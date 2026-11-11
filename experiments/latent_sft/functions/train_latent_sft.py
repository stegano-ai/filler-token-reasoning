import asyncio
import json
import math
import random
from pathlib import Path

import tinker
from tinker_cookbook import checkpoint_utils, model_info, renderers
from tinker_cookbook.eval.evaluators import TrainingClientEvaluator
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.types import SupervisedDataset
from tinker_cookbook.tokenizer_utils import get_tokenizer

from lib.side_tasks import get_side_task

RECIPE_NAME = "filler_token_reasoning_latent_sft"


class ThresholdReached(Exception):
    def __init__(self, state_path: str, metrics: dict[str, float], step: int):
        self.state_path = state_path
        self.metrics = metrics
        self.step = step
        super().__init__(f"early-stop threshold reached: {metrics}")


class EarlyStop(TrainingClientEvaluator):
    """Stop once `metric` clears `threshold`, but never before `min_steps` training steps.
    The trainer hands evaluators no step number, so steps are derived from the call count
    times `eval_every` — the same convention as the sft experiment's EarlyStopEvaluator."""

    def __init__(self, inner: TrainingClientEvaluator, metric: str, threshold: float,
                 min_steps: int, eval_every: int, log_path: str):
        self.inner = inner
        self.metric = metric
        self.threshold = threshold
        self.min_steps = min_steps
        self.eval_every = eval_every
        self.log_path = log_path
        self._n_evals = 0

    async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
        metrics = await self.inner(training_client)
        value = metrics[self.metric]
        steps_done = self._n_evals * self.eval_every
        print(f"Early-stop check: {self.metric}={value:.4f} threshold={self.threshold} "
              f"step={steps_done} min_steps={self.min_steps}")
        if steps_done >= self.min_steps and value >= self.threshold:
            print("Threshold reached, stopping")
            paths = await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"early_stop_{self._n_evals}",
                log_path=self.log_path,
                loop_state={"batch": self._n_evals},
                kind="both",
            )
            raise ThresholdReached(paths["state_path"], metrics, steps_done)
        self._n_evals += 1
        return metrics


class SFTDataset(SupervisedDataset):
    def __init__(self, data: list[tinker.Datum], batch_size: int, seed: int = 0):
        self.data = data
        self.batch_size = batch_size
        self.seed = seed
        random.Random(seed).shuffle(self.data)

    def get_batch(self, index: int) -> list[tinker.Datum]:
        start = index * self.batch_size
        return self.data[start:min(start + self.batch_size, len(self.data))]

    def __len__(self) -> int:
        return (len(self.data) + self.batch_size - 1) // self.batch_size

    def set_epoch(self, seed: int = 0):
        random.Random(self.seed + seed).shuffle(self.data)


def single_token(tokenizer, text: str) -> int:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) != 1:
        raise ValueError(f"{text!r} tokenizes to {len(ids)} tokens {ids}; need exactly one")
    return ids[0]


def make_datum(prefix: list[int], dot: int, sep: list[int], stop: list[int],
               digit_tokens: dict[int, int], states: list[int], answer: int,
               supervise_states: bool = True, use_filler: bool = True) -> tinker.Datum:
    """A Datum whose assistant body is one dot per side-task step, `sep` and the answer.

    Teacher-forcing targets are NOT the shifted input at the dot positions: the target at
    the position reading dot i is the token of state i, so the loss trains the forward pass
    to compute the running state at each filler position while the visible sequence stays
    dots. The answer keeps its ordinary next-token target after `sep`. Only the dot and
    answer positions carry loss weight.

    `use_filler=False` drops the dots entirely: the assistant turn is just `sep` and the
    answer, so the model answers straight from the question with no extra forward
    positions to compute in. That is the no-CoT floor."""
    n = len(states) if use_filler else 0
    full = list(prefix) + [dot] * n + list(sep) + [digit_tokens[answer]] + list(stop)
    inp, tgt = full[:-1], full[1:]
    weights = [0.0] * len(tgt)
    p = len(prefix)
    for i in range(n):
        tgt[p + i] = digit_tokens[states[i]]
        weights[p + i] = 1.0 if supervise_states else 0.0
    weights[p + n + len(sep) - 1] = 1.0
    total = sum(weights)
    weights = [w / total for w in weights]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(inp),
        loss_fn_inputs={
            "weights": tinker.TensorData(data=weights, dtype="float32", shape=[len(weights)]),
            "target_tokens": tinker.TensorData(data=tgt, dtype="int64", shape=[len(tgt)]),
        },
    )


def make_open_datum(prefix: list[int], sep: list[int], stop: list[int],
                    digit_tokens: dict[int, int], states: list[int],
                    answer: int) -> tinker.Datum:
    """The open-computation twin of make_datum ("dot by dot" mixing): the assistant body is
    the running states written as real digit tokens, supervised with ordinary next-token
    targets (position reading state i-1 predicts state i; the prefix end predicts state 0),
    plus the usual answer target. Same loss positions per item as the latent datum, so a
    mixed batch weights both modes equally."""
    n = len(states)
    full = (list(prefix) + [digit_tokens[s] for s in states] + list(sep)
            + [digit_tokens[answer]] + list(stop))
    inp, tgt = full[:-1], full[1:]
    weights = [0.0] * len(tgt)
    p = len(prefix)
    for i in range(n):
        weights[p - 1 + i] = 1.0
    weights[p + n + len(sep) - 1] = 1.0
    total = sum(weights)
    weights = [w / total for w in weights]
    return tinker.Datum(
        model_input=tinker.ModelInput.from_ints(inp),
        loss_fn_inputs={
            "weights": tinker.TensorData(data=weights, dtype="float32", shape=[len(weights)]),
            "target_tokens": tinker.TensorData(data=tgt, dtype="int64", shape=[len(tgt)]),
        },
    )


class LatentLogitEvaluator(TrainingClientEvaluator):
    """Logit-based eval: the whole sequence (dots and `#### `) is prefilled and nothing is
    sampled. For each item one datum per alphabet value carries that value's token as the
    target at every probed position; a forward pass then yields, per position, the logprob
    of each candidate, and the prediction is the argmax over candidates. Reported:
    eval/output_correct (answer position), eval/state_correct (mean over dot positions),
    eval/state_correct_last (final dot), eval/gold_answer_logprob."""

    def __init__(self, items: list[dict], prefix_by_item: list[list[int]], dot: int,
                 sep: list[int], stop: list[int], digit_tokens: dict[int, int],
                 alphabet: list[int], open_input: bool = False,
                 metric_prefix: str = "eval", use_filler: bool = True):
        self.items = items
        self.alphabet = alphabet
        self.metric_prefix = metric_prefix
        self.gold_prob_curve: list[float] | None = None
        self.datums, self.probe_meta = [], []
        for item, prefix in zip(items, prefix_by_item):
            states = item["reasoning"]
            n, p = len(states), len(prefix)
            if open_input:
                # Open-mode probing: the input body is the GOLD state digits (teacher
                # forced); each candidate datum overrides the targets at the state-
                # prediction slots (p-1+i predicts state i) and the answer slot with
                # candidate v. Same argmax-over-candidates read-out as latent mode.
                gold = [digit_tokens[s] for s in states]
                for v in alphabet:
                    full = (list(prefix) + gold + list(sep)
                            + [digit_tokens[v]] + list(stop))
                    inp, tgt = full[:-1], full[1:]
                    for i in range(n):
                        tgt[p - 1 + i] = digit_tokens[v]
                    weights = [1.0] * len(tgt)
                    self.datums.append(tinker.Datum(
                        model_input=tinker.ModelInput.from_ints(inp),
                        loss_fn_inputs={
                            "weights": tinker.TensorData(
                                data=weights, dtype="float32", shape=[len(weights)]),
                            "target_tokens": tinker.TensorData(
                                data=tgt, dtype="int64", shape=[len(tgt)]),
                        },
                    ))
                self.probe_meta.append((p - 1, n, p + n + len(sep) - 1))
            else:
                m = n if use_filler else 0
                for v in alphabet:
                    self.datums.append(make_datum(
                        prefix, dot, sep, stop, digit_tokens, [v] * n, v,
                        use_filler=use_filler))
                self.probe_meta.append((p, m, p + m + len(sep) - 1))

    async def __call__(self, training_client: tinker.TrainingClient) -> dict[str, float]:
        future = await training_client.forward_async(self.datums, loss_fn="cross_entropy")
        result = await future.result_async()
        logprobs = [list(x["logprobs"].data) for x in result.loss_fn_outputs]
        m = len(self.alphabet)
        n_ans, n_state, n_state_last, n_pos, gold_lp = 0, 0, 0, 0, 0.0
        dec_hits, dec_tot = [0] * 10, [0] * 10
        max_n = max([n for _, n, _ in self.probe_meta] + [0])
        pos_lp, pos_cnt = [0.0] * max_n, [0] * max_n
        for j, (item, (p, n, ans_t)) in enumerate(zip(self.items, self.probe_meta)):
            lps = logprobs[j * m:(j + 1) * m]
            states, answer = item["reasoning"], item["output"][-1]
            pred_ans = self.alphabet[max(range(m), key=lambda k: lps[k][ans_t])]
            n_ans += pred_ans == answer
            gold_lp += lps[self.alphabet.index(answer)][ans_t]
            for i in range(n):
                pred = self.alphabet[max(range(m), key=lambda k: lps[k][p + i])]
                pos_lp[i] += math.exp(lps[self.alphabet.index(states[i])][p + i])
                pos_cnt[i] += 1
                n_state += pred == states[i]
                n_pos += 1
                if i == n - 1:
                    n_state_last += pred == states[i]
                dec = min(9, (10 * i) // n)
                dec_hits[dec] += pred == states[i]
                dec_tot[dec] += 1
        self.gold_prob_curve = [s / c for s, c in zip(pos_lp, pos_cnt)]  # empty if no filler
        n_items = len(self.items)
        out = {
            f"{self.metric_prefix}/output_correct": n_ans / n_items,
            f"{self.metric_prefix}/state_correct": (n_state / n_pos) if n_pos else 0.0,
            f"{self.metric_prefix}/state_correct_last": n_state_last / n_items,
            f"{self.metric_prefix}/gold_answer_logprob": gold_lp / n_items,
        }
        for d in range(10):
            if dec_tot[d]:
                out[f"{self.metric_prefix}/state_correct_dec{d + 1}"] = dec_hits[d] / dec_tot[d]
        return out


async def train_latent_sft(
    model_name: str,
    log_path: str,
    sft: dict,
    task_prompt: str,
    n_train: int,
    n_final_eval: int,
    n_eval: int,
    batch_size: int,
    side_task: dict,
    renderer_name: str | None,
    wandb_project: str | None,
    wandb_name: str | None,
    seed: int,
    early_stop_metric: str | None,
    early_stop_threshold: float | None,
    early_stop_min_steps: int,
    open_fraction: float = 0.0,
    state_supervision: bool = True,
    use_filler: bool = True,
) -> tuple[str, bool, list[float]]:
    resolved_renderer_name = renderer_name or model_info.get_recommended_renderer_name(model_name)
    Path(log_path).mkdir(parents=True, exist_ok=True)

    side = get_side_task(**side_task)
    tokenizer = get_tokenizer(model_name)
    renderer = renderers.get_renderer(resolved_renderer_name, tokenizer=tokenizer)

    dot = single_token(tokenizer, " .")
    digit_tokens = {v: single_token(tokenizer, str(v)) for v in side.alphabet}
    sep = tokenizer.encode("\n\n#### ", add_special_tokens=False)
    stop = renderer.get_stop_sequences()
    if not all(isinstance(t, int) for t in stop):
        raise ValueError(f"expected token-id stop sequences, got {stop!r}")

    def prefix_for(item: dict) -> list[int]:
        user_text = task_prompt.format(
            SIDE_INPUT=item["input"],
            N_VALUES=len(item["reasoning"]),
            MAX_VALUE=max(side.alphabet),
            SIDE_TASK_EXPLANATION=side.explanation,
        )
        return renderer.build_generation_prompt(
            [{"role": "user", "content": user_text}]).to_ints()

    def gold_datum(item: dict) -> tinker.Datum:
        return make_datum(prefix_for(item), dot, sep, stop, digit_tokens,
                          item["reasoning"], item["output"][-1], state_supervision,
                          use_filler)

    if n_eval + n_final_eval > len(side.test_data):
        raise ValueError(f"n_eval + n_final_eval = {n_eval + n_final_eval} exceeds the "
                         f"{len(side.test_data)} test items the side task provides")
    if open_fraction > 0.0:
        # "Dot by dot" mixing: each training PRESENTATION is independently rendered open
        # (states as real digit tokens, next-token supervised) with probability
        # open_fraction, else latent (dots). Deterministic in the seed; a cycled item can
        # appear in both modes. open_fraction 0.0 reproduces the pure-latent pipeline
        # byte-identically.
        mix_rng = random.Random(seed + 1)

        def open_datum(item: dict) -> tinker.Datum:
            return make_open_datum(prefix_for(item), sep, stop, digit_tokens,
                                   item["reasoning"], item["output"][-1])

        train_data = [
            (open_datum if mix_rng.random() < open_fraction else gold_datum)(
                side.train_data[i % len(side.train_data)])
            for i in range(n_train)
        ]
    else:
        train_data = [gold_datum(side.train_data[i % len(side.train_data)]) for i in range(n_train)]
    eval_items = side.test_data[:n_eval]
    final_items = side.test_data[n_eval:n_eval + n_final_eval]
    print(f"Train: {len(train_data)}, Eval: {len(eval_items)}, "
          f"Final eval: {len(final_items)} (open_fraction {open_fraction})")

    logit_evaluator = LatentLogitEvaluator(
        items=eval_items,
        prefix_by_item=[prefix_for(item) for item in eval_items],
        dot=dot, sep=sep, stop=stop, digit_tokens=digit_tokens, alphabet=side.alphabet,
        use_filler=use_filler,
    )
    final_evaluator = LatentLogitEvaluator(
        items=final_items,
        prefix_by_item=[prefix_for(item) for item in final_items],
        dot=dot, sep=sep, stop=stop, digit_tokens=digit_tokens, alphabet=side.alphabet,
        metric_prefix="eval_final", use_filler=use_filler,
    )
    open_evaluator = None
    if open_fraction > 0.0:
        # Separate success tracking for the open mode, under eval_open/* metric names.
        open_evaluator = LatentLogitEvaluator(
            items=eval_items,
            prefix_by_item=[prefix_for(item) for item in eval_items],
            dot=dot, sep=sep, stop=stop, digit_tokens=digit_tokens,
            alphabet=side.alphabet, open_input=True, metric_prefix="eval_open",
        )
    if early_stop_metric is not None:
        if early_stop_threshold is None:
            raise ValueError("early_stop_threshold required when early_stop_metric is set")
        # The wrapper must sit on the evaluator that produces the gated metric.
        if early_stop_metric.startswith("eval_open/"):
            if open_evaluator is None:
                raise ValueError(f"{early_stop_metric} requires open_fraction > 0")
            open_evaluator = EarlyStop(
                open_evaluator, early_stop_metric, early_stop_threshold,
                early_stop_min_steps, sft["eval_every"], log_path)
        else:
            logit_evaluator = EarlyStop(
                logit_evaluator, early_stop_metric, early_stop_threshold,
                early_stop_min_steps, sft["eval_every"], log_path)

    sft = dict(sft)
    print(f"LR {sft['learning_rate']:.3e}")
    config = train.Config(
        **sft,
        model_name=model_name,
        recipe_name=RECIPE_NAME,
        renderer_name=renderer_name,
        log_path=log_path,
        dataset_builder=lambda: (SFTDataset(train_data, batch_size, seed), None),
        evaluator_builders=([lambda: logit_evaluator]
                            + ([lambda: open_evaluator] if open_evaluator else [])),
        wandb_project=wandb_project,
        wandb_name=wandb_name,
    )
    try:
        await train.main(config)
        state_path, reached = checkpoint_utils.get_last_checkpoint(log_path).state_path, False
    except ThresholdReached as e:
        print(f"Early-stop threshold reached ({early_stop_metric}="
              f"{e.metrics[early_stop_metric]:.4f} >= {early_stop_threshold}) at step {e.step}")
        state_path, reached = e.state_path, True

    # One large final eval against the returned weights (eval_final/*), appended to
    # metrics.jsonl so the curriculum tailer forwards it like any eval row. The
    # per-position gold-state prob curve comes from this eval.
    client = await tinker.ServiceClient().create_training_client_from_state_async(state_path)
    metrics = await final_evaluator(client)
    print(f"Final eval: {metrics}")
    metrics_path = Path(log_path) / "metrics.jsonl"
    last_step = json.loads(metrics_path.read_text().splitlines()[-1])["step"]
    with metrics_path.open("a") as f:
        f.write(json.dumps({"step": last_step + 1, **metrics}) + "\n")
    return state_path, reached, final_evaluator.gold_prob_curve


def main(**kwargs):
    asyncio.run(train_latent_sft(**kwargs))

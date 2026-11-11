# Markdown tables for the post:
#   results.md          per task x model: the latent max with its source run, the curriculum walls with 200-item accuracy, the
#                       one-stage frontier, the no-filler and open controls (all from runs.summary / runs.campaign)
#   hyperparameters.md  the shared training setup, read from the composed run configs
#   runs.md             every run behind a figure, with its final Tinker checkpoint (checkpoints expire after 7 days)
#   uv run --no-project --with pyyaml python figures/tables/make_tables.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runs import MODELS, TASKS, campaign, certified_max, frontier, summary

OUT = Path(__file__).resolve().parent
S = summary()
CUR = {(r["task"], r["model"]): r for r in campaign("curriculum")}
DIR = {(r["task"], r["model"]): r for r in campaign("direct")}


def wall(r):
    s = r["stages"]
    passed = [x for x in s if x["certified"]]
    a = f"{passed[-1]['size']} ({passed[-1]['final']['output_correct']:.2f})" if passed else "none"
    b = "ladder ran out" if certified_max(r)[1] else f"{s[-1]['size']} ({s[-1]['final']['output_correct']:.2f})"
    return a, b


lines = ["| task | model | latent max | from | curriculum: last passed (acc) | curriculum: wall (acc) | "
         "one-stage frontier / target | no filler | open |", "|" + "---|" * 9]
for t, (label, _) in TASKS.items():
    for m, (mlabel, _, _) in MODELS.items():
        c = S[t, m]
        passed, walled = wall(CUR[t, m])
        d = DIR[t, m]
        lines.append(f"| {label.replace(chr(10), ' ')} | {mlabel} | {'≥' if c['latent_open_ended'] else ''}{c['latent']:g} | "
                     f"`{c['latent_src']}` | {passed} | {walled} | {frontier(d):g} / {d['stages'][0]['size']} | "
                     f"{c['nocot']} | ≥{c['open']} |")
(OUT / "results.md").write_text(
    "Lengths in task steps. acc = answer accuracy on the 200-item final eval (100 on first rungs) of that stage's checkpoint; "
    "a stage passes when the 20-item gate reads >= 0.9, so a passed stage can sit below 0.9 here. Frontier = deepest trace "
    "decile with every decile up to it at >= 0.95 state accuracy, in steps.\n\n" + "\n".join(lines) + "\n")

fk = CUR["mod2", "qwen"]["kwargs"]
rows = [("LoRA rank", fk["sft"]["lora_rank"]), ("learning rate (Qwen / gpt-oss)",
        f"{fk['sft']['learning_rate']:.1e} / {CUR['mod2', 'oss20b']['kwargs']['sft']['learning_rate']:.1e}"),
        ("LR schedule", fk["sft"]["lr_schedule"]), ("batch size", fk["batch_size"]), ("train examples per stage", fk["n_train"]),
        ("epochs per stage (max)", fk["sft"]["num_epochs"]),
        ("max steps per stage", fk["n_train"] * fk["sft"]["num_epochs"] // fk["batch_size"]),
        ("eval every", f"{fk['sft']['eval_every']} steps, {fk['n_eval']} items"),
        ("pass gate", f"{fk['early_stop_metric']} >= {fk['early_stop_threshold']} after >= {fk['early_stop_min_steps']} steps"),
        ("final eval", f"{fk['n_final_eval']} items"), ("renderer (Qwen / gpt-oss)",
        f"{fk['renderer_name']} / {CUR['mod2', 'oss20b']['kwargs']['renderer_name']}")]
(OUT / "hyperparameters.md").write_text("| | |\n|---|---|\n" + "\n".join(f"| {k} | {v} |" for k, v in rows) + "\n")

lines = ["| campaign | run | stages (size: passed) | final checkpoint |", "|---|---|---|---|"]
for name in ["curriculum", "direct", "comparison_method", "nocot", "open"]:
    for r in campaign(name):
        st = ", ".join(f"{s['size']}:{'y' if s['certified'] else 'n'}" for s in r["stages"])
        lines.append(f"| {name} | `{r['name']}` | {st} | `{r['stages'][-1]['state_path']}` |")
(OUT / "runs.md").write_text("Runs in `experiments/latent_sft/results/<campaign>/<run>`. Tinker checkpoints carry a 7-day TTL, "
                             "so the paths are a record, not something that still loads.\n\n" + "\n".join(lines) + "\n")

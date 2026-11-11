import asyncio
import json
from pathlib import Path

import wandb
from research_scaffold.util import recursive_dict_update

from .train_latent_sft import train_latent_sft


async def _tail_metrics(metrics_path: Path, run, offset: int, stop: asyncio.Event) -> int:
    """Asynchronously stream a stage's metrics.jsonl rows into the curriculum's wandb run,
    shifting steps past the previous stages'. Same mechanism as the sft experiment."""
    last_step = offset - 1
    pos = 0
    while True:
        if metrics_path.exists():
            with metrics_path.open() as f:
                f.seek(pos)
                while True:
                    line = f.readline()
                    if not line.endswith("\n"):
                        break
                    pos = f.tell()
                    row = json.loads(line)
                    step = row.pop("step", None)
                    if step is None:
                        continue
                    last_step = step + offset
                    run.log(row, step=last_step)
        if stop.is_set():
            return last_step
        await asyncio.sleep(5)


DONE_MARKER = ".complete"


async def train_curriculum(stages: list[dict], log_path: str, wandb_name: str,
                           **shared_kwargs):
    if not stages:
        raise ValueError("stages must be a non-empty list")

    # Re-running a config only runs what is missing: a run that reached a definitive
    # end (all stages passed, or a stage walled) left a marker and is skipped. A
    # crashed or interrupted run has no marker and is re-run from scratch.
    done_path = Path(log_path) / DONE_MARKER
    if done_path.exists():
        print(f"{wandb_name}: already complete ({done_path}), skipping")
        return None

    run = wandb.init(
        project=shared_kwargs["wandb_project"],
        name=wandb_name,
        config={"stages": stages, **shared_kwargs},
    )
    offset = 0
    state_path = None
    gold_prob_curve = None
    try:
        for i, stage in enumerate(stages):
            stage_kwargs = recursive_dict_update(shared_kwargs, stage)
            stage_kwargs["log_path"] = f"{log_path}/stage{i + 1}"
            stage_kwargs["wandb_name"] = wandb_name
            stage_kwargs["wandb_project"] = None  # stage stays headless; metrics go to jsonl
            if state_path is not None:
                stage_kwargs["sft"].update({"load_checkpoint_path": state_path})

            print(f"=== Curriculum stage {i + 1}/{len(stages)} ===")
            metrics_path = Path(stage_kwargs["log_path"]) / "metrics.jsonl"
            stop = asyncio.Event()
            tailer = asyncio.create_task(_tail_metrics(metrics_path, run, offset, stop))
            try:
                state_path, reached, gold_prob_curve = await train_latent_sft(**stage_kwargs)
            finally:
                stop.set()
                offset = await tailer + 1

            if stage_kwargs["early_stop_metric"] is not None and not reached:
                # The ladder found its wall — a result, not an error. Stop here; the
                # stage's metrics record where it stalled.
                print(f"Stage {i + 1} did not reach "
                      f"{stage_kwargs['early_stop_metric']} >= "
                      f"{stage_kwargs['early_stop_threshold']}; curriculum walled here "
                      f"(last checkpoint: {state_path})")
                break
            done = "reached threshold" if reached else "finished all steps"
            print(f"Stage {i + 1} {done}; checkpoint: {state_path}")
        done_path.write_text("")
    finally:
        # Final model's per-position gold-state probability, as a native line panel
        # (x-axis: state_position). Logged once, after all step-indexed logging is done.
        if gold_prob_curve is not None:
            run.define_metric("eval/state_gold_prob_final", step_metric="state_position")
            for i, prob in enumerate(gold_prob_curve):
                run.log({"eval/state_gold_prob_final": prob, "state_position": i + 1})
        wandb.finish()

    return state_path


def main(**kwargs):
    asyncio.run(train_curriculum(**kwargs))

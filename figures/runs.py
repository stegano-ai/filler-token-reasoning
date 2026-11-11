# Reads the latent_sft and theory results into plain records. Every figure and table in figures/ goes through here,
# so "certified", "size" and "frontier" mean the same thing everywhere:
#   size       the task's step count at a stage (seq_length / num_steps / hops / n_hops)
#   certified  the stage's 0.9 early-stop gate fired (a checkpoint named early_stop_* exists)
#   final      the 200-item (100 on small first rungs) eval of the stage's returned checkpoint
#   frontier   a one-stage-at-target run: the deepest trace decile with every decile up to it at >= 0.95 state accuracy,
#              converted to steps (decile d of a target-T run = d/10 * T)
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
LATENT = ROOT / "experiments/latent_sft/results"
THEORY = ROOT / "experiments/theory/frontier.json"

TASKS = {  # name in run folders -> (label, theory key); ordered by longest length tested, 100 -> 25.
    "mod2": ("mod sum", "rolling_mod_sum"),
    "a5": ("A5", "ball_and_cups_a5"),
    "gl32": ("GL(3,2)", "gl32"),
    "arith8mod": ("modular\narithmetic", "limited_arithmetic"),
    "chain": ("ProntoQA", "prontoqa_chain"),
    "horn": ("HornQA", "horn"),
    "fc": ("function\ncomposition", "function_composition"),
}
MODELS = {"qwen": ("Qwen3.6-35B-A3B", "qwen", 40), "oss20b": ("gpt-oss-20b", "oss", 24)}  # -> (label, theory preset, layers)
SIZE_KEYS = ("seq_length", "num_steps", "hops", "n_hops")


def merge(a, b):
    return {k: merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v for k, v in {**a, **b}.items()}


def rows(path):
    return [json.loads(l) for l in path.read_text().splitlines()]


def run(path):
    """One run folder -> {name, task, model, kwargs, stages: [{size, certified, steps, curve, final}]}."""
    fk = yaml.safe_load((path / "config.yaml").read_text())["function_kwargs"]
    stages = []
    for i, patch in enumerate(fk["stages"]):
        d = path / f"stage{i + 1}"
        if not (d / "metrics.jsonl").exists():
            break
        st = merge(fk, patch)["side_task"]
        m = rows(d / "metrics.jsonl")
        ck = (d / "checkpoints.jsonl").read_text() if (d / "checkpoints.jsonl").exists() else ""
        final = next(r for r in m if "eval_final/output_correct" in r)
        stages.append({
            "size": next(st[k] for k in SIZE_KEYS if k in st),
            "certified": "early_stop_" in ck,
            "steps": max(r["step"] for r in m if "train_mean_nll" in r) + 1,
            "curve": [(r["step"], r["eval/output_correct"]) for r in m if "eval/output_correct" in r],
            "final": {k.split("/")[1]: v for k, v in final.items() if k.startswith("eval_final/")},
            "state_path": json.loads(ck.splitlines()[-1])["state_path"] if ck else None,
        })
    task = path.name.split("_")[1]
    return {"name": path.name, "task": task, "model": "oss20b" if "oss" in fk["model_name"] else "qwen", "kwargs": fk,
            "stages": stages}


def campaign(name):
    """The runs of one results campaign, restricted to the tasks in TASKS."""
    return [r for r in (run(p) for p in sorted((LATENT / name).iterdir()) if (p / "config.yaml").exists()) if r["task"] in TASKS]


def certified_max(r):
    """Largest certified size, and whether the ladder ran out above it (so the wall is unmeasured)."""
    sizes = [s["size"] for s in r["stages"] if s["certified"]]
    return (max(sizes) if sizes else 0), len(sizes) == len(r["kwargs"]["stages"])


def frontier(r):
    """Solid-prefix length of a one-stage run, in task steps."""
    s = r["stages"][0]
    d = 0
    while d < 10 and s["final"][f"state_correct_dec{d + 1}"] >= 0.95:
        d += 1
    return d / 10 * s["size"]


def theory():
    return json.loads(THEORY.read_text())


def summary():
    """(task, model) -> {latent, latent_src, latent_open_ended, nocot, open, target}: the largest length reached per condition.
    latent = max over the curriculum, the method-comparison runs of our method (ladder and one-stage, all seeds) and the
    one-stage-at-target frontier; open_ended = the ladder ran out of rungs, so the wall is above the number."""
    out = {(t, m): {"latent": 0, "latent_src": "none", "latent_open_ended": False, "nocot": 0, "open": 0}
           for t in TASKS for m in MODELS}
    for r in campaign("curriculum") + [r for r in campaign("comparison_method") if "_ours" in r["name"]]:
        n, ran_out = certified_max(r)
        c = out[r["task"], r["model"]]
        if n > c["latent"]:
            c.update(latent=n, latent_src=r["name"], latent_open_ended=ran_out)
    for r in campaign("direct"):
        if frontier(r) > out[r["task"], r["model"]]["latent"]:
            out[r["task"], r["model"]].update(latent=frontier(r), latent_src=r["name"] + " (frontier)", latent_open_ended=False)
    for r in campaign("curriculum"):  # the longest length tested: the top of the ladder
        st = r["kwargs"]["side_task"]
        out[r["task"], r["model"]]["target"] = next(st[k] for k in SIZE_KEYS if k in st)
    for key, name in [("nocot", "nocot"), ("open", "open")]:
        for r in campaign(name):
            out[r["task"], r["model"]][key] = certified_max(r)[0]
    return out

# Our method vs the alternatives: the longest length each run certified (0.9 gate on the answer) on a 10 -> 40 ladder.
# One group per task and model; light bar = mean over seeds, dots = individual seeds. ours = dots with per-dot state targets;
# Pfau et al. = their dot-by-dot recipe (50% open CoT, 50% dots, loss on the answer only); answer only = no dots at all, the
# answer straight after the question (seed 0 from results/nocot; seeds 1-2 from configs/comparison_nocot_seeds.yaml land in
# results/comparison_method and are picked up automatically). 0 = failed the first rung. Dotted line = target (40).
#   uv run --no-project --with pyyaml --with matplotlib python figures/method_comparison/plot.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from runs import MODELS, TASKS, campaign, certified_max
from style import ARMS, FULL, MUTED, SURFACE, legend_below, save, tint

RUNS = campaign("comparison_method") + [{**r, "name": r["name"] + "_seed0"} for r in campaign("nocot")]  # seed 0 of the no-dots arm
GROUPS = [("mod2", "qwen"), ("mod2", "oss20b"), ("gl32", "qwen")]
ARMS_SHOWN = [("ours", "ours"), ("dotbydot", "Pfau et al."), ("nocot", "answer only")]
w = 0.28

fig, ax = plt.subplots(figsize=(FULL, 2.4))
for g, (task, model) in enumerate(GROUPS):
    for k, (arm, label) in enumerate(ARMS_SHOWN):
        v = np.array([certified_max(r)[0] for r in RUNS
                      if r["task"] == task and r["model"] == model and f"_{arm}_seed" in r["name"]])
        x = g + (k - 1) * w
        ax.bar(x, v.mean(), w * 0.88, color=tint(ARMS[arm], 0.35), lw=0, label=label if g == 0 else None)
        ax.plot(x + np.linspace(-0.07, 0.07, len(v)), v, "o", ms=4.5, color=ARMS[arm], mec=SURFACE, mew=0.7, zorder=3)
ax.axhline(40, color=MUTED, ls=":", lw=1)
ax.set_xticks(range(len(GROUPS)), [f"{TASKS[t][0]}\n{MODELS[m][0]}" for t, m in GROUPS])
ax.tick_params(axis="x", length=0)
ax.set(ylabel="task length (steps)", yticks=[0, 10, 20, 30, 40], ylim=(-2, 44))
legend_below(fig, ncol=3, pad=0.02)
save(fig, __file__, "comparison.png")

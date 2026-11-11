# Theory: layers of the shallowest circuit verified at each task length, one panel per model architecture (Qwen preset:
# 40 layers, linear/linear/linear/full; gpt-oss preset: 24 layers, window(128)/full). An upper bound on the depth a task needs,
# not a minimum. Dotted line = the model's depth. Lines are nudged apart vertically (±6% of a doubling per task step) so ties stay visible. Data: experiments/theory/frontier.json.
#   uv run --no-project --with pyyaml --with matplotlib python figures/theory_frontier/plot.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from runs import MODELS, TASKS, theory
from style import FULL, MUTED, TASK_COLOR, TASK_MARKER, legend_below, save

T = theory()
fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.8), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, (model, (label, preset, depth)) in zip(axes, MODELS.items()):
    for i, (task, (tlabel, key)) in enumerate(TASKS.items()):
        f = {int(n): v["layers"] for n, v in T[preset]["frontier"][key].items() if v["layers"] > 0}
        nudge = 2 ** (0.06 * (i - (len(TASKS) - 1) / 2))  # small vertical offset per task so equal values do not hide each other
        ax.plot(list(f), [L * nudge for L in f.values()], color=TASK_COLOR[task], marker=TASK_MARKER[task], ms=3.5, mec="white", mew=0.4,
                markevery=max(1, len(f) // 8), label=tlabel.replace("\n", " "), zorder=3 if task == "mod2" else 2)
    ax.axhline(depth, color=MUTED, ls=":", lw=1)
    ax.text(1, depth * 1.08, f"model depth: {depth} layers", color=MUTED, fontsize=7, va="bottom")
    ax.set(title=label, xlabel="task length (steps)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256, 1024], ["1", "4", "16", "64", "256", "1024"])
    ax.set_yticks([1, 2, 4, 8, 16, 32, 64, 128], ["1", "2", "4", "8", "16", "32", "64", "128"])
    ax.minorticks_off()
axes[0].set_ylabel("layers of shallowest\nverified circuit")
legend_below(fig, ncol=4, pad=0.02)
save(fig, __file__, "frontier.png")

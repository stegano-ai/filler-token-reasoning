# Controls, one panel per model: longest length certified with no filler at all (immediate answer), with latent dot
# reasoning (our method), and with the reasoning written out as digit tokens (open). Open runs were
# trained only at the target length and all passed at the first check. Dotted line = the longest length tested for that task
# (top of its ladder), so a bar that reaches it never hit a wall. Exact numbers are in tables/results.md.
#   uv run --no-project --with pyyaml --with matplotlib python figures/controls/plot.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from runs import MODELS, TASKS, summary
from style import ARMS, FULL, MUTED, legend_below, save

S = summary()
ARMS_SHOWN = [("nocot", "answer only"), ("latent", "filler-token reasoning"), ("open", "open reasoning")]
fig, axes = plt.subplots(2, 1, figsize=(FULL, 4.4), sharex=True, gridspec_kw={"hspace": 0.3})
w = 0.29
for ax, (model, (label, _, _)) in zip(axes, MODELS.items()):
    for k, (arm, alabel) in enumerate(ARMS_SHOWN):
        for i, task in enumerate(TASKS):
            v, x = S[task, model][arm], i + (k - 1) * w
            ax.bar(x, v, w * 0.92, color=ARMS["ours" if arm == "latent" else arm], label=alabel if i == 0 else None)
    for i, task in enumerate(TASKS):
        ax.plot([i - w * 1.6, i + w * 1.6], [S[task, model]["target"]] * 2, color=MUTED, ls=":", lw=1)
    ax.set(title=label, ylim=(0, 108))
    ax.set_ylabel("task length (steps)")
axes[1].set_xticks(range(len(TASKS)), [label for label, _ in TASKS.values()])
axes[1].tick_params(axis="x", length=0)
legend_below(fig, ncol=3, pad=0.0)
save(fig, __file__, "controls.png")

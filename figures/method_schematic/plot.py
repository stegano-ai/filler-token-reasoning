# Schematic: what the model is trained on, ours vs Pfau et al., on 2 × 2 + 4 (states 2, 4, 8; answer 8). Each example is a
# row of input tokens (boxes), the model (bar), and the next token above each position: green = trained, grey = no loss.
# Ours: dots in, the reasoning state out at every dot, then the answer. Pfau et al.: a 50/50 mix of open chain of thought
# (ordinary next-token targets) and filler dots trained only on the answer (make_open_datum / state_supervision: false).
#   uv run --no-project --with pyyaml --with matplotlib python figures/method_schematic/plot.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from style import FULL, GRID, MUTED, TEXT, save, tint

GREEN = "#008300"
MODEL = tint(MUTED, 0.35)
P = r"2 $\times$ 2 + 4"  # cmss10 has no ×, – or →: math glyphs instead
EXAMPLES = {  # (inputs, targets); "~" marks the next token at a position without loss, None = nothing read off the prompt
    "ours": ([P, ".", ".", ".", "####"], [None, "2", "4", "8", "8"]),
    "open": ([P, "2", "4", "8", "####"], ["2", "4", "8", "~####", "8"]),
    "filler": ([P, ".", ".", ".", "####"], [None, "~.", "~.", "~####", "8"]),
}
XS = [0, 1.65, 2.45, 3.25, 4.2]  # the prompt and separator boxes are wider
WID = [1.45, 0.65, 0.65, 0.65, 0.95]


def example(ax, y, key, label=None):
    inputs, targets = EXAMPLES[key]
    x0, x1 = XS[0] - WID[0] / 2, XS[-1] + WID[-1] / 2
    ax.add_patch(FancyBboxPatch((x0, y + 0.42), x1 - x0, 0.3, boxstyle="round,pad=0,rounding_size=0.08", fc=MODEL,
                                ec="none"))  # the model
    for x, w, tok, tgt in zip(XS, WID, inputs, targets):
        trained = tgt is not None and not tgt.startswith("~")
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.2), w, 0.4, boxstyle="round,pad=0,rounding_size=0.06", fc=GRID,
                                    ec="none"))
        ax.text(x, y, tok, ha="center", va="center", fontsize=9 if tok == "####" else 10.5,
                color=MUTED if tok in (".", "####") else TEXT)
        if tok == P:
            x += w / 2 - 0.17  # the prompt enters the model at its last token, the 4
        segments = [(y + 0.22, y + 0.4, False), (y + 0.74, y + 0.98, True)]
        if tok == P and tgt is None:
            segments = segments[:1]  # the prompt goes into the model, but nothing is read off it
        for ya, yb, head in segments:
            ax.add_patch(FancyArrowPatch((x, ya), (x, yb), arrowstyle="-|>" if head else "-", mutation_scale=6, lw=0.8,
                                         color=GREEN if trained else GRID, shrinkA=0, shrinkB=0))
        if len(segments) == 1:
            continue
        shown = tgt.lstrip("~")
        ax.text(x, y + 1.13, shown, ha="center", va="center", fontsize=11.5 if trained else 9 if shown == "####" else 10.5,
                color=GREEN if trained else MUTED)
    if label:
        ax.text(x0 - 0.15, y + 0.55, label, ha="right", va="center", fontsize=10.5, color=MUTED)


fig = plt.figure(figsize=(FULL, 2.3))
left, right = fig.add_axes([0.0, 0.0, 0.44, 1.0]), fig.add_axes([0.53, 0.0, 0.47, 1.0])
example(left, 0.85, "ours")
left.set_title("Ours", loc="left", fontsize=11.5)
example(right, 1.85, "open", "50%")
example(right, 0.0, "filler", "50%")
right.set_title("Pfau et al.", loc="left", fontsize=11.5)
for ax, x_start in [(left, -0.8), (right, -1.3)]:  # the right panel needs room for the 50% labels
    ax.set(xlim=(x_start, 4.7), ylim=(-0.4, 3.25))
    ax.axis("off")
fig.add_artist(plt.Line2D([0.485, 0.485], [0.05, 0.9], color=GRID, lw=0.8))
save(fig, __file__, "method_schematic.png")

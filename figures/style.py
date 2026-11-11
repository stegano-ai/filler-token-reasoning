# Shared look of every figure (rcParams copied from the eNTK report). Colour = task, the same in every figure, in the
# fixed order of the validated default categorical palette (dataviz skill; passes CVD on adjacent pairs). Tasks are also
# told apart by marker, so identity never rests on colour alone. Models: Qwen solid, gpt-oss hatched / dashed.
# Method arms (method comparison, controls) use ARMS instead, never a task colour.
# Sizes: FULL = 6.5in text width, HALF = two side by side; 8pt text prints at 8pt.
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

TASK_COLOR = {"mod2": "#2a78d6", "a5": "#eb6834", "gl32": "#1baf7a", "arith8mod": "#eda100", "fc": "#e87ba4",
              "chain": "#008300", "horn": "#e34948"}
TASK_MARKER = {"mod2": "o", "a5": "s", "gl32": "D", "arith8mod": "^", "fc": "v", "chain": "P", "horn": "*"}
ARMS = {"ours": "#2a78d6", "dotbydot": "#eb6834", "ansonly": "#a8a7a2", "nocot": "#5c5b57", "open": "#1baf7a"}
INK = "#1a1a1a"
TEXT, MUTED, GRID, SURFACE = "#1a1a1a", "#5c5b57", "#ebeae6", "#ffffff"
MODEL_HATCH = {"qwen": None, "oss20b": "////"}
MODEL_LS = {"qwen": "-", "oss20b": "--"}

FULL, HALF = 6.5, 3.15  # figure widths in inches: full text width, half (two side by side)

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["cmss10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "font.size": 8, "axes.titlesize": 8.5, "axes.titleweight": "normal", "axes.titlelocation": "left",
    "axes.titlepad": 4, "axes.labelsize": 8, "axes.labelcolor": TEXT, "axes.labelpad": 3, "axes.edgecolor": MUTED,
    "axes.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": TEXT, "ytick.labelcolor": TEXT, "xtick.major.width": 0.5,
    "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "xtick.minor.size": 1.5, "ytick.minor.size": 1.5, "legend.frameon": False, "legend.fontsize": 7.5, "legend.handlelength": 2.2,
    "legend.columnspacing": 1.4, "legend.handletextpad": 0.5, "lines.linewidth": 1.4, "lines.markersize": 4,
    "lines.solid_capstyle": "round", "errorbar.capsize": 0, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.03,
    "figure.figsize": (FULL, 2.6),
})


def tint(color, a=0.28):
    """The colour mixed with white (bar fills behind full-colour seed dots)."""
    c = np.array(plt.matplotlib.colors.to_rgb(color))
    return tuple(1 - a * (1 - c))


def legend_below(fig, ax=None, ncol=3, pad=0.0, **kw):
    """One legend for the whole figure, centred under it (handles from ax, default: the first axis). pad moves it further down
    (figure fraction). The tight bounding box at save time includes it."""
    ax = ax or fig.axes[0]
    return fig.legend(*ax.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, -pad), ncol=ncol, **kw)


def panel_label(ax, s):
    """Bold panel letter (CM math bold: cmss10 has no bold) ("a", "b", ...) at the top left of the axis, outside the plot area."""
    ax.text(-0.02, 1.02, rf"$\mathbf{{{s}}}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color=TEXT)


def save(fig, script, name):
    fig.savefig(Path(script).resolve().parent / name)
    plt.close(fig)

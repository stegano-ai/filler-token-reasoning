# Schematic: where the computation of 2 * 2 + 4 = 8 flows in open, steganographic and latent reasoning. Grid = transformer
# (columns = token positions, the prompt's collapsed into one; rows = layers), prompt below, sampled tokens above.
# Green = the path the reasoning takes. Open / steganographic: one serial path, up through every layer to a sampled token
# and (thin, arrow) back in at the next position. Latent: the sampled tokens are filler (not fed back); the work spreads from the
# prompt along the first layer, moves up inside the network and converges on the last position, which emits the answer.
#   uv run --no-project --with pyyaml --with matplotlib python figures/reasoning_modes/plot.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, PathPatch
from matplotlib.path import Path as MPath
from style import FULL, GRID, MUTED, TEXT, save, tint

GREEN = "#008300"
MODES = [("Open reasoning", ["2", "4", "8"]), ("Steganographic reasoning", ["hm", "good", "question"]),
         ("Filler-token reasoning", [".", ".", "."])]
DX, L, DY, W, H = 1.0, 3, 0.7, 0.62, 0.38  # column spacing, layers, layer spacing, cell size
XS = [DX * i for i in range(4)]  # positions that emit a token: the prompt's last one, then the reasoning positions
TOP, Y_IN, Y_OUT = (L - 1) * DY, -0.62, (L - 1) * DY + 0.55
TOK = dict(ha="center", va="center", fontsize=7.5, bbox=dict(fc="white", ec="none", pad=1.2), zorder=5)


def rounded(pts, r):  # polyline with rounded corners (radius capped at half of each adjacent segment)
    verts, codes = [pts[0]], [MPath.MOVETO]
    for a, b, c in zip(pts, pts[1:], pts[2:]):
        u, v = np.subtract(a, b), np.subtract(c, b)
        rr = min(r, np.hypot(*u) / 2, np.hypot(*v) / 2)
        verts += [b + u / np.hypot(*u) * rr, b, b + v / np.hypot(*v) * rr]
        codes += [MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
    return MPath(np.array(verts + [pts[-1]], float), codes + [MPath.LINETO])


def draw(ax, path, lw=1.5, color=GREEN, zorder=3):
    ax.add_patch(PathPatch(path, fc="none", ec=color, lw=lw, capstyle="round", joinstyle="round", zorder=zorder))


def stream(x):  # latent: from the prompt along the first layer to position x, up, along the last layer to the last position
    return rounded(list(dict.fromkeys([(0, Y_IN + 0.14), (0, 0), (x, 0), (x, TOP), (XS[-1], TOP), (XS[-1], Y_OUT - 0.2)])), 0.5)


fig, axes = plt.subplots(1, 3, figsize=(FULL, 1.6), gridspec_kw={"wspace": 0.1})
fig.subplots_adjust(left=0, right=1)
for ax, (title, gen) in zip(axes, MODES):
    latent = title.startswith("Filler")
    for l in range(L):  # the prompt is one wide bar per layer, each reasoning position one cell
        for x0, w in [(-1.0, 1.0 + W / 2)] + [(x - W / 2, W) for x in XS[1:]]:
            ax.add_patch(FancyBboxPatch((x0, l * DY - H / 2), w, H, boxstyle="round,pad=0,rounding_size=0.05", fc=GRID,
                                        ec="none", zorder=1))
    ax.text(-0.14, Y_IN, r"2 $\times$ 2 +", **{**TOK, "ha": "right"}, color=TEXT)
    ax.text(0, Y_IN, "4", **TOK, color=TEXT)
    for x, tok in zip(XS, gen + ["8"]):
        if tok == ".":  # filler: a dot marker, the font's period is a speck
            ax.plot(x, Y_OUT, "o", ms=2, color="#a8a7a2", zorder=5)
        else:
            ax.text(x, Y_OUT, tok, **{**TOK, "fontsize": 6.5 if len(tok) > 4 else 7.5}, color=TEXT if x == XS[-1] else GREEN)
    loop = "#cfcec9" if latent else GREEN
    for a, b in zip(XS, XS[1:]) if not latent else []:  # each sampled token back in at the next position, down the gap between columns
        g = (a + b) / 2
        draw(ax, rounded([(a + 0.12, Y_OUT), (g, Y_OUT), (g, Y_IN), (b, Y_IN), (b, -H / 2)], 0.14), lw=0.8, color=loop,
             zorder=1.5)
        ax.add_patch(FancyArrowPatch((g, DY / 2 + 0.1), (g, DY / 2 - 0.1), arrowstyle="-|>", mutation_scale=6.5, lw=0,
                                     color=loop, zorder=1.5))
    if latent:
        o, i = stream(XS[-1]), stream(XS[1])  # band = outer streamline, then the inner one reversed
        band = MPath(np.vstack([o.vertices, i.vertices[::-1]]), list(o.codes) + [MPath.LINETO] + list(i.codes[1:][::-1]))
        ax.add_patch(PathPatch(band, fc=tint(GREEN, 0.18), ec="none", zorder=2))
        for x in (XS[1], XS[-1]):  # the band's edges, solid where the flow enters and leaves
            draw(ax, stream(x), lw=0.7, color=tint(GREEN, 0.55))
        draw(ax, rounded([(0, Y_IN + 0.14), (0, 0), (XS[1] - 0.5, 0)], 0.5))
        draw(ax, MPath([(XS[-1], TOP), (XS[-1], Y_OUT - 0.2)]))
        for x in XS[:-1]:  # the filler is still sampled at every position, it just carries nothing
            draw(ax, MPath([(x, TOP + H / 2 + 0.04), (x, Y_OUT - 0.12)]), lw=0.8, color=loop, zorder=1.5)
    else:
        for x in XS:
            draw(ax, MPath([(x, Y_IN + (0.14 if x == 0 else 0.25)), (x, Y_OUT - 0.2)]))
    ax.add_patch(FancyArrowPatch((XS[-1], Y_OUT - 0.4), (XS[-1], Y_OUT - 0.14), arrowstyle="-|>", mutation_scale=7, lw=0,
                                 color=GREEN, zorder=4))
    ax.set_title(title)
    ax.set(xlim=(-1.05, XS[-1] + 0.4), ylim=(Y_IN - 0.15, Y_OUT + 0.15), aspect="equal")
    ax.axis("off")
save(fig, __file__, "reasoning_modes.png")

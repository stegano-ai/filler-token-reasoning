import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

from .rs_transformer import ARCHS

PROBLEMS = Path(__file__).resolve().parents[1] / "problems"
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cc7", "#8c6d31"]


def frontier(tasks: list[str], arch: str) -> tuple[dict, dict]:
    data, excluded = {}, {}
    for task in tasks:  # every idea folder with a result for this arch takes part
        results = [json.loads(f.read_text()) for f in sorted((PROBLEMS / task / "ideas").glob(f"*/results_{arch}.json"))]
        excluded[task] = [r["circuit"] for r in results if not r["depth_ok"] or not r["verified_n"]]
        best = {}
        for r in results:
            if not r["depth_ok"]:
                continue
            for n in r["verified_n"]:  # a circuit counts only at lengths where its correctness was checked
                L = r["layers"][str(n)]
                if n not in best or L < best[n]["layers"]:
                    best[n] = {"layers": L, "circuit": r["circuit"]}
        data[task] = dict(sorted(best.items()))
    return data, excluded


def main(tasks: list[str], archs: list[str], out_dir: str, name: str) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(archs), figsize=(7 * len(archs), 4.5), squeeze=False)
    report = {}
    for ax, arch in zip(axes[0], archs):
        data, excluded = frontier(tasks, arch)
        report[arch] = {"frontier": data, "excluded_no_verified_length": excluded}
        assert len(data) <= len(PALETTE), "extend PALETTE"
        for (task, best), color in zip(data.items(), PALETTE):
            ns = [n for n in best if best[n]["layers"] > 0]  # log axis: a 0-layer (readout-only) point is not drawable
            ax.plot(ns, [best[n]["layers"] for n in ns], color=color, lw=2, marker="o", ms=4, label=task)
            for i, n in enumerate(ns):
                if i == 0 or best[n]["circuit"] != best[ns[i - 1]]["circuit"]:
                    ax.annotate(best[n]["circuit"], (n, best[n]["layers"]), fontsize=7, color=color,
                                xytext=(3, 4), textcoords="offset points")
        A = ARCHS[arch]
        ax.axhline(A.layers, color="black", ls=":", lw=1)
        ax.annotate(f"{A.layers} layers, pattern {'/'.join(A.pattern)}, H={A.H}", (1024, A.layers), fontsize=7,
                    xytext=(0, 3), textcoords="offset points", ha="right")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_title(arch)
        ax.set_xlabel("instance length n")
        ax.set_ylabel("layers of shallowest circuit verified at n")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, fontsize=8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for task, best in data.items():
            logger.info("[%s] %s: %s", arch, task, {n: v["layers"] for n, v in best.items()})
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=150)
    (out / f"{name}.json").write_text(json.dumps(report, indent=1) + "\n")
    return report

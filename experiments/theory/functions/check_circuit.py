import dataclasses
import importlib.util
import json
import logging
import time
from pathlib import Path

import numpy as np

from .rs_transformer import ARCHS

logger = logging.getLogger(__name__)

PROBLEMS = Path(__file__).resolve().parents[1] / "problems"
CHECKPOINTS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024]


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_instances(circuit, data: list[dict], n: int, arch) -> dict:
    P = circuit.build(n, arch)
    fails = []
    for d in data:
        try:
            reasoning, output = P.readout(P.run(np.asarray(d["input"])))
        except AssertionError as e:  # a resource-limit violation is a failed check, not a crash
            fails.append({"input": d["input"], "gold": d["reasoning"], "got": f"resource violation: {e}"})
            break
        if reasoning != d["reasoning"] or output != d["output"]:
            fails.append({"input": d["input"], "gold": d["reasoning"], "got": reasoning})
    return {"n": n, "depth": P.depth, "tested": len(data), "failures": len(fails),
            "first_failure": fails[0] if fails else None}


def main(problem: str, circuit: str, arch: str, min_n: int, max_n: int, max_layers: int,
         quick_instances: int, final_seconds: float, seed: int) -> dict:
    A = ARCHS[arch]
    folder = PROBLEMS / problem
    task = load_module(folder / "problem.py")
    idea = folder / "ideas" / circuit
    circ = load_module(idea / "circuit.py")

    layers, claimed = {}, {}
    for n in [c for c in CHECKPOINTS if min_n <= c <= max_n]:
        layers[n], claimed[n] = circ.build(n, A).depth, circ.claimed_layers(n, A)
        if layers[n] > max_layers:
            break
    depth_ok = all(layers[n] == claimed[n] for n in layers)

    quick = []
    for n in layers:
        if layers[n] > max_layers:
            continue
        r = run_instances(circ, task.instances(n, quick_instances, seed + n), n, A)
        quick.append(r)
        if r["failures"]:
            break
    quick_ok = not any(r["failures"] for r in quick)
    verified_n = [r["n"] for r in quick if not r["failures"]]

    final = None
    if final_seconds > 0:
        t0 = time.time()
        variants = [A, dataclasses.replace(A, p=A.p - 1), dataclasses.replace(A, c_beta=1.5)]
        tested, failures, rnd = {}, [], 0
        runnable = [n for n in layers if layers[n] <= max_layers]
        while time.time() - t0 < final_seconds:
            for n in runnable:
                if time.time() - t0 >= final_seconds:
                    break
                av = variants[rnd % len(variants)]
                r = run_instances(circ, task.instances(n, 1, seed + 10_000 * (rnd + 1) + n), n, av)
                tested[n] = tested.get(n, 0) + 1
                if r["failures"]:
                    failures.append({**r["first_failure"], "n": n, "arch": dataclasses.asdict(av)})
            rnd += 1
        final = {"seconds": round(time.time() - t0, 1), "tested_per_n": tested,
                 "n_failures": len(failures), "failures": failures[:5]}

    result = {"problem": problem, "circuit": circuit, "arch": arch, "arch_params": dataclasses.asdict(A),
              "layers": layers, "claimed_layers": claimed, "depth_ok": depth_ok,
              "quick": quick, "quick_ok": quick_ok, "verified_n": verified_n, "final": final,
              "pass": depth_ok and quick_ok and (final is None or final["n_failures"] == 0)}
    (idea / f"results_{arch}.json").write_text(json.dumps(result, indent=1) + "\n")
    logger.info("%s/%s [%s] depth_ok=%s verified_n=%s pass=%s", problem, circuit, arch, depth_ok, verified_n, result["pass"])
    return result

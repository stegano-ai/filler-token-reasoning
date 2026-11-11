import importlib
import sys
from pathlib import Path

import pytest
import yaml

THEORY = Path(__file__).resolve().parents[1] / "experiments" / "theory"
sys.path.insert(0, str(THEORY))

cc = importlib.import_module("functions.check_circuit")

BASE = yaml.safe_load((THEORY / "configs" / "base.yaml").read_text())["function_kwargs"]
SLOW = {"horn"}
CIRCUITS = [(p.parents[2].name, p.parent.name) for p in THEORY.glob("problems/*/ideas/*/circuit.py")]


def kwargs(problem: str, circuit: str) -> dict:
    meta = yaml.safe_load((THEORY / "configs" / f"{problem}.yaml").read_text())
    override = next(e["config"]["function_kwargs"] for e in meta["experiments"]
                    if e["config"]["function_kwargs"].get("circuit") == circuit)
    kw = {**BASE, **override, "problem": problem, "circuit": circuit, "final_seconds": 0, "arch": "bf16"}
    kw["max_n"] = min(kw["max_n"], 64)
    kw["quick_instances"] = 3
    return kw


@pytest.fixture(autouse=True)
def no_json_writes(monkeypatch):
    monkeypatch.setattr(cc.Path, "write_text", lambda self, s: None)


@pytest.mark.parametrize("problem,circuit", [pytest.param(p, c, marks=[pytest.mark.slow] * (p in SLOW))
                                              for p, c in CIRCUITS], ids=[f"{p}/{c}" for p, c in CIRCUITS])
def test_circuit_passes(problem, circuit):
    assert cc.main(**kwargs(problem, circuit))["pass"]


def test_wrong_depth_claim_fails(monkeypatch):
    real = cc.load_module

    def patched(path):
        mod = real(path)
        if path.parent.name == "serial":
            mod.claimed_layers = lambda n, arch: n + 2
        return mod

    monkeypatch.setattr(cc, "load_module", patched)
    assert not cc.main(**kwargs("function_composition", "serial"))["pass"]

import numpy as np

from functions.rs_transformer import BF16, Clause, Program, Q
from problems.gl32.group import STEP


def claimed_layers(n: int, arch) -> int:
    return n - 1


def build(n: int, arch) -> Program:
    n_scr = int(n == 1)
    n_pos = n + n_scr
    fields = {"tok": 2, "pos": n_pos, "out": 8}

    def embed(tokens, n_pos):
        tok = np.zeros(n_pos, dtype=np.int64)
        tok[:n] = tokens
        return {"tok": tok, "pos": np.arange(n_pos), "out": STEP[tok, 1]}

    def readout(st):
        vals = st["out"][:n].tolist()
        return vals, vals

    P = Program(arch, fields, n, n_scr, embed, readout)
    for _ in range(n - 1):
        lay = P.layer()
        lay.head([Clause(arch.M, {"pos": Q("pos", -1)}), Clause(-arch.M, {"pos": Q("pos")})], ["out"])
        lay.mlp(["tok", "pos", "out"],
                lambda f, h: {"out": np.where(f["pos"] >= 1, STEP[f["tok"], h[0].sym("out")], f["out"])})
    return P

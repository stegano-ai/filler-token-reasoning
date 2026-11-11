import numpy as np

from functions.rs_transformer import BF16, Clause, Program, Q
from problems.gl32.group import APPLY1, INS, MUL

W1, R = BF16.H + 1, BF16.mlp_bits // 8
E2G = np.array([0, INS[0], INS[1]])


def claimed_layers(n: int, arch) -> int:
    L, W = 1, W1
    while W < n:
        L, W = L + 1, R * W
    return L


def build(n: int, arch) -> Program:
    L = claimed_layers(n, arch)
    n_pos = n + 1
    fields = {"tok": 2, "e": 3, "pos": n_pos, "ipos": n_pos + 1, "one": 2, "out": 8,
              **({"g": len(MUL)} if L >= 2 else {})}
    M = arch.M

    def embed(tokens, n_pos):
        tok = np.zeros(n_pos, dtype=np.int64)
        tok[1:] = tokens
        pos = np.arange(n_pos)
        return {"tok": tok, "e": np.where(pos >= 1, 1 + tok, 0), "pos": pos,
                "ipos": np.where(pos >= 1, pos, n_pos), "one": np.ones(n_pos, dtype=np.int64)}

    def readout(st):
        vals = st["out"][1:].tolist()
        return vals, vals

    def fold(g, parts):
        for part in parts:
            g = MUL[g, part]
        return {"g": g, "out": APPLY1[g]} if L >= 2 else {"out": APPLY1[g]}

    P = Program(arch, fields, n, 1, embed, readout)
    W = 1
    for j in range(L):
        r, src = (W1, "e") if j == 0 else (R, "g")
        lay = P.layer()
        for m in range(1, r):
            lay.head([Clause(-M, {"one": 1}), Clause(2 * M, {"ipos": Q("pos", -m * W)}), Clause(M, {"pos": 0})], [src])
        if j == 0:
            lay.mlp(["e"], lambda f, h: fold(E2G[f["e"]], [E2G[o.sym("e")] for o in h]))
        else:
            lay.mlp(["g"], lambda f, h: fold(f["g"], [o.sym("g") for o in h]))
        W *= r
    return P

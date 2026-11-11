import math

import numpy as np

from functions.rs_transformer import BF16, Clause, Program, Q

D = 4


def claimed_layers(n: int, arch) -> int:
    return 2 + math.ceil(math.log2(n)) if n > 1 else 2


def build(n: int, arch) -> Program:
    n_pos = 1 + n * D
    G = [f"g{v}" for v in range(D)]
    fields = {"tok": D, "pos": n_pos, "table": n + 1, "slot": D, "out": D, **{g: D for g in G}}

    def embed(tokens, n_pos):
        pos = np.arange(n_pos)
        return {"tok": np.asarray(tokens), "pos": pos,
                "table": np.where(pos >= 1, (pos - 1) // D + 1, 0),
                "slot": np.where(pos >= 1, (pos - 1) % D, 0)}

    def readout(st):
        vals = [int(st["out"][k * D]) for k in range(1, n + 1)]
        return vals, vals

    P = Program(arch, fields, n_pos, 0, embed, readout)
    M = arch.M
    isT = lambda f: (f["slot"] == D - 1) & (f["table"] >= 1)

    lay = P.layer()
    for v in range(D):
        lay.head([Clause(M, {"table": Q("table"), "slot": v})], ["tok"])
    lay.mlp(["table", "slot"] + G,
            lambda f, h: {g: np.where(isT(f), h[v].sym("tok"), 0) for v, g in enumerate(G)})

    for j in range(claimed_layers(n, arch) - 2):
        lay = P.layer()
        lay.head([Clause(M, {"pos": Q("pos", -D * 2 ** j), "slot": D - 1})], G)

        def fn(f, h, j=j):
            own = np.stack([f[g] for g in G], 1)
            partner = np.stack([h[0].sym(g) for g in G], 1)
            composed = own[np.arange(len(own))[:, None], partner]
            valid = isT(f) & (f["table"] > 2 ** j)
            return {g: np.where(valid, composed[:, v], f[g]) for v, g in enumerate(G)}

        lay.mlp(["table", "slot"] + G, fn)

    lay = P.layer()
    lay.head([Clause(M, {"pos": 0})], ["tok"])
    lay.mlp(["table", "slot"] + G,
            lambda f, h: {"out": np.where(isT(f), np.stack([f[g] for g in G], 1)[np.arange(n_pos), h[0].sym("tok")], 0)})
    return P

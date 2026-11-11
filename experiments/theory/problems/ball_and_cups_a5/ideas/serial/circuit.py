import numpy as np

from functions.rs_transformer import Clause, Program, Q

PERMS = np.array([[1, 2, 3, 4, 0], [1, 0, 2, 4, 3]])  # rotate, double-swap: c -> image of c


def claimed_layers(n: int, arch) -> int:
    return n


def build(n: int, arch) -> Program:
    n_pos = n + 1  # one pad position so alphabet 5 fits the 4*n_pos^2 cap at n = 1
    fields = {"tok": 2, "pos": n_pos, "cur": 5}

    def embed(tokens, n_pos):
        tok = np.zeros(n_pos, dtype=np.int64)
        tok[:n] = tokens
        return {"tok": tok, "pos": np.arange(n_pos)}

    def readout(st):
        vals = st["cur"][:n].tolist()
        return vals, vals

    P = Program(arch, fields, n, 1, embed, readout)
    for L in range(1, n + 1):
        lay = P.layer()
        if L >= 2:
            lay.head([Clause(arch.M, {"pos": Q("pos", -1)})], ["cur"])

        def fn(f, h, L=L):
            prev = h[0].sym("cur") if L >= 2 else 0
            return {"cur": np.where(f["pos"] == L - 1, PERMS[f["tok"], prev], f["cur"])}

        lay.mlp(["tok", "pos", "cur"], fn)
    return P

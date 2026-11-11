import numpy as np

from functions.rs_transformer import BF16, Clause, Program, Q

MOD = 2


def claimed_layers(n: int, arch) -> int:
    return n - 1


def build(n: int, arch) -> Program:
    fields = {"tok": MOD, "pos": n, "s": MOD}

    def embed(tokens, n_pos):
        return {"tok": np.asarray(tokens), "pos": np.arange(n_pos), "s": np.asarray(tokens)}

    def readout(st):
        vals = st["s"].tolist()
        return vals, vals[-1:]

    P = Program(arch, fields, n, 0, embed, readout)
    for L in range(1, n):
        lay = P.layer()
        lay.head([Clause(arch.M, {"pos": Q("pos", -1)})], ["s"])
        # after layer L position i holds the sum of tok[i-L..i]; only positions i >= L still need to grow
        lay.mlp(["pos", "s", "tok"],
                lambda f, h, L=L: {"s": np.where(f["pos"] >= L, (np.round(h[0].frac("s")).astype(np.int64) + f["tok"]) % MOD, f["s"])})
    return P

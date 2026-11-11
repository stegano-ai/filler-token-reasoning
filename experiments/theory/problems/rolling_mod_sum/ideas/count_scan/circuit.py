import math

import numpy as np

from functions.rs_transformer import BF16, Clause, Program, Q

MOD = 2
C = 64                 # keys averaged per head: k/C stays exact after rounding to 2^-7
H = BF16.H
R = H * C              # radix per layer


def claimed_layers(n: int, arch) -> int:
    return math.ceil(math.log(n, R))


def build(n: int, arch) -> Program:
    L = claimed_layers(n, arch)
    fields = {"tok": MOD, "pos": n, "s": MOD}
    for l in range(1, L + 1):
        fields[f"blk{l}"] = n // R**l + 1
        fields[f"sub{l}"] = H
        fields[f"end{l}"] = 2

    def embed(tokens, n_pos):
        pos = np.arange(n_pos)
        st = {"tok": np.asarray(tokens), "pos": pos, "s": np.asarray(tokens)}
        for l in range(1, L + 1):
            u = R ** (l - 1)
            st[f"blk{l}"] = pos // (u * R)
            st[f"sub{l}"] = (pos // u) % R // C
            st[f"end{l}"] = pos % u == u - 1
        return st

    def readout(st):
        vals = st["s"].tolist()
        return vals, vals[-1:]

    P = Program(arch, fields, n, 0, embed, readout)
    for l in range(1, L + 1):
        u = R ** (l - 1)
        lay = P.layer()
        for j in range(H):
            lay.head([Clause(arch.M, {f"blk{l}": Q(f"blk{l}"), f"sub{l}": j, f"end{l}": 1})], ["s"])

        def fn(f, h, u=u):
            q = (f["pos"] // u) % R           # index of own level-(l-1) block within the level-l block
            own_end = f["pos"] % u == u - 1
            total = np.where(own_end, 0, f["s"])
            for j in range(H):
                cnt = np.where(j < q // C, C, np.where(j == q // C, q % C + own_end, 0))
                total = total + np.round(h[j].frac("s") * cnt).astype(np.int64)
            return {"s": total % MOD}

        lay.mlp(["pos", "s"], fn)
    return P

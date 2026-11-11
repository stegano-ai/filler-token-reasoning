import math

import numpy as np

from functions.rs_transformer import Clause, Program, Q
from problems.gl32.group import APPLY1, INS, MUL

N = len(MUL)
COMP = np.zeros((N + 1, N + 1), dtype=np.int64)  # slot N = identity, so sym()'s -1 (no match) composes as identity
COMP[:N, :N] = MUL
COMP[N, :] = np.arange(N + 1)
COMP[:, N] = np.arange(N + 1)
A1 = np.append(APPLY1, APPLY1[0])
R = 4  # windows merged per layer: own + 3 partners = 7.4 + 3*8 = 31.4 <= 32 mlp_bits


def first_block(arch) -> int:
    b = 1
    while b < arch.H + 1 and 1 + math.log2(b + 2) + b <= arch.mlp_bits:
        b += 1
    return b


def _steps(n: int, arch, B1: int) -> list:
    steps, B = [("fold", B1)], B1
    while B < n and len(steps) < arch.max_depth:
        kind = arch.pattern[len(steps) % len(arch.pattern)]
        lim = arch.window - 1 if kind == "window" else n
        r = 1 if kind == "linear" else max(q for q in range(1, R + 1) if (q - 1) * B <= lim)
        if r > 1:
            steps.append(("step", B, r))
            B *= r
        else:
            steps.append(("noop",))
    return steps


def schedule(n: int, arch) -> list:
    cands = [_steps(n, arch, b) for b in range(1, first_block(arch) + 1)]
    return min(cands, key=lambda s: (len(s), -s[0][1]))


def claimed_layers(n: int, arch) -> int:
    return len(schedule(n, arch))


def build(n: int, arch) -> Program:
    sched = schedule(n, arch)
    B1 = sched[0][1]
    n_pos = max(n + 1, 3)
    has_g = n_pos >= 7  # alphabet 169 needs 4 * n_pos^2 >= 169
    fields = {"tok": 2, "pos": n_pos, "cnt": B1 + 1, "out": 8, "anch": 2}
    if has_g:
        fields["g"] = N + 1

    def embed(tokens, n_pos):
        pos = np.arange(n_pos)
        real = (pos >= 1) & (pos <= n)
        tok = np.zeros(n_pos, dtype=np.int64)
        tok[1:1 + n] = tokens
        return {"tok": tok, "pos": pos, "cnt": np.where(real, np.minimum(pos, B1), 0),
                "anch": (pos == 0).astype(np.int64)}

    def readout(st):
        vals = st["out"][1:1 + n].tolist()
        return vals, vals

    def write(acc):
        return {"out": A1[acc], "g": acc} if has_g else {"out": A1[acc]}

    P = Program(arch, fields, n, n_pos - n, embed, readout)
    for s in sched:
        lay = P.layer()
        if s[0] == "noop":
            continue
        if s[0] == "fold":
            for m in range(1, min(B1, n)):
                lay.head([Clause(arch.M, {"pos": Q("pos", -m)})], ["tok"])

            def fn(f, h):
                acc = np.full(len(f["tok"]), N, dtype=np.int64)
                for m in reversed(range(len(h) + 1)):
                    t = f["tok"] if m == 0 else h[m - 1].sym("tok")
                    acc = np.where(m < f["cnt"], COMP[INS[t], acc], acc)
                return write(acc)

            lay.mlp(["tok", "cnt"], fn)
            continue
        B, r = s[1], s[2]
        for m in range(1, r):  # tiers 4 / 2 / 0: the partner mB back, the identity anchor, the rest
            lay.head([Clause(arch.M, {"pos": Q("pos", -m * B), "anch": 0}), Clause(2, {"anch": 1})], ["g"])

        def fn(f, h):
            acc = f["g"]
            for hd in h:
                acc = COMP[acc, hd.sym("g")]
            return write(acc)

        lay.mlp(["g"], fn)
    return P

import math

import numpy as np

from functions.rs_transformer import Clause, Program, Q

ROT, SWAP = (1, 2, 3, 4, 0), (1, 0, 2, 4, 3)
ELEMS = [tuple(range(5))]
for a in ELEMS:  # closure of {rotate, double-swap} = A5 (60 elements), a[c] = image of cup c
    for gen in (ROT, SWAP):
        if (b := tuple(gen[a[c]] for c in range(5))) not in ELEMS:
            ELEMS.append(b)
IDX = {e: i for i, e in enumerate(ELEMS)}
COMP = np.zeros((61, 61), dtype=np.int64)  # slot 60 = identity, so sym()'s -1 (no match) composes as identity
for ia in range(61):
    for ib in range(61):
        COMP[ia, ib] = ib if ia == 60 else ia if ib == 60 else IDX[tuple(ELEMS[ia][ELEMS[ib][c]] for c in range(5))]
INS = np.array([IDX[ROT], IDX[SWAP], 0])
IMG0 = np.array([e[0] for e in ELEMS])
R = 5  # blocks merged per layer: own + 4 partners = 5.93 + 4*6 = 29.93 <= 32 mlp_bits


def first_block(arch) -> int:
    b = 1
    while b < arch.H + 1 and 1 + math.log2(b + 2) + b <= arch.mlp_bits:
        b += 1
    return b


def schedule(n: int, arch) -> list:
    B = first_block(arch)
    steps, level = [("fold", 1, B)], 1
    while B < n:
        kind = arch.pattern[len(steps) % len(arch.pattern)]
        keys = -(-n // B) + 1
        if kind == "full" or (kind == "linear" and keys <= arch.lin_keys):
            steps.append(("step", level, B, kind))
            level, B = level + 1, B * R
        else:
            steps.append(("noop",))
    return steps


def claimed_layers(n: int, arch) -> int:
    return len(schedule(n, arch))


def build(n: int, arch) -> Program:
    sched = schedule(n, arch)
    B1 = sched[0][2]
    n_pos = max(n + 1, 4)  # anchor at position 0, tokens at 1..n; >= 4 so alphabet 61 fits 4*n_pos^2
    sizes = {1: B1}
    for s in sched:
        if s[0] == "step":
            sizes[s[1] + 1] = s[2] * R
    flags = sorted({s[1] for s in sched if s[0] == "step" and s[3] != "linear"})
    fields = {"tok": 2, "pos": n_pos, "pm": B1, "cnt": B1 + 1, "g": 61, "anch": 2}
    fields.update({f"k{L}": n_pos + 1 for L in sizes})
    fields.update({f"e{L}": 2 for L in flags})

    def embed(tokens, n_pos):
        pos = np.arange(n_pos)
        i = pos - 1
        real = (i >= 0) & (i < n)
        tok = np.zeros(n_pos, dtype=np.int64)
        tok[1:1 + n] = tokens
        st = {"tok": tok, "pos": pos, "pm": np.where(real, i % B1, 0),
              "cnt": np.where(real, i % B1 + 1, 0), "anch": (pos == 0).astype(np.int64)}
        for L, B in sizes.items():
            st[f"k{L}"] = np.where(real, i // B + 1, 0)
            if L in flags:
                st[f"e{L}"] = (real & (i % B == B - 1) | (pos == 0)).astype(np.int64)
        return st

    def readout(st):
        vals = IMG0[st["g"][1:1 + n]].tolist()
        return vals, vals

    P = Program(arch, fields, n, n_pos - n, embed, readout)
    for s in sched:
        lay = P.layer()
        assert s[0] != "step" or lay.kind == s[3]
        if s[0] == "noop":
            continue
        if s[0] == "fold":
            ref = "pm" if lay.kind == "linear" else "pos"
            for m in range(1, min(B1, n)):
                lay.head([Clause(arch.M, {ref: Q(ref, -m)})], ["tok"])

            def fn(f, h):
                acc = np.zeros(n_pos, dtype=np.int64)
                for m in reversed(range(len(h) + 1)):
                    t = f["tok"] if m == 0 else h[m - 1].sym("tok")
                    acc = np.where(m < f["cnt"], COMP[INS[t], acc], acc)
                return {"g": acc}

            lay.mlp(["tok", "cnt"], fn)
            continue
        L = s[1]
        a, b = f"k{L}", f"k{L + 1}"
        for m in range(1, R):
            if lay.kind == "linear":
                lay.head([Clause(arch.M, {a: Q(a, -m), b: Q(b, 0)})], ["g"])
            else:  # score tiers 4 / 2 / 0 / -2: the partner, the identity anchor, other block ends, the rest
                lay.head([Clause(arch.M, {a: Q(a, -m), b: Q(b, 0), f"e{L}": 1}),
                          Clause(2, {"anch": 1}), Clause(-2, {f"e{L}": 0})], ["g"])

        def fn(f, h):
            acc = f["g"]
            for hd in h:
                acc = COMP[acc, hd.sym("g")]
            return {"g": acc}

        lay.mlp(["g"], fn)
    return P

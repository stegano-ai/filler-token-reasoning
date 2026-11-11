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
R = 4  # blocks merged per layer: own + 3 partners = 7.4 + 3*8 = 31.4 <= 32 mlp_bits


def first_block(arch) -> int:
    b = 1
    while b < arch.H + 1 and 1 + math.log2(b + 2) + b <= arch.mlp_bits:
        b += 1
    return b


def schedule(n: int, arch) -> list:
    B = first_block(arch)
    steps, level = [("fold", 1, B)], 1
    while B < n and len(steps) < arch.max_depth:
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
    n_pos = max(n + 1, 3)
    has_g = n_pos >= 7  # alphabet 169 needs 4 * n_pos^2 >= 169
    sizes = {1: B1}
    for s in sched:
        if s[0] == "step":
            sizes[s[1] + 1] = s[2] * R
    flags = sorted({s[1] for s in sched if s[0] == "step" and s[3] != "linear"})
    fields = {"tok": 2, "pos": n_pos, "pm": B1, "cnt": B1 + 1, "out": 8, "anch": 2}
    fields.update({f"k{L}": n_pos + 1 for L in sizes})
    fields.update({f"e{L}": 2 for L in flags})
    if has_g:
        fields["g"] = N + 1

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
        vals = st["out"][1:1 + n].tolist()
        return vals, vals

    def write(acc):
        return {"out": A1[acc], "g": acc} if has_g else {"out": A1[acc]}

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
                acc = np.full(len(f["tok"]), N, dtype=np.int64)
                for m in reversed(range(len(h) + 1)):
                    t = f["tok"] if m == 0 else h[m - 1].sym("tok")
                    acc = np.where(m < f["cnt"], COMP[INS[t], acc], acc)
                return write(acc)

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
            return write(acc)

        lay.mlp(["g"], fn)
    return P

import math

import numpy as np

from functions.rs_transformer import Clause, Program, Q

MOD = 2
C = 64                 # keys averaged per counting head: k/C stays exact after rounding to 2^-7


def _count_heads(arch) -> int:
    h = arch.H
    while h > 1 and h + math.log2(h * C) + 1 > arch.mlp_bits:
        h -= 1
    return h


def _radix(kind: str, u: int, n: int, arch) -> int:
    if kind == "linear":
        return arch.H + 1 if math.ceil(n / u) <= arch.lin_keys else arch.H // 2 + 1
    r = _count_heads(arch) * C
    return max(min(r, arch.window // u) if kind == "window" else r, 1)


def plan(n: int, arch) -> list:
    steps, u, d = [], 1, 0
    while u < n:
        kind = arch.pattern[d % len(arch.pattern)]
        r = _radix(kind, u, n, arch)
        steps.append((kind, u, r))
        u, d = u * r, d + 1
    return steps


def claimed_layers(n: int, arch) -> int:
    return len(plan(n, arch))


def build(n: int, arch) -> Program:
    steps = plan(n, arch)
    fields, meta = {"tok": MOD, "s": MOD, "pos": max(n, 2)}, []
    for l, (kind, u, r) in enumerate(steps):
        nb = math.ceil(n / u)
        rq = min(r, nb)
        if r == 1:
            meta.append(None)
        elif kind == "linear":
            plain = nb <= arch.lin_keys
            meta.append({"kind": kind, "u": u, "r": r, "plain": plain,
                         "kmax": min(arch.H if plain else arch.H // 2, rq - 1)})
            fields[f"b{l}"] = nb if plain else arch.lin_keys
            fields[f"q{l}"] = rq
        else:
            nh = (rq - 2) // C + 1
            meta.append({"kind": kind, "u": u, "r": r, "nh": nh})
            fields[f"hi{l}"] = (nb - 1) // r + 1
            fields[f"sub{l}"] = nh + 1
            fields[f"q{l}"] = rq

    def embed(tokens, n_pos):
        pos = np.arange(n_pos)
        st = {"tok": np.asarray(tokens), "s": np.asarray(tokens), "pos": pos}
        for l, m in enumerate(meta):
            if m is None:
                continue
            blk, u, r = pos // m["u"], m["u"], m["r"]
            st[f"q{l}"] = blk % r
            if m["kind"] == "linear":
                st[f"b{l}"] = blk if m["plain"] else blk % arch.lin_keys
            else:
                st[f"hi{l}"] = blk // r
                st[f"sub{l}"] = np.where(pos % u == u - 1, (blk % r) // C, m["nh"])
        return st

    def readout(st):
        vals = st["s"].tolist()
        return vals, vals[-1:]

    P = Program(arch, fields, n, 0, embed, readout)
    for l, m in enumerate(meta):
        lay = P.layer()
        if m is None:
            continue
        qn = f"q{l}"
        if m["kind"] == "linear":
            bn = f"b{l}"
            for k in range(1, m["kmax"] + 1):
                lay.head([Clause(arch.M, {bn: Q(bn, -k)})], ["s"])
                if not m["plain"]:  # second head for the residues that wrapped past 0
                    lay.head([Clause(arch.M, {bn: Q(bn, arch.lin_keys - k)})], ["s"])

            def fn(f, h, m=m, qn=qn):
                tot = f["s"]
                for i in range(m["kmax"]):
                    v = h[i].sym("s") if m["plain"] else np.where(h[2 * i].sym("s") >= 0, h[2 * i].sym("s"), h[2 * i + 1].sym("s"))
                    tot = tot + np.where((i + 1 <= f[qn]) & (v >= 0), np.maximum(v, 0), 0)
                return {"s": tot % MOD}
        else:
            for j in range(m["nh"]):
                lay.head([Clause(arch.M, {f"hi{l}": Q(f"hi{l}"), f"sub{l}": j}),
                          Clause(-arch.M, {"pos": Q("pos")})], ["s"])

            def fn(f, h, m=m, qn=qn):
                tot = f["s"]
                for j in range(m["nh"]):
                    cnt = np.clip(f[qn] - j * C, 0, C)
                    # a head that matched no key averages an empty set; it contributes nothing
                    tot = tot + np.round(np.where(cnt > 0, h[j].frac("s"), 0.0) * cnt).astype(np.int64)
                return {"s": tot % MOD}

        lay.mlp([qn, "s"], fn)
    return P

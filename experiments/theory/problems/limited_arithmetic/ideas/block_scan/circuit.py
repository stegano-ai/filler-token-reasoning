import numpy as np

from functions.rs_transformer import Clause, Program, Q

MOD = 8
R = 5
A_VALS = np.array([0, 1, 3, 5, 7])
A_IDX = np.zeros(MOD, dtype=np.int64)
A_IDX[A_VALS] = np.arange(5)


def first_radix(arch) -> int:
    # a linear layer 0 gates itself, a softmax one needs the gate field in the MLP and one head fewer
    return R if arch.pattern[0] == "linear" else 4


def levels(n: int, arch) -> list[int]:
    out, B = [], 1
    while B < n + 1:
        out.append(B)
        B *= first_radix(arch) if len(out) == 1 else R
    return out


def claimed_layers(n: int, arch) -> int:
    return len(levels(n, arch))


def unpack(ai, b):  # a head that found no key gives sym = -1: the partner block is outside mine, so identity
    return np.where(ai >= 0, A_VALS[np.maximum(ai, 0)], 1), np.where(ai >= 0, b, 0)


def pack(a, b):
    return {"a": A_IDX[a % MOD], "b": b % MOD}


def compose(first, then):
    return (then[0] * first[0]) % MOD, (then[0] * first[1] + then[1]) % MOD


def elem(op, opr):
    a = np.select([opr < 0, opr >= 8, op == 1, op == 2, op == 3], [1, 0, 1, 1, opr], 0) % MOD
    b = np.select([opr < 0, opr >= 8, op == 1, op == 2], [0, opr & 7, opr, -opr], 0) % MOD
    return a, b


def gated(a, b, on):
    return np.where(on, a, 1), np.where(on, b, 0)


def build(n: int, arch) -> Program:
    n_pos = 2 * n + 1
    Bs = levels(n, arch)
    r0 = first_radix(arch)
    pos = np.arange(n_pos)
    e = (pos + 1) // 2
    even = pos % 2 == 0
    kinds = [arch.pattern[i % len(arch.pattern)] for i in range(len(Bs))]
    fields = {"a": 5, "b": MOD, "pos": n_pos, "op": 4, "opr": 16, "one": 2}
    aux = {"pos": pos, "one": np.ones(n_pos, dtype=np.int64)}
    if kinds[0] == "linear":
        fields["ce"], aux["ce"] = R + 1, np.where(even, e % R, R)
        fields["co"], aux["co"] = R + 1, np.where(even, R, e % R)
    else:
        fields["g0"], aux["g0"] = r0, e % r0
    for i, B in enumerate(Bs[1:], 1):
        idx = e // B
        if kinds[i] == "linear":
            fields[f"c{i}"], aux[f"c{i}"] = R + 1, np.where(even, idx % R, R)
        else:
            nb = int(idx.max()) + 1
            fields[f"i{i}"], aux[f"i{i}"] = nb + 1, idx
            fields[f"e{i}"], aux[f"e{i}"] = nb + 1, np.where(even & ((e + 1) % B == 0), idx, nb)
            fields[f"g{i}"], aux[f"g{i}"] = R, idx % R

    def embed(tokens, n_pos):
        tok = np.asarray(tokens)
        is_op = tok >= 8
        return {**aux, "op": np.where(is_op, tok - 7, 0),
                "opr": np.where(is_op, 0, np.where(pos == 0, tok + 8, tok))}

    def readout(st):
        vals = [int(st["b"][2 * k]) for k in range(1, n + 1)]
        return vals, vals[-1:]

    P = Program(arch, fields, n_pos, 0, embed, readout)
    M = arch.M
    bg = Clause(-M, {"one": 1})  # softmax heads only: pushes the background to -M, doubling the score gap
    for i, B in enumerate(Bs):
        lay = P.layer()
        if i == 0 and kinds[0] == "linear":
            lay.head([Clause(M, {"pos": Q("pos", -1)})], ["op"])
            for t in range(1, r0):
                lay.head([Clause(M, {"ce": Q("ce", -t)})], ["opr"])
                lay.head([Clause(M, {"co": Q("ce", -t)})], ["op"])

            def fn0(f, h):
                acc = elem(h[0].sym("op"), f["opr"])
                for t in range(1, r0):
                    acc = compose(elem(h[2 * t].sym("op"), h[2 * t - 1].sym("opr")), acc)
                return pack(*acc)

            lay.mlp(["opr"], fn0)
        elif i == 0:
            lay.head([bg, Clause(2 * M, {"pos": Q("pos", -1)})], ["op"])
            for t in range(1, r0):
                lay.head([bg, Clause(2 * M, {"pos": Q("pos", -2 * t)})], ["opr"])
                lay.head([bg, Clause(2 * M, {"pos": Q("pos", -2 * t - 1)})], ["op"])

            def fn0s(f, h):
                acc = elem(h[0].sym("op"), f["opr"])
                for t in range(1, r0):
                    acc = compose(gated(*elem(h[2 * t].sym("op"), h[2 * t - 1].sym("opr")), f["g0"] >= t), acc)
                return pack(*acc)

            lay.mlp(["opr", "g0"], fn0s)
        elif kinds[i] == "linear":
            for t in range(1, R):
                lay.head([Clause(M, {f"c{i}": Q(f"c{i}", -t)})], ["a", "b"])

            def fn(f, h):
                acc = unpack(f["a"], f["b"])
                for t in range(1, R):
                    acc = compose(unpack(h[t - 1].sym("a"), h[t - 1].sym("b")), acc)
                return pack(*acc)

            lay.mlp(["a", "b"], fn)
        else:
            for t in range(1, R):
                lay.head([bg, Clause(2 * M, {f"e{i}": Q(f"i{i}", -t)})], ["a", "b"])

            def fnf(f, h, i=i):
                acc = unpack(f["a"], f["b"])
                for t in range(1, R):
                    acc = compose(gated(*unpack(h[t - 1].sym("a"), h[t - 1].sym("b")), f[f"g{i}"] >= t), acc)
                return pack(*acc)

            lay.mlp(["a", "b", f"g{i}"], fnf)
    return P

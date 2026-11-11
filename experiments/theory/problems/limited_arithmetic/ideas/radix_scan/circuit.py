import numpy as np

from functions.rs_transformer import BF16, Clause, Program, Q

MOD = 8
R0, R = 4, 5  # layer 1: own op + 3 (op, operand) partner pairs = 7 heads; later: own + 4 partners x 6 bits = 30 bits


def blocks(n: int) -> list[int]:
    out = [R0]
    while out[-1] < n + 1:
        out.append(out[-1] * R)
    return out


def claimed_layers(n: int, arch) -> int:
    return len(blocks(n))


def elem(op, opr):  # (op code, operand code) -> affine map x -> a x + b; opr >= 8 marks the start constant
    start = opr >= 8
    a = np.select([start, op == 1, op == 2, op == 3], [0, 1, 1, opr], 0) % MOD
    b = np.select([start, op == 1, op == 2], [opr & 7, opr, -opr], 0) % MOD
    return a, b


def compose(first, then):  # apply `first`, then `then`; once `then` holds the start constant (a = 0) `first` is irrelevant
    return (then[0] * first[0]) % MOD, (then[0] * first[1] + then[1]) % MOD


def build(n: int, arch) -> Program:
    n_pos = 2 * n + 1
    fields = {"op": 4, "opr": 16, "pos": n_pos, "a": MOD, "b": MOD}
    M = arch.M

    def embed(tokens, n_pos):
        tok, pos = np.asarray(tokens), np.arange(n_pos)
        is_op = tok >= 8
        return {"op": np.where(is_op, tok - 7, 0), "opr": np.where(is_op, 0, np.where(pos == 0, tok + 8, tok)),
                "pos": pos, "a": 0, "b": np.where(pos == 0, tok, 0)}

    def readout(st):
        vals = [int(st["b"][2 * k]) for k in range(1, n + 1)]
        return vals, vals[-1:]

    P = Program(arch, fields, n_pos, 0, embed, readout)

    lay = P.layer()
    lay.head([Clause(M, {"pos": Q("pos", -1)})], ["op"])
    for t in range(1, R0):
        lay.head([Clause(M, {"pos": Q("pos", -2 * t)})], ["opr"])
        lay.head([Clause(M, {"pos": Q("pos", -2 * t - 1)})], ["op"])

    def fn1(f, h):
        acc = elem(h[0].sym("op"), f["opr"])
        for t in range(1, R0):  # partners below pos 0 are garbage but acc already has a = 0 by then
            acc = compose(elem(h[2 * t].sym("op"), h[2 * t - 1].sym("opr")), acc)
        return {"a": acc[0], "b": acc[1]}

    lay.mlp(["opr"], fn1)

    for B in blocks(n)[:-1]:
        lay = P.layer()
        for t in range(1, R):
            lay.head([Clause(M, {"pos": Q("pos", -2 * t * B)})], ["a", "b"])

        def fn(f, h):
            acc = f["a"], f["b"]
            for t in range(1, R):
                acc = compose((h[t - 1].sym("a"), h[t - 1].sym("b")), acc)
            return {"a": acc[0], "b": acc[1]}

        lay.mlp(["a", "b"], fn)
    return P

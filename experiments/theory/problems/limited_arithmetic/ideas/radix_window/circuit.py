import numpy as np

from functions.rs_transformer import Clause, Program, Q

MOD = 8
R = 5
A_VALS = np.array([0, 1, 3, 5, 7])
A_IDX = np.zeros(MOD, dtype=np.int64)
A_IDX[A_VALS] = np.arange(5)


def schedule(n: int, arch) -> list[int]:
    # radix of each layer: the shallowest way to reach coverage n+1 when a window layer can only
    # reach 2*t*L positions back, so its radix is capped once the block L it composes gets long
    target, cur, i = n + 1, {1: []}, 0
    while True:
        reached = [s for B, s in cur.items() if B >= target]
        if reached:
            return reached[0]
        nxt = {}
        for B, s in cur.items():
            for r in range(1, R + 1):
                if arch.pattern[i % len(arch.pattern)] == "window" and 2 * (r - 1) * B >= arch.window:
                    continue
                if i == 0 and 2 * r - 1 > arch.H:
                    continue
                nxt.setdefault(min(B * r, target), s + [r])
        cur, i = nxt, i + 1


def claimed_layers(n: int, arch) -> int:
    return len(schedule(n, arch))


def unpack(ai, b):  # sym = -1 is a partner off the left end: the zero map, annihilated by the start constant
    return np.where(ai >= 0, A_VALS[np.maximum(ai, 0)], 0), np.where(ai >= 0, b, 0)


def pack(a, b):
    return {"a": A_IDX[a % MOD], "b": b % MOD}


def compose(first, then):
    return (then[0] * first[0]) % MOD, (then[0] * first[1] + then[1]) % MOD


def elem(op, opr):
    a = np.select([opr >= 8, op == 1, op == 2, op == 3], [0, 1, 1, opr], 0) % MOD
    b = np.select([opr >= 8, op == 1, op == 2], [opr & 7, opr, -opr], 0) % MOD
    return a, b


def build(n: int, arch) -> Program:
    n_pos = 2 * n + 1
    fields = {"a": 5, "b": MOD, "pos": n_pos, "op": 4, "opr": 16, "one": 2}
    pos = np.arange(n_pos)

    def embed(tokens, n_pos):
        tok = np.asarray(tokens)
        is_op = tok >= 8
        return {"pos": pos, "one": np.ones(n_pos, dtype=np.int64), "op": np.where(is_op, tok - 7, 0),
                "opr": np.where(is_op, 0, np.where(pos == 0, tok + 8, tok))}

    def readout(st):
        vals = [int(st["b"][2 * k]) for k in range(1, n + 1)]
        return vals, vals[-1:]

    P = Program(arch, fields, n_pos, 0, embed, readout)
    M, L = arch.M, 1
    bg = Clause(-arch.M, {"one": 1})  # pushes every non-matching key to -M, doubling the score gap
    for i, r in enumerate(schedule(n, arch)):
        lay = P.layer()
        if i == 0:
            lay.head([bg, Clause(2 * M, {"pos": Q("pos", -1)})], ["op"])
            for t in range(1, r):
                lay.head([bg, Clause(2 * M, {"pos": Q("pos", -2 * t)})], ["opr"])
                lay.head([bg, Clause(2 * M, {"pos": Q("pos", -2 * t - 1)})], ["op"])

            def fn0(f, h, r=r):
                acc = elem(h[0].sym("op"), f["opr"])
                for t in range(1, r):
                    acc = compose(elem(h[2 * t].sym("op"), h[2 * t - 1].sym("opr")), acc)
                return pack(*acc)

            lay.mlp(["opr"], fn0)
        elif r > 1:
            for t in range(1, r):
                lay.head([bg, Clause(2 * M, {"pos": Q("pos", -2 * t * L)})], ["a", "b"])

            def fn(f, h, r=r):
                acc = unpack(f["a"], f["b"])
                for t in range(1, r):
                    acc = compose(unpack(h[t - 1].sym("a"), h[t - 1].sym("b")), acc)
                return pack(*acc)

            lay.mlp(["a", "b"], fn)
        L *= r
    return P

import numpy as np

from functions.rs_transformer import BF16, Clause, Program, Q

MOD = 8


def claimed_layers(n: int, arch) -> int:
    return n


def step(op, b, x):  # a non-op token (start position, op positions) leaves x unchanged
    return np.where(op == 8, x + b, np.where(op == 9, x - b, np.where(op == 10, x * b, x))) % MOD


def build(n: int, arch) -> Program:
    n_pos = 2 * n + 1
    fields = {"tok": 11, "pos": n_pos, "x": MOD}

    def embed(tokens, n_pos):
        pos = np.arange(n_pos)
        return {"tok": np.asarray(tokens), "pos": pos, "x": np.where(pos == 0, tokens[0], 0)}

    def readout(st):
        vals = [int(st["x"][2 * k]) for k in range(1, n + 1)]
        return vals, vals[-1:]

    P = Program(arch, fields, n_pos, 0, embed, readout)
    M = arch.M
    for _ in range(n):
        lay = P.layer()
        lay.head([Clause(M, {"pos": Q("pos", -2)})], ["x"])
        lay.head([Clause(M, {"pos": Q("pos", -1)})], ["tok"])
        lay.mlp(["tok"], lambda f, h: {"x": step(h[1].sym("tok"), f["tok"], h[0].sym("x"))})
    return P

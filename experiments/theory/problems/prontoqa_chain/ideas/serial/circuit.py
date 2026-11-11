import numpy as np

from functions.rs_transformer import BF16, Clause, Program, Q
from problems.prontoqa_chain.parse import parse_layer
from problems.prontoqa_chain.problem import V, n_premises


def claimed_layers(n: int, arch) -> int:
    return n + 2


def build(n: int, arch) -> Program:
    n_in = 8 * n_premises(n) + 8
    names = V + n_premises(n)
    fields = {"tok": names, "pos": n_in, "kind": 4, "idx": n_premises(n) + 1, "subj": names,
              "obj": names, "isprem": 2, "cur": names, "ans": 2}
    qpos = n_in - 2

    def embed(tokens, n_pos):
        return {"tok": np.asarray(tokens), "pos": np.arange(n_pos)}

    def readout(st):
        return [int(st["ans"][qpos])], [int(st["ans"][qpos])]

    P = Program(arch, fields, n_in, 0, embed, readout)
    parse_layer(P, n_in)
    for L in range(n + 1):
        lay = P.layer()
        lay.head([Clause(arch.M, {"subj": Q("subj" if L == 0 else "cur"), "isprem": 1})], ["obj"])

        def fn(f, h, L=L):
            cur = np.where(f["pos"] == qpos, h[0].sym("obj"), f["cur"])
            upd = {"cur": cur}
            if L == n:
                upd["ans"] = np.where(f["pos"] == qpos, cur == f["obj"], 0)
            return upd

        lay.mlp(["pos", "cur", "obj"], fn)
    return P

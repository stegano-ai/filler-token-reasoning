import numpy as np

from functions.rs_transformer import BF16, Clause, Program, Q
from problems.prontoqa_chain.parse import parse_layer
from problems.prontoqa_chain.problem import V, n_premises


def rounds(n: int) -> int:
    k = 0
    while 3 ** k < n + 1:
        k += 1
    return k


def claimed_layers(n: int, arch) -> int:
    return rounds(n) + 3


def build(n: int, arch) -> Program:
    NP, K = n_premises(n), rounds(n)
    n_in = 8 * NP + 8
    n_pos = n_in + 1 + (K + 1) * NP + 1
    names = V + NP
    fields = {"tok": names, "pos": n_pos, "inp": 2, "kind": 4, "idx": NP + 1, "subj": names,
              "obj": names, "isprem": 2, "isrule": 2, "blk": K + 2, "sent": 2, "L": names,
              "F": names, "ans": 2}
    qs = n_pos - 1
    M = arch.M
    SENT = Clause(M - 1, {"sent": 1})

    # block b (b = 0..K) holds one copy of every premise; block b runs rounds 1..b, reading
    # block b-1, so every block's state is a valid radius-r_b interval and later blocks see it
    def embed(tokens, n_pos):
        pos = np.arange(n_pos)
        tok = np.zeros(n_pos, dtype=np.int64)
        tok[:n_in] = tokens
        s = pos - n_in - 1
        isrule = (s >= 0) & (s < (K + 1) * NP)
        return {"tok": tok, "pos": pos, "inp": pos < n_in, "sent": pos == n_in, "isrule": isrule,
                "idx": np.where(isrule, s % NP + 1, 0), "blk": np.where(isrule, s // NP + 1, 0)}

    def readout(st):
        return [int(st["ans"][qs])], [int(st["ans"][qs])]

    P = Program(arch, fields, n_in, n_pos - n_in, embed, readout)
    parse_layer(P, n_in)

    lay = P.layer()
    lay.head([Clause(M, {"idx": Q("idx"), "inp": 1})], ["subj", "obj"])
    lay.head([Clause(M, {"kind": 3, "inp": 1})], ["subj", "obj"])
    lay.mlp(["isrule", "pos", "subj", "obj"], lambda f, h: {
        "subj": np.where(f["isrule"], h[0].sym("subj"), np.where(f["pos"] == qs, h[1].sym("subj"), f["subj"])),
        "obj": np.where(f["isrule"], h[0].sym("obj"), np.where(f["pos"] == qs, h[1].sym("obj"), f["obj"])),
        "L": np.where(f["isrule"], h[0].sym("subj"), 0),
        "F": np.where(f["isrule"], h[0].sym("obj"), 0)})

    # radius r -> 3r+1: adopt the interval that ends at my L (starts at my F); when none exists
    # the chain boundary is within reach and the anchored rule at my L (F) carries it
    for k in range(1, K + 1):
        lay = P.layer()
        prev = {"isrule": 1, "blk": Q("blk", -1)}
        lay.head([Clause(M, {"F": Q("L"), **prev}), SENT], ["L", "sent"])
        lay.head([Clause(M, {"obj": Q("L"), **prev}), SENT], ["L", "sent"])
        lay.head([Clause(M, {"L": Q("F"), **prev}), SENT], ["F", "sent"])
        lay.head([Clause(M, {"subj": Q("F"), **prev}), SENT], ["F", "sent"])

        def fn(f, h, k=k):
            pick = lambda a, b, fld: np.where(h[a].sym("sent") == 0, h[a].sym(fld),
                                              np.where(h[b].sym("sent") == 0, h[b].sym(fld), f[fld]))
            act = f["isrule"] & (f["blk"] > k)
            return {"L": np.where(act, pick(0, 1, "L"), f["L"]), "F": np.where(act, pick(2, 3, "F"), f["F"])}

        lay.mlp(["isrule", "blk", "L", "F"], fn)

    lay = P.layer()
    lay.head([Clause(M, {"L": Q("subj"), "F": Q("obj"), "isrule": 1, "blk": K + 1}), SENT], ["sent"])
    lay.mlp(["pos"], lambda f, h: {"ans": np.where(f["pos"] == qs, h[0].sym("sent") == 0, 0)})
    return P

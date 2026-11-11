import numpy as np

from functions.rs_transformer import Clause, Program, Q
from problems.prontoqa_chain.problem import V, n_premises
from problems.prontoqa_chain.sched import parse_into, premise_fetch, query_fetch, round_layer, to_full


def rounds(n: int) -> int:
    k = 0
    while 3 ** k < n + 1:
        k += 1
    return k


def claimed_layers(n: int, arch) -> int:
    K = rounds(n)
    if arch.pattern[0] == "linear":
        return 4 * K + 5
    if len(arch.pattern) > 1:
        return 2 * K + 4
    return K + 3


def build(n: int, arch) -> Program:
    NP, K, M = n_premises(n), rounds(n), arch.M
    n_in = 8 * NP + 8
    n_pos = n_in + 1 + (K + 1) * NP + 1
    names, qs = V + NP, n_in + 1 + (K + 1) * NP
    fields = {"tok": names, "pos": n_pos, "kind": 4, "idx": NP + 1, "subj": names, "obj": names,
              "blk": K + 2, "sent": 2, "L": names, "F": names, "E": names, "T": names,
              "hit": 2, "ans": 2}
    if arch.pattern[0] == "linear":
        fields.update({"t1": names, "t3": names})

    def embed(tokens, n_pos):
        pos = np.arange(n_pos)
        tok = np.zeros(n_pos, dtype=np.int64)
        tok[:n_in] = tokens
        s = pos - n_in - 1
        copy = (s >= 0) & (s < (K + 1) * NP)
        return {"tok": tok, "pos": pos, "sent": pos == n_in,
                "idx": np.where(copy, s % NP + 1, 0), "blk": np.where(copy, s // NP + 1, 0)}

    def readout(st):
        return [int(st["ans"][qs])], [int(st["ans"][qs])]

    P = Program(arch, fields, n_in, n_pos - n_in, embed, readout)
    parse_into(P, n_in)

    # (e, T) reaches every copy from the query's object token: a bounded-key associative read, so
    # a linear layer does it; otherwise it rides along on the mirror's full layer
    on_mirror = P.next_kind() != "linear"
    if not on_mirror:
        lay = P.layer()
        lay.head(query_fetch(M), ["subj", "obj"])
        lay.mlp([], lambda f, h: {"E": np.maximum(h[0].sym("subj"), 0),
                                  "T": np.maximum(h[0].sym("obj"), 0)})
    to_full(P)

    lay = P.layer()
    lay.head(premise_fetch(M), ["subj", "obj"])
    if on_mirror:
        lay.head(query_fetch(M), ["subj", "obj"])

    def mirror(f, h):
        c = f["blk"] > 0
        s, o = np.where(c, h[0].sym("subj"), f["subj"]), np.where(c, h[0].sym("obj"), f["obj"])
        upd = {"subj": s, "obj": o, "L": np.where(c, s, 0), "F": np.where(c, o, 0)}
        if on_mirror:
            upd.update({"E": np.maximum(h[1].sym("subj"), 0), "T": np.maximum(h[1].sym("obj"), 0)})
        return upd

    lay.mlp(["blk", "subj", "obj"], mirror)

    for k in range(1, K + 1):
        to_full(P)
        hit = (["E", "T"], lambda f, L, F: {
            "hit": ((f["blk"] == K + 1) & (L == f["E"]) & (F == f["T"])).astype(np.int64)})
        round_layer(P, k, hit if k == K else None)

    # exists a copy whose interval runs from the entity to the queried category: a one-bit flag
    # aggregated by a single key, so a linear layer serves when the pattern offers one
    if P.next_kind() != "linear":
        to_full(P)
    lay = P.layer()
    lay.head([Clause(M, {"hit": 1})], ["hit"])
    lay.mlp(["pos"], lambda f, h: {"ans": np.where(f["pos"] == qs, h[0].sym("hit") == 1, 0)})
    return P

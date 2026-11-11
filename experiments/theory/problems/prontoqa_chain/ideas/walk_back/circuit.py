import numpy as np

from functions.rs_transformer import Program
from problems.prontoqa_chain.problem import V, n_premises
from problems.prontoqa_chain.sched import parse_into, premise_fetch, query_fetch, round_layer, to_full


def rounds(n: int) -> int:
    k = 0
    while 3 ** k < 2 * n + 3:
        k += 1
    return k


def claimed_layers(n: int, arch) -> int:
    K = rounds(n)
    if arch.pattern[0] == "linear":
        return 4 * K + 4
    if len(arch.pattern) > 1:
        return 2 * K + 2
    return K + 2


def build(n: int, arch) -> Program:
    NP, K, M = n_premises(n), rounds(n), arch.M
    n_in = 8 * NP + 8
    n_pos = n_in + 1 + (K + 1) * NP + 1
    names, qs = V + NP, n_in + 1 + (K + 1) * NP
    fields = {"tok": names, "pos": n_pos, "kind": 4, "idx": NP + 1, "subj": names, "obj": names,
              "blk": K + 3, "sent": 2, "L": names, "F": names, "ans": 2}
    if arch.pattern[0] == "linear":
        fields.update({"t1": names, "t3": names})

    # the query slot is one more interval copy, sitting in block K+2 so it reads block K+1
    def embed(tokens, n_pos):
        pos = np.arange(n_pos)
        tok = np.zeros(n_pos, dtype=np.int64)
        tok[:n_in] = tokens
        s = pos - n_in - 1
        copy = (s >= 0) & (s < (K + 1) * NP)
        return {"tok": tok, "pos": pos, "sent": pos == n_in,
                "idx": np.where(copy, s % NP + 1, 0),
                "blk": np.where(pos == qs, K + 2, np.where(copy, s // NP + 1, 0))}

    def readout(st):
        return [int(st["ans"][qs])], [int(st["ans"][qs])]

    P = Program(arch, fields, n_in, n_pos - n_in, embed, readout)
    parse_into(P, n_in)
    to_full(P)

    lay = P.layer()
    lay.head(premise_fetch(M), ["subj", "obj"])
    lay.head(query_fetch(M), ["subj", "obj"])

    def mirror(f, h):
        isq = f["pos"] == qs
        c = (f["blk"] > 0) & ~isq
        s = np.where(isq, h[1].sym("subj"), np.where(c, h[0].sym("subj"), f["subj"]))
        o = np.where(isq, h[1].sym("obj"), np.where(c, h[0].sym("obj"), f["obj"]))
        return {"subj": s, "obj": o, "L": np.where(isq, o, np.where(c, s, 0)),
                "F": np.where(c | isq, o, 0)}

    lay.mlp(["pos", "blk", "subj", "obj"], mirror)

    # the query slot walks back from T by 3^(k-1) edges in round k and clamps at its chain's first
    # node, so after K rounds it holds the entity iff T ends the entity's chain
    for k in range(1, K + 1):
        to_full(P)
        done = (["pos", "subj"], lambda f, L, F: {"ans": np.where(f["pos"] == qs, L == f["subj"], 0)})
        round_layer(P, k, done if k == K else None)
    return P

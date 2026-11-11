import numpy as np

from functions.rs_transformer import Clause, Q
from problems.prontoqa_chain.problem import A, COLON, DOT, EVERY, V

OFFS = [1, 3, 4, 5, 6]


def to_full(P):
    while P.next_kind() != "full":
        P.layer()


# A linear layer only shifts by -3..0 (the conv kernel), so under that pattern the -4/-5/-6 reads
# of the parse are composed out of two shifts; a full or window layer does all five at once.
def parse_into(P, n_in):
    M = P.arch.M
    if P.next_kind() == "linear":
        lay = P.layer()
        lay.head([Clause(M, {"pos": Q("pos", -1)})], ["tok"])
        lay.head([Clause(M, {"pos": Q("pos", -3)})], ["tok"])
        lay.mlp([], lambda f, h: {"t1": np.maximum(h[0].sym("tok"), 0),
                                  "t3": np.maximum(h[1].sym("tok"), 0)})
        lay = P.layer()
        lay.head([Clause(M, {"pos": Q("pos", -3)})], ["t1"])
        lay.head([Clause(M, {"pos": Q("pos", -2)})], ["t3"])
        lay.head([Clause(M, {"pos": Q("pos", -3)})], ["t3"])
        reads = ["pos", "tok", "idx", "t1", "t3"]
        get = lambda f, h: {1: f["t1"], 3: f["t3"], 4: h[0].sym("t1"),
                            5: h[1].sym("t3"), 6: h[2].sym("t3")}
    else:
        lay = P.layer()
        for k in OFFS:
            lay.head([Clause(M, {"pos": Q("pos", -k)})], ["tok"])
        reads = ["pos", "tok", "idx"]
        get = lambda f, h: {k: h[i].sym("tok") for i, k in enumerate(OFFS)}

    def fn(f, h):
        m = get(f, h)
        is_obj = (f["pos"] < n_in) & (m[1] == A)
        kind = np.where(m[4] == EVERY, 1, np.where(m[4] == DOT, 2, np.where(m[4] == COLON, 3, 0)))
        kind = np.where(is_obj, kind, 0)
        idx = np.where(kind == 1, m[6] - V + 1, np.where(kind == 2, m[5] - V + 1, f["idx"]))
        return {"kind": kind, "idx": idx, "subj": np.where(is_obj, m[3], 0),
                "obj": np.where(is_obj, f["tok"], 0)}

    lay.mlp(reads, fn)


def premise_fetch(M):
    return [Clause(M, {"idx": Q("idx"), "kind": 1}), Clause(M, {"idx": Q("idx"), "kind": 2})]


def query_fetch(M):
    return [Clause(M, {"kind": 3})]


# radius r -> 3r+1: adopt the interval that ends at my L (starts at my F); when none exists the
# chain boundary is within reach and the anchored rule at my L (F) carries it. Block b (blk > k)
# reads block b-1, which sits entirely before it and has completed exactly k-1 rounds.
def round_layer(P, k, extra=None):
    M = P.arch.M
    S = Clause(M - 2, {"sent": 1})  # margin 2 on both sides: the sentinel loses to a real match and beats the background
    prev = {"blk": Q("blk", -1)}
    lay = P.layer()
    lay.head([Clause(M, {"F": Q("L"), **prev}), S], ["L", "sent"])
    lay.head([Clause(M, {"obj": Q("L"), **prev}), S], ["L", "sent"])
    lay.head([Clause(M, {"L": Q("F"), **prev}), S], ["F", "sent"])
    lay.head([Clause(M, {"subj": Q("F"), **prev}), S], ["F", "sent"])
    reads, tail = extra if extra else ([], None)

    def fn(f, h):
        pick = lambda a, b, fld: np.where(h[a].sym("sent") == 0, h[a].sym(fld),
                                          np.where(h[b].sym("sent") == 0, h[b].sym(fld), f[fld]))
        act = f["blk"] > k
        L = np.where(act, pick(0, 1, "L"), f["L"])
        F = np.where(act, pick(2, 3, "F"), f["F"])
        return {"L": L, "F": F, **(tail(f, L, F) if tail else {})}

    lay.mlp(["blk", "L", "F"] + reads, fn)

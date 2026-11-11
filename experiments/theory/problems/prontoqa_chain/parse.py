import numpy as np

from functions.rs_transformer import BF16, Clause, Q
from problems.prontoqa_chain.problem import A, COLON, DOT, EVERY, V

OFFS = [1, 3, 4, 5, 6]


# Object tokens (tok[-1] == "a") read their line: kind from tok[-4] (Every / . / :), subject from
# tok[-3], premise number from tok[-6] (rule) or tok[-5] (base fact).
def parse_layer(P, n_in):
    lay = P.layer()
    for k in OFFS:
        lay.head([Clause(BF16.M, {"pos": Q("pos", -k)})], ["tok"])

    def fn(f, h):
        m = {k: h[i].sym("tok") for i, k in enumerate(OFFS)}
        is_obj = (f["pos"] < n_in) & (m[1] == A)
        kind = np.where(m[4] == EVERY, 1, np.where(m[4] == DOT, 2, np.where(m[4] == COLON, 3, 0)))
        kind = np.where(is_obj, kind, 0)
        idx = np.where(kind == 1, m[6] - V + 1, np.where(kind == 2, m[5] - V + 1, f["idx"]))
        return {"kind": kind, "idx": idx, "subj": np.where(is_obj, m[3], 0),
                "obj": np.where(is_obj, f["tok"], 0), "isprem": (kind == 1) | (kind == 2)}

    lay.mlp(["pos", "tok", "idx"], fn)

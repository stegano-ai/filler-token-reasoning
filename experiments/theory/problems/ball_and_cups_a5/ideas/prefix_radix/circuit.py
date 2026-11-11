import numpy as np

from functions.rs_transformer import BF16, Clause, Program, Q

ROT, SWAP = (1, 2, 3, 4, 0), (1, 0, 2, 4, 3)
ELEMS = [tuple(range(5))]
for a in ELEMS:  # closure of {rotate, double-swap} = A5 (60 elements), a[c] = image of cup c
    for gen in (ROT, SWAP):
        if (b := tuple(gen[a[c]] for c in range(5))) not in ELEMS:
            ELEMS.append(b)
IDX = {e: i for i, e in enumerate(ELEMS)}
COMP = np.array([[IDX[tuple(a[b[c]] for c in range(5))] for b in ELEMS] for a in ELEMS])  # a∘b (b first)
INS = np.array([IDX[ROT], IDX[SWAP], 0])  # instruction symbol -> element; 2 = no instruction
IMG0 = np.array([e[0] for e in ELEMS])
K = 8  # instruction symbols folded by layer 1 (8 heads, 1 bit each)
R = 4  # partner blocks folded per later layer (4 x 6 bits) + one instruction (2 bits) + own (5.9 bits)


def blocks(n: int) -> list:
    b = [K]
    while b[-1] < n:
        b.append((R + 1) * b[-1] + 1)
    return b


def claimed_layers(n: int, arch) -> int:
    return len(blocks(n))


def build(n: int, arch) -> Program:
    n_scr = max(2 * n, 4)  # n identity anchors + n workers; >= 4 so alphabet 60 fits 4*n_pos^2 at n = 1
    n_pos = n + n_scr
    W = 2 * n  # workers sit at W + i; every position below W holds the identity forever
    fields = {"tok": 2, "pos": n_pos, "g": 60, "ins": 3, "v1": K + 1}

    def embed(tokens, n_pos):
        pos = np.arange(n_pos)
        tok = np.zeros(n_pos, dtype=np.int64)
        tok[:n] = tokens
        i = pos - W
        return {"tok": tok, "pos": pos, "ins": np.full(n_pos, 2),
                "v1": np.where((i >= 0) & (i < n), np.minimum(K, i + 1), 0)}

    def readout(st):
        vals = IMG0[st["g"][W:W + n]].tolist()
        return vals, vals

    P = Program(arch, fields, n, n_scr, embed, readout)
    M = arch.M
    lay = P.layer()
    for m in range(min(K, n)):
        lay.head([Clause(M, {"pos": Q("pos", -(W + m))})], ["tok"])

    def fn1(f, h):
        acc = np.zeros(n_pos, dtype=np.int64)
        for m in reversed(range(len(h))):
            acc = np.where(m < f["v1"], COMP[INS[h[m].sym("tok")], acc], acc)
        return {"g": acc, "ins": np.where(f["v1"] > 0, h[0].sym("tok"), 2) if h else f["ins"]}

    lay.mlp(["v1", "ins"], fn1)

    bs = blocks(n)
    for b in bs[:-1]:
        lay = P.layer()
        parts = [m for m in range(1, R + 1) if m * b <= n - 1]
        for m in parts:
            lay.head([Clause(M, {"pos": Q("pos", -m * b)})], ["g"])
        tail = (R + 1) * b <= n - 1
        if tail:
            lay.head([Clause(M, {"pos": Q("pos", -(R + 1) * b)})], ["ins"])

        def fn(f, h, parts=parts, tail=tail):
            acc = f["g"]
            for k in range(len(parts)):
                acc = COMP[acc, h[k].sym("g")]
            if tail:
                acc = COMP[acc, INS[h[-1].sym("ins")]]
            return {"g": acc}

        lay.mlp(["g"], fn)
    return P

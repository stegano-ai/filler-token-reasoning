import numpy as np

from functions.rs_transformer import Clause, Program, Q

D = 4
FUNCS = [tuple((i >> (2 * v)) & 3 for v in range(D)) for i in range(256)]
_i = FUNCS.index(tuple(range(D)))
FUNCS[_i], FUNCS[255] = FUNCS[255], FUNCS[_i]  # identity at 255 so sym()'s -1 (no match) composes as identity
IDX = {f: i for i, f in enumerate(FUNCS)}
COMP = np.array([[IDX[tuple(FUNCS[a][FUNCS[b][x]] for x in range(D))] for b in range(256)] for a in range(256)])
ENC = np.array([IDX[tuple((c >> (2 * v)) & 3 for v in range(D))] for c in range(256)])
R = 4  # functions merged per layer: 8 + 3 * 8 = 32 <= mlp_bits
W0 = 2  # tables folded by the gather layer: 2 + 2.33 + 7 * 2 <= mlp_bits


def n_levels(n: int) -> int:
    L, W = 0, W0
    while W < n + 1:
        L, W = L + 1, W * R
    return L


def schedule(n: int, arch) -> list:
    steps, L = [("gather",)], 1
    while L <= n_levels(n):
        kind = arch.pattern[len(steps) % len(arch.pattern)]
        if kind == "full" or (kind == "window" and (R - 1) * W0 * R ** L < arch.window):
            steps.append(("level", L))
            L += 1
        else:
            steps.append(("noop",))
    return steps


def claimed_layers(n: int, arch) -> int:
    return len(schedule(n, arch))


def build(n: int, arch) -> Program:
    n_in = 1 + n * D
    n_pos = max(n_in + 1, 8)  # >= 8 so alphabet 256 fits 4 * n_pos ** 2
    fields = {"tok": D, "pos": n_pos, "r": 4, "g": 256}

    def embed(tokens, n_pos):
        pos = np.arange(n_pos)
        tok = np.zeros(n_pos, dtype=np.int64)
        tok[1:1 + n_in] = tokens
        end = (pos % D == 1) & (pos <= n_in)
        return {"tok": tok, "pos": pos,
                "r": np.where(pos == 1, 1, np.where(pos == 1 + D, 2, np.where(end, 3, 0)))}

    def readout(st):
        vals = [FUNCS[int(st["g"][D * k + 1])][0] for k in range(1, n + 1)]
        return vals, vals

    P = Program(arch, fields, n_in, n_pos - n_in, embed, readout)
    for s in schedule(n, arch):
        lay = P.layer()
        if s[0] == "noop":
            continue
        if s[0] == "gather":
            for m in range(1, 2 * D):
                lay.head([Clause(arch.M, {"pos": Q("pos", -m)})], ["tok"])
            lay.mlp(["tok", "r"], gather_fn)
            continue
        for m in range(1, R):
            lay.head([Clause(arch.M, {"pos": Q("pos", -m * W0 * R ** s[1])})], ["g"])
        lay.mlp(["g"], scan_fn)
    return P


def gather_fn(f, h):
    t = [hd.sym("tok") for hd in h]
    own = ENC[np.clip(t[2] + 4 * t[1] + 16 * t[0] + 64 * f["tok"], 0, 255)]
    prev = ENC[np.clip(t[6] + 4 * t[5] + 16 * t[4] + 64 * t[3], 0, 255)]
    const = ENC[85 * np.clip(t[3], 0, 3)]
    return {"g": np.where(f["r"] == 3, COMP[own, prev],
                          np.where(f["r"] == 2, COMP[own, const],
                                   np.where(f["r"] == 1, ENC[85 * f["tok"]], 255)))}


def scan_fn(f, h):
    acc = f["g"]
    for hd in h:
        acc = COMP[acc, hd.sym("g")]
    return {"g": acc}

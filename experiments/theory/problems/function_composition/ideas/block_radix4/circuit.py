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
        if kind != "window" or (R - 1) * W0 * R ** L < arch.window:
            steps.append(("level", L, kind))
            L += 1
        else:
            steps.append(("noop",))
    return steps


def claimed_layers(n: int, arch) -> int:
    return len(schedule(n, arch))


def build(n: int, arch) -> Program:
    sched = schedule(n, arch)
    n_in = 1 + n * D
    n_pos = max(n_in + 1, 8)  # >= 8 so alphabet 256 fits 4 * n_pos ** 2
    fields = {"tok": D, "pos": n_pos, "r": 5, "g": 256, "slot": D, "par": 2, "parq": 2}
    for s in sched:
        if s[0] != "level":
            continue
        fields[f"m{s[1]}"] = R
        if s[2] != "linear":
            fields[f"k{s[1]}"] = n + 2
            fields[f"e{s[1]}"] = 2

    def embed(tokens, n_pos):
        pos = np.arange(n_pos)
        tok = np.zeros(n_pos, dtype=np.int64)
        tok[1:1 + n_in] = tokens
        elem = (pos + 2) // D
        end = (pos % D == 1) & (pos <= n_in)
        odd = end & (elem % 2 == 1)
        st = {"tok": tok, "pos": pos, "slot": (pos - 2) % D,
              "par": elem % 2, "parq": 1 - elem % 2,
              "r": np.where(pos == 1, 1, np.where(pos == 1 + D, 2,
                            np.where(odd, 3, np.where(end, 4, 0))))}
        for s in sched:
            if s[0] != "level":
                continue
            B = W0 * R ** (s[1] - 1)
            st[f"m{s[1]}"] = (elem // B) % R
            if s[2] != "linear":
                st[f"k{s[1]}"] = elem // B + 1
                st[f"e{s[1]}"] = (end & ((elem + 1) % B == 0)).astype(np.int64)
        return st

    def readout(st):
        vals = [FUNCS[int(st["g"][D * k + 1])][0] for k in range(1, n + 1)]
        return vals, vals

    P = Program(arch, fields, n_in, n_pos - n_in, embed, readout)
    for s in sched:
        lay = P.layer()
        if s[0] == "noop":
            continue
        if s[0] == "gather":
            for m in range(1, D):
                lay.head([Clause(arch.M, {"pos": Q("pos", -m)})], ["tok"])
            for v in range(D - 1, -1, -1):  # the previous table: most recent position of the other parity
                lay.head([Clause(arch.M, {"par": Q("parq"), "slot": v})] if lay.kind == "linear"
                         else [Clause(arch.M, {"pos": Q("pos", -(2 * D - 1 - v))})], ["tok"])
            lay.mlp(["tok", "r"], gather_fn)
            continue
        L = s[1]
        for m in range(1, R):
            if s[2] == "linear":  # the delta rule's most recent match on the block residue is the block end
                lay.head([Clause(arch.M, {f"m{L}": Q(f"m{L}", -m)})], ["g"])
            else:  # m keeps the match inside the level-(L+1) block
                lay.head([Clause(arch.M, {f"k{L}": Q(f"k{L}", -m), f"m{L}": Q(f"m{L}", -m),
                                          f"e{L}": 1})], ["g"])
        lay.mlp(["g"], scan_fn)
    return P


def gather_fn(f, h):
    t = [hd.sym("tok") for hd in h]
    own = ENC[np.clip(t[2] + 4 * t[1] + 16 * t[0] + 64 * f["tok"], 0, 255)]
    prev = ENC[np.clip(t[6] + 4 * t[5] + 16 * t[4] + 64 * t[3], 0, 255)]
    const = ENC[85 * np.clip(t[3], 0, 3)]
    return {"g": np.where(f["r"] == 3, COMP[own, prev],
                          np.where(f["r"] == 2, COMP[own, const],
                                   np.where(f["r"] == 4, own,
                                            np.where(f["r"] == 1, ENC[85 * f["tok"]], 255))))}


def scan_fn(f, h):
    acc = f["g"]
    for hd in h:
        acc = COMP[acc, hd.sym("g")]
    return {"g": acc}

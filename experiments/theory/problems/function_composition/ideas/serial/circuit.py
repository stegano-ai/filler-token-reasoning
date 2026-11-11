import numpy as np

from functions.rs_transformer import BF16, Clause, Program, Q

D = 4


def claimed_layers(n: int, arch) -> int:
    return n + 1


def build(n: int, arch) -> Program:
    n_in, n_scr = 1 + n * D, 1 + n
    n_pos = n_in + n_scr
    fields = {"tok": D, "pos": n_pos, "table": n + 1, "slot": D, "role": 3, "step": n + 1,
              "cur": D, "out": D}

    def embed(tokens, n_pos):
        pos = np.arange(n_pos)
        inp = pos < n_in
        tok = np.zeros(n_pos, dtype=np.int64)
        tok[:n_in] = tokens
        table = np.where(inp & (pos >= 1), (pos - 1) // D + 1, 0)
        slot = np.where(inp & (pos >= 1), (pos - 1) % D, 0)
        role = np.where(inp, 0, np.where(pos == n_in, 1, 2))
        step = np.where(role == 2, pos - n_in, 0)
        cur = np.where(role == 1, tokens[0], 0)
        return {"tok": tok, "pos": pos, "table": table, "slot": slot, "role": role,
                "step": step, "cur": cur}

    def readout(st):
        order = np.argsort(st["step"][n_in + 1:])
        vals = st["out"][n_in + 1:][order].tolist()
        return vals, vals

    P = Program(arch, fields, n_in, n_scr, embed, readout)
    M = arch.M
    for L in range(1, n + 2):
        lay = P.layer()
        if L <= n:
            lay.head([Clause(M, {"table": L, "slot": Q("cur")})], ["tok"])
        if L >= 2:
            lay.head([Clause(M, {"role": 1})], ["cur"])

        def fn(f, h, L=L):
            upd, hi = {}, 0
            if L <= n:
                upd["cur"] = np.where(f["role"] == 1, h[hi].sym("tok"), f["cur"])
                hi += 1
            if L >= 2:
                upd["out"] = np.where((f["role"] == 2) & (f["step"] == L - 1), h[hi].sym("cur"), f["out"])
            return upd

        lay.mlp(["role", "step", "cur", "out"], fn)
    return P

import numpy as np

from functions.rs_transformer import BF16, Clause, Program, Q

ENT0, NUM0 = 5, 414
PLAIN, FACT, RULE, SLOT, HOLDER, OUT, QE, QX = range(8)
HIT, MISS = BF16.M, BF16.M // 2


def nbits(x):
    return (np.floor(np.log2(np.maximum(x, 1))) + 1).astype(np.int64)


class Geo:
    def __init__(self, t):
        self.t = t
        self.n_in = 28 * t + 53
        self.n_idx = 4 * t + 8
        self.n_sites = self.n_idx // 2
        self.dmax = (self.n_idx - 1).bit_length()
        self.n_hold = t + 2
        self.t_max = self.n_hold * (self.dmax + 1) + 1
        self.QEP, self.QXP, self.NULLP = self.n_in - 3, self.n_in - 1, self.n_in
        self.FH0, self.S0 = self.n_in + 1, self.n_in + 3
        self.k = (4 * self.n_in - 3 - self.n_hold - self.t_max) // self.n_sites
        self.H0 = self.S0 + self.k * self.n_sites
        self.O0 = self.H0 + self.n_hold
        self.n_pos = self.O0 + self.t_max
        self.npairs = -(-(self.dmax + 1) // 2)
        self.emit_layers = -(-self.npairs // 3)


def claimed_layers(t: int, arch) -> int:
    return 6 + (t + 1) + Geo(t).emit_layers


def build(t: int, arch) -> Program:
    g = Geo(t)
    NULLP, S0, H0, O0 = g.NULLP, g.S0, g.H0, g.O0
    n_in, n_idx, n_sites = g.n_in, g.n_idx, g.n_sites
    ALPH = NUM0 + n_idx
    fields = {"pos": g.n_pos, "role": 8, "tok": ALPH, "A1": NUM0, "B1": NUM0, "H1": NUM0,
              "A2": NUM0, "B2": NUM0, "H2": NUM0, "I1": n_idx + 1, "S2": max(t + 2, 9),
              "est1": 2, "est2": 2, "fresh": 3, "isX": 4, "end": g.n_pos}

    pos = np.arange(g.n_pos)
    slotm = (pos >= S0) & (pos < H0)
    site = np.where(slotm, (pos - S0) % n_sites, 0)
    fhm = (pos >= g.FH0) & (pos < S0)
    holdm = (pos >= H0) & (pos < O0)
    outm = pos >= O0
    hs = pos - H0                              # holder step index
    base_role = np.select([pos == g.QEP, pos == g.QXP, pos < n_in, fhm, slotm, holdm, outm],
                          [QE, QX, PLAIN, SLOT, SLOT, HOLDER, OUT], PLAIN)

    def embed(tokens, n_pos):
        tok = np.zeros(n_pos, dtype=np.int64)
        tok[:n_in] = tokens
        inp = pos < n_in
        return {"pos": pos, "tok": tok, "role": base_role,
                "H1": np.where(inp & (tok < NUM0), tok, 0),
                "A1": np.where(pos == g.QEP, np.maximum(tok - ENT0 + 1, 0), 0),
                "I1": np.select([inp & (tok >= NUM0), slotm, pos == NULLP],
                                [tok - NUM0, 2 * site, n_idx], 0),
                "est1": (pos == g.FH0 + 1).astype(np.int64),
                "isX": np.where(pos == NULLP, 3, 0)}

    def readout(st):
        total = st["end"][H0 + t + 1] - O0 + 1
        vals = st["isX"][O0:O0 + total].tolist()
        return vals, vals[-1:]

    P = Program(arch, fields, n_in, g.n_pos - n_in, embed, readout)

    # L1 — parse the head of each premise line from the fixed offsets behind its newline
    lay = P.layer()
    for d in (1, 2, 3):
        lay.head([Clause(HIT, {"pos": Q("pos", -d)})], ["tok"])

    def parse_a(f, h):
        g1, g2, g3 = (x.sym("tok") for x in h)
        anchor = f["pos"] < n_in - 4
        fact, rule = anchor & (g2 == 2), anchor & (g2 == 3)
        return {"role": np.select([fact, rule], [FACT, RULE], base_role),
                "A2": np.where(fact | rule, g1, 0), "B2": np.where(rule, g3, 0),
                "S2": np.where(fact, g3 - ENT0 + 1, 0),
                "isX": np.select([fact, pos == NULLP], [1, 3], 0)}

    lay.mlp(["pos"], parse_a)

    # L2 — the remaining line fields: rule leg A and the premise numeral
    lay = P.layer()
    lay.head([Clause(HIT, {"pos": Q("pos", -4)})], ["tok"])
    lay.head([Clause(HIT, {"pos": Q("pos", -5)})], ["I1"])
    lay.head([Clause(HIT, {"pos": Q("pos", -6)})], ["I1"])

    def parse_b(f, h):
        g4, g5, g6 = h[0].sym("tok"), h[1].sym("I1"), h[2].sym("I1")
        rule, fact = f["role"] == RULE, f["role"] == FACT
        return {"H2": np.where(rule, g4, 0),
                "I1": np.select([rule, fact, slotm, pos == NULLP], [g6, g5, 2 * site, n_idx], 0)}

    lay.mlp(["role"], parse_b)

    def load(off, fld):
        return [Clause(HIT, {"role": RULE, "I1": Q("I1", off)}), Clause(MISS, {"pos": NULLP})], [fld]

    # L3 — relay the queried entity to the scratch rows, rank the fact pair, load both leg A's
    lay = P.layer()
    lay.head([Clause(HIT, {"role": QE}), Clause(MISS, {"pos": NULLP})], ["A1"])
    lay.head([Clause(HIT, {"role": FACT, "S2": Q("S2")}), Clause(-HIT, {"pos": Q("pos")}),
              Clause(MISS, {"pos": 0})], ["isX"])   # position 0 is the fallback the input rows can see
    lay.head(*load(0, "H2"))
    lay.head(*load(1, "H2"))

    def bind(f, h):
        ent, partner = h[0].sym("A1"), h[1].sym("isX") == 1
        return {"end": np.where(fhm | (holdm & (hs < 2)), ent, 0),
                "est1": np.where((f["role"] == FACT) & partner, 1, (pos == g.FH0 + 1).astype(np.int64)),
                "A1": np.where(slotm, h[2].sym("H2"), 0), "B1": np.where(slotm, h[3].sym("H2"), 0)}

    lay.mlp(["pos", "role"], bind)

    # L4 — both leg B's, and the queried entity's two base facts become pre-fired slots
    lay = P.layer()
    lay.head(*load(0, "B2"))
    lay.head(*load(1, "B2"))
    lay.head([Clause(HIT, {"role": FACT, "S2": Q("end"), "est1": Q("est1")}),
              Clause(MISS, {"pos": NULLP})], ["A2"])

    def legb(f, h):
        return {"tok": np.where(slotm, h[0].sym("B2"), 0), "H2": np.where(slotm, h[1].sym("B2"), 0),
                "B2": np.where(fhm, h[2].sym("A2"), 0), "est1": np.where(fhm, 1, f["est1"])}

    lay.mlp(["pos", "est1"], legb)

    # L5 — both head categories
    lay = P.layer()
    lay.head(*load(0, "A2"))
    lay.head(*load(1, "A2"))

    def heads(f, h):
        return {"B2": np.where(slotm, h[0].sym("A2"), f["B2"]), "A2": np.where(slotm, h[1].sym("A2"), 0)}

    lay.mlp(["pos", "B2"], heads)

    # L6 — holders 0/1 take the two base-fact premise numbers; slots test their heads against the query
    lay = P.layer()
    for r in (0, 1):
        lay.head([Clause(HIT, {"role": FACT, "S2": Q("end"), "est1": r}),
                  Clause(MISS, {"pos": NULLP})], ["I1"])
    lay.head([Clause(HIT, {"role": QX, "H1": Q("B2")}), Clause(MISS, {"pos": NULLP})], ["isX"])
    lay.head([Clause(HIT, {"role": QX, "H1": Q("A2")}), Clause(MISS, {"pos": NULLP})], ["isX"])

    def seed(f, h):
        i0, i1 = h[0].sym("I1"), h[1].sym("I1")
        l0, l1 = nbits(i0) + 1, nbits(i1) + 1
        x1, x2 = h[2].sym("isX") == 0, h[3].sym("isX") == 0
        return {"I1": np.select([holdm & (hs == 0), holdm & (hs == 1), slotm, pos == NULLP],
                                [i0, i1, 2 * site, n_idx], 0),
                "end": np.select([holdm & (hs == 0), holdm & (hs == 1)],
                                 [O0 + l0, O0 + l0 + l1], 0),
                "isX": np.select([slotm & x1, slotm & x2, pos == NULLP], [1, 2, 3], 0)}

    lay.mlp(["pos"], seed)

    # L7..L(t+7) — one derivation step per layer; holder s collects the slot that fired last layer
    def leg(q):
        return [Clause(HIT, {"role": SLOT, "est1": 1, "B2": Q(q)}),
                Clause(HIT, {"role": SLOT, "est2": 1, "A2": Q(q)}), Clause(MISS, {"pos": NULLP})]

    for j in range(t + 1):
        lay = P.layer()
        for q in ("A1", "tok", "B1", "H2"):
            lay.head(leg(q), ["isX"])
        lay.head([Clause(HIT, {"role": SLOT, "fresh": 1}), Clause(HIT, {"role": SLOT, "fresh": 2}),
                  Clause(MISS, {"pos": NULLP})], ["I1", "fresh"])
        lay.head([Clause(HIT, {"role": HOLDER, "pos": Q("pos", -1)}), Clause(MISS, {"pos": NULLP})],
                 ["end"])

        def step(f, h, s=j + 1):
            live = [x.sym("isX") != 3 for x in h[:4]]
            fire1 = slotm & (f["est1"] == 0) & live[0] & live[1]
            fire2 = slotm & (f["est2"] == 0) & live[2] & live[3]
            fr = h[4].sym("fresh")
            idx = h[4].sym("I1") + (fr == 2)
            got = holdm & (hs == s) & (fr > 0)
            prev = h[5].sym("end")
            return {"est1": np.where(fire1, 1, f["est1"]), "est2": np.where(fire2, 1, f["est2"]),
                    "fresh": np.select([fire1, fire2], [1, 2], 0),
                    "I1": np.where(got, idx, f["I1"]),
                    "end": np.where(got & (prev >= O0), prev + nbits(idx) + 1, f["end"])}

        lay.mlp(["pos", "est1", "est2", "I1", "end"], step)

    # L(t+8).. — every output position window-matches its holder and writes one symbol
    pairs = [(2 * i + 1, 2 * i + 2) for i in range(g.npairs)]
    for c in range(g.emit_layers):
        chunk = pairs[3 * c:3 * c + 3]
        lay = P.layer()
        for e, e2 in chunk:
            lay.head([Clause(HIT, {"role": HOLDER, "end": Q("pos", e)}),
                      Clause(HIT, {"role": HOLDER, "end": Q("pos", e2)}),
                      Clause(MISS, {"pos": NULLP})], ["end", "I1"])
        if c == 0:
            lay.head([Clause(HIT, {"role": HOLDER, "pos": H0 + t + 1, "end": Q("pos", 0)}),
                      Clause(MISS, {"pos": NULLP})], ["pos"])
            lay.head([Clause(HIT, {"role": SLOT, "est1": 1, "isX": 1}),
                      Clause(HIT, {"role": SLOT, "est2": 1, "isX": 2}),
                      Clause(MISS, {"pos": NULLP})], ["isX"])

        def emit(f, h, c=c, n=len(chunk)):
            sym = f["isX"]
            for x in h[:n]:
                e = x.sym("end") - pos
                idx = x.sym("I1")
                ok = outm & (e >= 1) & (e <= nbits(idx) + 1)
                sym = np.where(ok, np.where(e == 1, 2, (idx >> np.maximum(e - 2, 0)) & 1), sym)
            if c == 0:
                sym = np.where(outm & (h[n].sym("pos") != NULLP), (h[n + 1].sym("isX") != 3).astype(np.int64), sym)
            return {"isX": sym}

        lay.mlp(["pos", "isX"], emit)

    return P

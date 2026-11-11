import numpy as np

from functions.rs_transformer import BF16, Clause, Program, Q

ENT0, NUM0 = 5, 414
PLAIN, FACT, RULE, SLOT, HOLDER, OUT, QE, QX = range(8)
HIT, MISS = BF16.M, BF16.M // 2
ANY = ("linear", "window", "full")
CHEAP = ("linear", "full")
FULL = ("full",)


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
        self.emit_heads = -(-self.dmax // 3)   # three offsets each, weights 4/3/2
        self.msb_off = self.dmax + 1 if self.dmax + 1 > 3 * self.emit_heads else None
        self.emit_layers = 1


def _place(d, pat, req):
    while pat[d % len(pat)] not in req:
        d += 1
    return d + 1


def _init_reqs(arch):
    # linear presets: two full load layers (3 category loads each) with the broadcast, the two
    # base-fact reads and the pad on the cheap layers between them
    if "linear" in arch.pattern:
        return [ANY, ANY, CHEAP, FULL, CHEAP, CHEAP, ANY, FULL]
    return [ANY, ANY, FULL, FULL, FULL, FULL]


def _schedule(t, arch):
    # (depth, index of the first step layer, [(kind, fires?, firing in `fresh`, firing in `I1`)])
    pat = arch.pattern
    d = 0
    for req in _init_reqs(arch):
        d = _place(d, pat, req)
    d0, steps, done = d, [], 0
    if "linear" in pat:
        # a linear step layer tags its firing in `fresh`, a full one in `I1`; the next linear layer
        # collects both tags, so a step layer of either kind advances the wavefront by one rule
        gf = gi = None
        while done < t:
            i, kind = len(steps), pat[(d0 + len(steps)) % len(pat)]
            if kind == "linear":
                done += sum(x is not None and x < t for x in (gf, gi))
                steps.append((kind, True, gf, gi))
                gf, gi = i, None
            else:
                steps.append((kind, gi is None, None, None))
                gi = i if gi is None else gi
        d = d0 + len(steps)
    else:
        steps = [("full", True, i - 1 if 1 <= i <= t else None, None) for i in range(t + 1)]
        for _ in steps:
            d = _place(d, pat, FULL)
    for _ in range(Geo(t).emit_layers):
        d = _place(d, pat, FULL)
    return d, d0, steps


def claimed_layers(t: int, arch) -> int:
    return _schedule(t, arch)[0]


def build(t: int, arch) -> Program:
    g = Geo(t)
    NULLP, S0, H0, O0 = g.NULLP, g.S0, g.H0, g.O0
    n_in, n_idx, n_sites = g.n_in, g.n_idx, g.n_sites
    ALPH = NUM0 + n_idx
    memo = "linear" in arch.pattern
    steps = _schedule(t, arch)[2]
    fields = {"pos": g.n_pos, "role": 8, "tok": ALPH, "A1": NUM0, "B1": ALPH, "E1": NUM0,
              "A2": NUM0, "B2": NUM0, "H2": NUM0, "I1": n_idx + 1, "E2": NUM0,
              "est1": 2, "est2": 2, "fresh": 3, "isX": 4, "end": g.n_pos}

    pos = np.arange(g.n_pos)
    slotm = (pos >= S0) & (pos < H0)
    site = np.where(slotm, (pos - S0) % n_sites, 0)
    fhm = (pos >= g.FH0) & (pos < S0)
    holdm = (pos >= H0) & (pos < O0)
    outm = pos >= O0
    hs = pos - H0
    base_role = np.select([pos == g.QEP, pos == g.QXP, pos < n_in, fhm, slotm, holdm, outm],
                          [QE, QX, PLAIN, SLOT, SLOT, HOLDER, OUT], PLAIN)

    def embed(tokens, n_pos):
        tok = np.zeros(n_pos, dtype=np.int64)
        tok[:n_in] = tokens
        inp = pos < n_in
        return {"pos": pos, "tok": tok, "role": base_role,
                "E1": np.where(inp & (tok < NUM0), tok, 0),
                "A1": np.where(pos == g.QEP, np.maximum(tok - ENT0 + 1, 0), 0),
                "I1": np.select([inp & (tok >= NUM0), slotm, pos == NULLP],
                                [tok - NUM0, 2 * site, n_idx], 0),
                "est1": (pos == g.FH0 + 1).astype(np.int64),
                # the line marker ':' / '>' pre-coded so L1 can read it in 2 bits instead of 10
                "isX": np.select([pos == NULLP, inp & (tok == 2), inp & (tok == 3)], [3, 1, 2], 0)}

    def readout(st):
        total = st["end"][H0 + t + 1] - O0 + 1
        vals = st["isX"][O0:O0 + total].tolist()
        return vals, vals[-1:]

    P = Program(arch, fields, n_in, g.n_pos - n_in, embed, readout)
    reqs = iter(_init_reqs(arch))

    def need(req):
        while P.next_kind() not in req:
            P.layer()
        return P.layer()

    def load(off, flds):
        return [Clause(HIT, {"role": RULE, "I1": Q("I1", off)}), Clause(MISS, {"pos": NULLP})], flds

    # L1 — parse the head of each premise line from the fixed offsets behind its newline
    lay = need(next(reqs))
    lay.head([Clause(HIT, {"pos": Q("pos", -1)})], ["tok"])
    lay.head([Clause(HIT, {"pos": Q("pos", -2)})], ["isX"])
    lay.head([Clause(HIT, {"pos": Q("pos", -3)})], ["tok", "I1"])

    def parse_a(f, h):
        g1, g2, g3, g3i = h[0].sym("tok"), h[1].sym("isX"), h[2].sym("tok"), h[2].sym("I1")
        anchor = f["pos"] < n_in - 4
        fact, rule = anchor & (g2 == 1), anchor & (g2 == 2)
        deep = pos >= 3
        base = {"role": np.select([fact, rule], [FACT, RULE], base_role),
                "isX": np.select([fact, pos == NULLP], [1, 3], 0),
                "E2": np.where(fact, g3 - ENT0 + 1, 0),
                "B1": np.where(deep, g3, 0), "H2": np.where(deep, g3i, 0)}
        if memo:  # rule head -> B2, fact category -> E1, rule leg B rides B1's raw copy
            return {**base, "B2": np.where(rule, g1, 0), "E1": np.where(fact, g1, 0)}
        return {**base, "A2": np.where(fact | rule, g1, 0), "B2": np.where(rule, g3, 0)}

    lay.mlp(["pos"], parse_a)

    # L2 — offsets 4/5/6 reached as short shifts of L1's raw copies, so this layer stays local
    lay = need(next(reqs))
    lay.head([Clause(HIT, {"pos": Q("pos", -1)})], ["B1"])
    lay.head([Clause(HIT, {"pos": Q("pos", -2)})], ["H2"])
    lay.head([Clause(HIT, {"pos": Q("pos", -3)})], ["H2"])

    def parse_b(f, h):
        g4, g5, g6 = h[0].sym("B1"), h[1].sym("H2"), h[2].sym("H2")
        rule, fact = f["role"] == RULE, f["role"] == FACT
        return {"H2": np.where(rule, g4, 0),
                "I1": np.select([rule, fact, slotm, pos == NULLP], [g6, g5, 2 * site, n_idx], 0)}

    lay.mlp(["role"], parse_b)

    if memo:
        # L3 — relay the queried entity and the queried category (key = role: 8 tuples, linear-safe)
        lay = need(next(reqs))
        lay.head([Clause(HIT, {"role": QE})], ["A1"])
        lay.head([Clause(HIT, {"role": QX})], ["tok"])
        lay.mlp(["E2"], lambda f, h: {"end": np.maximum(h[0].sym("A1"), 0),
                                      "E2": np.where(slotm, h[1].sym("tok"), f["E2"])})

        # L4 — both head categories, leg A of the first rule, the fact pair rank, the X test
        lay = need(next(reqs))
        lay.head(*load(0, ["B2"]))
        lay.head(*load(1, ["B2"]))
        lay.head(*load(0, ["H2"]))
        lay.head([Clause(HIT, {"role": FACT, "E2": Q("E2")}), Clause(-HIT, {"pos": Q("pos")}),
                  Clause(MISS, {"pos": 0})], ["isX"])

        def bind(f, h):
            return {"B2": np.where(slotm, h[0].sym("B2"), 0), "A2": np.where(slotm, h[1].sym("B2"), 0),
                    "A1": np.where(slotm, h[2].sym("H2"), 0),
                    "est1": ((h[3].sym("isX") == 1) | (pos == g.FH0 + 1)).astype(np.int64)}

        lay.mlp(["pos"], bind)

        # L5 — the queried entity's two base facts become pre-fired slots
        lay = need(next(reqs))
        for r in (0, 1):
            lay.head([Clause(HIT, {"role": FACT, "E2": Q("end"), "est1": r})], ["E1"])

        def basefacts(f, h):
            c = np.where(pos == g.FH0, h[0].sym("E1"), h[1].sym("E1"))
            return {"B2": np.where(fhm, c, f["B2"]), "E1": np.where(fhm, c, 0),
                    "est1": np.where(fhm | (pos == H0 + 1), 1, f["est1"])}

        lay.mlp(["pos", "B2", "est1"], basefacts)

        # L6 — holders 0/1 take the two base-fact premise numbers and set the output offsets
        lay = need(next(reqs))
        for r in (0, 1):
            lay.head([Clause(HIT, {"role": FACT, "E2": Q("end"), "est1": r})], ["I1"])

        def seed(f, h):
            i0, i1 = h[0].sym("I1"), h[1].sym("I1")
            l0, l1 = nbits(i0) + 1, nbits(i1) + 1
            return {"B2": np.select([holdm & (hs == 0), holdm & (hs == 1)], [i0, i1], f["B2"]),
                    "end": np.select([holdm & (hs == 0), holdm & (hs == 1)],
                                     [O0 + l0, O0 + l0 + l1], 0)}

        lay.mlp(["pos", "B2"], seed)

        # L7 — each sub-slot tests its head category against the relayed query (no head needed)
        lay = need(next(reqs))

        def xtest(f, h):
            x = f["E2"]
            return {"isX": np.select([slotm & (f["B2"] == x), slotm & (f["A2"] == x), pos == NULLP],
                                     [1, 2, 3], 0),
                    "E2": np.zeros(g.n_pos, dtype=np.int64)}

        lay.mlp(["E2", "B2", "A2"], xtest)

        # L8 — leg A of the second rule and both leg B's
        lay = need(next(reqs))
        lay.head(*load(1, ["H2"]))
        lay.head(*load(0, ["B1"]))
        lay.head(*load(1, ["B1"]))

        def legs(f, h):
            return {"B1": np.where(slotm, h[0].sym("H2"), 0), "tok": np.where(slotm, h[1].sym("B1"), 0),
                    "H2": np.where(slotm, h[2].sym("B1"), 0),
                    "I1": np.zeros(g.n_pos, dtype=np.int64),
                    "fresh": np.zeros(g.n_pos, dtype=np.int64)}

        lay.mlp(["pos"], legs)
    else:
        # L3 — relay the queried entity, rank the fact pair, load both leg A's
        lay = need(next(reqs))
        lay.head([Clause(HIT, {"role": QE}), Clause(MISS, {"pos": NULLP})], ["A1"])
        lay.head([Clause(HIT, {"role": FACT, "E2": Q("E2")}), Clause(-HIT, {"pos": Q("pos")}),
                  Clause(MISS, {"pos": 0})], ["isX"])
        lay.head(*load(0, ["H2"]))
        lay.head(*load(1, ["H2"]))

        def bind(f, h):
            ent, partner = h[0].sym("A1"), h[1].sym("isX") == 1
            return {"end": np.where(fhm | (holdm & (hs < 2)), ent, 0),
                    "est1": np.where((f["role"] == FACT) & partner, 1, (pos == g.FH0 + 1).astype(np.int64)),
                    "A1": np.where(slotm, h[2].sym("H2"), 0), "B1": np.where(slotm, h[3].sym("H2"), 0)}

        lay.mlp(["pos", "role"], bind)

        # L4 — both leg B's, and the queried entity's two base facts become pre-fired slots
        lay = need(next(reqs))
        lay.head(*load(0, ["B2"]))
        lay.head(*load(1, ["B2"]))
        lay.head([Clause(HIT, {"role": FACT, "E2": Q("end"), "est1": Q("est1")}),
                  Clause(MISS, {"pos": NULLP})], ["A2"])

        def legb(f, h):
            return {"tok": np.where(slotm, h[0].sym("B2"), 0), "H2": np.where(slotm, h[1].sym("B2"), 0),
                    "B2": np.where(fhm, h[2].sym("A2"), 0), "est1": np.where(fhm, 1, f["est1"])}

        lay.mlp(["pos", "est1"], legb)

        # L5 — both head categories
        lay = need(next(reqs))
        lay.head(*load(0, ["A2"]))
        lay.head(*load(1, ["A2"]))

        def heads(f, h):
            return {"B2": np.where(slotm, h[0].sym("A2"), f["B2"]), "A2": np.where(slotm, h[1].sym("A2"), 0)}

        lay.mlp(["pos", "B2"], heads)

        # L6 — holders 0/1 take the two base-fact premise numbers; slots test their heads
        lay = need(next(reqs))
        for r in (0, 1):
            lay.head([Clause(HIT, {"role": FACT, "E2": Q("end"), "est1": r}),
                      Clause(MISS, {"pos": NULLP})], ["I1"])
        lay.head([Clause(HIT, {"role": QX, "E1": Q("B2")}), Clause(MISS, {"pos": NULLP})], ["isX"])
        lay.head([Clause(HIT, {"role": QX, "E1": Q("A2")}), Clause(MISS, {"pos": NULLP})], ["isX"])

        def seed(f, h):
            i0, i1 = h[0].sym("I1"), h[1].sym("I1")
            l0, l1 = nbits(i0) + 1, nbits(i1) + 1
            x1, x2 = h[2].sym("isX") == 0, h[3].sym("isX") == 0
            return {"B2": np.select([holdm & (hs == 0), holdm & (hs == 1)], [i0, i1], f["B2"]),
                    "end": np.select([holdm & (hs == 0), holdm & (hs == 1)],
                                     [O0 + l0, O0 + l0 + l1], 0),
                    "isX": np.select([slotm & x1, slotm & x2, pos == NULLP], [1, 2, 3], 0),
                    "est1": fhm.astype(np.int64), "fresh": np.zeros(g.n_pos, dtype=np.int64),
                    "E1": np.where(fhm, f["B2"], 0), "E2": np.zeros(g.n_pos, dtype=np.int64)}

        lay.mlp(["pos", "B2"], seed)

    # L8.. — one derivation step per step layer; holder s collects the slot that fired last step
    def classic_leg(q):
        return [Clause(HIT, {"role": SLOT, "est1": 1, "B2": Q(q)}),
                Clause(HIT, {"role": SLOT, "est2": 1, "A2": Q(q)}), Clause(MISS, {"pos": NULLP})]

    def decode(p1, p2):
        p = np.where(p1 >= 0, p1, p2)
        return (p1 >= 0) | (p2 >= 0), 2 * ((p - S0) % n_sites) + ((p1 < 0) & (p2 >= 0))

    for kind, fires, gf, gi in steps:
        if not fires:
            need((kind,))
            continue
        lay = need((kind,))
        if memo:
            for q in ("A1", "tok", "B1", "H2"):
                tail = [] if kind == "linear" else [Clause(MISS, {"pos": NULLP})]
                lay.head([Clause(HIT, {"role": SLOT, "E1": Q(q)})] + tail, ["est1"])
                lay.head([Clause(HIT, {"role": SLOT, "E2": Q(q)})] + tail, ["est2"])
            if kind == "linear":
                for fld in ("fresh", "I1"):
                    for v in (1, 2):
                        lay.head([Clause(HIT, {fld: v})], ["pos"])
                for off in (-1, -2):
                    lay.head([Clause(HIT, {"role": HOLDER, "pos": Q("pos", off)})], ["end"])

            def memo_step(f, h, kind=kind, gf=gf, gi=gi):
                live = [(h[2 * i].sym("est1") == 1) | (h[2 * i + 1].sym("est2") == 1) for i in range(4)]
                fire1 = slotm & live[0] & live[1]
                fire2 = slotm & live[2] & live[3]
                est1 = np.where(fhm, 1, fire1).astype(np.int64)
                tag = np.select([fire1 & (f["est1"] == 0), fire2 & (f["est2"] == 0)], [1, 2], 0)
                upd = {"est1": est1, "est2": fire2.astype(np.int64),
                       "E1": np.where(est1 == 1, f["B2"], 0), "E2": np.where(fire2, f["A2"], 0)}
                if kind != "linear":
                    return {**upd, "I1": tag}
                hitf, idxf = decode(h[8].sym("pos"), h[9].sym("pos"))
                hiti, idxi = decode(h[10].sym("pos"), h[11].sym("pos"))
                e1, e2 = h[12].sym("end"), h[13].sym("end")
                end, B2 = f["end"], f["B2"]
                if gf is not None:
                    got = holdm & (hs == gf + 2) & hitf
                    end = np.where(got & (e1 >= O0), e1 + nbits(idxf) + 1, end)
                    B2 = np.where(got, idxf, B2)
                if gi is not None:
                    base = e2 + nbits(idxf) + 1
                    got = holdm & (hs == gi + 2) & hiti & hitf
                    end = np.where(got & (e2 >= O0), base + nbits(idxi) + 1, end)
                    B2 = np.where(got, idxi, B2)
                return {**upd, "fresh": tag, "I1": np.zeros(g.n_pos, dtype=np.int64),
                        "B2": B2, "end": end}

            lay.mlp(["pos", "est1", "est2", "B2", "A2", "end"], memo_step)
        else:
            for q in ("A1", "tok", "B1", "H2"):
                lay.head(classic_leg(q), ["isX"])
            lay.head([Clause(HIT, {"role": SLOT, "fresh": 1}), Clause(HIT, {"role": SLOT, "fresh": 2}),
                      Clause(MISS, {"pos": NULLP})], ["I1", "fresh"])
            lay.head([Clause(HIT, {"role": HOLDER, "pos": Q("pos", -1)}), Clause(MISS, {"pos": NULLP})],
                     ["end"])

            def step(f, h, gf=gf):
                live = [x.sym("isX") != 3 for x in h[:4]]
                fire1 = slotm & (f["est1"] == 0) & live[0] & live[1]
                fire2 = slotm & (f["est2"] == 0) & live[2] & live[3]
                fr = h[4].sym("fresh")
                idx = h[4].sym("I1") + (fr == 2)
                got = holdm & (hs == (0 if gf is None else gf + 2)) & (fr > 0) & (gf is not None)
                prev = h[5].sym("end")
                return {"est1": np.where(fire1, 1, f["est1"]), "est2": np.where(fire2, 1, f["est2"]),
                        "fresh": np.select([fire1, fire2], [1, 2], 0),
                        "B2": np.where(got, idx, f["B2"]),
                        "end": np.where(got & (prev >= O0), prev + nbits(idx) + 1, f["end"])}

            lay.mlp(["pos", "est1", "est2", "B2", "end"], step)

    # emit — three offsets per head, ranked by weight so the nearest holder wins; the entry's
    # leading bit is always 1, so the longest offset needs a bare match and no premise number
    nb, msb = g.emit_heads, g.msb_off
    lay = need(FULL)
    for j in range(nb):
        lay.head([Clause(HIT - w, {"role": HOLDER, "end": Q("pos", 3 * j + 1 + w)}) for w in range(3)],
                 ["end", "B2"])
    if msb is not None:
        lay.head([Clause(HIT, {"role": HOLDER, "end": Q("pos", msb)})], ["end"])
    lay.head([Clause(HIT, {"role": HOLDER, "pos": H0 + t + 1, "end": Q("pos", 0)}),
              Clause(MISS, {"pos": NULLP})], ["pos"])
    lay.head([Clause(HIT, {"role": SLOT, "est1": 1, "isX": 1}),
              Clause(HIT, {"role": SLOT, "est2": 1, "isX": 2}),
              Clause(MISS, {"pos": NULLP})], ["isX"])

    def emit(f, h):
        sym = f["isX"]
        if msb is not None:
            sym = np.where(outm & (h[nb].sym("end") - pos == msb), 1, sym)
        for x in reversed(h[:nb]):
            e = x.sym("end") - pos
            idx = x.sym("B2")
            ok = outm & (e >= 1) & (e <= nbits(idx) + 1)
            sym = np.where(ok, np.where(e == 1, 2, (idx >> np.maximum(e - 2, 0)) & 1), sym)
        n = nb + (msb is not None)
        return {"isX": np.where(outm & (h[n].sym("pos") != NULLP),
                                (h[n + 1].sym("isX") != 3).astype(np.int64), sym)}

    lay.mlp(["pos", "isX"], emit)

    return P

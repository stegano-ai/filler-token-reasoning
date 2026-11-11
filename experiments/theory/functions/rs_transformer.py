import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Arch:
    p: int = 8            # precision bits of every aggregated (attention-averaged) value
    H: int = 8            # heads per layer
    F: int = 16           # fields per position
    M: int = 4            # max |attention score|
    C: int = 4            # max clauses per head
    c_beta: float = 2.0   # logits = c_beta * ln(n_pos) * score
    scratch_exp: float = 1.0
    alpha_exp: float = 2.0
    const_alphabet: int = 1024   # fields with a larger alphabet are positional codes, exempt from mlp_bits
    mlp_bits: int = 32           # max log2(table size) of one MLP over its constant-alphabet inputs
    max_depth: int = 200
    layers: int = 200            # the model's depth (drawn on the frontier; circuits may exceed it)
    pattern: tuple = ("full",)   # layer kinds by index, cycled: full | window | linear
    window: int = 128            # keys visible to a window layer: j > i - window
    lin_keys: int = 128          # distinct key tuples a linear-attention head can hold (delta-rule memory)
    conv: int = 3                # linear layers may copy from pos-1..pos-conv (depthwise conv kernel 4)

    def beta(self, n_pos: int) -> float:
        return self.c_beta * math.log(max(n_pos, 2))


BF16 = Arch()
# per-model presets from the HF configs (2026-09-22): heads = query heads, pattern = layer_types
QWEN = Arch(H=16, layers=40, pattern=("linear", "linear", "linear", "full"), lin_keys=128, conv=3)
OSS = Arch(H=64, layers=24, pattern=("window", "full"), window=128)
ARCHS = {"bf16": BF16, "qwen": QWEN, "oss": OSS}


@dataclass(frozen=True)
class Q:
    field: str
    offset: int = 0


@dataclass
class Clause:
    weight: int
    conds: dict  # key field -> Q(query field, offset) | int constant


@dataclass
class Head:
    score: list  # of Clause
    value: list  # key fields whose bit codes are averaged


def bits_for(alphabet: int) -> int:
    return max(1, math.ceil(math.log2(alphabet)))


def to_bits(x: np.ndarray, nbits: int) -> np.ndarray:
    return ((x[:, None] >> np.arange(nbits - 1, -1, -1)[None, :]) & 1).astype(np.float32)


class HeadOut:
    def __init__(self, out: np.ndarray, layout: list):
        self._out, self._layout = out, layout

    def _bits(self, fld: str) -> np.ndarray:
        for name, lo, hi, _ in self._layout:
            if name == fld:
                return self._out[:, lo:hi]
        raise KeyError(fld)

    def frac(self, fld: str) -> np.ndarray:
        # the averaged value of a 1-bit field: the only aggregate the MLP may read (a count / 2^p)
        assert next(a for n, _, _, a in self._layout if n == fld) == 2, f"frac() only on 1-bit fields, not {fld}"
        return self._bits(fld)[:, 0]

    def sym(self, fld: str) -> np.ndarray:
        # exact copy where every bit is exactly 0 or 1 (a unique match); -1 anywhere the head averaged
        b = self._bits(fld)
        ok = ((b == 0) | (b == 1)).all(1)
        v = ((b > 0.5) * (1 << np.arange(b.shape[1] - 1, -1, -1))[None, :]).sum(1).astype(np.int64)
        return np.where(ok, v, -1)


@dataclass
class Layer:
    kind: str = "full"
    heads: list = field(default_factory=list)
    reads: list = field(default_factory=list)
    fn: object = None

    def head(self, score: list, value: list) -> "Layer":
        self.heads.append(Head(score, value))
        return self

    def mlp(self, reads: list, fn) -> "Layer":
        self.reads, self.fn = reads, fn
        return self


class Program:
    def __init__(self, arch: Arch, fields: dict, n_input: int, n_scratch: int, embed, readout):
        self.arch, self.fields, self.n_input, self.n_scratch = arch, fields, n_input, n_scratch
        self.embed, self.readout, self.layers = embed, readout, []
        self.n_pos = n_input + n_scratch
        assert len(fields) <= arch.F, f"{len(fields)} fields > F={arch.F}"
        assert n_scratch <= 4 * n_input ** arch.scratch_exp, f"scratch {n_scratch} exceeds budget"
        assert max(fields.values()) <= 4 * self.n_pos ** arch.alpha_exp, "alphabet exceeds budget"

    @property
    def depth(self) -> int:
        return len(self.layers)

    def next_kind(self) -> str:
        return self.arch.pattern[self.depth % len(self.arch.pattern)]

    def layer(self) -> Layer:
        L = Layer(kind=self.next_kind())
        self.layers.append(L)
        return L

    def _pairs(self, cl: Clause, st: dict) -> tuple:
        # every (query i, key j <= i) matching all conds of the clause, via hashing key tuples
        n = self.n_pos
        keyt = np.stack([st[f] for f in cl.conds], 1)
        qryt = np.stack([(st[r.field] + r.offset) if isinstance(r, Q) else np.full(n, r) for r in cl.conds.values()], 1)
        _, inv = np.unique(np.concatenate([keyt, qryt]), axis=0, return_inverse=True)
        kid, qid = inv[:n].ravel(), inv[n:].ravel()
        order = np.argsort(kid, kind="stable")
        composite = kid[order].astype(np.int64) * n + order
        lo = np.searchsorted(composite, qid * n, "left")
        hi = np.searchsorted(composite, qid * n + np.arange(n), "right")
        cnt = hi - lo
        idx = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)
        return np.repeat(np.arange(n), cnt), order[idx]

    def _linear(self, h: Head, st: dict) -> "HeadOut":
        # gated-delta-rule associative memory: one clause; a query reads the value written by the most
        # recent key (j <= i) matching it; <= lin_keys distinct key tuples per head; pos offsets within
        # the conv kernel are short-range shifts and exempt from the key budget
        n = self.n_pos
        assert len(h.score) == 1, "a linear-attention head has exactly one clause"
        cl = h.score[0]
        content = {kf: r for kf, r in cl.conds.items()
                   if not (kf == "pos" and isinstance(r, Q) and r.field == "pos" and -self.arch.conv <= r.offset <= 0)}
        if content:
            keyt = np.stack([st[f] for f in content], 1)
            assert len(np.unique(keyt, axis=0)) <= self.arch.lin_keys, f"linear head sees > lin_keys={self.arch.lin_keys} distinct keys"
        i, j = self._pairs(cl, st)
        last = np.full(n, -1, dtype=np.int64)
        np.maximum.at(last, i, j)
        layout, cols, lo = [], [], 0
        for f in h.value:
            nb = bits_for(self.fields[f])
            cols.append(to_bits(st[f], nb).astype(np.float64))
            layout.append((f, lo, lo + nb, self.fields[f]))
            lo += nb
        V = np.concatenate(cols, 1)
        out = np.where((last >= 0)[:, None], V[np.maximum(last, 0)], 0.5)
        return HeadOut(out, layout)

    def _attend(self, L: Layer, st: dict, beta: float) -> list:
        # exact causal softmax: constant-only clauses give every key a query-independent base score s0(j),
        # handled by prefix sums; clauses with query refs contribute sparse (i, j) pairs on top
        n, outs = self.n_pos, []
        w = self.arch.window if L.kind == "window" else n
        for h in L.heads:
            if L.kind == "linear":
                outs.append(self._linear(h, st))
                continue
            assert len(h.score) <= self.arch.C, f"{len(h.score)} clauses > C={self.arch.C}"
            s0 = np.zeros(n)
            I, J, Wt = [np.zeros(0, dtype=np.int64)], [np.zeros(0, dtype=np.int64)], [np.zeros(0)]
            for cl in h.score:
                if any(isinstance(r, Q) for r in cl.conds.values()):
                    i, j = self._pairs(cl, st)
                    keep = j > i - w
                    i, j = i[keep], j[keep]
                    I.append(i); J.append(j); Wt.append(np.full(len(i), float(cl.weight)))
                else:
                    m = np.ones(n, dtype=bool)
                    for kf, ref in cl.conds.items():
                        m &= st[kf] == ref
                    s0 += cl.weight * m
            I, J, Wt = np.concatenate(I), np.concatenate(J), np.concatenate(Wt)
            if len(I) and len(h.score) > 1:
                pair, inv = np.unique(I * n + J, return_inverse=True)
                I, J, Wt = pair // n, pair % n, np.bincount(inv, weights=Wt)
            pi, pj = I, J
            S = s0[pj] + Wt
            assert abs(s0).max() <= self.arch.M and (not len(S) or abs(S).max() <= self.arch.M), f"|score| > M={self.arch.M}"
            layout, cols, lo = [], [], 0
            for f in h.value:
                nb = bits_for(self.fields[f])
                cols.append(to_bits(st[f], nb).astype(np.float64))
                layout.append((f, lo, lo + nb, self.fields[f]))
                lo += nb
            V = np.concatenate(cols, 1)
            e0 = np.exp(beta * s0)
            delta = np.exp(beta * S) - e0[pj]
            if w < n:  # window layer: background is the sum over the last w keys, summed directly (no cancellation)
                win = lambda x: np.convolve(x, np.ones(w))[:n]
                c1 = win(e0)
                cV = np.stack([win(e0 * V[:, b]) for b in range(V.shape[1])], 1)
            else:
                c1, cV = np.cumsum(e0), np.cumsum(e0[:, None] * V, 0)
            denom = c1 + np.bincount(pi, weights=delta, minlength=n)
            numer = cV
            Vp = delta[:, None] * V[pj]
            numer += np.stack([np.bincount(pi, weights=Vp[:, b], minlength=n) for b in range(V.shape[1])], 1)
            out = np.round(numer / denom[:, None] * 2 ** self.arch.p) / 2 ** self.arch.p
            outs.append(HeadOut(out, layout))
        return outs

    def run(self, tokens: np.ndarray) -> dict:
        assert len(tokens) == self.n_input
        st = {f: np.zeros(self.n_pos, dtype=np.int64) for f in self.fields}
        st.update(self.embed(tokens, self.n_pos))
        beta = self.arch.beta(self.n_pos)
        for L in self.layers:
            assert len(L.heads) <= self.arch.H, f"{len(L.heads)} heads > H={self.arch.H}"
            outs = self._attend(L, st, beta)
            if L.fn is not None:
                bits = sum(math.log2(self.fields[f]) for f in L.reads if self.fields[f] <= self.arch.const_alphabet)
                bits += sum(hi - lo if a <= self.arch.const_alphabet else 0
                            for o in outs for _, lo, hi, a in o._layout)
                assert bits <= self.arch.mlp_bits, f"MLP reads {bits:.1f} bits > mlp_bits={self.arch.mlp_bits}"
                upd = L.fn({f: st[f] for f in L.reads}, outs)
                for f, v in upd.items():
                    v = np.asarray(v, dtype=np.int64)
                    assert ((v >= 0) & (v < self.fields[f])).all(), f"field {f} left its alphabet"
                    st[f] = np.broadcast_to(v, (self.n_pos,)).copy()
        return st

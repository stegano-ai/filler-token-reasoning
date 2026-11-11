# radix_window (oss preset)

Sliding radix scan over the affine monoid of radix_scan (element k at pos 2k, `a` = index into
{0,1,3,5,7} and `b` = offset, 40 elements), re-scheduled for gpt-oss-20b's window/full pattern.

Invariant: after a layer with running coverage L, position 2e holds the product over
[max(0, e-L+1), e]. Layer 0 builds L = r_0 elements from the raw tokens (head 0 = own op at pos-1,
then for t = 1..r_0-1 the operand at pos-2t and the op at pos-2t-1); layer i >= 1 with radix r reads
partners at pos - 2tL for t = 1..r-1 and multiplies L by r. Partners off the left end return sym = -1,
the zero map, which the accumulator (already holding the start constant, a = 0) annihilates.

The only preset-specific constraint: an **even (window) layer sees keys j > i - 128**, so a partner at
2tL positions back is invisible once 2(r-1)L >= 128. The radix per layer is therefore not fixed —
`schedule()` searches for the shallowest sequence of radices (each <= 5: 32 MLP bits allow own 5.32 +
4 partners * 6; layer 0 also needs 2r-1 <= H heads) with the window constraint applied on even layer
indices. It makes window layers earn their depth while L is short (e.g. n = 384: radices 4, 5, 4, 5 —
the third layer is a window layer composing L = 20 at distance 120 < 128) and spends them as no-ops
(radix 1) once L > 63 elements. Every head also carries a constant clause Clause(-M, {one: 1}) so the
non-matching background sits at -M and the matched key at +M: the score gap is 2M, which keeps the copy
exact at n_pos = 3 under the c_beta = 1.5 robustness variant.

layers(n): 1 (n <= 4), 2 (<= 24), 3 (<= 74), 4 (<= 399), 6 (<= 1999) — window layers cost the extra
layer at n >= 400, where the ideal model needs 5. Verified n = 1..1024 under oss (depth 6 at n = 1024,
model depth 24), robustness pass clean.

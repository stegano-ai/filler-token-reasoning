# hs_scan — Hillis–Steele radix-4 sliding-window scan (best for oss)

Invariant after layer L: `g` at position i is the product of the instructions in the last B_L
positions ending at i (saturating at the sequence start), `out = g(1)`. Unlike the block-aligned
scan the partner of a step is a *fixed* positional offset, so a partner is missing only at the very
start of the sequence — which is exactly where the identity anchor at position 0 is inside a
window layer's 128-key view. That is what lets a window layer carry a step.

**Layer 1 (fold), B_1 = min(H + 1, largest B with 1 + log2(B+1) + (B−1) ≤ 32 mlp_bits)** = 27 oss,
9 bf16. B_1 − 1 heads copy `tok` from `pos == pos_q − m`; the MLP folds head m iff m < `cnt` =
min(i, B_1), so the window saturates at the start and the anchor's `tok` is never read.

**Each later layer, B → rB (r ≤ 4).** r − 1 heads with two clauses: weight 4 on
`pos == pos_q − mB AND anch == 0`, weight 2 on `anch == 1`. Tiers 4 / 2 / 0 — partner, anchor,
rest; `anch == 0` on the partner clause keeps the two from stacking to 6 > M when the partner *is*
position 0. Missing partner (i < mB) or partner = anchor → an exact identity read. MLP composes
own ∘ p₁ ∘ p₂ ∘ p₃ = log2(169) + 3·8 = 31.4 ≤ 32 bits, so r ≤ 4.

Schedule: a full layer takes r = 4; a **window** layer takes the largest r with (r − 1)B ≤ 127;
a linear layer cannot do a long positional offset at all, so this idea is not for qwen. B_1 is
chosen per n as the value minimising the layer count (ties → larger B_1), which is why a window
step is worth taking only when it is not too late: oss reaches B = 27, 108, 216, 864 on layers
1..4 and 3456 on layer 6, i.e. layers(n) = 1, 2, 3, 4, 6 for n ≤ 27, 108, 216, 864, 3456.
bf16 (all full) → 9, 36, 144, 576, 2304, the same 1 + ⌈log₄(n/9)⌉ as block_scan.
Verified at every checkpoint to n = 1024 (45 s robustness pass): oss 4 layers of 24 up to n = 768,
6 at n = 1024; bf16 5.

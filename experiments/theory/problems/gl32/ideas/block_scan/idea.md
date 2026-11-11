# block_scan — block-aligned radix-4 prefix scan, scheduled onto the presets (best for qwen)

Invariant after the layer that establishes level L: `g` at position i is the product of the
instructions from the start of i's level-L block to i (block size B_L, aligned to the token index);
`out = g(1)` is rewritten every layer and is the readout. Tokens sit at 1..n, position 0 is an
identity anchor, no worker region. `g` has alphabet 169 with slot 168 = identity, so `sym()`'s −1
("no match") composes as identity and the missing-partner case costs no MLP bits.

**Layer 1 (fold), B_1 = min(H + 1, largest B with 1 + log2(B+1) + (B−1) ≤ 32 mlp_bits)** = 17 qwen,
27 oss, 9 bf16. B_1 − 1 heads copy `tok` from the previous positions; the MLP reads own `tok`,
the heads (1 bit each) and `cnt` = (i mod B_1) + 1 and folds its own block's instructions.
On a softmax layer the head key is `pos == pos_q − m`; on a linear layer it is `pm == pm_q − m`
with pm = i mod B_1 (17 ≤ lin_keys distinct keys, and the query key goes negative exactly where
`cnt` masks the head out).

**Each later layer, B → 4B.** 3 heads fetch the totals of the 3 preceding level-L blocks inside the
same level-(L+1) block (a block's total is its last position's `g`) and the MLP composes
own ∘ p₁ ∘ p₂ ∘ p₃ — log2(169) + 3·8 = 31.4 ≤ 32 bits, so radix 4 is the MLP's maximum for a
168-element group. Head m:
- **linear layer** (qwen): one clause `k_L == k_L(q) − m AND k_{L+1} == k_{L+1}(q)`. The delta rule
  returns the most recent match = the last position of block k − m, so no block-end flag is needed;
  a partner in the previous parent has no key at all → identity. Distinct key tuples =
  ⌈n/B_L⌉ + 1 ≤ lin_keys = 128, which is why the fold must come first.
- **softmax layer** (full): the same two conditions plus `e_L == 1` (block end), score tiers
  4 (partner) / 2 (anchor) / 0 (other block ends) / −2 (rest). The anchor tier makes a missing
  partner read an exact identity instead of an average. A window layer can never reach
  4·B_L − 1 > 127 back, so it is left empty.

Schedule: qwen (linear,linear,linear,full) steps on every layer → B = 17, 68, 272, 1088,
layers(n) = 1 + ⌈log₄(n/17)⌉; bf16 → B = 9, 36, 144, 576, 2304. Verified at every checkpoint to
n = 1024 (45 s robustness pass, p / p−1 / c_beta 1.5): qwen 4 layers of 40, bf16 5.

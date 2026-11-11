# block_scan — block-aligned radix-5 prefix scan, scheduled onto the model presets

Invariant after the layer that establishes level L: `g` at position i is the product of the
instructions from the start of i's level-L block to i (block size B_L, blocks aligned to the token
index). The answer is `g` once B_L ≥ n, read out as the image of cup 0. Everything is computed in
place on the token positions; position 0 is an identity anchor (g = identity forever), tokens sit at
1..n, so no scratch workers and no long hop to a worker region. `g` has alphabet 61 with slot 60 =
identity, so `sym()`'s −1 for "no match" indexes the last column of COMP and composes as identity —
the missing-partner case needs no extra MLP bits.

**Layer 1 (fold), B_1 = min(H + 1, largest B with 1 + log2(B+1) + (B−1) ≤ 32 mlp_bits)** = 17 qwen,
27 oss, 9 bf16. B_1 − 1 heads copy `tok` from the previous B_1 − 1 positions; the MLP reads own
`tok` (1 bit), the heads (1 bit each) and `cnt` = (i mod B_1) + 1, and folds the `cnt` instructions
of its own block. Head form by layer kind: a softmax layer (full / window) uses the exact offset
`pos == pos_q − m` (distance ≤ 26 ≪ window 128); a linear layer uses the key `pm` = i mod B_1
(B_1 ≤ 128 distinct keys, query `pm_q − m`, which is negative exactly when m leaves the block, so
the association misses precisely where `cnt` masks it out).

**Each later layer, B → 5B.** 4 heads fetch the totals of the 4 preceding level-L blocks inside the
same level-(L+1) block — the total of a block is its last position's `g` — and the MLP composes
own ∘ p₁ ∘ p₂ ∘ p₃ ∘ p₄ (5.93 + 4·6 = 29.93 ≤ 32 bits, so radix 5 is the MLP's maximum). Head m:
- **linear layer** (qwen): one clause `k_L == k_L(q) − m AND k_{L+1} == k_{L+1}(q)`. The delta-rule
  memory returns the *most recent* match = exactly the last position of block k−m, so no
  block-end flag is needed, and a block in the previous parent simply has no key → identity.
  Distinct key tuples = ⌈n/B_L⌉ + 1 ≤ lin_keys = 128, which is why the fold must come first.
- **softmax layer** (full; a window layer can never reach 4·B_L > 127 keys back and is left empty):
  the same two conditions plus `e_L == 1` (last position of its level-L block) for a unique match,
  with score tiers 4 (the partner) / 2 (the anchor, which also carries `e_L = 1`) / 0 (other block
  ends) / −2 (everything else, via a constant clause on `e_L == 0`). The anchor tier is what makes
  a missing partner read an exact identity instead of an average of the background.

Schedule: layer kind by index decides. qwen (linear,linear,linear,full) runs every step, so
B = 17, 85, 425, 2125 → layers(n) = 1 + ⌈log₅(n/17)⌉. oss (window,full) can only step on the odd
layers, so B = 27, 135, ·, 675, ·, 3375 → layers(n) = 1 + 2⌈log₅(n/27)⌉. bf16: B = 9, 45, 225, 1125.
Verified at every checkpoint to n = 1024: qwen 4 layers (of 40), oss 6 (of 24), bf16 4.

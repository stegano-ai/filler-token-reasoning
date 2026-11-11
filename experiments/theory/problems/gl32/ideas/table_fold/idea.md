# table_fold — 1 layer for n ≤ 9, then 1 + ⌈log₄(n/9)⌉ (window 9, then ×4 per layer)

Position 0 is a scratch cell holding the identity of GL(3,2) (`g = 0`, `e = 0`); inputs sit at
1..n. Position i keeps `g` = product of the instructions in its window (i − W + 1 .. i), one
field of alphabet 168 (table MUL from group.py); `out = g(1)` is rewritten every layer.
Every head has three clauses: −M on everything (`one == 1`), +2M on the partner `ipos == pos − mW`
(ipos = n_pos at the scratch cell so it never matches), +M on `pos == 0`. Partner present → score 4,
unique, exact copy; absent → the identity cell wins at score 0 (everything else at −4). So the
fold never needs a guard and the MLP reads no positional field.
Layer 1: 8 heads copy `e` (identity / shift / couple, alphabet 3) from pos − 1..−8; MLP folds 9
symbols (1.6 + 8·2 = 17.6 bits) → W = 9. Layer j ≥ 2: 3 heads copy `g` from pos − W, −2W, −3W;
MLP folds 4 elements (log₂168 + 3·8 = 31.4 ≤ 32 bits) → W ×= 4.
v2 deviation from the doc (1 for n ≤ 11; 2 + ⌈log₃(n/44)⌉): the 11-wide first layer needed two
bits per head via slot packing, which `frac` now forbids, so layer 1 is capped at H + 1 = 9; the
doc's ×4 then ×3 tail is replaced by ×4 throughout, the largest fold that fits mlp_bits = 32.
Equal to the doc at every checkpoint except n = 512 (4 vs 5). radix9 (72-bit fold) is gone.

# serial — n − 1 layers

Embedding: every position holds its instruction bit and `out = STEP[tok, 1]` (correct only at
position 0). Layer j: one head copies `out` from pos − 1 (score +M on pos − 1, −M on self, so
the match is unique and the copy exact even at n_pos = 2); the MLP applies the position's own
instruction to it, guarded by pos ≥ 1. After layer j positions ≤ j are final, so n − 1 layers
finish the chain; the readout is the `out` field.
v2 change: added the self-exclusion clause — without it the self key's e^0 leak at n_pos = 2
(1/257 > 2^-9) made `sym` return −1.

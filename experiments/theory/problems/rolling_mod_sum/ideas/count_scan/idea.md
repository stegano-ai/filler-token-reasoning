# count_scan — radix-512 counting scan, layers(n) = ceil(log_512 n)

Uses the model's per-head counting capacity: a head whose clause matches exactly c <= 64 keys
with equal score averages their 1-bit `s` (frac(), the one aggregate v2 allows), giving k/c,
which survives rounding to 2^-8 (and 2^-7; c <= 2^(p-1)) and the MLP inverts it to the exact
count k = round(frac * c); c is known from pos. Single-clause heads; MLP reads pos + s + 8 head
bits <= 19 bits <= mlp_bits. Semantics v2 changed nothing here except frac() being 1-D.

Layer l (block size u = 512^(l-1) at level l-1, 512^l at level l): head j attends to the ends
of the level-(l-1) blocks lying in sub-range j (64 blocks each) of the query's level-l block
(causal, so only blocks at or before the query). The MLP knows from pos how many keys each
head matched (64, partial, or 0), recovers the counts, and sets s(i) = own level-(l-1) prefix
+ sum of preceding complete level-(l-1) blocks = prefix within the level-l block. After
ceil(log_512 n) layers every position holds its full prefix parity; readout as in serial.

Deviation from the doc: the doc's ceil(log_r n) with r = heads+1 uses one unique-match fetch
per head (r = 9 here); counting 64 keys per head gives r = 8*64 = 512 instead. n = 1 needs 0
layers (embedding seeds s = tok), so no max(1, .) floor. Uses only 0/1 operands and mod 2.

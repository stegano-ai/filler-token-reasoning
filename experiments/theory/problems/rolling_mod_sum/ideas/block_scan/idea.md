# block_scan — one block-prefix scan layer per level, radix set by the layer's kind

Invariant after layer l: every position holds the parity of its own level-l block up to itself
(u_0 = 1, u_{l+1} = u_l · R_{l+1}). A layer raises the level: the query at block index
q = (i/u) mod R adds the sums of the q preceding level-l blocks, which sit at those blocks' END
positions (there the invariant is the whole block sum). Done when u ≥ n; readout = s.
The radix R of each layer is whatever its kind can reach, so the same circuit schedules onto
all three presets:

- **full** (bf16 all layers, qwen layer 3/7/…, oss odd layers): tie-counting. Head j matches
  {same level-(l+1) block, sub == j, block end} with a −M self-exclusion clause, so it averages
  exactly cnt_j = clip(q − 64j, 0, 64) ones and the MLP reads the count back as
  round(frac·cnt_j). R = 64·h, h = the largest head count with h + log2(64h) + 1 ≤ 32 bits
  (bf16 8 → R=512, qwen 16 → R=1024, oss 20 of its 64 heads → R=1280).
- **window** (oss even layers): the same head, but the farthest key is u·R−1 back, so R ≤ 128/u.
  Only layer 0 is useful (R = 128); at u ≥ 128 the layer is a forced no-op.
- **linear** (qwen layers 0,1,2, …): no averaging, so each of the 16 heads brings back one block.
  Key field b = pos/u; head k asks for b == b(query) − k and the delta rule returns the most
  recent such position, i.e. that block's end. R = 17. If the sequence holds more than
  lin_keys = 128 blocks (only layer 0, where u = 1 and n > 128) the key must be b mod 128 and a
  second head per shift covers the residues that wrapped past 0, halving it to R = 9.
  MLP reads q (log2 R) + own s + one bit per head ≤ 21 bits.

Closed form, layers(n) = number of steps until ∏R ≥ n:
- **qwen** (all linear, no full layer needed): 1 for n ≤ 17 (b = pos, 16 shifts), 2 for n ≤ 153
  (9·17), 3 for n ≤ 1377 (9·17·17 = 2601 while n ≤ 1152, else 9·9·17), 4 for n ≤ 746496 (layer 3
  is the first full layer, radix 1024). The key budget is the bottleneck: a linear layer drops to
  radix 9 whenever the sequence holds more than 128 blocks of the current size. Verified 2…1024.
- **oss**: 1 for n ≤ 128 (the window), 2 for n ≤ 163840 (128·1280); beyond that the next window
  layer is a forced no-op, so 4. Verified 2…1024.
- **bf16**: ⌈log_512 n⌉ — same as count_scan. Verified 2…1024.

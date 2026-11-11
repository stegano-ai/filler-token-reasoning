# contain_hit — interval merge with a local containment flag, scheduled onto the model presets

Same 3-way bidirectional interval merge as `interval_merge3` (radius r -> 3r+1 per round, so
K = ceil(log3(h+1)) rounds), but the final containment test is a one-bit flag computed inside the
last round's MLP and aggregated by a single-key head, which lets the cheap layer kinds carry it.

Layers, by kind:

- **parse** — 1 full/window layer (offsets -1,-3,-4,-5,-6 of `tok`, all inside the 128-key window);
  under the qwen pattern a linear layer only shifts by -3..0 (the conv kernel), so it becomes 2
  linear layers: `t1,t3 = tok@-1,-3`, then `t1@-3, t3@-2, t3@-3` = `tok@-4,-5,-6`.
- **broadcast (e, T)** — every copy needs the query's entity and target to test containment locally.
  One clause keyed on `kind == 3` is a bounded-key associative read (4 distinct keys << lin_keys),
  so a **linear** layer does it; under window/full patterns it instead rides as a second head on the
  mirror layer, free.
- **mirror** — 1 **full** layer: copy of premise i fetches (subj, obj) by `idx` (NP+1 distinct keys
  and arbitrary distance, so neither a linear nor a window layer can do it), setting L = subj,
  F = obj. Blocks 1..K+1, block b running rounds 1..b-1 off block b-1, as in interval_merge3.
- **round k (k = 1..K)** — 1 **full** layer each: content matches at arbitrary distance (block b
  reads block b-1, exactly NP positions back, which is beyond the 128-key window), on key spaces of
  size Theta(n). The last round's MLP also writes `hit = (blk == K+1) & (L == e) & (F == T)`.
- **exists** — `hit` is a 1-bit field with 2 distinct key values, so the aggregation
  "some copy spans e..T" is a **linear** layer under qwen (most recent j with `hit == 1`), and a
  constant-clause softmax head elsewhere. Both read out as `sym(hit) == 1`.

Per preset (K = ceil(log3(h+1))):

| preset | schedule | layers(h) | h = 512 |
|---|---|---|---|
| bf16 | parse, mirror+bcast, K rounds, exists | **K + 3** | 9 |
| oss (window/full) | window parse, full mirror+bcast, K rounds on the odd layers, full exists | **2K + 4** | 16 |
| qwen (lin,lin,lin,full) | 2 linear parse, linear bcast, full mirror, K rounds one per 4 layers, linear exists | **4K + 5** | 29 |

Both fit their model (24 / 40 layers) for every h <= 1024. Verified to n = 512 under all three.

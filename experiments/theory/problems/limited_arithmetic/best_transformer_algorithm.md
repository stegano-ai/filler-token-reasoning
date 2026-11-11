# limited_arithmetic (modular) — best known transformer algorithm

**layers(n) = ⌈log_r n⌉ + 1**, r = heads + 1 (one extra layer folds the start value;
otherwise identical to the mod-sum scan over Aff(Z_m)). Audited wf_d60028f2.

Per model preset (2026-09-22), the certified constructions are `ideas/block_scan` and `ideas/radix_window`:
- qwen: **⌈log_5(n+1)⌉** (block_scan) — every linear layer folds radix 5 through a mod-5 key field whose
  negative query value gates the block boundary for free, so no layer is wasted; verified n ≤ 1024 (5 layers).
- oss: **1, 2, 3, 4, 6 layers for n ≤ 4, 24, 74, 399, 1999** (radix_window) — the 128-key window caps a
  window layer's radix at 1 + ⌊63/L⌋, so its depth-4 reach is 400 and it spends one no-op layer above that;
  verified n ≤ 1024 (6 layers). r = 5 on both: the 32-bit MLP cap, not the head count, limits the radix.

Prior version and full audit trail: git history + workflow journals wf_46c71f48, wf_9af6b9bf, wf_d60028f2.

# rolling_mod_sum — best known transformer algorithm

**layers(n) = number of block-scan levels until the product of the per-layer radices reaches n**
(idea `block_scan`, certified per preset; each layer's radix is what its kind can reach).

- bf16: ⌈log_512 n⌉ — a full layer counts 64 tied keys on each of 8 heads.
- oss-20b: 1 for n ≤ 128, 2 for n ≤ 163840 — layer 0 is a window layer (reach 128), layer 1 a
  full layer with radix 64·20 = 1280 (20 heads is what the 32-bit MLP allows).
- Qwen3.6-35B-A3B: 1 for n ≤ 17, 2 for n ≤ 153, 3 for n ≤ 1377, 4 for n ≤ 746496 — its first
  three layers are linear-attention, where each of the 16 heads returns one block end (radix 17);
  the delta-rule key budget of 128 forces radix 9 on any layer whose block size leaves more than
  128 blocks in the sequence (always layer 0 once n > 128).

The older r = (heads + 1) unique-fetch form (audit wf_d60028f2) is superseded by tie-counting on
full layers; the linear-layer radix 17 is that form. Audit trail: git history + workflow journals
wf_46c71f48, wf_9af6b9bf, wf_d60028f2.

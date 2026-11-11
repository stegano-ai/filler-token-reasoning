# gl32 — best known transformer algorithm

**layers(n) = 1 for n ≤ 11; 2 + max(0, ⌈log₃(n/44)⌉) for larger n.** Same audited
table-folding construction as ball_and_cups (|GL(3,2)| = 168 gives slightly smaller
windows). n = 1024 → 5 layers.

Prior version and full audit trail: git history + workflow journals wf_46c71f48, wf_9af6b9bf, wf_d60028f2.

**Certified per preset (2026-09-22, verified at every checkpoint to n = 1024 with a 45 s
robustness pass).** Both new ideas are radix-4 scans over the 168-element group: one fold layer
turns B₁ = min(H + 1, 27) instructions into a group element (17 qwen, 27 oss, 9 bf16), and each
later layer multiplies the covered length by 4 (own ∘ 3 partners = log₂169 + 3·8 = 31.4 ≤ 32
mlp_bits, so radix 4 is the MLP's maximum for this group).

- qwen: **1 + ⌈log₄(n/17)⌉** (`ideas/block_scan`, block-aligned) — a linear layer steps too,
  because the delta-rule memory keyed by (block, parent block) returns the last position of a
  preceding block, so every layer steps. B = 17, 68, 272, 1088 → **4 layers at n = 1024** (of 40).
- oss: **1, 2, 3, 4, 6 layers for n ≤ 27, 108, 216, 864, 3456** (`ideas/hs_scan`,
  Hillis–Steele sliding window) — with fixed positional offsets a partner is missing only at the
  sequence start, where the identity anchor is still inside a window layer's 128-key view, so a
  window layer can step whenever (r − 1)B ≤ 127. **4 layers to n = 768, 6 at n = 1024** (of 24).
  The old `table_fold` fails oss beyond n = 512 (its block partners leave the window).
- bf16: 1 + ⌈log₄(n/9)⌉ for both, 5 at n = 1024 — equal to this doc's form.

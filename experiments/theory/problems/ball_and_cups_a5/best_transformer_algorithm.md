# ball_and_cups (A5) — best known transformer algorithm

**layers(n) = 1 for n ≤ 11; 2 + max(0, ⌈log₄(n/55)⌉) for larger n.** Audit round
wf_d60028f2 overturned the strict-log₂ claim: gathering k instruction symbols into a
composite and folding via one |A5|^k-entry P4 table is legal within the alphabet cap
for moderate k, giving an 11-symbol first layer and radix-4-ish tree above it.
Non-solvability still forbids the head-radix (linear-sum) trick — this is table
folding, capped by width, not by heads. n = 1024 → 5 layers.

Prior version and full audit trail: git history + workflow journals wf_46c71f48, wf_9af6b9bf, wf_d60028f2.

**Certified per preset (2026-09-22, `ideas/block_scan`, verified at every checkpoint to n = 1024).**
A block-aligned radix-5 scan: layer 1 folds B₁ = min(H + 1, 27) instructions into one A5 element
(B₁ = 17 qwen, 27 oss, 9 bf16 — the fold is capped by heads or by the 32-bit MLP, whichever binds),
every later layer multiplies the block size by 5 (own ∘ 4 block totals = 5.93 + 4·6 ≤ 32 bits, so
radix 5 is the MLP's maximum). layers(n) = 1 + ⌈log₅(n/B₁)⌉ wherever the model can run a step on
every layer — qwen's linear layers can (the delta-rule memory keyed by (block, parent block) returns
the last position of a preceding block), so qwen gets 4 layers at n = 1024. An oss window layer
cannot reach 4·B_L > 127 keys back, so only its full layers step: layers(n) = 1 + 2⌈log₅(n/27)⌉, 6
at n = 1024. bf16: 1 + ⌈log₅(n/9)⌉, 4 at n = 1024 — one below this doc's ⌈log₄⌉ form.

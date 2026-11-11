# prontoqa chain — best known transformer algorithm

**layers(h) = ceil(log3(2h+3)) + 2 ≈ 0.63·log₂ h + 3 (ideal bf16 model)** — the 3-way
bidirectional interval merge, with the query slot joining the merge as one more interval copy so
the answer falls out of the last round (`ideas/walk_back`). h=16 → 6 layers, h=1024 → 9.
`ideas/contain_hit` is the same merge with a separate containment test, ceil(log3(h+1)) + 3.
Both are verified to h = 512 under the robustness pass.

Per model preset, where a merge round needs a full-attention layer (arbitrary distance, Theta(n)
distinct keys) and only the parse, the (e,T) broadcast and the final one-bit aggregation fit on the
cheap layer kinds — see each idea.md:

| preset | contain_hit | walk_back | best at h = 512 |
|---|---|---|---|
| bf16 (all full) | K + 3 | K' + 2 | 9 |
| oss (window/full, 24 layers) | 2K + 4 | 2K' + 2 | 16 |
| qwen (lin,lin,lin,full, 40 layers) | 4K + 5 | 4K' + 4 | 29 |

K = ceil(log3(h+1)), K' = ceil(log3(2h+3)) ∈ {K, K+1}; whichever idea is smaller wins at a given h.

Prior version and full audit trail: git history + workflow journals wf_46c71f48, wf_9af6b9bf, wf_d60028f2.

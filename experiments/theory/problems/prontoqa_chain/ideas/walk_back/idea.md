# walk_back — the query slot is one more interval copy, walking back from T

Same 3-way interval merge as `contain_hit`, but nothing is tested after the last round: the query
slot joins the merge as an extra copy in block K+2, so it reads block K+1 like any other copy and
the answer falls out of the last round's MLP. That removes the aggregation layer (1 / 2 / 4 layers)
at the price of a slightly larger radius requirement.

The query slot is initialised to the zero-width interval [T, T] (L = F = T, from the query token's
parsed subj = e, obj = T). In round k it adopts the L of the block-(K+1) copy whose F equals its L,
i.e. it steps back over that copy's whole interval, 2*r_{k-1} + 1 = 3^(k-1) edges. After K rounds it
has walked back sum_{k<=K} 3^(k-1) = (3^K - 1)/2 edges, clamping at its chain's first node (no rule
ends there, and the anchored fallback head carries the boundary node exactly as in the merge). T is
always a chain endpoint, so the forward half of its update never matches and F stays T.

The entity's chain is base fact + h rules = h + 1 edges from e to c_h, and the other chain's first
node is never e, so `ans = (L == subj)` at the query slot is the answer as soon as
(3^K - 1)/2 >= h + 1, i.e. **K = ceil(log3(2h + 3))** — at most one round more than
ceil(log3(h+1)), and equal to it for roughly half of all h.

Layers, by kind (parse and mirror exactly as in contain_hit; no broadcast, no exists):

| preset | schedule | layers(h) | h = 512 |
|---|---|---|---|
| bf16 | parse, mirror, K rounds | **K + 2** | 9 |
| oss (window/full) | window parse, full mirror, K rounds on the odd layers | **2K + 2** | 16 |
| qwen (lin,lin,lin,full) | 2 linear parse, pad, full mirror, K rounds one per 4 layers | **4K + 4** | 32 |

Cheaper than contain_hit whenever 3^K already covers 2h+3 (oss: always at least as cheap; qwen:
cheaper at h = 3, 12, 32, 96, 256, ...). Verified to n = 512 under all three presets.

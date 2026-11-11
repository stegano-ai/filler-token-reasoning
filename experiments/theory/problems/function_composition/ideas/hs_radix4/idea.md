# hs_radix4 — radix-4 Hillis-Steele scan over table-functions (bf16, oss)

An element is one table: a function on D = 4 points, 8 bits, so the 32-bit MLP cap lets one
layer compose **own + 3 partners** (8 + 3*8 = 32). Functions are stored in one field `g`
(alphabet 256) with the identity at index 255, so `sym()`'s -1 (no match) composes as the
identity — a missing partner at the left edge is free.

Layout: position 0 is a spare, the n*D+1 tokens sit at 1..n*D+1. Element 0 is x0 at position 1
(as the constant function x |-> x0), element k is table k, ending at position 4k+1. Composing the
prefix therefore already applies x0, and the answer for hop k is `FUNCS[g[4k+1]][0]` — no final
layer to apply the input.

| layer | kind | what |
|---|---|---|
| 0 | any | **gather**: 7 heads at pos offsets -1..-7 read the 8 tokens of this table and the previous one; MLP writes `g` = f_k o f_{k-1} (W = 2 elements). 2 + 2 + 7*2 = 18 bits. |
| j | full / window | **level L**: 3 heads at pos offsets -m * W0 * 4^L (m = 1,2,3); MLP `acc = own o p1 o p2 o p3`, W *= 4. 8 + 3*8 = 32 bits. |
| j | window, too far | **no-op** |

A level needs exact positional matching, so it wants a full layer; a window layer can run level L
only while the farthest partner still fits, `3 * W0 * 4^L < window` (L <= 2 for window = 128).
Under oss (window, full) levels 1 and 2 ride the window layers, everything above level 3 wastes
one window layer per level. Linear layers cannot do this (a pos offset beyond the conv kernel
blows the 128-key delta-rule budget), so qwen uses `block_radix4` instead.

Missing partners are safe without an anchor tier: position 0 keeps g = identity for ever, and it
is always inside the key set (full) or the window (a fallback can only happen at a position closer
to the start than the partner distance, which is < window), so a "background average" is either
mixed (sym = -1 = identity) or uniformly 255 = identity.

Closed form, W0 = 2, R = 4, L(n) = ceil(log4((n+1)/2)):
- bf16: `1 + L(n)` — 6 layers at n = 1024.
- oss: `1 + L(n)` for L <= 3, else `2*L(n) - 2` — 8 layers at n = 1024 (model depth 24).

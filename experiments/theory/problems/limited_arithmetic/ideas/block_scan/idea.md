# block_scan (qwen preset; also valid on all-softmax presets)

Same monoid as radix_scan: every token is an affine map x -> ax+b over Z_8 with a in {0,1,3,5,7}
(40 elements, stored as `a` = index into {0,1,3,5,7} and `b`), element k at pos 2k, the start value is
the constant map (0, x0), and the prefix product at element k has a = 0 and b = intermediate k.

Under qwen only every 4th layer is softmax, and a linear (Gated DeltaNet) layer cannot match on a
position offset beyond the conv kernel once n_pos > lin_keys = 128 (that is exactly where radix_scan
dies at n = 64). So the scan is **block-aligned** rather than sliding: after layer j every even
position holds the product over [B_j*floor(e/B_j), e]. A linear head then needs a key field of
alphabet 6 only, and the delta-rule "most recent matching key" lands on precisely the partner block.

- Layer 0, linear (9 heads): `ce` = e mod 5 at even positions / 5 at odd, `co` = e mod 5 at odd / 5 at
  even. Head 0 copies the own op from pos-1 (conv kernel, free). For t = 1..4 two heads read the
  operand (`ce` == query `ce` - t) and the op (`co` == query `ce` - t) of element e-t. **The gate is
  free**: for t > e mod 5 the query value is negative, no key carries it, the head returns sym = -1 and
  the MLP folds an identity, so the product never leaves the level-1 block. 4 + 2 + 4*6 = 30 MLP bits.
- Layer j >= 1, linear (4 heads): `c_j` = floor(e/B_j) mod 5 at even positions, 5 at odd; head t
  matches `c_j` == query `c_j` - t, i.e. the last position of block idx-t, auto-gated the same way.
  2.32 + 3 + 4*6 = 29.3 MLP bits. Odd positions are never keys, so their garbage never propagates.
- Any **softmax** layer instead needs a unique key and an explicit gate: `e_j` = floor(e/B_j) at the
  last even position of each complete block and a sentinel elsewhere, head t matches
  `e_j` == query `i_j` - t, and the MLP also reads `g_j` = floor(e/B_j) mod 5, keeping partner t only
  for t <= g_j (2.32 + 3 + 2.32 + 4*6 = 31.6 bits). A softmax layer 0 uses pos-2t / pos-2t-1 instead,
  with radix 4 so that 7 heads and 26 MLP bits fit the ideal model too. Every softmax head carries
  Clause(-M, {one: 1}) so the background sits at -M and the match at +M: a 2M gap keeps the copy exact
  at n_pos = 3 under the c_beta = 1.5 robustness variant.

layers(n) = ceil(log_5(n+1)) under qwen — every layer kind does a radix-5 fold, no no-ops, which is the
depth the ideal bf16 model reaches, and strictly below radix_scan's 1 + ceil(log_5((n+1)/4)) at
n = 4, 24, 512. Under an all-softmax preset layer 0 drops to radix 4 and the claim becomes that same
1 + ceil(log_5((n+1)/4)). Verified n = 1..1024 under qwen (depth 5 at n = 1024, model depth 40),
robustness pass clean. Not listed under oss: window layers cannot reach a block partner at
2tB positions, and the delta-rule gating trick has no softmax equivalent.

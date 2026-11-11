# wavefront_lean — the same wavefront with the additive overhead cut

Same algorithm as `ideas/wavefront` / `ideas/wavefront_sched` (one derivation step per step layer,
holders linearise the fired premise numbers), with two changes that only touch the constant:

1. **emit is one layer instead of two** (all presets), and
2. **init is 8 layers instead of 16 under qwen** — two full layers carry all six category loads,
   everything else rides the linear layers that the pattern hands out for free.

| preset | old | new | init | steps | emit |
| --- | --- | --- | --- | --- | --- |
| bf16 | t + 9 | **t + 8** | 6 | t + 1 | 1 |
| qwen | t + 24 | **t + 12** | 8 | t + 1 | 1 full layer, 2 pads |
| oss | 2t + 16 | **2t + 14** | 10 | 2(t + 1) | 1 full layer, 1 pad |

Certified at hops 16, 24, 32, 48, 64.

## Emit in one layer: weighted offsets instead of unique pairs

An output position q belongs to the holder with the smallest `end > q`. `wavefront` found it with
heads of two offset clauses, relying on consecutive `end`s differing by ≥ 2 for uniqueness — 2
offsets per head, and 3 heads per layer is all the 9-bit premise number leaves under `mlp_bits`, so
the `dmax + 1 = 10` offsets at t = 64 needed two layers.

Here one head carries **three** offsets with weights 4, 3, 2. Several holders may now match at once;
the softmax keeps the highest weight, and the weight order is the offset order, so the head returns
exactly the nearest holder (the ratio between adjacent weights is `n_pos^-c_beta`, far below the
p = 8 rounding step). The smallest weight has to stay at 2: a weight-1 clause sits only
`n_pos^(c_beta-1)` above the unweighted background of every other key, which is 1/68 at
`c_beta = 1.5` — above the rounding step, and the robustness pass catches it.

Three heads therefore reach offset 9. Offset `dmax + 1` — reached only at t = 64 — is covered by a
fourth head that carries **no premise number at all**: a position that far from its holder's `end`
holds the leading bit of a minimal-width binary number, which is 1 by definition. That head's only
value field is `end` (a positional code, exempt), so it is free, and the layer stays at 31 of 32
bits. The MLP resolves across heads by scanning them from the largest offset group down.

## The lean init (linear presets only; oss/bf16 keep the 6-layer init, which packs better)

Six category loads (leg A, leg B and the head of each of a slot's two rules) are the whole reason
init needs full attention: each is keyed by premise number, ~4t+8 distinct keys, far past
`lin_keys = 128`. At 9 bits per loaded category only three fit under `mlp_bits = 32`, so exactly
**two** full layers are needed — and under qwen those are layer 3 and layer 7, which pins init at 8.
Everything else was moved onto the linear layers in between:

| idx | kind | work |
| --- | --- | --- |
| 0 | linear | parse A — conv-range shifts −1…−3 behind each newline |
| 1 | linear | parse B — offsets −4…−6 as shifts of L0's staged raw copies |
| 2 | linear | relay the queried entity **and** the queried category: key = `role`, ≤ 8 distinct tuples |
| 3 | full | load head0, head1, legA0; rank the queried entity's fact pair (3 clauses) |
| 4 | linear | the two base facts' categories → the pre-fired slots (key `(role, E2, est1)`, ≤ 32 tuples) |
| 5 | linear | the two base facts' premise numbers → holders 0/1 and their output offsets |
| 6 | linear | each sub-slot compares its head category with the relayed query — MLP only, no head |
| 7 | full | load legA1, legB0, legB1 |

Why the cheap layers are legal:

* **relays** (`{role: QE}`, `{role: QX}`) key on one field with 8 values, and the query row is the
  most recent one before every scratch row, so "most recent match" = "the match".
* **base-fact reads** key on `(role, E2, est1)` — 8 entities × 2 ranks — and the match is unique.
* the **X test** needs no attention at all once the query category has been relayed: it is a
  comparison of two fields the slot already holds. It cannot be done on a premise row (the query
  line sits at the end of the input, so no premise row can attend to it) and it cannot share layer
  7 (31 of 32 MLP bits are spent there), which is what the otherwise-idle layer 6 is for.

The field budget is what forces the rest. With 16 fields and four categories per premise row alive
at once, a load's destination must be a field whose input-row value is already dead: layer 3 writes
`B2`/`A2`/`A1` (rule head consumed in that same layer; `A1` held only the queried entity), layer 7
writes `B1`/`tok`/`H2` (leg A and leg B sources, consumed there). The rule's leg B is never copied
anywhere — it rides `B1`, the raw −3 staging copy, which at a rule anchor already *is* leg B.

## What did not work

* **Four offsets per emit head** (weights 4/3/2/1), which would have covered every t ≤ 254 with
  three heads: the weight-1 clause loses to the background at `c_beta = 1.5`.
* **Loads on a linear layer.** Premise numbers exceed `lin_keys` for t > 30 and sharding them into
  ≤ 128-key fields needs one head per shard per load — over both the head and the bit budget.
* **One full init layer.** 6 × 9 = 54 bits of category > `mlp_bits`; two layers is the floor, and
  under qwen two full layers are 8 layers of depth.
* **A derivation step on an oss window layer.** A step matches a slot's leg against *any*
  established slot in the grid, which spans many copies of the premise array — nowhere near the
  128-key window. The 2t stands.
* **A shorter oss init.** 4 full layers of bit-packed work; moving the relay+rank to layer 1 (so the
  fulls are 1/3/5/7 and init ends at 8, saving 2 more) leaves no layer where both head categories
  are loaded *and* the X test can still read the fact categories it would clobber.

## Caveats

Unchanged from `wavefront`: the scratch grid is `k ≈ 50` copies of the premise array, enough for the
copy descents the generator produces up to t = 64 and not claimed beyond it; `B2` doubles as the
holder's premise number, so `4t + 8 ≤ 414`. Sub-slots whose premise index happens to be a base fact
load the null row's staged token (an entity name, 5…12) instead of the old circuit's 0 — never a
category, so they still never fire.

# wavefront_sched — the wavefront scheduled onto the model presets

Same algorithm as `ideas/wavefront` (one derivation step per step layer, holders linearise the
fired premise numbers, two emit layers write the trace), but every layer is placed on a layer of
the kind it actually needs, and the derivation step has a second implementation that runs on a
**linear** (Gated DeltaNet) layer. Certified at hops 16, 24, 32, 48, 64.

| preset | layers(t), 16 ≤ t ≤ 64 | why |
| --- | --- | --- |
| bf16 | t + 9 | every layer is full, nothing is padded — identical to `wavefront` |
| qwen | t + 24 | one step per layer, linear **and** full; 16 layers of padded init |
| oss  | 2t + 16 | a step needs a global match, so only the odd (full) layers can carry one |

## Layer kinds

* **L1 parse A** — three conv-range shifts (`pos−1…−3`) of `tok`, the pre-coded line marker and the
  premise numeral. Offsets inside the conv kernel and no content key, so this runs on **any** kind.
* **L2 parse B** — the rule's leg A and the premise numeral sit at `pos−4…−6`, outside the kernel.
  L1 therefore also stores raw copies of `tok@−3` (in `B1`) and of the numeral (in `H2`), and L2
  reaches −4/−5/−6 as −1/−2/−3 of those. **Any** kind again; under qwen these two layers are the
  first two linear layers, so the init costs no extra full layer.
* **L3–L6 bind / leg B / heads / seed** — each matches a premise by its number (`I1`) or a fact by
  (entity, rank) across the whole input, over hundreds of distinct keys and with 3–4 clauses:
  **full** only. Under qwen they land on layer 3, 7, 11, 15, so init ends at depth 16.
* **step layers** — see below. **linear or full** under qwen, **full** under oss/bf16.
* **emit** — output position q matches its holder by `end` with two clauses plus a null fallback:
  **full**.

## The step layer on a linear layer (`E1`/`E2`, the establishment memory)

A linear head is an associative memory: one clause, the query reads the value written by the most
recent matching key, at most 128 distinct key tuples. The classic leg test
(`role SLOT ∧ est ∧ head-category = leg`) is three clauses and keys on the category field, which
carries ~2t+6 distinct values — over the budget at t = 64 and not expressible in one clause.

Two fields that die after L6 (the input-row category copy and the fact entity) are recycled as
**`E1`, `E2`: the category a slot has established, 0 while it has not**. `E1` is rewritten every
step layer as `est1 ? B2 : 0` (and `E2` as `est2 ? A2 : 0`), which costs nothing extra because the
MLP already reads `B2`/`A2`. Gating the write rather than the read is what makes this work:

* the key field then holds only `{0} ∪ established categories`, ≤ t+3 ≈ 35 distinct tuples at
  t = 64 — comfortably inside `lin_keys = 128`, while an ungated `E1` would hold ~103;
* every position the head can match is established, so "most recent match" (linear) and "any match"
  (full) agree, and the same head reads `est1`/`est2` as an exact 1-bit answer under both kinds.
  A category can head a rule in two chains, and only one of them fires for the queried entity —
  with an ungated key the most recent match would be the wrong chain's rule.

A leg test is one head per establishment field, so four legs cost 8 heads (qwen has 16). Under a
full layer the same head gets the usual `MISS` null fallback and returns `est = 0` on no match;
under a linear layer there is no fallback and no match gives `sym = −1`. `est1 = live(A1) ∧
live(tok)` is monotone, so it is recomputed rather than read.

## Collecting the fired rule when the collecting layer is not the firing layer

The holder for step s must learn which slot fired. Under bf16/oss every layer is full and collects
its predecessor's firing, exactly as in `wavefront`. Under qwen a *full* step layer cannot also
collect — the collect head would add 9 more MLP bits on top of the 19.4 the `B2`/`A2` reads already
cost. So a full step layer only fires, and the next linear layer collects two firings at once.
The two tags must live in **two different fields**: a site holds two rules and both can fire in
consecutive layers, so a single tag field would lose the older one.

* a **linear** step layer tags its firing in `fresh` (1 = first sub-slot, 2 = second) and clears `I1`;
* a **full** step layer tags its firing in `I1`, leaving `fresh` untouched;
* a linear layer reads `{fresh = 1}`, `{fresh = 2}`, `{I1 = 1}`, `{I1 = 2}` — four one-clause heads
  whose value is `pos`, which is a positional code and free of the MLP table cap. From the position
  the MLP recovers the premise number as `2·((p − S0) mod n_sites) + sub`. Under full attention the
  copies of one firing sit at different positions, so there the head returns `I1` (identical across
  copies) instead.

The two holders filled in one layer are consecutive (s and s+1), so `end` still chains: holder s
reads `end` at `pos−1`, holder s+1 reads `pos−2` and adds both premise lengths. Both offsets are
inside the conv kernel.

Every step layer stays inside the caps: linear reads `est1, est2, B2, A2` (19.4 bits) plus 8 one-bit
head outputs = 27.4; full is the same without the collect heads; the classic (bf16/oss) step reads
`est1, est2, B2` and collects `I1, fresh` = 29.7.

## Closed forms

`emit = ⌈⌈(⌈log2(4t+8)⌉+1)/2⌉/3⌉ = 2` for 16 ≤ t ≤ 64.

* **bf16** — 6 init + (t+1) steps + emit = t + 9.
* **oss** (window, full) — init lands on layers 0, 1, 3, 5, 7, 9; each of the t+1 full step layers
  and each emit layer costs a window pad: 10 + 2(t+1) + 2·emit = 2t + 16.
* **qwen** (linear, linear, linear, full) — init ends at depth 16. Step layers are consecutive from
  there; per four of them (L,L,L,F) four firings are collected, so m = t+1 step layers suffice for
  t ≡ 0 (mod 4), and the two emit layers cost three pads each: 16 + (t+1) + 7 = t + 24.

## Caveats

The scratch layout is the one `wavefront` documents: k ≈ 50 copies of the premise array, enough for
the copy descents the generator produces up to t = 64 and not claimed beyond it. `B2` doubles as
the holder's premise number, so it needs `4t+8 ≤ 414`, i.e. t ≤ 101.

# wavefront — layers(t) = t + 9 (one derivation step per layer, 6 init + 2 emit + 1 collect tail)

Certified against the v2 interpreter (`functions/rs_transformer.py`) at hops 16, 24, 32, 48, 64.
The t + 4 figure that stood under the older, looser interpreter is not reachable under v2: the
init and emit stages are bit-bound, not layer-bound, and they cost 8 constant layers instead of 3.

## Layer by layer

**L1 parse A.** Every premise line ends in a newline; the three tokens behind it are gathered by
fixed offsets (`Q("pos", -d)`). Offset −2 is `:` for a fact line and `>` for a rule line and occurs
nowhere else, so the line's kind is read off it; offset −1 is the head/fact category, offset −3 the
rule's leg B or the fact's entity. Writes `role`, `A2` (head category), `B2` (leg B), `S2` (entity).

**L2 parse B.** Offsets −4/−5/−6 give the rule's leg A and the premise numeral (facts are 6 tokens,
rules 7, so the numeral sits at −5 and −6 respectively). Writes `H2` (leg A), `I1` (premise number).

**L3 bind.** The query's entity is relayed from the `? E : X` line to the scratch rows (`end`); each
fact anchor learns whether an earlier fact of the same entity exists (a self-excluding negative-weight
clause, `est1` = its rank in the pair); every slot loads both of its rules' leg A.

**L4 leg B + base facts.** Both leg Bs load; the two positions FH0/FH1 pull the queried entity's two
facts by (entity, rank) and become pre-fired slots (`est1 = 1`, category in `B2`).

**L5 heads.** Both rules' head categories load.

**L6 seed.** Holders 0 and 1 take the two base-fact premise numbers and set their output-row `end`
offsets; each slot compares both of its head categories against the queried category X and records
which (if either) matches, in `isX`.

**L7 … L(t+7) wavefront, one layer per derivation step.** The scratch slot grid holds `k` copies of
the premise array (2 rules per position). Each unfired sub-slot polls its two body categories against
the established keys — a single head per leg, matching `{role SLOT, est1, B2}` or `{role SLOT, est2, A2}`
with a weight-2 fallback on the null position, so the head returns the null sentinel `isX = 3` exactly
when the category is not established. A sub-slot fires when both legs are established; the layer's own
index is *not* used as the step tag. Instead a 3-valued `fresh` field (pure write, no read) marks the
sub-slots that fired in the previous layer, and holder s = layer − 6 collects them: every copy that
fires does so in the same layer with an identical payload, so the head's average *is* the value and
`sym` is exact. `end_s = end_{s−1} + nbits(idx_s) + 1` rides the same layer from the preceding holder.

**L(t+8), L(t+9) emit.** Output position q window-matches its holder on `end`: one head per pair of
offsets `{end == q + e, end == q + e + 1}`, which is unique because consecutive `end`s differ by at
least 2. `end` and `pos` have alphabet > 1024 and are exempt from the MLP table cap, so the match costs
only the 9 bits of the premise number. The delimiter, the addressed MSB-first binary digit, and the
final answer bit (a head asking whether any fired slot carries `isX`, i.e. whether X was derived for
the queried entity) are written into `isX`, which the readout slices.

## What changed for v2

* **6 clauses per head → ≤ 4.** Facts are relayed into two pre-fired scratch slots, so a leg test is
  one clause for each of the two sub-slots plus the null fallback.
* **The 2-tie "average minus own bits" pair head is gone.** `sym` is −1 on any average now, so the
  later fact of a pair identifies itself with a negative-weight self-exclusion clause instead
  (`Clause(-M, {"pos": Q("pos")})`), and every head in the circuit is either a unique match or a match
  over keys with identical payloads.
* **MLPs read 76–84 bits → ≤ 32.** Every layer's MLP reads `pos` (alphabet > 1024, exempt) and derives
  each row's role from build-time position ranges instead of reading wide fields. Every written field
  is a pure function of the heads and the few read fields, which is why the parse and load stages are
  spread over six layers: 6 category fields at 9 bits each cannot be loaded in fewer.
* **Step tags by `fresh`, not by counting or by the layer constant**, which removes the step field from
  the wavefront MLP's reads.
* 16 fields exactly, at most 7 heads per layer, ≤ 32 MLP bits per layer, all checked by the interpreter.

## Scratch caveat — certified range only

The documented O(t²) step-replicated grid does not fit the 4·n_input scratch budget. What fits is
`k` copies of the premise array; a derivation step can stay in its copy only while the premise order
ascends, so the chain moves to the next copy at every descent. Measured over generator instances the
deepest copy reached is 10 at t = 16, 18 at t = 32, 36 at t = 64, against k ≈ 50 copies available at
every t in that range — so the construction is correct on the range it is certified for and has roughly
a 1.4× margin at t = 64. It is *not* claimed beyond t = 64: descents grow like t/2 while k stays ≈ 50,
so the copies run out somewhere above t ≈ 90.

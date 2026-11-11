# What the two files per problem mean

Every problem folder holds exactly two documents:

**bound.md — the theoretical serial-computation bound.** How the number of serial
computation steps must scale with instance length n, for ANY fixed-width,
limited-precision computer of the transformer kind (formally: the RS-transformer
model — n input + n filler positions, one causal forward pass, constant hidden width,
constant-margin features, O(log n)-bit attention-score path). Written as a lower
bound Ω(·) with its proof route, plus any accounting caveats. This is the floor no
algorithm can beat.

**best_transformer_algorithm.md — the best known construction.** An explicit
transformer circuit with a closed-form layer count layers(n), all constants numeric.
This is the ceiling we currently know how to achieve, certified by construction
(and adversarially audited by agent swarms wf_46c71f48 / wf_9af6b9bf).

Formally, define RS-depth D(n) = the minimum L such that one fixed transformer with
L layers (within the resource limits above) is correct on all instance lengths
m ≤ n. Then bound.md states D(n) ≥ f(n) and best_transformer_algorithm.md states
D(n) ≤ g(n).

**What counts as a better result:**
- a better ALGORITHM = a construction with layers(n) = o(g(n)) — or the same order
  with a smaller leading constant — meeting every resource limit (the usual failure
  mode is smuggling in poly(n) width, log n-bit value paths, recency tie-breaks, or
  forward-looking attention; all are excluded);
- a better BOUND = a proof that D(n) ≥ f'(n) with f' growing faster than f;
- the problem is CLOSED when f and g meet up to constants (status per task:
  mod_sum, arithmetic, a5, gl32, chain closed up to log log factors; horn wide open
  between Ω(1) and O(t)).

Expressibility ≠ learnability: these are floors a model COULD reach; what SGD finds
under a given curriculum is the project's empirical question, not this folder's.

## Executable model: `experiments/theory/functions/rs_transformer.py`

The RS-transformer is defined operationally by `functions/rs_transformer.py`, an idealised causal
transformer run numerically. Every closed form in this folder is certified only relative to it.

- Positions hold ≤ F = 16 symbol fields, each with a declared alphabet (poly(n) allowed); a field is
  stored as its binary code. n input positions plus ≤ 4·n scratch positions.
- A head scores key j for query i by Σ over ≤ C = 4 clauses of weight × AND(key field == query field
  + offset | constant), |score| ≤ M = 4, negative weights allowed (self-exclusion). Logits are
  c_beta·ln(n_pos)·score with c_beta = 2, causal softmax (self included), output = weighted average of
  the value fields' bit codes rounded to p = 8 bits (constant-precision regime; p = 8 is the bf16
  instance, the value is a model parameter, not a measurement).
- The MLP reads only what a constant-margin readout can see: `sym(field)` is the exact copied symbol
  where every bit is exactly 0/1 (a unique match) and −1 wherever the head averaged; `frac(field)` is
  allowed only on 1-bit fields and is the averaged flag, so a count over c ≤ 2^(p−1) tied keys is
  exact and that is the only aggregate available. No exact counting beyond that, no bit-slot packing.
- MLP table cap: Σ log2(alphabet) over the fields it reads with alphabet ≤ 1024, plus the bits of
  head-output fields with alphabet ≤ 1024, ≤ mlp_bits = 32 per layer. Larger alphabets are positional
  codes (positions, pointers) and exempt; they must not be used to pack constants.
- Depth = number of layers; embedding and readout are free.

Constructions live at `problems/<task>/ideas/<idea>/circuit.py` (`build(n) -> Program`,
`claimed_layers(n)`) with `problems/<task>/problem.py` (`instances(n, k, seed)`), a short `idea.md`,
and the checker's `results.json`. `python main.py -c configs/<task>.yaml` runs `check_circuit`: depth
at length checkpoints up to 1024 (stopping past 200 layers) against the claim, correctness spot checks,
and an optional time-budgeted robustness pass (`final_seconds`: fresh instances cycling p, p−1 and
c_beta 1.5). `configs/frontier.yaml` plots, per task and length, the shallowest passing idea
(`frontier.png`, `frontier.json`). Nothing in prose is load-bearing: a construction exists when its
results.json passes.

### Model presets (2026-09-22)

`functions/rs_transformer.py` carries three presets, and every closed form is now certified per
preset (`results_<arch>.json`, one frontier panel each):

- `bf16`: the ideal model above, all layers full attention, H = 8.
- `qwen` (Qwen3.6-35B-A3B, HF config): 40 layers, 16 heads, pattern linear/linear/linear/full. A
  linear-attention (Gated DeltaNet) head is an associative memory: one clause, the query reads the value
  written by the most recent matching key, at most 128 distinct key tuples per head, no averaging; pos
  offsets within the conv kernel (−3..0) are free shifts. Only every 4th layer can do softmax matching.
- `oss` (gpt-oss-20b, HF config): 24 layers, 64 heads, pattern window/full; a window layer sees only
  the last 128 keys.

A layer's kind is assigned by its index from the preset's pattern, so a circuit is scheduled onto the
model rather than the model bent to the circuit. Depth beyond the model's layer count is still recorded;
the dotted line on each panel is the model's depth.

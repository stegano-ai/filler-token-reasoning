# horn — best known transformer algorithm

**layers(t) = t + 4** — audit round wf_d60028f2 superseded sequential firing (2t+4)
with a wavefront variant: init 2, one layer per derivation step, collect 1, emit 1.
t=16 → 20 layers. NOTE this flips the expressibility reading: horn-16 fits BOTH
studied models' depth (20 ≤ 24 ≤ 40), so the universal latent horn-16 failure is a
LEARNING failure, not an expressibility ceiling. Δ-fan-out and encoding caveats per
the audit record (wf_d60028f2 journal).

**CERTIFIED (2026-09-17): layers(t) = t + 9**, `ideas/wavefront/`, passing the v2 interpreter
at t = 16, 24, 32, 48, 64 (t=16 -> 25 layers). The t + 4 figure above was never executable: under
v2's resource limits the init and emit stages are bit-bound (<= 32 MLP bits per layer, <= 4 clauses
per head, 16 fields), which costs 8 constant layers rather than 3. The loop constant 1 survives.
The expressibility reading is unchanged in kind: horn-16 at 25 layers still fits both studied
models' depth (25 <= 40) though no longer the 24-layer one.

Prior version and full audit trail: git history + workflow journals wf_46c71f48, wf_9af6b9bf, wf_d60028f2.

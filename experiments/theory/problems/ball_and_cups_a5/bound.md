# ball_and_cups (A5) — theoretical serial-computation bound

**Scaling: Θ(log n), base-2 forced** for fixed width/precision.

- Lower bound: **Ω(log n / log log n)** — Z_2 ≤ A5, parity transfers. Context:
  the A5 word problem is NC1-complete (Barrington), so no sub-log shortcut is
  expected in any reasonable accounting.
- Non-solvability blocks the head-radix speedup: composing r group elements in one
  MLP layer needs a |G|^r lookup (no linear pre-activation trick exists for
  non-solvable groups) — the radix caps at 2–3 REGARDLESS of head count. This is
  Barrington's theorem materializing as an architecture constraint.

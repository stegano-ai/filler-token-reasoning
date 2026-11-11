# limited_arithmetic (modular) — theoretical serial-computation bound

**Scaling: Θ(log n)** — same class as rolling_mod_sum.

- Lower bound: **Ω(log n / log log n)** — Z_2 embeds in Aff(Z_m) via translations,
  so the parity bound transfers (see rolling_mod_sum/bound.md for the bridge).
- The affine monoid Aff(Z_m) is finite and SOLVABLE, which is what keeps the task
  in the head-radix class (unit products fold via discrete logs, the additive part
  via one more sum-and-threshold) — unlike the non-solvable group tasks.

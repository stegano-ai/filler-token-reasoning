# rolling_mod_sum — theoretical serial-computation bound

**Scaling: Θ(log n) serial steps** for a fixed-width, limited-precision computer.

- Lower bound: **Ω(log n / log log n)** — parity is not in AC0 (Furst–Saxe–Sipser /
  Hastad); a constant-precision transformer layer is simulable by O(1) depth of
  poly-size AC0-with-approximate-majority (promise-gate semantics, Ajtai/Viola
  substitution), so L layers give depth O(L) and the AC0 size-depth tradeoff forces
  L = Ω(log n / log log n). Valid average-case (Hastad correlation bounds).
- The aggregation-unit depth of 2 (count, then read parity of the count) is
  unrealizable at fixed width and precision: the count needs either an n-wide
  threshold bank at one position or log n-bit exact aggregation, neither available.
- Upper bound met by the algorithm file: **Θ(log n), essentially tight.**

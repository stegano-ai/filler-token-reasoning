# prontoqa chain — theoretical serial-computation bound

**Scaling: Θ̃(log h), average-case over the task's random premise order.**

- Lower bound: **Ω(log h / log log h)** via a 2-track parity reduction (each bit
  crosses or preserves two parallel chains; endpoint track = parity). Holds
  average-case (parity is average-case hard for AC0), so it genuinely matches the
  w.h.p. upper bound.
- Regime restriction: log h = O(d) — the Θ(h) distinct category symbols must be
  margin-separable in the model's hidden dimension.
- Worst-case premise order at linear positions: only O(√h) is verified;
  O(log h) worst-case at linear positions is open.

# horn — theoretical serial-computation bound

**Scaling: open between Ω(1) and O(t).** The widest gap in the suite.

- Only Ω(1) is PROVED for transformer layers: the parity-embedding gadget violates
  horn's own promise constraints (fresh heads per rule, pair-privacy), and no
  parity-sensitive subfamily inside the promise has been constructed.
- The measured 2.9·t^0.49 span is in JOIN units (one relational aggregation = 1 op)
  with poly-width relational state; porting it to transformer layers awaits the
  layer cost of one relational JOIN with Θ(log t)-bit keys. The ≤256-hop data
  equally fits 0.67·log² t.
- Residual hardness attribution: planted 3-coloring of the auxiliary graph at the
  Kesten–Stigum threshold + Θ(t) cutwidth.

**Sub-linear swarm round (2026-09-16, killed before synthesis; journal wf_dcf34ec6):** 8 designers.
Two sub-linear claims — pair-relation pointer jumping at Θ(√t·log t) w.h.p. (executable check and
scaling audit did not refute; model-compliance lens refuted it) and a shortcut on the shipped
distribution (unaudited); two polylog lower-bound proposals (unaudited); pointer jumping, chain-
membership recovery, meet-in-the-middle and frontier doubling returned negative results. Nothing
here changes the headline until re-audited against the executable checker.

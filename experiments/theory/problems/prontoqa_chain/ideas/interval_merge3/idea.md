# interval_merge3 — 3-way bidirectional interval merge, layers(h) = ceil(log3(h+1)) + 3

Format: answer-only bit (see problem.py). Scratch: sentinel + (K+1) blocks of one copy per premise + query slot.

1. parse (as serial). 2. mirror: every block copy fetches (subj, obj) of its premise by number; L = subj,
   F = obj (radius 0); the query slot fetches (e, T).
3..K+2. round k (K = ceil(log3(h+1))): copy in block b >= k, reading block b-1 (causality: a round's
   readers sit after everything it reads), radius r -> 3r+1: L <- L of the rule whose F == my L, F <- F of
   the rule whose L == my F. No match (sentinel wins) means the chain boundary is within reach, and the
   anchored rule with obj == my L (subj == my F) then carries exactly the boundary node, so every copy
   holds node-interval [max(x-1-r,0), min(x+r,N)] with r = (3^k-1)/2 — tie-free, all keys unique.
K+3. test: ans = exists rule copy in block K with L == e and F == T (needs r >= h/2, i.e. 3^K >= h+1).

Uses: fresh globally unique names, out/in-degree <= 1, chain 0 = base fact + h rules with e appearing
only there. Deviation from best_transformer_algorithm.md (2 ceil(log3(2h-1)) + 1): one layer per round
instead of two (per-round scratch blocks replace the copy step) and a containment test instead of
walking back from T, so the count is ceil(log3(h+1)) + 3 (h=16: 6 vs 9; h=1024: 10 vs 15).

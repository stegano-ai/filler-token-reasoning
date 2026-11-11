# serial — one hop per layer, layers(h) = h + 2

Format: answer-only bit (see problem.py). Tokens are words; no scratch positions.

1. parse: every object token (tok[-1] == "a") reads its line via fixed offsets (-1,-3,-4,-5,-6):
   kind (rule / base fact / query from tok[-4]), subject, premise number. The query's object
   token (the last input token but one) holds e = entity and T = target.
2..h+2. the query token walks the chain: cur <- obj of the unique premise with subj == cur
   (base fact first, keyed on e; out-degree <= 1 makes every match unique). h+1 hops reach c_h;
   the last MLP sets ans = (cur == T).

Uses: subjects are unique across premises (base fact + rules), the query is the last line so it
sees every premise causally, chain 0 has exactly h rules.

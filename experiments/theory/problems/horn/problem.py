from lib.side_tasks.prontoqa import _CHAIN_ENTITIES, _COMPACT_CATEGORIES, _horn_data

N_CHAINS, RHO, SURFACE = 4, 2, "compact"
MIN_HOPS = 16
MAX_HOPS = (len(_COMPACT_CATEGORIES) - N_CHAINS) * RHO // N_CHAINS - 1  # compact pool cap: 197

# Tokeniser: one token per line element. Vocabulary: 0 newline, 1 '.', 2 ':', 3 '>', 4 '?',
# 5..12 the 8 entity names, 13..413 the 401 compact categories, 414+i the premise numeral i.
# A rule line "i. A B > C" is 7 tokens, a fact line "i. E : A" is 6, the query "? E : X" is 4
# and the blank line before it is a lone newline.
MARKERS = {"\n": 0, ".": 1, ":": 2, ">": 3, "?": 4}
ENT0, CAT0, NUM0 = 5, 13, 414
VOCAB = {**MARKERS, **{e: ENT0 + i for i, e in enumerate(_CHAIN_ENTITIES)},
         **{c: CAT0 + i for i, c in enumerate(_COMPACT_CATEGORIES)}}


def tokenise(text: str) -> list[int]:
    toks = []
    for line in text.split("\n"):
        for w in line.split():
            if w[:-1].isdigit() and w.endswith("."):
                toks += [NUM0 + int(w[:-1]), MARKERS["."]]
            else:
                toks.append(VOCAB[w])
        toks.append(MARKERS["\n"])
    return toks[:-1]


def instances(n: int, k: int, seed: int) -> list[dict]:
    assert MIN_HOPS <= n <= MAX_HOPS, f"horn generator supports {MIN_HOPS} <= hops <= {MAX_HOPS}, got {n}"
    return [{"input": tokenise(d["input"]), "reasoning": d["reasoning"], "output": d["output"]}
            for d in _horn_data(n, N_CHAINS, RHO, k, seed, SURFACE)]

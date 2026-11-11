import re

from lib.side_tasks.prontoqa import _CHAIN_ENTITIES, _chain_data, _chain_name_pool

# Reasoning format: answer-only bit. bound.md / best_transformer_algorithm.md (and the
# historical best_known.md / review.md) bound "decide f^h(start) = target"; the review notes the
# gold trace is chain 0's walk independent of the answer, so the log-depth closed form is about
# the answer, not the proof_trace. Instances therefore carry reasoning = output = [answer bit].
N_CHAINS = 2

# Tokeniser: one token per word or punctuation mark (regex [A-Za-z]+|\d+|[.:], newlines dropped).
# Fixed vocabulary WORDS = punctuation + keywords + entities + the generator's name pool; a
# premise number k tokenises to V + k.
WORDS = [".", ":", "Every", "is", "a", "True", "or", "false"] + _CHAIN_ENTITIES + _chain_name_pool()
V = len(WORDS)
DOT, COLON, EVERY, IS, A = 0, 1, 2, 3, 4
_ID = {w: i for i, w in enumerate(WORDS)}


def tokenize(text: str) -> list[int]:
    return [V + int(w) if w.isdigit() else _ID[w] for w in re.findall(r"[A-Za-z]+|\d+|[.:]", text)]


def n_premises(n: int) -> int:
    return N_CHAINS * n + 1


def instances(n: int, k: int, seed: int) -> list[dict]:
    return [{"input": tokenize(d["input"]), "reasoning": d["output"], "output": d["output"]}
            for d in _chain_data(n, N_CHAINS, k, seed)]

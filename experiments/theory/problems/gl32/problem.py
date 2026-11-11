import random

from lib.side_tasks.gl32 import _lamp_patterns

REASONING_IN_OUTPUT = True


def instances(n: int, k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(k):
        ins = [rng.randint(0, 1) for _ in range(n)]
        r = _lamp_patterns(ins)
        out.append({"input": ins, "reasoning": r, "output": list(r) if REASONING_IN_OUTPUT else r[-1:]})
    return out

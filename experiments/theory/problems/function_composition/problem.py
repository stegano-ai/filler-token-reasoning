import random

D = 4  # domain size; n = number of tables (hops)


def instances(n: int, k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(k):
        x0 = rng.randrange(D)
        tables = [[rng.randrange(D) for _ in range(D)] for _ in range(n)]
        x, reasoning = x0, []
        for t in tables:
            x = t[x]
            reasoning.append(x)
        out.append({"input": [x0] + [v for t in tables for v in t],
                    "reasoning": reasoning, "output": reasoning})
    return out

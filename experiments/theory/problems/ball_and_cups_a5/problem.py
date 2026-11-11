import random

from lib.side_tasks.ball_and_cups import _ball_positions

# n = seq_length; instructions are drawn uniformly from {0: rotate, 1: double-swap} as in make_task.draw


def instances(n: int, k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    out = []
    for _ in range(k):
        instructions = [rng.randint(0, 1) for _ in range(n)]
        reasoning = _ball_positions(instructions)
        out.append({"input": instructions, "reasoning": reasoning, "output": list(reasoning)})
    return out

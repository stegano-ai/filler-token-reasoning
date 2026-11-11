import random

from .base import SideTaskDefinition, sample_split

# 0 = ROTATE, 1 = DOUBLE-SWAP. These generate A5 (non-solvable), so tracking the
# ball is NC1-complete (Barrington); do not change the instruction set casually —
# e.g. the mirror swap (0<->4, 1<->3) instead yields solvable D5.
_DOUBLE_SWAP = {0: 1, 1: 0, 2: 2, 3: 4, 4: 3}


def _ball_positions(instructions: list[int]) -> list[int]:
    pos = 0
    return [(pos := (pos + 1) % 5 if i == 0 else _DOUBLE_SWAP[pos]) for i in instructions]


def make_task(
            n_samples: int,
            seq_length: int,
            reasoning_in_output: bool = True,
            seed: int = 42,
        ) -> SideTaskDefinition:
    rng = random.Random(seed)

    def draw() -> dict:
        instructions = [rng.randint(0, 1) for _ in range(seq_length)]
        reasoning = _ball_positions(instructions)
        output = list(reasoning) if reasoning_in_output else [reasoning[-1]]
        return {"input": instructions, "reasoning": reasoning, "output": output}

    # Non-commutative (that is the point of A5), so the exact move sequence is the problem.
    train_data, test_data = sample_split(
        n_samples, draw, lambda d: tuple(d["input"]), "ball_and_cups")
    return SideTaskDefinition(
        name="ball_and_cups",
        train_data=train_data,
        test_data=test_data,
        explanation=(
            f"Five cups sit at positions 0-4, with a ball under the cup at position 0. "
            f"You will receive a sequence of {seq_length} instructions. "
            f"Rotate: every cup moves one position up, wrapping from 4 back to 0, "
            f"so the ball's position p becomes (p+1) mod 5. "
            f"Double-Swap: the cups at positions 0 and 1 swap and the cups at "
            f"positions 3 and 4 swap; the cup at position 2 stays put. "
            f"Track the ball's position after each instruction as your reasoning, "
            f"then output those positions. "
            f"For example, given [Double-Swap, Rotate, Rotate, Double-Swap]: "
            f"reasoning is [1, 2, 3, 4], output is [1, 2, 3, 4]."
        ),
        alphabet=list(range(5)),
    )

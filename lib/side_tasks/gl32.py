import random

from .base import SideTaskDefinition, sample_split

# 0 = SHIFT (cyclic bit rotation), 1 = COUPLE (CNOT: if lamp A is on, toggle lamp B).
# Both are invertible linear maps on the 3 lamp bits and together generate
# GL(3,2) = PSL(2,7) (order 168, non-solvable, contains no A5), so tracking the pattern
# is NC1-complete like ball_and_cups but through a different group — do not change the
# instruction set casually. Linear maps fix the zero vector, so pattern 0 is unreachable
# from the lamp-A start and the alphabet's 0 never appears in gold data.


def _shift(v: int) -> int:
    return ((v << 1) & 0b111) | (v >> 2)


def _lamp_patterns(instructions: list[int]) -> list[int]:
    v = 1
    return [(v := _shift(v) if i == 0 else (v ^ 0b010 if v & 1 else v))
            for i in instructions]


def make_task(
            n_samples: int,
            seq_length: int,
            reasoning_in_output: bool = True,
            seed: int = 42,
        ) -> SideTaskDefinition:
    rng = random.Random(seed)

    def draw() -> dict:
        instructions = [rng.randint(0, 1) for _ in range(seq_length)]
        reasoning = _lamp_patterns(instructions)
        output = list(reasoning) if reasoning_in_output else [reasoning[-1]]
        return {"input": instructions, "reasoning": reasoning, "output": output}

    train_data, test_data = sample_split(
        n_samples, draw, lambda d: tuple(d["input"]), "gl32")
    return SideTaskDefinition(
        name="gl32",
        train_data=train_data,
        test_data=test_data,
        explanation=(
            f"Three lamps A, B, C sit in a ring; lamp A starts on, lamps B and C start off. "
            f"You will receive a sequence of {seq_length} instructions. "
            f"Shift: every lamp passes its on/off state to the next lamp — "
            f"A's state moves to B, B's to C, C's to A. "
            f"Couple: if lamp A is on, lamp B toggles; if lamp A is off, nothing happens. "
            f"Write a pattern as the number A + 2*B + 4*C, counting an on lamp as 1. "
            f"Track the pattern after each instruction as your reasoning, "
            f"then output those patterns. "
            f"For example, given [Couple, Shift, Couple, Shift]: "
            f"reasoning is [3, 6, 6, 5], output is [3, 6, 6, 5]."
        ),
        alphabet=list(range(8)),
    )

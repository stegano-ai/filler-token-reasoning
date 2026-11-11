import random

from .base import SideTaskDefinition, sample_split


def _running_sum_mod(numbers: list[int], modulus: int) -> list[int]:
    running = 0
    return [(running := (running + n) % modulus) for n in numbers]


def make_task(
            n_samples: int,
            seq_length: int,
            operand_range: int,
            modulus: int,
            reasoning_in_output: bool = True,
            seed: int = 42,
            min_operand: int = 1,
        ) -> SideTaskDefinition:
    rng = random.Random(seed)

    def draw() -> dict:
        nums = [rng.randint(min_operand, operand_range) for _ in range(seq_length)]
        reasoning = _running_sum_mod(nums, modulus)
        output = list(reasoning) if reasoning_in_output else [reasoning[-1]]
        return {"input": nums, "reasoning": reasoning, "output": output}

    # Keyed by the exact operand sequence: permutations count as distinct problems. This
    # keeps binary/low-entropy operand settings splittable (under multiset equivalence a
    # 0/1 task has only seq_length + 1 problems), at the price that a test sequence may be
    # a permutation of a train sequence.
    train_data, test_data = sample_split(
        n_samples, draw, lambda d: tuple(d["input"]), "rolling_mod_sum")
    example = [3 % modulus, 4 % modulus, 6 % modulus]
    if reasoning_in_output:
        output_clause, example_output = "then output those running-sum values", example
    else:
        output_clause, example_output = "then output the final running-sum value", example[-1:]
    return SideTaskDefinition(
        name="rolling_mod_sum",
        train_data=train_data,
        test_data=test_data,
        explanation=(
            f"You will receive a sequence of {seq_length} numbers. "
            f"Compute the running sum modulo {modulus} of these numbers as your reasoning, "
            f"{output_clause}. "
            f"For example, given [3, 1, 2] with mod {modulus}: "
            f"reasoning is {example}, output is {example_output}."
        ),
        alphabet=list(range(modulus)),
    )

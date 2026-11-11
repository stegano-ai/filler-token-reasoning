import math
import random

from .base import SideTaskDefinition, sample_split


def _binary_cot(intermediates: list[int]) -> list[int]:
    flat = []
    for i, v in enumerate(intermediates):
        if i > 0:
            flat.append(2)
        flat.extend(int(b) for b in bin(v)[2:])
    return flat


# The operand pool depends on the modulus alone, never on the state: +/- take any nonzero
# shift, * takes the units of Z_m other than 1, so every step map is an invertible affine map
# of Z_m and no operator letter says anything about the state it acts on. Units rather than a
# prime modulus because max_value is already configured at 3, 8 and 20 across the repo, and
# m - 1 is a unit for every m >= 3, so the * pool is never empty.
def modular_operands(max_value: int) -> dict[str, list[int]]:
    shifts = list(range(1, max_value))
    units = [b for b in range(2, max_value) if math.gcd(b, max_value) == 1]
    return {"+": shifts, "-": shifts, "*": units}


def modular_step(op: str, b: int, current: int, max_value: int) -> int:
    if op == "+":
        return (current + b) % max_value
    if op == "-":
        return (current - b) % max_value
    return (current * b) % max_value


def _generate_modular(rng, num_steps, max_value, binary_cot, operands):
    current = rng.randrange(max_value)
    parts = [str(current)]
    intermediates = []
    for _ in range(num_steps):
        op = rng.choice(["+", "-", "*"])
        b = rng.choice(operands[op])
        current = modular_step(op, b, current, max_value)
        parts.append(f"{op} {b}")
        intermediates.append(current)
    reasoning = _binary_cot(intermediates) if binary_cot else intermediates
    return {"input": " ".join(parts), "reasoning": reasoning, "output": [intermediates[-1]]}


def make_task(n_samples: int, num_steps: int, max_value: int, seed: int = 42,
              binary_cot: bool = False) -> SideTaskDefinition:
    assert max_value >= 3, "modular limited_arithmetic needs max_value >= 3 for a nonempty * pool"
    rng = random.Random(seed)
    operands = modular_operands(max_value)
    train_data, test_data = sample_split(
        n_samples, lambda: _generate_modular(rng, num_steps, max_value, binary_cot, operands),
        lambda d: d["input"], "limited_arithmetic")
    base = (
        f"You will receive a starting value followed by {num_steps} operations using +, -, *. "
        f"Evaluate left-to-right with no operator precedence, reducing modulo {max_value} after "
        f"every operation, so every intermediate result is an integer in [0, {max_value - 1}]. "
    )
    if binary_cot:
        explanation = base + (
            f"Use the {num_steps} intermediate results as your reasoning, written in binary "
            f"(most significant bit first) with a single 2 separating consecutive values. "
            f"For example, with modulus 7, given '5 + 4 * 3 - 2' (3 steps): step 1: 5+4=9=2 mod 7, "
            f"step 2: 2*3=6=6 mod 7, step 3: 6-2=4=4 mod 7. "
            f"Intermediates [2, 6, 4] become the reasoning [1,0, 2, 1,1,0, 2, 1,0,0]. Output is [4]."
        )
    else:
        explanation = base + (
            f"Use the {num_steps} intermediate results as your reasoning, and output the final result. "
            f"For example, with modulus 7, given '5 + 4 * 3 - 2' (3 steps): step 1: 5+4=9=2 mod 7, "
            f"step 2: 2*3=6=6 mod 7, step 3: 6-2=4=4 mod 7. Reasoning is [2, 6, 4], output is [4]."
        )
    return SideTaskDefinition(
        name="limited_arithmetic",
        train_data=train_data,
        test_data=test_data,
        explanation=explanation,
        alphabet=[0, 1, 2] if binary_cot else list(range(max_value)),
    )

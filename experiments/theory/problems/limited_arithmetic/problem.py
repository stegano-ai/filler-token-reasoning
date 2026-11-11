import random

from lib.side_tasks.limited_arithmetic import _generate_modular, modular_operands

MAX_VALUE = 8       # modulus; n = num_steps
BINARY_COT = False
TOKENS = {"+": 8, "-": 9, "*": 10}


def instances(n: int, k: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    ops = modular_operands(MAX_VALUE)
    out = []
    for _ in range(k):
        d = _generate_modular(rng, n, MAX_VALUE, BINARY_COT, ops)
        d["input"] = [TOKENS[w] if w in TOKENS else int(w) for w in d["input"].split()]
        out.append(d)
    return out

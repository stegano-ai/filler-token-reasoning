import random

from .base import SideTaskDefinition, sample_split

# Pointer chasing / k-hop in lookup-table form — the task family with the strongest
# known transformer depth lower bounds: Omega(log k) depth for width-bounded softmax
# transformers CONDITIONAL on the MPC 1-vs-2-cycle conjecture (Sanford-Hsu-Telgarsky
# 2024 Cor 4.3); unconditionally, L >= k or huge width against linear-attention/SSM
# layers (SHT Cor 5.2; Ye et al. 2026 for full/linear hybrids), and an unconditional
# decoder-only depth hierarchy at small L with enormous domains (Chen-Peng-Wu 2024).
# Tables are ALWAYS random permutations, not configurable: compositions of
# independent uniform non-bijective tables coalesce — two walks merge per table with
# prob ~1/m, so p(final value independent of x0) = 1-(1-1/m)^(k+1), and at m=8 a
# reader of only the last ~17 tables that ignores x0 scores >= 0.9 at EVERY k,
# silently voiding the task (this bit the 2026-09 fc runs; see the lab notebook's
# 2026-09-22 entry). Permutation tables have no coalescence, chance floor exactly
# 1/m at every k, every table relevant, span ceil(log2(k+1)) tight — honestly
# NC1-hard via S_m.
# CAVEAT: the depth-lower-bound pedigree also requires domain_size to GROW with n_hops — a block of tables over a fixed
# domain composes into one m-entry table (m*log2(m) bits), so constant-domain configs
# are compressible and fall back to the a5/gl32 slot (conditional NC1-hardness via the
# transformation monoid T_m, non-solvable for m >= 5). Configure domain_size ~ n_hops
# (or larger) when the unconditional pedigree is the point. The tables must stay
# independent uniform permutations — a single iterated table can admit shortcuts.
# Class 2 (state tracking), but each per-step input is a whole table,
# which is what blocks local summarization in the growing-domain regime. Chance
# floor: the final value of a random-table walk is ~uniform, so ~1/domain_size.


def _trace(x: int, tables: list[list[int]]) -> list[int]:
    return [(x := t[x]) for t in tables]


def make_task(
            n_samples: int,
            n_hops: int,
            domain_size: int,
            reasoning_in_output: bool = True,
            seed: int = 42,
        ) -> SideTaskDefinition:
    rng = random.Random(seed)

    def draw() -> dict:
        x0 = rng.randrange(domain_size)
        tables = [rng.sample(range(domain_size), domain_size) for _ in range(n_hops)]
        reasoning = _trace(x0, tables)
        output = list(reasoning) if reasoning_in_output else [reasoning[-1]]
        return {"input": [x0] + [v for t in tables for v in t],
                "reasoning": reasoning, "output": output}

    train_data, test_data = sample_split(
        n_samples, draw, lambda d: tuple(d["input"]), "function_composition")
    return SideTaskDefinition(
        name="function_composition",
        train_data=train_data,
        test_data=test_data,
        explanation=(
            f"You will receive a list of integers: a start value, then {n_hops} lookup "
            f"tables of {domain_size} entries each, written one after another. "
            f"Each table maps a value v (from 0 to {domain_size - 1}) to its entry at "
            f"index v. Starting from the start value, apply table 1, then table 2, "
            f"and so on, feeding each result into the next table. "
            f"Track the value after each table as your reasoning, "
            f"then output those values. "
            f"For example, with 2 tables of 4 entries each, the input "
            f"[1, 3, 0, 2, 1, 1, 3, 0, 2] means start value 1, table 1 = [3, 0, 2, 1], "
            f"table 2 = [1, 3, 0, 2]: table 1 maps 1 to 0, then table 2 maps 0 to 1, "
            f"so reasoning is [0, 1], output is [0, 1]."
        ),
        alphabet=list(range(domain_size)),
    )

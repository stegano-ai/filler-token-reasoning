import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pytest

from lib.side_tasks import get_side_task
from lib.side_tasks.ball_and_cups import _ball_positions
from lib.side_tasks.function_composition import _trace
from lib.side_tasks.gl32 import _lamp_patterns
from lib.side_tasks.limited_arithmetic import modular_operands, modular_step
from lib.side_tasks.prontoqa import _COMPACT_CATEGORIES, _encode_indices, _split_premises
from lib.side_tasks.rolling_mod_sum import _running_sum_mod


def _decode_trace(trace: list[int]) -> list[int]:
    """Inverse of _encode_indices: split on the delimiter 2, read each group as binary."""
    groups, current = [], []
    for v in trace:
        if v == 2:
            groups.append(current)
            current = []
        else:
            current.append(v)
    groups.append(current)
    return [int("".join(str(b) for b in g), 2) for g in groups]


def _singular(category: str) -> str:
    category = category.lower()
    return category[:-2] if category.endswith("uses") else category


def _rule(sentence: str) -> tuple[str, bool, str] | None:
    """Antecedent, negation and consequent of "Every X is (not) Y." / "Xes are (not) Y."."""
    m = re.fullmatch(r"(?:Every|Each) (\w+) is (?:an? )?(not )?(\w+)\.", sentence) or re.fullmatch(
        r"(\w+) are (?:an? )?(not )?(\w+)\.", sentence)
    return (_singular(m.group(1)), m.group(2) is not None, _singular(m.group(3))) if m else None


def _proof_conclusion(lines: list[str], indices: list[int]) -> tuple[str, bool] | None:
    """Walk the cited premises as a proof chain starting from the base fact, returning the
    (category, negated) it reaches, or None if a rule does not apply to the category reached
    so far — i.e. the trace is not a connected proof."""
    base, *rules = indices
    m = re.fullmatch(r"\w+ is (?:an? )?(\w+)\.", lines[base])
    if m is None:
        return None
    current, negated = _singular(m.group(1)), False
    for i in rules:
        parsed = _rule(lines[i])
        if parsed is None or parsed[0] != current:
            return None
        _, negated, current = parsed
    return current, negated


def _unnumbered(premises: str) -> list[str]:
    return [line.split(". ", 1)[1] for line in premises.splitlines()]


# ── running sum mod computation ──


class TestRunnningSumMod:
    def test_basic_mod3(self):
        assert _running_sum_mod([3, 1, 2], 3) == [0, 1, 0]

    def test_single_element(self):
        assert _running_sum_mod([5], 3) == [2]

    def test_all_zeros(self):
        assert _running_sum_mod([0, 0, 0], 3) == [0, 0, 0]

    def test_mod2(self):
        assert _running_sum_mod([1, 1, 1, 1], 2) == [1, 0, 1, 0]

    def test_accumulates(self):
        assert _running_sum_mod([1, 2, 3, 4], 5) == [1, 3, 1, 0]


# ── rolling mod sum side task ──


class TestRollingModSum:
    def test_loads(self):
        task = get_side_task("rolling_mod_sum", n_samples=100, seq_length=9, operand_range=10, modulus=3)
        assert len(task.train_data) == 80
        assert len(task.test_data) == 20

    def test_reasoning_is_running_sum_mod(self):
        task = get_side_task("rolling_mod_sum", n_samples=50, seq_length=5, operand_range=10, modulus=3)
        for item in task.train_data:
            assert item["reasoning"] == _running_sum_mod(item["input"], 3)

    def test_output_is_reasoning(self):
        task = get_side_task("rolling_mod_sum", n_samples=50, seq_length=5, operand_range=10, modulus=3)
        for item in task.train_data:
            assert item["output"] == item["reasoning"]

    def test_values_in_range(self):
        task = get_side_task("rolling_mod_sum", n_samples=50, seq_length=5, operand_range=10, modulus=3)
        for item in task.train_data:
            assert all(0 <= v <= 2 for v in item["reasoning"])
            assert all(0 <= v <= 2 for v in item["output"])

    def test_lengths_match(self):
        task = get_side_task("rolling_mod_sum", n_samples=50, seq_length=9, operand_range=10, modulus=3)
        for item in task.train_data:
            assert len(item["input"]) == len(item["reasoning"]) == len(item["output"]) == 9

    def test_different_modulus(self):
        task = get_side_task("rolling_mod_sum", n_samples=50, seq_length=5, operand_range=10, modulus=5)
        for item in task.train_data:
            assert all(0 <= v <= 4 for v in item["reasoning"])


# ── ball and cups side task ──


class TestBallAndCups:
    def test_loads(self):
        task = get_side_task("ball_and_cups", n_samples=100, seq_length=9)
        assert len(task.train_data) == 80
        assert len(task.test_data) == 20

    def test_example_from_explanation(self):
        assert _ball_positions([1, 0, 0, 1]) == [1, 2, 3, 4]

    def test_rotate_cycles(self):
        assert _ball_positions([0, 0, 0, 0, 0]) == [1, 2, 3, 4, 0]

    def test_double_swap_is_involution(self):
        assert _ball_positions([1, 1]) == [1, 0]

    def test_output_is_reasoning(self):
        task = get_side_task("ball_and_cups", n_samples=50, seq_length=12)
        for item in task.train_data:
            assert item["output"] == item["reasoning"]
            assert item["reasoning"] == _ball_positions(item["input"])

    def test_values_in_range(self):
        task = get_side_task("ball_and_cups", n_samples=50, seq_length=12)
        assert task.alphabet == [0, 1, 2, 3, 4]
        for item in task.train_data:
            assert all(v in (0, 1) for v in item["input"])
            assert all(0 <= v <= 4 for v in item["reasoning"])


# ── gl32 side task ──


class TestGl32:
    def test_loads(self):
        task = get_side_task("gl32", n_samples=100, seq_length=9)
        assert len(task.train_data) == 80
        assert len(task.test_data) == 20

    def test_example_from_explanation(self):
        assert _lamp_patterns([1, 0, 1, 0]) == [3, 6, 6, 5]

    def test_shift_cycles(self):
        assert _lamp_patterns([0, 0, 0]) == [2, 4, 1]

    def test_couple_is_involution(self):
        assert _lamp_patterns([1, 1]) == [3, 1]

    def test_output_is_reasoning(self):
        task = get_side_task("gl32", n_samples=50, seq_length=12)
        for item in task.train_data:
            assert item["output"] == item["reasoning"]
            assert item["reasoning"] == _lamp_patterns(item["input"])

    def test_values_in_range_and_zero_unreachable(self):
        task = get_side_task("gl32", n_samples=50, seq_length=12)
        assert task.alphabet == list(range(8))
        for item in task.train_data:
            assert all(v in (0, 1) for v in item["input"])
            assert all(1 <= v <= 7 for v in item["reasoning"])


# ── function composition side task ──


class TestFunctionComposition:
    def test_loads(self):
        task = get_side_task("function_composition", n_samples=100, n_hops=4, domain_size=8)
        assert len(task.train_data) == 80
        assert len(task.test_data) == 20

    def test_example_from_explanation(self):
        assert _trace(1, [[3, 0, 2, 1], [1, 3, 0, 2]]) == [0, 1]

    def test_trace_matches_recomputation(self):
        n_hops, m = 4, 8
        task = get_side_task("function_composition", n_samples=50, n_hops=n_hops, domain_size=m)
        for item in task.train_data:
            x, *flat = item["input"]
            assert len(flat) == n_hops * m
            expected = []
            for i in range(n_hops):
                x = flat[i * m + x]
                expected.append(x)
            assert item["reasoning"] == expected
            assert item["output"] == item["reasoning"]

    def test_values_in_range(self):
        task = get_side_task("function_composition", n_samples=50, n_hops=3, domain_size=5)
        assert task.alphabet == [0, 1, 2, 3, 4]
        for item in task.train_data:
            assert all(0 <= v <= 4 for v in item["input"])
            assert all(0 <= v <= 4 for v in item["reasoning"])
            assert len(item["reasoning"]) == 3

    def test_answer_only_output_is_final_value(self):
        task = get_side_task("function_composition", n_samples=50, n_hops=3, domain_size=5,
                             reasoning_in_output=False)
        for item in task.train_data:
            assert item["output"] == [item["reasoning"][-1]]

    def test_tables_are_permutations(self):
        m = 6
        task = get_side_task("function_composition", n_samples=30, n_hops=4,
                             domain_size=m)
        for item in task.train_data:
            _, *flat = item["input"]
            for i in range(4):
                assert sorted(flat[i * m:(i + 1) * m]) == list(range(m))

    def test_seed_is_deterministic(self):
        kwargs = dict(n_samples=20, n_hops=3, domain_size=5)
        a = get_side_task("function_composition", seed=7, **kwargs)
        b = get_side_task("function_composition", seed=7, **kwargs)
        c = get_side_task("function_composition", seed=8, **kwargs)
        assert a.train_data == b.train_data
        assert a.train_data != c.train_data


# ── limited arithmetic side task ──


# ── limited arithmetic (modular) ──


def _mod_parse(text: str) -> tuple[int, list[tuple[str, int]]]:
    tokens = text.split()
    return int(tokens[0]), [(tokens[i], int(tokens[i + 1])) for i in range(1, len(tokens), 2)]


def _mod_run(start: int, letters: list[tuple[str, int]], m: int) -> list[int]:
    state, states = start, []
    for op, b in letters:
        state = modular_step(op, b, state, m)
        states.append(state)
    return states


def _mod_alphabet(m: int) -> list[tuple[str, int]]:
    return [(op, b) for op, pool in modular_operands(m).items() for b in pool]


def _mod_task(m: int, steps: int, n_samples: int, **kwargs):
    return get_side_task("limited_arithmetic", n_samples=n_samples,
                         num_steps=steps, max_value=m, **kwargs)


def _mod_items(m: int, steps: int, n_samples: int, **kwargs) -> list[dict]:
    task = _mod_task(m, steps, n_samples, **kwargs)
    return task.train_data + task.test_data


def _held_out_map_accuracy(fit: list[tuple], evaluate: list[tuple]) -> float:
    """Accuracy of the empirical maximum-a-posteriori predictor: fit key -> target counts on
    one half of the instances, predict the modal target on the other half, backing off to the
    global mode for keys never seen. This is the best a bounded-window reader can do."""
    counts = defaultdict(Counter)
    for key, target in fit:
        counts[key][target] += 1
    fallback = Counter(target for _, target in fit).most_common(1)[0][0]
    hits = sum(1 for key, target in evaluate
               if (counts[key].most_common(1)[0][0] if key in counts else fallback) == target)
    return hits / len(evaluate)


def _compose(f: tuple, g: tuple) -> tuple:
    return tuple(f[x] for x in g)


def _inverse(f: tuple) -> tuple:
    inverse = [0] * len(f)
    for x, y in enumerate(f):
        inverse[y] = x
    return tuple(inverse)


def _closure(generators: set) -> set:
    elements, frontier = set(generators), list(generators)
    while frontier:
        f = frontier.pop()
        for g in list(elements):
            for product in (_compose(f, g), _compose(g, f)):
                if product not in elements:
                    elements.add(product)
                    frontier.append(product)
    return elements


class TestLimitedArithmeticModular:
    def test_loads(self):
        task = _mod_task(8, 5, 100)
        assert task.name == "limited_arithmetic"
        assert len(task.train_data) == 80 and len(task.test_data) == 20
        assert task.alphabet == list(range(8))

    def test_shape_matches_the_contract(self):
        for item in _mod_items(8, 6, 50):
            assert isinstance(item["input"], str)
            assert len(item["reasoning"]) == 6
            assert item["output"] == [item["reasoning"][-1]]
            assert all(0 <= v < 8 for v in item["reasoning"])

    def test_trace_is_the_modular_state_sequence(self):
        for item in _mod_items(8, 10, 100):
            start, letters = _mod_parse(item["input"])
            assert _mod_run(start, letters, 8) == item["reasoning"]

    def test_binary_cot_encodes_the_same_states(self):
        task = _mod_task(8, 6, 60, binary_cot=True)
        assert task.alphabet == [0, 1, 2]
        for item in task.train_data:
            start, letters = _mod_parse(item["input"])
            assert _decode_trace(item["reasoning"]) == _mod_run(start, letters, 8)
            assert item["output"] == [_mod_run(start, letters, 8)[-1]]

    def test_seed_is_deterministic(self):
        a, b, c = _mod_task(8, 6, 40, seed=7), _mod_task(8, 6, 40, seed=7), _mod_task(8, 6, 40, seed=8)
        assert a.train_data == b.train_data
        assert a.train_data != c.train_data

    def test_split_is_disjoint(self):
        task = _mod_task(8, 5, 200)
        assert not ({d["input"] for d in task.train_data} & {d["input"] for d in task.test_data})

    def test_small_modulus_rejected(self):
        with pytest.raises(AssertionError, match="max_value >= 3"):
            _mod_task(2, 4, 10)

    # ── acceptance: the trace must not be window-computable ──

    def test_every_step_map_is_a_bijection_of_z_m(self):
        for m in (3, 5, 7, 8, 9, 20):
            for op, b in _mod_alphabet(m):
                assert sorted(modular_step(op, b, x, m) for x in range(m)) == list(range(m))

    def test_step_maps_generate_the_solvable_affine_group_of_z_m(self):
        for m in (5, 8, 9):
            generators = {tuple(modular_step(op, b, x, m) for x in range(m))
                          for op, b in _mod_alphabet(m)}
            group = _closure(generators)
            units = [a for a in range(m) if math.gcd(a, m) == 1]
            assert group == {tuple((a * x + c) % m for x in range(m))
                             for a in units for c in range(m)}
            assert len(group) == m * len(units)
            assert tuple(range(m)) in group
            assert all(_inverse(f) in group for f in group)
            derived = _closure({_compose(_compose(f, g), _compose(_inverse(f), _inverse(g)))
                                for f in group for g in group})
            assert all(_compose(f, g) == _compose(g, f) for f in derived for g in derived)

    def test_operator_menu_does_not_depend_on_the_state(self):
        m, steps = 5, 12
        conditional, pooled = defaultdict(Counter), Counter()
        for item in _mod_items(m, steps, 800):
            start, letters = _mod_parse(item["input"])
            states = [start] + item["reasoning"]
            for i, letter in enumerate(letters):
                conditional[states[i]][letter] += 1
                pooled[letter] += 1
        total = sum(pooled.values())
        for state in range(m):
            counts = conditional[state]
            n = sum(counts.values())
            assert set(counts) == set(pooled), state
            assert 0.5 * sum(abs(counts[l] / n - pooled[l] / total) for l in pooled) < 0.05

    def test_window_composite_is_a_bijection_so_the_window_posterior_is_uniform(self):
        m, steps = 8, 40
        for item in _mod_items(m, steps, 300):
            _, letters = _mod_parse(item["input"])
            for w in (4, 8, 16):
                for i in (w, steps // 2, steps - 1):
                    window = letters[i - w + 1:i + 1]
                    assert sorted(_mod_run(x, window, m)[-1] for x in range(m)) == list(range(m))

    def test_window_conditional_state_distribution_is_uniform(self):
        m, steps, w = 3, 20, 4
        buckets = defaultdict(Counter)
        for item in _mod_items(m, steps, 2000):
            _, letters = _mod_parse(item["input"])
            for i in range(w, steps):
                buckets[tuple(letters[i - w + 1:i + 1])][item["reasoning"][i]] += 1
        kept = [counts for counts in buckets.values() if sum(counts.values()) >= 30]
        assert len(kept) > 300
        chi_square, deviations = 0.0, []
        for counts in kept:
            n = sum(counts.values())
            chi_square += sum((counts[s] - n / m) ** 2 / (n / m) for s in range(m))
            deviations.append(0.5 * sum(abs(counts[s] / n - 1 / m) for s in range(m)))
        assert chi_square / (len(kept) * (m - 1)) < 1.3
        assert sum(deviations) / len(deviations) < 0.10

    def test_best_bounded_window_predictor_is_at_the_1_over_m_chance_floor(self):
        for m, steps in ((8, 24), (3, 24)):
            items = _mod_items(m, steps, 1200)
            parsed = [(_mod_parse(item["input"])[1], item["reasoning"]) for item in items]
            for w in (4, 8, 16):
                fit, evaluate = [], []
                for index, (letters, states) in enumerate(parsed):
                    target = fit if index % 2 == 0 else evaluate
                    for i in range(w, steps):
                        target.append(((i, tuple(letters[i - w + 1:i + 1])), states[i]))
                accuracy = _held_out_map_accuracy(fit, evaluate)
                assert abs(accuracy - 1 / m) < 0.02, \
                    f"m={m} w={w}: window predictor {accuracy:.4f} vs chance floor 1/m={1 / m:.4f}"

    def test_operator_sequence_alone_is_at_the_1_over_m_chance_floor(self):
        m, steps = 8, 12
        items = _mod_items(m, steps, 1200)
        fit, evaluate = [], []
        for index, item in enumerate(items):
            _, letters = _mod_parse(item["input"])
            assert len({_mod_run(x, letters, m)[-1] for x in range(m)}) == m
            (fit if index % 2 == 0 else evaluate).append((tuple(letters), item["output"][0]))
        accuracy = _held_out_map_accuracy(fit, evaluate)
        assert accuracy < 1 / m + 0.02, \
            f"operator-only predictor {accuracy:.4f} vs chance floor 1/m={1 / m:.4f}"

    def test_final_value_is_uniform_at_the_1_over_m_chance_floor(self):
        for m, steps in ((3, 9), (8, 10), (8, 40)):
            items = _mod_items(m, steps, 3000)
            counts = Counter(item["output"][0] for item in items)
            for v in range(m):
                assert abs(counts[v] / len(items) - 1 / m) < 0.02, \
                    (f"m={m} L={steps} value {v}: {counts[v] / len(items):.4f}, "
                     f"chance floor is 1/m={1 / m:.4f}")

    def test_state_marginals_are_uniform_at_every_position(self):
        m, steps = 8, 20
        items = _mod_items(m, steps, 2000)
        for i in range(steps):
            counts = Counter(item["reasoning"][i] for item in items)
            assert max(abs(counts[v] / len(items) - 1 / m) for v in range(m)) < 0.03, i

    def test_persistence_baseline_is_below_the_1_over_m_chance_floor(self):
        m, steps = 8, 20
        items = _mod_items(m, steps, 2000)
        repeats = sum(1 for item in items for i in range(1, steps)
                      if item["reasoning"][i] == item["reasoning"][i - 1])
        assert repeats / (len(items) * (steps - 1)) < 1 / m

    def test_changing_any_operator_changes_every_later_state(self):
        m, steps = 7, 8
        alphabet = _mod_alphabet(m)
        for item in _mod_items(m, steps, 60)[:20]:
            start, letters = _mod_parse(item["input"])
            states = item["reasoning"]
            for j in range(steps):
                previous = start if j == 0 else states[j - 1]
                alternatives = [letter for letter in alphabet
                                if modular_step(*letter, previous, m) != states[j]]
                assert alternatives
                for letter in alternatives:
                    changed = _mod_run(start, letters[:j] + [letter] + letters[j + 1:], m)
                    assert all(changed[k] != states[k] for k in range(j, steps))


# ── prontoqa premise-index encoding ──


class TestEncodeIndices:
    def test_example_from_docstring(self):
        assert _encode_indices([5, 2]) == [1, 0, 1, 2, 1, 0]

    def test_single_index_has_no_delimiter(self):
        assert _encode_indices([3]) == [1, 1]

    def test_zero_is_a_single_zero_bit(self):
        assert _encode_indices([0, 1]) == [0, 2, 1]

    def test_no_leading_zeros(self):
        assert _encode_indices([16]) == [1, 0, 0, 0, 0]

    def test_roundtrips(self):
        assert _decode_trace(_encode_indices([0, 7, 12, 5])) == [0, 7, 12, 5]


class TestSplitPremises:
    def test_splits_on_sentence_boundary(self):
        assert _split_premises("Every wumpus is a vumpus. Stella is a wumpus.") == [
            "Every wumpus is a vumpus.", "Stella is a wumpus.",
        ]

    def test_ignores_trailing_whitespace(self):
        assert _split_premises("  Alex is a jompus.  ") == ["Alex is a jompus."]


# ── prontoqa side task (answer only) ──


class TestProntoQAAnswerOnly:
    def test_loads(self):
        task = get_side_task("prontoqa", strategy="answer_only", hops=3, n_samples=100)
        assert len(task.train_data) + len(task.test_data) <= 100
        assert min(len(task.train_data), len(task.test_data)) > 0

    def test_output_is_reasoning(self):
        task = get_side_task("prontoqa", strategy="answer_only", hops=3, n_samples=50)
        for item in task.train_data:
            assert item["output"] == item["reasoning"]

    def test_values_in_range(self):
        task = get_side_task("prontoqa", strategy="answer_only", hops=3, n_samples=50)
        assert task.alphabet == [0, 1]
        for item in task.train_data:
            assert len(item["reasoning"]) == 1
            assert item["reasoning"][0] in (0, 1)

    def test_input_is_unnumbered_premises_plus_query(self):
        task = get_side_task("prontoqa", strategy="answer_only", hops=3, n_samples=20)
        for item in task.train_data:
            premises, query = item["input"].split("\n\n")
            assert query.startswith("True or false:")
            assert not premises.startswith("0. ")

    def test_both_answers_occur(self):
        task = get_side_task("prontoqa", strategy="answer_only", hops=3, n_samples=100)
        assert {item["output"][0] for item in task.train_data} == {0, 1}


# ── prontoqa side task (proof trace) ──


class TestProntoQAProofTrace:
    def test_loads(self):
        task = get_side_task("prontoqa", strategy="proof_trace", hops=3, n_samples=100)
        assert len(task.train_data) + len(task.test_data) <= 100
        assert min(len(task.train_data), len(task.test_data)) > 0

    def test_alphabet(self):
        task = get_side_task("prontoqa", strategy="proof_trace", hops=3, n_samples=50)
        assert task.alphabet == [0, 1, 2]
        for item in task.train_data:
            assert all(v in (0, 1, 2) for v in item["reasoning"])

    def test_output_is_answer_bit(self):
        task = get_side_task("prontoqa", strategy="proof_trace", hops=3, n_samples=50)
        for item in task.train_data:
            assert len(item["output"]) == 1
            assert item["output"][0] in (0, 1)

    def test_reasoning_ends_with_delimiter_and_answer(self):
        task = get_side_task("prontoqa", strategy="proof_trace", hops=3, n_samples=50)
        for item in task.train_data:
            assert item["reasoning"][-2:] == [2, item["output"][0]]

    def test_delimiter_count_is_hops_plus_one(self):
        # base fact + hops rules = hops+1 entries (hops delimiters), plus the delimiter
        # preceding the final answer bit
        for hops in (1, 3):
            task = get_side_task("prontoqa", strategy="proof_trace", hops=hops, n_samples=50)
            for item in task.train_data:
                assert item["reasoning"].count(2) == hops + 1

    def test_trace_decodes_to_valid_premise_indices(self):
        hops = 3
        task = get_side_task("prontoqa", strategy="proof_trace", hops=hops, n_samples=50)
        for item in task.train_data:
            premises, _ = item["input"].split("\n\n")
            indices = _decode_trace(item["reasoning"][:-2])
            assert len(indices) == hops + 1
            assert len(set(indices)) == hops + 1  # no premise explored twice
            assert all(0 <= i < len(premises.splitlines()) for i in indices)

    def test_input_premises_are_numbered(self):
        task = get_side_task("prontoqa", strategy="proof_trace", hops=3, n_samples=20)
        for item in task.train_data:
            premises, query = item["input"].split("\n\n")
            lines = premises.splitlines()
            assert query.startswith("True or false:")
            assert all(line.startswith(f"{i}. ") for i, line in enumerate(lines))

    def test_base_fact_then_rules(self):
        task = get_side_task("prontoqa", strategy="proof_trace", hops=3, n_samples=50)
        for item in task.train_data:
            premises, query = item["input"].split("\n\n")
            lines = premises.splitlines()
            entity = query.removeprefix("True or false: ").split()[0]
            base, *rules = _decode_trace(item["reasoning"][:-2])
            # the first entry is the base fact about the entity, the rest are quantified rules
            assert entity in lines[base]
            assert all(entity not in lines[i] for i in rules)

    def test_trace_is_a_connected_proof_reaching_the_answer(self):
        task = get_side_task("prontoqa", strategy="proof_trace", hops=3, n_samples=50)
        for item in task.train_data + task.test_data:
            premises, query = item["input"].split("\n\n")
            lines = _unnumbered(premises)
            reached = _proof_conclusion(lines, _decode_trace(item["reasoning"][:-2]))
            assert reached is not None, f"trace is not a connected proof:\n{item['input']}"
            m = re.fullmatch(r"True or false: \w+ is (?:an? )?(not )?(\w+)\.", query)
            queried = (_singular(m.group(2)), m.group(1) is not None)
            assert item["output"][0] == int(reached == queried), (
                f"reached {reached} but query asks {queried}:\n{item['input']}")

    def test_rule_positions_are_not_predictable(self):
        # the proof path must not sit at fixed relative positions in the premise list: with the
        # upstream postorder ordering it does, which leaks the answer. Guards _ORDERING="random".
        task = get_side_task("prontoqa", strategy="proof_trace", hops=3, n_samples=60)
        items = task.train_data + task.test_data
        patterns = set()
        for item in items:
            premises, _ = item["input"].split("\n\n")
            n = len(premises.splitlines())
            _, *rules = _decode_trace(item["reasoning"][:-2])
            patterns.add(tuple(round(i / (n - 1), 1) for i in rules))
        assert len(patterns) > len(items) // 2

    def test_seed_is_deterministic(self):
        a = get_side_task("prontoqa", strategy="proof_trace", hops=3, n_samples=20, seed=7)
        b = get_side_task("prontoqa", strategy="proof_trace", hops=3, n_samples=20, seed=7)
        c = get_side_task("prontoqa", strategy="proof_trace", hops=3, n_samples=20, seed=8)
        assert a.train_data == b.train_data
        assert a.train_data != c.train_data

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown prontoqa strategy"):
            get_side_task("prontoqa", strategy="nonexistent", hops=3, n_samples=10)


# ── prontoqa side task (chain strategy, direct construction, no vendored generator) ──


class TestProntoQAChain:
    HOPS = 8

    def _items(self, **kw):
        task = get_side_task("prontoqa", strategy="chain", hops=self.HOPS,
                             n_samples=80, **kw)
        return task.train_data + task.test_data

    @staticmethod
    def _parse(item):
        premise_block, query = item["input"].rsplit("\n\n", 1)
        premises = [line.split(". ", 1)[1] for line in premise_block.split("\n")]
        target = re.fullmatch(r"True or false: \w+ is a (\w+)\.", query).group(1)
        return premises, target

    def test_trace_walks_the_entity_chain(self):
        for item in self._items():
            premises, target = self._parse(item)
            *indices, bit = _decode_trace(item["reasoning"])
            current = re.fullmatch(r"\w+ is a (\w+)\.", premises[indices[0]]).group(1)
            for idx in indices[1:]:
                lhs, rhs = re.fullmatch(r"Every (\w+) is a (\w+)\.", premises[idx]).groups()
                assert lhs == current
                current = rhs
            assert bit == (current == target)
            assert item["output"] == [bit]
            assert len(indices) == 1 + self.HOPS

    def test_premise_count_and_alphabet(self):
        task = get_side_task("prontoqa", strategy="chain", hops=self.HOPS, n_samples=20)
        assert task.alphabet == [0, 1, 2]
        for item in task.train_data:
            premises, _ = self._parse(item)
            assert len(premises) == 1 + 2 * self.HOPS

    def test_answers_balanced(self):
        outputs = [item["output"][0] for item in self._items()]
        assert 0.3 <= sum(outputs) / len(outputs) <= 0.7

    def test_no_local_leak(self):
        # the queried category terminates a chain and is mentioned exactly once,
        # whether the answer is 0 or 1 — nothing local to the target separates the cases
        for item in self._items():
            premises, target = self._parse(item)
            text = " ".join(premises)
            assert text.count(f" {target}.") == 1
            assert not any(p.startswith(f"Every {target} ") for p in premises)

    def test_premise_order_is_shuffled(self):
        base_positions = {_decode_trace(item["reasoning"])[0] for item in self._items()}
        assert len(base_positions) > 5

    def test_deterministic_and_seed_sensitive(self):
        a = get_side_task("prontoqa", strategy="chain", hops=4, n_samples=20, seed=7)
        b = get_side_task("prontoqa", strategy="chain", hops=4, n_samples=20, seed=7)
        c = get_side_task("prontoqa", strategy="chain", hops=4, n_samples=20, seed=8)
        assert a.train_data == b.train_data
        assert a.train_data != c.train_data

    def test_scales_to_deep_chains(self):
        task = get_side_task("prontoqa", strategy="chain", hops=64, n_samples=10)
        for item in task.train_data + task.test_data:
            premises, _ = self._parse(item)
            assert len(premises) == 1 + 2 * 64

    def test_more_chains(self):
        task = get_side_task("prontoqa", strategy="chain", hops=4, n_samples=30, n_chains=4)
        for item in task.train_data:
            premises, _ = self._parse(item)
            assert len(premises) == 1 + 4 * 4


# ── prontoqa side task (horn: shared category pool, unbounded free legs, pair privacy) ──
#
# The attacks below see only item["input"] — they parse the numbered premises and the query and
# never touch the generator internals. The bounded-window predecessor design died to the first
# of them (review.md §3), so it is shipped here as a must-FAIL regression test.


# Both horn surfaces, parsed by the same layer: the attacks and invariants below work on the
# recovered structure, so a rendering change must leave every one of them intact.
_COMPACT_RULE_RE = re.compile(r"(\w+) (\w+) > (\w+)")
_COMPACT_FACT_RE = re.compile(r"(\w+) : (\w+)")
_COMPACT_QUERY_RE = re.compile(r"\? (\w+) : (\w+)")
_RULE_RES = (re.compile(r"Everything that is a (\w+) and a (\w+) is a (\w+)\."), _COMPACT_RULE_RE)
_FACT_RES = (re.compile(r"(\w+) is a (\w+)\."), _COMPACT_FACT_RE)
_QUERY_RES = (re.compile(r"True or false: (\w+) is a (\w+)\."), _COMPACT_QUERY_RE)


def _match(regexes, sentence):
    for r in regexes:
        m = r.fullmatch(sentence)
        if m:
            return m.groups()
    return None


def _parse_horn(item):
    premise_block, query = item["input"].rsplit("\n\n", 1)
    premises = [line.split(". ", 1)[1] for line in premise_block.split("\n")]
    entity, target = _match(_QUERY_RES, query)
    return premises, entity, target


def _forward_chain(premises, entity):
    """Fixpoint forward chaining for `entity` from the premises themselves: returns the
    established categories in derivation order and the fired rule indices in firing order
    (unique because exactly one un-fired rule is ever applicable)."""
    established = [f[1] for p in premises if (f := _match(_FACT_RES, p)) and f[0] == entity]
    rules = {i: r for i, p in enumerate(premises) if (r := _match(_RULE_RES, p))}
    fired = []
    progress = True
    while progress:
        progress = False
        for i, (a, b, head) in rules.items():
            if i not in fired and a in established and b in established:
                established.append(head)
                fired.append(i)
                progress = True
    return established, fired


def _relaxed_closure(premises, entity):
    """One-leg-relaxed fixpoint: a rule fires when EITHER body category is established."""
    established = {f[1] for p in premises if (f := _match(_FACT_RES, p)) and f[0] == entity}
    rules = [r for p in premises if (r := _match(_RULE_RES, p))]
    progress = True
    while progress:
        progress = False
        for a, b, head in rules:
            if head not in established and (a in established or b in established):
                established.add(head)
                progress = True
    return established


def _one_component(premises):
    """Union-find over category mentions: categories co-mentioned in a premise are joined."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            x = parent[x]
        return x

    for p in premises:
        cats = list(_match(_RULE_RES, p) or _match(_FACT_RES, p)[1:])
        for c in cats[1:]:
            parent[find(c)] = find(cats[0])
    return len({find(x) for x in parent}) == 1


def _horn_graph(item):
    """Attacker view: rules {premise index: (a, b, head)}, facts {premise index: (entity,
    category)}, the queried entity and the queried category."""
    block, query = item["input"].rsplit("\n\n", 1)
    rules, facts = {}, {}
    for line in block.split("\n"):
        i, sentence = line.split(". ", 1)
        rule = _match(_RULE_RES, sentence)
        if rule:
            rules[int(i)] = rule
        else:
            facts[int(i)] = _match(_FACT_RES, sentence)
    entity, target = _match(_QUERY_RES, query)
    return rules, facts, entity, target


def _horn_items(hops, n_samples, n_chains=4, rho=2, seed=0, surface="prose"):
    task = get_side_task("prontoqa", strategy="horn", hops=hops, n_samples=n_samples,
                         n_chains=n_chains, rho=rho, seed=seed, surface=surface)
    return task.train_data + task.test_data


def _gold(item):
    """(premise index trace, answer bit)."""
    return _decode_trace(item["reasoning"][:-2]), item["reasoning"][-1]


# ---- attack (a): FRONTIER-DOUBLING, ported from tmp/verify_horn2_span.py -------------------
# Evaluate the derivation as a width-w frontier automaton and square its transition table
# ceil(log2 hops) times. On horn2 this recovered the full trace and the answer at every setting;
# horn's Theta(hops) cutwidth must make the transition undefined, so the doubling never
# completes.

def _frontier_domain(rules, w):
    succ = defaultdict(set)
    for a, b, h in rules.values():
        succ[a].add(h)
        succ[b].add(h)

    def grow(seeds, upto):
        out, cur = set(seeds), list(seeds)
        for _ in range(upto):
            cur = [p + (h,) for p in cur for h in succ[p[-1]] if h not in p]
            out |= set(cur)
        return out

    anchored = grow([r for r in rules.values()], max(w - 1, 0))
    paths = grow([(c,) for c in succ], w + 1)
    return {p for p in anchored | paths if 3 <= len(p) <= w + 2}


def _frontier_transition(rules, w):
    by_leg = defaultdict(list)
    for i, (a, b, _) in rules.items():
        by_leg[a].append(i)
        by_leg[b].append(i)

    def f(state):
        front, newest = set(state), state[-1]
        hits = [i for i in by_leg[newest]
                if set(rules[i][:2]) <= front and rules[i][2] not in front]
        if len(hits) != 1:
            return None
        return (state + (rules[hits[0]][2],))[-(w + 2):], hits[0]

    return f


def _frontier_start(rules, facts, entity):
    """Constant span: the first fired rule is the unique one whose body is exactly the queried
    entity's two base categories."""
    base = {c for (e, c) in facts.values() if e == entity}
    hits = [i for i, (a, b, _) in rules.items() if {a, b} == base]
    if len(hits) != 1:
        return None, None
    prefix = sorted(k for k, (e, _) in facts.items() if e == entity) + [hits[0]]
    return rules[hits[0]], prefix


def _frontier_doubling(item, w, hops):
    """-> (trace, answer) or (None, None) if the bounded-frontier automaton gets stuck."""
    rules, facts, entity, target = _horn_graph(item)
    f = _frontier_transition(rules, w)
    domain = _frontier_domain(rules, w)
    state, trace = _frontier_start(rules, facts, entity)
    if state is None or state not in domain:
        return None, None
    step = {s: f(s) for s in domain}
    tables = [{s: (v[0], (v[1],)) for s, v in step.items() if v is not None}]
    for _ in range(max(0, math.ceil(math.log2(max(hops - 1, 2))))):
        prev = tables[-1]
        tables.append({s: (prev[prev[s][0]][0], prev[s][1] + prev[prev[s][0]][1])
                       for s in prev if prev[s][0] in prev})
    for k, bit in enumerate(reversed(bin(hops - 1)[2:])):
        if bit == "1":
            if state not in tables[k]:
                return None, None
            state, fired = tables[k][state]
            trace += list(fired)
    return trace, int(state[-1] == target)


def _run_frontier_doubling(hops, n_samples, windows, seed=0, surface="prose"):
    data = _horn_items(hops, n_samples, seed=seed, surface=surface)
    report = {}
    for w in windows:
        exact = answered = correct = 0
        for item in data:
            trace, answer = _frontier_doubling(item, w, hops)
            gold_trace, gold_answer = _gold(item)
            if trace is None:
                continue
            answered += 1
            exact += trace == gold_trace
            correct += answer == gold_answer
        # an automaton that gets stuck is left guessing the answer
        report[w] = (exact, answered, (correct + 0.5 * (len(data) - answered)) / len(data))
    return report


# ---- attack (b): spectral clustering of the rule co-occurrence graph -----------------------

def _kmeans(X, k, seed=0, iters=30, restarts=10):
    rs = np.random.RandomState(seed)
    best, best_inertia = None, np.inf
    for _ in range(restarts):
        centres = X[rs.choice(len(X), k, replace=False)]
        labels = np.zeros(len(X), int)
        for _ in range(iters):
            labels = np.argmin(((X[:, None] - centres[None]) ** 2).sum(2), 1)
            centres = np.array([X[labels == c].mean(0) if (labels == c).any() else centres[c]
                                for c in range(k)])
        inertia = ((X - centres[labels]) ** 2).sum()
        if inertia < best_inertia:
            best, best_inertia = labels, inertia
    return best


def _spectral_labels(A, k):
    """Normalised Laplacian D^-1/2 A D^-1/2; the trivial (largest) eigenvector is dropped by
    row-normalising the top-k embedding, then k-means."""
    d = A.sum(1)
    L = A / np.sqrt(np.outer(d, d) + 1e-12)
    _, vecs = np.linalg.eigh(L)
    X = vecs[:, -k:]
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    return _kmeans(X, k)


def _spectral_answer_accuracy(hops, n_samples, n_chains=4, seed=0):
    """Cluster rules by shared categories into n_chains groups; predict answer 1 iff the rule
    heading the queried category lands in the cluster that dominates the rules touching the
    queried entity's base categories."""
    correct = 0
    data = _horn_items(hops, n_samples, n_chains=n_chains, seed=seed)
    for item in data:
        rules, facts, entity, target = _horn_graph(item)
        keys = sorted(rules)
        triples = [set(rules[i]) for i in keys]
        A = np.array([[len(x & y) if x is not y else 0 for y in triples] for x in triples],
                     float)
        labels = _spectral_labels(A, n_chains)
        q_rule = next(p for p, i in enumerate(keys) if rules[i][2] == target)
        base = {c for (e, c) in facts.values() if e == entity}
        near = Counter(labels[p] for p, t in enumerate(triples) if t & base)
        pred = int(labels[q_rule] == near.most_common(1)[0][0])
        correct += pred == item["output"][0]
    return correct / len(data)


# ---- attack (c): common-neighbour statistic ------------------------------------------------

def _common_neighbour_features(data):
    stats, labels = [], []
    for item in data:
        rules, facts, entity, target = _horn_graph(item)
        nb = defaultdict(set)
        for triple in rules.values():
            for x, y in combinations(triple, 2):
                nb[x].add(y)
                nb[y].add(x)
        qa, qb, _ = next(r for r in rules.values() if r[2] == target)
        base = [c for (e, c) in facts.values() if e == entity]
        near_q, near_e = nb[qa] | nb[qb], nb[base[0]] | nb[base[1]]
        shared = len(near_q & near_e)
        stats.append([shared, shared / max(len(near_q | near_e), 1)])
        labels.append(item["output"][0])
    return np.array(stats, float), np.array(labels)


def _best_threshold_accuracy(x, y):
    """The best accuracy any threshold on x (either polarity) achieves against y — an
    in-sample optimum, so it is an upper bound on what the statistic carries."""
    best = 0.0
    for thr in np.unique(x):
        for sign in (1, -1):
            best = max(best, (((x - thr) * sign >= 0).astype(int) == y).mean())
    return best


def _common_neighbour_accuracy(hops, n_samples, seed=0):
    X, y = _common_neighbour_features(_horn_items(hops, n_samples, seed=seed))
    return max(_best_threshold_accuracy(X[:, j], y) for j in range(X.shape[1]))


# ---- attack (d): trace-side, per-position prediction of the fired rule ---------------------

def _rule_features(rules, facts, entity):
    """Surface statistics of every rule premise: how often each of its categories is used as a
    body leg or as a head, and whether it touches the queried entity's base facts."""
    body_count, head_count = Counter(), Counter()
    for a, b, h in rules.values():
        body_count[a] += 1
        body_count[b] += 1
        head_count[h] += 1
    base = {c for (e, c) in facts.values() if e == entity}
    keys = sorted(rules)
    features = []
    for i in keys:
        a, b, h = rules[i]
        features.append([body_count[a], body_count[b], body_count[h],
                         head_count[a], head_count[b], head_count[h],
                         body_count[a] + body_count[b], abs(body_count[a] - body_count[b]),
                         (a in base) + (b in base), float(h in base), i / max(keys)])
    return keys, np.array(features, float)


def _position_prediction_accuracy(hops, n_samples, seed=0):
    """Nearest-centroid attack: fit one centroid per trace position on the first half of the
    instances (gold labels handed over), then on the held-out half predict, for each position,
    the rule premise closest to that position's centroid. Returns the per-position top-1
    accuracies and the chance rate."""
    data = _horn_items(hops, n_samples, seed=seed)
    parsed, traces = [], []
    for item in data:
        rules, facts, entity, _ = _horn_graph(item)
        keys, F = _rule_features(rules, facts, entity)
        parsed.append((keys, F, {k: p for p, k in enumerate(keys)}))
        traces.append(_gold(item)[0][2:])
    stacked = np.concatenate([F for _, F, _ in parsed])
    mu, sd = stacked.mean(0), stacked.std(0) + 1e-9
    half = len(data) // 2
    centroids = [np.mean([(parsed[i][1][parsed[i][2][traces[i][p]]] - mu) / sd
                          for i in range(half)], 0) for p in range(hops)]
    accuracy = []
    for p in range(hops):
        hits = 0
        for i in range(half, len(data)):
            keys, F, _ = parsed[i]
            Z = (F - mu) / sd
            hits += keys[int(np.argmin(((Z - centroids[p]) ** 2).sum(1)))] == traces[i][p]
        accuracy.append(hits / (len(data) - half))
    return accuracy, 1 / len(parsed[0][0])


class TestProntoQAHorn:
    HOPS = 16
    CHAINS = 4
    RHO = 2

    def _items(self, hops=None, n_samples=40, **kw):
        return _horn_items(hops or self.HOPS, n_samples, kw.pop("n_chains", self.CHAINS),
                            kw.pop("rho", self.RHO), **kw)

    @staticmethod
    def _closures(item):
        """Every entity's gated closure, in derivation order."""
        premises, _, _ = _parse_horn(item)
        entities = {f[0] for p in premises if (f := _match(_FACT_RES, p))}
        return {e: _forward_chain(premises, e) for e in entities}

    @pytest.mark.parametrize("surface", ["prose", "compact"])
    def test_trace_matches_forward_chaining(self, surface):
        for item in self._items(surface=surface):
            premises, entity, target = _parse_horn(item)
            *indices, bit = _decode_trace(item["reasoning"])
            assert len(indices) == 2 + self.HOPS
            base = {i for i, p in enumerate(premises)
                    if (f := _match(_FACT_RES, p)) and f[0] == entity}
            assert set(indices[:2]) == base
            assert indices[0] < indices[1]  # the two base facts are interchangeable
            established, fired = _forward_chain(premises, entity)
            assert indices[2:] == fired
            assert bit == (established[-1] == target)
            assert item["output"] == [bit]
            # every entity derives exactly its own chain: hops rules, hops+2 categories
            for other, (est, fires) in self._closures(item).items():
                assert len(fires) == self.HOPS
                assert len(est) == self.HOPS + 2 == len(set(est))

    def test_exactly_one_rule_applicable_per_step(self):
        # the derivation order is unique — this is what the pair-privacy promise P1 buys, and
        # it is the reason the gold trace is well defined without a per-entity namespace
        for item in self._items(n_samples=20):
            rules, facts, entity, _ = _horn_graph(item)
            established = {c for (e, c) in facts.values() if e == entity}
            for _ in range(self.HOPS):
                applicable = [i for i, (a, b, h) in rules.items()
                              if {a, b} <= established and h not in established]
                assert len(applicable) == 1
                established.add(rules[applicable[0]][2])

    @pytest.mark.parametrize("surface", ["prose", "compact"])
    def test_pair_privacy_and_shared_categories(self, surface):
        for item in self._items(n_samples=20, surface=surface):
            rules, facts, _, _ = _horn_graph(item)
            closures = {e: set(est) for e, (est, _) in self._closures(item).items()}
            for a, b, _ in rules.values():
                # P1: every body pair lies in exactly one entity's chain, so every rule is
                # some entity's real rule and it fires for exactly that one entity
                assert sum({a, b} <= c for c in closures.values()) == 1
            # ... while single categories are shared by exactly rho entities (the terminals,
            # private by construction, are the only exception)
            terminals = {est[-1] for est, _ in self._closures(item).values()}
            owners = Counter(c for closure in closures.values() for c in closure)
            for c, k in owners.items():
                assert k == (1 if c in terminals else self.RHO)

    def test_free_legs_are_unbounded(self):
        # review.md §6.4.3: horn2 died because the free leg lived in a constant-size window, so
        # the derivation DAG had bounded bandwidth. Here the back-distance must be Theta(hops)
        # and no constant window may cover most steps.
        distances = []
        for item in self._items(n_samples=30, hops=32):
            rules, facts, entity, _ = _horn_graph(item)
            indices, _ = _gold(item)
            established = [facts[i][1] for i in indices[:2]]
            for k in indices[2:]:
                a, b, head = rules[k]
                free = a if b == established[-1] else b
                assert established[-1] in (a, b)  # the pinned leg is the newest category
                distances.append(len(established) - 1 - established.index(free))
                established.append(head)
        d = np.array(distances)
        assert np.median(d) >= 32 / 8, np.median(d)
        assert d.max() >= 32 / 2
        for w in (1, 2, 4, 8):
            assert (d <= w).mean() <= 0.25 + 0.05 * w, (w, (d <= w).mean())

    @pytest.mark.parametrize("surface", ["prose", "compact"])
    def test_no_partition_and_no_relaxation_shortcut(self, surface):
        for item in self._items(n_samples=20, surface=surface):
            premises, entity, _ = _parse_horn(item)
            rules, _, _, _ = _horn_graph(item)
            categories = {c for triple in rules.values() for c in triple}
            closures = self._closures(item)
            terminals = {est[-1] for est, _ in closures.values()}
            # the premise graph is one component and one-leg relaxation reaches EVERY
            # terminal from the queried entity's base facts, so neither connectivity nor the
            # relaxed closure separates the answers — only the gated closure does
            assert _one_component(premises)
            assert terminals <= _relaxed_closure(premises, entity)
            gated = set(closures[entity][0])
            assert gated & terminals == {closures[entity][0][-1]}
            # the gated closure is a strict subset of the pool: no per-entity namespace
            assert gated < categories

    def test_answer_symmetry(self):
        # all n_chains terminals are structurally identical: head of exactly one rule, in no
        # body, mentioned once. Nothing local to the target separates answer 0 from answer 1.
        by_answer = defaultdict(list)
        for item in self._items(n_samples=120):
            premises, entity, target = _parse_horn(item)
            rules, _, _, _ = _horn_graph(item)
            bodies = Counter(c for a, b, _ in rules.values() for c in (a, b))
            heads = Counter(h for _, _, h in rules.values())
            terminals = [c for c in heads if bodies[c] == 0]
            assert len(terminals) == self.CHAINS and target in terminals
            for t in terminals:
                assert heads[t] == 1 and " ".join(premises).count(f" {t}.") == 1
            assert sum(p.startswith(f"{entity} is a ") for p in premises) == 2
            qa, qb, _ = next(r for r in rules.values() if r[2] == target)
            by_answer[item["output"][0]].append(
                [len(premises), len(bodies), np.mean(list(bodies.values())),
                 max(bodies.values()), np.mean(list(heads.values())),
                 bodies[qa] + bodies[qb], abs(bodies[qa] - bodies[qb]),
                 heads[qa] + heads[qb]])
        assert set(by_answer) == {0, 1}
        # the two cases are exchangeable: every global statistic, and every statistic of the
        # queried terminal's own deriving rule, agrees within two standard errors
        a, b = np.array(by_answer[0]), np.array(by_answer[1])
        gap = np.abs(a.mean(0) - b.mean(0))
        tol = 2 * np.sqrt(a.var(0) / len(a) + b.var(0) / len(b)) + 1e-9
        assert (gap <= tol).all(), f"answer-conditioned statistics differ: {gap} vs {tol}"

    @pytest.mark.parametrize("surface", ["prose", "compact"])
    def test_premise_count_and_alphabet(self, surface):
        task = get_side_task("prontoqa", strategy="horn", hops=self.HOPS, n_samples=20,
                             n_chains=self.CHAINS, rho=self.RHO, surface=surface)
        assert task.alphabet == [0, 1, 2]
        for item in task.train_data:
            premises, _, _ = _parse_horn(item)
            # n_chains chains of hops rules, plus two base facts per chain, no distractors
            assert len(premises) == self.CHAINS * (self.HOPS + 2)
            assert all(v in (0, 1, 2) for v in item["reasoning"])
        for n_chains, hops in ((3, 9), (5, 7)):
            other = get_side_task("prontoqa", strategy="horn", hops=hops, n_samples=5,
                                  n_chains=n_chains, rho=2, surface=surface)
            for item in other.train_data:
                assert len(_parse_horn(item)[0]) == n_chains * (hops + 2)

    @pytest.mark.parametrize("surface", ["prose", "compact"])
    def test_delimiter_count(self, surface):
        for item in self._items(n_samples=20, surface=surface):
            assert item["reasoning"].count(2) == self.HOPS + 2
            assert item["reasoning"][-2] == 2

    @pytest.mark.parametrize("surface", ["prose", "compact"])
    def test_answers_balanced(self, surface):
        outputs = [item["output"][0] for item in self._items(n_samples=100, surface=surface)]
        assert 0.35 <= sum(outputs) / len(outputs) <= 0.65

    def test_premise_order_is_shuffled(self):
        assert len({_decode_trace(item["reasoning"])[0] for item in self._items()}) > 5

    @pytest.mark.parametrize("surface", ["prose", "compact"])
    def test_deterministic_and_seed_sensitive(self, surface):
        kw = dict(strategy="horn", hops=7, n_samples=20, n_chains=4, rho=2, surface=surface)
        a = get_side_task("prontoqa", seed=7, **kw)
        b = get_side_task("prontoqa", seed=7, **kw)
        c = get_side_task("prontoqa", seed=8, **kw)
        d = get_side_task("prontoqa", **{**kw, "n_chains": 5}, seed=7)
        assert a.train_data == b.train_data
        assert a.train_data != c.train_data
        assert a.train_data != d.train_data

    def test_feasibility_asserts(self):
        # P1 is infeasible unless m > rho: at m == rho every non-terminal lies on every chain
        with pytest.raises(AssertionError, match="infeasible"):
            get_side_task("prontoqa", strategy="horn", hops=8, n_samples=2, n_chains=2, rho=2)
        with pytest.raises(AssertionError, match="infeasible"):
            get_side_task("prontoqa", strategy="horn", hops=8, n_samples=2, n_chains=3, rho=3)
        with pytest.raises(AssertionError, match="rho >= 2"):
            get_side_task("prontoqa", strategy="horn", hops=8, n_samples=2, n_chains=3, rho=1)
        with pytest.raises(AssertionError, match="divisible"):
            get_side_task("prontoqa", strategy="horn", hops=8, n_samples=2, n_chains=3, rho=2)


class TestProntoQAHornAttacks:
    """The attack battery. Each attacker sees only item["input"]."""

    @pytest.mark.parametrize("surface", ["prose", "compact"])
    def test_frontier_doubling_fails(self, surface):
        # review.md §6.4.7: rerun the attack that killed horn2 and require it to fail. On horn2
        # this recovered trace AND answer 20/20 at every window; here the transition is
        # undefined almost immediately because the free leg leaves the window.
        report = _run_frontier_doubling(16, 12, (1, 2, 4), surface=surface)
        for w, (exact, answered, accuracy) in report.items():
            assert exact == 0, f"w={w}: recovered {exact} exact traces"
            assert accuracy <= 0.65, f"w={w}: answer accuracy {accuracy} ({answered} completed)"

    @pytest.mark.slow
    def test_frontier_doubling_fails_at_depth(self):
        report = _run_frontier_doubling(32, 10, (2, 4))
        for w, (exact, answered, accuracy) in report.items():
            assert (exact, answered) == (0, 0), f"w={w}: {exact} traces, {answered} completed"
            assert accuracy == 0.5, f"w={w}: answer accuracy {accuracy}"

    @pytest.mark.slow
    def test_spectral_clustering_is_at_chance(self):
        # clustering the rule co-occurrence graph into n_chains groups must not recover chain
        # membership: P1 makes membership a property of body PAIRS, not of categories
        n = 60
        accuracy = _spectral_answer_accuracy(16, n)
        sigma = math.sqrt(0.25 / n)
        assert abs(accuracy - 0.5) <= 2 * sigma, f"spectral answer accuracy {accuracy:.3f}"

    def test_common_neighbour_leak_is_small_and_flat(self):
        # the tolerated leak: sharing puts a weak similarity signal between the queried
        # terminal's rule body and the queried entity's base facts. It must stay near chance
        # and must NOT grow with hops (a growing leak would mean the shared pool leaks more
        # the longer the chains are).
        hops = (8, 16, 32)
        accuracy = {h: _common_neighbour_accuracy(h, 200) for h in hops}
        report = " ".join(f"hops={h}: {accuracy[h]:.3f}" for h in hops)
        for h in hops:
            assert accuracy[h] <= 0.65, f"common-neighbour accuracy too high — {report}"
        assert accuracy[hops[-1]] - accuracy[hops[0]] < 0.08, f"leak grows with hops — {report}"

    def test_trace_positions_are_not_predictable(self):
        # surface statistics must not say which premise fires at which step. Two positions are
        # structurally exposed and are checked separately: step 0 fires the unique rule whose
        # body is the two visible base facts (a constant-span anchor, as in horn2), and the
        # last step fires one of the n_chains rules heading a terminal, so 1/n_chains is its
        # floor. Everything in between must stay near the 1/n_premises chance rate.
        accuracy, chance = _position_prediction_accuracy(16, 80)
        interior = accuracy[2:-1]
        assert np.mean(interior) <= 4 * chance, (
            f"interior top-1 {np.mean(interior):.3f} vs chance {chance:.3f}: {accuracy}")
        assert max(interior) <= 0.15, f"per-position top-1 {accuracy}"
        assert accuracy[-1] <= 0.45, (
            f"final-step top-1 {accuracy[-1]:.3f} (1/n_chains floor is 0.25): {accuracy}")


class TestProntoQAHornCompact:
    """The compact surface is a pure rendering change: the structural invariants are covered by
    the parametrised tests above, so what is left is that prose did not move, that the words
    carry no signal, and that the rendering is actually shorter."""

    # digests of the horn task exactly as the generator produced it before the surface knob
    # existed — surface="prose" (and the default) must stay byte-identical to them forever
    PROSE_DIGESTS = {
        (7, 20, 4, 2, 7): "d222615025c013d9a728c13fbcf243d7a2b380b196225fa3f1daf2cd4d8cd988",
        (16, 8, 4, 2, 3): "4a33f7f5b5ffed19176b2bbbc576073b6a41dec632122290063d28d3419a4a99",
    }

    @staticmethod
    def _digest(task):
        blob = json.dumps([task.train_data, task.test_data, task.explanation], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def test_prose_surface_is_byte_identical(self):
        for (hops, n, n_chains, rho, seed), digest in self.PROSE_DIGESTS.items():
            kw = dict(strategy="horn", hops=hops, n_samples=n, n_chains=n_chains, rho=rho,
                      seed=seed)
            assert self._digest(get_side_task("prontoqa", **kw)) == digest
            assert self._digest(get_side_task("prontoqa", surface="prose", **kw)) == digest

    def test_compact_rendering(self):
        for item in _horn_items(8, 10, surface="compact"):
            block, query = item["input"].rsplit("\n\n", 1)
            for line in block.split("\n"):
                sentence = line.split(". ", 1)[1]
                assert _COMPACT_RULE_RE.fullmatch(sentence) or _COMPACT_FACT_RE.fullmatch(sentence)
            assert _COMPACT_QUERY_RE.fullmatch(query)
        assert len(set(_COMPACT_CATEGORIES)) == len(_COMPACT_CATEGORIES)

    def test_pool_too_small_raises(self):
        # hops=64 at n_chains=4, rho=2 needs 134 distinct categories; the shipped list carries
        # them, and a config that outgrows it must say so instead of crashing inside sample()
        get_side_task("prontoqa", strategy="horn", hops=64, n_samples=1, n_chains=4, rho=2,
                      surface="compact")
        with pytest.raises(AssertionError, match="category pool"):
            get_side_task("prontoqa", strategy="horn", hops=400, n_samples=1, n_chains=4,
                          rho=2, surface="compact")

    def test_unknown_surface_raises(self):
        with pytest.raises(AssertionError, match="unknown horn surface"):
            get_side_task("prontoqa", strategy="horn", hops=7, n_samples=1, n_chains=4, rho=2,
                          surface="terse")

    def test_words_carry_no_signal(self):
        # the category words are common nouns, so an attacker could hope that *which* word is
        # used says something. It cannot: words are drawn uniformly and assigned to chains
        # independently of identity, so per-word rates of (i) landing in the queried entity's
        # chain and (ii) co-occurring with answer 1 must be flat up to binomial noise.
        data = _horn_items(16, 200, surface="compact")
        membership, answer = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
        for item in data:
            premises, entity, _ = _parse_horn(item)
            words = {c for p in premises
                     for c in (_match(_RULE_RES, p) or _match(_FACT_RES, p)[1:])}
            gated = set(_forward_chain(premises, entity)[0])
            bit = item["output"][0]
            for w in words:
                membership[w][0] += w in gated
                membership[w][1] += 1
                answer[w][0] += bit
                answer[w][1] += 1
        for name, counts in (("chain membership", membership), ("answer", answer)):
            used = [(k, n) for k, n in counts.values() if n >= 8]
            p = sum(k for k, _ in used) / sum(n for _, n in used)
            chi2 = sum((k - n * p) ** 2 / (n * p * (1 - p)) for k, n in used)
            df = len(used)
            assert df >= 300, df  # the whole pool has to be exercised for this to mean anything
            assert chi2 <= df + 6 * math.sqrt(2 * df), (
                f"{name} is not word-independent: chi2 {chi2:.0f} on {df} words (p={p:.3f})")

    def test_compact_is_much_shorter(self):
        # chars stand in for tokens in the fast tests; the token measurement is the slow test
        prose = _horn_items(16, 20)
        compact = _horn_items(16, 20, surface="compact")
        ratio = np.mean([len(c["input"]) / len(p["input"]) for p, c in zip(prose, compact)])
        assert ratio < 0.5, ratio

    @pytest.mark.slow
    def test_compact_categories_are_single_tokens(self):
        # the point of the compact surface: every category costs ONE token where it appears,
        # which is mid-line, i.e. with a leading space
        from tinker_cookbook.tokenizer_utils import get_tokenizer
        tokenizer = get_tokenizer("Qwen/Qwen3.6-35B-A3B")
        multi = [w for w in _COMPACT_CATEGORIES if len(tokenizer.encode(f" {w}")) != 1]
        assert not multi, f"{len(multi)} categories are not single tokens: {multi}"
        prose = _horn_items(16, 20)
        compact = _horn_items(16, 20, surface="compact")
        ratio = np.mean([len(tokenizer.encode(c["input"])) / len(tokenizer.encode(p["input"]))
                         for p, c in zip(prose, compact)])
        assert ratio < 0.4, ratio


# ── get_side_task ──


class TestGetSideTask:
    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Unknown side task type"):
            get_side_task("nonexistent")

    def test_all_tasks_have_name(self):
        tasks = [
            get_side_task("rolling_mod_sum", n_samples=10, seq_length=5, operand_range=10, modulus=3),
            get_side_task("limited_arithmetic", n_samples=10, num_steps=3, max_value=20),
            get_side_task("ball_and_cups", n_samples=10, seq_length=5),
            get_side_task("gl32", n_samples=10, seq_length=5),
            get_side_task("function_composition", n_samples=10, n_hops=3, domain_size=5),
            get_side_task("prontoqa", strategy="answer_only", hops=2, n_samples=10),
            get_side_task("prontoqa", strategy="proof_trace", hops=2, n_samples=10),
            get_side_task("prontoqa", strategy="chain", hops=2, n_samples=10),
            get_side_task("prontoqa", strategy="horn", hops=7, n_samples=10, n_chains=4, rho=2),
        ]
        assert [t.name for t in tasks] == [
            "rolling_mod_sum", "limited_arithmetic", "ball_and_cups",
            "gl32", "function_composition",
            "prontoqa_answer_only", "prontoqa_proof_trace",
            "prontoqa_chain", "prontoqa_horn",
        ]


# ── disjoint train/test split (hash-partitioned by problem, lib/side_tasks/base.py) ──


class TestDisjointSplit:
    def test_sampled_tasks_no_key_overlap(self):
        cases = [
            (dict(type="rolling_mod_sum", n_samples=200, seq_length=6, operand_range=10, modulus=3),
             lambda d: tuple(d["input"])),
            (dict(type="ball_and_cups", n_samples=200, seq_length=12),
             lambda d: tuple(d["input"])),
            (dict(type="gl32", n_samples=200, seq_length=12),
             lambda d: tuple(d["input"])),
            (dict(type="function_composition", n_samples=200, n_hops=3, domain_size=4),
             lambda d: tuple(d["input"])),
            (dict(type="limited_arithmetic", n_samples=200, num_steps=3, max_value=20),
             lambda d: d["input"]),
        ]
        for cfg, key in cases:
            task = get_side_task(**cfg)
            train_keys = {key(d) for d in task.train_data}
            test_keys = {key(d) for d in task.test_data}
            assert not (train_keys & test_keys), cfg["type"]
            assert len(test_keys) == len(task.test_data), cfg["type"]  # test deduplicated

    def test_rolling_mod_sum_binary_operands_are_splittable(self):
        # the exact-sequence key keeps low-entropy operand settings splittable: with 0/1
        # operands a multiset key would leave only seq_length + 1 problems
        task = get_side_task("rolling_mod_sum", n_samples=200, seq_length=10,
                             operand_range=1, min_operand=0, modulus=2)
        train_keys = {tuple(d["input"]) for d in task.train_data}
        assert all(tuple(d["input"]) not in train_keys for d in task.test_data)

    def test_partition_tasks_no_input_overlap(self):
        for cfg in [dict(type="prontoqa", strategy="answer_only", hops=3, n_samples=60)]:
            task = get_side_task(**cfg)
            train_inputs = {repr(d["input"]) for d in task.train_data}
            test_inputs = {repr(d["input"]) for d in task.test_data}
            assert not (train_inputs & test_inputs), cfg["type"]

    def test_same_seed_same_data(self):
        cfg = dict(type="rolling_mod_sum", n_samples=100, seq_length=6, operand_range=10, modulus=3)
        a, b = get_side_task(**cfg), get_side_task(**cfg)
        assert a.train_data == b.train_data and a.test_data == b.test_data

    def test_too_small_space_raises(self):
        with pytest.raises(ValueError, match="input space is too small"):
            get_side_task("rolling_mod_sum", n_samples=1000, seq_length=2, operand_range=3, modulus=3)

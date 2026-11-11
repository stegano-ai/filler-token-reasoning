"""Runtime loader for PrOntoQA (github.com/asaparov/prontoqa, Apache-2.0).

The upstream repo has no PyPI package or HF dataset — examples are produced by
running ``run_experiment.py --model-name json``. We vendor the generation modules
unchanged under ``vendor/`` (see vendor/LICENSE, pinned at commit
0a6412b6fddf46324a1cb96e066dd7b3d89b87d6) and drive their JSON path here.

Each example is a dict with ``in_context_example{i}`` entries and a ``test_example``,
carrying ``question``, ``query``, ``chain_of_thought`` and (unless proofs_only) ``answer``.
"""
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

_VENDOR = Path(__file__).parent / "vendor"

# The vendored fictional ontology has a hardcoded 17-name concept pool
# (run_experiment.generate_question), which with "relevant" distractors exhausts around 6
# deduction steps — the generator then retries "Could not extend ontology" forever. For
# hops > 5 we substitute a 270-name pool of the same wumpus-like shape; hops <= 5 keeps the
# upstream pool so existing prontoqa data stays byte-identical under the same seed.
_ONSETS = ["wor", "yer", "zil", "drom", "quim", "bler", "snor", "grum", "tral", "clop",
           "brin", "stel", "frap", "glin", "hax", "jun", "kel", "lum", "mip", "ned",
           "pog", "quat", "rup", "sib", "tev", "vun", "wex", "yob", "zam", "crul"]
_SUFFIXES = ["pus", "ple", "git", "per", "pist", "pant", "por", "pee", "pin"]
_active_pool: list[str] | None = None

# generate_ontology also consumes one property family (adjective set) per tree level, from
# a ~15-entry literal local to generate_question, which caps ontology depth the same way
# the concept pool does. patched generate_theory tops the passed-in list up with these
# synthetic families (again only in hops > 5 mode).
_ADJ_STEMS = [c + v for c in ("bl", "cr", "dr", "fl", "gr", "pl", "sk", "tr", "vr",
                              "sn", "kl", "zw")
              for v in ("arp", "elt", "imb", "orn", "ust", "eev", "oam", "ilk")]
_EXTRA_FAMILIES = [[s + end for end in ("y", "ish", "ous", "ful")] for s in _ADJ_STEMS]


def _concept_pool() -> list[str]:
    return [o + s for s in _SUFFIXES for o in _ONSETS]

# Upstream run_experiment.py argparse defaults, mirrored so behaviour matches the CLI.
_DEFAULTS = dict(
    resume=False, model_size="", ordering="postorder", test_ordering=None,
    num_trials=500, few_shot_examples=8, ontology="fictional",
    distractors="relevant", test_distractors="relevant", no_adjectives=False,
    proofs_only=False, DFS="none", disjoint_concept_names=False, OOD=False,
    proof_width=2, test_width_diff=0, repetitions_per_test=1, rule_types=3,
    prompting="COT", deduction_rule="ModusPonens", generate_non_atomic_steps=False,
    seed=62471893,
)


def _load_vendor():
    if "run_experiment" in sys.modules:
        return sys.modules["run_experiment"]
    # Vendor modules use bare sibling imports (`from theory import *`) and read
    # bad_patterns.txt relative to CWD at import time, so scope both the path and
    # CWD to the import itself instead of leaking them into the process.
    cwd = os.getcwd()
    sys.path.insert(0, str(_VENDOR))
    try:
        os.chdir(_VENDOR)
        import run_experiment
    finally:
        os.chdir(cwd)
        sys.path.remove(str(_VENDOR))
    for name in _concept_pool():
        if name not in run_experiment.morphology.plural_nouns:
            run_experiment.morphology.add_noun(name, name + ("es" if name.endswith("s") else "s"))
    orig = run_experiment.generate_question

    def patched(num_deduction_steps, available_concept_names, *a, **kw):
        if available_concept_names is None and _active_pool is not None:
            available_concept_names = list(_active_pool)
        return orig(num_deduction_steps, available_concept_names, *a, **kw)

    run_experiment.generate_question = patched
    orig_theory = run_experiment.generate_theory

    def patched_theory(available_concept_names, available_property_families, config):
        if _active_pool is not None:
            have = {f[0] for f in available_property_families}
            available_property_families.extend(
                list(f) for f in _EXTRA_FAMILIES if f[0] not in have)
        return orig_theory(available_concept_names, available_property_families, config)

    run_experiment.generate_theory = patched_theory
    return run_experiment


def generate(hops: int, **overrides) -> list[dict]:
    """Generate PrOntoQA examples with ``1 + hops`` deduction steps.

    ``overrides`` set any of the upstream flags in ``_DEFAULTS`` (e.g. ``ontology``,
    ``num_trials``, ``distractors``, ``ordering``, ``proofs_only``, ``seed``).
    Returns the list of example dicts, one per trial.
    """
    unknown = set(overrides) - set(_DEFAULTS)
    if unknown:
        raise ValueError(f"Unknown PrOntoQA option(s): {sorted(unknown)}")
    run_experiment = _load_vendor()
    global _active_pool
    _active_pool = _concept_pool() if hops > 5 else None
    args = SimpleNamespace(**{**_DEFAULTS, **overrides})
    if args.test_ordering is None:
        args.test_ordering = args.ordering
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "examples.json"
        # The vendored generator prints benign "Could not extend ontology" retry warnings to
        # stdout (thousands of them with relevant distractors); its actual output goes to `out`,
        # so suppress stdout here. Real errors raise as exceptions and are unaffected.
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            run_experiment.run_experiment("json", args, 1 + hops, 1 + hops, str(out))
        return list(json.loads(out.read_text()).values())

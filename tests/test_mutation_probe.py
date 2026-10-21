"""Cheap guard for scripts/mutation_probe.py (the probe itself runs as a CI step).

Every mutant anchor must still occur exactly once in the current source, or
the probe would silently stop testing what it claims to test.
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("probe", ROOT / "scripts" / "mutation_probe.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


@pytest.mark.parametrize("mid", sorted(probe.MUTANTS))
def test_anchor_occurs_exactly_once(mid):
    rel, old, new, _, expected = probe.MUTANTS[mid]
    text = (ROOT / "src" / rel).read_text(encoding="utf-8")
    assert text.count(old) == 1
    assert old != new
    assert expected and all((ROOT / suite).is_file() for suite in expected)
    assert set(expected.values()) <= {"relation", "guard", "survives"}


def test_catalogue_contains_a_known_survivor_and_a_policy_only_mutant():
    """The probe documents limits, not only successes."""
    verdicts = {(m, s): v for m, row in probe.MUTANTS.items() for s, v in row[4].items()}
    assert verdicts[("M7_rates_swapped_everywhere", probe.METAMORPHIC)] == "survives"
    assert verdicts[("M7_rates_swapped_everywhere", probe.SIMULATION)] == "relation"
    assert verdicts[("M8_policy_looks_one_step_short", probe.CROSS)] == "relation"


def test_unknown_mutant_is_rejected():
    with pytest.raises(ValueError, match="unknown mutants"):
        probe.probe(["M99"])


@pytest.mark.parametrize(
    "message, verdict",
    [
        ("AssertionError: assert 0.1 == 0.2", "relation"),
        ("assert False", "relation"),
        ("Failed: DID NOT RAISE <class 'ValueError'>", "relation"),
        ("ValueError: belief must sum to one", "guard"),
        ("ArithmeticError: outcome probabilities do not sum to one", "guard"),
        ("ZeroDivisionError: float division by zero", "guard"),
        ("TypeError: unsupported operand", "crash"),
        ("collection failure", "crash"),
        ("", "crash"),
    ],
)
def test_failure_classification(message, verdict):
    assert probe.classify(message) == verdict

"""Package predictions vs a Monte-Carlo simulation of the generative story.

The oracle is a second transcription of the same equations, so an error made
when turning the verbal model into equations would be shared by both (e.g. the
two rates swapped everywhere: every metamorphic relation still holds, see
scripts/mutation_probe.py). The simulator in tests/_simulator.py samples the
verbal model directly (Gillespie jumps, no matrices). Agreement is checked
per reading pattern -- its probability and the posterior of x0, x, "changed"
and theta -- with binomial z-scores.

Threshold: |z| <= 5 for every score. With at most about 110 scores per case
and a normal approximation, a false alarm has probability below 1e-4 per
case; the runs are seeded, so the outcome is reproducible. Measured at
100 000 runs: null max |z| 2.0-2.7; wrong stories 28-91. Each case is paired with power
controls: deliberately wrong stories must exceed the threshold by far.
"""

import importlib.util
from pathlib import Path

import _simulator as S
import pytest

from atomic_information_lab import Hypothesis, Problem

SPEC = importlib.util.spec_from_file_location("gate", Path(__file__).parents[1] / "scripts" / "research_gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

RUNS = 100_000
Z_MAX = 5.0
CASES = {
    "default_precise": (((1.0, 0.07, 0.045, 0.025),), ("reference", "precise", "precise")),
    "fast_dynamics": (((1.0, 3.0, 2.0, 0.025),), ("reference", "fast", "fast")),
    "two_models": (((0.5, 0.07, 0.045, 0.025), (0.5, 0.23, 0.16, 0.10)), ("reference", "fast", "precise")),
    "retention_three": (
        ((0.2, 0.07, 0.045, 0.025), (0.3, 0.23, 0.16, 0.10), (0.5, 0.0, 0.3, 0.2)),
        ("fast", "reference", "precise", "fast"),
    ),
}


def predictions(components, sequence):
    """Package prediction for every reading pattern of a fixed sequence."""
    pr = Problem(tuple(Hypothesis(k01, k10, f) for _, k01, k10, f in components), tuple(c[0] for c in components))
    out = {}

    def walk(belief, readings, p):
        if len(readings) == len(sequence):
            m = pr.marginals(belief)
            out[tuple(readings)] = (
                p,
                m["p_initial_one"],
                m["p_current_one"],
                m["p_changed"],
                pr.model_posterior(belief),
            )
            return
        for y, py, post in pr.branches(belief, pr.action(sequence[len(readings)])):
            walk(post, readings + [y], p * py)

    walk(pr.initial_belief(), [], 1.0)
    return out


def max_z(name, variant=None, seed=20261021):
    components, sequence = CASES[name]
    table = S.simulate(components, sequence, RUNS, seed, variant=variant)
    scores = S.z_scores(table, RUNS, predictions(components, sequence))
    return max(scores), len(scores)


@pytest.mark.parametrize("name", sorted(CASES))
def test_package_agrees_with_the_simulated_story(name):
    worst, count = max_z(name)
    assert count >= 30
    assert worst <= Z_MAX, f"{name}: max |z| = {worst:.2f} over {count} scores"


@pytest.mark.parametrize(
    "name, variant, floor",
    [
        ("default_precise", "swap_rates", 20.0),  # the error no metamorphic relation sees
        ("fast_dynamics", "swap_rates", 20.0),
        ("two_models", "swap_rates", 15.0),
        ("default_precise", "read_before_evolve", 10.0),
        ("two_models", "bias_x4", 10.0),
    ],
)
def test_power_wrong_stories_are_detected(name, variant, floor):
    worst, _ = max_z(name, variant)
    assert worst > floor > Z_MAX


def test_seed_is_used_and_null_holds_for_another_seed():
    components, sequence = CASES["default_precise"]
    a = S.simulate(components, sequence, 2_000, 1)
    b = S.simulate(components, sequence, 2_000, 2)
    assert a != b
    assert max_z("default_precise", seed=7)[0] <= Z_MAX


def test_impossible_pattern_or_exact_prediction_violation_scores_inf():
    components, sequence = CASES["default_precise"]
    pred = predictions(components, sequence)
    table = S.simulate(components, sequence, 1_000, 3)
    missing = dict(pred)
    missing.pop(next(iter(table)))  # the prediction now calls an observed pattern impossible
    assert max(S.z_scores(table, 1_000, missing)) == float("inf")
    with pytest.raises(ValueError):
        S.simulate(components, sequence, 10, 0, variant="bogus")


def test_simulator_is_independent_of_package_and_oracle():
    ok, why = gate.oracle_independence(S, allowed=gate.ORACLE_ALLOWED_IMPORTS | {"random"})
    assert ok, why
    # the same rule without "random" refuses it: the allowlist is really applied
    assert gate.oracle_independence(S)[0] is False

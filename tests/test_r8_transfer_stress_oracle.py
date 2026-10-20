"""R8 seed-fixed independent policy-transfer stress, synthetic finite grid only."""

from __future__ import annotations

import random

import pytest

from atomic_information_lab import Hypothesis, Objective, Problem
from atomic_information_lab.planner import nominal_policy
from atomic_information_lab.robustness import policy_transfer_report
from tests._oracle import Model

SEED = 20261020
CASES = 20


def make_cases():
    rng = random.Random(SEED)
    return [
        (
            round(rng.uniform(0.002, 0.25), 6),
            round(rng.uniform(0.003, 0.20), 6),
            round(rng.uniform(0.001, 0.06), 6),
            round(rng.uniform(0.002, 0.06), 6),
        )
        for _ in range(CASES)
    ]


def _problem(k01, k10, bias, lam):
    return Problem(hypotheses=(Hypothesis(k01, k10, bias),), objective=Objective(energy_lambda=lam))


@pytest.mark.parametrize("k01,k10,bias,lam", make_cases())
def test_r8_transfer_matches_independent_rational_oracle(k01, k10, bias, lam):
    nominal = _problem(0.07, 0.045, 0.025, 0.012)
    truth = _problem(k01, k10, bias, lam)
    measured = policy_transfer_report(nominal, [truth], 2)["scenarios"][0]
    oracle = Model(((1.0, k01, k10, bias),), lam=lam)
    nominal_choice = nominal_policy(nominal)

    def fixed_policy(_message, history, h):
        name = nominal_choice((), history, h)
        return None if name == "stop" else next(a for a in oracle.actions if a[0] == name)

    independent_value, independent_steps = oracle.policy_value(oracle.root(), 2, fixed_policy)
    true_min = oracle.value(oracle.root(), 2)
    assert abs(float(independent_value) - measured["nominal_policy_under_truth"]) <= 3e-11
    assert abs(float(independent_steps) - measured["expected_steps"]) <= 3e-11
    assert abs(float(true_min) - measured["true_optimal_value"]) <= 3e-11


def test_r8_oracle_can_reject_a_biased_solver_value():
    nominal = _problem(0.07, 0.045, 0.025, 0.012)
    truth = _problem(0.15, 0.035, 0.025, 0.04)
    report = policy_transfer_report(nominal, [truth], 2)["scenarios"][0]
    corrupted = report["true_optimal_value"] + 0.0001
    oracle = Model(((1.0, 0.15, 0.035, 0.025),), lam=0.04)
    assert abs(float(oracle.value(oracle.root(), 2)) - corrupted) > 3e-11

"""R8 seed-fixed independent policy-transfer stress, synthetic finite grid only."""

from __future__ import annotations

import random
from dataclasses import replace

import pytest
from _oracle import Model
from _transfer_check import oracle_reference, transfer_mismatches

from atomic_information_lab import Hypothesis, Objective, Problem, robustness
from atomic_information_lab.planner import nominal_policy
from atomic_information_lab.robustness import policy_transfer_report

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


def _scenario(k01, k10, bias, lam):
    nominal = _problem(0.07, 0.045, 0.025, 0.012)
    measured = policy_transfer_report(nominal, [_problem(k01, k10, bias, lam)], 2)["scenarios"][0]
    reference = oracle_reference(Model(((1.0, k01, k10, bias),), lam=lam), nominal_policy(nominal), 2)
    return measured, reference


@pytest.mark.parametrize("k01,k10,bias,lam", make_cases())
def test_r8_transfer_matches_independent_rational_oracle(k01, k10, bias, lam):
    measured, reference = _scenario(k01, k10, bias, lam)
    assert transfer_mismatches(measured, reference) == []


def test_r8_oracle_can_reject_a_biased_solver_value(monkeypatch):
    """The solver's optimum is forged by +1e-10 INSIDE the package; the shared check must reject it.

    The earlier version added 1e-4 to the reported value in the test itself and
    compared inline, i.e. it only showed 1e-4 > 3e-11. Here the forgery flows
    through ``policy_transfer_report`` into the same check the stress tests use.
    """
    params = (0.15, 0.035, 0.025, 0.04)
    clean, reference = _scenario(*params)
    assert transfer_mismatches(clean, reference) == []  # positive control

    real = robustness.solve

    def biased_solve(*args, **kwargs):
        decision = real(*args, **kwargs)
        return replace(decision, value=decision.value + 1e-10)

    monkeypatch.setattr(robustness, "solve", biased_solve)
    forged, _ = _scenario(*params)
    assert forged["true_optimal_value"] != clean["true_optimal_value"]  # the forgery reached the report
    assert transfer_mismatches(forged, reference) == ["true_optimal_value"]

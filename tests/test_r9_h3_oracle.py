"""R9 independent held-out policy-transfer checks at horizon 3, still synthetic."""

import random

import pytest
from _oracle import Model

from atomic_information_lab import Hypothesis, Objective, Problem
from atomic_information_lab.planner import nominal_policy
from atomic_information_lab.robustness import policy_transfer_report


def cases():
    rng = random.Random(2026102009)
    return [
        (
            round(rng.uniform(0.005, 0.16), 6),
            round(rng.uniform(0.004, 0.13), 6),
            round(rng.uniform(0.003, 0.045), 6),
            round(rng.uniform(0.004, 0.05), 6),
        )
        for _ in range(8)
    ]


@pytest.mark.parametrize("k01,k10,bias,lam", cases())
def test_h3_transfer_independent_oracle(k01, k10, bias, lam):
    nominal = Problem(hypotheses=(Hypothesis(0.07, 0.045, 0.025),), objective=Objective(energy_lambda=0.012))
    truth = Problem(hypotheses=(Hypothesis(k01, k10, bias),), objective=Objective(energy_lambda=lam))
    result = policy_transfer_report(nominal, [truth], 3)["scenarios"][0]
    oracle = Model(((1.0, k01, k10, bias),), lam=lam)
    pick = nominal_policy(nominal)

    def frozen_policy(_message, history, h):
        name = pick((), history, h)
        return None if name == "stop" else next(a for a in oracle.actions if a[0] == name)

    val, steps = oracle.policy_value(oracle.root(), 3, frozen_policy)
    optimum = oracle.value(oracle.root(), 3)
    assert abs(result["nominal_policy_under_truth"] - float(val)) <= 3e-11
    assert abs(result["expected_steps"] - float(steps)) <= 3e-11
    assert abs(result["true_optimal_value"] - float(optimum)) <= 3e-11
    assert result["signed_regret"] >= -result["tolerance"]


def test_h3_negative_control_detects_value_bias():
    k01, k10, bias, lam = cases()[0]
    nominal = Problem(hypotheses=(Hypothesis(0.07, 0.045, 0.025),), objective=Objective(energy_lambda=0.012))
    truth = Problem(hypotheses=(Hypothesis(k01, k10, bias),), objective=Objective(energy_lambda=lam))
    val = policy_transfer_report(nominal, [truth], 3)["scenarios"][0]["true_optimal_value"]
    oracle = Model(((1.0, k01, k10, bias),), lam=lam)
    assert abs((val + 1e-5) - float(oracle.value(oracle.root(), 3))) > 3e-11

"""Independent TEST-ONLY oracle evaluates a frozen nominal policy under truth."""

import pytest
from _oracle import Model

from atomic_information_lab import Hypothesis, Objective, Problem
from atomic_information_lab.planner import nominal_policy
from atomic_information_lab.robustness import policy_transfer_report


@pytest.mark.parametrize(
    "k01,k10,lam", [(0.01, 0.01, 0.012), (0.07, 0.045, 0.012), (0.2, 0.18, 0.012), (0.07, 0.045, 0.035)]
)
def test_independent_transfer_policy_value(k01, k10, lam):
    nominal = Problem(hypotheses=(Hypothesis(0.07, 0.045, 0.025),), objective=Objective(energy_lambda=0.012))
    truth = Problem(hypotheses=(Hypothesis(k01, k10, 0.025),), objective=Objective(energy_lambda=lam))
    reported = policy_transfer_report(nominal, [truth], 2)["scenarios"][0]
    ref = Model(((1.0, k01, k10, 0.025),), lam=lam)
    choice = nominal_policy(nominal)

    def independent_choice(_msg, hist, h):
        action = choice((), hist, h)
        return None if action == "stop" else next(a for a in ref.actions if a[0] == action)

    independent_value, independent_steps = ref.policy_value(ref.root(), 2, independent_choice)
    assert abs(float(independent_value) - reported["nominal_policy_under_truth"]) < 3e-11
    assert abs(float(independent_steps) - reported["expected_steps"]) < 3e-11

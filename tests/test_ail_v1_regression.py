"""Run after applying hardening in atomic-information-lab checkout (pip install -e .)."""

import math

import pytest

from atomic_information_lab.model import Objective, Problem
from atomic_information_lab.planner import _pick, evaluate_policy, open_loop_value, solve


def test_action_energy_overflow_fails_closed():
    prob = Problem(objective=Objective(energy_lambda=1e308))
    with pytest.raises(ArithmeticError, match="non-finite"):
        prob.step_cost(prob.action("precise"))


def test_forced_bellman_overflow_fails_closed():
    prob = Problem(objective=Objective(energy_lambda=1e308))
    with pytest.raises(ArithmeticError):
        solve(prob, horizon=1, allow_stop=False)


def test_open_loop_and_policy_overflow_fail_closed():
    prob = Problem(objective=Objective(energy_lambda=1e308))
    with pytest.raises(ArithmeticError):
        open_loop_value(prob, ["precise"])
    with pytest.raises(ArithmeticError):
        evaluate_policy(prob, lambda belief, history, h: "precise", 1)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_candidate_cannot_be_ignored(bad):
    with pytest.raises(ArithmeticError):
        _pick([(0.25, "stop"), (bad, "fast")])


def test_normal_risk_still_finite():
    p = Problem()
    v = solve(p, horizon=1).value
    assert math.isfinite(v) and v >= 0

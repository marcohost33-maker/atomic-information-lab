"""R6 empirical robustness matrix; separate rational oracle cross-checks."""
from __future__ import annotations
import math
import pytest
from atomic_information_lab import Problem, Hypothesis, Objective
from atomic_information_lab.robustness import policy_transfer_report
from tests._oracle import Model


def _p(k01, k10, lam=0.012):
    return Problem(hypotheses=(Hypothesis(k01, k10, 0.025),),
                   objective=Objective(energy_lambda=lam))


def test_robustness_grid_is_finite_and_nonnegative_regret_within_tolerance():
    nominal = _p(0.07, 0.045)
    truths = [nominal, _p(0.01, 0.01), _p(0.10, 0.08), _p(0.2, 0.18)]
    out = policy_transfer_report(nominal, truths, 3)
    assert out["independent_oracle"] is False
    assert len(out["scenarios"]) == 4
    assert all(math.isfinite(x["signed_regret"]) and
               x["signed_regret"] >= -x["tolerance"] for x in out["scenarios"])
    assert abs(out["scenarios"][0]["signed_regret"]) < 1e-10


@pytest.mark.parametrize("k01,k10,lam", [(.01,.01,.012),(.07,.045,.012),(.2,.18,.012),(.07,.045,.035)])
def test_true_optimum_checked_against_separate_oracle(k01, k10, lam):
    p = _p(k01,k10,lam)
    out = policy_transfer_report(_p(.07,.045), [p], 2)
    independent = Model(((1.0,k01,k10,.025),),lam=lam)
    oracle = float(independent.value(independent.root(),2))
    assert abs(oracle-out["scenarios"][0]["true_optimal_value"]) < 3e-11


def test_refuse_incompatible_action_semantics():
    from atomic_information_lab import Action
    nominal = _p(.07,.045)
    altered = Problem(actions=(Action("fast",1,0.12,200),)+nominal.actions[1:])
    with pytest.raises(ValueError,match="catalogue"):
        policy_transfer_report(nominal,[altered],2)


@pytest.mark.parametrize("bad", [[], [None], [Problem()]*101])
def test_invalid_truth_grids_fail_closed(bad):
    with pytest.raises(ValueError):
        policy_transfer_report(Problem(),bad,2)

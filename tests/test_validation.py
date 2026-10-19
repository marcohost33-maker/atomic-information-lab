"""Fail-closed input handling. Every rejection test carries a positive control."""

import math

import pytest

from atomic_information_lab import (
    DEFAULT_ACTIONS,
    DEFAULT_HYPOTHESIS,
    Action,
    Hypothesis,
    Objective,
    Problem,
    solve,
)
from atomic_information_lab.planner import best_open_loop, open_loop_value

BAD_NUMBERS = [math.nan, math.inf, -math.inf, -1.0, True, "0.1", None]


@pytest.mark.parametrize("field", ["k01", "k10", "bias_flip"])
@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_hypothesis_rejects_bad_numbers(field, bad):
    good = {"k01": 0.07, "k10": 0.045, "bias_flip": 0.025}
    Hypothesis(**good)  # positive control
    with pytest.raises(ValueError):
        Hypothesis(**{**good, field: bad})


def test_hypothesis_bias_flip_range():
    Hypothesis(0.1, 0.1, 0.5)  # boundary is allowed
    with pytest.raises(ValueError):
        Hypothesis(0.1, 0.1, 0.5000001)


@pytest.mark.parametrize("field", ["backaction", "error", "energy"])
@pytest.mark.parametrize("bad", BAD_NUMBERS)
def test_action_rejects_bad_numbers(field, bad):
    good = {"name": "a", "backaction": 1.0, "error": 0.1, "energy": 1.0}
    Action(**good)
    with pytest.raises(ValueError):
        Action(**{**good, field: bad})


@pytest.mark.parametrize("name", ["", "stop", 3, None])
def test_action_rejects_bad_names(name):
    Action("ok", 1.0, 0.1, 1.0)
    with pytest.raises(ValueError):
        Action(name, 1.0, 0.1, 1.0)


def test_action_error_above_half_rejected():
    Action("a", 1.0, 0.5, 1.0)
    with pytest.raises(ValueError):
        Action("a", 1.0, 0.51, 1.0)


@pytest.mark.parametrize(
    "kw",
    [
        {"alpha": 1.01},
        {"alpha": -0.01},
        {"energy_lambda": -1e-9},
        {"retention_weight": -0.1},
        {"alpha": math.nan},
        {"energy_lambda": math.inf},
    ],
)
def test_objective_rejects(kw):
    Objective()
    with pytest.raises(ValueError):
        Objective(**kw)


@pytest.mark.parametrize("k01", [1e308, 6e307])  # 6e307 * 3.5 > float max
def test_rate_overflow_fails_closed(k01):
    """Regression for the historic R10-R12 silent failure.

    There ``k01 * backaction`` overflowed to inf, the posterior became NaN,
    ``if z > 0`` dropped BOTH observation branches, and the planner reported
    the bare energy cost (0.0336) as the optimal risk.
    """
    Problem((Hypothesis(1e300, 0.045, 0.025),))  # large but finite: allowed
    with pytest.raises(ValueError, match="overflow"):
        Problem((Hypothesis(k01, 0.045, 0.025),))


def test_backaction_overflow_fails_closed():
    Problem(actions=(Action("x", 1e300, 0.1, 1.0),))
    with pytest.raises(ValueError, match="overflow"):
        Problem(actions=(Action("x", 1e308, 0.1, 1.0),), hypotheses=(Hypothesis(10.0, 0.0, 0.0),))


def test_branch_guard_rejects_nonfinite_internal_state():
    """Defence in depth: a corrupted kernel must raise, not drop branches."""
    pr = Problem()
    b = pr.initial_belief()
    assert len(pr.branches(b, DEFAULT_ACTIONS[0])) == 2  # positive control
    key = (0, "fast")
    row = list(pr._kernel[key][0])
    row[0] = math.nan
    pr._kernel[key] = (tuple(row),) + pr._kernel[key][1:]
    with pytest.raises(ArithmeticError):
        pr.branches(b, DEFAULT_ACTIONS[0])


def test_branch_guard_rejects_mass_leak():
    pr = Problem()
    b = pr.initial_belief()
    key = (0, "fast")
    row = [v * 0.5 for v in pr._kernel[key][0]]  # sub-stochastic row leaks mass
    pr._kernel[key] = (tuple(row),) + pr._kernel[key][1:]
    with pytest.raises(ArithmeticError, match="sum to one"):
        pr.branches(b, DEFAULT_ACTIONS[0])


@pytest.mark.parametrize("h", [-1, 6, 2.0, True, None, "3"])
def test_horizon_rejected(h):
    pr = Problem()
    solve(pr, horizon=5)
    with pytest.raises(ValueError):
        solve(pr, horizon=h)


def test_allow_stop_must_be_bool():
    with pytest.raises(ValueError):
        solve(Problem(), horizon=1, allow_stop=0)


@pytest.mark.parametrize(
    "acts",
    [
        (),
        DEFAULT_ACTIONS + (Action("x", 1, 0.1, 1), Action("y", 1, 0.1, 1)),
        (DEFAULT_ACTIONS[0], DEFAULT_ACTIONS[0]),
        ("fast",),
    ],
)
def test_bad_action_sets(acts):
    Problem(actions=DEFAULT_ACTIONS)
    with pytest.raises(ValueError):
        Problem(actions=acts)


@pytest.mark.parametrize("w", [(0.6, 0.6), (math.nan, 1.0), (-0.5, 1.5), (1.0,), (0.5, 0.5, 0.0)])
def test_bad_weights(w):
    hyps = (DEFAULT_HYPOTHESIS, Hypothesis(0.2, 0.1, 0.1))
    Problem(hyps, (0.5, 0.5))
    with pytest.raises(ValueError):
        Problem(hyps, w)


def test_too_many_hypotheses():
    Problem((DEFAULT_HYPOTHESIS,) * 4)
    with pytest.raises(ValueError):
        Problem((DEFAULT_HYPOTHESIS,) * 5)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b[:-1],
        lambda b: (b[0] + 0.01,) + b[1:],
        lambda b: (math.nan,) + b[1:],
        lambda b: (-0.25, 0.5) + b[2:],
    ],
)
def test_unnormalised_belief_rejected(mutate):
    pr = Problem()
    b = pr.initial_belief()
    pr.validate_belief(b)
    with pytest.raises(ValueError):
        solve(pr, mutate(b), horizon=1)


def test_unknown_action_in_sequence():
    pr = Problem()
    open_loop_value(pr, ("fast",))
    with pytest.raises(ValueError):
        open_loop_value(pr, ("teleport",))
    with pytest.raises(ValueError):
        best_open_loop(pr, 6)


def test_action_from_other_problem_rejected():
    pr = Problem()
    with pytest.raises(ValueError):
        pr.branches(pr.initial_belief(), Action("fast", 2.0, 0.12, 1.0))

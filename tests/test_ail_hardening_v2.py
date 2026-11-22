"""Additional AIL model numerical checks for pinned upstream + local hardening."""

from __future__ import annotations

import math
import random

import pytest

from atomic_information_lab import (
    Action,
    Hypothesis,
    Objective,
    Problem,
    best_open_loop,
    evaluate_policy,
    optimal_policy,
    solve,
)
from atomic_information_lab.planner import _pick


@pytest.mark.parametrize("energy,lam", [(1e308, 2.0), (1e300, 1e300), (1e308, 1e308)])
def test_composed_overflow_fails_closed_even_with_valid_stop(energy, lam):
    pr = Problem(actions=(Action("huge", 1.0, 0.05, energy),), objective=Objective(energy_lambda=lam))
    assert math.isfinite(pr.terminal_loss(pr.initial_belief()))
    with pytest.raises(ArithmeticError):
        solve(pr, horizon=1, allow_stop=True)
    with pytest.raises(ArithmeticError):
        best_open_loop(pr, horizon=1)


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_choice_rejects_an_invalid_alternative_even_if_first_is_finite(bad):
    with pytest.raises(ArithmeticError):
        _pick([(0.25, "stop"), (bad, "huge")])


@pytest.mark.parametrize("seed", list(range(12)))
def test_finite_decision_policy_decomposition(seed):
    rng = random.Random(seed)
    hyp = Hypothesis(rng.uniform(0, 0.4), rng.uniform(0, 0.4), rng.uniform(0, 0.5))
    lam = rng.uniform(0.001, 0.03)
    beta = rng.uniform(0.0, 0.5)
    pr = Problem(hypotheses=(hyp,), objective=Objective(alpha=rng.random(), energy_lambda=lam, retention_weight=beta))
    for horizon in (0, 1, 2):
        dec = solve(pr, horizon=horizon)
        stats = evaluate_policy(pr, optimal_policy(pr), horizon)
        ol, _ = best_open_loop(pr, horizon)
        assert math.isfinite(dec.value)
        assert math.isfinite(stats.value)
        assert abs(stats.value - dec.value) < 2e-12
        assert abs(stats.value - (stats.expected_terminal_loss + lam * stats.expected_energy)) < 2e-12
        assert dec.value <= ol + 1e-12

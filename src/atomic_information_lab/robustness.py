"""Deterministic *synthetic* policy-transfer sensitivity analysis.

A nominal model plans without seeing the true parameters. Each candidate
truth evaluates the frozen nominal policy under the candidate's own dynamics.
This is not Bayesian model identification or experimental quantum evidence.
"""
from __future__ import annotations

from collections.abc import Sequence
from math import isfinite

from .model import Problem
from .planner import MAX_HORIZON, TIE_TOL, evaluate_policy, nominal_policy, solve


def policy_transfer_report(nominal: Problem, truths: Sequence[Problem], horizon: int) -> dict:
    """Report signed regret and expected observation cost for controlled scenarios.

    A true optimum is relative to each TRUE model (not to the nominal model).
    If a nominal event has zero likelihood but occurs under truth, policy
    evaluation fails closed; it is not silently skipped.
    """
    if type(nominal) is not Problem:
        raise TypeError("nominal must be a Problem")
    if type(horizon) is not int or not 0 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be integer in 0..{MAX_HORIZON}")
    truths = tuple(truths)
    if not truths or len(truths) > 100 or any(type(t) is not Problem for t in truths):
        raise ValueError("truths must contain 1..100 Problem instances")
    nom_policy = nominal_policy(nominal)
    scenarios = []
    for index, truth in enumerate(truths):
        if truth.actions != nominal.actions:
            raise ValueError("action catalogue semantics must match across models")
        actual = evaluate_policy(truth, nom_policy, horizon)
        true_d = solve(truth, horizon=horizon)
        regret = actual.value - true_d.value
        v = (actual.value, true_d.value, regret, actual.expected_steps,
             actual.expected_energy, actual.expected_terminal_loss)
        if any(not isfinite(x) for x in v):
            raise ArithmeticError("non-finite transfer result")
        tol = 2e-10 + (horizon + 1) * TIE_TOL * max(1.0, abs(actual.value), abs(true_d.value))
        if regret < -tol:
            raise ArithmeticError("transferred policy outperformed claimed optimum beyond tolerance")
        scenarios.append({
            "scenario": index,
            "true_optimal_value": true_d.value,
            "nominal_policy_under_truth": actual.value,
            "signed_regret": regret,
            "expected_steps": actual.expected_steps,
            "expected_energy": actual.expected_energy,
            "expected_terminal_loss": actual.expected_terminal_loss,
            "tolerance": tol,
        })
    return {
        "schema": "ail-synthetic-policy-transfer-v1",
        "scope": "classical two-state, supplied finite candidate grid, no worst-case guarantee outside grid",
        "independent_oracle": False,
        "horizon": horizon,
        "scenarios": scenarios,
        "max_observed_regret": max(x["signed_regret"] for x in scenarios),
        "min_observed_regret": min(x["signed_regret"] for x in scenarios),
    }

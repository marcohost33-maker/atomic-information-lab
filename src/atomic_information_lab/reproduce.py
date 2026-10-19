"""Reproduce the reference tables (historic R11 / R12 numbers) from the consolidated code."""

from __future__ import annotations

from dataclasses import replace

from .model import DEFAULT_ACTIONS, DEFAULT_HYPOTHESIS, TWO_HYPOTHESES, Action, Objective, Problem
from .planner import best_open_loop, evaluate_policy, greedy_policy, nominal_policy, optimal_policy, solve

__all__ = ["single_model_table", "two_model_table", "two_model_sensitivity"]


def single_model_table(max_horizon: int = 5, objective: Objective = Objective()) -> list[dict]:
    """Optional STOP vs forced horizon vs best open loop vs greedy (historic R11 table)."""
    pr = Problem((DEFAULT_HYPOTHESIS,), actions=DEFAULT_ACTIONS, objective=objective)
    rows = []
    for h in range(max_horizon + 1):
        d = solve(pr, horizon=h)
        stats = evaluate_policy(pr, optimal_policy(pr), h)
        fixed, seq = best_open_loop(pr, h)
        rows.append(
            {
                "horizon": h,
                "optimal_with_stop": d.value,
                "first_action": d.action,
                "forced_exactly_h": solve(pr, horizon=h, allow_stop=False).value,
                "best_open_loop": fixed,
                "best_open_loop_sequence": list(seq),
                "greedy_with_stop": evaluate_policy(pr, greedy_policy(pr), h).value,
                "expected_steps": stats.expected_steps,
                "expected_terminal_loss": stats.expected_terminal_loss,
            }
        )
    return rows


def two_model_table(max_horizon: int = 4, objective: Objective = Objective()) -> list[dict]:
    """Two static model hypotheses, prior (0.5, 0.5) (historic R12 table)."""
    pr = Problem(TWO_HYPOTHESES, (0.5, 0.5), DEFAULT_ACTIONS, objective)
    nominal = Problem((TWO_HYPOTHESES[0],), actions=DEFAULT_ACTIONS, objective=replace(objective, retention_weight=0.0))
    rows = []
    for h in range(max_horizon + 1):
        d = solve(pr, horizon=h)
        fixed, seq = best_open_loop(pr, h)
        rows.append(
            {
                "horizon": h,
                "adaptive_with_stop": d.value,
                "first_action": d.action,
                "best_open_loop": fixed,
                "best_open_loop_sequence": list(seq),
                "nominal_single_model_policy": evaluate_policy(pr, nominal_policy(nominal), h).value,
            }
        )
    return rows


def two_model_sensitivity(horizon: int = 4) -> list[dict]:
    """Retention weight x reference back-action grid (historic R12 sensitivity)."""
    rows = []
    for beta in (0.0, 0.1, 0.2, 0.4):
        for ref_back in (0.0, 0.55, 1.0):
            acts = (
                DEFAULT_ACTIONS[0],
                DEFAULT_ACTIONS[1],
                Action("reference", ref_back, 0.025, 0.8, reads_reference=True),
            )
            pr = Problem(TWO_HYPOTHESES, (0.5, 0.5), acts, Objective(retention_weight=beta))
            d = solve(pr, horizon=horizon)
            fixed, seq = best_open_loop(pr, horizon)
            rows.append(
                {
                    "retention_weight": beta,
                    "reference_backaction": ref_back,
                    "adaptive_with_stop": d.value,
                    "first_action": d.action,
                    "best_open_loop": fixed,
                    "best_open_loop_sequence": list(seq),
                    "gap": fixed - d.value,
                }
            )
    return rows

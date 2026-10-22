"""Reproduce the reference tables (historic R11 / R12 numbers) from the consolidated code."""

from __future__ import annotations

from dataclasses import replace

from .model import DEFAULT_ACTIONS, DEFAULT_HYPOTHESIS, TWO_HYPOTHESES, Action, Objective, Problem
from .planner import best_open_loop, evaluate_policy, greedy_policy, nominal_policy, optimal_policy, solve
from .retention import best_open_loop_retention, solve_retention

__all__ = ["single_model_table", "two_model_table", "two_model_sensitivity", "retention_table"]

RETENTION_DELTAS = (0.0, 0.01, 0.05, 0.08, 0.1, 0.13, 1.0)


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


def retention_table(horizon: int = 4, deltas=RETENTION_DELTAS) -> list[dict]:
    """Retention limit P(x != x0) <= delta, single default model (roadmap R13/P0, F29)."""
    pr = Problem((DEFAULT_HYPOTHESIS,), actions=DEFAULT_ACTIONS)
    rows = []
    for delta in deltas:
        p = solve_retention(pr, delta, horizon, "pathwise")
        e = solve_retention(pr, delta, horizon, "expected")
        fixed, seq = best_open_loop_retention(pr, horizon, delta, "pathwise")
        rows.append(
            {
                "horizon": horizon,
                "delta": delta,
                "unconstrained": p.unconstrained_value,
                "unconstrained_p_changed": p.unconstrained_p_changed,
                "unconstrained_p_violation": p.unconstrained_p_violation,
                "pathwise": p.value,
                "pathwise_first_action": p.first_actions[0],
                "pathwise_best_fixed": fixed,
                "pathwise_best_fixed_sequence": list(seq),
                "expected_randomised": e.value,
                "expected_multiplier": e.multiplier,
                "expected_mixture": [[w, a] for (w, _), a in zip(e.components, e.first_actions, strict=True)],
                "expected_deterministic": e.deterministic_value,
                "expected_duality_gap": e.duality_gap,
            }
        )
    return rows


IDENTIFIABILITY_DESIGN = ("fast", "reference", "precise", "reference", "fast", "precise")


def identifiability_scenarios():
    """(label, calibration, design, free) of the roadmap R13/P0 identifiability table."""
    from .identifiability import Calibration, Design, parameter_names

    cal = Calibration(prep_error=0.02)
    every = parameter_names(cal)
    anchored = tuple(f for f in every if f != "backaction:fast")  # fast read defines the rate unit
    target_only = ("k01", "k10", "bias_flip", "error:fast", "error:precise", "backaction:precise")
    d6 = IDENTIFIABILITY_DESIGN
    return [
        ("all free, unknown writes", cal, Design(d6), every),
        ("all free, known writes", cal, Design(d6, True), every),
        ("fast back-action fixed, unknown writes", cal, Design(d6), anchored),
        (
            "fast back-action fixed, known writes, periodic F-R-P",
            cal,
            Design(("fast", "reference", "precise") * 2, True),
            anchored,
        ),
        ("fast back-action fixed, known writes", cal, Design(d6, True), anchored),
        ("no reference read (target reads only)", cal, Design(("fast", "precise") * 2), target_only),
        ("best 4-step design (D-optimal)", cal, Design(("fast", "precise", "reference", "precise"), True), anchored),
    ]


def identifiability_table() -> list[dict]:
    """Which parameters calibration sessions can separate (roadmap R13/P0)."""
    from .identifiability import identifiability

    rows = []
    for label, cal, design, free in identifiability_scenarios():
        r = identifiability(cal, design, free)
        worst = None
        if r.locally_identifiable and None not in r.sessions_for_target:
            i = max(range(len(r.free)), key=lambda k: r.sessions_for_target[k])
            worst = [r.free[i], r.sessions_for_target[i]]
        rows.append(
            {
                "scenario": label,
                "design": list(design.sequence),
                "known_write": design.known_write,
                "free": len(r.free),
                "rank": r.rank,
                "verdict": r.verdict,
                "scope": r.scope,
                "witnesses": [w.name for w in r.witnesses],
                "zero_information": list(r.zero_information),
                "worst_parameter_sessions_for_10pct": worst,
                "sessions_for_10pct": dict(zip(r.free, r.sessions_for_target, strict=True)),
                "sessions_scope": r.sessions_scope,
            }
        )
    return rows

"""Research-only, deterministic internal consistency checks.

This checks Bellman value versus the *same implementation family* evaluated
as a policy. It is NOT an independent mathematical oracle or field validation.
"""

from __future__ import annotations

from math import isfinite

from .model import Problem
from .planner import TIE_TOL, best_open_loop, evaluate_policy, optimal_policy, solve


def self_audit(problem: Problem, horizon: int) -> dict:
    """Fail if self-consistency/dominance identities are violated.

    The internal tolerance scales with horizon and magnitude. A caller must
    independently benchmark this oracle before claiming scientific correctness.
    """
    d = solve(problem, horizon=horizon, allow_stop=True)
    st = evaluate_policy(problem, optimal_policy(problem), horizon)
    fixed, sequence = best_open_loop(problem, horizon)
    forced = solve(problem, horizon=horizon, allow_stop=False).value
    prev = solve(problem, horizon=max(0, horizon - 1)).value
    reconstructed = st.expected_terminal_loss + problem.objective.energy_lambda * st.expected_energy
    vals = (
        d.value,
        st.value,
        fixed,
        forced,
        prev,
        reconstructed,
        st.expected_steps,
        st.expected_energy,
        st.expected_terminal_loss,
    )
    if any(not isfinite(v) for v in vals):
        raise ArithmeticError("self-audit detected non-finite value")
    # Never let a huge, unrelated forced-policy value inflate the tolerance
    # for the STOP-policy comparison (an adversarial numerical blind spot).
    factor = (horizon + 2) * TIE_TOL
    tol = max(1e-11, factor * max(1.0, abs(d.value), abs(st.value), abs(reconstructed)))
    if abs(d.value - st.value) > tol:
        raise ArithmeticError("Bellman / explicit policy disagreement")
    if abs(st.value - reconstructed) > tol:
        raise ArithmeticError("policy cost decomposition disagreement")
    for bound in (fixed, forced, prev):
        bound_tol = max(1e-11, factor * max(1.0, abs(d.value), abs(bound)))
        if d.value > bound + bound_tol:
            raise ArithmeticError("Bellman dominance / horizon monotonicity violated")
    if not (0.0 <= st.expected_steps <= horizon + factor):
        raise ArithmeticError("invalid expected measurement count")
    return {
        "schema": "ail-research-self-audit-v1",
        "claim_scope": "synthetic_classical_two_state_only",
        "independent_oracle": False,
        "horizon": horizon,
        "optimal_with_stop": d.value,
        "first_action": d.action,
        "evaluated_policy_value": st.value,
        "best_open_loop": fixed,
        "best_open_loop_sequence": list(sequence),
        "forced_value": forced,
        "previous_horizon_value": prev,
        "expected_steps": st.expected_steps,
        "expected_energy": st.expected_energy,
        "expected_terminal_loss": st.expected_terminal_loss,
        "max_residual": max(abs(d.value - st.value), abs(st.value - reconstructed)),
        "tolerance": tol,
        "verdict": "internal_consistency_pass",
    }

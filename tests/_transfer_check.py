"""Shared comparison of a policy-transfer scenario against the independent oracle.

The R8/R9 transfer tests and their negative controls must use the SAME check.
A negative control that compares its own forged value inline only proves that
the forgery is larger than the tolerance; it says nothing about whether the
check the positive tests rely on would reject it. Keeping the comparison here,
and sending forged scenarios through it, closes that gap.

Imports nothing from ``atomic_information_lab``: the caller passes the frozen
nominal policy (a callable) and the oracle model.
"""

from __future__ import annotations

ABS_TOL = 3e-11
FIELDS = ("nominal_policy_under_truth", "expected_steps", "true_optimal_value")


def oracle_reference(oracle, nominal_choice, horizon: int) -> dict:
    """Independent values for one scenario: the frozen nominal policy and the true optimum."""

    def frozen_policy(_message, history, h):
        name = nominal_choice((), history, h)
        return None if name == "stop" else next(a for a in oracle.actions if a[0] == name)

    value, steps = oracle.policy_value(oracle.root(), horizon, frozen_policy)
    optimum = oracle.value(oracle.root(), horizon)
    return {
        "nominal_policy_under_truth": float(value),
        "expected_steps": float(steps),
        "true_optimal_value": float(optimum),
    }


def transfer_mismatches(measured: dict, reference: dict, tol: float = ABS_TOL) -> list[str]:
    """Names of the fields in which ``measured`` disagrees with ``reference``.

    ``not (|a - b| <= tol)`` so that NaN and a missing tolerance comparison both
    count as a mismatch (fail closed).
    """
    return [key for key in FIELDS if not abs(measured[key] - reference[key]) <= tol]

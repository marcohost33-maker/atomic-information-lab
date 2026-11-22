"""Separate outputs of a policy (roadmap R29-02).

``planner.solve`` returns one number: expected terminal loss plus priced
energy. That number mixes quantities which answer different questions and can
move in opposite directions (finding N-2: the risk can fall while nothing is
learned about the written bit). ``policy_profile`` evaluates one policy on the
full observation tree and reports each quantity on its own, exactly:

``bayes_risk``
    expected terminal loss ``E[L(b_T)]`` at the stopping time ``T``, and its
    three ingredients, unweighted: ``brier_initial = E[p0(1-p0)]``,
    ``brier_current = E[pt(1-pt)]`` and ``p_changed = P(x_T != x0)``, so that
    ``bayes_risk = alpha*brier_initial + (1-alpha)*brier_current
    + retention_weight*p_changed``.
``information_initial_bits``
    mutual information ``I(X0; D_T) = H(X0) - E[H(X0 | D_T)]`` between the
    written bit and everything recorded up to the stopping time (actions and
    readings; for a deterministic policy the actions are functions of the
    readings). In bits. It is NOT the objective the planner minimises.
``information_model_bits``
    the same for the model index, ``I(theta; D_T)`` (0 with one hypothesis).
``entropy_current_bits``
    ``E[H(X_T | D_T)]``, the remaining uncertainty about the *current* bit.
    No mutual information is reported for ``X_T``: its marginal distribution
    changes with every measurement (back-action), so "prior entropy minus
    posterior entropy" would not be a mutual information of two fixed
    variables.
``expected_energy`` / ``energy_cost`` / ``expected_steps`` / ``steps_distribution``
    energy in the abstract units of the action catalogue, its price
    ``energy_lambda * expected_energy``, the expected number of measurements
    and the distribution ``P(T = k)``.

``value = bayes_risk + energy_cost`` is what ``planner.evaluate_policy``
reports; the identity is checked on every call (``ArithmeticError`` otherwise).

Scope (``PROFILE_SCOPE``): every figure is exact within the assumed synthetic
model for the given policy and horizon, up to floating-point rounding. No
figure is a statement about an instrument.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, log2

from .model import STOP, Problem
from .planner import History, Policy, _check_horizon

__all__ = ["PROFILE_SCOPE", "PROFILE_SCOPE_TEXT", "PolicyProfile", "binary_entropy_bits", "policy_profile"]

PROFILE_SCOPE = "MODEL_INTERNAL_EXACT_FOR_THIS_POLICY"
PROFILE_SCOPE_TEXT = (
    "separate outputs of one policy (scope: MODEL_INTERNAL_EXACT_FOR_THIS_POLICY): exact within the assumed "
    "synthetic model up to rounding; mutual information is reported next to the Bayes risk, it is not the "
    "objective, and neither figure describes an instrument"
)
_IDENTITY_TOL = 1e-12  # value == bayes_risk + energy_cost, masses sum to one, information >= 0


def binary_entropy_bits(p: float) -> float:
    """``-p log2 p - (1-p) log2 (1-p)`` with ``0 log 0 = 0``; refuses values outside [0, 1]."""
    if isinstance(p, bool) or not isinstance(p, (int, float)) or not isfinite(p) or not 0.0 <= p <= 1.0:
        raise ValueError("probability must be a finite number in [0, 1]")
    if p == 0.0 or p == 1.0:
        return 0.0
    return -(p * log2(p) + (1.0 - p) * log2(1.0 - p))


def _entropy_bits(dist: Sequence[float]) -> float:
    return -sum(p * log2(p) for p in dist if p > 0.0)


@dataclass(frozen=True)
class PolicyProfile:
    horizon: int
    value: float  # bayes_risk + energy_cost (what planner.evaluate_policy reports)
    bayes_risk: float  # expected terminal loss at the stopping time
    brier_initial: float  # E[p0 (1 - p0)], unweighted
    brier_current: float  # E[pt (1 - pt)], unweighted
    p_changed: float  # P(x_T != x0) at the stopping time
    information_initial_bits: float  # I(X0; D_T)
    entropy_initial_prior_bits: float  # H(X0) under the start belief (upper bound of the line above)
    information_model_bits: float  # I(theta; D_T)
    entropy_current_bits: float  # E[H(X_T | D_T)]
    expected_energy: float  # abstract energy units
    energy_cost: float  # energy_lambda * expected_energy
    expected_steps: float
    steps_distribution: tuple[float, ...]  # P(T = k) for k = 0..horizon
    scope: str = PROFILE_SCOPE
    notes: tuple[str, ...] = (PROFILE_SCOPE_TEXT,)


def _clip_information(x: float, upper: float, what: str) -> float:
    """Information is in [0, prior entropy]; rounding may leave it outside by a few ulp, more is an error."""
    if not isfinite(x) or x < -_IDENTITY_TOL or x > upper + _IDENTITY_TOL:
        raise ArithmeticError(f"{what} outside [0, prior entropy]")
    return min(x, upper) if x > 0.0 else 0.0


def policy_profile(
    problem: Problem, policy: Policy, horizon: int, belief: Sequence[float] | None = None
) -> PolicyProfile:
    """Exact separate outputs of ``policy`` over ``horizon`` steps from ``belief`` (default: initial belief).

    ``policy(belief, history, steps_left)`` returns ``STOP`` or an action name,
    as for ``planner.evaluate_policy``; at ``steps_left == 0`` the policy is
    not asked and the process stops.
    """
    _check_horizon(horizon)
    start = problem.initial_belief() if belief is None else problem.validate_belief(belief)
    o = problem.objective
    acc = {"risk": 0.0, "b0": 0.0, "bt": 0.0, "chg": 0.0, "h0": 0.0, "hm": 0.0, "ht": 0.0, "energy": 0.0}
    stop_mass = [0.0] * (horizon + 1)

    def rec(b, hist: History, h: int, mass: float, energy: float) -> None:
        choice = policy(b, hist, h) if h > 0 else STOP
        if choice == STOP:
            m = problem.marginals(b)
            p0, pt = m["p_initial_one"], m["p_current_one"]
            acc["risk"] += mass * problem.terminal_loss(b)
            acc["b0"] += mass * p0 * (1.0 - p0)
            acc["bt"] += mass * pt * (1.0 - pt)
            acc["chg"] += mass * m["p_changed"]
            # marginals are sums of up to n_states floats: a probability may exceed 1 by an ulp
            acc["h0"] += mass * binary_entropy_bits(min(max(p0, 0.0), 1.0))
            acc["ht"] += mass * binary_entropy_bits(min(max(pt, 0.0), 1.0))
            acc["hm"] += mass * _entropy_bits(problem.model_posterior(b))
            acc["energy"] += mass * energy
            stop_mass[len(hist)] += mass
            return
        a = problem.action(choice)
        for y, p, post in problem.branches(b, a):
            rec(post, hist + ((a.name, y),), h - 1, mass * p, energy + a.energy)

    rec(start, (), horizon, 1.0, 0.0)
    if not all(isfinite(v) for v in acc.values()) or not abs(sum(stop_mass) - 1.0) <= _IDENTITY_TOL:
        raise ArithmeticError("policy profile is not finite or its stopping masses do not sum to one")
    prior = problem.marginals(start)["p_initial_one"]
    h_prior = binary_entropy_bits(min(max(prior, 0.0), 1.0))
    h_model = _entropy_bits(problem.model_posterior(start))
    info0 = _clip_information(h_prior - acc["h0"], h_prior, "information about the written bit")
    info_m = _clip_information(h_model - acc["hm"], h_model, "information about the model index")
    energy_cost = o.energy_lambda * acc["energy"]
    value = acc["risk"] + energy_cost
    parts = o.alpha * acc["b0"] + (1.0 - o.alpha) * acc["bt"] + o.retention_weight * acc["chg"]
    if not isfinite(value) or not abs(parts - acc["risk"]) <= _IDENTITY_TOL * max(1.0, abs(parts)):
        raise ArithmeticError("Bayes risk does not equal the weighted sum of its parts")
    steps = sum(k * p for k, p in enumerate(stop_mass))
    return PolicyProfile(
        horizon=horizon,
        value=value,
        bayes_risk=acc["risk"],
        brier_initial=acc["b0"],
        brier_current=acc["bt"],
        p_changed=acc["chg"],
        information_initial_bits=info0,
        entropy_initial_prior_bits=h_prior,
        information_model_bits=info_m,
        entropy_current_bits=acc["ht"],
        expected_energy=acc["energy"],
        energy_cost=energy_cost,
        expected_steps=steps,
        steps_distribution=tuple(stop_mass),
    )

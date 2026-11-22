"""Exact finite-horizon planning with an optional STOP action.

Bellman recursion over the full binary observation tree::

    V_0(b) = L(b)
    V_h(b) = min( L(b),  min_a [ lambda*E_a + sum_y P(y|b,a) * V_{h-1}(tau(b,a,y)) ] )

With ``allow_stop=False`` the STOP option is removed for ``h > 0`` (exactly
``h`` measurements, the historic R10 "forced" planner).

Ties: options whose values differ by at most ``TIE_TOL`` are treated as equal
and resolved deterministically -- STOP first, then actions in catalogue order.
This makes the chosen policy independent of last-bit floating point noise; the
reported value can exceed the exact optimum by at most ``horizon * TIE_TOL``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import product
from math import isfinite

from .model import STOP, Action, Problem

__all__ = [
    "MAX_HORIZON",
    "TIE_TOL",
    "Decision",
    "PolicyStats",
    "solve",
    "evaluate_policy",
    "optimal_policy",
    "greedy_policy",
    "open_loop_value",
    "best_open_loop",
    "nominal_policy",
]

MAX_HORIZON = 5
TIE_TOL = 1e-12

History = tuple[tuple[str, int], ...]
Policy = Callable[[tuple[float, ...], History, int], str]


@dataclass(frozen=True)
class Decision:
    value: float
    action: str  # STOP or an action name


@dataclass(frozen=True)
class PolicyStats:
    value: float  # expected terminal loss + expected energy cost
    expected_steps: float
    expected_energy: float
    expected_terminal_loss: float


def _check_horizon(horizon: int) -> None:
    if type(horizon) is not int or not 0 <= horizon <= MAX_HORIZON:
        raise ValueError(f"horizon must be an int in 0..{MAX_HORIZON}")


def _pick(options: Sequence[tuple[float, str]]) -> Decision:
    """Return a finite minimum; never silently ignore NaN or overflowed options."""
    if not options or any(not isfinite(v) for v, _ in options):
        raise ArithmeticError("non-finite or empty Bellman choice set")
    best_v, best_a = options[0]
    for v, a in options[1:]:
        if v < best_v - TIE_TOL:
            best_v, best_a = v, a
    return Decision(best_v, best_a)


def _action_value(problem: Problem, belief, action: Action, horizon: int, allow_stop: bool) -> float:
    value = problem.step_cost(action)
    for _, p, post in problem.branches(belief, action):
        value += p * _solve(problem, post, horizon - 1, allow_stop).value
        if not isfinite(value):
            raise ArithmeticError("non-finite Bellman recursion value")
    return value


def _solve(problem: Problem, belief, horizon: int, allow_stop: bool) -> Decision:
    if horizon == 0:
        return Decision(problem.terminal_loss(belief), STOP)
    options = [(problem.terminal_loss(belief), STOP)] if allow_stop else []
    options += [(_action_value(problem, belief, a, horizon, allow_stop), a.name) for a in problem.actions]
    return _pick(options)


def solve(
    problem: Problem, belief: Sequence[float] | None = None, horizon: int = 3, *, allow_stop: bool = True
) -> Decision:
    """Optimal expected loss and first decision from ``belief`` (default: initial belief)."""
    _check_horizon(horizon)
    if not isinstance(allow_stop, bool):
        raise ValueError("allow_stop must be bool")
    belief = problem.initial_belief() if belief is None else problem.validate_belief(belief)
    return _solve(problem, belief, horizon, allow_stop)


def evaluate_policy(
    problem: Problem, policy: Policy, horizon: int, belief: Sequence[float] | None = None
) -> PolicyStats:
    """Exact expected value of an arbitrary (history-dependent) policy."""
    _check_horizon(horizon)
    belief = problem.initial_belief() if belief is None else problem.validate_belief(belief)

    def rec(b, hist: History, h: int) -> tuple[float, float, float, float]:
        choice = policy(b, hist, h) if h > 0 else STOP
        if choice == STOP:
            loss = problem.terminal_loss(b)
            return loss, 0.0, 0.0, loss
        a = problem.action(choice)
        v, s, e, t = problem.step_cost(a), 1.0, a.energy, 0.0
        for y, p, post in problem.branches(b, a):
            cv, cs, ce, ct = rec(post, hist + ((a.name, y),), h - 1)
            v += p * cv
            s += p * cs
            e += p * ce
            t += p * ct
        return v, s, e, t

    values = rec(belief, (), horizon)
    if any(not isfinite(v) for v in values):
        raise ArithmeticError("non-finite policy evaluation")
    return PolicyStats(*values)


def optimal_policy(problem: Problem, *, allow_stop: bool = True) -> Policy:
    def choose(b, hist, h):
        return _solve(problem, b, h, allow_stop).action

    return choose


def greedy_policy(problem: Problem) -> Policy:
    """Myopic rule: compare STOP with every one-step lookahead, then repeat."""

    def choose(b, hist, h):
        options = [(problem.terminal_loss(b), STOP)]
        for a in problem.actions:
            v = problem.step_cost(a) + sum(p * problem.terminal_loss(post) for _, p, post in problem.branches(b, a))
            options.append((v, a.name))
        return _pick(options).action

    return choose


def open_loop_value(problem: Problem, sequence: Sequence[str], belief: Sequence[float] | None = None) -> float:
    """Value of measuring the fixed sequence (no adaptation), then STOP."""
    seq = tuple(sequence)
    _check_horizon(len(seq))
    belief = problem.initial_belief() if belief is None else problem.validate_belief(belief)
    acts = tuple(problem.action(n) for n in seq)

    def rec(b, k):
        if k == len(acts):
            return problem.terminal_loss(b)
        a = acts[k]
        return problem.step_cost(a) + sum(p * rec(post, k + 1) for _, p, post in problem.branches(b, a))

    value = rec(belief, 0)
    if not isfinite(value):
        raise ArithmeticError("non-finite open-loop evaluation")
    return value


def best_open_loop(
    problem: Problem, horizon: int, belief: Sequence[float] | None = None
) -> tuple[float, tuple[str, ...]]:
    """Best fixed sequence of length 0..horizon (shorter preferred on ties)."""
    _check_horizon(horizon)
    best = (open_loop_value(problem, (), belief), ())
    for n in range(1, horizon + 1):
        for seq in product((a.name for a in problem.actions), repeat=n):
            v = open_loop_value(problem, seq, belief)
            if v < best[0] - TIE_TOL:
                best = (v, seq)
    return best


def nominal_policy(nominal: Problem) -> Policy:
    """Act optimally for a *nominal* single model, updating its own belief from the history.

    Evaluate it under a different (true) problem with ``evaluate_policy`` to get
    the Bayes risk of model misspecification. The nominal problem must share
    action names with the problem it is evaluated under.
    """

    def choose(_b, hist, h):
        nb = nominal.initial_belief()
        for name, y in hist:
            match = [post for yy, _, post in nominal.branches(nb, nominal.action(name)) if yy == y]
            if not match:
                raise ArithmeticError("observation impossible under the nominal model")
            nb = match[0]
        return _solve(nominal, nb, h, True).action

    return choose

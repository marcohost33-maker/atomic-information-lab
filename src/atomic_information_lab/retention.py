"""Planning under a hard retention limit ``P(x != x0) <= delta`` (roadmap R13/P0, formula F29).

The R12 objective prices retention with a weight ``beta``; a limit is a
different statement: some risk of having corrupted the stored bit is simply
not acceptable. Two readings are kept apart, because they answer different
questions and have different optimal policies:

``pathwise`` -- every time the policy stops, the posterior probability that
    the bit has changed, ``P(x != x0 | data)``, is at most ``delta``. A
    measurement is admissible only if every outcome still leaves a way to stop
    within the limit; the planner never walks into a state it cannot leave
    legally. Deterministic policies are optimal.

``expected`` -- the probability over all outcomes, ``P(x_T != x0)`` at the
    stopping time, is at most ``delta`` (formula F29). This is a constrained
    MDP with one constraint: its optimum is a deterministic policy or a mix of
    two (Beutler & Ross 1985), found through the Lagrangian
    ``L(b) + mu * P(x != x0 | b)`` -- which is the R12 retention penalty with
    weight ``beta + mu``. The multiplier ``mu*`` is located exactly by a
    breakpoint search on the finite set of deterministic policies; the result
    carries its own certificate (dual bound and duality gap). The mix draws
    one of the two policies once, before the first reading.

Randomisation is not always acceptable. For the expected mode the result
therefore also carries the best DETERMINISTIC policy meeting the limit, found
exactly by dynamic programming over Pareto fronts of (risk, cost) pairs
(``best_deterministic_retention``); a constrained MDP generally has a duality
gap over deterministic policies, so this value can be strictly worse than the
mixture and is not what the Lagrangian alone would pick.

Every result also reports what the unconstrained optimum would have done --
its value and its retention risk -- so the cost of the limit and the risk an
unconstrained figure hides are both visible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from math import isfinite

from .model import STOP, Problem
from .planner import TIE_TOL, Policy, _check_horizon

__all__ = [
    "RETENTION_MODES",
    "RetentionResult",
    "solve_retention",
    "pathwise_policy",
    "lagrangian_policy",
    "retention_profile",
    "best_open_loop_retention",
    "best_deterministic_retention",
]

RETENTION_MODES = ("pathwise", "expected")
_MAX_BREAKPOINT_STEPS = 500
_MU_CAP = 1e12


@dataclass(frozen=True)
class RetentionResult:
    mode: str
    delta: float
    horizon: int
    value: float  # expected terminal loss + energy cost of the returned (possibly mixed) policy
    p_changed: float  # P(x != x0) at the stopping time under that policy
    first_actions: tuple[str, ...]  # one entry per mixture component
    components: tuple[tuple[float, float], ...]  # (weight, multiplier the component is optimal for)
    multiplier: float | None  # mu* for "expected"; None for "pathwise"
    dual_bound: float | None  # lower bound on every policy meeting the limit ("expected" only)
    duality_gap: float | None
    deterministic_value: float  # best DETERMINISTIC policy meeting the limit (exact, Pareto-front DP)
    deterministic_p_changed: float
    deterministic_first_action: str
    unconstrained_value: float
    unconstrained_p_changed: float
    unconstrained_p_violation: float  # probability that the unconstrained optimum stops above delta


def _check_delta(delta: float) -> float:
    if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not isfinite(delta) or not 0 <= delta <= 1:
        raise ValueError("delta must be a finite number in [0, 1]")
    return float(delta)


def _changed(problem: Problem, belief) -> float:
    return problem.marginals(belief)["p_changed"]


def _better(v: float, best: float) -> bool:
    return v < best - TIE_TOL  # same tie rule as the planner: earlier option wins within TIE_TOL


# ---------------------------------------------------------------- Lagrangian (expected mode)
def _lagrangian(problem: Problem, belief, h: int, mu: float) -> tuple[float, float, float, str]:
    """(min of L + mu*changed, its loss+energy part, its changed part, first decision)."""
    loss, pc = problem.terminal_loss(belief), _changed(problem, belief)
    best = (loss + mu * pc, loss, pc, STOP)
    if h == 0:
        return best
    for a in problem.actions:
        cost = problem.step_cost(a)
        lv, jv, cv = cost, cost, 0.0
        for _, p, post in problem.branches(belief, a):
            cl, cj, cc, _ = _lagrangian(problem, post, h - 1, mu)
            lv, jv, cv = lv + p * cl, jv + p * cj, cv + p * cc
        if not (isfinite(lv) and isfinite(jv) and isfinite(cv)):
            raise ArithmeticError("non-finite Lagrangian recursion value")
        if _better(lv, best[0]):
            best = (lv, jv, cv, a.name)
    return best


def lagrangian_policy(problem: Problem, mu: float) -> Policy:
    """Deterministic policy minimising ``loss + energy + mu * P(x != x0)`` (R12 penalty with weight +mu)."""
    if not isfinite(mu) or mu < 0:
        raise ValueError("mu must be finite and >= 0")

    def choose(b, hist, h):
        return _lagrangian(problem, b, h, mu)[3]

    return choose


# ---------------------------------------------------------------- pathwise mode
def _pathwise(problem: Problem, belief, h: int, delta: float) -> tuple[float, float, str] | None:
    """(value, changed, first decision) of the best policy that stops only within the limit; None if none."""
    best = None
    pc = _changed(problem, belief)
    if pc <= delta:
        best = (problem.terminal_loss(belief), pc, STOP)
    if h == 0:
        return best
    for a in problem.actions:
        v, c = problem.step_cost(a), 0.0
        for _, p, post in problem.branches(belief, a):
            child = _pathwise(problem, post, h - 1, delta)
            if child is None:
                break  # some outcome would trap the policy above the limit
            v, c = v + p * child[0], c + p * child[1]
        else:
            if not (isfinite(v) and isfinite(c)):
                raise ArithmeticError("non-finite constrained recursion value")
            if best is None or _better(v, best[0]):
                best = (v, c, a.name)
    return best


def pathwise_policy(problem: Problem, delta: float) -> Policy:
    """Best policy whose every STOP satisfies ``P(x != x0 | data) <= delta``."""
    delta = _check_delta(delta)

    def choose(b, hist, h):
        r = _pathwise(problem, b, h, delta)
        if r is None:
            raise ArithmeticError("no admissible decision: the retention limit cannot be met from here")
        return r[2]

    return choose


def _min_changed(problem: Problem, belief, h: int) -> float:
    """Smallest achievable P(x_T != x0) from ``belief`` (feasibility of the expected limit)."""
    best = _changed(problem, belief)
    if h == 0:
        return best
    for a in problem.actions:
        c = sum(p * _min_changed(problem, post, h - 1) for _, p, post in problem.branches(belief, a))
        best = min(best, c)
    return best


# ---------------------------------------------------------------- best deterministic (expected mode)
def _prune(points):
    """Keep the Pareto front of (changed, value, plan): no other point is lower in both.

    A point within TIE_TOL of a kept one is dropped, so the front stays small;
    this can cost at most TIE_TOL per level in value.
    """
    front, best_j = [], None
    for c, j, plan in sorted(points, key=lambda t: (t[0], t[1])):
        if best_j is None or j < best_j - TIE_TOL:
            front.append((c, j, plan))
            best_j = j
    return front


def _front(problem: Problem, belief, h: int):
    """Pareto front over all deterministic policies from ``belief``; plan = None (STOP) or
    (action name, ((y, sub-plan), ...))."""
    points = [(_changed(problem, belief), problem.terminal_loss(belief), None)]
    if h > 0:
        for a in problem.actions:
            partial = [(0.0, problem.step_cost(a), ())]
            for y, p, post in problem.branches(belief, a):
                child = _front(problem, post, h - 1)
                partial = _prune(
                    [
                        (c0 + p * c1, j0 + p * j1, plan0 + ((y, plan1),))
                        for c0, j0, plan0 in partial
                        for c1, j1, plan1 in child
                    ]
                )
            points += [(c, j, (a.name, plan)) for c, j, plan in partial]
    front = _prune(points)
    if not all(isfinite(c) and isfinite(j) for c, j, _ in front):
        raise ArithmeticError("non-finite Pareto front")
    return front


def _plan_policy(root_plan) -> Policy:
    def choose(_b, hist, _h):
        plan = root_plan
        for name, y in hist:
            if plan is None or plan[0] != name:
                raise ArithmeticError("history does not follow the deterministic plan")
            plan = dict(plan[1])[y]
        return STOP if plan is None else plan[0]

    return choose


def best_deterministic_retention(
    problem: Problem, delta: float, horizon: int, belief: Sequence[float] | None = None
) -> tuple[float, float, Policy, int]:
    """Best deterministic policy with ``P(x_T != x0) <= delta``: (value, changed, policy, front size).

    Exact over all deterministic policies (up to TIE_TOL per level); raises
    ``ValueError`` if none meets the limit.
    """
    _check_horizon(horizon)
    delta = _check_delta(delta)
    belief = problem.initial_belief() if belief is None else problem.validate_belief(belief)
    front = _front(problem, belief, horizon)
    feasible = [(j, c, plan) for c, j, plan in front if c <= delta]
    if not feasible:
        raise ValueError("infeasible: no deterministic policy keeps P(x_T != x0) <= delta")
    j, c, plan = min(feasible, key=lambda t: t[0])
    return j, c, _plan_policy(plan), len(front)


# ---------------------------------------------------------------- evaluation
def retention_profile(
    problem: Problem, policy: Policy, horizon: int, delta: float, belief: Sequence[float] | None = None
) -> dict:
    """Exact value, P(x_T != x0) and probability of stopping above ``delta`` for any policy."""
    _check_horizon(horizon)
    delta = _check_delta(delta)
    belief = problem.initial_belief() if belief is None else problem.validate_belief(belief)

    def rec(b, hist, h):
        choice = policy(b, hist, h) if h > 0 else STOP
        if choice == STOP:
            pc = _changed(problem, b)
            return problem.terminal_loss(b), pc, float(pc > delta)
        a = problem.action(choice)
        v, c, viol = problem.step_cost(a), 0.0, 0.0
        for y, p, post in problem.branches(b, a):
            cv, cc, cx = rec(post, hist + ((a.name, y),), h - 1)
            v, c, viol = v + p * cv, c + p * cc, viol + p * cx
        return v, c, viol

    value, changed, violation = rec(belief, (), horizon)
    if not all(isfinite(x) for x in (value, changed, violation)):
        raise ArithmeticError("non-finite retention profile")
    return {"value": value, "p_changed": changed, "p_violation": violation}


# ---------------------------------------------------------------- solver
def solve_retention(
    problem: Problem,
    delta: float,
    horizon: int,
    mode: str = "pathwise",
    belief: Sequence[float] | None = None,
) -> RetentionResult:
    """Optimal planning under the retention limit ``delta`` (see the module docstring for the two modes).

    Raises ``ValueError`` if no policy can meet the limit from ``belief``
    (never from the initial belief, where ``x == x0`` and STOP is admissible).
    """
    _check_horizon(horizon)
    delta = _check_delta(delta)
    if mode not in RETENTION_MODES:
        raise ValueError(f"mode must be one of {RETENTION_MODES}")
    belief = problem.initial_belief() if belief is None else problem.validate_belief(belief)

    _, j0, c0, a0 = _lagrangian(problem, belief, horizon, 0.0)
    plain = retention_profile(problem, lagrangian_policy(problem, 0.0), horizon, delta, belief)
    common = {
        "mode": mode,
        "delta": delta,
        "horizon": horizon,
        "unconstrained_value": j0,
        "unconstrained_p_changed": c0,
        "unconstrained_p_violation": plain["p_violation"],
    }

    if mode == "pathwise":
        r = _pathwise(problem, belief, horizon, delta)
        if r is None:
            raise ValueError("infeasible: no policy keeps P(x != x0 | data) <= delta at every stop")
        return RetentionResult(
            value=r[0],
            p_changed=r[1],
            first_actions=(r[2],),
            components=((1.0, 0.0),),
            multiplier=None,
            dual_bound=None,
            duality_gap=None,
            deterministic_value=r[0],
            deterministic_p_changed=r[1],
            deterministic_first_action=r[2],
            **common,
        )

    if _min_changed(problem, belief, horizon) > delta:
        raise ValueError("infeasible: no policy keeps P(x_T != x0) <= delta")
    if c0 <= delta:  # the limit is not binding
        return RetentionResult(
            value=j0,
            p_changed=c0,
            first_actions=(a0,),
            components=((1.0, 0.0),),
            multiplier=0.0,
            dual_bound=j0,
            duality_gap=0.0,
            deterministic_value=j0,  # the unconstrained optimum is deterministic and feasible
            deterministic_p_changed=c0,
            deterministic_first_action=a0,
            **common,
        )

    lo = (0.0, j0, c0, a0)  # (mu, J, C, first action) with C > delta
    mu = 1.0
    while True:
        _, j, c, a = _lagrangian(problem, belief, horizon, mu)
        if c <= delta:
            hi = (mu, j, c, a)
            break
        lo = (mu, j, c, a)
        mu *= 4.0
        if mu > _MU_CAP:
            raise ArithmeticError("retention multiplier search did not bracket the limit")

    tol = 4 * (horizon + 1) * TIE_TOL
    for _ in range(_MAX_BREAKPOINT_STEPS):
        # Lines J + mu*C of the two bracketing policies cross at mu_x (lo_mu <= mu_x <= hi_mu).
        mu_x = (hi[1] - lo[1]) / (lo[2] - hi[2])
        lx, jx, cx, ax = _lagrangian(problem, belief, horizon, mu_x)
        line = lo[1] + mu_x * lo[2]
        if lx >= line - tol * max(1.0, abs(line), mu_x):
            break  # nothing beats both lines at mu_x: mu_x is the breakpoint mu*
        if cx > delta:
            lo = (mu_x, jx, cx, ax)
        else:
            hi = (mu_x, jx, cx, ax)
    else:
        raise ArithmeticError("retention breakpoint search did not converge")

    q = (delta - hi[2]) / (lo[2] - hi[2])  # weight of the policy above the limit
    value = q * lo[1] + (1.0 - q) * hi[1]
    changed = q * lo[2] + (1.0 - q) * hi[2]
    dual = lx - mu_x * delta
    gap = value - dual
    if not all(isfinite(x) for x in (q, value, changed, dual, gap)) or not -1e-12 <= q <= 1 + 1e-12:
        raise ArithmeticError("invalid retention mixture")
    if gap < -tol * max(1.0, abs(value), mu_x):
        raise ArithmeticError("dual bound exceeds the primal value: weak duality violated")
    det_value, det_changed, det_policy, _ = best_deterministic_retention(problem, delta, horizon, belief)
    if det_value < value - tol * max(1.0, abs(value)):
        raise ArithmeticError("a deterministic policy beats the randomised optimum: mixture search is wrong")
    parts = [(q, lo[0], lo[3]), (1.0 - q, hi[0], hi[3])]
    parts = [p for p in parts if p[0] > 1e-15] or [parts[1]]
    return RetentionResult(
        value=value,
        p_changed=changed,
        first_actions=tuple(p[2] for p in parts),
        components=tuple((p[0], p[1]) for p in parts),
        multiplier=mu_x,
        dual_bound=dual,
        duality_gap=gap,
        deterministic_value=det_value,
        deterministic_p_changed=det_changed,
        deterministic_first_action=det_policy(belief, (), horizon),
        **common,
    )


def best_open_loop_retention(
    problem: Problem, horizon: int, delta: float, mode: str = "pathwise"
) -> tuple[float, tuple[str, ...]]:
    """Best FIXED reading sequence (length 0..horizon) that meets the limit -- the baseline the
    roadmap asks for. Deterministic sequences only, also in the expected mode."""
    _check_horizon(horizon)
    delta = _check_delta(delta)
    if mode not in RETENTION_MODES:
        raise ValueError(f"mode must be one of {RETENTION_MODES}")

    def rec(b, acts):
        if not acts:
            pc = _changed(problem, b)
            return problem.terminal_loss(b), pc, pc
        a = acts[0]
        v, c, worst = problem.step_cost(a), 0.0, 0.0
        for _, p, post in problem.branches(b, a):
            cv, cc, cw = rec(post, acts[1:])
            v, c, worst = v + p * cv, c + p * cc, max(worst, cw)
        return v, c, worst

    root = problem.initial_belief()
    best = None
    for n in range(horizon + 1):
        for seq in product(problem.actions, repeat=n):
            v, c, worst = rec(root, seq)
            ok = worst <= delta if mode == "pathwise" else c <= delta
            if ok and (best is None or _better(v, best[0])):
                best = (v, tuple(a.name for a in seq))
    return best  # never None: the empty sequence stops at the root, where x == x0

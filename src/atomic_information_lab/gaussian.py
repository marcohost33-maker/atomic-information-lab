"""Linear-Gaussian analogue of the measurement model: Kalman filter, STOP, look-ahead (roadmap R13/P0).

The continuous line of the earlier research rounds (R8/R9: Gaussian drift, reference slots,
online myopic selection) was left out of 0.1.0 because it had no exact oracle. This module
brings it back in the one form that *has* an exact answer.

Model (all quantities dimensionless; synthetic, no instrument)
--------------------------------------------------------------
Hidden state ``s = (x0, x, b)``:

* ``x0`` -- the value that was written (never changes),
* ``x``  -- the current value of the stored quantity,
* ``b``  -- a slowly drifting sensor offset.

Prior: ``x0 ~ N(0, prior_var)``, ``x = x0``, ``b ~ N(0, bias_prior_var)`` independent of ``x0``.
One measurement with action ``a``:

1. ``x <- x + w``,  ``w ~ N(0, drift_var * backaction(a))``  (back-action speeds up the drift),
2. ``b <- b + u``,  ``u ~ N(0, bias_drift_var)``,
3. ``y = x + b + v`` for a target read, ``y = b + v`` for a read of a known reference,
   ``v ~ N(0, noise_var(a))``.

Terminal loss (expected squared error of the posterior-mean estimates, plus a retention term)::

    L = alpha * Var(x0 | D) + (1 - alpha) * Var(x | D) + retention_weight * E[(x - x0)^2]

``E[(x - x0)^2] = drift_var * sum(backaction)`` is the damage the readings did to the stored
value, averaged over the data (the expected retention term of the discrete planner). Its
posterior counterpart ``E[(x - x0)^2 | D] = Var(x - x0 | D) + E[x - x0 | D]^2``
(``conditional_damage``) depends on the data; its data average is the damage above. Every
reading costs ``energy_lambda * energy``.

Why this line has an exact oracle -- and where that stops
--------------------------------------------------------
Three conditions together: (i) linear-Gaussian dynamics and readings, (ii) known parameters,
(iii) a terminal loss that depends on the data only through the posterior covariance (here:
posterior variances and the data-averaged damage). Then the Kalman covariance does not depend
on the observed values, the conditional value of every continuation is the same for every data
set, and the Bayes-optimal *adaptive* policy with STOP is an open-loop plan "measure
``a_1 .. a_n``, then stop", found by a deterministic Bellman recursion over covariances
(measurement control can be optimised a priori: Meier, Peschon & Dressler 1967,
doi:10.1109/TAC.1967.1098668; no dual effect: Bar-Shalom & Tse 1974,
doi:10.1109/TAC.1974.1100635). Under these conditions the value of adaptivity is exactly zero
(finding N-47). Drop (iii) and it returns: with a 0-1 loss for the sign of ``x0`` the
terminal risk ``Phi(-|m| / sqrt(v))`` depends on the posterior mean, and a stopping rule that
looks at the data beats every fixed plan (finding N-50). A retention limit on the posterior
damage ``conditional_damage`` would also be data-dependent; the limits implemented here are on
the data-averaged damage only. Dropping (ii) (unknown variances) breaks the linearity of the
posterior.

A limit on the data-averaged damage ``E[(x - x0)^2] <= delta`` is a constraint on an
*expectation*, and it changes the picture (R35/R35b, Codex review of PR #8, findings N-52,
N-53). Every policy -- randomised or data-dependent -- induces a distribution over the reading
sequences it carries out, and its expected value and expected damage are the matching averages
of the plan values ``V(s)`` and plan damages ``D(s)`` (the posterior variances do not depend on
the data, and a reading's drift has mean zero given everything decided before it). The
achievable ``(damage, value)`` pairs are therefore the convex hull of the plan points, and the
optimum under the limit is the lower convex hull at ``delta``: one plan or a mix of two (a
linear programme over plan weights with one equality row, the weights sum to one, and one
inequality row, the damage, has a basic optimum with at most two non-zero weights; the
infinite-horizon, time-average analogue for constrained Markov chains is Beutler & Ross
1985, doi:10.1016/0022-247X(85)90288-4). Three nested policy classes, all solved exactly by
``solve_gaussian_retention`` (default model, suprema over ``delta`` for ``h >= 2``):

* best single plan (``solve_gaussian(..., retention_limit=delta)``): up to 52.76 % above the
  optimum (left limit at ``delta -> 0.02``);
* best deterministic policy: the first reading is fixed, but from then on the standardised
  innovation of a reading is a uniform draw supplied by the data, so the policy can mix any
  continuations that share that first reading -- the best over first readings of the hull of the
  plans starting with it (N-53). Up to 41.63 % better than the best single plan (left limit at
  ``delta -> 0.031``), so under an expected limit a data-dependent rule beats every fixed plan;
* randomised optimum: what an external coin before the first reading adds over the best
  deterministic policy -- up to 29.03 % (``delta ~ 0.01303``). When both hull plans share their
  first reading, the two coincide. Against randomised open-loop plans, adaptivity is worth
  exactly zero here as well (the hull is reached by both).

The independent oracle ``tests/_gaussian_oracle.py`` recomputes every posterior in exact
rational arithmetic by batch least squares (information form over the noise sequence), sharing
no code with this file. Cost of the exact search: ``sum_k |A|^k`` covariance updates
(1 092 for three actions and ``h = 6``, about 0.03 s; ``h = 10``: about 2.6 s).

Every figure carries ``scope = GAUSSIAN_SCOPE``: exact inside the assumed linear-Gaussian
model, up to floating-point rounding; model error is not quantified.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import isfinite

from .model import STOP

__all__ = [
    "GAUSSIAN_SCOPE",
    "MAX_GAUSSIAN_HORIZON",
    "GaussianAction",
    "GaussianModel",
    "GaussianObjective",
    "GaussianState",
    "GaussianPlan",
    "GaussianRetentionResult",
    "DEFAULT_GAUSSIAN_ACTIONS",
    "DEFAULT_GAUSSIAN_MODEL",
    "initial_state",
    "predict",
    "update",
    "terminal_loss",
    "conditional_damage",
    "solve_gaussian",
    "solve_gaussian_retention",
    "gaussian_retention_frontier",
    "innovation",
    "plan_value",
    "lookahead_plan",
    "reference_plan",
    "model_reference_plan",
    "fixed_plan",
    "best_fixed_plan",
    "gaussian_table",
]

GAUSSIAN_SCOPE = "MODEL_INTERNAL_EXACT_LINEAR_GAUSSIAN"
MAX_GAUSSIAN_HORIZON = 10
TIE_TOL = 1e-12

Matrix = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]


def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v) for v in values)


@dataclass(frozen=True)
class GaussianAction:
    """A measurement option of the linear-Gaussian model."""

    name: str
    backaction: float
    noise_var: float
    energy: float
    reads_reference: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name == STOP:
            raise ValueError("action name must be a non-empty string other than 'stop'")
        if not _finite(self.backaction, self.noise_var, self.energy):
            raise ValueError(f"action {self.name!r}: parameters must be finite real numbers")
        # noise_var > 0: a noiseless reading makes the innovation variance singular for a
        # degenerate prior; refused instead of special-cased.
        if self.backaction < 0 or self.energy < 0 or not self.noise_var > 0:
            raise ValueError(f"action {self.name!r}: need backaction>=0, energy>=0, noise_var>0")
        if not isinstance(self.reads_reference, bool):
            raise ValueError("reads_reference must be bool")


@dataclass(frozen=True)
class GaussianObjective:
    alpha: float = 0.5
    energy_lambda: float = 0.012
    retention_weight: float = 0.0

    def __post_init__(self) -> None:
        if not _finite(self.alpha, self.energy_lambda, self.retention_weight):
            raise ValueError("objective parameters must be finite real numbers")
        if not 0 <= self.alpha <= 1 or self.energy_lambda < 0 or self.retention_weight < 0:
            raise ValueError("need 0<=alpha<=1, energy_lambda>=0, retention_weight>=0")


DEFAULT_GAUSSIAN_ACTIONS: tuple[GaussianAction, ...] = (
    GaussianAction("fast", 1.0, 0.1, 1.0),
    GaussianAction("precise", 3.5, 0.03, 2.8),
    GaussianAction("reference", 0.55, 0.025, 0.8, reads_reference=True),
)


@dataclass(frozen=True)
class GaussianModel:
    """Variances of prior and drifts, action catalogue and objective.

    Defaults mirror the discrete model: prior variance 1/4 (the variance of a fair bit, so the
    loss before any reading is 0.25 in both models), same back-action factors and energies.
    """

    prior_var: float = 0.25
    bias_prior_var: float = 0.25
    drift_var: float = 0.02
    bias_drift_var: float = 0.01
    actions: tuple[GaussianAction, ...] = DEFAULT_GAUSSIAN_ACTIONS
    objective: GaussianObjective = field(default_factory=GaussianObjective)

    def __post_init__(self) -> None:
        if not _finite(self.prior_var, self.bias_prior_var, self.drift_var, self.bias_drift_var):
            raise ValueError("variances must be finite real numbers")
        if min(self.prior_var, self.bias_prior_var, self.drift_var, self.bias_drift_var) < 0:
            raise ValueError("variances must be non-negative")
        actions = tuple(self.actions)
        object.__setattr__(self, "actions", actions)
        if not 1 <= len(actions) <= 4 or any(type(a) is not GaussianAction for a in actions):
            raise ValueError("need 1..4 GaussianAction objects")
        if len({a.name for a in actions}) != len(actions):
            raise ValueError("action names must be unique")
        if type(self.objective) is not GaussianObjective:
            raise ValueError("objective must be a GaussianObjective")
        for a in actions:
            if not isfinite(self.drift_var * a.backaction):
                raise ValueError(f"drift_var * backaction overflows for action {a.name!r}")

    def action(self, name: str) -> GaussianAction:
        for a in self.actions:
            if a.name == name:
                return a
        raise ValueError(f"unknown action {name!r}")


DEFAULT_GAUSSIAN_MODEL = GaussianModel()


_PSD_TOL = 1e-12


@dataclass(frozen=True)
class GaussianState:
    """Posterior ``N(mean, cov)`` over ``(x0, x, b)`` plus the accumulated drift variance ``E[(x-x0)^2]``.

    Validated on construction (fail-closed): a finite 3-vector, a finite symmetric positive
    semi-definite 3x3 matrix (every principal minor >= -1e-12 relative to the scale of the
    diagonal), and ``damage >= 0``. Sequences are stored as tuples.
    """

    mean: tuple[float, float, float]
    cov: Matrix
    damage: float = 0.0

    def __post_init__(self) -> None:
        try:
            mean = tuple(self.mean)
            cov = tuple(tuple(r) for r in self.cov)
        except TypeError as exc:
            raise ValueError("state must hold a 3-vector and a 3x3 matrix") from exc
        if len(mean) != 3 or len(cov) != 3 or any(len(r) != 3 for r in cov):
            raise ValueError("state must hold a 3-vector and a 3x3 matrix")
        if not _finite(*mean, *(c for r in cov for c in r), self.damage):
            raise ValueError("state entries must be finite real numbers")
        if self.damage < 0:
            raise ValueError("damage must be >= 0")
        scale = max(abs(cov[i][i]) for i in range(3))
        if any(abs(cov[i][j] - cov[j][i]) > _PSD_TOL * max(scale, 1.0) for i in range(3) for j in range(i)):
            raise ValueError("covariance must be symmetric")
        tol = _PSD_TOL
        minors1 = [cov[i][i] for i in range(3)]
        minors2 = [cov[i][i] * cov[j][j] - cov[i][j] * cov[j][i] for i in range(3) for j in range(i + 1, 3)]
        det = (
            cov[0][0] * (cov[1][1] * cov[2][2] - cov[1][2] * cov[2][1])
            - cov[0][1] * (cov[1][0] * cov[2][2] - cov[1][2] * cov[2][0])
            + cov[0][2] * (cov[1][0] * cov[2][1] - cov[1][1] * cov[2][0])
        )
        if (
            min(minors1) < -tol * scale
            or min(minors2) < -tol * scale**2
            or det < -tol * scale**3
            or (scale == 0 and min(minors1) < 0)
        ):
            raise ValueError("covariance must be positive semi-definite")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "cov", cov)


def initial_state(model: GaussianModel) -> GaussianState:
    v, vb = model.prior_var, model.bias_prior_var
    return GaussianState((0.0, 0.0, 0.0), ((v, v, 0.0), (v, v, 0.0), (0.0, 0.0, vb)), 0.0)


def _check_state(state: GaussianState) -> None:
    # contents are validated by GaussianState itself; only the type is checked here
    if type(state) is not GaussianState:
        raise ValueError("state must be a GaussianState")


def predict(model: GaussianModel, state: GaussianState, action: GaussianAction) -> GaussianState:
    """Prior of the next reading: add drift (scaled by back-action) and offset drift."""
    _check_state(state)
    q = model.drift_var * action.backaction
    p = [list(r) for r in state.cov]
    p[1][1] += q
    p[2][2] += model.bias_drift_var
    cov = tuple(tuple(r) for r in p)
    return GaussianState(state.mean, cov, state.damage + q)  # type: ignore[arg-type]


def _h(action: GaussianAction) -> tuple[float, float, float]:
    return (0.0, 0.0, 1.0) if action.reads_reference else (0.0, 1.0, 1.0)


def update(model: GaussianModel, state: GaussianState, action: GaussianAction, y: float | None = None) -> GaussianState:
    """One Kalman step (predict, then condition on ``y``).

    With ``y=None`` only the covariance is updated (the mean is left at its prediction); the
    covariance never depends on ``y``. Joseph form keeps the covariance symmetric and
    positive semi-definite under rounding.
    """
    if action not in model.actions:
        raise ValueError(f"action {action!r} not in this model")
    if y is not None and not _finite(y):
        raise ValueError("observation must be a finite real number")
    pr = predict(model, state, action)
    h = _h(action)
    P = pr.cov
    ph = tuple(sum(P[i][j] * h[j] for j in range(3)) for i in range(3))
    s = sum(h[i] * ph[i] for i in range(3)) + action.noise_var
    if not (isfinite(s) and s > 0):
        raise ArithmeticError("innovation variance is not positive and finite")
    k = tuple(v / s for v in ph)
    # Joseph form: (I - K h) P (I - K h)^T + K r K^T
    a = tuple(tuple((1.0 if i == j else 0.0) - k[i] * h[j] for j in range(3)) for i in range(3))
    ap = tuple(tuple(sum(a[i][m] * P[m][j] for m in range(3)) for j in range(3)) for i in range(3))
    cov = tuple(
        tuple(sum(ap[i][m] * a[j][m] for m in range(3)) + k[i] * action.noise_var * k[j] for j in range(3))
        for i in range(3)
    )
    cov = tuple(tuple(0.5 * (cov[i][j] + cov[j][i]) for j in range(3)) for i in range(3))
    mean = pr.mean
    if y is not None:
        innov = y - sum(h[i] * mean[i] for i in range(3))
        mean = tuple(mean[i] + k[i] * innov for i in range(3))
    out = GaussianState(mean, cov, pr.damage)  # type: ignore[arg-type]
    _check_state(out)
    return out


def terminal_loss(model: GaussianModel, state: GaussianState) -> float:
    _check_state(state)
    o = model.objective
    loss = o.alpha * state.cov[0][0] + (1.0 - o.alpha) * state.cov[1][1] + o.retention_weight * state.damage
    if not isfinite(loss):
        raise ArithmeticError("non-finite terminal loss")
    return loss


def conditional_damage(state: GaussianState) -> float:
    """Posterior expected damage ``E[(x - x0)^2 | D] = Var(x - x0 | D) + E[x - x0 | D]^2``.

    Data-dependent (through the posterior mean), unlike ``state.damage``, which is its average
    over the data. Not used by the planner; see the module docstring.
    """
    _check_state(state)
    c = state.cov
    var = c[1][1] + c[0][0] - 2.0 * c[0][1]
    diff = state.mean[1] - state.mean[0]
    out = max(var, 0.0) + diff * diff
    if not isfinite(out):
        raise ArithmeticError("non-finite conditional damage")
    return out


def _step_cost(model: GaussianModel, action: GaussianAction) -> float:
    return model.objective.energy_lambda * action.energy


@dataclass(frozen=True)
class GaussianPlan:
    """Value of a plan "measure ``sequence``, then STOP" and its parts. All exact within the model."""

    value: float
    sequence: tuple[str, ...]
    terminal_loss: float
    energy: float
    var_initial: float
    var_current: float
    damage: float
    scope: str = GAUSSIAN_SCOPE


def _check_horizon(horizon: int) -> None:
    if type(horizon) is not int or not 0 <= horizon <= MAX_GAUSSIAN_HORIZON:
        raise ValueError(f"horizon must be an int in 0..{MAX_GAUSSIAN_HORIZON}")


def _check_limit(limit: float | None) -> None:
    if limit is not None and (not _finite(limit) or limit < 0):
        raise ValueError("retention_limit must be None or a finite number >= 0")


def plan_value(model: GaussianModel, sequence: Sequence[str], state: GaussianState | None = None) -> GaussianPlan:
    """Exact expected value of measuring ``sequence`` and stopping (data-independent)."""
    seq = tuple(sequence)
    _check_horizon(len(seq))
    st = initial_state(model) if state is None else state
    _check_state(st)
    energy = 0.0
    cost = 0.0
    for name in seq:
        a = model.action(name)
        st = update(model, st, a)
        energy += a.energy
        cost += _step_cost(model, a)
    loss = terminal_loss(model, st)
    value = cost + loss
    if not isfinite(value):
        raise ArithmeticError("non-finite plan value")
    return GaussianPlan(value, seq, loss, energy, st.cov[0][0], st.cov[1][1], st.damage)


def _pick(options: list[tuple[float, tuple[str, ...]]]) -> tuple[float, tuple[str, ...]]:
    if not options or any(not isfinite(v) for v, _ in options):
        raise ArithmeticError("non-finite or empty Bellman choice set")
    best = options[0]
    for opt in options[1:]:
        if opt[0] < best[0] - TIE_TOL:
            best = opt
    return best


def _bellman(model: GaussianModel, st: GaussianState, h: int, limit: float | None) -> tuple[float, tuple[str, ...]]:
    options = [(terminal_loss(model, st), ())]  # STOP first: wins ties
    if h > 0:
        for a in model.actions:
            if limit is not None and st.damage + model.drift_var * a.backaction > limit:
                continue  # the reading would break the retention limit: inadmissible
            v, rest = _bellman(model, update(model, st, a), h - 1, limit)
            options.append((_step_cost(model, a) + v, (a.name, *rest)))
    return _pick(options)


def solve_gaussian(
    model: GaussianModel = DEFAULT_GAUSSIAN_MODEL,
    horizon: int = 5,
    state: GaussianState | None = None,
    *,
    retention_limit: float | None = None,
) -> GaussianPlan:
    """Best open-loop plan "measure ``a_1 .. a_n``, then STOP".

    Without ``retention_limit`` this is the Bayes-optimal policy with STOP (N-47).

    With ``retention_limit`` (``delta``) it is the best *deterministic plan* whose data-averaged
    damage ``E[(x - x0)^2]`` stays ``<= delta`` -- not the optimum under that expected limit:
    a mix of two plans, or a data-dependent rule, can do strictly better (N-52, N-53; use
    ``solve_gaussian_retention``). The limit is not pathwise either: the posterior damage
    ``conditional_damage`` depends on the data and is unbounded under Gaussian tails, so a hard
    pathwise version would admit only readings that carry no information about ``x - x0``
    (finding N-51). Ties within ``1e-12``: STOP first, then catalogue order.
    """
    _check_horizon(horizon)
    _check_limit(retention_limit)
    st = initial_state(model) if state is None else state
    _check_state(st)
    if retention_limit is not None and st.damage > retention_limit:
        raise ValueError("the start state already violates the retention limit")
    _, seq = _bellman(model, st, horizon, retention_limit)
    return plan_value(model, seq, st)


def innovation(model: GaussianModel, state: GaussianState, action: GaussianAction, y: float) -> tuple[float, float]:
    """Innovation ``y - E[y | data]`` of one reading and its variance (before the update).

    The standardised innovation ``e / sqrt(s)`` is standard normal and independent of the data
    seen before; ``Phi`` of it is a uniform draw supplied by the data (used by N-53).
    """
    if action not in model.actions:
        raise ValueError(f"action {action!r} not in this model")
    if not _finite(y):
        raise ValueError("observation must be a finite real number")
    pr = predict(model, state, action)
    h = _h(action)
    s = sum(h[i] * pr.cov[i][j] * h[j] for i in range(3) for j in range(3)) + action.noise_var
    if not (isfinite(s) and s > 0):
        raise ArithmeticError("innovation variance is not positive and finite")
    return y - sum(h[i] * pr.mean[i] for i in range(3)), s


@dataclass(frozen=True)
class GaussianRetentionResult:
    """Optimum under the expected limit ``E[(x - x0)^2] <= delta`` (see the module docstring).

    Three nested policy classes, each solved exactly (``value <= deterministic_value <=
    best_single_plan.value``):

    * ``components`` / ``value``: the optimum over *all* policies -- one plan, or two plans with
      their weights (lower convex hull at ``delta``). The mix draws a plan before the first
      reading; if ``shared_prefix`` is not empty, a threshold on the standardised innovation of
      the last shared reading draws it instead, without a coin.
    * ``deterministic_components`` / ``deterministic_value``: the best *deterministic* policy.
      Its first reading is fixed (no data yet); from the first reading on the data can draw, so it
      is the best over first readings ``a`` of the lower hull of the plans that start with ``a``
      (N-53). Its components always share their first reading; the split is on the innovation of
      the last shared reading.
    * ``best_single_plan``: the best open-loop plan meeting the limit (``solve_gaussian``).

    ``multiplier`` is the price ``mu*`` of the limit (slope of the lower hull at ``delta``);
    ``dual_bound = min_s V(s) + mu* (D(s) - delta)`` over *all* plans is a lower bound on every
    policy meeting the limit, so ``duality_gap`` certifies ``value``. ``unconstrained`` is the
    optimum without the limit.
    """

    delta: float
    horizon: int
    value: float
    damage: float
    components: tuple[tuple[float, GaussianPlan], ...]
    shared_prefix: tuple[str, ...]
    multiplier: float
    dual_bound: float
    duality_gap: float
    deterministic_value: float
    deterministic_damage: float
    deterministic_components: tuple[tuple[float, GaussianPlan], ...]
    best_single_plan: GaussianPlan
    unconstrained: GaussianPlan
    scope: str = GAUSSIAN_SCOPE


Point = tuple[float, float, tuple[str, ...]]


def _plan_points(model: GaussianModel, st: GaussianState, horizon: int) -> list[Point]:
    """(damage, value, sequence) of every plan of length 0..horizon, STOP first, catalogue order."""
    out: list[Point] = []

    def rec(s: GaussianState, seq: tuple[str, ...], cost: float) -> None:
        out.append((s.damage, cost + terminal_loss(model, s), seq))
        if len(seq) < horizon:
            for a in model.actions:
                rec(update(model, s, a), (*seq, a.name), cost + _step_cost(model, a))

    rec(st, (), 0.0)
    if not all(isfinite(d) and isfinite(v) for d, v, _ in out):
        raise ArithmeticError("non-finite plan point")
    return out


def _lower_hull(points: list[Point]) -> list[Point]:
    """Vertices of the lower-left convex hull, damage increasing and value strictly decreasing.

    Points within ``TIE_TOL`` in value keep the earlier one (STOP first, then catalogue order).
    """
    order = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1], i))
    front: list[Point] = []
    for i in order:  # Pareto front: each kept point is strictly cheaper than all with less damage
        if not front or points[i][1] < front[-1][1] - TIE_TOL:
            front.append(points[i])
    hull: list[Point] = []
    for p in front:
        while len(hull) >= 2:
            (d1, v1, _), (d2, v2, _) = hull[-2], hull[-1]
            # drop the middle vertex unless it lies strictly below the chord from hull[-2] to p
            if (v2 - v1) * (p[0] - d1) >= (p[1] - v1) * (d2 - d1) - TIE_TOL * (p[0] - d1):
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def _hull_at(hull: list[Point], delta: float) -> tuple[float, list[tuple[float, tuple[str, ...]]], float] | None:
    """(value, [(weight, sequence)], slope price) of a lower hull at ``delta``; None if infeasible."""
    if not hull or hull[0][0] > delta:
        return None
    if hull[-1][0] <= delta:  # the limit does not bind on this hull
        return hull[-1][1], [(1.0, hull[-1][2])], 0.0
    k = max(i for i, (d, _, _) in enumerate(hull) if d <= delta)
    (d1, v1, s1), (d2, v2, s2) = hull[k], hull[k + 1]
    mu = (v1 - v2) / (d2 - d1)
    q = (d2 - delta) / (d2 - d1)  # weight of the plan with less damage
    if q >= 1.0 - 1e-15:
        return v1, [(1.0, s1)], mu
    return q * v1 + (1.0 - q) * v2, [(q, s1), (1.0 - q, s2)], mu


def _shared(a: tuple[str, ...], b: tuple[str, ...]) -> tuple[str, ...]:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def gaussian_retention_frontier(
    model: GaussianModel = DEFAULT_GAUSSIAN_MODEL, horizon: int = 5, state: GaussianState | None = None
) -> tuple[tuple[float, float, tuple[str, ...]], ...]:
    """Vertices ``(damage, value, plan)`` of the lower convex hull: the optimum under the expected
    limit for *every* ``delta`` at once (linear between vertices, flat after the last one).

    The slope between two vertices is the price of the limit there; read it to choose ``delta``.
    """
    _check_horizon(horizon)
    st = initial_state(model) if state is None else state
    _check_state(st)
    return tuple(_lower_hull(_plan_points(model, st, horizon)))


def solve_gaussian_retention(
    model: GaussianModel = DEFAULT_GAUSSIAN_MODEL,
    horizon: int = 5,
    delta: float = 0.0,
    state: GaussianState | None = None,
) -> GaussianRetentionResult:
    """Bayes-optimal policy under the expected limit ``E[(x - x0)^2] <= delta`` (N-52), with the
    best deterministic policy (N-53) and the best single plan next to it.

    Exact over all policies (randomised or data-dependent): the lower convex hull of the plan
    points ``(D(s), V(s))`` at ``delta``. Cost: every plan is evaluated once
    (``sum_k |A|^k`` covariance updates, as ``solve_gaussian``).
    """
    _check_horizon(horizon)
    if delta is None:
        raise ValueError("delta must be a finite number >= 0")
    _check_limit(delta)
    delta = float(delta)
    st = initial_state(model) if state is None else state
    _check_state(st)
    if st.damage > delta:
        raise ValueError("the start state already violates the retention limit")
    points = _plan_points(model, st, horizon)
    hull = _lower_hull(points)
    unconstrained = plan_value(model, hull[-1][2], st)
    found = _hull_at(hull, delta)
    if found is None:  # unreachable: the empty plan has the start damage
        raise ArithmeticError("no plan meets the limit")
    _, mix, mu = found
    comps = tuple((w, plan_value(model, seq, st)) for w, seq in mix)

    # best deterministic policy: fixed first reading (or STOP), then the data may draw
    det: tuple[float, list[tuple[float, tuple[str, ...]]]] = (points[0][1], [(1.0, ())])  # STOP
    for a in model.actions:
        sub = _hull_at(_lower_hull([p for p in points if p[2][:1] == (a.name,)]), delta)
        if sub is not None and sub[0] < det[0] - TIE_TOL:
            det = (sub[0], sub[1])
    det_comps = tuple((w, plan_value(model, seq, st)) for w, seq in det[1])

    feasible = [(v, i) for i, (d, v, _) in enumerate(points) if d <= delta]
    single = plan_value(model, points[min(feasible, key=lambda t: (t[0], t[1]))[1]][2], st)

    value = sum(w * p.value for w, p in comps)
    damage = sum(w * p.damage for w, p in comps)
    det_value = sum(w * p.value for w, p in det_comps)
    det_damage = sum(w * p.damage for w, p in det_comps)
    dual = min(v + mu * (d - delta) for d, v, _ in points)
    gap = value - dual
    tol = 4 * (horizon + 1) * TIE_TOL * max(1.0, abs(value), mu)
    if not all(isfinite(x) for x in (value, damage, mu, dual, gap, det_value, det_damage)):
        raise ArithmeticError("non-finite retention optimum")
    if damage > delta + tol or det_damage > delta + tol or abs(gap) > tol:
        raise ArithmeticError("retention optimum fails its own certificate")
    if not value - tol <= det_value <= single.value + tol:
        raise ArithmeticError("policy classes out of order: mixture <= deterministic <= single plan violated")
    if len(det_comps) == 2 and not _shared(det_comps[0][1].sequence, det_comps[1][1].sequence):
        raise ArithmeticError("deterministic components must share their first reading")
    shared = _shared(comps[0][1].sequence, comps[1][1].sequence) if len(comps) == 2 else ()
    return GaussianRetentionResult(
        delta=delta,
        horizon=horizon,
        value=value,
        damage=damage,
        components=comps,
        shared_prefix=shared,
        multiplier=mu,
        dual_bound=dual,
        duality_gap=gap,
        deterministic_value=det_value,
        deterministic_damage=det_damage,
        deterministic_components=det_comps,
        best_single_plan=single,
        unconstrained=unconstrained,
    )


def lookahead_plan(
    model: GaussianModel = DEFAULT_GAUSSIAN_MODEL,
    horizon: int = 5,
    depth: int = 2,
    *,
    retention_limit: float | None = None,
) -> GaussianPlan:
    """Receding-horizon rule: at each step solve the ``depth``-step problem, take its first action.

    ``depth = 1`` is the greedy (myopic) rule, ``depth = 2`` the two-step look-ahead of the
    roadmap. The rule stops as soon as the truncated problem prefers STOP or the horizon is used.
    With ``retention_limit`` the truncated problems are those of ``solve_gaussian`` (deterministic
    plans under the limit), so the result is compared with that plan, not with the mixture
    optimum of ``solve_gaussian_retention``.
    """
    _check_horizon(horizon)
    _check_limit(retention_limit)
    if type(depth) is not int or depth < 1:
        raise ValueError("depth must be an int >= 1")
    st = initial_state(model)
    seq: list[str] = []
    for remaining in range(horizon, 0, -1):
        _, first = _bellman(model, st, min(depth, remaining), retention_limit)
        if not first:
            break
        a = model.action(first[0])
        st = update(model, st, a)
        seq.append(a.name)
    return plan_value(model, seq)


def reference_plan(
    horizon: int, target: str = "fast", reference: str = "reference", period: int = 2
) -> tuple[str, ...]:
    """Fixed reference-slot plan of the earlier R8/R9 line: a reference read every ``period`` slots.

    ``period=2`` gives ``reference, target, reference, target, ...``.
    """
    _check_horizon(horizon)
    if type(period) is not int or period < 1:
        raise ValueError("period must be an int >= 1")
    return tuple(reference if k % period == 0 else target for k in range(horizon))


def fixed_plan(model: GaussianModel, sequence: Sequence[str]) -> GaussianPlan:
    """A fixed plan measured to its end (no stopping) -- the baseline of the roadmap item."""
    return plan_value(model, sequence)


def best_fixed_plan(model: GaussianModel, horizon: int) -> GaussianPlan:
    """Best plan of *exactly* ``horizon`` readings (the forced planner of the discrete line)."""
    _check_horizon(horizon)
    st = initial_state(model)

    def rec(s: GaussianState, h: int) -> tuple[float, tuple[str, ...]]:
        if h == 0:
            return terminal_loss(model, s), ()
        opts = []
        for a in model.actions:
            v, rest = rec(update(model, s, a), h - 1)
            opts.append((_step_cost(model, a) + v, (a.name, *rest)))
        return _pick(opts)

    _, seq = rec(st, horizon)
    return plan_value(model, seq)


def model_reference_plan(model: GaussianModel, horizon: int) -> tuple[str, ...]:
    """``reference_plan`` with the model's own first reference and first target action.

    A catalogue without a reference action gives a target-only plan and vice versa.
    """
    _check_horizon(horizon)
    refs = [a.name for a in model.actions if a.reads_reference]
    targets = [a.name for a in model.actions if not a.reads_reference]
    if not refs or not targets:
        return ((refs or targets)[0],) * horizon
    return reference_plan(horizon, target=targets[0], reference=refs[0])


def gaussian_table(model: GaussianModel = DEFAULT_GAUSSIAN_MODEL, max_horizon: int = 8) -> list[dict]:
    """Optimum vs. two-step look-ahead vs. greedy vs. fixed reference plan, ``h = 0..max_horizon``."""
    _check_horizon(max_horizon)
    rows = []
    for h in range(max_horizon + 1):
        opt = solve_gaussian(model, h)
        two = lookahead_plan(model, h, 2)
        greedy = lookahead_plan(model, h, 1)
        forced = best_fixed_plan(model, h)
        ref = fixed_plan(model, model_reference_plan(model, h))
        rows.append(
            {
                "horizon": h,
                "optimal": opt.value,
                "optimal_plan": list(opt.sequence),
                "two_step_lookahead": two.value,
                "two_step_plan": list(two.sequence),
                "greedy": greedy.value,
                "greedy_plan": list(greedy.sequence),
                "best_forced_h": forced.value,
                "reference_plan": ref.value,
                "reference_plan_sequence": list(ref.sequence),
                "optimal_var_initial": opt.var_initial,
                "optimal_damage": opt.damage,
                "scope": GAUSSIAN_SCOPE,
            }
        )
    return rows

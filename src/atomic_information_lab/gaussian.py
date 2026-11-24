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
value; it is averaged over the data, exactly like the expected retention term of the discrete
planner. Every reading costs ``energy_lambda * energy``.

Why this line has an exact oracle
---------------------------------
The Kalman covariance recursion does not depend on the observed values, and the loss depends
on the data only through that covariance. Hence the conditional value of every continuation
is the same for every data set: the Bayes-optimal *adaptive* policy with STOP is an open-loop
plan "measure ``a_1 .. a_n``, then stop", found by a deterministic Bellman recursion over
covariances (measurement control can be optimised a priori: Meier, Peschon & Dressler 1967,
doi:10.1109/TAC.1967.1098668; tree search over schedules: Vitus et al., Automatica 2012).
Consequence: in this model class adaptivity is worth exactly zero, whereas in the discrete
model it is worth 9.6 % at ``h = 5`` (finding N-47). The independent oracle
``tests/_gaussian_oracle.py`` recomputes every posterior in exact rational arithmetic by
batch least squares (information form over the noise sequence), sharing no code with this file.

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
    "DEFAULT_GAUSSIAN_ACTIONS",
    "DEFAULT_GAUSSIAN_MODEL",
    "initial_state",
    "predict",
    "update",
    "terminal_loss",
    "solve_gaussian",
    "plan_value",
    "lookahead_plan",
    "reference_plan",
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


@dataclass(frozen=True)
class GaussianState:
    """Posterior ``N(mean, cov)`` over ``(x0, x, b)`` plus the accumulated drift variance ``E[(x-x0)^2]``."""

    mean: tuple[float, float, float]
    cov: Matrix
    damage: float = 0.0


def initial_state(model: GaussianModel) -> GaussianState:
    v, vb = model.prior_var, model.bias_prior_var
    return GaussianState((0.0, 0.0, 0.0), ((v, v, 0.0), (v, v, 0.0), (0.0, 0.0, vb)), 0.0)


def _check_state(state: GaussianState) -> None:
    if type(state) is not GaussianState:
        raise ValueError("state must be a GaussianState")
    flat = (*state.mean, *(c for row in state.cov for c in row), state.damage)
    if len(state.mean) != 3 or len(state.cov) != 3 or any(len(r) != 3 for r in state.cov) or not _finite(*flat):
        raise ValueError("state must hold a finite 3-vector and a finite 3x3 matrix")


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
    """Bayes-optimal policy with STOP (an open-loop plan, see the module docstring).

    ``retention_limit`` (``delta``): a reading is admissible only if ``E[(x - x0)^2]`` stays
    ``<= delta`` afterwards; because the damage is data-independent, the pathwise and the
    expected reading of the limit coincide here. Ties within ``1e-12``: STOP first, then
    catalogue order.
    """
    _check_horizon(horizon)
    _check_limit(retention_limit)
    st = initial_state(model) if state is None else state
    _check_state(st)
    if retention_limit is not None and st.damage > retention_limit:
        raise ValueError("the start state already violates the retention limit")
    _, seq = _bellman(model, st, horizon, retention_limit)
    return plan_value(model, seq, st)


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


def gaussian_table(model: GaussianModel = DEFAULT_GAUSSIAN_MODEL, max_horizon: int = 8) -> list[dict]:
    """Optimum vs. two-step look-ahead vs. greedy vs. fixed reference plan, ``h = 0..max_horizon``."""
    _check_horizon(max_horizon)
    rows = []
    for h in range(max_horizon + 1):
        opt = solve_gaussian(model, h)
        two = lookahead_plan(model, h, 2)
        greedy = lookahead_plan(model, h, 1)
        forced = best_fixed_plan(model, h)
        ref = fixed_plan(model, reference_plan(h))
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

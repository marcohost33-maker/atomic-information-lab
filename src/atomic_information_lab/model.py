"""Discrete two-state measurement model with unknown sensor inversion.

Hidden state per hypothesis ``theta``: ``(x0, x, b)``

* ``x0`` -- the bit that was originally written (never changes),
* ``x``  -- the current state of the stored bit (two-state CTMC),
* ``b``  -- a persistent but slowly flipping sensor-inversion bit.

One measurement step with action ``a``:

1. the bit evolves for one time unit under the generator with rates
   ``k01 * backaction(a)`` and ``k10 * backaction(a)``,
2. the inversion bit flips with probability ``bias_flip``,
3. the sensor reports ``target XOR noise`` where ``target = x XOR b`` for a
   target read and ``target = 0 XOR b`` for a read of a known reference
   (the reference therefore reveals ``b``), and ``noise ~ Bernoulli(error)``.

All quantities are dimensionless model parameters. ``energy`` is an abstract
cost unit, not joules. This is a synthetic classical model; it does not
describe any specific instrument and makes no quantum claim.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import product
from math import expm1, isfinite

__all__ = [
    "Action",
    "Hypothesis",
    "Objective",
    "Problem",
    "DEFAULT_ACTIONS",
    "DEFAULT_HYPOTHESIS",
    "TWO_HYPOTHESES",
    "MAX_HYPOTHESES",
    "MAX_ACTIONS",
    "STOP",
]

STOP = "stop"
MAX_HYPOTHESES = 4
MAX_ACTIONS = 4
_SUM_TOL = 1e-9
_LOCAL = tuple(product((0, 1), repeat=3))  # (x0, x, b) inside one hypothesis


def _finite(*values: float) -> bool:
    return all(isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v) for v in values)


@dataclass(frozen=True)
class Action:
    """A measurement option."""

    name: str
    backaction: float
    error: float
    energy: float
    reads_reference: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name or self.name == STOP:
            raise ValueError("action name must be a non-empty string other than 'stop'")
        if not _finite(self.backaction, self.error, self.energy):
            raise ValueError(f"action {self.name!r}: parameters must be finite real numbers")
        if self.backaction < 0 or self.energy < 0 or not 0 <= self.error <= 0.5:
            raise ValueError(f"action {self.name!r}: need backaction>=0, energy>=0, 0<=error<=0.5")
        if not isinstance(self.reads_reference, bool):
            raise ValueError("reads_reference must be bool")


@dataclass(frozen=True)
class Hypothesis:
    """Transition rates of the stored bit and flip probability of the sensor inversion."""

    k01: float
    k10: float
    bias_flip: float

    def __post_init__(self) -> None:
        if not _finite(self.k01, self.k10, self.bias_flip):
            raise ValueError("hypothesis parameters must be finite real numbers")
        if self.k01 < 0 or self.k10 < 0 or not 0 <= self.bias_flip <= 0.5:
            raise ValueError("need k01>=0, k10>=0, 0<=bias_flip<=0.5")


@dataclass(frozen=True)
class Objective:
    """Terminal loss weights and energy price.

    ``L(b) = alpha*Var(x0) + (1-alpha)*Var(x) + retention_weight*P(x != x0)``;
    each measurement costs ``energy_lambda * energy``.
    """

    alpha: float = 0.5
    energy_lambda: float = 0.012
    retention_weight: float = 0.0

    def __post_init__(self) -> None:
        if not _finite(self.alpha, self.energy_lambda, self.retention_weight):
            raise ValueError("objective parameters must be finite real numbers")
        if not 0 <= self.alpha <= 1 or self.energy_lambda < 0 or self.retention_weight < 0:
            raise ValueError("need 0<=alpha<=1, energy_lambda>=0, retention_weight>=0")


DEFAULT_ACTIONS: tuple[Action, ...] = (
    Action("fast", 1.0, 0.12, 1.0),
    Action("precise", 3.5, 0.035, 2.8),
    Action("reference", 0.55, 0.025, 0.8, reads_reference=True),
)
DEFAULT_HYPOTHESIS = Hypothesis(0.07, 0.045, 0.025)
TWO_HYPOTHESES = (Hypothesis(0.07, 0.045, 0.025), Hypothesis(0.23, 0.16, 0.10))


def _step_probabilities(k01: float, k10: float) -> tuple[float, float]:
    """Closed form of exp(Q) for the two-state generator over one time unit."""
    total = k01 + k10
    if not isfinite(total):
        raise ValueError("effective transition rate overflows to infinity")
    if total == 0.0:
        return 0.0, 0.0
    q = -expm1(-total)
    return k01 * q / total, k10 * q / total


class Problem:
    """A finite mixture of hypotheses plus action catalogue and objective.

    Beliefs are tuples of length ``8 * len(hypotheses)`` ordered as
    ``(theta, x0, x, b)`` with ``theta`` slowest.
    """

    def __init__(
        self,
        hypotheses: Sequence[Hypothesis] = (DEFAULT_HYPOTHESIS,),
        weights: Sequence[float] | None = None,
        actions: Sequence[Action] = DEFAULT_ACTIONS,
        objective: Objective = Objective(),
    ) -> None:
        hypotheses = tuple(hypotheses)
        actions = tuple(actions)
        if not 1 <= len(hypotheses) <= MAX_HYPOTHESES or any(type(h) is not Hypothesis for h in hypotheses):
            raise ValueError(f"need 1..{MAX_HYPOTHESES} Hypothesis objects")
        if not 1 <= len(actions) <= MAX_ACTIONS or any(type(a) is not Action for a in actions):
            raise ValueError(f"need 1..{MAX_ACTIONS} Action objects")
        if len({a.name for a in actions}) != len(actions):
            raise ValueError("action names must be unique")
        if type(objective) is not Objective:
            raise ValueError("objective must be an Objective")
        if weights is None:
            weights = (1.0 / len(hypotheses),) * len(hypotheses)
        weights = tuple(weights)
        if len(weights) != len(hypotheses) or not _finite(*weights) or any(w < 0 for w in weights):
            raise ValueError("weights must be finite, non-negative, one per hypothesis")
        if abs(sum(weights) - 1.0) > 1e-12:
            raise ValueError("weights must sum to one")
        self.hypotheses = hypotheses
        self.weights = weights
        self.actions = actions
        self.objective = objective
        self.n_states = 8 * len(hypotheses)
        # Per (hypothesis, action): 8x8 transition matrix over the local states.
        self._kernel: dict[tuple[int, str], tuple[tuple[float, ...], ...]] = {}
        for t, hyp in enumerate(hypotheses):
            for a in actions:
                r01, r10 = hyp.k01 * a.backaction, hyp.k10 * a.backaction
                if not (isfinite(r01) and isfinite(r10)):
                    raise ValueError(f"rate * backaction overflows for action {a.name!r}")
                p01, p10 = _step_probabilities(r01, r10)
                move = ((1.0 - p01, p01), (p10, 1.0 - p10))
                f = hyp.bias_flip
                self._kernel[(t, a.name)] = tuple(
                    tuple((move[x][x2] * (f if b != b2 else 1.0 - f)) if x0 == y0 else 0.0 for (y0, x2, b2) in _LOCAL)
                    for (x0, x, b) in _LOCAL
                )

    # ------------------------------------------------------------ beliefs
    def initial_belief(self) -> tuple[float, ...]:
        """x0 uniform, x == x0, b uniform and independent; hypotheses by weight."""
        return tuple(w * (0.25 if x == x0 else 0.0) for w in self.weights for (x0, x, b) in _LOCAL)

    def validate_belief(self, belief: Sequence[float]) -> tuple[float, ...]:
        belief = tuple(belief)
        if len(belief) != self.n_states or not _finite(*belief) or any(p < 0 for p in belief):
            raise ValueError(f"belief must be {self.n_states} finite non-negative numbers")
        if abs(sum(belief) - 1.0) > _SUM_TOL:
            raise ValueError("belief must sum to one")
        return belief

    def action(self, name: str) -> Action:
        for a in self.actions:
            if a.name == name:
                return a
        raise ValueError(f"unknown action {name!r}")

    def branches(self, belief: Sequence[float], action: Action) -> tuple[tuple[int, float, tuple[float, ...]], ...]:
        """All observation outcomes ``(y, P(y), posterior)`` with ``P(y) > 0``.

        Fails closed (ArithmeticError) if the outcome masses are not finite or
        do not sum to one -- a silently dropped branch would make the risk look
        smaller than it is.
        """
        belief = self.validate_belief(belief)
        if action not in self.actions:
            raise ValueError(f"action {action!r} not in this problem")
        pred = [0.0] * self.n_states
        for t in range(len(self.hypotheses)):
            kern = self._kernel[(t, action.name)]
            off = 8 * t
            for i in range(8):
                p = belief[off + i]
                if p:
                    row = kern[i]
                    for j in range(8):
                        pred[off + j] += p * row[j]
        out = []
        masses = []
        for y in (0, 1):
            joint = []
            for idx in range(self.n_states):
                _, x, b = _LOCAL[idx % 8]
                target = (0 if action.reads_reference else x) ^ b
                like = (1.0 - action.error) if target == y else action.error
                joint.append(pred[idx] * like)
            z = sum(joint)
            masses.append(z)
            if not isfinite(z) or z < 0:
                raise ArithmeticError("non-finite or negative outcome probability")
            if z > 0:
                out.append((y, z, tuple(v / z for v in joint)))
        if not abs(sum(masses) - 1.0) <= _SUM_TOL:  # NaN-safe: NaN fails this test
            raise ArithmeticError("outcome probabilities do not sum to one")
        return tuple(out)

    # ------------------------------------------------------------ marginals
    def marginals(self, belief: Sequence[float]) -> dict[str, float]:
        belief = self.validate_belief(belief)
        p0 = pt = mismatch = 0.0
        for idx, p in enumerate(belief):
            x0, x, _ = _LOCAL[idx % 8]
            p0 += p * x0
            pt += p * x
            mismatch += p * (x0 != x)
        return {"p_initial_one": p0, "p_current_one": pt, "p_changed": mismatch}

    def model_posterior(self, belief: Sequence[float]) -> tuple[float, ...]:
        belief = self.validate_belief(belief)
        return tuple(sum(belief[8 * t : 8 * (t + 1)]) for t in range(len(self.hypotheses)))

    def terminal_loss(self, belief: Sequence[float]) -> float:
        """Expected Brier score of the posterior-mean forecasts plus retention penalty."""
        m = self.marginals(belief)
        o = self.objective
        p0, pt = m["p_initial_one"], m["p_current_one"]
        return o.alpha * p0 * (1.0 - p0) + (1.0 - o.alpha) * pt * (1.0 - pt) + o.retention_weight * m["p_changed"]

    def step_cost(self, action: Action) -> float:
        return self.objective.energy_lambda * action.energy

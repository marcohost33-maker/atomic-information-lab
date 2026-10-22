"""Experimental identifiability of the measurement model (roadmap R13/P0).

Question of the roadmap: can preparation error, sensor bias, drift and
back-action be estimated *separately* from calibration sessions -- or is it
NO-GO? This module answers it for the model in ``model.py`` extended by two
calibration parameters:

* ``prep_error`` -- probability that writing ``x0`` leaves the bit in the
  other state before the first reading (``x_start = x0 XOR Bernoulli(eps)``);
  the planning model assumes ``prep_error = 0``;
* ``bias_prior`` -- probability that the sensor starts inverted, ``P(b = 1)``;
  the planning model assumes ``0.5``.

A *calibration session* runs a fixed sequence of actions (a ``Design``) and
records the binary readings. When ``known_write`` is set, the experimenter
also records which bit was written (half of the sessions write 0, half 1);
otherwise ``x0`` is uniform and unobserved, as in the planning model.
Sessions are independent, so N sessions carry N times the information of one.

Three tools, each with its own evidence:

1. **Exact structural witnesses** (global statements). A witness is a second,
   different parameter point that produces the *same* distribution of
   session outcomes; it is checked numerically every time it is reported
   (max. absolute difference <= 1e-13, else ``ArithmeticError``). Known
   witnesses: drift/back-action scale (only products ``k * beta`` enter the
   model), hidden preparation error (without known writes ``x_start`` is
   uniform for every ``eps``) and the label swap (``x <-> 1-x``,
   ``k01 <-> k10``, ``b <-> 1-b``) that only a reference read breaks.
2. **Fisher information** per session, from exact forward-mode derivatives of
   every outcome probability. By Rothenberg (1971), under regularity
   conditions a parameter point is *locally* identifiable iff the Fisher
   information is non-singular. Rank is decided on the scale-free matrix
   ``D^-1/2 F D^-1/2`` (``D = diag F``) with tolerance ``RANK_TOL``; its
   null space is reported in the original coordinates.
3. **Cramer-Rao bounds**: ``sqrt((F^-1)_ii / N)`` is the smallest standard
   error any unbiased estimator can reach with N sessions; reported as the
   number of sessions needed for a 10 % relative standard error. A full rank
   with an astronomically large session count is identifiable in principle
   but not in practice.

``discrimination`` compares two calibrations on one design (KL divergence and
Chernoff information, with the number of sessions after which the equal-prior
Bayes error is guaranteed below ``alpha``); ``design_search`` ranks all fixed
sequences of a given length.

Limits: synthetic classical model; the regularity conditions of Rothenberg's
theorem fail on the boundary of the parameter space (an ``error`` of 0 makes
outcome probabilities vanish and is refused); a full-rank Fisher matrix is a
numerical statement with the tolerance stated above, a witness is an exact
statement checked numerically. Nothing here describes a real instrument.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
from itertools import product
from math import atan2, ceil, cos, exp, expm1, isfinite, log, sin, sqrt

from .model import DEFAULT_ACTIONS, DEFAULT_HYPOTHESIS, MAX_ACTIONS, Action, Hypothesis

__all__ = [
    "Calibration",
    "Design",
    "Witness",
    "IdentifiabilityReport",
    "Discrimination",
    "DesignScore",
    "MAX_DESIGN_LENGTH",
    "MAX_SEARCH_DESIGNS",
    "RANK_TOL",
    "ZERO_INFO_TOL",
    "WITNESS_TOL",
    "parameter_names",
    "outcome_distribution",
    "fisher_information",
    "identifiability",
    "discrimination",
    "design_search",
]

MAX_DESIGN_LENGTH = 10
MAX_SEARCH_DESIGNS = 2048
RANK_TOL = 1e-9  # eigenvalue of the scale-free Fisher matrix (diagonal 1) counted as zero
# Information about a relative change of a parameter, F_ii * value**2 (value 0: F_ii), below which the
# design carries no information about it. Cancellation that is exact in theory (e.g. a back-action factor
# after the last target read) leaves rounding noise; over all 726 designs of length 1..5 with the default
# actions the largest such value is 1.5e-33 and the smallest real one 2.8e-7 (tests/test_identifiability.py).
ZERO_INFO_TOL = 1e-20
WITNESS_TOL = 1e-13  # max. absolute difference of two outcome distributions called "equal"
_SUM_TOL = 1e-12
_BASE = ("k01", "k10", "bias_flip", "prep_error", "bias_prior")


def _real(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v)


# ---------------------------------------------------------------- forward-mode derivatives
class _D:
    """Value with a gradient vector; just enough arithmetic for the forward algorithm."""

    __slots__ = ("v", "g")

    def __init__(self, v: float, g: tuple[float, ...]) -> None:
        self.v, self.g = v, g

    def _lift(self, o):
        return o if isinstance(o, _D) else _D(float(o), (0.0,) * len(self.g))

    def __add__(self, o):
        o = self._lift(o)
        return _D(self.v + o.v, tuple(a + b for a, b in zip(self.g, o.g, strict=True)))

    __radd__ = __add__

    def __sub__(self, o):
        o = self._lift(o)
        return _D(self.v - o.v, tuple(a - b for a, b in zip(self.g, o.g, strict=True)))

    def __rsub__(self, o):
        return self._lift(o) - self

    def __mul__(self, o):
        o = self._lift(o)
        return _D(self.v * o.v, tuple(self.v * b + o.v * a for a, b in zip(self.g, o.g, strict=True)))

    __rmul__ = __mul__

    def __truediv__(self, o):
        o = self._lift(o)
        q = self.v / o.v
        return _D(q, tuple((a - q * b) / o.v for a, b in zip(self.g, o.g, strict=True)))


def _one_minus_exp_neg(t: _D) -> _D:
    """1 - exp(-t) with its derivative exp(-t) * dt."""
    d = exp(-t.v)
    return _D(-expm1(-t.v), tuple(d * a for a in t.g))


# ---------------------------------------------------------------- model of a calibration session
@dataclass(frozen=True)
class Calibration:
    """All parameters that a calibration campaign would have to determine."""

    hypothesis: Hypothesis = DEFAULT_HYPOTHESIS
    actions: tuple[Action, ...] = DEFAULT_ACTIONS
    prep_error: float = 0.0
    bias_prior: float = 0.5

    def __post_init__(self) -> None:
        if type(self.hypothesis) is not Hypothesis:
            raise ValueError("hypothesis must be a Hypothesis")
        acts = tuple(self.actions) if isinstance(self.actions, (tuple, list)) else None
        if acts is None or not 1 <= len(acts) <= MAX_ACTIONS or any(type(a) is not Action for a in acts):
            raise ValueError(f"actions must be 1..{MAX_ACTIONS} Action objects")
        if len({a.name for a in acts}) != len(acts):
            raise ValueError("action names must be unique")
        object.__setattr__(self, "actions", acts)
        if not _real(self.prep_error) or not 0 <= self.prep_error <= 0.5:
            raise ValueError("prep_error must be a finite number in [0, 0.5]")
        if not _real(self.bias_prior) or not 0 <= self.bias_prior <= 1:
            raise ValueError("bias_prior must be a finite number in [0, 1]")

    def parameters(self) -> dict[str, float]:
        """Every parameter by name, in the canonical order of ``parameter_names``."""
        h = self.hypothesis
        out = {
            "k01": float(h.k01),
            "k10": float(h.k10),
            "bias_flip": float(h.bias_flip),
            "prep_error": float(self.prep_error),
            "bias_prior": float(self.bias_prior),
        }
        for a in self.actions:
            out[f"error:{a.name}"] = float(a.error)
        for a in self.actions:
            out[f"backaction:{a.name}"] = float(a.backaction)
        return out

    def with_values(self, values: Mapping[str, float]) -> Calibration:
        """A copy with some parameters replaced; every value goes through the usual validation."""
        unknown = set(values) - set(self.parameters())
        if unknown:
            raise ValueError(f"unknown parameter(s): {sorted(unknown)}")
        p = {**self.parameters(), **values}
        hyp = Hypothesis(p["k01"], p["k10"], p["bias_flip"])
        acts = tuple(
            Action(a.name, p[f"backaction:{a.name}"], p[f"error:{a.name}"], a.energy, a.reads_reference)
            for a in self.actions
        )
        return Calibration(hyp, acts, p["prep_error"], p["bias_prior"])

    def action(self, name: str) -> Action:
        for a in self.actions:
            if a.name == name:
                return a
        raise ValueError(f"unknown action {name!r}")


def parameter_names(calibration: Calibration) -> tuple[str, ...]:
    return tuple(calibration.parameters())


@dataclass(frozen=True)
class Design:
    """One calibration session: a fixed action sequence; ``known_write`` records the written bit."""

    sequence: tuple[str, ...]
    known_write: bool = False

    def __post_init__(self) -> None:
        seq = self.sequence
        if isinstance(seq, str) or not isinstance(seq, (tuple, list)):
            raise ValueError("sequence must be a tuple of action names")
        seq = tuple(seq)
        if not 1 <= len(seq) <= MAX_DESIGN_LENGTH or not all(isinstance(s, str) and s for s in seq):
            raise ValueError(f"sequence must hold 1..{MAX_DESIGN_LENGTH} action names")
        if not isinstance(self.known_write, bool):
            raise ValueError("known_write must be bool")
        object.__setattr__(self, "sequence", seq)


def _check_free(calibration: Calibration, free: Iterable[str]) -> tuple[str, ...]:
    if isinstance(free, str):
        raise ValueError("free must be a sequence of parameter names, not a string")
    free = tuple(free)
    names = calibration.parameters()
    if not free or len(set(free)) != len(free) or any(f not in names for f in free):
        raise ValueError(f"free must be distinct names out of {list(names)}")
    return tuple(n for n in names if n in free)  # canonical order


def _leaves(calibration: Calibration, design: Design, free: tuple[str, ...]):
    """Yield (outcome, probability as _D) for every session outcome, by depth-first forward recursion."""
    acts = {a.name: a for a in calibration.actions}
    for name in design.sequence:
        if name not in acts:
            raise ValueError(f"design uses unknown action {name!r}")
    n = len(free)
    par = {
        k: _D(v, tuple(1.0 if f == k else 0.0 for f in free)) if k in free else _D(v, (0.0,) * n)
        for k, v in calibration.parameters().items()
    }
    steps = []
    for name in design.sequence:
        r01 = par["k01"] * par[f"backaction:{name}"]
        r10 = par["k10"] * par[f"backaction:{name}"]
        tot = r01 + r10
        if not isfinite(tot.v):
            raise ValueError(f"rate * backaction overflows for action {name!r}")
        if tot.v == 0.0:
            p01, p10 = r01, r10  # limit of r * (1 - e^-tot) / tot, exact to first order
        else:
            q = _one_minus_exp_neg(tot)
            p01, p10 = r01 * q / tot, r10 * q / tot
        steps.append((acts[name].reads_reference, p01, p10, par[f"error:{name}"]))
    f, eps, pb = par["bias_flip"], par["prep_error"], par["bias_prior"]
    starts = (
        [((0,), 0.5, eps), ((1,), 0.5, 1 - eps)] if design.known_write else [((), 1.0, 0.5 * eps + 0.5 * (1 - eps))]
    )

    def rec(t, alpha, prefix):
        if t == len(steps):
            yield prefix, alpha[0] + alpha[1] + alpha[2] + alpha[3]
            return
        ref, p01, p10, e = steps[t]
        move = ((1 - p01, p01), (p10, 1 - p10))
        a2 = [alpha[0 + b] * move[0][x2] + alpha[2 + b] * move[1][x2] for x2 in (0, 1) for b in (0, 1)]
        a3 = [
            a2[2 * x + 0] * (f if b2 else 1 - f) + a2[2 * x + 1] * (1 - f if b2 else f) for x in (0, 1) for b2 in (0, 1)
        ]
        for y in (0, 1):
            nxt = []
            for s in range(4):
                x, b = divmod(s, 2)
                target = (0 if ref else x) ^ b
                nxt.append(a3[s] * ((1 - e) if target == y else e))
            yield from rec(t + 1, nxt, prefix + (y,))

    for label, w, px1 in starts:
        alpha = [(1 - px1) * (1 - pb) * w, (1 - px1) * pb * w, px1 * (1 - pb) * w, px1 * pb * w]
        yield from rec(0, alpha, label)


def outcome_distribution(calibration: Calibration, design: Design) -> dict[tuple[int, ...], float]:
    """P(outcome) for every outcome ``(x0?, y1, ..., yn)``; ``x0`` only with ``known_write``."""
    out = {o: p.v for o, p in _leaves(calibration, design, ())}
    total = sum(out.values())
    if not all(isfinite(p) and p >= 0 for p in out.values()) or not abs(total - 1.0) <= _SUM_TOL:
        raise ArithmeticError("outcome probabilities are not a distribution")
    return out


def fisher_information(calibration: Calibration, design: Design, free: Iterable[str]) -> tuple[tuple[float, ...], ...]:
    """Per-session Fisher information ``sum_o grad p(o) grad p(o)^T / p(o)`` in the canonical order of ``free``.

    Refuses (ArithmeticError) an outcome of probability 0 whose probability
    still moves with the parameters: there the information is not defined
    (boundary of the parameter space).
    """
    free = _check_free(calibration, free)
    n = len(free)
    fim = [[0.0] * n for _ in range(n)]
    total = 0.0
    for _, p in _leaves(calibration, design, free):
        if not (isfinite(p.v) and all(isfinite(g) for g in p.g)) or p.v < 0:
            raise ArithmeticError("non-finite or negative outcome probability")
        total += p.v
        if p.v == 0.0:
            if any(p.g):
                raise ArithmeticError(
                    "outcome of probability 0 with non-zero derivative: boundary, no Fisher information"
                )
            continue
        for i in range(n):
            gi = p.g[i] / p.v
            if gi:
                row = fim[i]
                for j in range(n):
                    row[j] += gi * p.g[j]
    if not abs(total - 1.0) <= _SUM_TOL:
        raise ArithmeticError("outcome probabilities do not sum to one")
    for i in range(n):  # symmetrise rounding
        for j in range(i):
            fim[i][j] = fim[j][i] = 0.5 * (fim[i][j] + fim[j][i])
    if not all(isfinite(v) for row in fim for v in row):
        raise ArithmeticError("non-finite Fisher information")
    return tuple(tuple(r) for r in fim)


# ---------------------------------------------------------------- symmetric eigen-decomposition (Jacobi)
def _eigh(a: Sequence[Sequence[float]]) -> tuple[list[float], list[list[float]]]:
    """Eigenvalues and eigenvectors (columns of V) of a small symmetric matrix, cyclic Jacobi."""
    n = len(a)
    m = [list(map(float, r)) for r in a]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    scale = sum(m[i][i] ** 2 for i in range(n)) or 1.0
    for _ in range(100):
        off = sum(m[i][j] ** 2 for i in range(n) for j in range(i + 1, n))
        if off <= 1e-30 * scale:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if m[p][q] == 0.0:
                    continue
                phi = 0.5 * atan2(2.0 * m[p][q], m[q][q] - m[p][p])
                c, s = cos(phi), sin(phi)
                for k in range(n):
                    mkp, mkq = m[k][p], m[k][q]
                    m[k][p], m[k][q] = c * mkp - s * mkq, s * mkp + c * mkq
                for k in range(n):
                    mpk, mqk = m[p][k], m[q][k]
                    m[p][k], m[q][k] = c * mpk - s * mqk, s * mpk + c * mqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p], v[k][q] = c * vkp - s * vkq, s * vkp + c * vkq
    else:
        raise ArithmeticError("Jacobi eigen-decomposition did not converge")
    return [m[i][i] for i in range(n)], v


# ---------------------------------------------------------------- structural witnesses
@dataclass(frozen=True)
class Witness:
    """A different parameter point with the same outcome distribution (checked, not assumed)."""

    name: str
    description: str
    other: tuple[tuple[str, float], ...]  # the parameters that differ, at the other point
    max_abs_difference: float  # measured between the two outcome distributions


def _max_diff(p: Mapping, q: Mapping) -> float:
    return max(abs(p.get(o, 0.0) - q.get(o, 0.0)) for o in set(p) | set(q))


def _windows(cal: Calibration, design: Design) -> list[dict[str, int]]:
    """Back-action counts per window: the steps after one target read up to and including the next."""
    out, cur = [], {}
    for name in design.sequence:
        cur[name] = cur.get(name, 0) + 1
        if not cal.action(name).reads_reference:
            out.append(cur)
            cur = {}
    return out  # steps after the last target read never influence an x-reading


def _window_null_vector(cal: Calibration, design: Design, free: tuple[str, ...]):
    """A non-zero change of the free back-action factors that keeps every window sum, or None.

    Exact rational Gaussian elimination on the integer window-count matrix;
    returns (change per parameter name, dimension of the null space).
    """
    cols = [a for a in sorted(set(design.sequence)) if f"backaction:{a}" in free]
    if not cols:
        return None
    rows = [[Fraction(w.get(a, 0)) for a in cols] for w in _windows(cal, design)]
    pivots, r = [], 0
    for c in range(len(cols)):
        piv = next((i for i in range(r, len(rows)) if rows[i][c] != 0), None)
        if piv is None:
            continue
        rows[r], rows[piv] = rows[piv], rows[r]
        rows[r] = [x / rows[r][c] for x in rows[r]]
        for i in range(len(rows)):
            if i != r and rows[i][c] != 0:
                rows[i] = [x - rows[i][c] * y for x, y in zip(rows[i], rows[r], strict=True)]
        pivots.append(c)
        r += 1
    free_cols = [c for c in range(len(cols)) if c not in pivots]
    if not free_cols:
        return None
    fc = free_cols[0]
    vec = [Fraction(0)] * len(cols)
    vec[fc] = Fraction(1)
    for i, pc in enumerate(pivots):
        vec[pc] = -rows[i][fc]
    return {f"backaction:{a}": float(x) for a, x in zip(cols, vec, strict=True) if x != 0}, len(free_cols)


def _witnesses(cal: Calibration, design: Design, free: tuple[str, ...]) -> tuple[Witness, ...]:
    used = sorted(set(design.sequence))
    v = cal.parameters()
    cands = []
    betas = [f"backaction:{a}" for a in used]
    if {"k01", "k10", *betas} <= set(free) and (v["k01"] + v["k10"]) > 0:
        other = {"k01": 2 * v["k01"], "k10": 2 * v["k10"], **{b: v[b] / 2 for b in betas}}
        cands.append(
            (
                "drift_backaction_scale",
                "only the products k * backaction enter: doubling both rates and halving every used back-action "
                "factor changes nothing; fix one back-action factor (an anchor) to define the rate unit",
                other,
            )
        )
    if "prep_error" in free and not design.known_write:
        cands.append(
            (
                "prep_error_hidden",
                "without recording the written bit, x_start is uniform for every preparation error; "
                "sessions with known writes are required",
                {"prep_error": 0.25 if v["prep_error"] != 0.25 else 0.4},
            )
        )
    window = _window_null_vector(cal, design, free)
    if window is not None:
        delta, dim = window
        step = None
        for sign in (1, -1):  # largest move along +delta or -delta that keeps every factor >= 0
            room = [v[b] / abs(d) for b, d in delta.items() if sign * d < 0]
            if not room:
                step = sign * (max(v[b] for b in delta) or 1.0) * 0.5
                break
            if min(room) > 0:
                step = sign * 0.5 * min(room)
                break
        if step is not None:
            cands.append(
                (
                    "backaction_window",
                    "x is seen only by target reads, and all step generators are multiples of one rate matrix, so "
                    "they commute: only the sum of back-action factors between two target reads enters "
                    f"(window null space of dimension {dim} over the free back-action factors)",
                    {b: v[b] + step * d for b, d in delta.items()},
                )
            )
    no_ref = not any(cal.action(a).reads_reference for a in used)
    pb_ok = v["bias_prior"] == 0.5 or "bias_prior" in free
    if no_ref and not design.known_write and {"k01", "k10"} <= set(free) and v["k01"] != v["k10"] and pb_ok:
        cands.append(
            (
                "label_swap",
                "without a reference read, relabelling x <-> 1-x and b <-> 1-b with k01 <-> k10 leaves every "
                "target reading unchanged; one reference read breaks the symmetry",
                {"k01": v["k10"], "k10": v["k01"], "bias_prior": 1 - v["bias_prior"]},
            )
        )
    base = outcome_distribution(cal, design)
    out = []
    for name, text, other in cands:
        other = {k: x for k, x in other.items() if x != v[k]}
        d = _max_diff(base, outcome_distribution(cal.with_values(other), design))
        if not d <= WITNESS_TOL:
            raise ArithmeticError(f"witness {name!r} does not reproduce the distribution (diff {d:.3e})")
        out.append(Witness(name, text, tuple(sorted(other.items())), d))
    return tuple(out)


# ---------------------------------------------------------------- identifiability report
@dataclass(frozen=True)
class IdentifiabilityReport:
    design: Design
    free: tuple[str, ...]
    values: tuple[float, ...]
    fisher: tuple[tuple[float, ...], ...]  # per session
    rank: int
    locally_identifiable: bool  # Fisher information non-singular (Rothenberg 1971)
    verdict: str  # "GO", "NO-GO" (rank deficient) or "AMBIGUOUS" (full rank but a global witness)
    min_normalised_eigenvalue: float
    zero_information: tuple[str, ...]  # parameters the design carries no information about
    null_directions: tuple[tuple[float, ...], ...]  # original coordinates, largest entry +1
    witnesses: tuple[Witness, ...]
    relative_crlb: tuple[float | None, ...]  # per session: sqrt((F^-1)_ii) / |value|
    sessions_for_target: tuple[int | None, ...]  # sessions for relative SE <= target_rse
    target_rse: float
    energy_per_session: float
    notes: tuple[str, ...] = field(default=())


def identifiability(
    calibration: Calibration, design: Design, free: Iterable[str], target_rse: float = 0.1
) -> IdentifiabilityReport:
    """Local (Fisher) and global (witness) identifiability of ``free`` from sessions of ``design``."""
    if not _real(target_rse) or not target_rse > 0:
        raise ValueError("target_rse must be a positive finite number")
    free = _check_free(calibration, free)
    values = tuple(calibration.parameters()[f] for f in free)
    fim = fisher_information(calibration, design, free)
    n = len(free)
    diag = [fim[i][i] for i in range(n)]
    dead = [diag[i] * (values[i] if values[i] else 1.0) ** 2 <= ZERO_INFO_TOL for i in range(n)]
    zero = tuple(f for f, d in zip(free, dead, strict=True) if d)
    live = [i for i in range(n) if not dead[i]]
    nulls: list[list[float]] = [[1.0 if j == i else 0.0 for j in range(n)] for i in range(n) if dead[i]]
    inv = None
    min_eig = 0.0
    if live:
        s = [sqrt(diag[i]) for i in live]
        norm = [[fim[i][j] / (s[a] * s[b]) for b, j in enumerate(live)] for a, i in enumerate(live)]
        lam, vec = _eigh(norm)
        min_eig = min(lam)
        for k, lk in enumerate(lam):
            if lk <= RANK_TOL:
                u = [0.0] * n
                for a, i in enumerate(live):
                    u[i] = vec[a][k] / s[a]
                nulls.append(u)
        if not zero and min_eig > RANK_TOL:
            m = len(live)
            ninv = [[sum(vec[a][k] * vec[b][k] / lam[k] for k in range(m)) for b in range(m)] for a in range(m)]
            inv = [[ninv[a][b] / (s[a] * s[b]) for b in range(m)] for a in range(m)]
    if zero:
        min_eig = 0.0
    null_dirs = []
    for u in nulls:
        big = max(u, key=abs)
        null_dirs.append(tuple(x / big if abs(x / big) > 1e-12 else 0.0 for x in u))
    rank = n - len(null_dirs)
    witnesses = _witnesses(calibration, design, free)
    local = rank == n
    if not local:
        verdict = "NO-GO"
    elif witnesses:
        verdict = "AMBIGUOUS"
    else:
        verdict = "GO"
    rel: list[float | None] = []
    sess: list[int | None] = []
    for i in range(n):
        if inv is None or values[i] == 0.0:
            rel.append(None)
            sess.append(None)
            continue
        r = sqrt(inv[i][i]) / abs(values[i])
        rel.append(r)
        sess.append(max(1, ceil((r / target_rse) ** 2)))
    notes = []
    if any(v == 0.0 for v in values):
        notes.append("a free parameter sits on the boundary (value 0): Rothenberg's regularity conditions fail there")
    energy = sum(calibration.action(a).energy for a in design.sequence)
    return IdentifiabilityReport(
        design,
        free,
        values,
        fim,
        rank,
        local,
        verdict,
        min_eig,
        zero,
        tuple(null_dirs),
        witnesses,
        tuple(rel),
        tuple(sess),
        float(target_rse),
        energy,
        tuple(notes),
    )


# ---------------------------------------------------------------- model discrimination
@dataclass(frozen=True)
class Discrimination:
    design: Design
    kl_ab: float  # KL(P_a || P_b) per session, nats
    kl_ba: float
    chernoff: float  # Chernoff information per session, nats
    chernoff_lambda: float
    sessions_for_error: int | None  # N with equal-prior Bayes error <= 0.5 * exp(-N C) <= alpha
    alpha: float
    energy_per_session: float


def _kl(p: Mapping, q: Mapping) -> float:
    s = 0.0
    for o, po in p.items():
        if po > 0:
            qo = q.get(o, 0.0)
            if qo <= 0:
                return float("inf")
            s += po * log(po / qo)
    return max(s, 0.0)


def _chernoff(p: Mapping, q: Mapping) -> tuple[float, float]:
    keys = [o for o in set(p) | set(q) if p.get(o, 0.0) > 0 and q.get(o, 0.0) > 0]

    def g(lam: float) -> float:
        z = sum(p[o] ** lam * q[o] ** (1 - lam) for o in keys)
        return log(z) if z > 0 else float("-inf")

    lo, hi = 0.0, 1.0
    r = (sqrt(5) - 1) / 2
    a, b = hi - r * (hi - lo), lo + r * (hi - lo)
    ga, gb = g(a), g(b)
    for _ in range(200):  # g is convex in lambda: golden-section search for its minimum
        if ga <= gb:
            hi, b, gb = b, a, ga
            a = hi - r * (hi - lo)
            ga = g(a)
        else:
            lo, a, ga = a, b, gb
            b = lo + r * (hi - lo)
            gb = g(b)
    lam = 0.5 * (lo + hi)
    return max(-g(lam), 0.0), lam


def discrimination(cal_a: Calibration, cal_b: Calibration, design: Design, alpha: float = 0.01) -> Discrimination:
    """How well sessions of ``design`` tell calibration ``a`` from ``b`` (e.g. two model hypotheses)."""
    if not _real(alpha) or not 0 < alpha < 0.5:
        raise ValueError("alpha must be a finite number in (0, 0.5)")
    if [(a.name, a.energy, a.reads_reference) for a in cal_a.actions] != [
        (a.name, a.energy, a.reads_reference) for a in cal_b.actions
    ]:
        raise ValueError("both calibrations need the same action catalogue (names, energy, reference flag)")
    p, q = outcome_distribution(cal_a, design), outcome_distribution(cal_b, design)
    c, lam = _chernoff(p, q)
    if c <= 1e-15:  # rounding floor of sum p^l q^(1-l) near one
        c = 0.0
    n = None if c == 0.0 else max(1, ceil(log(1 / (2 * alpha)) / c))
    energy = sum(cal_a.action(a).energy for a in design.sequence)
    return Discrimination(design, _kl(p, q), _kl(q, p), c, lam, n, float(alpha), energy)


# ---------------------------------------------------------------- design search
@dataclass(frozen=True)
class DesignScore:
    design: Design
    verdict: str
    log_det: float | None  # log det of the per-session Fisher information (D-optimality); None if singular
    worst_sessions: int | None  # max over parameters of the sessions needed for target_rse
    energy_per_session: float


def design_search(
    calibration: Calibration,
    free: Iterable[str],
    length: int,
    known_write: Sequence[bool] = (False, True),
    target_rse: float = 0.1,
) -> list[DesignScore]:
    """Score every fixed sequence of ``length`` actions; best first.

    Order: verdict GO first, then larger ``log_det`` (D-optimal), then fewer
    sessions for the worst parameter, then less energy, then catalogue order.
    """
    if isinstance(length, bool) or not isinstance(length, int) or not 1 <= length <= MAX_DESIGN_LENGTH:
        raise ValueError(f"length must be an int in 1..{MAX_DESIGN_LENGTH}")
    modes = tuple(known_write)
    if not modes or any(not isinstance(m, bool) for m in modes) or len(set(modes)) != len(modes):
        raise ValueError("known_write must be distinct booleans")
    names = [a.name for a in calibration.actions]
    count = len(names) ** length * len(modes)
    if count > MAX_SEARCH_DESIGNS:
        raise ValueError(f"{count} designs exceed MAX_SEARCH_DESIGNS={MAX_SEARCH_DESIGNS}")
    free = _check_free(calibration, free)
    scores = []
    for order, (mode, seq) in enumerate((m, s) for m in modes for s in product(names, repeat=length)):
        rep = identifiability(calibration, Design(seq, mode), free, target_rse)
        ld = None
        if rep.locally_identifiable:
            lam, _ = _eigh(rep.fisher)
            ld = sum(log(x) for x in lam) if min(lam) > 0 else None
        worst = None if None in rep.sessions_for_target else max(rep.sessions_for_target)
        scores.append((order, DesignScore(rep.design, rep.verdict, ld, worst, rep.energy_per_session)))
    rank = {"GO": 0, "AMBIGUOUS": 1, "NO-GO": 2}

    def key(t):
        order, s = t
        return (
            rank[s.verdict],
            -(s.log_det if s.log_det is not None else float("-inf")),
            s.worst_sessions if s.worst_sessions is not None else float("inf"),
            s.energy_per_session,
            order,
        )

    return [s for _, s in sorted(scores, key=key)]

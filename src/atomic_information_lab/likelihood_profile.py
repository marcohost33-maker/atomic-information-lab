"""Profile likelihood for practical identifiability (roadmap R29 / P1; Raue et al. 2009).

The Cramer-Rao counts of ``identifiability`` are *local*: they read the
curvature of the log-likelihood at one point and assume it stays quadratic.
Raue et al. (2009, Bioinformatics 25:1923, doi:10.1093/bioinformatics/btp358)
replace the curvature by the **profile likelihood**: fix one parameter at a
value ``c``, re-optimise all others, and see how fast the fit gets worse. A
parameter is *practically* identifiable from given data when its
likelihood-based confidence interval ``{c : PL statistic(c) <= Delta}`` is
bounded on both sides; a *structurally* non-identifiable parameter has a flat
profile. Wieland et al. (2021, Curr. Opin. Syst. Biol. 25:60,
doi:10.1016/j.coisb.2021.03.005) recommend it over the Fisher matrix for
exactly this reason.

**Expected data, no sampling.** This module profiles the *expected* outcome
counts ``N * p*(o)`` of N sessions at the true point (the "Asimov" data set of
Cowan, Cranmer, Gross & Vitells 2011, Eur. Phys. J. C 71:1554,
doi:10.1140/epjc/s10052-011-1554-0). The maximum-likelihood estimate on these
data is the truth itself, and the profile likelihood-ratio statistic is

    Lambda_N(c) = 2 N * min over the other free parameters of KL(p* || p_(c, others)),

i.e. ``N`` times a per-session statistic ``pi(c)`` that is computed once. Hence

* the **95 % profile interval** after N sessions is the connected set of ``c``
  around the truth with ``pi(c) <= Delta / N``, ``Delta = 3.8415`` (the 95 %
  quantile of chi-square with one degree of freedom: pointwise, Wilks);
* the interval is **bounded within the scan** once ``N > Delta / max pi`` on
  each side -- ``bounded_sessions``; with fewer sessions it runs into a domain
  boundary or the scan limit (practical non-identifiability in Raue's sense);
* ``profile_sessions`` is the likelihood-ratio analogue of the Cramer-Rao
  count: the smallest N whose ``Delta = 1`` interval lies within
  ``truth * (1 +- target_rse)``. If the profile were exactly quadratic it would
  equal the Cramer-Rao count, so ``profile_to_crlb`` measures how far the
  local picture is off, and ``asymmetry`` = ``pi(+r) / pi(-r)`` in which
  direction.

What this is and is not (``PL_SCOPE``): exact within the synthetic model up to
the inner optimisation, at one parameter point and one design, for the
expected data set -- it describes the *median-like* experiment, not the
spread of intervals over repeated experiments, and Wilks' chi-square
calibration is asymptotic (its coverage at finite N is not measured here). The
scan is a finite grid up to ``SCAN_CAP`` times the true value (or the domain
bound): "bounded" means bounded within that grid, and counts read off the grid
are conservative (a finer grid can only lower them). Nothing here describes an
instrument.

Inner optimisation: Fisher scoring with Levenberg damping on the per-session
Kullback-Leibler divergence, started twice per grid point (from the neighbour
towards the truth and from the true values of the other parameters); the lower
value is kept. A point whose optimisation does not converge is reported and is
never used as evidence that an interval is bounded (a non-converged value
over-estimates the profile).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import exp, floor, isfinite, log1p
from statistics import NormalDist

from .identifiability import SESSIONS_SCOPE, Calibration, Design, _check_free, _leaves, _real, identifiability

__all__ = [
    "PL_SCOPE",
    "PL_SCOPE_TEXT",
    "SCAN_CAP",
    "FLAT_TOL",
    "ProfilePoint",
    "LikelihoodInterval",
    "ParameterProfile",
    "profile_statistic",
    "profile_likelihood",
    "profile_report",
    "chi2_1_quantile",
]

PL_SCOPE = "EXPECTED_DATA_PROFILE_MODEL_INTERNAL_ONE_POINT"
PL_SCOPE_TEXT = (
    "profile likelihood on the EXPECTED data of N sessions (Asimov data set), model-internal, one parameter "
    "point, one design (pl_scope: EXPECTED_DATA_PROFILE_MODEL_INTERNAL_ONE_POINT): intervals use Wilks' "
    "asymptotic chi-square threshold, pointwise (1 degree of freedom); 'bounded' means bounded within the "
    "scan range, and counts read off the scan grid are conservative; the spread of intervals over repeated "
    "experiments and their finite-N coverage are not measured"
)
SCAN_CAP = 64.0  # upper scan limit for parameters without a finite upper bound, in multiples of the truth
FLAT_TOL = 1e-12  # per-session statistic below which a profile point counts as flat (no information)
_GRID_DEPTH = 10  # the scan approaches a boundary up to 2**-10 of the distance
_INNER_ITER = 200
_INNER_REL_STEP = 1e-10
_INNER_DECREMENT = 1e-10  # converged when scoring promises < 1e-10 relative improvement of the divergence
_LOWER = 1e-12  # nuisance parameters stay >= this (the domain is open at 0 for relative scales)
_BISECT_REL = 1e-4  # interval end points to 1e-4 of their distance from the truth
_UPPER = {"bias_flip": 0.5, "prep_error": 0.5, "bias_prior": 1.0}


def chi2_1_quantile(level: float) -> float:
    """Quantile of chi-square with one degree of freedom: ``z_((1+level)/2) ** 2``."""
    if not _real(level) or not 0 < level < 1:
        raise ValueError("level must be a number in (0, 1)")
    z = NormalDist().inv_cdf(0.5 + level / 2)
    return z * z


def _upper_bound(name: str) -> float | None:
    if name in _UPPER:
        return _UPPER[name]
    if name.startswith("error:"):
        return 0.5
    return None  # rates and back-action factors: unbounded above


def _h(p: float, q: float) -> float:
    """``q * (u - 1 - log u)`` with ``u = p / q``: one term of KL(q || p), >= 0, accurate near u = 1."""
    d = (p - q) / q
    if abs(d) < 1e-3:  # series of d - log1p(d) = d^2/2 - d^3/3 + ..., no cancellation
        s, t = 0.0, d * d
        for k in range(2, 12):
            s += t / k if k % 2 == 0 else -t / k
            t *= d
        return q * s
    return q * (d - log1p(d))


def _kl(target: dict, cal: Calibration, design: Design) -> float | None:
    total = 0.0
    for o, p in _leaves(cal, design, ()):
        q = target[o]
        if q == 0.0:
            continue
        if not p.v > 0.0:
            return None
        total += _h(p.v, q)
    return total


def _solve_damped(fim: list[list[float]], score: list[float]) -> list[float] | None:
    """Levenberg-damped Fisher-scoring step; damping grows until the system is solvable."""
    from .estimation import _solve

    n = len(score)
    floor_ = 1e-12 * max((fim[i][i] for i in range(n)), default=1.0) or 1.0
    diag = [max(fim[i][i], floor_) for i in range(n)]
    for mu in (0.0, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1.0):
        a = [[fim[i][j] + (mu * diag[i] if i == j else 0.0) for j in range(n)] for i in range(n)]
        step = _solve(a, score)
        if step is not None and all(isfinite(s) for s in step):
            return step
    return None


@dataclass(frozen=True)
class ProfilePoint:
    value: float  # the profiled parameter is fixed here
    statistic: float  # per session: 2 * min KL(p* || p); N sessions give N times this value
    converged: bool
    reason: str  # "converged" or why not
    nuisance: tuple[tuple[str, float], ...]  # optimum of the other free parameters at this point


def _inner(
    truth: Calibration, design: Design, name: str, value: float, nuis: tuple[str, ...], start: dict, target: dict
) -> tuple[float, dict, bool, str]:
    """min over ``nuis`` of KL(p* || p) with ``name`` fixed at ``value``, inside the parameter domain.

    Projected Fisher scoring in log coordinates: a nuisance parameter that sits on a domain bound
    while the gradient pushes it outwards is held there (active set), the
    others take a damped scoring step, the trial point is projected into the
    domain and halved until the divergence does not increase. Converged when
    the scoring decrement promises less than ``_INNER_DECREMENT`` relative
    improvement, or the step is below ``_INNER_REL_STEP`` relative.
    """
    theta = dict(start)
    try:
        cal = truth.with_values({name: value, **theta})
    except ValueError:
        return float("inf"), theta, False, "start outside the domain"
    cur = _kl(target, cal, design)
    if cur is None:
        return float("inf"), theta, False, "start has an impossible observed outcome"
    if not nuis:
        return cur, theta, True, "converged"
    k = len(nuis)
    hi = [_upper_bound(f) for f in nuis]
    for _ in range(_INNER_ITER):
        score = [0.0] * k
        fim = [[0.0] * k for _ in range(k)]
        for o, p in _leaves(cal, design, nuis):
            if not p.v > 0.0:
                continue
            r = (target[o] - p.v) / p.v  # score of the expected log-likelihood, cancellation-free
            for i in range(k):
                gi = p.g[i]
                if gi:
                    score[i] += r * gi
                    gi /= p.v
                    row = fim[i]
                    for j in range(k):
                        row[j] += gi * p.g[j]
        vals = [theta[f] for f in nuis]
        # log coordinates (u = log value): rates and back-action factors span orders of magnitude, and
        # the drift/back-action valley (only k * beta enters) is straight in log space
        for i in range(k):
            score[i] *= vals[i]
            for j in range(k):
                fim[i][j] *= vals[i] * vals[j]
        at_low = [vals[i] <= _LOWER * (1 + 1e-9) and score[i] < 0 for i in range(k)]
        at_high = [hi[i] is not None and vals[i] >= hi[i] and score[i] > 0 for i in range(k)]
        held = [a or b for a, b in zip(at_low, at_high, strict=True)]
        # a parameter the data say nothing about here (e.g. a drift rate while every reading is pure noise)
        # cannot move the divergence: it keeps its value instead of making the system singular
        top = max(fim[i][i] for i in range(k))
        mute = [fim[i][i] <= 1e-24 * top for i in range(k)]
        idx = [i for i in range(k) if not held[i] and not mute[i]]
        if not idx:
            return cur, theta, True, _bound_reason(nuis, held)
        sub = _solve_damped([[fim[i][j] for j in idx] for i in idx], [score[i] for i in idx])
        if sub is None:
            return cur, theta, False, "singular Fisher information"
        step = [0.0] * k
        for a, i in enumerate(idx):
            step[i] = sub[a]
        decrement = sum(step[i] * score[i] for i in idx)  # about twice the promised improvement
        if cur == 0.0 or decrement <= _INNER_DECREMENT * cur:
            return cur, theta, True, _bound_reason(nuis, held)
        t = 1.0
        while True:
            new = {f: _project(vals[i] * _exp(t * step[i]), hi[i]) for i, f in enumerate(nuis)}
            cand = None
            try:
                cal_new = truth.with_values({name: value, **new})
                cand = _kl(target, cal_new, design)
            except (ValueError, ArithmeticError):
                cand = None
            if cand is not None and cand <= cur * (1 + 1e-12):
                break
            t *= 0.5
            if t < 2**-40:
                return cur, theta, False, "no admissible descent step"
        rel = max(abs(new[f] - vals[i]) / max(abs(new[f]), _LOWER) for i, f in enumerate(nuis))
        theta, cal, cur = new, cal_new, cand
        if rel < _INNER_REL_STEP:
            return cur, theta, True, _bound_reason(nuis, held)
    return cur, theta, False, "iteration limit"


def _exp(u: float) -> float:
    return exp(min(u, 50.0))  # a log step above 50 (factor 5e21) is cut; the line search halves it anyway


def _project(v: float, hi: float | None) -> float:
    v = max(v, _LOWER)
    return v if hi is None else min(v, hi)


def _bound_reason(nuis: tuple[str, ...], held: list[bool]) -> str:
    at = [f for f, h in zip(nuis, held, strict=True) if h]
    return "converged" if not at else "converged on the domain bound of " + ", ".join(at)


def profile_statistic(
    truth: Calibration, design: Design, free: Iterable[str], name: str, value: float, start: dict | None = None
) -> ProfilePoint:
    """Per-session profile statistic ``2 * min KL(p* || p)`` with ``name`` fixed at ``value``.

    The minimum runs over the other parameters in ``free``; it is started from
    ``start`` (if given) and from their true values, and the lower result is kept.
    """
    free = _check_free(truth, free)
    if name not in free:
        raise ValueError(f"{name!r} is not a free parameter")
    if not _real(value) or not value > 0:
        raise ValueError("value must be a positive finite number")
    ub = _upper_bound(name)
    if ub is not None and value > ub:
        raise ValueError(f"value must be <= {ub} for {name}")
    truth_values = truth.parameters()
    nuis = tuple(f for f in free if f != name)
    target = {o: p.v for o, p in _leaves(truth, design, ())}
    if not all(isfinite(q) and q >= 0 for q in target.values()) or abs(sum(target.values()) - 1.0) > 1e-12:
        raise ArithmeticError("outcome probabilities are not a distribution")
    cold = {f: truth_values[f] for f in nuis}
    starts = [cold] if start is None else [dict(start), cold]
    best = None
    for s in starts:
        v, th, ok, why = _inner(truth, design, name, value, nuis, s, target)
        if best is None or (ok and not best[2]) or (ok == best[2] and v < best[0]):
            best = (v, th, ok, why)
    v, th, ok, why = best
    stat = 2.0 * v if isfinite(v) else float("inf")
    return ProfilePoint(float(value), stat, ok, why, tuple(sorted(th.items())))


@dataclass(frozen=True)
class LikelihoodInterval:
    sessions: int
    level: float
    threshold: float  # Delta / N on the per-session statistic
    low: float | None  # None: no converged point above the threshold down to the scan limit
    high: float | None
    low_open: bool
    high_open: bool
    bounded: bool  # both ends found within the scan (practically identifiable at this N, in Raue's sense)
    pl_scope: str = PL_SCOPE


@dataclass(frozen=True)
class ParameterProfile:
    name: str
    true_value: float
    target_rse: float
    crlb_rse: float | None  # per session, relative (identifiability.relative_crlb)
    crlb_sessions: int | None  # Cramer-Rao count for target_rse (lower bound for unbiased estimators)
    profile_sessions: int | None  # likelihood-ratio analogue (Delta = 1 interval within +-target_rse)
    profile_to_crlb: float | None  # unrounded ratio of the two counts; 1 = the profile is locally quadratic
    asymmetry: float | None  # pi(truth*(1+r)) / pi(truth*(1-r)); < 1: flatter above the truth
    low_plateau: float  # largest converged statistic on the lower scan
    high_plateau: float
    bounded_sessions: int | None  # smallest N whose 95 % interval is bounded within the scan; None: never
    flat: bool  # every converged statistic <= FLAT_TOL: no information (structural)
    scan_low: float  # smallest value scanned
    scan_high: float  # largest value scanned
    points: tuple[ProfilePoint, ...]  # ascending in value; the truth itself is not a point (statistic 0)
    not_converged: int
    intervals: tuple[LikelihoodInterval, ...]
    # Outward monotonicity per side (converged points only). A side that rises and falls again has a
    # second valley: the likelihood-ratio confidence SET is then not one interval, and the dip is the
    # smallest statistic after the rise (0 for an exact global ambiguity such as the label swap).
    low_monotone: bool = True
    high_monotone: bool = True
    low_dip: float | None = None
    high_dip: float | None = None
    pl_scope: str = PL_SCOPE  # scope of every profile figure (profile_sessions, bounded_sessions, intervals)
    sessions_scope: str = SESSIONS_SCOPE  # scope of the Cramer-Rao figures crlb_rse / crlb_sessions


def _dip(stats: list[float]) -> float | None:
    """Smallest value after the running maximum fell back by more than 1e-6 relative (else None)."""
    top, dip = 0.0, None
    for v in stats:
        if v > top:
            top = v
        elif v < top * (1 - 1e-6) and top > FLAT_TOL:
            dip = v if dip is None else min(dip, v)
    return dip


def _side_offsets(distance: float, step: float) -> list[float]:
    """Positive offsets from the truth towards a boundary ``distance`` away, finest near the truth."""
    near = [step / 4, step / 2, step]
    mid = [step * 2.0**k for k in range(1, 64) if step * 2.0**k < distance / 2]
    far = [distance * (1 - 2.0**-k) for k in range(1, _GRID_DEPTH + 1)]
    return sorted({x for x in near + mid + far if 0 < x < distance})


def _scan(truth, design, free, name, offsets, sign):
    """March from the truth outwards; every point is warm-started from its inner neighbour."""
    tv = truth.parameters()[name]
    nuis = [f for f in free if f != name]
    warm = {f: truth.parameters()[f] for f in nuis}
    out = []
    for d in offsets:
        pt = profile_statistic(truth, design, free, name, tv + sign * d, warm)
        out.append(pt)
        if pt.converged:
            warm = dict(pt.nuisance)
    return out


def _bisect(truth, design, free, name, inner_pt, outer_pt, threshold):
    """End point between a converged point below and a converged point above the threshold."""
    a, b = inner_pt, outer_pt
    tv = truth.parameters()[name]
    while abs(b.value - a.value) > _BISECT_REL * abs(b.value - tv):
        mid = 0.5 * (a.value + b.value)
        m = profile_statistic(truth, design, free, name, mid, dict(a.nuisance))
        if not m.converged:
            return b.value  # conservative: the outer, verified point
        if m.statistic > threshold:
            b = m
        else:
            a = m
    return b.value


def _interval(truth, design, free, name, low_pts, high_pts, sessions, level) -> LikelihoodInterval:
    threshold = chi2_1_quantile(level) / sessions
    tv = truth.parameters()[name]
    nuis0 = tuple(sorted((f, truth.parameters()[f]) for f in free if f != name))
    centre = ProfilePoint(tv, 0.0, True, "truth", nuis0)
    ends = []
    for pts in (low_pts, high_pts):
        prev, end = centre, None
        for pt in pts:  # outward order
            if pt.converged and pt.statistic > threshold:
                end = _bisect(truth, design, free, name, prev, pt, threshold)
                break
            if pt.converged:
                prev = pt
        ends.append(end)
    low, high = ends
    return LikelihoodInterval(
        sessions, float(level), threshold, low, high, low is None, high is None, low is not None and high is not None
    )


def profile_likelihood(
    truth: Calibration,
    design: Design,
    free: Iterable[str],
    name: str,
    target_rse: float = 0.1,
    sessions: Sequence[int] = (),
    level: float = 0.95,
) -> ParameterProfile:
    """Profile of one free parameter on the expected data, with intervals at the given session counts."""
    free = _check_free(truth, free)
    if name not in free:
        raise ValueError(f"{name!r} is not a free parameter")
    if not _real(target_rse) or not 0 < target_rse < 1:
        raise ValueError("target_rse must be a number in (0, 1)")
    if isinstance(sessions, (str, bytes)) or not all(
        isinstance(n, int) and not isinstance(n, bool) and 1 <= n <= 10**15 for n in sessions
    ):
        raise ValueError("sessions must be ints in 1..1e15")
    delta = chi2_1_quantile(level)
    tv = truth.parameters()[name]
    if not tv > 0:
        raise ValueError(f"{name} must be > 0 at the truth (interior point; relative offsets)")
    ub = _upper_bound(name)
    if ub is not None and not tv < ub:
        raise ValueError(f"{name} must lie below its upper bound {ub} at the truth")
    report = identifiability(truth, design, free, target_rse)
    i = report.free.index(name)
    crlb_rse = report.relative_crlb[i]
    crlb_sessions = report.sessions_for_target[i]
    step = target_rse * tv
    hi_dist = (ub - tv) if ub is not None else (SCAN_CAP - 1.0) * tv
    if step >= hi_dist:
        raise ValueError("truth * (1 + target_rse) reaches the upper domain bound; choose a smaller target_rse")
    low_off = _side_offsets(tv, step)
    high_off = _side_offsets(hi_dist, step)
    low_pts = _scan(truth, design, free, name, low_off, -1.0)
    high_pts = _scan(truth, design, free, name, high_off, +1.0)
    conv_low = [p.statistic for p in low_pts if p.converged]
    conv_high = [p.statistic for p in high_pts if p.converged]
    low_plateau = max(conv_low, default=0.0)
    high_plateau = max(conv_high, default=0.0)
    flat = max(low_plateau, high_plateau) <= FLAT_TOL
    pr = min(low_plateau, high_plateau)
    bounded_sessions = None if pr <= FLAT_TOL else floor(delta / pr) + 1
    low_dip, high_dip = _dip(conv_low), _dip(conv_high)

    def at(pts, off):
        for p, d in zip(pts, off, strict=True):
            if d == step:
                return p
        raise AssertionError("target offset missing from the grid")

    lo_r, hi_r = at(low_pts, low_off), at(high_pts, high_off)
    # Delta = 1 interval within +-r: some converged point on each side within r must exceed 1/N
    in_low = [p.statistic for p, d in zip(low_pts, low_off, strict=True) if d <= step and p.converged]
    in_high = [p.statistic for p, d in zip(high_pts, high_off, strict=True) if d <= step and p.converged]
    m = min(max(in_low, default=0.0), max(in_high, default=0.0))
    profile_sessions = None if m <= FLAT_TOL else floor(1.0 / m) + 1
    ratio = None
    if profile_sessions is not None and crlb_rse is not None:
        ratio = (1.0 / m) / (crlb_rse / target_rse) ** 2
    asym = None
    if lo_r.converged and hi_r.converged and lo_r.statistic > FLAT_TOL:
        asym = hi_r.statistic / lo_r.statistic
    intervals = tuple(_interval(truth, design, free, name, low_pts, high_pts, n, level) for n in sessions)
    points = tuple(reversed(low_pts)) + tuple(high_pts)
    return ParameterProfile(
        name,
        tv,
        float(target_rse),
        crlb_rse,
        crlb_sessions,
        profile_sessions,
        ratio,
        asym,
        low_plateau,
        high_plateau,
        bounded_sessions,
        flat,
        points[0].value,
        points[-1].value,
        points,
        sum(not p.converged for p in points),
        intervals,
        low_dip is None,
        high_dip is None,
        low_dip,
        high_dip,
    )


def profile_report(
    truth: Calibration,
    design: Design,
    free: Iterable[str],
    names: Iterable[str] | None = None,
    target_rse: float = 0.1,
    sessions: Sequence[int] = (),
    level: float = 0.95,
) -> tuple[ParameterProfile, ...]:
    """``profile_likelihood`` for several free parameters (default: all of them)."""
    free = _check_free(truth, free)
    names = free if names is None else tuple(names)
    if isinstance(names, str) or not names or any(n not in free for n in names):
        raise ValueError("names must be free parameters")
    return tuple(profile_likelihood(truth, design, free, n, target_rse, sessions, level) for n in names)

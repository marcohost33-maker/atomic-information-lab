"""Retention limit P(x != x0) <= delta (roadmap R13/P0, F29) against an exact, independent reference.

Reference: the oracle's rational forward messages (tests/_oracle.py), with
EVERY deterministic policy enumerated (7204 at h = 3 with three actions).
Each policy is reduced to exact numbers (J, C, worst stopping posterior):

* pathwise optimum  = min J over policies whose every stop has P(x != x0 | data) <= delta;
* expected optimum  = lower convex hull of all (C, J) at C <= delta (mixtures of
  deterministic policies span every randomised policy of a finite tree);
* deterministic expected optimum = min J over C <= delta.

All of it in Fractions; nothing is shared with the package's recursion.
"""

import math
from fractions import Fraction as Fr

import pytest
from test_oracle_agreement import CASES, make

from atomic_information_lab import STOP, Hypothesis, Problem, solve
from atomic_information_lab.reproduce import retention_table
from atomic_information_lab.retention import (
    best_deterministic_retention,
    best_open_loop_retention,
    lagrangian_policy,
    pathwise_policy,
    retention_profile,
    solve_retention,
)

TOL = 1e-10
DELTAS = (0.0, 0.001, 0.02, 0.05, 0.08, 0.1, 0.12, 0.2, 1.0)


def enumerate_policies(model, m, h):
    """Every deterministic policy from message m as (J, C, worst stop posterior), mass-weighted, exact."""
    tot = sum(m)
    if tot == 0:
        return [(Fr(0), Fr(0), Fr(0))]
    mis = sum(v for v, s in zip(m, model.states, strict=True) if s[1] != s[2])
    out = [(model.stop_mass(m), mis, mis / tot)]
    if h == 0:
        return out
    for a in model.actions:
        c = model.cost(a) * tot
        k0 = enumerate_policies(model, model.step(m, a, 0), h - 1)
        k1 = enumerate_policies(model, model.step(m, a, 1), h - 1)
        out += [(c + j0 + j1, c0 + c1, max(p0, p1)) for j0, c0, p0 in k0 for j1, c1, p1 in k1]
    return out


def lower_hull(points):
    pts = sorted(set((c, j) for j, c, _ in points))
    hull = []
    for p in pts:
        while len(hull) >= 2:
            (c1, j1), (c2, j2) = hull[-2], hull[-1]
            if (j2 - j1) * (p[0] - c1) >= (p[1] - j1) * (c2 - c1):  # hull[-1] not strictly below the chord
                hull.pop()
            else:
                break
        hull.append(p)
    return hull


def hull_min(hull, delta):
    """min J over the convex hull restricted to C <= delta (the hull is convex, so check vertices and delta)."""
    best = min((j for c, j in hull if c <= delta), default=None)
    for (c1, j1), (c2, j2) in zip(hull, hull[1:], strict=False):
        if c1 <= delta < c2:
            at = j1 + (j2 - j1) * (delta - c1) / (c2 - c1)
            best = at if best is None else min(best, at)
    return best


REFERENCE_CASES = ["default", "two_models", "retention", "lambda0", "fast_dynamics", "flip_half"]


@pytest.fixture(scope="module")
def references():
    out = {}
    for name in REFERENCE_CASES:
        pkg, orc = make(**CASES[name])
        model = orc(True)
        pols = enumerate_policies(model, model.root(), 3)
        out[name] = (pkg, pols, lower_hull(pols))
    return out


def test_enumeration_counts_every_policy(references):
    _, pols, _ = references["default"]
    assert len(pols) == 7204  # the README's count of deterministic policies with STOP at h = 3


@pytest.mark.parametrize("name", REFERENCE_CASES)
def test_pathwise_matches_exhaustive_search(references, name):
    pkg, pols, _ = references[name]
    for delta in DELTAS:
        d = Fr(delta)
        exact = min(j for j, _, worst in pols if worst <= d)
        r = solve_retention(pkg, delta, 3, "pathwise")
        assert r.value == pytest.approx(float(exact), abs=TOL), delta
        assert r.p_changed <= delta + 1e-15


@pytest.mark.parametrize("name", REFERENCE_CASES)
def test_expected_matches_the_convex_hull(references, name):
    pkg, pols, hull = references[name]
    for delta in DELTAS:
        d = Fr(delta)
        r = solve_retention(pkg, delta, 3, "expected")
        assert r.value == pytest.approx(float(hull_min(hull, d)), abs=TOL), delta
        assert r.p_changed <= delta + 1e-12
        assert r.duality_gap == pytest.approx(0.0, abs=1e-10)
        assert r.dual_bound <= r.value + 1e-10
        # the best deterministic policy (Pareto-front DP) equals the exhaustive deterministic optimum
        best_det = min(j for j, c, _ in pols if c <= d)
        assert r.deterministic_p_changed <= delta + 1e-12
        assert r.deterministic_value == pytest.approx(float(best_det), abs=TOL), delta


def test_randomisation_is_worth_something_and_never_hurts(references):
    """Constrained MDPs have a duality gap over deterministic policies: the mixture can be strictly
    better than every deterministic policy -- recorded, because a deterministic user pays it."""
    gaps = []
    for name in REFERENCE_CASES:
        pkg, _, _ = references[name]
        for delta in DELTAS:
            r = solve_retention(pkg, delta, 3, "expected")
            gaps.append(r.deterministic_value - r.value)
    assert min(gaps) >= -TOL
    assert max(gaps) > 1e-3


@pytest.mark.parametrize("delta", [0.0, 0.05, 0.08, 0.13])
def test_deterministic_plan_reproduces_its_value(delta):
    pr = Problem((Hypothesis(0.07, 0.045, 0.025), Hypothesis(0.23, 0.16, 0.1)), (0.5, 0.5))
    value, changed, policy, size = best_deterministic_retention(pr, delta, 4)
    prof = retention_profile(pr, policy, 4, 1.0)
    assert prof["value"] == pytest.approx(value, abs=1e-12)
    assert prof["p_changed"] == pytest.approx(changed, abs=1e-12) and changed <= delta + 1e-15
    assert size >= 1


# ---------------------------------------------------------------- mixture and certificate
@pytest.mark.parametrize("delta", [0.001, 0.05, 0.1])
def test_mixture_components_reproduce_value_and_risk(delta):
    pr = Problem()
    r = solve_retention(pr, delta, 4, "expected")
    assert len(r.components) == 2 and math.isclose(sum(w for w, _ in r.components), 1.0)
    parts = [retention_profile(pr, lagrangian_policy(pr, mu), 4, delta) for _, mu in r.components]
    value = sum(w * p["value"] for (w, _), p in zip(r.components, parts, strict=True))
    changed = sum(w * p["p_changed"] for (w, _), p in zip(r.components, parts, strict=True))
    assert value == pytest.approx(r.value, abs=1e-12)
    assert changed == pytest.approx(delta, abs=1e-12)  # the limit is active and met exactly


def test_limit_that_does_not_bind_returns_the_unconstrained_optimum():
    pr = Problem()
    for mode in ("pathwise", "expected"):
        r = solve_retention(pr, 1.0, 4, mode)
        assert r.value == pytest.approx(solve(pr, horizon=4).value, abs=1e-15)
        assert r.multiplier in (None, 0.0)


# ---------------------------------------------------------------- properties
@pytest.mark.parametrize("h", [2, 3, 4])
def test_ordering_and_monotonicity_in_delta(h):
    pr = Problem((Hypothesis(0.07, 0.045, 0.025), Hypothesis(0.23, 0.16, 0.1)), (0.5, 0.5))
    unconstrained = solve(pr, horizon=h).value
    prev_p = prev_e = math.inf
    for delta in DELTAS:
        p = solve_retention(pr, delta, h, "pathwise")
        e = solve_retention(pr, delta, h, "expected")
        assert unconstrained - 1e-12 <= e.value <= e.deterministic_value + 1e-12 <= p.value + 1e-12
        assert p.value <= pr.terminal_loss(pr.initial_belief()) + 1e-12  # STOP is always admissible
        assert p.value <= prev_p + 1e-12 and e.value <= prev_e + 1e-12
        prev_p, prev_e = p.value, e.value


def test_zero_limit_allows_measuring_only_when_nothing_can_change():
    moving = Problem()
    assert solve_retention(moving, 0.0, 3, "pathwise").first_actions == (STOP,)
    static = Problem((Hypothesis(0.0, 0.0, 0.025),))  # rates 0: reading cannot change the bit
    r = solve_retention(static, 0.0, 3, "pathwise")
    assert r.first_actions != (STOP,)
    assert r.value == pytest.approx(solve(static, horizon=3).value, abs=1e-15)


def test_unconstrained_optimum_hides_a_large_corruption_risk():
    """The figure a limit-free planner reports, and what it does not say (README table)."""
    r = solve_retention(Problem(), 0.05, 3, "pathwise")
    assert r.unconstrained_value == pytest.approx(0.15691449841, abs=1e-10)
    assert r.unconstrained_p_changed > 0.12
    assert r.unconstrained_p_violation == pytest.approx(1.0)  # it always stops above 5 %
    assert retention_profile(Problem(), pathwise_policy(Problem(), 0.05), 3, 0.05)["p_violation"] == 0.0


@pytest.mark.parametrize("delta", [0.0, 0.05, 0.1, 0.2])
def test_pathwise_policy_never_stops_above_the_limit(delta):
    pr = Problem((Hypothesis(0.07, 0.045, 0.025), Hypothesis(0.23, 0.16, 0.1)), (0.5, 0.5))
    prof = retention_profile(pr, pathwise_policy(pr, delta), 4, delta)
    assert prof["p_violation"] == 0.0
    assert prof["value"] == pytest.approx(solve_retention(pr, delta, 4, "pathwise").value, abs=1e-15)


def test_best_fixed_sequence_is_a_baseline_never_better_than_adaptive():
    pr = Problem()
    for delta in DELTAS:
        for mode in ("pathwise", "expected"):
            fixed, seq = best_open_loop_retention(pr, 3, delta, mode)
            adaptive = solve_retention(pr, delta, 3, mode)
            assert adaptive.value <= fixed + 1e-12  # a fixed sequence is one admissible policy
            assert len(seq) <= 3


# ---------------------------------------------------------------- fail closed
@pytest.mark.parametrize("delta", [-0.1, 1.5, math.nan, math.inf, True, "0.1"])
def test_bad_delta_rejected(delta):
    with pytest.raises(ValueError):
        solve_retention(Problem(), delta, 2)


def test_bad_mode_and_horizon_rejected():
    with pytest.raises(ValueError):
        solve_retention(Problem(), 0.1, 2, "average")
    with pytest.raises(ValueError):
        solve_retention(Problem(), 0.1, 6)
    with pytest.raises(ValueError):
        lagrangian_policy(Problem(), -1.0)


def test_infeasible_start_is_refused_not_reported_optimistically():
    pr = Problem()
    changed = tuple(0.25 if x != x0 else 0.0 for x0 in (0, 1) for x in (0, 1) for _ in (0, 1))
    for mode in ("pathwise", "expected"):
        with pytest.raises(ValueError, match="infeasible"):
            solve_retention(pr, 0.1, 3, mode, belief=changed)
    with pytest.raises(ArithmeticError, match="no admissible decision"):
        pathwise_policy(pr, 0.1)(changed, (), 2)


# ---------------------------------------------------------------- h = 4: independent exact DP on oracle messages
def oracle_front(model, m, h):
    """Exact Pareto front {(C, J)} over deterministic policies (no tolerance), mass-weighted."""
    tot = sum(m)
    if tot == 0:
        return [(Fr(0), Fr(0))]
    mis = sum(v for v, s in zip(m, model.states, strict=True) if s[1] != s[2])
    pts = [(mis, model.stop_mass(m))]
    if h:
        for a in model.actions:
            f0 = oracle_front(model, model.step(m, a, 0), h - 1)
            f1 = oracle_front(model, model.step(m, a, 1), h - 1)
            c = model.cost(a) * tot
            pts += [(c0 + c1, c + j0 + j1) for c0, j0 in f0 for c1, j1 in f1]
    front, best = [], None
    for cc, jj in sorted(set(pts)):
        if best is None or jj < best:
            front.append((cc, jj))
            best = jj
    return front


def oracle_pathwise(model, m, h, delta):
    """Exact optimum with every stop at posterior P(x != x0) <= delta; None if impossible."""
    tot = sum(m)
    if tot == 0:
        return Fr(0)
    mis = sum(v for v, s in zip(m, model.states, strict=True) if s[1] != s[2])
    best = model.stop_mass(m) if mis <= delta * tot else None
    if h:
        for a in model.actions:
            kids = [oracle_pathwise(model, model.step(m, a, y), h - 1, delta) for y in (0, 1)]
            if None not in kids:
                v = model.cost(a) * tot + sum(kids)
                best = v if best is None or v < best else best
    return best


@pytest.mark.parametrize("name", ["default", "two_models", "lambda0"])
def test_horizon_4_against_independent_exact_dp(name):
    pkg, orc = make(**CASES[name])
    model = orc(True)
    front = oracle_front(model, model.root(), 4)
    hull = lower_hull([(j, c, None) for c, j in front])
    for delta in (0.0, 0.01, 0.05, 0.08, 0.1, 0.13, 1.0):
        d = Fr(delta)
        e = solve_retention(pkg, delta, 4, "expected")
        assert e.value == pytest.approx(float(hull_min(hull, d)), abs=TOL), delta
        assert e.deterministic_value == pytest.approx(float(min(j for c, j in front if c <= d)), abs=TOL), delta
        p = solve_retention(pkg, delta, 4, "pathwise")
        assert p.value == pytest.approx(float(oracle_pathwise(model, model.root(), 4, d)), abs=TOL), delta


def test_reference_table_values():
    """Pinned h = 4 numbers of the README retention table (cross-checked by the test above)."""
    rows = {r["delta"]: r for r in retention_table(4)}
    r = rows[0.08]
    assert r["unconstrained"] == pytest.approx(0.14624660362779907, abs=1e-12)
    assert r["unconstrained_p_changed"] == pytest.approx(0.13481597640625403, abs=1e-12)
    assert r["unconstrained_p_violation"] == pytest.approx(0.6543709473301715, abs=1e-12)
    assert r["pathwise"] == pytest.approx(0.25, abs=1e-12)
    assert r["expected_randomised"] == pytest.approx(0.1731213658073808, abs=1e-12)
    assert r["expected_deterministic"] == pytest.approx(0.21100430571967638, abs=1e-12)
    assert rows[0.1]["pathwise"] == pytest.approx(0.171552, abs=1e-6)


def test_readme_retention_table_matches_the_code():
    """The README table is pinned to the code, so a hand-edited number cannot drift."""
    import pathlib
    import re

    text = (pathlib.Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    rows = {r["delta"]: r for r in retention_table(4)}
    row = re.compile(r"\| (0|0\.05|0\.08|0\.10|1) " + r"\| ([\d.]+) " * 5 + r"% \|")
    found = 0
    for line in text.splitlines():
        m = row.match(line)
        if not m:
            continue
        r = rows[float(m.group(1))]
        keys = ("pathwise", "pathwise_best_fixed", "expected_randomised", "expected_deterministic")
        for key, cell in zip(keys, m.groups()[1:5], strict=True):
            assert float(cell) == pytest.approx(r[key], abs=6e-11), (m.group(1), key)
        assert float(m.group(6)) == pytest.approx(100 * r["unconstrained_p_violation"], abs=0.06)
        found += 1
    assert found == 5

"""R35 / Codex review of PR #8 (P2): optimum under the expected retention limit in the Gaussian line.

``solve_gaussian_retention`` takes the lower convex hull of the plan points (damage, value).
Checked here against an independent rational oracle that uses a different algorithm (every
single plan and every pair of plans, exact linear interpolation in ``Fraction``), against the
counter-example of the review, against its own duality certificate, and -- for the
data-dependent realisation of N-53 -- against a generative simulation of the verbal model with
a negative control.
"""

from __future__ import annotations

import random
from fractions import Fraction as F
from itertools import product
from math import erf, sqrt

import pytest
from _gaussian_oracle import plan_value as oracle_plan_value
from _gaussian_oracle import posterior as oracle_posterior

from atomic_information_lab.gaussian import (
    DEFAULT_GAUSSIAN_MODEL,
    GAUSSIAN_SCOPE,
    GaussianAction,
    GaussianModel,
    GaussianObjective,
    gaussian_retention_frontier,
    initial_state,
    innovation,
    plan_value,
    solve_gaussian,
    solve_gaussian_retention,
    update,
)

TOL = 1e-12

ALT = GaussianModel(
    prior_var=0.5,
    bias_prior_var=1.0,
    drift_var=0.07,
    bias_drift_var=0.0,
    actions=(
        GaussianAction("slow", 0.0, 0.2, 0.5),
        GaussianAction("hard", 4.0, 0.01, 3.0),
        GaussianAction("ref", 0.25, 0.05, 1.0, reads_reference=True),
    ),
    objective=GaussianObjective(alpha=0.3, energy_lambda=0.02, retention_weight=0.4),
)


def _params(model):
    return {
        "prior_var": model.prior_var,
        "bias_prior_var": model.bias_prior_var,
        "drift_var": model.drift_var,
        "bias_drift_var": model.bias_drift_var,
        "actions": {a.name: (a.backaction, a.noise_var, a.energy, a.reads_reference) for a in model.actions},
    }


def _objective(model):
    o = model.objective
    return (o.alpha, o.energy_lambda, o.retention_weight)


_ORACLE_CACHE: dict = {}


def _oracle_points(model, horizon):
    key = (id(model), horizon)
    if key not in _ORACLE_CACHE:
        prm, obj = _params(model), _objective(model)
        pts = []
        for n in range(horizon + 1):
            for seq in product([a.name for a in model.actions], repeat=n):
                pts.append((oracle_posterior(prm, seq)["damage"], oracle_plan_value(prm, obj, seq), seq))
        _ORACLE_CACHE[key] = pts
    return _ORACLE_CACHE[key]


def _oracle_optimum(model, horizon, delta):
    """Exact optimum over all mixtures: no hull, every single plan and every pair (rational)."""
    d = F(delta)
    pts = _oracle_points(model, horizon)
    best = min(v for dd, v, _ in pts if dd <= d)
    for d1, v1, _ in pts:
        if d1 > d:
            continue
        for d2, v2, _ in pts:
            if d2 > d:
                q = (d2 - d) / (d2 - d1)
                best = min(best, q * v1 + (1 - q) * v2)
    return best


DELTAS = [0.0, 0.004, 0.01, 0.0155, 0.025, 0.031, 0.04, 0.07, 0.11, 0.3]


@pytest.mark.parametrize("model", [DEFAULT_GAUSSIAN_MODEL, ALT], ids=["default", "alt"])
@pytest.mark.parametrize("horizon", [0, 1, 2, 3])
def test_optimum_matches_the_rational_pairwise_oracle(model, horizon):
    for delta in DELTAS:
        r = solve_gaussian_retention(model, horizon, delta)
        assert r.value == pytest.approx(float(_oracle_optimum(model, horizon, delta)), abs=TOL), delta
        assert r.scope == GAUSSIAN_SCOPE


def test_codex_counterexample_of_the_review():
    """Default model, h = 1, delta = 0.01: the single plan is STOP (0.25); the half/half mix of STOP
    and one fast reading meets the limit exactly and is worth 0.2072698413 (review of PR #8)."""
    r = solve_gaussian_retention(DEFAULT_GAUSSIAN_MODEL, 1, 0.01)
    assert [(round(w, 12), p.sequence) for w, p in r.components] == [(0.5, ()), (0.5, ("fast",))]
    assert r.value == pytest.approx(0.2072698413, abs=1e-10)
    assert r.damage == pytest.approx(0.01, abs=TOL)
    assert r.best_single_plan.value == pytest.approx(0.25, abs=TOL) and r.best_single_plan.sequence == ()
    assert solve_gaussian(DEFAULT_GAUSSIAN_MODEL, 1, retention_limit=0.01).value == r.best_single_plan.value


def test_n52_pinned_gap_of_the_single_plan():
    r = solve_gaussian_retention(DEFAULT_GAUSSIAN_MODEL, 2, 0.03)
    gap = r.best_single_plan.value / r.value - 1
    assert round(gap, 4) == 0.3656
    assert r.best_single_plan.sequence == ("fast",)
    assert [p.sequence for _, p in r.components] == [(), ("reference", "fast")]


@pytest.mark.parametrize("model", [DEFAULT_GAUSSIAN_MODEL, ALT], ids=["default", "alt"])
def test_certificate_and_shape(model):
    """Value is non-increasing and convex in delta; dual gap 0; never worse than the single plan;
    equal to the unconstrained optimum once the limit stops binding."""
    h = 4
    grid = [i * 0.005 for i in range(0, 41)]
    vals = []
    for d in grid:
        r = solve_gaussian_retention(model, h, d)
        assert r.damage <= d + TOL
        assert abs(r.duality_gap) <= 1e-10
        assert r.value <= r.best_single_plan.value + TOL
        assert r.best_single_plan.value == pytest.approx(solve_gaussian(model, h, retention_limit=d).value, abs=TOL)
        assert len(r.components) in (1, 2) and sum(w for w, _ in r.components) == pytest.approx(1.0, abs=1e-15)
        if len(r.components) == 2:  # the two hull plans bracket the limit
            assert r.components[0][1].damage <= d < r.components[1][1].damage
        else:
            assert r.components[0][1].damage <= d + TOL
        assert r.value - TOL <= r.deterministic_value <= r.best_single_plan.value + TOL
        assert r.deterministic_damage <= d + TOL
        vals.append(r.value)
    assert all(b <= a + TOL for a, b in zip(vals, vals[1:], strict=False))
    assert all(vals[i - 1] + vals[i + 1] - 2 * vals[i] >= -1e-10 for i in range(1, len(vals) - 1))
    free = solve_gaussian(model, h)
    r = solve_gaussian_retention(model, h, free.damage + 1.0)
    assert r.value == pytest.approx(free.value, abs=TOL) and r.multiplier == 0.0
    assert r.unconstrained.value == pytest.approx(free.value, abs=TOL)


def test_zero_limit_allows_only_damage_free_plans():
    r = solve_gaussian_retention(DEFAULT_GAUSSIAN_MODEL, 3, 0.0)
    assert [p.sequence for _, p in r.components] == [()] and r.value == pytest.approx(0.25, abs=TOL)
    # ALT has a reading without back-action, so a zero limit still allows measuring
    r = solve_gaussian_retention(ALT, 3, 0.0)
    assert r.damage == 0.0 and all(set(p.sequence) <= {"slow"} for _, p in r.components)


@pytest.mark.parametrize("delta", [-0.1, float("nan"), float("inf"), None])
def test_invalid_limits_are_rejected(delta):
    with pytest.raises(ValueError):
        solve_gaussian_retention(DEFAULT_GAUSSIAN_MODEL, 2, delta)


def test_start_state_above_the_limit_is_rejected():
    st = update(DEFAULT_GAUSSIAN_MODEL, initial_state(DEFAULT_GAUSSIAN_MODEL), DEFAULT_GAUSSIAN_MODEL.action("precise"))
    with pytest.raises(ValueError):
        solve_gaussian_retention(DEFAULT_GAUSSIAN_MODEL, 2, 0.01, st)
    r = solve_gaussian_retention(DEFAULT_GAUSSIAN_MODEL, 2, 0.1, st)
    assert r.value <= plan_value(DEFAULT_GAUSSIAN_MODEL, (), st).value + TOL


# ------------------------------------------------------------------ N-53: the data as the coin
def _phi(z):
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _common(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return a[:n]


def _run_split(model, result, rng, swap=False, components=None):
    """One run of the verbal model under the deterministic split rule of N-53.

    Read the shared prefix; then follow component 0 iff Phi(standardised innovation of the last
    shared reading) < its weight. ``swap`` uses the other weight (negative control).
    ``components`` defaults to the mixture optimum; pass ``deterministic_components`` to run the
    best deterministic policy.
    """
    (w0, p0), (_w1, p1) = result.components if components is None else components
    if swap:
        w0 = 1.0 - w0
    o = model.objective
    x0 = rng.gauss(0, sqrt(model.prior_var))
    x, b = x0, rng.gauss(0, sqrt(model.bias_prior_var))
    st = initial_state(model)
    shared = _common(p0.sequence, p1.sequence)
    assert shared, "a deterministic split needs a shared first reading"
    k = len(shared)
    seq = list(shared)
    energy, i, u = 0.0, 0, None
    while i < len(seq):
        a = model.action(seq[i])
        x += rng.gauss(0, sqrt(model.drift_var * a.backaction))
        b += rng.gauss(0, sqrt(model.bias_drift_var))
        y = (b if a.reads_reference else x + b) + rng.gauss(0, sqrt(a.noise_var))
        if i == k - 1:
            e, s = innovation(model, st, a, y)
            u = _phi(e / sqrt(s))
        st = update(model, st, a, y)
        energy += a.energy
        i += 1
        if i == k:
            seq += list((p0 if u < w0 else p1).sequence[k:])
    loss = (
        o.alpha * (st.mean[0] - x0) ** 2
        + (1 - o.alpha) * (st.mean[1] - x) ** 2
        + o.retention_weight * (x - x0) ** 2
        + o.energy_lambda * energy
    )
    return loss, (x - x0) ** 2


def _z(samples, target):
    n = len(samples)
    m = sum(samples) / n
    sd = sqrt(sum((v - m) ** 2 for v in samples) / (n - 1) / n)
    return (m - target) / sd


def test_n53_case_has_a_shared_prefix_and_beats_every_plan():
    r = solve_gaussian_retention(DEFAULT_GAUSSIAN_MODEL, 2, 0.08)
    assert r.shared_prefix == ("reference",)
    assert [p.sequence for _, p in r.components] == [("reference", "fast"), ("reference", "precise")]
    assert r.value < r.best_single_plan.value - 1e-4  # 0.115260 vs 0.116174


def test_n53_data_split_realises_the_mixture_without_a_coin():
    """Generative simulation: the deterministic, data-dependent rule attains the mixture's value
    and damage; the swapped weights (negative control) miss the damage by far."""
    m = DEFAULT_GAUSSIAN_MODEL
    r = solve_gaussian_retention(m, 2, 0.08)
    rng = random.Random(53)
    runs = [_run_split(m, r, rng) for _ in range(20000)]
    assert abs(_z([lo for lo, _ in runs], r.value)) < 4
    assert abs(_z([d for _, d in runs], r.damage)) < 4
    rng = random.Random(530)
    bad = [_run_split(m, r, rng, swap=True) for _ in range(20000)]
    assert abs(_z([d for _, d in bad], r.damage)) > 4


def test_innovation_matches_the_update():
    m = DEFAULT_GAUSSIAN_MODEL
    st = update(m, initial_state(m), m.action("fast"), 0.3)
    a = m.action("reference")
    e, s = innovation(m, st, a, -0.7)
    after = update(m, st, a, -0.7)
    pre = update(m, st, a)  # covariance only
    assert s > 0 and after.cov == pre.cov
    # the update moves the offset mean by its gain times the innovation
    assert after.mean[2] - st.mean[2] == pytest.approx(e * (st.cov[2][2] + m.bias_drift_var) / s, abs=1e-12)
    with pytest.raises(ValueError):
        innovation(m, st, a, float("nan"))


# ------------------------------------------------------------------ P1 of the review: squashed history
REGISTRATION = "f1eb3539ab2f1ae42a01b8c4a292ed8eae1e9e4f"


def _git_has(rev):
    import shutil
    import subprocess
    from pathlib import Path

    if shutil.which("git") is None:
        return False
    root = Path(__file__).resolve().parents[1]
    r = subprocess.run(["git", "-C", str(root), "cat-file", "-e", f"{rev}^{{commit}}"], capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(
    not _git_has(REGISTRATION), reason="needs git history with the R34 registration (CI: fetch-depth 0)"
)
def test_squashed_history_fails_closed_with_a_diagnosis(tmp_path):
    """The shape Codex reviewed on PR #8: the R34 branch squashed onto its base. The chain must fail
    (C2, C4) and name the cause; the unsquashed head and a merge commit pass without a diagnosis."""
    import json
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    script = root / "scripts" / "verify_preregistration.py"

    def git(*args, cwd=tmp_path / "c"):
        return subprocess.run(
            ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    def check(rev):
        r = subprocess.run(
            [sys.executable, str(script), "--repo", str(tmp_path / "c"), "--chain", "r34", "--head", rev],
            capture_output=True,
            text=True,
        )
        return r.returncode, json.loads(r.stdout)

    subprocess.run(["git", "clone", "-q", "--shared", "--no-checkout", str(root), str(tmp_path / "c")], check=True)
    head = git("rev-parse", "HEAD", cwd=root)
    git("checkout", "-q", "--detach", f"{REGISTRATION}^")
    base = git("rev-parse", "HEAD")
    git("merge", "-q", "--squash", head)
    git("commit", "-q", "-m", "squash")
    squashed = git("rev-parse", "HEAD")
    rc, rep = check(squashed)
    assert rc == 1 and not rep["all_pass"]
    assert not rep["checks"]["C2_results_added_in_a_descendant"]["pass"]
    assert len(rep["diagnosis"]) == 1 and "squashed or rebased" in rep["diagnosis"][0]
    git("checkout", "-q", "--detach", base)
    git("merge", "-q", "--no-ff", "-m", "merge", head)
    for rev in (head, git("rev-parse", "HEAD")):
        rc, rep = check(rev)
        assert rc == 0 and rep["all_pass"] and rep["diagnosis"] == []


# ------------------------------------------------------------------ R35b: three nested policy classes
def _oracle_deterministic(model, horizon, delta):
    """Best deterministic policy, rational and without hulls: STOP, or for each first reading the
    best single plan / pair of plans starting with it (the data draw from the first reading on)."""
    d = F(delta)
    pts = _oracle_points(model, horizon)
    best = next(v for dd, v, s in pts if s == ())
    for a in model.actions:
        group = [(dd, v) for dd, v, s in pts if s[:1] == (a.name,)]
        for d1, v1 in group:
            if d1 <= d:
                best = min(best, v1)
                for d2, v2 in group:
                    if d2 > d:
                        q = (d2 - d) / (d2 - d1)
                        best = min(best, q * v1 + (1 - q) * v2)
    return best


@pytest.mark.parametrize("model", [DEFAULT_GAUSSIAN_MODEL, ALT], ids=["default", "alt"])
@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_deterministic_policy_matches_the_rational_oracle(model, horizon):
    for delta in DELTAS:
        r = solve_gaussian_retention(model, horizon, delta)
        assert r.deterministic_value == pytest.approx(float(_oracle_deterministic(model, horizon, delta)), abs=TOL)
        comps = r.deterministic_components
        assert len(comps) == 1 or _common(comps[0][1].sequence, comps[1][1].sequence)


def test_one_reading_leaves_no_data_to_draw_with():
    """h = 1: the only decision comes before any data, so deterministic = best single plan."""
    for d in DELTAS:
        r = solve_gaussian_retention(DEFAULT_GAUSSIAN_MODEL, 1, d)
        assert r.deterministic_value == pytest.approx(r.best_single_plan.value, abs=TOL)


def test_r35b_pinned_example_of_the_three_classes():
    r = solve_gaussian_retention(DEFAULT_GAUSSIAN_MODEL, 2, 0.03)
    assert round(r.value, 6) == 0.120491
    assert round(r.deterministic_value, 6) == 0.12242
    assert round(r.best_single_plan.value, 6) == 0.16454
    assert [p.sequence for _, p in r.deterministic_components] == [("fast",), ("fast", "reference")]
    assert r.shared_prefix == ()  # the randomised optimum needs a coin here


def test_exact_suprema_quoted_in_the_docs():
    """Suprema over delta (default model, h = 2; the same for h = 3..6 by the R35b audit): the
    ratios are linear-fractional between breakpoints, so the suprema sit at breakpoints or their
    left limits; evaluated here just below / at them."""
    m = DEFAULT_GAUSSIAN_MODEL

    def at(d):
        return solve_gaussian_retention(m, 2, d)

    r = at(0.02 - 1e-12)  # single vs mixture, left limit at 0.02
    assert round(r.best_single_plan.value / r.value - 1, 4) == 0.5276
    r = at(0.031 - 1e-12)  # single vs deterministic, left limit at 0.031
    assert round(r.best_single_plan.value / r.deterministic_value - 1, 4) == 0.4163
    grid = [0.0130 + i * 1e-7 for i in range(1, 600)]  # deterministic vs mixture, attained near 0.013028
    best = max(at(d).deterministic_value / at(d).value - 1 for d in grid)
    assert round(best, 4) == 0.2903
    assert at(0.02).best_single_plan.sequence == ("fast",)


def test_n53_best_deterministic_policy_is_realised_by_the_data():
    """h = 2, delta = 0.03: the hull plans do not share a first reading (a coin is needed for the
    optimum), but the best deterministic policy -- read 'fast', then stop or read 'reference' by
    the innovation -- is attained without a coin and is 25.6 % below the best single plan."""
    m = DEFAULT_GAUSSIAN_MODEL
    r = solve_gaussian_retention(m, 2, 0.03)
    rng = random.Random(5353)
    runs = [_run_split(m, r, rng, components=r.deterministic_components) for _ in range(20000)]
    assert abs(_z([lo for lo, _ in runs], r.deterministic_value)) < 4
    assert abs(_z([d for _, d in runs], r.deterministic_damage)) < 4
    rng = random.Random(53530)
    bad = [_run_split(m, r, rng, swap=True, components=r.deterministic_components) for _ in range(20000)]
    assert abs(_z([d for _, d in bad], r.deterministic_damage)) > 4
    assert round(1 - r.deterministic_value / r.best_single_plan.value, 3) == 0.256


@pytest.mark.parametrize("model", [DEFAULT_GAUSSIAN_MODEL, ALT], ids=["default", "alt"])
def test_multiplier_is_the_retention_weight_that_makes_both_plans_optimal(model):
    """Gaussian analogue of N-15: pricing the damage at retention_weight + mu* makes both hull plans
    optimal for the unconstrained planner, and its value is dual_bound + mu* delta."""
    from dataclasses import replace as dc_replace

    for d in (0.01, 0.03, 0.08):
        r = solve_gaussian_retention(model, 3, d)
        if r.multiplier == 0.0:
            continue
        o = model.objective
        priced = dc_replace(model, objective=dc_replace(o, retention_weight=o.retention_weight + r.multiplier))
        free = solve_gaussian(priced, 3)
        assert free.value == pytest.approx(r.dual_bound + r.multiplier * d, abs=1e-10)
        for _, p in r.components:
            assert plan_value(priced, p.sequence).value == pytest.approx(free.value, abs=1e-10)


@pytest.mark.parametrize("model", [DEFAULT_GAUSSIAN_MODEL, ALT], ids=["default", "alt"])
def test_frontier_is_the_optimum_for_every_limit(model):
    f = gaussian_retention_frontier(model, 3)
    assert all(a[0] < b[0] and a[1] > b[1] for a, b in zip(f, f[1:], strict=False))
    slopes = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(f, f[1:], strict=False)]
    assert all(s1 <= s2 + 1e-12 for s1, s2 in zip(slopes, slopes[1:], strict=False))  # convex
    for d, v, seq in f:
        r = solve_gaussian_retention(model, 3, d)
        assert r.value == pytest.approx(v, abs=TOL)
        assert plan_value(model, seq).value == pytest.approx(v, abs=TOL)


@pytest.mark.parametrize("delta", [True, "0.1", 1j])
def test_non_numeric_limits_are_rejected(delta):
    with pytest.raises(ValueError):
        solve_gaussian_retention(DEFAULT_GAUSSIAN_MODEL, 2, delta)
    with pytest.raises(ValueError):
        solve_gaussian(DEFAULT_GAUSSIAN_MODEL, 2, retention_limit=delta)

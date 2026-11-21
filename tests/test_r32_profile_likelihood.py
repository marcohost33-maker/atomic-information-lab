"""R32: profile likelihood for practical identifiability (roadmap R29 / P1, Raue et al. 2009).

What is checked, and against what:

* **Independent oracle.** The per-session profile statistic ``2 min KL`` is
  recomputed from the hidden-path oracle (``tests/_ident_oracle.py``,
  uniformisation, no package code) with its own global minimiser (log grid,
  then golden section) -- with no nuisance parameter, and with one.
* **Local limit.** Near the truth the profile must have the Cramer-Rao
  curvature (``identifiability.relative_crlb``, a different code path), and
  its odd deviation must grow linearly with the offset (cubic term).
* **Structural witnesses** (exact global statements of ``identifiability``):
  flat profiles for the drift/back-action scale and the hidden preparation
  error; the label swap shows as a second zero (non-monotone side).
* **Definitions** checked exactly: ``bounded_sessions`` is the smallest N whose
  interval is bounded; intervals are nested in N and contain the truth;
  non-converged points never bound an interval; Wilks' threshold.
* **Consumer contract:** every JSON object with a count carries both scopes.
* **Pinned evidence:** ``docs/PROFILE_LIKELIHOOD.json`` (reproduce table) is
  checked against a fresh run of one row, and the README table against it.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import subprocess
import sys

import pytest
from _ident_oracle import session_distribution, spec_of

from atomic_information_lab import profile_likelihood as public_profile_likelihood
from atomic_information_lab.identifiability import Calibration, Design, identifiability, parameter_names
from atomic_information_lab.likelihood_profile import (
    FLAT_TOL,
    PL_SCOPE,
    PL_SCOPE_TEXT,
    ParameterProfile,
    chi2_1_quantile,
    profile_likelihood,
    profile_report,
    profile_statistic,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
ENV = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"}
# literal on purpose: comparing with the imported constants would follow a mutation of them
PL_TOKEN = "EXPECTED_DATA_PROFILE_MODEL_INTERNAL_ONE_POINT"
CRLB_TOKEN = "CRLB_UNBIASED_REGULAR_LOWER_BOUND_ONLY"
CHI2_1_95 = 3.841458820694124  # 1.959963984540054 ** 2

CAL = Calibration(prep_error=0.02)
EVERY = parameter_names(CAL)
ANCHORED = tuple(f for f in EVERY if f != "backaction:fast")
D6 = Design(("fast", "reference", "precise", "reference", "fast", "precise"), True)
D3 = Design(("fast", "reference", "precise"), True)
PAIR = ("k01", "backaction:precise")  # GO on D3; strongly non-quadratic profile of k01
FIVE = ("k01", "k10", "prep_error", "bias_prior", "error:precise")  # GO on D3; open intervals at small N


# ---------------------------------------------------------------- independent oracle
def _oracle_kl(p: dict, cal: Calibration, design: Design) -> float:
    q = session_distribution(spec_of(cal), design.sequence, design.known_write)
    assert set(q) == set(p)
    return math.fsum(p[o] * math.log(p[o] / q[o]) for o in p if p[o] > 0)


def _oracle_min_over(p, truth, design, fixed: dict, nuisance: str, lo: float, hi: float) -> tuple[float, float]:
    """Global minimum of KL over one nuisance parameter: log grid, then golden section."""

    def f(u: float) -> float:
        return _oracle_kl(p, truth.with_values({**fixed, nuisance: math.exp(u)}), design)

    a, b = math.log(lo), math.log(hi)
    grid = [a + (b - a) * k / 120 for k in range(121)]
    vals = [f(u) for u in grid]
    k = min(range(len(vals)), key=vals.__getitem__)
    a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
    g = (math.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(80):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    u = (a + b) / 2
    return 2 * f(u), math.exp(u)


@pytest.mark.parametrize("name, factor", [("k01", 1.5), ("backaction:precise", 0.6), ("prep_error", 3.0)])
def test_no_nuisance_statistic_is_twice_the_oracle_kl(name, factor):
    p = session_distribution(spec_of(CAL), D3.sequence, D3.known_write)
    value = CAL.parameters()[name] * factor
    pt = profile_statistic(CAL, D3, (name,), name, value)
    want = 2 * _oracle_kl(p, CAL.with_values({name: value}), D3)
    assert pt.converged and want > 1e-4
    assert pt.statistic == pytest.approx(want, rel=1e-9)


@pytest.mark.parametrize("factor", [0.5, 0.8, 1.3, 2.5])
def test_profile_with_one_nuisance_matches_the_oracle_global_minimum(factor):
    p = session_distribution(spec_of(CAL), D3.sequence, D3.known_write)
    value = 0.07 * factor
    want, beta = _oracle_min_over(p, CAL, D3, {"k01": value}, "backaction:precise", 3.5 / 50, 3.5 * 50)
    pt = profile_statistic(CAL, D3, PAIR, "k01", value)
    assert pt.converged
    assert pt.statistic == pytest.approx(want, rel=1e-6)
    assert dict(pt.nuisance)["backaction:precise"] == pytest.approx(beta, rel=1e-4)
    # the profile is strictly below the slice (nuisance held at the truth) away from the truth
    slice_ = 2 * _oracle_kl(p, CAL.with_values({"k01": value}), D3)
    assert pt.statistic < slice_ * (1 - 1e-3)


# ---------------------------------------------------------------- local limit: Cramer-Rao curvature
@pytest.mark.parametrize("name", ["prep_error", "backaction:reference"])
def test_local_limit_is_the_cramer_rao_curvature(name):
    rep = identifiability(CAL, D6, ANCHORED)
    i = rep.free.index(name)
    rho, tv = rep.relative_crlb[i], rep.values[i]
    ratio = {}
    for r in (1e-3, -1e-3, 2e-3, -2e-3):
        pt = profile_statistic(CAL, D6, ANCHORED, name, tv * (1 + r))
        assert pt.converged
        ratio[r] = pt.statistic / (r * r / rho**2)
    assert all(abs(v - 1) < 5e-3 for v in ratio.values()), ratio
    odd1 = (ratio[1e-3] - ratio[-1e-3]) / 2
    odd2 = (ratio[2e-3] - ratio[-2e-3]) / 2
    assert abs(odd1) > 1e-6  # these two profiles are not symmetric
    assert 1.8 < odd2 / odd1 < 2.2  # the deviation from the quadratic is the cubic term, linear in r


def test_profile_sessions_reduce_to_the_crlb_count_for_a_symmetric_parameter():
    # bias_prior = 0.5 is a symmetry point (b <-> 1-b): the profile is even, the counts agree to rounding
    r = profile_likelihood(CAL, D3, FIVE, "bias_prior")
    assert r.asymmetry == pytest.approx(1.0, abs=1e-9)
    assert abs(r.profile_sessions - r.crlb_sessions) <= 2
    assert r.profile_to_crlb == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------- structural witnesses
def test_drift_backaction_scale_gives_a_flat_profile():
    r = profile_likelihood(CAL, D6, EVERY, "k01", sessions=(10**9,))
    assert identifiability(CAL, D6, EVERY).verdict == "NO-GO"
    assert r.flat and max(p.statistic for p in r.points) <= FLAT_TOL
    assert r.bounded_sessions is None and r.profile_sessions is None and r.crlb_sessions is None
    assert r.intervals[0].low_open and r.intervals[0].high_open and not r.intervals[0].bounded


def test_hidden_preparation_error_gives_a_flat_profile():
    r = profile_likelihood(CAL, Design(D6.sequence, False), ANCHORED, "prep_error", sessions=(10**12,))
    assert r.flat and r.bounded_sessions is None and not r.intervals[0].bounded


def test_label_swap_shows_as_a_second_zero():
    d = Design(("fast", "precise") * 2)
    free = [f for f in EVERY if f not in ("backaction:fast", "prep_error", "error:reference", "backaction:reference")]
    assert identifiability(CAL, d, free).verdict == "AMBIGUOUS"
    swapped = profile_statistic(CAL, d, free, "k01", 0.045)  # k01 <-> k10 with b <-> 1-b
    assert swapped.converged and swapped.statistic <= 1e-20
    r = profile_likelihood(CAL, d, free, "k01")
    assert not r.low_monotone and r.low_dip is not None and r.low_dip < 1e-6
    assert r.high_monotone and r.high_dip is None


# ---------------------------------------------------------------- definitions, exactly
def test_bounded_sessions_is_the_smallest_n_with_a_bounded_interval():
    r = profile_likelihood(CAL, D3, FIVE, "k10")
    n = r.bounded_sessions
    assert n is not None and n > 1
    below, at = profile_likelihood(CAL, D3, FIVE, "k10", sessions=(n - 1, n)).intervals
    assert at.bounded and not below.bounded


def test_bounded_sessions_follows_the_flatter_side():
    r = profile_likelihood(CAL, D3, PAIR, "k01")
    assert r.low_plateau < r.high_plateau / 2  # strongly asymmetric
    assert r.bounded_sessions == math.floor(CHI2_1_95 / r.low_plateau) + 1


def test_intervals_contain_the_truth_and_are_nested_in_n():
    ns = (100, 1_000, 10_000, 100_000)
    r = profile_likelihood(CAL, D3, PAIR, "k01", sessions=ns)
    prev = None
    for iv in r.intervals:
        assert iv.bounded and iv.low < 0.07 < iv.high
        if prev is not None:
            assert prev.low <= iv.low and iv.high <= prev.high
        prev = iv


def test_large_n_interval_has_the_wald_half_width():
    # far into the asymptotic regime the 95 % interval is truth +- z * CRLB standard error
    r = profile_likelihood(CAL, D3, PAIR, "backaction:precise", sessions=(10**9,))
    iv = r.intervals[0]
    half = 1.959963984540054 * r.crlb_rse * 3.5 / math.sqrt(10**9)
    assert (iv.high - iv.low) / 2 == pytest.approx(half, rel=0.02)


def test_wilks_threshold_literal():
    assert chi2_1_quantile(0.95) == pytest.approx(CHI2_1_95, rel=1e-12)
    assert chi2_1_quantile(0.6826894921370859) == pytest.approx(1.0, rel=1e-9)


def test_unconverged_points_never_bound_an_interval(monkeypatch):
    mod = sys.modules["atomic_information_lab.likelihood_profile"]

    def stuck(truth, design, name, value, nuis, start, target):
        return 1e9, dict(start), False, "iteration limit"

    monkeypatch.setattr(mod, "_inner", stuck)
    r = profile_likelihood(CAL, D3, PAIR, "k01", sessions=(10**6,))
    assert r.not_converged == len(r.points)
    assert r.bounded_sessions is None and r.profile_sessions is None
    iv = r.intervals[0]
    assert iv.low_open and iv.high_open and not iv.bounded


def test_public_export_is_the_module_function():
    assert public_profile_likelihood is profile_likelihood


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"name": "k10"}, "not a free parameter"),
        ({"target_rse": 0.0}, "target_rse"),
        ({"target_rse": 1.0}, "target_rse"),
        ({"sessions": (0,)}, "sessions"),
        ({"sessions": (True,)}, "sessions"),
        ({"sessions": "100"}, "sessions"),
        ({"level": 1.0}, "level"),
        ({"level": float("nan")}, "level"),
    ],
)
def test_profile_likelihood_fails_closed(kwargs, match):
    args = {"name": "k01", **kwargs}
    with pytest.raises(ValueError, match=match):
        profile_likelihood(CAL, D3, PAIR, **args)


@pytest.mark.parametrize(
    "name, value, match",
    [("k10", 0.05, "not a free parameter"), ("k01", 0.0, "positive"), ("k01", -1.0, "positive")],
)
def test_profile_statistic_fails_closed(name, value, match):
    with pytest.raises(ValueError, match=match):
        profile_statistic(CAL, D3, PAIR, name, value)


def test_upper_domain_bounds_are_enforced():
    with pytest.raises(ValueError, match="<= 0.5"):
        profile_statistic(CAL, D3, ("prep_error", "k01"), "prep_error", 0.6)
    with pytest.raises(ValueError, match="upper domain bound"):
        profile_likelihood(Calibration(prep_error=0.48), D3, ("prep_error", "k01"), "prep_error", target_rse=0.1)
    with pytest.raises(ValueError, match="> 0 at the truth"):
        profile_likelihood(Calibration(prep_error=0.0), D3, ("prep_error", "k01"), "prep_error")


def test_profile_report_defaults_to_every_free_parameter_and_rejects_others():
    rows = profile_report(CAL, D3, PAIR)
    assert [r.name for r in rows] == list(PAIR) and all(isinstance(r, ParameterProfile) for r in rows)
    with pytest.raises(ValueError, match="names"):
        profile_report(CAL, D3, PAIR, ("k10",))
    with pytest.raises(ValueError, match="names"):
        profile_report(CAL, D3, PAIR, "k01")


# ---------------------------------------------------------------- consumer contract (CLI)
COUNT_KEY = re.compile(r"sessions|crlb")


def _sites(obj, path="$"):
    if isinstance(obj, dict):
        if any(COUNT_KEY.search(k) and k != "sessions_scope" for k in obj):
            yield path, obj
        for k, v in obj.items():
            yield from _sites(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _sites(v, f"{path}[{i}]")


def test_cli_every_count_carries_both_scopes_and_their_words():
    p = subprocess.run(
        [sys.executable, "-m", "atomic_information_lab", "profile-likelihood", "--design", "fast,reference,precise"]
        + ["--known-write", "--free", "k01,backaction:precise", "--param", "k01", "--sessions", "100,100000"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=ENV,
        timeout=600,
    )
    assert p.returncode == 0, p.stderr
    data = json.loads(p.stdout)
    sites = list(_sites(data))
    assert len(sites) >= 3  # positive control: the row and its two intervals
    for path, obj in sites:
        assert obj.get("sessions_scope") == CRLB_TOKEN, path
        assert obj.get("pl_scope") == PL_TOKEN, path
    words = " ".join(data["notes"])
    for needle in ("EXPECTED data", "Asimov", "Wilks", "not measured", "conservative", "lower bound", "UNBIASED"):
        assert needle in words


def test_cli_rejects_bad_sessions():
    p = subprocess.run(
        [sys.executable, "-m", "atomic_information_lab", "profile-likelihood", "--design", "fast,precise"]
        + ["--free", "k01,backaction:precise", "--sessions", "ten"],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=ENV,
        timeout=120,
    )
    assert p.returncode == 2 and "--sessions" in p.stderr


def test_scope_text_states_its_limits():
    assert PL_SCOPE == PL_TOKEN
    for needle in ("EXPECTED", "model-internal", "one parameter point", "Wilks", "scan range", "coverage"):
        assert needle in PL_SCOPE_TEXT


# ---------------------------------------------------------------- pinned evidence
TABLE = ROOT / "docs" / "PROFILE_LIKELIHOOD.json"


def _table() -> dict:
    return json.loads(TABLE.read_text(encoding="utf-8"))["profile_likelihood"]


def test_registered_design_table_row_is_reproduced():
    from atomic_information_lab.reproduce import PL_SESSIONS, profile_row

    t = _table()
    assert t["design"] == list(D6.sequence) and t["free"] == list(ANCHORED) and t["sessions"] == list(PL_SESSIONS)
    stored = {row["name"]: row for row in t["parameters"]}
    assert set(stored) == set(ANCHORED)
    fresh = profile_row(profile_likelihood(CAL, D6, ANCHORED, "bias_prior", 0.1, PL_SESSIONS))
    want = json.loads(json.dumps(fresh))
    assert want.keys() == stored["bias_prior"].keys()
    for k, v in want.items():
        if isinstance(v, float):
            assert stored["bias_prior"][k] == pytest.approx(v, rel=1e-9, abs=1e-15), k
        elif k == "points":
            assert len(v) == len(stored["bias_prior"][k])
        else:
            assert stored["bias_prior"][k] == v, k


def test_table_is_complete_and_its_figures_rest_on_converged_points():
    total = 0
    for row in _table()["parameters"]:
        total += row["not_converged"]
        stuck = [p for p in row["points"] if not p[2]]
        assert len(stuck) == row["not_converged"]
        converged = [p[1] for p in row["points"] if p[2]]
        # a stuck point over-estimates the profile; it must not be what a plateau or threshold rests on
        assert max(row["low_plateau"], row["high_plateau"]) == max(converged)
        for p in stuck:
            assert p[1] > 1e3 * max(iv["sessions"] and 3.85 / iv["sessions"] for iv in row["intervals"])
        assert row["low_monotone"] and row["high_monotone"], row["name"]
        assert all(iv["bounded"] for iv in row["intervals"]), row["name"]
        assert row["sessions_scope"] == CRLB_TOKEN and row["pl_scope"] == PL_TOKEN
    assert total == 1  # bias_flip = 0.381: read errors run to 0 and a back-action factor to infinity (N-40)


def test_post_hoc_the_two_most_non_quadratic_parameters_are_the_two_biased_ones():
    """Descriptive, labelled post hoc (finding N-36): the registered MLE run at count/16 against the profile."""
    rows = {r["name"]: r for r in _table()["parameters"]}
    top2 = sorted(rows, key=lambda n: rows[n]["profile_to_crlb"], reverse=True)[:2]
    reg = json.loads((ROOT / "docs" / "ESTIMATOR_CALIBRATION_RESULTS.json").read_text(encoding="utf-8"))
    run = next(r for r in reg["runs"] if r["sessions"] == 23_372)
    flagged = sorted(
        p["name"]
        for p in run["parameters"]
        if p["relative_bias"] > 0 and p["relative_bias"] > 3 * p["relative_bias_mcse"]
    )
    assert sorted(top2) == flagged == ["backaction:precise", "backaction:reference"]
    for n in flagged:  # flatter above the truth <-> positive bias
        assert rows[n]["asymmetry"] < 1


def test_readme_table_matches_the_pinned_json():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    sec = text.split("### Profile likelihood", 1)[1].split("\n## ", 1)[0]
    rows = {r["name"]: r for r in _table()["parameters"]}
    seen = 0
    for line in sec.splitlines():
        m = re.match(r"\| `([^`]+)` \| ([\d ]+) \| ([\d ]+) \| ([\d.]+) \| ([\d.]+) \| ([\d ]+) \|", line)
        if not m:
            continue
        row = rows[m.group(1)]
        assert int(m.group(2).replace(" ", "")) == row["crlb_sessions"]
        assert int(m.group(3).replace(" ", "")) == row["profile_sessions"]
        assert m.group(4) == f"{row['profile_to_crlb']:.3f}"
        assert m.group(5) == f"{row['asymmetry']:.3f}"
        assert int(m.group(6).replace(" ", "")) == row["bounded_sessions"]
        seen += 1
    assert seen == len(rows)

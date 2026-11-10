"""R30: the three cross-family findings on PR #4 @35aef81 (GPT-6, 2026-11-05).

* AIL4-01 / AIL-01 -- the Cramér–Rao scope must reach every Python consumer of a Cramér–Rao
  figure, not only the CLI: ``ParameterCalibration.crlb_rse`` had no ``sessions_scope``, and the
  results JSON written by ``dataclasses.asdict`` inherited the gap.
* AIL4-03 / AIL-02 -- "efficient within Monte-Carlo error" needs the Monte-Carlo error of the RSE
  itself, and coverage over converged fits only is a survivor statistic: it is now bounded over
  all replications.
* AIL4-04 / AIL-03 -- the pre-registration chain is checked from git history
  (``scripts/verify_preregistration.py``), with negative controls on fabricated histories so the
  check cannot pass vacuously; the fixed setting of the protocol is checked against the runner
  without git as well.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.util
import inspect
import json
import random
import re
import shutil
import subprocess
from math import ceil, sqrt
from pathlib import Path

import pytest

from atomic_information_lab.estimation import ParameterCalibration, estimator_calibration, sd_mcse
from atomic_information_lab.identifiability import Calibration, Design

ROOT = Path(__file__).resolve().parents[1]
TOKEN = "CRLB_UNBIASED_REGULAR_LOWER_BOUND_ONLY"  # literal on purpose (a mutated constant must not follow)
identifiability = importlib.import_module("atomic_information_lab.identifiability")
estimation = importlib.import_module("atomic_information_lab.estimation")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ AIL4-01: scope on every CRLB figure
def _public_dataclasses():
    for mod in (identifiability, estimation):
        for name in mod.__all__:
            obj = getattr(mod, name)
            if isinstance(obj, type) and dataclasses.is_dataclass(obj):
                yield obj


def test_every_public_type_with_a_crlb_figure_carries_the_lower_bound_scope():
    found = []
    for cls in _public_dataclasses():
        fields = {f.name: f for f in dataclasses.fields(cls)}
        if any("crlb" in n for n in fields):
            found.append(cls.__name__)
            assert "sessions_scope" in fields, f"{cls.__name__} holds a Cramér–Rao figure without sessions_scope"
    assert "ParameterCalibration" in found  # positive control: the type of the finding is seen


def _small_run(**kw):
    d = Design(("reference", "fast", "reference"), True)
    cal = Calibration(prep_error=0.05, bias_prior=0.3)
    return estimator_calibration(cal, d, ("error:fast", "error:reference", "prep_error"), **kw)


def _crlb_sites(obj, path="$"):
    if isinstance(obj, dict):
        if any("crlb" in k for k in obj):
            yield path, obj
        for k, v in obj.items():
            yield from _crlb_sites(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _crlb_sites(v, f"{path}[{i}]")


def test_estimator_api_and_its_json_carry_the_token_on_every_crlb_site():
    r = _small_run(sessions=20_000, replications=12, seed=5)
    data = json.loads(json.dumps(dataclasses.asdict(r)))
    sites = list(_crlb_sites(data))
    assert len(sites) == 3  # positive control: one per free parameter
    for path, obj in sites:
        assert obj.get("sessions_scope") == TOKEN, path


def test_parameter_calibration_positional_construction_unchanged():
    nan = float("nan")
    p = ParameterCalibration("k01", 0.07, 0.1, nan, nan, nan, nan, nan, nan, nan)
    assert p.sessions_scope == TOKEN


# frozen evidence: the registered results file predates the field and is NOT rewritten.
REGISTERED = "docs/ESTIMATOR_CALIBRATION_RESULTS.json"
REGISTERED_SHA256 = "1dc62757d2df8714662549846693746d2365ba5bbfbb4195b204f55802f21308"


def test_docs_json_carries_the_scope_except_the_pinned_registered_result():
    files = sorted((ROOT / "docs").glob("*.json"))
    assert ROOT / REGISTERED in files
    seen_sites = 0
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        missing = [p for p, o in _crlb_sites(data) if o.get("sessions_scope") != TOKEN]
        seen_sites += sum(1 for _ in _crlb_sites(data))
        if f.relative_to(ROOT).as_posix() == REGISTERED:
            # the one allowed exception is tamper-evident: its bytes are pinned
            assert hashlib.sha256(f.read_bytes()).hexdigest() == REGISTERED_SHA256
            assert missing, "control: the registered file is the known gap; if it has the scope, drop the exception"
        else:
            assert missing == [], (f.name, missing[:3])
    assert seen_sites >= 40  # positive control: the walker finds the registered file's 40 sites


# ------------------------------------------------------------------ AIL4-03: MC error of the RSE, survivors
def test_sd_mcse_matches_normal_theory_and_the_observed_spread_of_s():
    rng = random.Random(20261110)
    xs = [rng.gauss(0.0, 2.0) for _ in range(200_000)]
    s = sqrt(sum(x * x for x in xs) / (len(xs) - 1))
    assert sd_mcse(xs) == pytest.approx(s / sqrt(2 * (len(xs) - 1)), rel=0.02)  # Morris et al. 2019, Table 6
    # independent control: the standard deviation of s over many samples of size m
    m, reps = 60, 3000
    ss, est = [], []
    for _ in range(reps):
        ys = [rng.gauss(0.0, 1.0) for _ in range(m)]
        mu = sum(ys) / m
        ss.append(sqrt(sum((y - mu) ** 2 for y in ys) / (m - 1)))
        est.append(sd_mcse(ys))
    mu_s = sum(ss) / reps
    observed = sqrt(sum((v - mu_s) ** 2 for v in ss) / (reps - 1))
    assert sum(est) / reps == pytest.approx(observed, rel=0.08)


def test_sd_mcse_grows_with_heavy_tails():
    """Laplace data: excess kurtosis 3, so Var(s^2) ~ 5 sigma^4/m instead of 2 sigma^4/m (ratio sqrt 2.5)."""
    rng = random.Random(7)
    xs = [rng.expovariate(1.0) * (1 if rng.random() < 0.5 else -1) for _ in range(200_000)]
    s = sqrt(sum(x * x for x in xs) / (len(xs) - 1))
    assert sd_mcse(xs) / (s / sqrt(2 * (len(xs) - 1))) == pytest.approx(sqrt(2.5), rel=0.06)


@pytest.mark.parametrize("bad", [[1.0, 2.0, 3.0], [1.0, 2.0, float("nan"), 4.0], [1.0, 2.0, float("inf"), 0.0]])
def test_sd_mcse_fails_closed(bad):
    with pytest.raises(ValueError):
        sd_mcse(bad)


def test_sd_mcse_constant_data_is_zero():
    assert sd_mcse([0.5] * 10) == 0.0


def test_coverage_over_all_replications_brackets_the_survivor_coverage():
    """Protocol design at N = 3000: about a fifth of the fits fail, the rest cover below 1.

    The converged-only coverage must lie strictly between the two all-replication bounds. (A case with
    coverage 1 would hide a bound that ignores the failures, mutant M25: hits/m = (hits+f)/R = 1.)
    """
    from test_identifiability import ANCHORED, D6

    r = estimator_calibration(Calibration(prep_error=0.02), Design(D6, True), ANCHORED, 3000, 60, seed=0)
    failed = r.replications - r.converged
    below_one = [p for p in r.parameters if p.coverage < 1]
    assert failed > 0 and len(below_one) >= 5  # positive control: not degenerate
    for p in r.parameters:
        hits = round(p.coverage * r.converged)
        assert p.coverage_all_low == pytest.approx(hits / r.replications)
        assert p.coverage_all_high == pytest.approx((hits + failed) / r.replications)
        assert p.coverage_all_low < p.coverage <= p.coverage_all_high
        if p.coverage < 1:
            assert p.coverage < p.coverage_all_high
        assert p.coverage_all_high - p.coverage_all_low == pytest.approx(failed / r.replications)
        assert p.empirical_rse_mcse > 0 or r.converged < 4


def test_two_or_three_converged_fits_do_not_crash_the_report():
    """Regression (found while writing R30): sd_mcse needs 4 values; 2-3 fits must give nan, not raise."""
    d = Design(("reference", "fast", "reference"), True)
    cal = Calibration(prep_error=0.05, bias_prior=0.3)
    r = estimator_calibration(cal, d, ("error:fast", "prep_error"), 80, 6, seed=6)
    assert r.converged == 2  # positive control: this seed really hits the case
    for p in r.parameters:
        assert p.empirical_rse_mcse != p.empirical_rse_mcse  # nan, reported, not raised
        assert p.empirical_rse == p.empirical_rse  # the other measures are still computed


def test_without_failures_the_bounds_coincide_with_the_coverage():
    r = _small_run(sessions=2_000_000, replications=30, seed=3)
    assert r.converged == r.replications  # control: large N, no failed fit
    for p in r.parameters:
        assert p.coverage_all_low == p.coverage == p.coverage_all_high


# ------------------------------------------------------------------ AIL4-04: pre-registration chain
def test_protocol_setting_equals_the_runner_without_git():
    vp = _load_script("verify_preregistration")
    run = _load_script("estimator_calibration")
    want = vp.protocol_setting((ROOT / "docs/ESTIMATOR_PROTOCOL.md").read_text(encoding="utf-8"))
    assert want["sessions"] == [ceil(run.CRLB_COUNT * m) for m in run.MULTIPLIERS]
    assert want["seed_base"] == run.BASE_SEED
    assert want["design"] == run.DESIGN and want["known_write"] is True
    assert want["replications"] == 1000 and want["start_offset"] == 0.2
    assert run.REPLICATIONS == want["replications"]
    # the runner leaves the start offset to the library default
    assert inspect.signature(estimator_calibration).parameters["start_offset"].default == want["start_offset"]


def test_registered_result_setting_matches_the_protocol_without_git():
    vp = _load_script("verify_preregistration")
    want = vp.protocol_setting((ROOT / "docs/ESTIMATOR_PROTOCOL.md").read_text(encoding="utf-8"))
    got = vp.results_setting(json.loads((ROOT / REGISTERED).read_text(encoding="utf-8")))
    assert got["sessions"] == want["sessions"]
    assert got["seeds"] == [want["seed_base"] + i for i in range(4)]
    assert got["replications"] == [want["replications"]]


HAS_GIT = shutil.which("git") is not None


def _git_has(rev: str) -> bool:
    r = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-e", f"{rev}^{{commit}}"], capture_output=True)
    return r.returncode == 0


@pytest.mark.skipif(not (HAS_GIT and _git_has("1097c2d")), reason="needs git history with 1097c2d (CI: fetch-depth 0)")
def test_real_history_passes_the_chain():
    rep = _load_script("verify_preregistration").verify(ROOT, "1097c2d")
    assert rep["all_pass"], {k: v["pass"] for k, v in rep["checks"].items()}
    assert rep["checks"]["C2_results_added_in_a_descendant"]["results_commit"].startswith("35aef81")
    assert rep["limits"] and "wall-clock" in rep["limits"][0]


# fabricated histories: each breaks exactly one link and must fail exactly that check
PROTO = """# protocol

* Design: `fast, reference`, `known_write=True`.
* Sessions: N ∈ {10, 40}.
* Replications: R = 5 per N. Seed: `7 + index of N` (0..1).
* Estimator: started at `true · 1.2` for every free parameter.

## Deviations

None so far.
"""


def _results(seeds=(7, 8)):
    run = {"design": {"sequence": ["fast", "reference"], "known_write": True}, "replications": 5, "start_offset": 0.2}
    return json.dumps({"runs": [{**run, "sessions": n, "seed": s} for n, s in zip((10, 40), seeds, strict=True)]})


def _repo(tmp: Path, steps):
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@example.invalid", "PATH": __import__("os").environ["PATH"],
           "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1"}  # fmt: skip
    subprocess.run(["git", "init", "-q", str(tmp)], check=True, env=env)
    shas = []
    for files in steps:
        for rel, text in files.items():
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")
        subprocess.run(["git", "-C", str(tmp), "add", "-A"], check=True, env=env)
        subprocess.run(["git", "-C", str(tmp), "commit", "-q", "-m", "s"], check=True, env=env)
        out = subprocess.run(["git", "-C", str(tmp), "rev-parse", "HEAD"], capture_output=True, text=True, env=env)
        shas.append(out.stdout.strip())
    return shas


P, R, S = "docs/ESTIMATOR_PROTOCOL.md", "docs/ESTIMATOR_CALIBRATION_RESULTS.json", "scripts/run.py"
GOOD = [{P: PROTO, S: "x = 1\n"}, {R: _results(), P: PROTO.replace("None so far.", "None.\n\n## Outcome\n\nok")}]
BROKEN = {
    "C1_protocol_registered_before_results": [{P: PROTO, S: "x = 1\n", R: _results()}],
    "C3_frozen_protocol_unchanged": [{P: PROTO, S: "x = 1\n"}, {R: _results(), P: PROTO.replace("R = 5", "R = 6")}],
    "C4_registered_code_unchanged_at_run": [{P: PROTO, S: "x = 1\n"}, {R: _results(), S: "x = 2\n"}],
    "C5_setting_matches_protocol": [{P: PROTO, S: "x = 1\n"}, {R: _results(seeds=(7, 9))}],
}


@pytest.mark.skipif(not HAS_GIT, reason="needs git")
def test_fabricated_good_history_passes(tmp_path):
    shas = _repo(tmp_path, GOOD)
    rep = _load_script("verify_preregistration").verify(tmp_path, shas[0])
    assert rep["all_pass"], {k: v["pass"] for k, v in rep["checks"].items()}


@pytest.mark.skipif(not HAS_GIT, reason="needs git")
@pytest.mark.parametrize("check", sorted(BROKEN))
def test_fabricated_broken_link_fails_its_check(tmp_path, check):
    shas = _repo(tmp_path, BROKEN[check])
    rep = _load_script("verify_preregistration").verify(tmp_path, shas[0])
    assert not rep["all_pass"]
    assert rep["checks"][check]["pass"] is False
    if check != "C1_protocol_registered_before_results":  # C1 also breaks C2 (same commit); others are isolated
        assert [k for k, v in rep["checks"].items() if not v["pass"]] == [check]


def test_protocol_parser_reads_thin_space_numbers():
    vp = _load_script("verify_preregistration")
    text = (ROOT / "docs/ESTIMATOR_PROTOCOL.md").read_text(encoding="utf-8")
    assert re.search(r"23 372", text)  # control: the protocol writes numbers with spaces
    assert vp.protocol_setting(text)["sessions"][0] == 23372

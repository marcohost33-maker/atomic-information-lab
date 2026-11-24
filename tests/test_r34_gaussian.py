"""R34 / roadmap R13-P0: linear-Gaussian line (Kalman filter, STOP, look-ahead, reference plan).

Checked against the independent rational oracle ``tests/_gaussian_oracle.py`` (batch
information form, no Kalman recursion) and against a generative simulation of the verbal model.
"""

from __future__ import annotations

import random
from dataclasses import replace
from itertools import product
from math import sqrt

import pytest
from _gaussian_oracle import optimum as oracle_optimum
from _gaussian_oracle import posterior as oracle_posterior

from atomic_information_lab.gaussian import (
    DEFAULT_GAUSSIAN_MODEL,
    GAUSSIAN_SCOPE,
    MAX_GAUSSIAN_HORIZON,
    GaussianAction,
    GaussianModel,
    GaussianObjective,
    gaussian_table,
    initial_state,
    lookahead_plan,
    plan_value,
    reference_plan,
    solve_gaussian,
    terminal_loss,
    update,
)

TOL = 1e-12

ALT = GaussianModel(
    prior_var=0.5,
    bias_prior_var=1.0,
    drift_var=0.07,
    bias_drift_var=0.0,  # constant offset: a dropped oracle variable
    actions=(
        GaussianAction("slow", 0.0, 0.2, 0.5),  # no back-action: another dropped variable
        GaussianAction("hard", 4.0, 0.01, 3.0),
        GaussianAction("ref", 0.25, 0.05, 1.0, reads_reference=True),
    ),
    objective=GaussianObjective(alpha=0.3, energy_lambda=0.02, retention_weight=0.4),
)


def params(model):
    return {
        "prior_var": model.prior_var,
        "bias_prior_var": model.bias_prior_var,
        "drift_var": model.drift_var,
        "bias_drift_var": model.bias_drift_var,
        "actions": {a.name: (a.backaction, a.noise_var, a.energy, a.reads_reference) for a in model.actions},
    }


def objective(model):
    o = model.objective
    return (o.alpha, o.energy_lambda, o.retention_weight)


def sequences(model, max_len):
    names = [a.name for a in model.actions]
    for n in range(max_len + 1):
        yield from product(names, repeat=n)


@pytest.mark.parametrize("model", [DEFAULT_GAUSSIAN_MODEL, ALT], ids=["default", "alt"])
def test_posterior_variances_match_the_rational_batch_oracle(model):
    worst = 0.0
    for seq in sequences(model, 4):
        plan = plan_value(model, seq)
        ref = oracle_posterior(params(model), seq)
        worst = max(
            worst,
            abs(plan.var_initial - float(ref["var_initial"])),
            abs(plan.var_current - float(ref["var_current"])),
            abs(plan.damage - float(ref["damage"])),
            abs(plan.energy - float(ref["energy"])),
        )
    assert worst <= TOL


@pytest.mark.parametrize("model", [DEFAULT_GAUSSIAN_MODEL, ALT], ids=["default", "alt"])
def test_posterior_means_match_the_oracle_for_random_data(model):
    rng = random.Random(34)
    for seq in [s for s in sequences(model, 4) if s][::7]:
        ys = [round(rng.gauss(0, 1), 6) for _ in seq]
        st = initial_state(model)
        for name, y in zip(seq, ys, strict=True):
            st = update(model, st, model.action(name), y)
        ref = oracle_posterior(params(model), seq, ys)["mean"]
        assert max(abs(a - float(b)) for a, b in zip(st.mean, ref, strict=True)) <= 1e-11


@pytest.mark.parametrize("model", [DEFAULT_GAUSSIAN_MODEL, ALT], ids=["default", "alt"])
@pytest.mark.parametrize("horizon", range(5))
def test_optimum_matches_brute_force_over_all_plans(model, horizon):
    plan = solve_gaussian(model, horizon)
    value, _ = oracle_optimum(params(model), objective(model), horizon)
    assert plan.value == pytest.approx(float(value), abs=TOL)
    assert plan.scope == GAUSSIAN_SCOPE


@pytest.mark.parametrize("limit", [0.0, 0.01, 0.02, 0.05])
def test_retention_limit_matches_brute_force(limit):
    plan = solve_gaussian(DEFAULT_GAUSSIAN_MODEL, 4, retention_limit=limit)
    value, _ = oracle_optimum(params(DEFAULT_GAUSSIAN_MODEL), objective(DEFAULT_GAUSSIAN_MODEL), 4, limit)
    assert plan.value == pytest.approx(float(value), abs=TOL)
    assert plan.damage <= limit + TOL


def test_covariance_does_not_depend_on_the_data():
    """The structural fact behind 'adaptivity is worth zero' (N-47)."""
    m = DEFAULT_GAUSSIAN_MODEL
    for name in ("fast", "precise", "reference"):
        a = m.action(name)
        covs = {
            update(m, update(m, initial_state(m), m.action("fast"), y1), a, y2).cov
            for y1 in (-3, 0.1, 5)
            for y2 in (-1, 2)
        }
        assert len({tuple(round(c, 15) for r in cov for c in r) for cov in covs}) == 1


def test_deep_lookahead_equals_the_optimum_and_depth_one_is_greedy():
    m = DEFAULT_GAUSSIAN_MODEL
    for h in range(7):
        assert lookahead_plan(m, h, depth=h or 1).value == pytest.approx(solve_gaussian(m, h).value, abs=TOL)
    # depth 1 compares STOP with every single reading, as the discrete greedy rule does
    st = initial_state(m)
    one = min(m.objective.energy_lambda * a.energy + terminal_loss(m, update(m, st, a)) for a in m.actions)
    first = lookahead_plan(m, 1, 1)
    assert first.value == pytest.approx(min(one, terminal_loss(m, st)), abs=TOL)


def test_default_table_pins_and_ordering():
    rows = gaussian_table(max_horizon=5)
    for r in rows:
        assert r["optimal"] <= min(r["two_step_lookahead"], r["greedy"], r["best_forced_h"], r["reference_plan"]) + TOL
        assert r["scope"] == GAUSSIAN_SCOPE
    # exploratory finding N-48 at the default point: the two-step rule is worse than the greedy rule
    r5 = rows[5]
    assert r5["greedy"] == pytest.approx(r5["optimal"], abs=TOL)
    assert r5["two_step_lookahead"] > r5["greedy"] + 1e-3
    assert r5["two_step_plan"][0] == "reference" and r5["optimal_plan"][0] == "fast"


def test_reference_plan_shape():
    assert reference_plan(5) == ("reference", "fast", "reference", "fast", "reference")
    assert reference_plan(3, target="precise", period=3) == ("reference", "precise", "precise")
    assert reference_plan(0) == ()


def _simulate(model, seq, rng):
    x0 = rng.gauss(0, sqrt(model.prior_var))
    x, b = x0, rng.gauss(0, sqrt(model.bias_prior_var))
    ys = []
    for name in seq:
        a = model.action(name)
        x += rng.gauss(0, sqrt(model.drift_var * a.backaction))
        b += rng.gauss(0, sqrt(model.bias_drift_var))
        ys.append((b if a.reads_reference else x + b) + rng.gauss(0, sqrt(a.noise_var)))
    return x0, x, ys


def _calibration_z(truth_model, filter_model, seq, n, seed):
    """z-score of the empirical squared error of the x0 estimate against the reported variance."""
    rng = random.Random(seed)
    errs = []
    var = None
    for _ in range(n):
        x0, _x, ys = _simulate(truth_model, seq, rng)
        st = initial_state(filter_model)
        for name, y in zip(seq, ys, strict=True):
            st = update(filter_model, st, filter_model.action(name), y)
        var = st.cov[0][0]
        errs.append((st.mean[0] - x0) ** 2)
    mse = sum(errs) / n
    sd = sqrt(sum((e - mse) ** 2 for e in errs) / (n - 1) / n)
    return (mse - var) / sd


def test_generative_simulation_agrees_with_the_reported_variance():
    seq = ("fast", "reference", "fast", "precise")
    assert abs(_calibration_z(DEFAULT_GAUSSIAN_MODEL, DEFAULT_GAUSSIAN_MODEL, seq, 6000, 1)) < 4


def test_generative_simulation_has_power_against_a_wrong_model():
    """Negative control: a filter that assumes a 4x smaller offset prior is visibly miscalibrated."""
    wrong = replace(DEFAULT_GAUSSIAN_MODEL, bias_prior_var=DEFAULT_GAUSSIAN_MODEL.bias_prior_var / 4)
    seq = ("fast", "reference", "fast", "precise")
    assert _calibration_z(DEFAULT_GAUSSIAN_MODEL, wrong, seq, 6000, 1) > 4


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prior_var": -1.0},
        {"drift_var": float("nan")},
        {"actions": ()},
        {"actions": (GaussianAction("a", 1, 0.1, 1), GaussianAction("a", 1, 0.1, 1))},
        {"drift_var": 1e308, "actions": (GaussianAction("a", 10.0, 0.1, 1),)},
    ],
)
def test_invalid_models_are_rejected(kwargs):
    with pytest.raises(ValueError):
        GaussianModel(**kwargs)


@pytest.mark.parametrize(
    "args", [("stop", 1, 0.1, 1), ("a", -1, 0.1, 1), ("a", 1, 0.0, 1), ("a", 1, 0.1, float("inf"))]
)
def test_invalid_actions_are_rejected(args):
    with pytest.raises(ValueError):
        GaussianAction(*args)


def test_invalid_calls_are_rejected():
    m = DEFAULT_GAUSSIAN_MODEL
    with pytest.raises(ValueError):
        solve_gaussian(m, MAX_GAUSSIAN_HORIZON + 1)
    with pytest.raises(ValueError):
        solve_gaussian(m, 2, retention_limit=-0.1)
    with pytest.raises(ValueError):
        lookahead_plan(m, 2, depth=0)
    with pytest.raises(ValueError):
        plan_value(m, ("nope",))
    with pytest.raises(ValueError):
        update(m, initial_state(m), m.action("fast"), float("nan"))
    with pytest.raises(ValueError):
        update(m, initial_state(m), GaussianAction("other", 1, 0.1, 1))


# ------------------------------------------------------------------ pre-registered grid and CLI
def _load_runner():
    import importlib.util
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("gaussian_grid", root / "scripts" / "gaussian_grid.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, root


def test_registered_grid_results_reproduce_from_the_current_code():
    import json

    runner, root = _load_runner()
    stored = json.loads((root / "docs" / "GAUSSIAN_GRID_RESULTS.json").read_text(encoding="utf-8"))
    fresh = runner.run()
    assert {k: v["verdict"] for k, v in fresh["predictions"].items()} == {
        "G1": "CONFIRMED",
        "G2": "CONFIRMED",
        "G3": "CONFIRMED",
        "G4": "REFUTED",
        "G5": "CONFIRMED",
        "G6": "CONFIRMED",
    }
    assert {k: v["verdict"] for k, v in stored["predictions"].items()} == {
        k: v["verdict"] for k, v in fresh["predictions"].items()
    }
    assert len(stored["rows"]) == len(fresh["rows"]) == 27
    for a, b in zip(stored["rows"], fresh["rows"], strict=True):
        assert a["optimal_plan"] == b["optimal_plan"] and a["two_step_plan"] == b["two_step_plan"]
        for key in ("optimal", "two_step", "greedy", "reference_plan"):
            assert a[key] == pytest.approx(b[key], abs=TOL)
    # the results were produced by a clean tree at the registration commit or later
    assert stored["provenance"]["dirty"] is False


def test_cli_gaussian(capsys):
    import json

    from atomic_information_lab.__main__ import main

    assert main(["gaussian", "--horizon", "6"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["optimal"]["sequence"] == ["fast", "reference", "fast"]
    assert out["optimal"]["value"] == pytest.approx(solve_gaussian(DEFAULT_GAUSSIAN_MODEL, 6).value, abs=TOL)
    assert out["lookahead"]["depth"] == 2 and out["optimal"]["scope"] == GAUSSIAN_SCOPE
    assert main(["gaussian", "--horizon", "4", "--retention-limit", "0.03"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["optimal"]["damage"] <= 0.03 and "reference_plan" not in out
    for argv in (["gaussian", "--horizon", "11"], ["gaussian", "--horizon", "2", "--depth", "0"]):
        assert main(argv) == 2
    assert main(["reproduce", "--table", "gaussian", "--max-horizon", "2"]) == 0
    assert [r["horizon"] for r in json.loads(capsys.readouterr().out)["gaussian"]] == [0, 1, 2]


# ------------------------------------------------------------------ registration chain (from git)
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
def test_results_were_produced_by_the_registered_files():
    """Protocol, runner and module hashed in the results are the files of the registration commit,
    and the protocol was only extended below its rules afterwards (never edited above)."""
    import hashlib
    import json
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    stored = json.loads((root / "docs" / "GAUSSIAN_GRID_RESULTS.json").read_text(encoding="utf-8"))
    prov = stored["provenance"]
    assert prov["git_head"] == REGISTRATION and prov["dirty"] is False

    def at(path):
        return subprocess.run(
            ["git", "-C", str(root), "show", f"{REGISTRATION}:{path}"], capture_output=True, check=True
        ).stdout

    assert hashlib.sha256(at("docs/GAUSSIAN_PROTOCOL.md")).hexdigest() == prov["sha256_protocol"]
    assert hashlib.sha256(at("scripts/gaussian_grid.py")).hexdigest() == prov["sha256_runner"]
    assert hashlib.sha256(at("src/atomic_information_lab/gaussian.py")).hexdigest() == prov["sha256_module"]
    registered = at("docs/GAUSSIAN_PROTOCOL.md").decode("utf-8")
    current = (root / "docs" / "GAUSSIAN_PROTOCOL.md").read_text(encoding="utf-8")
    assert current.startswith(registered)
    # the results file is not part of the registration commit
    r = subprocess.run(
        ["git", "-C", str(root), "cat-file", "-e", f"{REGISTRATION}:docs/GAUSSIAN_GRID_RESULTS.json"],
        capture_output=True,
    )
    assert r.returncode != 0


def test_readme_table_matches_the_code():
    """Every figure of the README's Gaussian table is the code's value rounded to 7 digits."""
    from pathlib import Path

    text = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    rows = {r["horizon"]: r for r in gaussian_table(max_horizon=6)}
    keys = ("optimal", "two_step_lookahead", "greedy", "best_forced_h", "reference_plan")
    for h in (1, 2, 3, 6):
        r = rows[h]
        cells = [f"{r[k]:.7f}" for k in keys]
        line = f"| {h} | {cells[0]} | {', '.join(r['optimal_plan'])} | " + " | ".join(cells[1:]) + " |"
        assert line in text, line

"""Reproducible mutation probes: does a test suite really see a given model error?

Each mutant is one exact text replacement in a COPY of ``src/`` (the working
tree is never touched). The suite runs against that copy and every failure
is classified:

* ``relation`` -- at least one test failed on an assertion, i.e. the property
  the test states was violated;
* ``guard``    -- only fail-closed guards fired (ValueError/ArithmeticError);
  the mutant is caught, but not by the property under test;
* ``survives`` -- the suite stayed green;
* ``crash``    -- anything else (TypeError, collection error, ...); never an
  expected verdict, so a crashing suite fails the probe instead of counting
  as a kill.

A baseline run of the unmutated copy must pass every suite, and each run
checks that the package it imported is the mutated copy, so a kill cannot
come from a broken setup or a stale installed package.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "ail-mutation-probe-v1"

MODEL = "atomic_information_lab/model.py"
PLANNER = "atomic_information_lab/planner.py"
METAMORPHIC = "tests/test_metamorphic.py"
SIMULATION = "tests/test_generative_simulation.py"
CROSS = "tests/test_policy_cross_evaluation.py"
RETENTION_FILE = "atomic_information_lab/retention.py"
RETENTION = "tests/test_retention.py"
IDENT_FILE = "atomic_information_lab/identifiability.py"
IDENT = "tests/test_identifiability.py"
ESTIM_FILE = "atomic_information_lab/estimation.py"
R29 = "tests/test_r29_estimator_scope.py"
R30 = "tests/test_r30_crossfamily_ail4.py"
METRICS_FILE = "atomic_information_lab/metrics.py"
PROFILE = "tests/test_r31_policy_profile.py"
PARETO = "tests/test_r31_design_pareto.py"
PL_FILE = "atomic_information_lab/likelihood_profile.py"
PL = "tests/test_r32_profile_likelihood.py"
GAUSS_FILE = "atomic_information_lab/gaussian.py"
GAUSS = "tests/test_r34_gaussian.py"

# id: (file, old, new, what it breaks, {suite: expected verdict})
MUTANTS = {
    "M1_reference_reads_x": (
        MODEL,
        "target = (0 if action.reads_reference else x) ^ b",
        "target = x ^ b",
        "a reference read reports the stored bit instead of the bias",
        {METAMORPHIC: "relation", SIMULATION: "relation"},
    ),
    "M2_cost_not_lambda_times_energy": (
        MODEL,
        "cost = self.objective.energy_lambda * action.energy",
        "cost = self.objective.energy_lambda + 0.01 * action.energy",
        "step cost is not the product lambda * E",
        {METAMORPHIC: "relation"},
    ),
    "M3_one_kernel_for_all_hypotheses": (
        MODEL,
        "kern = self._kernel[(t, action.name)]",
        "kern = self._kernel[(0, action.name)]",
        "every hypothesis evolves with the first hypothesis' kernel",
        {METAMORPHIC: "relation", SIMULATION: "relation"},
    ),
    "M4_model_posterior_slice": (
        MODEL,
        "return tuple(sum(belief[8 * t : 8 * (t + 1)]) for t in range(len(self.hypotheses)))",
        "return tuple(sum(belief[8 * t : 8 * t + 4]) * 2 for t in range(len(self.hypotheses)))",
        "model posterior sums the wrong half of each block",
        {METAMORPHIC: "relation", SIMULATION: "relation"},
    ),
    "M5_k10_ignored": (
        MODEL,
        "p01, p10 = _step_probabilities(r01, r10)",
        "p01, p10 = _step_probabilities(r01, r01)",
        "the 1->0 rate is replaced by the 0->1 rate",
        {METAMORPHIC: "relation", SIMULATION: "relation"},
    ),
    "M6_bias_flip_depends_on_x": (
        MODEL,
        "(f if b != b2 else 1.0 - f)",
        "((f if x == 0 else min(0.5, 4 * f)) if b != b2 else 1.0 - (f if x == 0 else min(0.5, 4 * f)))",
        "the inversion bit flips four times as often when x = 1 (kernel stays stochastic)",
        {METAMORPHIC: "relation", SIMULATION: "relation"},
    ),
    "M7_rates_swapped_everywhere": (
        MODEL,
        "p01, p10 = _step_probabilities(r01, r10)",
        "p10, p01 = _step_probabilities(r01, r10)",
        "k01 and k10 swapped consistently (a specification-level error)",
        {METAMORPHIC: "survives", SIMULATION: "relation"},
    ),
    "M8_policy_looks_one_step_short": (
        PLANNER,
        "return _solve(problem, b, h, allow_stop).action",
        "return _solve(problem, b, max(h - 1, 1), allow_stop).action",
        "the decision rule plans one step short; solve() values are unchanged",
        {CROSS: "relation"},
    ),
    "M9_pathwise_limit_loosened": (
        RETENTION_FILE,
        "    if pc <= delta:\n",
        "    if pc <= delta + 0.02:\n",
        "the hard limit admits stops up to 2 percentage points above delta",
        {RETENTION: "relation"},
    ),
    "M10_mixture_weight_off": (
        RETENTION_FILE,
        "q = (delta - hi[2]) / (lo[2] - hi[2])",
        "q = 0.9 * (delta - hi[2]) / (lo[2] - hi[2])",
        "the randomised policy under-uses the allowed risk (still feasible, no longer optimal)",
        {RETENTION: "relation"},
    ),
    "M11_multiplier_ignored_at_stop": (
        RETENTION_FILE,
        "best = (loss + mu * pc, loss, pc, STOP)",
        "best = (loss, loss, pc, STOP)",
        "the Lagrangian forgets the retention price when stopping (the multiplier search then refuses)",
        {RETENTION: "guard"},
    ),
    "M12_pareto_front_collapsed": (
        RETENTION_FILE,
        "sorted(points, key=lambda t: (t[0], t[1]))",
        "sorted(points, key=lambda t: (t[1], t[0]))",
        "the Pareto pruning keeps only the cheapest point (the solver then refuses as infeasible)",
        {RETENTION: "guard"},
    ),
    "M13_pareto_pruning_too_coarse": (
        RETENTION_FILE,
        "if best_j is None or j < best_j - TIE_TOL:",
        "if best_j is None or j < best_j - 0.01:",
        "the Pareto pruning drops plans less than 0.01 cheaper: feasible but suboptimal deterministic values",
        {RETENTION: "relation"},
    ),
    "M14_prep_error_ignored_with_known_write": (
        IDENT_FILE,
        "[((0,), 0.5, eps), ((1,), 0.5, 1 - eps)]",
        "[((0,), 0.5, 0 * eps), ((1,), 0.5, 1 - 0 * eps)]",
        "a recorded write is treated as a perfect preparation",
        {IDENT: "relation"},
    ),
    "M15_fisher_divides_by_p_squared": (
        IDENT_FILE,
        "gi = p.g[i] / p.v",
        "gi = p.g[i] / p.v**2",
        "Fisher information with the wrong weight 1/p^2 (ranks often unchanged, values wrong)",
        {IDENT: "relation"},
    ),
    "M16_windows_ignore_reference_reads": (
        IDENT_FILE,
        "if not cal.action(name).reads_reference:",
        "if True:",
        "every read closes a back-action window, so the window witness is never found",
        {IDENT: "relation"},
    ),
    "M17_rank_tolerance_too_loose": (
        IDENT_FILE,
        "if lk <= RANK_TOL:",
        "if lk <= 1e-2:",
        "weakly informative directions are called non-identifiable",
        {IDENT: "relation"},
    ),
    "M18_no_zero_information_threshold": (
        IDENT_FILE,
        "ZERO_INFO_TOL = 1e-20",
        "ZERO_INFO_TOL = 0.0",
        "exact-in-theory zero information left as rounding noise (1e-33) is counted as information",
        {IDENT: "relation"},
    ),
    "M19_scope_back_to_necessary": (
        IDENT_FILE,
        'SESSIONS_SCOPE = "CRLB_UNBIASED_REGULAR_LOWER_BOUND_ONLY"',
        'SESSIONS_SCOPE = "CRLB_NECESSARY_LOWER_BOUND_ONLY"',
        "the R27 token returns: the count is called necessary without the unbiasedness condition (AIL-R29-01)",
        {R29: "relation"},
    ),
    "M20_discrimination_without_scope": (
        IDENT_FILE,
        "    sessions_scope: str = DISCRIMINATION_SCOPE\n",
        "",
        "the sufficient Chernoff count reaches Python consumers without a machine-readable scope",
        {R29: "relation"},
    ),
    "M21_chernoff_count_without_one_half": (
        IDENT_FILE,
        "ceil(log(1 / (2 * alpha)) / c)",
        "ceil(log(1 / alpha) / c)",
        "the factor 1/2 of the equal-prior Bayes error is dropped: the count is still sufficient but larger",
        {R29: "relation"},
    ),
    "M22_failed_fits_dropped": (
        ESTIM_FILE,
        "failures[reason] = failures.get(reason, 0) + 1",
        "pass",
        "failed MLE fits vanish from the report instead of being counted",
        {R29: "relation"},
    ),
    "M23_mle_stops_after_one_step": (
        ESTIM_FILE,
        "if rel < _REL_STEP_TOL:",
        "if True:",
        "Fisher scoring stops after one step and calls the result converged",
        {R29: "relation"},
    ),
    "M24_crlb_rse_without_lower_bound_scope": (
        ESTIM_FILE,
        "                SESSIONS_SCOPE,\n                sd_mcse",
        "                ESTIMATOR_SCOPE,\n                sd_mcse",
        "the Cramer-Rao figure of the estimator report travels with the Monte-Carlo scope instead (AIL4-01)",
        {R30: "relation"},
    ),
    "M25_all_replication_coverage_ignores_failures": (
        ESTIM_FILE,
        "(hits + replications - m) / replications",
        "hits / m",
        "the upper all-replication coverage bound silently becomes the survivor coverage (AIL4-03)",
        {R30: "relation"},
    ),
    "M26_rse_mcse_assumes_normality": (
        ESTIM_FILE,
        "var_s2 = max(m4 - m2 * m2 * (m - 3) / (m - 1), 0.0) / m",
        "var_s2 = 2 * (m2 * m / (m - 1)) ** 2 / (m - 1)",
        "the Monte-Carlo error of the RSE ignores the fourth moment: too small for heavy tails",
        {R30: "relation"},
    ),
    "M27_rse_mcse_on_two_fits": (
        ESTIM_FILE,
        'if m >= 4 else float("nan")',
        'if m >= 2 else float("nan")',
        "two or three converged fits make the whole calibration raise instead of reporting nan",
        {R30: "guard"},
    ),
    "M28_information_about_the_current_bit": (
        METRICS_FILE,
        'acc["h0"] += mass * binary_entropy_bits(min(max(p0, 0.0), 1.0))',
        'acc["h0"] += mass * binary_entropy_bits(min(max(pt, 0.0), 1.0))',
        "the information figure uses the posterior of the current bit instead of the written bit",
        {PROFILE: "relation"},
    ),
    "M29_retention_not_weighted_by_path": (
        METRICS_FILE,
        'acc["chg"] += mass * m["p_changed"]',
        'acc["chg"] += m["p_changed"] / (2 ** len(hist))',
        "P(x_T != x0) averages over leaves as if every outcome were equally likely",
        {PROFILE: "relation"},
    ),
    "M30_energy_price_counts_steps": (
        METRICS_FILE,
        'energy_cost = o.energy_lambda * acc["energy"]',
        "energy_cost = o.energy_lambda * sum(k * q for k, q in enumerate(stop_mass))",
        "the energy price counts readings instead of their energy",
        {PROFILE: "relation"},
    ),
    "M31_identical_designs_remove_each_other": (
        IDENT_FILE,
        "if j != i and cj != ci and all(a <= b for a, b in zip(cj, ci, strict=True)):",
        "if j != i and all(a <= b for a, b in zip(cj, ci, strict=True)):",
        "weak dominance: two designs with identical criteria both vanish from the Pareto list",
        {PARETO: "relation"},
    ),
    "M32_pareto_ignores_energy": (
        IDENT_FILE,
        "        float(score.energy_per_session),\n    )",
        "        0.0,\n    )",
        "energy per session is not a criterion: cheaper designs with a larger bound are dropped",
        {PARETO: "relation"},
    ),
    "M33_condition_number_inverted": (
        IDENT_FILE,
        "condition = max(lam) / min_eig",
        "condition = min_eig / max(lam)",
        "the condition number is reported as lambda_min / lambda_max",
        {PARETO: "relation"},
    ),
    "M34_profile_is_a_slice": (
        PL_FILE,
        "    truth_values = truth.parameters()\n    nuis = tuple(f for f in free if f != name)",
        "    truth_values = truth.parameters()\n    nuis = ()",
        "the other parameters are held at the truth instead of re-optimised (a slice, not a profile)",
        {PL: "relation"},
    ),
    "M35_wilks_threshold_is_z": (
        PL_FILE,
        "    return z * z",
        "    return z",
        "the interval threshold uses the normal quantile instead of chi-square(1)",
        {PL: "relation"},
    ),
    "M36_unconverged_point_bounds": (
        PL_FILE,
        "            if pt.converged and pt.statistic > threshold:",
        "            if pt.statistic > threshold:",
        "a non-converged (over-estimated) profile value may close an interval",
        {PL: "relation"},
    ),
    "M37_bounded_from_the_steeper_side": (
        PL_FILE,
        "    pr = min(low_plateau, high_plateau)",
        "    pr = max(low_plateau, high_plateau)",
        "bounded_sessions follows the steeper side, so one side is still open at that N",
        {PL: "relation"},
    ),
    "M38_second_valley_ignored": (
        PL_FILE,
        "    return dip",
        "    return None",
        "a profile that falls back to a second zero (global ambiguity) is reported as monotone",
        {PL: "relation"},
    ),
    "M39_converged_beats_lower": (
        PL_FILE,
        "        if best is None or v < best[0] or (v == best[0] and ok and not best[2]):",
        "        if best is None or (ok and not best[2]) or (ok == best[2] and v < best[0]):",
        "of two starts a converged but higher local minimum replaces a lower value (R32 behaviour)",
        {PL: "relation"},
    ),
    "M40_damped_decrement_converges": (
        PL_FILE,
        "        if cur == 0.0 or (mu == 0.0 and decrement <= _INNER_DECREMENT * cur):",
        "        if cur == 0.0 or decrement <= _INNER_DECREMENT * cur:",
        "the decrement of a damped step ends the search (R32 behaviour)",
        {PL: "relation"},
    ),
    "M41_shortened_step_converges": (
        PL_FILE,
        "        if rel < _INNER_REL_STEP and full and mu == 0.0:",
        "        if rel < _INNER_REL_STEP:",
        "a tiny step after many halvings counts as convergence (R32 behaviour)",
        {PL: "relation"},
    ),
    "M42_no_backward_sweep": (
        PL_FILE,
        "        if 2.0 * v < cur.statistic:",
        "        if False:",
        "only the forward continuation runs; a point stuck in a higher valley stays there",
        {PL: "relation"},
    ),
    "M43_observed_lambda_not_centred": (
        PL_FILE,
        "    lam = 2.0 * total * (v - kl_full)\n    return max(lam, 0.0), ok, lam < 0.0",
        "    lam = 2.0 * total * v\n    return max(lam, 0.0), ok, lam < 0.0",
        "the observed-data statistic forgets to subtract the divergence at the full estimate",
        {PL: "relation"},
    ),
    "M44_unobserved_mass_skipped": (
        PL_FILE,
        "            total += max(p.v, 0.0)\n            continue",
        "            continue",
        "outcomes with zero observed count are skipped in KL(emp || p) (cross-family review of PR #7)",
        {PL: "relation"},
    ),
    "M45_zero_start_frozen": (
        PL_FILE,
        "        scaled = [fim[i][i] * max(vals[i], 1e-6) ** 2 for i in range(k)]",
        "        scaled = [fim[i][i] * vals[i] ** 2 for i in range(k)]",
        "'no information' decided in log coordinates; SURVIVES by design: the zero start is defended twice "
        "(start projection, and the line search projects every value into the open domain), so removing one "
        "defence alone leaves the Codex case correct -- a documented limit of single-edit mutants (N-46)",
        {PL: "survives"},
    ),
    "M46_gauss_reference_reads_target": (
        GAUSS_FILE,
        "    return (0.0, 0.0, 1.0) if action.reads_reference else (0.0, 1.0, 1.0)",
        "    return (0.0, 1.0, 1.0)",
        "a Gaussian reference read sees the stored value instead of the offset alone (R34)",
        {GAUSS: "relation"},
    ),
    "M47_gauss_backaction_ignored": (
        GAUSS_FILE,
        "    q = model.drift_var * action.backaction",
        "    q = model.drift_var",
        "back-action no longer speeds up the drift of the stored value (R34)",
        {GAUSS: "relation"},
    ),
    "M48_gauss_no_stop_before_horizon": (
        GAUSS_FILE,
        "    options = [(terminal_loss(model, st), ())]  # STOP first: wins ties",
        "    options = [(terminal_loss(model, st), ())] if h == 0 else []",
        "the Gaussian planner must use its whole horizon (STOP removed) (R34)",
        {GAUSS: "relation"},
    ),
    "M49_gauss_innovation_sign": (
        GAUSS_FILE,
        "        mean = tuple(mean[i] + k[i] * innov for i in range(3))",
        "        mean = tuple(mean[i] - k[i] * innov for i in range(3))",
        "Kalman mean update with the wrong sign; the covariances (and every planning figure) stay exact, "
        "so only the oracle means and the generative simulation can see it (R34)",
        {GAUSS: "relation"},
    ),
    "M50_gauss_lookahead_takes_last_action": (
        GAUSS_FILE,
        "        a = model.action(first[0])",
        "        a = model.action(first[-1])",
        "the receding-horizon rule executes the last action of its plan instead of the first (R34)",
        {GAUSS: "relation"},
    ),
    "M51_gauss_state_det_unchecked": (
        GAUSS_FILE,
        "            or det < -tol * scale**3\n",
        "",
        "GaussianState accepts a matrix whose principal 1x1 and 2x2 minors are fine but whose "
        "determinant is negative (R34b defect 3)",
        {GAUSS: "relation"},
    ),
    "M52_gauss_table_default_names": (
        GAUSS_FILE,
        "        ref = fixed_plan(model, model_reference_plan(model, h))",
        "        ref = fixed_plan(model, reference_plan(h))",
        "the table's reference plan uses the default action names regardless of the catalogue (R34b defect 1); "
        "caught as a guard: the unknown action name raises ValueError, exactly how the defect showed",
        {GAUSS: "guard"},
    ),
}
_RELATION_PREFIXES = ("AssertionError", "assert ", "Failed: DID NOT RAISE")
_GUARD_ERRORS = ("ValueError", "ArithmeticError", "ZeroDivisionError", "OverflowError", "FloatingPointError")


def classify(message: str) -> str:
    """Verdict of one failing test from its junit failure message."""
    if message.startswith(_RELATION_PREFIXES):
        return "relation"
    if message.split(":", 1)[0].strip() in _GUARD_ERRORS:
        return "guard"
    return "crash"


def _copy_src(dest: Path) -> Path:
    src = dest / "src"
    shutil.copytree(ROOT / "src", src, ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"))
    return src


def _run_suite(src: Path, suite: str, work: Path) -> tuple[str, list[str]]:
    env = {**os.environ, "PYTHONPATH": str(src), "PYTHONDONTWRITEBYTECODE": "1"}
    probe = subprocess.run(
        [sys.executable, "-c", "import atomic_information_lab as m; print(m.__file__)"],
        capture_output=True,
        text=True,
        env=env,
        cwd=work,
    )
    if probe.returncode != 0 or not Path(probe.stdout.strip()).resolve().is_relative_to(src.resolve()):
        raise RuntimeError(f"probe would not test the copy: imported {probe.stdout.strip()!r} {probe.stderr}")
    report = work / "report.xml"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "pythonpath=",  # pyproject's pythonpath=src would shadow the copy
            f"--junitxml={report}",
            str(ROOT / suite),
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    failed, kinds = [], set()
    for case in ET.parse(report).getroot().iter("testcase"):
        for tag in ("failure", "error"):
            node = case.find(tag)
            if node is not None:
                failed.append(case.get("name"))
                kinds.add(classify(node.get("message") or ""))
    report.unlink()
    if not failed:
        return "survives", []
    for verdict in ("crash", "relation", "guard"):  # a crash anywhere makes the run untrustworthy
        if verdict in kinds:
            return verdict, failed


def probe(mutant_ids=None) -> dict:
    ids = list(MUTANTS) if mutant_ids is None else list(mutant_ids)
    unknown = set(ids) - set(MUTANTS)
    if unknown:
        raise ValueError(f"unknown mutants: {sorted(unknown)}")
    suites = sorted({s for i in ids for s in MUTANTS[i][4]})
    results = {}
    with tempfile.TemporaryDirectory(prefix="ail-mutation-") as tmp:
        work = Path(tmp)
        src = _copy_src(work / "baseline")
        for suite in suites:
            verdict, failed = _run_suite(src, suite, work)
            if verdict != "survives":
                raise RuntimeError(f"baseline is not green for {suite}: {failed[:5]}")
        for mid in ids:
            rel, old, new, what, expected = MUTANTS[mid]
            src = _copy_src(work / mid)
            target = src / rel
            text = target.read_text(encoding="utf-8")
            if text.count(old) != 1:
                raise RuntimeError(f"{mid}: anchor must occur exactly once in {rel}, found {text.count(old)}")
            target.write_text(text.replace(old, new), encoding="utf-8")
            row = {"breaks": what, "suites": {}}
            for suite, want in expected.items():
                verdict, failed = _run_suite(src, suite, work)
                row["suites"][suite] = {"expected": want, "observed": verdict, "failed_tests": len(failed)}
            row["as_expected"] = all(v["expected"] == v["observed"] for v in row["suites"].values())
            results[mid] = row
    return {
        "schema": SCHEMA,
        "baseline": "green",
        "mutants": results,
        "all_as_expected": all(r["as_expected"] for r in results.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mutants", nargs="*", help="mutant ids (default: all)")
    args = parser.parse_args()
    try:
        out = probe(args.mutants or None)
    except (ValueError, RuntimeError) as exc:
        print(json.dumps({"schema": SCHEMA, "verdict": "fail", "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if out["all_as_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

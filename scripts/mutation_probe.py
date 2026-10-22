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

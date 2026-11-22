"""Run the pre-registered falsification tests of docs/PL_FALSIFICATION_PROTOCOL.md (R33) and write JSON.

Usage: python scripts/pl_falsification.py [--part A|B|all] [--replications R] [--workers W] [--out PATH]

Part A: likelihood-ratio interval coverage on the registered data (same seeds as
docs/ESTIMATOR_CALIBRATION_RESULTS.json), with the provenance check A0.
Part B: Monte-Carlo MLE on the 4-step design at the two registered N, scored
against docs/PL_TRANSFER_PREDICTIONS.json (B1-B6).

The outcome counts are drawn sequentially from ``random.Random(seed)`` exactly as
``estimation.estimator_calibration`` draws them; only the fits run in parallel.
A smaller --replications is a smoke run and is recorded as such.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import random
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atomic_information_lab.estimation import Z95, estimator_calibration, fit_mle, sample_counts  # noqa: E402
from atomic_information_lab.identifiability import Calibration, Design, parameter_names  # noqa: E402

PROTOCOL = "docs/PL_FALSIFICATION_PROTOCOL.md"
PREDICTIONS = "docs/PL_TRANSFER_PREDICTIONS.json"
REGISTERED = "docs/ESTIMATOR_CALIBRATION_RESULTS.json"
REPLICATIONS = 1000
CHI2_95 = 3.841458820694124
START_OFFSET = 0.2
DESIGN_A = ("fast", "reference", "precise", "reference", "fast", "precise")
RUNS_A = ((23_372, 20261104), (373_950, 20261106))
DESIGN_B = ("fast", "precise", "reference", "precise")
RUNS_B = ((417_942, 20261122), (30_894, 20261123))
BOUND_FILES = (
    PROTOCOL,
    PREDICTIONS,
    "scripts/pl_falsification.py",
    "src/atomic_information_lab/estimation.py",
    "src/atomic_information_lab/identifiability.py",
    "src/atomic_information_lab/model.py",
)


def _setting():
    truth = Calibration(prep_error=0.02)
    free = tuple(f for f in parameter_names(truth) if f != "backaction:fast")
    start = truth.with_values({f: truth.parameters()[f] * (1 + START_OFFSET) for f in free})
    return truth, free, start


def _lr_one(args):
    """Full fit, then one restricted fit per parameter (fixed at the truth); Lambda per parameter."""
    design_seq, counts = args
    truth, free, start = _setting()
    design = Design(design_seq, True)
    full = fit_mle(start, design, free, counts)
    row = {"converged": full.converged and full.std_errors is not None, "reason": full.reason}
    if not row["converged"]:
        return row
    est = dict(zip(free, full.values, strict=True))
    row["estimates"] = list(full.values)
    row["std_errors"] = list(full.std_errors)
    lam, above, restr_fail = [], [], []
    tv = truth.parameters()
    for name in free:
        others = tuple(f for f in free if f != name)
        s = truth.with_values({**{f: est[f] for f in others}, name: tv[name]})
        r = fit_mle(s, design, others, counts)
        if not r.converged:
            restr_fail.append(name)
        d = 2 * (full.log_likelihood - r.log_likelihood)
        above.append(d < 0)
        lam.append(max(d, 0.0))
    row["lambda"] = lam
    row["restricted_above_full"] = above
    row["restricted_not_converged"] = restr_fail
    return row


def _draw(design_seq, sessions, seed, reps):
    truth, _, _ = _setting()
    rng = random.Random(seed)
    design = Design(design_seq, True)
    return [sample_counts(truth, design, sessions, rng) for _ in range(reps)]


def part_a(reps: int, workers: int) -> dict:
    truth, free, _ = _setting()
    tv = truth.parameters()
    reg = json.loads((ROOT / REGISTERED).read_text(encoding="utf-8"))
    out = []
    for sessions, seed in RUNS_A:
        samples = _draw(DESIGN_A, sessions, seed, reps)
        with ProcessPoolExecutor(max_workers=workers) as ex:
            rows = list(ex.map(_lr_one, [(DESIGN_A, c) for c in samples], chunksize=8))
        fits = [r for r in rows if r["converged"]]
        m = len(fits)
        params = []
        for i, name in enumerate(free):
            t = tv[name]
            xs = [r["estimates"][i] for r in fits]
            mean = sum(xs) / m
            sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (m - 1))
            wald_lo = sum(t < r["estimates"][i] - Z95 * r["std_errors"][i] for r in fits)  # truth below
            wald_hi = sum(t > r["estimates"][i] + Z95 * r["std_errors"][i] for r in fits)  # truth above
            lr_miss = [r for r in fits if r["lambda"][i] > CHI2_95]
            lr_lo = sum(r["estimates"][i] > t for r in lr_miss)  # interval lies above the truth
            lr_hi = sum(r["estimates"][i] <= t for r in lr_miss)
            hits = m - len(lr_miss)
            params.append(
                {
                    "name": name,
                    "relative_bias": (mean - t) / abs(t),
                    "empirical_rse": sd / abs(t),
                    "wald_coverage": (m - wald_lo - wald_hi) / m,
                    "wald_miss_truth_below": wald_lo,
                    "wald_miss_truth_above": wald_hi,
                    "lr_coverage": hits / m,
                    "lr_coverage_mcse": math.sqrt(max(hits / m * (1 - hits / m), 1e-300) / m),
                    "lr_coverage_all_low": hits / reps,
                    "lr_coverage_all_high": (hits + reps - m) / reps,
                    "lr_miss_truth_below": lr_lo,
                    "lr_miss_truth_above": lr_hi,
                    "restricted_above_full": sum(r["restricted_above_full"][i] for r in fits),
                    "restricted_not_converged": sum(name in r["restricted_not_converged"] for r in fits),
                }
            )
        regrun = next((r for r in reg["runs"] if r["sessions"] == sessions), None)
        a0 = None
        if regrun is not None and reps == reg["replications"]:
            regp = {p["name"]: p for p in regrun["parameters"]}
            dev = max(
                max(
                    abs(p["relative_bias"] - regp[p["name"]]["relative_bias"]),
                    abs(p["empirical_rse"] - regp[p["name"]]["empirical_rse"]),
                    abs(p["wald_coverage"] - regp[p["name"]]["coverage"]),
                )
                for p in params
            )
            a0 = {"max_abs_deviation": dev, "pass": dev <= 1e-12 and m == regrun["converged"]}
        a1 = all(0.93 <= p["lr_coverage"] <= 0.97 for p in params)
        out.append(
            {
                "sessions": sessions,
                "seed": seed,
                "replications": reps,
                "converged": m,
                "failures": sorted({r["reason"] for r in rows if not r["converged"]}),
                "failed_fits": reps - m,
                "parameters": params,
                "A0_provenance": a0,
                "A1_lr_coverage_in_screen": a1,
            }
        )
    a3 = {}
    first = out[0]
    for p in first["parameters"]:
        if p["name"] in ("backaction:precise", "backaction:reference"):
            lr = abs(p["lr_miss_truth_below"] - p["lr_miss_truth_above"])
            wd = abs(p["wald_miss_truth_below"] - p["wald_miss_truth_above"])
            a3[p["name"]] = {"lr_imbalance": lr, "wald_imbalance": wd, "pass": lr <= wd}
    return {"runs": out, "A3_tail_balance_at_23372": a3}


def part_b(reps: int) -> dict:
    truth, free, _ = _setting()
    pred = json.loads((ROOT / PREDICTIONS).read_text(encoding="utf-8"))
    pmap = {r["sessions"]: {p["name"]: p for p in r["parameters"]} for r in pred["runs"]}
    design = Design(DESIGN_B, True)
    runs = []
    for sessions, seed in RUNS_B:
        t0 = time.time()
        res = estimator_calibration(truth, design, free, sessions, reps, seed, START_OFFSET)
        rows = []
        for p in res.parameters:
            pr = pmap[sessions][p.name]
            flagged = abs(p.relative_bias) > 3 * p.relative_bias_mcse and abs(p.relative_bias) > 0.02
            pb, prr = pr["predicted_bias"], pr["predicted_rse_ratio"]
            ratio_mcse = p.empirical_rse_mcse / p.crlb_rse if math.isfinite(p.empirical_rse_mcse) else float("nan")
            rows.append(
                {
                    "name": p.name,
                    "relative_bias": p.relative_bias,
                    "relative_bias_mcse": p.relative_bias_mcse,
                    "rse_ratio": p.rse_ratio,
                    "rse_ratio_mcse": ratio_mcse,
                    "coverage": p.coverage,
                    "s2_flag": flagged,
                    "predicted_bias": pb,
                    "predicted_rse_ratio": prr,
                    "sign_ok": None if not flagged or pb is None else (p.relative_bias > 0) == (pb > 0),
                    "B4_bias_within_3mcse": None
                    if pb is None
                    else abs(p.relative_bias - pb) <= 3 * p.relative_bias_mcse,
                    "B5_rse_within_3mcse": None if prr is None else abs(p.rse_ratio - prr) <= 3 * ratio_mcse,
                }
            )
        runs.append(
            {
                "sessions": sessions,
                "seed": seed,
                "replications": reps,
                "converged": res.converged,
                "failures": [list(f) for f in res.failures],
                "failed_fits": reps - res.converged,
                "seconds": round(time.time() - t0, 1),
                "parameters": rows,
            }
        )
    by_n = {r["sessions"]: r for r in runs}
    flagged = [p for r in runs for p in r["parameters"] if p["s2_flag"]]
    b1 = {"flagged": [p["name"] for p in flagged], "pass": all(p["sign_ok"] for p in flagged), "vacuous": not flagged}
    small = {p["name"]: p for p in by_n[30_894]["parameters"]}
    b2 = all(
        small[n]["relative_bias"] > 3 * small[n]["relative_bias_mcse"]
        for n in ("backaction:precise", "backaction:reference")
    )
    b3 = not any(p["s2_flag"] for p in by_n[417_942]["parameters"])
    b4 = [p["B4_bias_within_3mcse"] for r in runs for p in r["parameters"] if p["B4_bias_within_3mcse"] is not None]
    b5 = [p["B5_rse_within_3mcse"] for r in runs for p in r["parameters"] if p["B5_rse_within_3mcse"] is not None]
    b6 = by_n[30_894]["failed_fits"] > by_n[417_942]["failed_fits"]
    verdict = "transferred" if (b1["pass"] and b2 and b3) else "not transferred"
    return {
        "runs": runs,
        "B1_sign": b1,
        "B2_effect_present_at_30894": b2,
        "B3_effect_absent_at_417942": b3,
        "B4_bias_within_3mcse": {"pass": sum(b4), "of": len(b4)},
        "B5_rse_within_3mcse": {"pass": sum(b5), "of": len(b5)},
        "B6_more_failures_where_interval_open": b6,
        "N37_verdict": verdict,
    }


def provenance() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "git_head": git("rev-parse", "HEAD"),
        "bound_files_dirty": bool(git("status", "--porcelain", "--", *BOUND_FILES)),
        "sha256": {f: hashlib.sha256((ROOT / f).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for f in BOUND_FILES},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--part", choices=("A", "B", "all"), default="all")
    ap.add_argument("--replications", type=int, default=REPLICATIONS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if not 2 <= a.replications <= 100_000:
        raise SystemExit("--replications must be in 2..100000")
    t0 = time.time()
    out = {
        "protocol": PROTOCOL,
        "replications": a.replications,
        "protocol_replications": REPLICATIONS,
        "python": platform.python_version(),
        "provenance": provenance(),
    }
    if a.part in ("A", "all"):
        out["part_a"] = part_a(a.replications, a.workers)
    if a.part in ("B", "all"):
        out["part_b"] = part_b(a.replications)
    out["seconds"] = round(time.time() - t0, 1)
    text = json.dumps(out, indent=2)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

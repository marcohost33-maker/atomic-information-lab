"""Runner for the pre-registered R34 grid study (docs/GAUSSIAN_PROTOCOL.md).

Writes docs/GAUSSIAN_GRID_RESULTS.json (or stdout with --stdout). Deterministic; no random numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atomic_information_lab.gaussian import (  # noqa: E402
    DEFAULT_GAUSSIAN_MODEL,
    GAUSSIAN_SCOPE,
    GaussianObjective,
    fixed_plan,
    lookahead_plan,
    reference_plan,
    solve_gaussian,
)

SCHEMA = "ail-gaussian-grid-v1"
HORIZON = 6
BIAS = (0.05, 0.25, 1.0)
DRIFT = (0.005, 0.02, 0.08)
LAMBDA = (0.004, 0.012, 0.036)
DEFAULT_POINT = (0.25, 0.02, 0.012)
STRICT = 1e-9


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run() -> dict:
    rows = []
    for bias, drift, lam in product(BIAS, DRIFT, LAMBDA):
        m = replace(
            DEFAULT_GAUSSIAN_MODEL,
            bias_prior_var=bias,
            drift_var=drift,
            objective=GaussianObjective(alpha=0.5, energy_lambda=lam, retention_weight=0.0),
        )
        opt = solve_gaussian(m, HORIZON)
        two = lookahead_plan(m, HORIZON, 2)
        greedy = lookahead_plan(m, HORIZON, 1)
        ref = fixed_plan(m, reference_plan(HORIZON))
        rows.append(
            {
                "bias_prior_var": bias,
                "drift_var": drift,
                "energy_lambda": lam,
                "optimal": opt.value,
                "optimal_plan": list(opt.sequence),
                "two_step": two.value,
                "two_step_plan": list(two.sequence),
                "greedy": greedy.value,
                "greedy_plan": list(greedy.sequence),
                "reference_plan": ref.value,
                "r_two_step": (two.value - opt.value) / opt.value,
                "r_greedy": (greedy.value - opt.value) / opt.value,
                "r_reference_plan": (ref.value - opt.value) / opt.value,
            }
        )
    n = len(rows)
    g1 = all(r["optimal"] <= min(r["two_step"], r["greedy"], r["reference_plan"]) + 1e-12 for r in rows)
    g2_points = [
        r
        for r in rows
        if r["two_step"] > r["greedy"] + STRICT
        and (r["bias_prior_var"], r["drift_var"], r["energy_lambda"]) != DEFAULT_POINT
    ]
    mean_two = sum(r["r_two_step"] for r in rows) / n
    mean_greedy = sum(r["r_greedy"] for r in rows) / n
    max_two = max(r["r_two_step"] for r in rows)
    g5_count = sum(r["r_reference_plan"] >= 0.05 for r in rows)
    reads_target = {a.name for a in DEFAULT_GAUSSIAN_MODEL.actions if not a.reads_reference}
    reads_ref = {a.name for a in DEFAULT_GAUSSIAN_MODEL.actions if a.reads_reference}
    g6_counter = [
        r
        for r in rows
        if r["bias_prior_var"] >= 0.25
        and reads_target & set(r["optimal_plan"])
        and not reads_ref & set(r["optimal_plan"])
    ]

    def verdict(ok: bool) -> str:
        return "CONFIRMED" if ok else "REFUTED"

    predictions = {
        "G1": {"verdict": verdict(g1)},
        "G2": {"verdict": verdict(len(g2_points) >= 2), "count_excluding_default": len(g2_points)},
        "G3": {"verdict": verdict(mean_two <= mean_greedy), "mean_r_two_step": mean_two, "mean_r_greedy": mean_greedy},
        "G4": {"verdict": verdict(max_two <= 0.10), "max_r_two_step": max_two},
        "G5": {"verdict": verdict(g5_count >= 20), "count": g5_count},
        "G6": {"verdict": verdict(not g6_counter), "counter_examples": len(g6_counter)},
    }
    return {
        "schema": SCHEMA,
        "scope": GAUSSIAN_SCOPE,
        "horizon": HORIZON,
        "provenance": {
            "git_head": _git("rev-parse", "HEAD"),
            "dirty": bool(_git("status", "--porcelain", "--untracked-files=no")),
            "sha256_protocol": _sha(ROOT / "docs" / "GAUSSIAN_PROTOCOL.md"),
            "sha256_runner": _sha(Path(__file__)),
            "sha256_module": _sha(ROOT / "src" / "atomic_information_lab" / "gaussian.py"),
        },
        "predictions": predictions,
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    out = json.dumps(run(), indent=2, sort_keys=True) + "\n"
    if args.stdout:
        sys.stdout.write(out)
    else:
        (ROOT / "docs" / "GAUSSIAN_GRID_RESULTS.json").write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

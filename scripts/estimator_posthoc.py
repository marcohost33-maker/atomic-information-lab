"""Post-hoc analysis of the pre-registered MLE calibration (R30; NOT confirmatory).

Usage:
    python scripts/estimator_posthoc.py --fresh RUN.json [--registered PATH] [--out PATH]

``RUN.json`` is a new run of ``scripts/estimator_calibration.py`` with the protocol setting.
The script first checks that the fresh run reproduces EVERY field of the registered results
exactly (float equality; only ``python``/``seconds`` may differ); then, from the fields the
registered file lacks (Monte-Carlo error of the RSE, coverage over all replications), it adds:

* ``rse_ratio_ci95``: ``(empirical_rse +- 1.96 * empirical_rse_mcse) / crlb_rse`` -- the only
  basis for a statement such as "efficient within Monte-Carlo error" (AIL-02);
* ``coverage_wilson95``: Wilson interval of the converged-fit coverage, and
  ``coverage_all_low/high`` over all replications (survivor selection, AIL4-03);
* coverage flags under the registered screen [0.93, 0.97] AND under a family-wise band
  (Bonferroni over all parameter x N cells), because 40 cells were screened (multiplicity).

The registered endpoints are the protocol's. Nothing here changes P1, and no equivalence
margin was registered, so the intervals are descriptive, not an equivalence test.
"""

from __future__ import annotations

import argparse
import json
import sys
from math import sqrt
from statistics import NormalDist

REGISTERED = "docs/ESTIMATOR_CALIBRATION_RESULTS.json"
Z95 = NormalDist().inv_cdf(0.975)
VOLATILE = {"python", "seconds", "provenance"}


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    lo = 0.0 if k == 0 else max(0.0, centre - half)  # exact at the edges (else 1e-17 rounding residue)
    hi = 1.0 if k == n else min(1.0, centre + half)
    return lo, hi


def reproduction(registered: dict, fresh: dict) -> dict:
    """Every registered leaf must be equal in the fresh run; extra fresh fields are allowed."""
    mismatches, compared = [], 0

    def walk(a, b, path):
        nonlocal compared
        if isinstance(a, dict):
            if not isinstance(b, dict):
                mismatches.append(path)
                return
            for k, v in a.items():
                if path == "$" and k in VOLATILE:
                    continue
                walk(v, b.get(k, _MISSING), f"{path}.{k}")
        elif isinstance(a, list):
            # "notes" may gain appended text (R30 added one); every registered note must stay as it was
            extendable = path.endswith(".notes")
            if not isinstance(b, list) or (len(b) < len(a) if extendable else len(a) != len(b)):
                mismatches.append(path)
                return
            for i, (x, y) in enumerate(zip(a, b[: len(a)], strict=True)):
                walk(x, y, f"{path}[{i}]")
        else:
            compared += 1
            if a != b:
                mismatches.append(path)

    walk(registered, fresh, "$")
    return {"leaves_compared": compared, "mismatches": mismatches, "exact": not mismatches and compared > 0}


_MISSING = object()


def analyse(fresh: dict) -> dict:
    runs = fresh["runs"]
    cells = sum(len(r["parameters"]) for r in runs)
    z_fw = NormalDist().inv_cdf(1 - 0.025 / cells)
    rows, summary = [], []
    for r in runs:
        m, rep = r["converged"], r["replications"]
        n_out_ci, cov_screen, cov_fw = 0, 0, 0
        for p in r["parameters"]:
            crlb, rse, se = p["crlb_rse"], p["empirical_rse"], p["empirical_rse_mcse"]
            lo, hi = (rse - Z95 * se) / crlb, (rse + Z95 * se) / crlb
            hits = round(p["coverage"] * m)
            band = z_fw * sqrt(0.95 * 0.05 / m)
            screen_flag = not 0.93 <= p["coverage"] <= 0.97
            fw_flag = abs(p["coverage"] - 0.95) > band
            n_out_ci += not lo <= 1 <= hi
            cov_screen += screen_flag
            cov_fw += fw_flag
            rows.append(
                {
                    "sessions": r["sessions"],
                    "parameter": p["name"],
                    "crlb_rse": crlb,
                    "sessions_scope": p["sessions_scope"],
                    "empirical_rse": rse,
                    "empirical_rse_mcse": se,
                    "rse_ratio": p["rse_ratio"],
                    "rse_ratio_ci95": [lo, hi],
                    "rse_ratio_ci_contains_1": lo <= 1 <= hi,
                    "relative_bias": p["relative_bias"],
                    "relative_bias_ci95": [
                        p["relative_bias"] - Z95 * p["relative_bias_mcse"],
                        p["relative_bias"] + Z95 * p["relative_bias_mcse"],
                    ],
                    "coverage": p["coverage"],
                    "coverage_wilson95": list(wilson(hits, m)),
                    "coverage_all_low": p["coverage_all_low"],
                    "coverage_all_high": p["coverage_all_high"],
                    "coverage_flag_registered_screen": screen_flag,
                    "coverage_flag_familywise": fw_flag,
                }
            )
        summary.append(
            {
                "sessions": r["sessions"],
                "converged": m,
                "replications": rep,
                "rse_ratio_ci_excludes_1": n_out_ci,
                "coverage_flags_registered_screen": cov_screen,
                "coverage_flags_familywise": cov_fw,
            }
        )
    return {"cells": cells, "familywise_z": z_fw, "per_n": summary, "rows": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fresh", required=True)
    ap.add_argument("--registered", default=REGISTERED)
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)
    with open(args.registered, encoding="utf-8") as fh:
        registered = json.load(fh)
    with open(args.fresh, encoding="utf-8") as fh:
        fresh = json.load(fh)
    rep = reproduction(registered, fresh)
    out = {
        "schema": "ail-estimator-posthoc-v1",
        "post_hoc": True,
        "confirmatory": False,
        "registered_results": args.registered,
        "fresh_run": {"python": fresh.get("python"), "provenance": fresh.get("provenance")},
        "reproduction": rep,
        "notes": [
            "post hoc, not confirmatory: the registered endpoints (P1, S1-S3) are those of docs/ESTIMATOR_PROTOCOL.md",
            "no equivalence margin was registered: rse_ratio_ci95 is descriptive, not an equivalence test",
            "coverage is over converged fits; coverage_all_low/high bound it over all replications",
            "familywise band: Bonferroni over all parameter x N cells, two-sided 5 %",
            "crlb_rse bounds unbiased regular estimators only (sessions_scope); model-internal, one point, one design",
        ],
        "analysis": analyse(fresh) if rep["exact"] else None,
    }
    text = json.dumps(out, indent=1)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
    return 0 if rep["exact"] else 1


if __name__ == "__main__":
    sys.exit(main())

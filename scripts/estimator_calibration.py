"""Run the pre-registered MLE calibration of docs/ESTIMATOR_PROTOCOL.md and write JSON.

Usage: python scripts/estimator_calibration.py [--replications R] [--out PATH]

The defaults are the protocol's (R = 1000); a smaller R is for smoke runs only
and is recorded in the output.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
import time
from math import ceil
from pathlib import Path

from atomic_information_lab.estimation import estimator_calibration
from atomic_information_lab.identifiability import Calibration, Design, parameter_names

CRLB_COUNT = 373_950
MULTIPLIERS = (1 / 16, 1 / 4, 1, 4)
BASE_SEED = 20261104
REPLICATIONS = 1000  # protocol R; a smaller --replications is a smoke run and is recorded as such
DESIGN = ("fast", "reference", "precise", "reference", "fast", "precise")
ROOT = Path(__file__).resolve().parents[1]
# Files whose bytes determine the numbers; their SHA-256 is written into every result (R30, AIL-03).
BOUND_FILES = (
    "docs/ESTIMATOR_PROTOCOL.md",
    "scripts/estimator_calibration.py",
    "src/atomic_information_lab/estimation.py",
    "src/atomic_information_lab/identifiability.py",
)


def provenance() -> dict:
    """Commit, dirty flag and file digests of the code that produced a result.

    Added after the registered run (R30): that run's JSON carries no such binding, which is why
    its link to the protocol commit had to be shown separately (scripts/verify_preregistration.py).
    """

    def git(*args):
        try:
            r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return None
        return r.stdout.strip() if r.returncode == 0 else None

    status = git("status", "--porcelain", "--", *BOUND_FILES)
    return {
        "git_head": git("rev-parse", "HEAD"),
        "bound_files_dirty": None if status is None else bool(status),
        "sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in BOUND_FILES},
    }


def endpoints(runs: list[dict]) -> dict:
    by_n = {r["sessions"]: r for r in runs}
    primary = None
    if CRLB_COUNT in by_n:
        pe = next(p for p in by_n[CRLB_COUNT]["parameters"] if p["name"] == "prep_error")
        primary = {"prep_error_empirical_rse": pe["empirical_rse"], "P1_pass": pe["empirical_rse"] <= 0.11}
    flags = []
    for r in runs:
        for p in r["parameters"]:
            if abs(p["relative_bias"]) > 3 * p["relative_bias_mcse"] and abs(p["relative_bias"]) > 0.02:
                flags.append({"sessions": r["sessions"], "parameter": p["name"], "S2_bias": p["relative_bias"]})
            if not 0.93 <= p["coverage"] <= 0.97:
                flags.append({"sessions": r["sessions"], "parameter": p["name"], "S3_coverage": p["coverage"]})
    return {"primary": primary, "flags": flags}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replications", type=int, default=REPLICATIONS)
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)
    cal = Calibration(prep_error=0.02)
    free = tuple(n for n in parameter_names(cal) if n != "backaction:fast")
    design = Design(DESIGN, True)
    runs = []
    t0 = time.perf_counter()
    for i, m in enumerate(MULTIPLIERS):
        n = ceil(CRLB_COUNT * m)
        r = estimator_calibration(cal, design, free, n, args.replications, BASE_SEED + i)
        runs.append(dataclasses.asdict(r))
        print(f"N={n}: {r.converged}/{r.replications} converged", file=sys.stderr)
    out = {
        "protocol": "docs/ESTIMATOR_PROTOCOL.md",
        "replications": args.replications,
        "protocol_replications": REPLICATIONS,
        "python": platform.python_version(),
        "seconds": round(time.perf_counter() - t0, 1),
        "provenance": provenance(),
        "runs": runs,
        "endpoints": endpoints(runs),
    }
    text = json.dumps(out, indent=1)
    if args.out == "-":
        print(text)
    else:
        with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

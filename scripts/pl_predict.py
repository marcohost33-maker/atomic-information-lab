"""Predict the finite-N behaviour of the MLE from the expected-data profile likelihood (R33).

For every free parameter the ``Delta = 1`` profile interval [c-, c+] after N
sessions (68.27 %, Wilks) gives two predictions under a normalising-transform
model (some monotone g with g(estimate) approximately normal):

* ``predicted_rse_ratio`` = ((c+ - c-) / 2) / (Cramer-Rao standard error at N)
  -- the sampling spread relative to the local (Wald) picture;
* ``predicted_bias``      = ((c+ + c-) / 2 - truth) / truth
  -- the mean shift; for g = log, (c+ + c-)/2 - truth = truth (cosh s - 1) ~ truth s^2 / 2,
  which is the mean shift of a lognormal estimate.

Both are model-internal predictions from EXPECTED data (no sampling). They are
used once in-sample (registered design, docs/ESTIMATOR_CALIBRATION_RESULTS.json)
and once out-of-sample (docs/PL_TRANSFER_PROTOCOL.md). Profiles run in
parallel processes, one per parameter.

Usage: python scripts/pl_predict.py --design fast,precise,reference,precise --known-write --sessions 417942
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atomic_information_lab.identifiability import Calibration, Design, parameter_names  # noqa: E402
from atomic_information_lab.likelihood_profile import PL_SCOPE, profile_likelihood  # noqa: E402

LEVEL_1SE = 0.6826894921370859  # chi-square(1) quantile 1.0


def _one(args):
    seq, known, name, sessions = args
    cal = Calibration(prep_error=0.02)
    free = tuple(f for f in parameter_names(cal) if f != "backaction:fast")
    r = profile_likelihood(cal, Design(seq, known), free, name, 0.1, (sessions,), LEVEL_1SE)
    iv = r.intervals[0]
    out = {
        "name": name,
        "true_value": r.true_value,
        "profile_to_crlb": r.profile_to_crlb,
        "asymmetry": r.asymmetry,
        "interval_1se": [iv.low, iv.high],
        "bounded": iv.bounded,
        "not_converged": r.not_converged,
        "predicted_rse_ratio": None,
        "predicted_bias": None,
    }
    if iv.bounded and r.crlb_rse is not None:
        se = r.crlb_rse * r.true_value / math.sqrt(sessions)
        out["predicted_rse_ratio"] = (iv.high - iv.low) / 2 / se
        out["predicted_bias"] = ((iv.high + iv.low) / 2 - r.true_value) / r.true_value
    return out


def predict(seq: tuple[str, ...], known: bool, sessions: int, workers: int = 4) -> dict:
    cal = Calibration(prep_error=0.02)
    free = tuple(f for f in parameter_names(cal) if f != "backaction:fast")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(_one, [(seq, known, n, sessions) for n in free]))
    return {
        "design": list(seq),
        "known_write": known,
        "sessions": sessions,
        "level": LEVEL_1SE,
        "parameters": rows,
        "pl_scope": PL_SCOPE,
        "seconds": round(time.time() - t0, 1),
        "notes": [
            "predictions from the expected-data profile (Delta = 1 interval) under a normalising-transform model; "
            "model-internal, one parameter point, one design"
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--design", required=True)
    ap.add_argument("--known-write", action="store_true")
    ap.add_argument("--sessions", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = predict(tuple(s.strip() for s in a.design.split(",")), a.known_write, a.sessions, a.workers)
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

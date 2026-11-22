"""Multi-start benchmark against the profile-likelihood table (R33): does any random start find a lower profile?

The profile value at a point is a MINIMUM over the nuisance parameters; the
table's inner optimisation is local (continuation in both directions plus a
start at the truth). This script tries to falsify it: for every parameter of
docs/PROFILE_LIKELIHOOD.json it takes the points at -r and +r and the
outermost point of each side, and runs the inner optimisation from K random
starts (each nuisance parameter = its true value * exp(sigma * N(0, 1)), cut to
its domain). A start that ends LOWER than the table by more than 1e-6
relative refutes the table value at that point.

Usage: python scripts/pl_multistart.py [--starts K] [--sigma S] [--seed SEED] [--workers W] [--out PATH]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atomic_information_lab import likelihood_profile as lp  # noqa: E402
from atomic_information_lab.identifiability import Calibration, Design, _leaves  # noqa: E402

TABLE = ROOT / "docs" / "PROFILE_LIKELIHOOD.json"
REFUTE_REL = 1e-6


def _job(args):
    name, value, table_stat, table_converged, starts, free, seq = args
    truth = Calibration(prep_error=0.02)
    design = Design(tuple(seq), True)
    nuis = tuple(f for f in free if f != name)
    target = {o: p.v for o, p in _leaves(truth, design, ())}
    best, best_ok, ran = math.inf, False, 0
    for s in starts:
        v, _, ok, _ = lp._inner(truth, design, name, value, nuis, s, target)
        ran += 1
        if 2 * v < best:
            best, best_ok = 2 * v, ok
    improvement = (table_stat - best) / table_stat if table_stat > 0 else 0.0
    return {
        "name": name,
        "value": value,
        "table_statistic": table_stat,
        "table_converged": table_converged,
        "best_random_statistic": best,
        "best_random_converged": best_ok,
        "relative_improvement": improvement,
        "refutes_table": improvement > REFUTE_REL,
        "starts": ran,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--starts", type=int, default=12)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    t = json.loads(TABLE.read_text(encoding="utf-8"))["profile_likelihood"]
    truth = Calibration(prep_error=0.02)
    tv = truth.parameters()
    free = tuple(t["free"])
    rng = random.Random(a.seed)
    jobs = []
    for row in t["parameters"]:
        name, true = row["name"], row["true_value"]
        pts = row["points"]
        below = [p for p in pts if p[0] < true]
        above = [p for p in pts if p[0] > true]
        r = row["target_rse"] * true
        chosen = [
            below[0],
            min(below, key=lambda p: abs(p[0] - (true - r))),
            min(above, key=lambda p: abs(p[0] - (true + r))),
            above[-1],
        ]
        for value, stat, conv in chosen:
            starts = []
            for _ in range(a.starts):
                s = {}
                for f in free:
                    if f == name:
                        continue
                    v = tv[f] * math.exp(a.sigma * rng.gauss(0.0, 1.0))
                    ub = lp._upper_bound(f)
                    s[f] = min(max(v, lp._LOWER), ub) if ub is not None else max(v, lp._LOWER)
                starts.append(s)
            jobs.append((name, value, stat, conv, starts, free, t["design"]))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(_job, jobs))
    conv_rows = [r for r in rows if r["table_converged"]]
    out = {
        "table": "docs/PROFILE_LIKELIHOOD.json",
        "starts_per_point": a.starts,
        "sigma": a.sigma,
        "seed": a.seed,
        "points": len(rows),
        "inner_runs": sum(r["starts"] for r in rows),
        "refuted_points": sum(r["refutes_table"] for r in rows),
        "refuted_converged_points": sum(r["refutes_table"] for r in conv_rows),
        "max_relative_improvement": max(r["relative_improvement"] for r in rows),
        "refute_threshold": REFUTE_REL,
        "seconds": round(time.time() - t0, 1),
        "rows": rows,
    }
    text = json.dumps(out, indent=2)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

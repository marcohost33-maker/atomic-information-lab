"""How conservative is the Chernoff count of the README (R12 pair, four precise reads, 68 sessions)?

Exact enumeration is out of reach there (8 likelihood-ratio classes, ~2e9 count
vectors at N = 68), so this estimates the equal-prior Bayes error of the
likelihood-ratio test by Monte Carlo with a fixed seed:
P_e(N) = 0.5 P_a(LLR < 0) + 0.5 P_b(LLR > 0), ties counted 1/2.
The exact small-case check is tests/test_r29_estimator_scope.py.

Usage: python scripts/chernoff_check.py [--replications R]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from math import exp, log, sqrt

from atomic_information_lab.estimation import sample_counts
from atomic_information_lab.identifiability import Calibration, Design, discrimination, outcome_distribution
from atomic_information_lab.model import TWO_HYPOTHESES

SEED = 20261104
GRID = (20, 30, 40, 50, 60, 68)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replications", type=int, default=20_000)
    args = ap.parse_args(argv)
    a, b = Calibration(TWO_HYPOTHESES[0]), Calibration(TWO_HYPOTHESES[1])
    d = Design(("precise",) * 4)
    disc = discrimination(a, b, d, alpha=0.01)
    p, q = outcome_distribution(a, d), outcome_distribution(b, d)
    llr = {o: log(p[o] / q[o]) for o in p}
    rng = random.Random(SEED)
    rows = []
    for n in GRID:
        err = 0.0
        for cal, wrong in ((a, lambda s: s < 0), (b, lambda s: s > 0)):
            miss = 0.0
            for _ in range(args.replications):
                s = sum(k * llr[o] for o, k in sample_counts(cal, d, n, rng).items())
                miss += 1.0 if wrong(s) else (0.5 if s == 0 else 0.0)
            err += 0.5 * miss / args.replications
        mcse = sqrt(err * (1 - err) / (2 * args.replications))
        rows.append(
            {"sessions": n, "bayes_error_mc": err, "mcse": mcse, "chernoff_bound": 0.5 * exp(-n * disc.chernoff)}
        )
    out = {
        "pair": "TWO_HYPOTHESES (R12)",
        "design": list(d.sequence),
        "alpha": 0.01,
        "chernoff": disc.chernoff,
        "sessions_for_error": disc.sessions_for_error,
        "sessions_scope": disc.sessions_scope,
        "seed": SEED,
        "replications_per_hypothesis": args.replications,
        "rows": rows,
    }
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

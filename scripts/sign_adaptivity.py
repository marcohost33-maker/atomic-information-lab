"""Where adaptivity returns in the linear-Gaussian line: a 0-1 loss for the sign of x0 (R34b, N-50).

N-47 says that adaptivity is worth exactly zero in the Gaussian line. That holds because the
terminal loss there depends on the data only through the posterior covariance. This script
checks the boundary of the claim with the same dynamics, readings and energy prices, but with
the terminal decision "report the sign of x0", loss 1 if wrong, plus ``lambda * E`` per reading.

Exact structure used (no approximation besides quadrature):

* the posterior of x0 after a fixed set of readings is ``N(m, v)``, ``v`` from the Kalman
  covariance (data-independent), and the Bayes risk of the sign decision is ``Phi(-|m| / sqrt v)``;
* under the prior predictive, the posterior mean is a Gaussian martingale: ``m_1 ~ N(0, v0 - v1)``,
  ``m_2 | m_1 ~ N(m_1, v1 - v2)``;
* therefore a fixed plan with final variance ``v`` has risk ``1/2 - arctan(sqrt((v0 - v) / v)) / pi``
  (closed form), and the Bayes-optimal adaptive policy for horizon 2 has value
  ``min(1/2, min_a [lambda E_a + E_{m_1} min(Phi(-|m_1|/sqrt v_a), min_b lambda E_b + g_ab(m_1))])``,
  ``g_ab(m_1) = E[Phi(-|m_2| / sqrt v_ab) | m_1]``.

Integrals are split at the kinks of ``|.|`` and computed by composite Simpson rules; the
quadrature error is estimated by halving the step. A Monte-Carlo simulation of the verbal
model checks the adaptive value independently of the martingale argument.

Exploratory (seen in a prototype before this script was written); not pre-registered.
Writes docs/SIGN_ADAPTIVITY.json (or stdout with --stdout).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from itertools import product
from math import atan, erf, exp, pi, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from atomic_information_lab.gaussian import (  # noqa: E402
    DEFAULT_GAUSSIAN_MODEL,
    GaussianObjective,
    initial_state,
    plan_value,
    update,
)

SCHEMA = "ail-sign-adaptivity-v1"
LAMBDAS = (0.004, 0.012, 0.036)
SPAN = 10.0  # integrate the outer Gaussian over +-SPAN standard deviations


def _phi_cdf(z: float) -> float:
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def _pdf(x: float, var: float) -> float:
    return exp(-x * x / (2.0 * var)) / sqrt(2.0 * pi * var)


def _simpson(f, a: float, b: float, n: int) -> float:
    if b <= a:
        return 0.0
    n += n % 2
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3.0


def open_loop_risk(v0: float, v: float) -> float:
    """Expected 0-1 risk of the sign decision after a fixed plan with final variance ``v``."""
    if v >= v0:
        return 0.5
    return 0.5 - atan(sqrt((v0 - v) / v)) / pi


def _g(m1: float, v_ab: float, tau: float, n: int) -> float:
    """E[Phi(-|m2| / sqrt v_ab)] for m2 ~ N(m1, tau); split at the kink m2 = 0."""
    if tau <= 0:
        return _phi_cdf(-abs(m1) / sqrt(v_ab))
    sd = sqrt(tau)
    lo, hi = m1 - SPAN * sd, m1 + SPAN * sd

    def f(z):
        return _phi_cdf(-abs(z) / sqrt(v_ab)) * _pdf(z - m1, tau)

    if lo < 0 < hi:
        return _simpson(f, lo, 0.0, n) + _simpson(f, 0.0, hi, n)
    return _simpson(f, lo, hi, 2 * n)


def adaptive_value(model, lam: float, n_outer: int, n_inner: int) -> dict:
    """Bayes-optimal adaptive value for horizon 2 and the decision of each first action."""
    v0 = model.prior_var
    names = [a.name for a in model.actions]
    best = {"value": 0.5, "first": None}
    per_first = {}
    for a in names:
        v1 = plan_value(model, (a,)).var_initial
        s1 = v0 - v1
        if s1 <= 0:
            continue
        conts = []
        for b in names:
            vab = plan_value(model, (a, b)).var_initial
            conts.append((b, vab, v1 - vab, lam * model.action(b).energy))

        def inner(m1, conts=conts, v1=v1):
            stop = _phi_cdf(-abs(m1) / sqrt(v1))
            return min([stop] + [cost + _g(m1, vab, tau, n_inner) for _b, vab, tau, cost in conts])

        sd1 = sqrt(s1)
        # integrand symmetric in m1: integrate over [0, SPAN sd] and double
        val = lam * model.action(a).energy + 2.0 * _simpson(
            lambda x, s1=s1: inner(x) * _pdf(x, s1), 0.0, SPAN * sd1, n_outer
        )
        per_first[a] = val
        if val < best["value"]:
            best = {"value": val, "first": a}
    return {**best, "per_first_action": per_first}


def open_loop_best(model, lam: float, horizon: int = 2) -> dict:
    v0 = model.prior_var
    best = {"value": 0.5, "plan": []}
    for k in range(1, horizon + 1):
        for seq in product([a.name for a in model.actions], repeat=k):
            p = plan_value(model, seq)
            val = lam * p.energy + open_loop_risk(v0, p.var_initial)
            if val < best["value"]:
                best = {"value": val, "plan": list(seq)}
    return best


def _decision(model, lam, a, m1, n_inner):
    v1 = plan_value(model, (a,)).var_initial
    opts = [(_phi_cdf(-abs(m1) / sqrt(v1)), None)]
    for b in (x.name for x in model.actions):
        vab = plan_value(model, (a, b)).var_initial
        opts.append((lam * model.action(b).energy + _g(m1, vab, v1 - vab, n_inner), b))
    return min(opts, key=lambda t: t[0])[1]


def monte_carlo(model, lam: float, first: str, runs: int, seed: int, n_inner: int = 60) -> dict:
    """Simulate the verbal model and run the adaptive policy with the Kalman filter.

    Decisions at m1 come from a lookup over |m1| on a grid of 0.01 posterior sd (the policy is
    symmetric in m1); the reported value is the empirical mean of energy cost + 0-1 loss.
    """
    rng = random.Random(seed)
    v0 = model.prior_var
    v1 = plan_value(model, (first,)).var_initial
    sd1 = sqrt(v0 - v1)
    grid = {k: _decision(model, lam, first, k * 0.01 * sd1, n_inner) for k in range(0, 1001)}
    total = total_sq = 0.0
    for _ in range(runs):
        x0 = rng.gauss(0, sqrt(model.prior_var))
        x, b = x0, rng.gauss(0, sqrt(model.bias_prior_var))
        st = initial_state(model)
        cost = 0.0
        plan = [first]
        k = 0
        while k < len(plan):
            act = model.action(plan[k])
            x += rng.gauss(0, sqrt(model.drift_var * act.backaction))
            b += rng.gauss(0, sqrt(model.bias_drift_var))
            y = (b if act.reads_reference else x + b) + rng.gauss(0, sqrt(act.noise_var))
            st = update(model, st, act, y)
            cost += lam * act.energy
            if k == 0:
                nxt = grid[min(1000, round(abs(st.mean[0]) / (0.01 * sd1)))]
                if nxt is not None:
                    plan.append(nxt)
            k += 1
        wrong = (st.mean[0] >= 0) != (x0 >= 0)
        loss = cost + (1.0 if wrong else 0.0)
        total += loss
        total_sq += loss * loss
    mean = total / runs
    se = sqrt(max(total_sq / runs - mean * mean, 0.0) / runs)
    return {"runs": runs, "seed": seed, "value": mean, "standard_error": se}


def run(n_outer: int = 800, n_inner: int = 200, mc_runs: int = 40000) -> dict:
    rows = []
    for lam in LAMBDAS:
        model = replace(DEFAULT_GAUSSIAN_MODEL, objective=GaussianObjective(energy_lambda=lam))
        fine = adaptive_value(model, lam, n_outer, n_inner)
        coarse = adaptive_value(model, lam, n_outer // 2, n_inner // 2)
        ol = open_loop_best(model, lam)
        mc = monte_carlo(model, lam, fine["first"], mc_runs, seed=50) if fine["first"] else None
        rows.append(
            {
                "energy_lambda": lam,
                "open_loop_best": ol,
                "adaptive": fine,
                "quadrature_error_estimate": abs(fine["value"] - coarse["value"]),
                "relative_gain": (ol["value"] - fine["value"]) / ol["value"],
                "monte_carlo_adaptive": mc,
            }
        )
    return {
        "schema": SCHEMA,
        "status": "exploratory, not pre-registered",
        "loss": "0-1 loss for the sign of x0 plus energy_lambda * energy; horizon 2",
        "quadrature": {"n_outer": n_outer, "n_inner": n_inner, "span_sd": SPAN},
        "rows": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stdout", action="store_true")
    args = ap.parse_args()
    out = json.dumps(run(), indent=2, sort_keys=True) + "\n"
    if args.stdout:
        sys.stdout.write(out)
    else:
        (ROOT / "docs" / "SIGN_ADAPTIVITY.json").write_text(out, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

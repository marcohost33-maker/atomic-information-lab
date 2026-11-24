"""Independent exact oracle for the linear-Gaussian line (shares no code with ``src/``).

Different algorithm from the package: no Kalman recursion. The whole history is written as
a linear function of independent Gaussian variables

    z = (x0, b0, w_1..w_T, u_1..u_T),   x_t = x0 + w_1 + .. + w_t,   b_t = b0 + u_1 + .. + u_t,

and the posterior is computed in one batch in information form,
``Lambda = diag(1/var) + sum_t h_t h_t^T / r_t``, ``C = Lambda^-1``, ``m = C sum_t h_t y_t / r_t``,
by Gauss-Jordan elimination in exact rational arithmetic (``fractions.Fraction``). Floats given
as inputs are converted exactly (``Fraction(0.1)`` is the binary value of the float).
Variables with zero prior variance are fixed at zero and dropped.
"""

from __future__ import annotations

from fractions import Fraction as F
from itertools import product


def _inv(m):
    n = len(m)
    a = [list(row) + [F(int(i == j)) for j in range(n)] for i, row in enumerate(m)]
    for c in range(n):
        piv = next(r for r in range(c, n) if a[r][c] != 0)
        a[c], a[piv] = a[piv], a[c]
        p = a[c][c]
        a[c] = [v / p for v in a[c]]
        for r in range(n):
            if r != c and a[r][c] != 0:
                f = a[r][c]
                a[r] = [vr - f * vc for vr, vc in zip(a[r], a[c], strict=True)]
    return [row[n:] for row in a]


def posterior(params, sequence, ys=None):
    """Return dict with exact Var(x0|D), Var(x_T|D), damage, energy and (if ``ys``) the means.

    ``params``: dict with prior_var, bias_prior_var, drift_var, bias_drift_var and
    ``actions`` = {name: (backaction, noise_var, energy, reads_reference)}.
    """
    T = len(sequence)
    acts = [params["actions"][n] for n in sequence]
    names = ["x0", "b0"] + [f"w{t}" for t in range(1, T + 1)] + [f"u{t}" for t in range(1, T + 1)]
    var = {"x0": F(params["prior_var"]), "b0": F(params["bias_prior_var"])}
    for t, (beta, _r, _e, _ref) in enumerate(acts, start=1):
        var[f"w{t}"] = F(params["drift_var"]) * F(beta)
        var[f"u{t}"] = F(params["bias_drift_var"])
    live = [n for n in names if var[n] != 0]
    idx = {n: i for i, n in enumerate(live)}
    k = len(live)

    def vec(coeffs):
        v = [F(0)] * k
        for n in coeffs:
            if n in idx:
                v[idx[n]] += 1
        return v

    lam = [[F(0)] * k for _ in range(k)]
    for n in live:
        lam[idx[n]][idx[n]] = 1 / var[n]
    info = [F(0)] * k
    for t, (_beta, r, _e, ref) in enumerate(acts, start=1):
        terms = ["b0"] + [f"u{s}" for s in range(1, t + 1)]
        if not ref:
            terms += ["x0"] + [f"w{s}" for s in range(1, t + 1)]
        h = vec(terms)
        r = F(r)
        for i in range(k):
            if h[i]:
                for j in range(k):
                    if h[j]:
                        lam[i][j] += h[i] * h[j] / r
        if ys is not None:
            y = F(ys[t - 1])
            for i in range(k):
                info[i] += h[i] * y / r
    cov = _inv(lam) if k else []
    g0 = vec(["x0"])
    gT = vec(["x0"] + [f"w{s}" for s in range(1, T + 1)])
    gb = vec(["b0"] + [f"u{s}" for s in range(1, T + 1)])

    def quad(g):
        return sum(g[i] * cov[i][j] * g[j] for i in range(k) for j in range(k))

    out = {
        "var_initial": quad(g0),
        "var_current": quad(gT),
        "damage": sum(F(params["drift_var"]) * F(a[0]) for a in acts),
        "energy": sum(F(a[2]) for a in acts),
    }
    if ys is not None:
        mean = [sum(cov[i][j] * info[j] for j in range(k)) for i in range(k)]
        out["mean"] = tuple(sum(g[i] * mean[i] for i in range(k)) for g in (g0, gT, gb))
    return out


def plan_value(params, objective, sequence):
    alpha, lam, beta = (F(v) for v in objective)
    p = posterior(params, sequence)
    return lam * p["energy"] + alpha * p["var_initial"] + (1 - alpha) * p["var_current"] + beta * p["damage"]


def optimum(params, objective, horizon, limit=None):
    """Brute force over every plan of length 0..horizon (an adaptive policy cannot do better:
    its value is a mixture of plan values, because no posterior variance depends on the data)."""
    best = None
    for n in range(horizon + 1):
        for seq in product(sorted(params["actions"]), repeat=n):
            if limit is not None and posterior(params, seq)["damage"] > F(limit):
                continue
            v = plan_value(params, objective, seq)
            if best is None or v < best[0]:
                best = (v, seq)
    return best

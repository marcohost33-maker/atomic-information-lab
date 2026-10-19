"""Independent test oracle for the discrete two-state measurement model.

Shares NO code with ``atomic_information_lab`` (nor with the historic R10-R12
sources); parameters are transcribed from the model specification. Two methods:
  A) exact rational DP over *unnormalised* forward messages per observation
     history (fractions.Fraction); CTMC step via uniformisation series.
  B) brute force over ALL deterministic policies (small h) with explicit
     hidden-path enumeration (floats) -- min over enumerated policies, no DP.
"""

from __future__ import annotations

import math
from fractions import Fraction as Fr
from itertools import product

# (name, backaction, error, energy, is_reference) -- values transcribed from the R10 spec text
DEFAULT_ACTIONS = (
    ("fast", 1.0, 0.12, 1.0, False),
    ("precise", 3.5, 0.035, 2.8, False),
    ("reference", 0.55, 0.025, 0.8, True),
)


def ctmc_uniformised(k01: float, k10: float, t: float = 1.0) -> tuple[float, float]:
    """P(0->1), P(1->0) over time t by uniformisation (not the closed form)."""
    q = max(k01, k10)
    if q == 0.0:
        return 0.0, 0.0
    if not math.isfinite(q * t) or q * t > 500.0:
        # exp(-qt) would underflow and the series would silently return zeros
        raise ValueError("outside the oracle's numerical domain (q*t > 500)")
    # uniformised DTMC U = I + Q/q ; P(t) = sum_n Pois(n; q t) U^n
    u = ((1 - k01 / q, k01 / q), (k10 / q, 1 - k10 / q))
    un = ((1.0, 0.0), (0.0, 1.0))
    qt = q * t
    w = math.exp(-qt)
    acc = [[0.0, 0.0], [0.0, 0.0]]
    n = 0
    while True:
        for i in range(2):
            for j in range(2):
                acc[i][j] += w * un[i][j]
        # Poisson terms decay monotonically once n > qt; stop when negligible
        if n > qt and w < 1e-20:
            break
        n += 1
        un = tuple(tuple(sum(un[i][k] * u[k][j] for k in range(2)) for j in range(2)) for i in range(2))
        w = w * qt / n
        if n > 10_000:
            raise RuntimeError("uniformisation did not converge")
    if abs(acc[0][0] + acc[0][1] - 1.0) > 1e-12 or abs(acc[1][0] + acc[1][1] - 1.0) > 1e-12:
        raise ArithmeticError("uniformisation rows do not sum to one")
    return acc[0][1], acc[1][0]


class Model:
    """Finite mixture over static model index theta; states (theta, x0, x, b)."""

    def __init__(self, components, actions=DEFAULT_ACTIONS, lam=0.012, alpha=0.5, beta=0.0, exact=True):
        # components: sequence of (weight, k01, k10, bias_flip)
        self.N = Fr if exact else float
        N = self.N
        self.components = tuple(components)
        self.actions = tuple(actions)
        self.lam = N(lam)
        self.alpha = N(alpha)
        self.beta = N(beta)
        self.states = [
            (th, x0, x, b) for th in range(len(self.components)) for x0 in (0, 1) for x in (0, 1) for b in (0, 1)
        ]
        self.index = {s: i for i, s in enumerate(self.states)}
        self.trans = {}
        for th, (_, k01, k10, f) in enumerate(self.components):
            for a in self.actions:
                p01, p10 = ctmc_uniformised(k01 * a[1], k10 * a[1])
                self.trans[(th, a[0])] = (N(p01), N(p10), N(f))

    def root(self):
        m = []
        for th, x0, x, _b in self.states:
            w = self.N(self.components[th][0])
            m.append(w / 4 if x == x0 else self.N(0))
        return tuple(m)

    def step(self, m, a, y):
        name, _, err, _, is_ref = a
        err = self.N(err)
        out = []
        for th, x0, x2, b2 in self.states:
            p01, p10, f = self.trans[(th, name)]
            tx = {(0, 0): 1 - p01, (0, 1): p01, (1, 0): p10, (1, 1): 1 - p10}
            acc = self.N(0)
            for x in (0, 1):
                for b in (0, 1):
                    src = m[self.index[(th, x0, x, b)]]
                    if src:
                        acc += src * tx[(x, x2)] * (f if b != b2 else 1 - f)
            target = b2 if is_ref else (x2 ^ b2)
            out.append(acc * ((1 - err) if y == target else err))
        if self.N is float and not all(math.isfinite(v) and v >= 0 for v in out):
            raise ArithmeticError("non-finite forward message")
        return tuple(out)

    def stop_mass(self, m):
        """Mass-weighted terminal loss = M * L(m/M)."""
        tot = sum(m)
        if tot == 0:
            return self.N(0)
        m0 = sum(v for v, s in zip(m, self.states, strict=True) if s[1] == 1)
        mt = sum(v for v, s in zip(m, self.states, strict=True) if s[2] == 1)
        mis = sum(v for v, s in zip(m, self.states, strict=True) if s[1] != s[2])
        return self.alpha * m0 * (tot - m0) / tot + (1 - self.alpha) * mt * (tot - mt) / tot + self.beta * mis

    def cost(self, a):
        return self.lam * self.N(a[3])

    def value(self, m, h, allow_stop=True):
        """Optimal mass-weighted value; allow_stop=False => exactly h measurements."""
        tot = sum(m)
        if tot == 0:
            return self.N(0)
        stop = self.stop_mass(m)
        if h == 0:
            return stop
        best = stop if allow_stop else None
        for a in self.actions:
            v = self.cost(a) * tot + sum(self.value(self.step(m, a, y), h - 1, allow_stop) for y in (0, 1))
            if best is None or v < best:
                best = v
        return best

    def policy_value(self, m, h, policy, hist=()):
        """Value + expected steps of an arbitrary policy(hist, h) -> action or None (stop)."""
        tot = sum(m)
        if tot == 0:
            return self.N(0), self.N(0)
        a = policy(m, hist, h) if h > 0 else None
        if a is None:
            return self.stop_mass(m), self.N(0)
        v = self.cost(a) * tot
        steps = tot
        for y in (0, 1):
            cv, cs = self.policy_value(self.step(m, a, y), h - 1, policy, hist + ((a[0], y),))
            v += cv
            steps += cs
        return v, steps

    def optimal_policy(self, tie_order=("stop", "fast", "precise", "reference")):
        """Exact Bellman policy; ties broken by tie_order (STOP first)."""

        def pol(m, hist, h):
            tot = sum(m)
            opts = {"stop": self.stop_mass(m)}
            for a in self.actions:
                opts[a[0]] = self.cost(a) * tot + sum(self.value(self.step(m, a, y), h - 1) for y in (0, 1))
            bestv = min(opts.values())
            for name in tie_order:
                if name in opts and opts[name] == bestv:
                    return None if name == "stop" else next(a for a in self.actions if a[0] == name)
            raise AssertionError

        return pol

    def greedy_policy(self):
        """Myopic: compare STOP with one-step lookahead; ties -> STOP, then name order."""

        def pol(m, hist, h):
            tot = sum(m)
            best_name, best_v = "stop", self.stop_mass(m)
            for a in sorted(self.actions, key=lambda a: a[0]):
                v = self.cost(a) * tot + sum(self.stop_mass(self.step(m, a, y)) for y in (0, 1))
                if v < best_v:
                    best_name, best_v = a[0], v
            return None if best_name == "stop" else next(a for a in self.actions if a[0] == best_name)

        return pol

    def open_loop(self, seq):
        """Risk of fixed action sequence then STOP."""

        def rec(m, rest):
            if sum(m) == 0:
                return self.N(0)
            if not rest:
                return self.stop_mass(m)
            a = rest[0]
            return self.cost(a) * sum(m) + sum(rec(self.step(m, a, y), rest[1:]) for y in (0, 1))

        return rec(self.root(), tuple(seq))

    def best_open_loop(self, h):
        best = (self.stop_mass(self.root()), ())
        for n in range(1, h + 1):
            for seq in product(self.actions, repeat=n):
                v = self.open_loop(seq)
                if v < best[0]:
                    best = (v, tuple(a[0] for a in seq))
        return best


# ---------------------------------------------------------------- method B
def hidden_path_table(components, actions, aseq):
    """Explicit hidden-path enumeration (floats). Returns {yseq: [mass, m_x0=1, m_xt=1, mismatch, brier_terms]}"""
    out = {}
    trans = {}
    for th, (_, k01, k10, f) in enumerate(components):
        for a in actions:
            trans[(th, a[0])] = ctmc_uniformised(k01 * a[1], k10 * a[1]) + (f,)
    n = len(aseq)
    paths = []
    for th, (w, *_rest) in enumerate(components):
        for x0, b0 in product((0, 1), (0, 1)):
            for hid in product(product((0, 1), (0, 1)), repeat=n):
                for ys in product((0, 1), repeat=n):
                    p = w * 0.25
                    x, b = x0, b0
                    for (x2, b2), y, a in zip(hid, ys, aseq, strict=True):
                        p01, p10, f = trans[(th, a[0])]
                        px = {(0, 0): 1 - p01, (0, 1): p01, (1, 0): p10, (1, 1): 1 - p10}[(x, x2)]
                        pb = f if b != b2 else 1 - f
                        target = b2 if a[4] else (x2 ^ b2)
                        pe = (1 - a[2]) if y == target else a[2]
                        p *= px * pb * pe
                        x, b = x2, b2
                    if p > 0:
                        paths.append((ys, p, x0, x))
    for ys, p, x0, x in paths:
        e = out.setdefault(ys, [0.0, 0.0, 0.0, 0.0])
        e[0] += p
        e[1] += p * x0
        e[2] += p * x
        e[3] += p * (x0 != x)
    # expected Brier computed directly from paths with the posterior-mean forecast
    brier = {}
    for ys, p, x0, x in paths:
        tot, m0, mt, _ = out[ys]
        q0, qt = m0 / tot, mt / tot
        b = brier.setdefault(ys, [0.0, 0.0])
        b[0] += p * (q0 - x0) ** 2
        b[1] += p * (qt - x) ** 2
    return out, brier


def brute_force_all_policies(components, actions, h, lam, alpha, beta=0.0, allow_stop=True):
    """Enumerate every deterministic policy of depth <= h; return (min value, #policies)."""
    tables = {(): ({(): [1.0, 0.5, 0.5, 0.0]}, None)}
    for n in range(1, h + 1):
        for aseq in product(actions, repeat=n):
            tables[tuple(a[0] for a in aseq)] = hidden_path_table(components, actions, aseq)

    def loss(entry):
        tot, m0, mt, mis = entry
        if tot <= 0:
            return 0.0
        return alpha * m0 * (tot - m0) / tot + (1 - alpha) * mt * (tot - mt) / tot + beta * mis

    def enum(anames, ys, left):
        entry = tables[anames][0].get(ys, [0.0, 0.0, 0.0, 0.0])
        vals = []
        if allow_stop or left == 0:
            vals.append(loss(entry))
        if left == 0:
            return vals
        for a in actions:
            c = lam * a[3] * entry[0]
            v0 = enum(anames + (a[0],), ys + (0,), left - 1)
            v1 = enum(anames + (a[0],), ys + (1,), left - 1)
            vals.extend(c + u + w for u in v0 for w in v1)
        return vals

    vals = enum((), (), h)
    return min(vals), len(vals)

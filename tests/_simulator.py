"""Monte-Carlo simulation of the generative story, written from the verbal model only.

Third, structurally different derivation next to the package (closed-form
two-state kernel, normalised belief vectors) and the oracle (uniformisation,
unnormalised rational messages): here nothing is a matrix. Each run samples
the hidden quantities and lets them evolve as the README describes them:

1. ``theta`` from the prior weights, ``x0`` and ``b`` uniform, ``x = x0``;
2. per reading: ``x`` jumps with exponential holding times at rate
   ``k01 * backaction`` (from 0) or ``k10 * backaction`` (from 1) for one time
   unit (Gillespie), then ``b`` flips with probability ``bias_flip``, then
   the sensor reports ``target XOR noise`` with ``target = x XOR b`` (target
   read) or ``b`` (reference read) and ``noise ~ Bernoulli(error)``.

It shares no code with the package or the oracle and imports only the
standard library. The agreement it can show is statistical, not exact.
"""

from __future__ import annotations

import math
import random
from itertools import accumulate

# (name, backaction, error, energy, is_reference) -- transcribed from the README text
DEFAULT_ACTIONS = (
    ("fast", 1.0, 0.12, 1.0, False),
    ("precise", 3.5, 0.035, 2.8, False),
    ("reference", 0.55, 0.025, 0.8, True),
)


def _evolve(rng: random.Random, x: int, up: float, down: float) -> int:
    """Two-state continuous-time jump process over one time unit (Gillespie)."""
    t = 0.0
    while True:
        rate = up if x == 0 else down
        if rate <= 0.0:
            return x
        t += rng.expovariate(rate)
        if t > 1.0:
            return x
        x ^= 1


def simulate(components, sequence, runs, seed, actions=DEFAULT_ACTIONS, *, variant=None):
    """Tally ``runs`` simulated experiments of a fixed reading ``sequence``.

    ``components``: ``(weight, k01, k10, bias_flip)`` per model hypothesis.
    Returns ``{readings: [count, sum_x0, sum_x, sum_changed, [count per theta]]}``.
    ``variant`` deliberately breaks the story for power controls only:
    ``"swap_rates"``, ``"bias_x4"``, ``"read_before_evolve"``.
    """
    if variant not in (None, "swap_rates", "bias_x4", "read_before_evolve"):
        raise ValueError(f"unknown variant {variant!r}")
    rng = random.Random(seed)
    by_name = {a[0]: a for a in actions}
    cumulative = list(accumulate(c[0] for c in components))
    table: dict[tuple[int, ...], list] = {}
    for _ in range(runs):
        u = rng.random() * cumulative[-1]
        theta = next(i for i, c in enumerate(cumulative) if u < c or i == len(cumulative) - 1)
        _, k01, k10, flip = components[theta]
        if variant == "swap_rates":
            k01, k10 = k10, k01
        if variant == "bias_x4":
            flip = min(0.5, 4 * flip)
        x0 = rng.getrandbits(1)
        x, b = x0, rng.getrandbits(1)
        readings = []
        for name in sequence:
            _, backaction, error, _, is_reference = by_name[name]
            if variant == "read_before_evolve":
                target = b if is_reference else x ^ b
                readings.append(target ^ (rng.random() < error))
            x = _evolve(rng, x, k01 * backaction, k10 * backaction)
            if rng.random() < flip:
                b ^= 1
            if variant != "read_before_evolve":
                target = b if is_reference else x ^ b
                readings.append(target ^ (rng.random() < error))
        entry = table.setdefault(tuple(readings), [0, 0, 0, 0, [0] * len(components)])
        entry[0] += 1
        entry[1] += x0
        entry[2] += x
        entry[3] += x0 != x
        entry[4][theta] += 1
    return table


def z_scores(table, runs, predicted, min_count=200):
    """Binomial z-scores of simulated frequencies against ``predicted``.

    ``predicted[readings] = (P(readings), p_x0_one, p_x_one, p_changed, model_posterior)``.
    One score per reading pattern (its frequency) and, where it occurred at
    least ``min_count`` times, one per conditional quantity. Exact 0/1
    predictions must be met exactly (an impossible event that occurs scores inf).
    """
    scores = []

    def score(observed, total, p):
        if p in (0.0, 1.0):
            return 0.0 if observed == p * total else math.inf
        return abs(observed - total * p) / math.sqrt(total * p * (1.0 - p))

    for readings, (p, q0, qx, qc, post) in predicted.items():
        entry = table.get(readings, [0, 0, 0, 0, [0] * len(post)])
        n = entry[0]
        scores.append(score(n, runs, p))
        if n >= min_count:
            scores += [score(entry[1], n, q0), score(entry[2], n, qx), score(entry[3], n, qc)]
            if len(post) > 1:
                scores += [score(c, n, q) for c, q in zip(entry[4], post, strict=True)]
    if set(table) - set(predicted):
        scores.append(math.inf)  # a reading pattern the prediction calls impossible
    return scores

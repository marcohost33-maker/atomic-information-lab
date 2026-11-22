"""Independent oracle for calibration-session distributions (identifiability tests).

Shares NO code with ``atomic_information_lab``. Written from the verbal model:
write x0 (uniform; recorded or not), the preparation flips it with
probability eps, the sensor starts inverted with probability pb; each step
the bit moves (0->1 with p01_a, 1->0 with p10_a), the inversion flips with
probability f, and the reading is target XOR noise with noise ~ Bernoulli(e_a),
target = x XOR b for a target read and b for a reference read.

Every hidden path (x0, preparation flip, b0, then per step: new x, new b,
noise) is enumerated explicitly -- no forward recursion, no matrix algebra.
Works with floats or with ``fractions.Fraction`` (pass exact step
probabilities via ``steps``); with floats the step probabilities come from the
uniformisation series of ``_oracle.ctmc_uniformised``, not the closed form.
"""

from __future__ import annotations

from itertools import product

from _oracle import ctmc_uniformised


def step_probabilities(spec, name):
    beta = spec["actions"][name][0]
    return ctmc_uniformised(spec["k01"] * beta, spec["k10"] * beta)


def session_distribution(spec, sequence, known_write, steps=None):
    """dict outcome -> probability; outcome = (x0, y1..yn) if known_write else (y1..yn).

    spec: k01, k10, f, eps, pb, actions = {name: (backaction, error, is_reference)}.
    steps: optional {name: (p01, p10)} overriding the CTMC (exact rational tests).
    """
    one = spec["f"] * 0 + 1  # 1 of the same number type as the spec
    half = one / 2
    if steps is None:
        steps = {a: step_probabilities(spec, a) for a in set(sequence)}
    f, eps, pb = spec["f"], spec["eps"], spec["pb"]
    n = len(sequence)
    out = {}
    for x0, flip, b0 in product((0, 1), repeat=3):
        w0 = half * (eps if flip else one - eps) * (pb if b0 else one - pb)
        for path in product(product((0, 1), repeat=3), repeat=n):  # per step: (x, b, noise)
            w, x, b, ys = w0, x0 ^ flip, b0, []
            for name, (x2, b2, noise) in zip(sequence, path, strict=True):
                p01, p10 = steps[name]
                _, err, is_ref = spec["actions"][name]
                if x == 0:
                    w *= p01 if x2 else one - p01
                else:
                    w *= one - p10 if x2 else p10
                w *= f if b2 != b else one - f
                w *= err if noise else one - err
                x, b = x2, b2
                ys.append(((0 if is_ref else x) ^ b) ^ noise)
                if w == 0:
                    break
            else:
                key = ((x0,) if known_write else ()) + tuple(ys)
                out[key] = out.get(key, 0 * one) + w
    return out


def spec_of(calibration):
    """Transcribe a package Calibration into a plain oracle spec (reads public attributes only)."""
    h = calibration.hypothesis
    return {
        "k01": h.k01,
        "k10": h.k10,
        "f": h.bias_flip,
        "eps": calibration.prep_error,
        "pb": calibration.bias_prior,
        "actions": {a.name: (a.backaction, a.error, a.reads_reference) for a in calibration.actions},
    }

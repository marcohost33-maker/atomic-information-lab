"""Input contract of the probability-mass tolerance (1e-12), stated and attacked.

Contract (``Problem.validate_belief``, ``Problem(weights=...)``): finite,
non-negative reals, float sum within 1e-12 of one; anything else is REJECTED,
never renormalised. The question from the cross-family review was whether
1e-12 is still satisfiable by real inputs. Rounding analysis (first order,
plain left-to-right summation): computing ``s = sum(v)`` costs ``(n-1)u``,
each ``v_i / s`` costs ``u``, summing the result again costs ``(n-1)u``, so a
float-normalised belief lands within ``(2n-1)u < 2 * n * 2**-53`` of one
(7.1e-15 for n = 32). The tests below try to break that bound with seeded
random and adversarial vectors and measure the margin to 1e-12, plus the
posteriors the package produces itself over a full planning tree.

CPython >= 3.12 ``sum()`` uses compensated (Neumaier) summation for exact
``float`` objects only; float subclasses such as ``numpy.float64`` take the
plain path. Margins measured with built-in floats alone would therefore be
optimistic; both paths are attacked here.
"""

import math
import random

import pytest

from atomic_information_lab import TWO_HYPOTHESES, Hypothesis, Problem, solve
from atomic_information_lab.model import _SUM_TOL

U = 2.0**-53
FOUR = Problem(
    (Hypothesis(0.07, 0.045, 0.025), Hypothesis(0.23, 0.16, 0.1), Hypothesis(0.0, 0.3, 0.2), Hypothesis(1.0, 2.0, 0.5)),
    (0.1, 0.2, 0.3, 0.4),
)
PROBLEMS = {8: Problem(), 16: Problem(TWO_HYPOTHESES, (0.5, 0.5)), 32: FOUR}


class Scalar(float):
    """Stands in for numpy.float64: a float subclass, summed without compensation."""


def naive_sum(values):
    total = 0.0
    for x in values:
        total += x
    return total


def normalised(v):
    s = sum(v)
    return tuple(x / s for x in v)


def normalised_naively(v):
    """How a caller normalises with a plain loop or with NumPy scalars."""
    s = naive_sum(v)
    return tuple(Scalar(x / s) for x in v)


def test_builtin_sum_is_compensated_only_for_exact_floats():
    """Documents the CPython behaviour the margins depend on (requires-python >= 3.12)."""
    assert sum([1.0, 1e100, 1.0, -1e100]) == 2.0
    assert sum([Scalar(1.0), Scalar(1e100), Scalar(1.0), Scalar(-1e100)]) == 0.0


def test_tolerance_is_the_documented_one():
    assert _SUM_TOL == 1e-12


@pytest.mark.parametrize("n", sorted(PROBLEMS))
def test_float_normalised_beliefs_are_always_accepted(n):
    """Falsification attempt: 20 000 seeded vectors per size, mixed magnitudes."""
    pr, rng, worst = PROBLEMS[n], random.Random(20261021 + n), 0.0
    for _ in range(20_000):
        v = [rng.random() ** rng.choice((1, 4, 12, 40)) for _ in range(n)]
        b = normalised(v)
        worst = max(worst, abs(sum(b) - 1.0))
        pr.validate_belief(b)
    assert worst <= 2 * n * U  # the rounding bound holds ...
    assert _SUM_TOL / (2 * n * U) > 100  # ... with more than two orders of magnitude to spare


@pytest.mark.parametrize("n", sorted(PROBLEMS))
def test_naively_normalised_scalar_beliefs_are_always_accepted(n):
    """Same attack without compensated summation anywhere: plain-loop normalisation,
    float-subclass entries (so validate_belief's own sum() is plain, too)."""
    pr, rng, worst = PROBLEMS[n], random.Random(20261022 + n), 0.0
    for _ in range(20_000):
        v = [rng.random() ** rng.choice((1, 4, 12, 40)) for _ in range(n)]
        b = normalised_naively(v)
        worst = max(worst, abs(naive_sum(b) - 1.0))
        pr.validate_belief(b)
    assert worst <= 2 * n * U
    assert worst > 2 * U or n == 8  # control: this path really is less accurate than the compensated one


@pytest.mark.parametrize("n", sorted(PROBLEMS))
def test_adversarial_normalised_beliefs_are_accepted(n):
    """One dominant entry plus tiny ones, thirds, and ascending/descending order."""
    pr = PROBLEMS[n]
    candidates = [
        normalised([1.0] + [1e-9] * (n - 1)),
        normalised([1e-300] * (n - 1) + [1.0]),
        normalised([1.0 / 3.0] * n),
        normalised([1.0 / (i + 1) for i in range(n)]),
        normalised([float(i + 1) for i in range(n)][::-1]),
        tuple([1.0 / 3.0] * 3 + [0.0] * (n - 3)),  # thirds without renormalising: sum is 1 - 1.1e-16
        normalised_naively([0.1] * n),
        normalised_naively([1.0] + [1e-9] * (n - 1)),
        normalised_naively([float(i + 1) for i in range(n)]),
    ]
    for b in candidates:
        assert abs(naive_sum(b) - 1.0) <= 2 * n * U
        pr.validate_belief(b)


def test_thirds_weights_are_accepted():
    pr = Problem((TWO_HYPOTHESES[0], TWO_HYPOTHESES[1], Hypothesis(0.0, 0.3, 0.2)), (1 / 3, 1 / 3, 1 / 3))
    assert abs(sum(pr.weights) - 1.0) <= 3 * U
    assert math.isfinite(solve(pr, horizon=2).value)


@pytest.mark.parametrize("excess, ok", [(5e-13, True), (9e-13, True), (2e-12, False), (1e-10, False), (1e-9, False)])
def test_boundary_just_inside_and_outside(excess, ok):
    pr = Problem()
    b = list(pr.initial_belief())
    b[0] += excess
    if ok:
        pr.validate_belief(b)
    else:
        with pytest.raises(ValueError, match="sum to one"):
            pr.validate_belief(b)


@pytest.mark.parametrize("excess, ok", [(5e-13, True), (2e-12, False), (1e-10, False)])
def test_weights_use_the_same_tolerance(excess, ok):
    w = (0.5 + excess, 0.5)
    if ok:
        Problem(TWO_HYPOTHESES, w)
    else:
        with pytest.raises(ValueError, match="sum to one"):
            Problem(TWO_HYPOTHESES, w)


def test_rejected_inputs_are_not_renormalised():
    """A belief scaled by 1 + 1e-10 is refused; the package does not quietly rescale it."""
    pr = Problem()
    b = tuple(p * (1 + 1e-10) for p in pr.initial_belief())
    with pytest.raises(ValueError):
        solve(pr, b, horizon=1)


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_non_finite_entries_are_rejected(bad):
    pr = Problem()
    b = list(pr.initial_belief())
    b[2] = bad
    with pytest.raises(ValueError, match="finite"):
        pr.validate_belief(b)


def test_bool_entry_is_rejected_even_when_the_sum_is_one():
    """``True == 1``: only the type rule, not the sum rule, can refuse this one-hot belief."""
    pr = Problem()
    one_hot = [1.0] + [0.0] * 7
    pr.validate_belief(one_hot)  # control: the float version is valid
    with pytest.raises(ValueError, match="finite"):
        pr.validate_belief([True] + [0.0] * 7)


def test_negative_zero_counts_as_zero():
    pr = Problem()
    b = pr.initial_belief()
    z = tuple(-0.0 if p == 0.0 else p for p in b)
    assert any(math.copysign(1.0, p) < 0 for p in z)
    assert pr.validate_belief(z) == b  # -0.0 == 0.0
    for h in range(4):
        assert solve(pr, z, horizon=h).value == solve(pr, b, horizon=h).value


@pytest.mark.parametrize("n, depth", [(8, 5), (16, 4), (32, 3)])
def test_package_posteriors_keep_the_margin(n, depth):
    """Every posterior and branch-mass sum the planner meets stays far inside the tolerance."""
    pr, worst, frontier = PROBLEMS[n], 0.0, [PROBLEMS[n].initial_belief()]
    for _ in range(depth):
        nxt = []
        for b in frontier:
            for a in pr.actions:
                br = pr.branches(b, a)
                worst = max(worst, abs(sum(p for _, p, _ in br) - 1.0))
                for _, _, post in br:
                    worst = max(worst, abs(sum(post) - 1.0))
                    nxt.append(post)
        frontier = nxt
    assert worst <= 2 * n * U

"""AIL-R29-01 and the R28 Q2 follow-up: what a session count may and may not claim.

* The Cramér–Rao count bounds UNBIASED regular estimators only; a biased estimator can have a
  smaller mean squared error (exact counter-example), so the scope token and text must carry the
  condition.
* Every public result type with a derived session count carries a machine-readable scope -- also
  in the Python API, not only in the CLI (``Discrimination`` had none).
* The Chernoff count of ``discrimination`` is sufficient and not necessary: checked against the
  exact Bayes error of the likelihood-ratio test, computed here by binomial enumeration without
  the module's Chernoff code.
* ``estimation``: the MLE reproduces a closed-form estimate, fails closed on bad input, reports
  failed fits instead of dropping them, and is reproducible from its seed.
"""

import dataclasses
import importlib
import math
import random
from fractions import Fraction as Fr
from math import comb

import pytest
from test_identifiability import ANCHORED, CAL, D6

from atomic_information_lab.estimation import ESTIMATOR_SCOPE, estimator_calibration, fit_mle, sample_counts
from atomic_information_lab.identifiability import (
    DISCRIMINATION_SCOPE,
    DISCRIMINATION_SCOPE_TEXT,
    SESSIONS_SCOPE,
    SESSIONS_SCOPE_TEXT,
    Calibration,
    Design,
    discrimination,
    outcome_distribution,
)
from atomic_information_lab.model import TWO_HYPOTHESES

# the package re-exports functions named like the modules, so take the modules from importlib
identifiability = importlib.import_module("atomic_information_lab.identifiability")
estimation = importlib.import_module("atomic_information_lab.estimation")


# ---------------------------------------------------------------- R29-01: the bound's condition
def test_biased_estimator_beats_the_crlb_at_its_point_exact():
    """Constant estimator theta_hat = 1/2 for a Bernoulli(p) mean, n draws, exact rationals.

    At p = 1/2 its MSE is 0 < CRLB = p(1-p)/n: the count is no lower bound for every estimator.
    Control 1: it is biased elsewhere (p = 1/4), so the CRLB's precondition fails -- no contradiction.
    Control 2: the unbiased sample mean attains the CRLB exactly.
    """
    n = 10
    for p in (Fr(1, 2), Fr(1, 4)):
        law = {k: comb(n, k) * p**k * (1 - p) ** (n - k) for k in range(n + 1)}
        assert sum(law.values()) == 1
        crlb = p * (1 - p) / n
        mse_const = sum(w * (Fr(1, 2) - p) ** 2 for w in law.values())
        bias_const = Fr(1, 2) - p
        mse_mean = sum(w * (Fr(k, n) - p) ** 2 for k, w in law.items())
        assert mse_mean == crlb  # unbiased estimator: bound attained
        if p == Fr(1, 2):
            assert mse_const == 0 < crlb and bias_const == 0  # unbiased AT this point, biased as a function
        else:
            assert bias_const != 0 and mse_const > crlb


def test_scope_token_and_text_carry_the_unbiasedness_condition():
    assert SESSIONS_SCOPE == "CRLB_UNBIASED_REGULAR_LOWER_BOUND_ONLY"
    assert "NECESSARY" not in SESSIONS_SCOPE
    text = SESSIONS_SCOPE_TEXT.lower()
    assert SESSIONS_SCOPE in SESSIONS_SCOPE_TEXT
    for needle in ("unbiased", "not sufficient", "no lower bound for biased", "this parameter point"):
        assert needle in text, needle
    assert SESSIONS_SCOPE_TEXT.isascii() and DISCRIMINATION_SCOPE_TEXT.isascii()
    rep = identifiability.identifiability(CAL, Design(D6, True), ANCHORED)
    assert rep.sessions_scope == SESSIONS_SCOPE and SESSIONS_SCOPE_TEXT in rep.notes


# ---------------------------------------------------------------- consumer inventory, Python API
COUNT_FIELDS = ("sessions_for_target", "worst_sessions", "sessions_for_error")


def _public_result_types():
    for mod in (identifiability, estimation):
        for name in mod.__all__:
            obj = getattr(mod, name)
            if isinstance(obj, type) and dataclasses.is_dataclass(obj):
                yield obj


def test_every_public_type_with_a_count_carries_a_scope():
    found = []
    for cls in _public_result_types():
        names = {f.name for f in dataclasses.fields(cls)}
        if names & set(COUNT_FIELDS):
            found.append(cls.__name__)
            assert "sessions_scope" in names, f"{cls.__name__} holds a session count without sessions_scope"
    # positive control: the inventory sees all three types that hold counts
    assert sorted(found) == ["DesignScore", "Discrimination", "IdentifiabilityReport"]


def test_discrimination_carries_the_sufficient_scope():
    a, b = Calibration(TWO_HYPOTHESES[0]), Calibration(TWO_HYPOTHESES[1])
    r = discrimination(a, b, Design(("precise",) * 4))
    assert r.sessions_for_error == 68
    assert getattr(r, "sessions_scope", None) == DISCRIMINATION_SCOPE == "CHERNOFF_SUFFICIENT_KNOWN_PAIR_ONLY"
    assert r.notes == (DISCRIMINATION_SCOPE_TEXT,)
    assert "sufficient, not necessary" in DISCRIMINATION_SCOPE_TEXT
    # positional construction of the old eight fields still works (fields appended with defaults)
    old = identifiability.Discrimination(r.design, 0.0, 0.0, 0.0, 0.5, None, 0.01, 0.0)
    assert getattr(old, "sessions_scope", None) == DISCRIMINATION_SCOPE


# ---------------------------------------------------------------- Chernoff count vs exact Bayes error
def _bayes_error(n: int, p1: float, q1: float) -> float:
    """Exact equal-prior Bayes error of n i.i.d. binary readings (sufficient statistic: number of ones)."""
    return 0.5 * sum(comb(n, k) * min(p1**k * (1 - p1) ** (n - k), q1**k * (1 - q1) ** (n - k)) for k in range(n + 1))


@pytest.mark.parametrize(("pb_b", "reported", "exact_min"), [(0.5, 47, 32), (0.3, 162, 112)])
def test_chernoff_count_is_sufficient_not_necessary_exact(pb_b, reported, exact_min):
    a = Calibration(bias_prior=0.1)
    b = a.with_values({"bias_prior": pb_b})
    d = Design(("reference",))
    f, e = a.hypothesis.bias_flip, a.action("reference").error

    def p_one(pb):  # by hand: b flips with f, the reference reading is b XOR Bernoulli(e)
        pb2 = pb * (1 - f) + (1 - pb) * f
        return pb2 * (1 - e) + (1 - pb2) * e

    p1, q1 = p_one(0.1), p_one(pb_b)
    assert outcome_distribution(a, d)[(1,)] == pytest.approx(p1, abs=1e-15)
    assert outcome_distribution(b, d)[(1,)] == pytest.approx(q1, abs=1e-15)
    r = discrimination(a, b, d, alpha=0.01)
    assert r.sessions_for_error == reported
    for n in range(1, reported + 1):  # the bound holds at every N ...
        assert _bayes_error(n, p1, q1) <= 0.5 * math.exp(-n * r.chernoff) * (1 + 1e-12)
    assert _bayes_error(reported, p1, q1) <= 0.01  # ... so the reported count is sufficient ...
    first = next(n for n in range(1, reported + 1) if _bayes_error(n, p1, q1) <= 0.01)
    assert first == exact_min < reported  # ... and not necessary
    assert _bayes_error(first - 1, p1, q1) > 0.01  # control: one session fewer misses alpha


# ---------------------------------------------------------------- estimation
def test_mle_matches_closed_form_single_parameter():
    """Reference read, bias_prior = 0.1: P(y=1) = c + (1 - 2 c) e with c = P(b=1 after the flip)."""
    cal = Calibration(bias_prior=0.1)
    d = Design(("reference",))
    f = cal.hypothesis.bias_flip
    c = 0.1 * (1 - f) + 0.9 * f
    counts = {(1,): 1460, (0,): 8540}
    fit = fit_mle(cal.with_values({"error:reference": 0.05}), d, ("error:reference",), counts)
    assert fit.converged and fit.reason == "converged"
    e_hat = (0.146 - c) / (1 - 2 * c)
    assert fit.values[0] == pytest.approx(e_hat, rel=1e-9)
    q = 0.146
    assert fit.std_errors[0] == pytest.approx(math.sqrt(q * (1 - q) / 10000) / (1 - 2 * c), rel=1e-6)


def test_sample_counts_reproducible_and_complete():
    d = Design(D6, True)
    a = sample_counts(CAL, d, 50_000, random.Random(7))
    b = sample_counts(CAL, d, 50_000, random.Random(7))
    assert a == b and sum(a.values()) == 50_000
    probs = outcome_distribution(CAL, d)
    assert set(a) <= set(probs)
    big = sample_counts(CAL, d, 10**9, random.Random(1))
    worst = max(abs(big.get(o, 0) / 10**9 - p) / math.sqrt(p * (1 - p) / 10**9) for o, p in probs.items() if p > 0)
    assert worst < 6  # every frequency within 6 binomial SDs (control on the sequential binomial scheme)


def test_estimator_calibration_reports_and_is_reproducible():
    d = Design(("reference", "fast", "reference"), True)
    cal = Calibration(prep_error=0.05, bias_prior=0.3)
    free = ("error:fast", "error:reference", "prep_error")
    r1 = estimator_calibration(cal, d, free, 20_000, 40, seed=3)
    r2 = estimator_calibration(cal, d, free, 20_000, 40, seed=3)
    assert r1 == r2
    assert r1.scope == ESTIMATOR_SCOPE and r1.converged + sum(n for _, n in r1.failures) == 40
    for p in r1.parameters:
        assert 0 <= p.coverage <= 1 and p.rse_ratio > 0 and p.crlb_rse > 0
        assert abs(p.relative_bias) < 5 * p.relative_bias_mcse + 0.05
    assert any("unbiased" in n for n in r1.notes)


def test_failed_fits_are_counted_not_dropped():
    """Tiny N: some fits must leave the interior; every replication is accounted for."""
    d = Design(("reference", "fast", "reference"), True)
    cal = Calibration(prep_error=0.05, bias_prior=0.3)
    r = estimator_calibration(cal, d, ("error:fast", "prep_error"), 30, 60, seed=11)
    assert r.converged + sum(n for _, n in r.failures) == 60
    assert r.failures, "positive control: at N = 30 some fits hit the boundary"


@pytest.mark.parametrize(
    "call",
    [
        lambda: sample_counts(CAL, Design(D6), 0, random.Random(0)),
        lambda: sample_counts(CAL, Design(D6), True, random.Random(0)),
        lambda: sample_counts(CAL, Design(D6), 10, 0),
        lambda: fit_mle(CAL, Design(D6), ("k01",), {}),
        lambda: fit_mle(CAL, Design(D6), ("k01",), {(9, 9): 3}),
        lambda: fit_mle(CAL, Design(("fast",)), ("k01",), {(0,): -1, (1,): 3}),
        lambda: fit_mle(CAL, Design(("fast",)), ("k01",), {(0,): 1.5}),
        lambda: estimator_calibration(CAL, Design(D6), ANCHORED, 100, 10, 0),  # NO-GO without known writes
        lambda: estimator_calibration(CAL, Design(D6, True), ANCHORED, 100, 1, 0),
        lambda: estimator_calibration(CAL, Design(D6, True), ANCHORED, 100, 10, -1),
        lambda: estimator_calibration(CAL, Design(D6, True), ANCHORED, 100, 10, 0, start_offset=float("nan")),
        lambda: estimator_calibration(CAL, Design(D6, True), ANCHORED, 100, 10, 0, start_offset=-1),
    ],
)
def test_estimation_fails_closed(call):
    with pytest.raises(ValueError):
        call()

"""R9 independent held-out policy-transfer checks at horizon 3, still synthetic."""

import hashlib
import json
import math
import random
from dataclasses import replace

import pytest
from _oracle import Model
from _transfer_check import oracle_reference, transfer_mismatches

from atomic_information_lab import Hypothesis, Objective, Problem, robustness
from atomic_information_lab.planner import nominal_policy
from atomic_information_lab.robustness import policy_transfer_report


def cases():
    rng = random.Random(2026102009)
    return [
        (
            round(rng.uniform(0.005, 0.16), 6),
            round(rng.uniform(0.004, 0.13), 6),
            round(rng.uniform(0.003, 0.045), 6),
            round(rng.uniform(0.004, 0.05), 6),
        )
        for _ in range(8)
    ]


# sha256 of json.dumps(cases(), separators=(",", ":")). Pins seed AND grid: the oracle
# agrees with the package for any parameters, so without this pin dropping the
# seed (random.Random()) kept every test green while the grid changed per run.
CASES_SHA256 = "7289970d00c19792ecfc8684957eddc45761f6741a173cc80c05d6cf7c9bb8c2"


def test_h3_case_grid_is_pinned():
    grid = cases()
    assert len(grid) == 8 and len(set(grid)) == 8
    assert grid[0] == (0.034986, 0.120934, 0.031815, 0.010726)
    assert hashlib.sha256(json.dumps(grid, separators=(",", ":")).encode()).hexdigest() == CASES_SHA256


def _scenario(k01, k10, bias, lam):
    nominal = Problem(hypotheses=(Hypothesis(0.07, 0.045, 0.025),), objective=Objective(energy_lambda=0.012))
    truth = Problem(hypotheses=(Hypothesis(k01, k10, bias),), objective=Objective(energy_lambda=lam))
    measured = policy_transfer_report(nominal, [truth], 3)["scenarios"][0]
    reference = oracle_reference(Model(((1.0, k01, k10, bias),), lam=lam), nominal_policy(nominal), 3)
    return measured, reference


@pytest.mark.parametrize("k01,k10,bias,lam", cases())
def test_h3_transfer_independent_oracle(k01, k10, bias, lam):
    result, reference = _scenario(k01, k10, bias, lam)
    assert transfer_mismatches(result, reference) == []
    assert result["signed_regret"] >= -result["tolerance"]


# Smallest forgery above the 3e-11 tolerance that the earlier inline control
# could not see (+1e-10 on true_optimal_value left it green).
FORGERY = 1e-10

# Where each reported field comes from inside the package. Deliberately NOT
# _transfer_check.FIELDS: if the check silently dropped a field, a control
# parametrised over FIELDS would lose that case with it and stay green.
SOURCES = {
    "nominal_policy_under_truth": ("evaluate_policy", "value"),
    "expected_steps": ("evaluate_policy", "expected_steps"),
    "true_optimal_value": ("solve", "value"),
}


def _forge(monkeypatch, field, delta):
    """Inject ``delta`` into ONE reported field inside the package, upstream of the report."""
    name, attr = SOURCES[field]
    real = getattr(robustness, name)

    def forged(*args, **kwargs):
        result = real(*args, **kwargs)
        return replace(result, **{attr: getattr(result, attr) + delta})

    monkeypatch.setattr(robustness, name, forged)


@pytest.mark.parametrize("field", SOURCES)
def test_h3_negative_control_detects_value_bias(monkeypatch, field):
    """A forged package value must be rejected by the SAME check the positive tests use."""
    params = cases()[0]
    clean, reference = _scenario(*params)
    assert transfer_mismatches(clean, reference) == []  # positive control: unforged passes

    _forge(monkeypatch, field, FORGERY)
    forged, _ = _scenario(*params)
    assert forged[field] != clean[field]  # the forgery reached the report
    assert transfer_mismatches(forged, reference) == [field]


@pytest.mark.parametrize("field", SOURCES)
def test_h3_check_accepts_deviation_inside_tolerance(monkeypatch, field):
    """Legitimate input passes: a deviation well inside 3e-11 is not a mismatch."""
    params = cases()[0]
    clean, _ = _scenario(*params)
    _forge(monkeypatch, field, 1e-12)
    measured, reference = _scenario(*params)
    assert measured[field] != clean[field]  # the deviation reached the report
    assert transfer_mismatches(measured, reference) == []


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_transfer_check_fails_closed_on_non_finite(bad):
    reference = dict.fromkeys(SOURCES, 0.25)
    for field in SOURCES:
        assert transfer_mismatches({**reference, field: bad}, reference) == [field]

import importlib.util
from pathlib import Path

import pytest

from atomic_information_lab.planner import Decision

SPEC = importlib.util.spec_from_file_location("gate", Path(__file__).parents[1] / "scripts" / "research_gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def test_reproducible_independent_oracle_positives():
    a = gate.certify(cases=5, horizon=2, seed=20261020)
    b = gate.certify(cases=5, horizon=2, seed=20261020)
    assert a == b and a["comparisons"] == 30
    # Pins the generated configs for this seed (a == b alone stays green if the
    # case generator changes), and shows the seed argument is actually used
    # (a == b also stays green if certify ignores it for a fixed internal seed).
    assert a["configs_sha256"] == "739112c8bab7deb636460c33c8c9b145e8a8a8ac8b89cafb48f7398e58332417"
    assert gate.certify(cases=5, horizon=2, seed=20261021)["configs_sha256"] != a["configs_sha256"]
    assert a["max_abs_residual"] <= a["abs_tolerance"]
    assert a["oracle_independent_of_package"]


def test_oracle_detects_mutated_value_negative_control():
    def corrupt(*a, **kw):
        return Decision(0.0, "stop")

    with pytest.raises(ArithmeticError, match="oracle disagreement"):
        gate.certify(cases=1, horizon=1, seed=20261020, solve_fn=corrupt)


@pytest.mark.parametrize(
    "params", [dict(cases=0), dict(cases=999), dict(horizon=4), dict(horizon=True), dict(seed=-1), dict(seed=True)]
)
def test_rejects_unbounded_or_invalid_campaign(params):
    with pytest.raises(ValueError):
        gate.certify(**params)

"""Independent and negative near-tie decision checks."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from near_tie_gate import run

from atomic_information_lab.planner import Decision


def test_near_tie_gate_positive():
    r = run()
    assert r["verdict"] == "pass"
    assert len(r["results"]) == 5
    assert any(x["tie_sensitive"] and x["regret"] > 0 for x in r["results"])
    assert any(x["decision"] == "reference" for x in r["results"])
    assert any(x["decision"] == "stop" for x in r["results"])


def test_near_tie_gate_negative_always_stop():
    def bad(problem, *, horizon, allow_stop):
        return Decision(problem.terminal_loss(problem.initial_belief()), "stop")

    with pytest.raises(AssertionError, match="regret|disagreement"):
        run(solve_fn=bad)

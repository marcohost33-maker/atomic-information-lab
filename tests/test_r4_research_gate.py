import importlib.util
import json
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


# ---- oracle_independent_of_package is measured, not asserted ----------------
ORACLE_SOURCE = Path(gate.oracle.__file__).read_text(encoding="utf-8")


def _oracle_copy(tmp_path, source, name):
    path = tmp_path / f"{name}.py"
    path.write_text(source, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_oracle_independence_positive_control(tmp_path):
    """An unchanged copy of the oracle passes; the flag in the result comes from that measurement."""
    assert gate.oracle_independence(gate.oracle)[0] is True
    copy = _oracle_copy(tmp_path, ORACLE_SOURCE, "oracle_copy_clean")
    assert gate.oracle_independence(copy)[0] is True
    assert gate.certify(cases=1, horizon=1, oracle_module=copy)["oracle_independent_of_package"] is True


COUPLINGS = {
    "top_import": "import atomic_information_lab\n",
    "from_import": "from atomic_information_lab.model import Problem\n",
    "nested_import": "def _late():\n    from atomic_information_lab import solve\n    return solve\n",
    "foreign_module": "import json\n",
    "dunder_import": "_pkg = __import__('atomic' + '_information_lab')\n",
    "importlib": "import importlib\n",
    "package_string": "_NAME = 'atomic_information_lab.planner'\n",
}


@pytest.mark.parametrize("coupling", COUPLINGS)
def test_gate_refuses_an_oracle_coupled_to_the_package(tmp_path, coupling):
    copy = _oracle_copy(tmp_path, ORACLE_SOURCE + "\n" + COUPLINGS[coupling], f"oracle_copy_{coupling}")
    assert gate.oracle_independence(copy)[0] is False
    with pytest.raises(RuntimeError, match="oracle not independent"):
        gate.certify(cases=1, horizon=1, oracle_module=copy)


def test_gate_refuses_package_object_injected_at_runtime(tmp_path):
    copy = _oracle_copy(tmp_path, ORACLE_SOURCE, "oracle_copy_injected")
    copy.solve = gate.solve  # source unchanged, namespace now holds package code
    with pytest.raises(RuntimeError, match="oracle not independent"):
        gate.certify(cases=1, horizon=1, oracle_module=copy)


def test_cli_reports_dependent_oracle_as_fail(monkeypatch, capsys):
    monkeypatch.setattr(gate, "oracle_independence", lambda _module: (False, "probe"))
    monkeypatch.setattr("sys.argv", ["research_gate.py", "--cases", "1", "--horizon", "1"])
    assert gate.main() == 2
    out = json.loads(capsys.readouterr().out)
    assert out["verdict"] == "fail" and "oracle not independent" in out["error"]

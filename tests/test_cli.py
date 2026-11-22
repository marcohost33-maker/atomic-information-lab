"""Command-line boundary: happy path plus negative inputs that must fail with exit code 2."""

import json
import os
import pathlib
import subprocess
import sys

import pytest

from atomic_information_lab.__main__ import main

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"


def test_solve_happy_path(capsys):
    assert main(["solve", "--horizon", "1"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["first_action"] == "stop"
    assert out["value"] == pytest.approx(0.25, abs=1e-12)


def test_solve_forced(capsys):
    assert main(["solve", "--horizon", "1", "--forced"]) == 0
    assert json.loads(capsys.readouterr().out)["value"] == pytest.approx(0.2595778082749113, abs=1e-12)


def test_reproduce_small(capsys):
    assert main(["reproduce", "--table", "single", "--max-horizon", "2"]) == 0
    rows = json.loads(capsys.readouterr().out)["single_model"]
    assert [r["horizon"] for r in rows] == [0, 1, 2]
    assert rows[2]["optimal_with_stop"] == pytest.approx(0.17021731780078578, abs=1e-12)


@pytest.mark.parametrize(
    "argv",
    [
        ["solve", "--horizon", "6"],
        ["solve", "--horizon", "-1"],
        ["solve", "--horizon", "2", "--energy-lambda", "-0.1"],
        ["solve", "--horizon", "2", "--alpha", "1.5"],
        ["solve", "--horizon", "2", "--bias-flip", "0.6"],
        ["solve", "--horizon", "2", "--k01", "1e308"],
        ["reproduce", "--max-horizon", "9"],
    ],
)
def test_invalid_values_exit_2(argv, capsys):
    assert main(argv) == 2
    assert "error:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["solve", "--horizon", "2", "--k01", "nan"],
        ["solve", "--horizon", "2", "--alpha", "inf"],
        ["solve", "--horizon", "x"],
        ["solve"],
        ["reproduce", "--table", "bogus"],
        [],
    ],
)
def test_parser_rejects_exit_2(argv):
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_module_entry_point_subprocess():
    env = {**os.environ, "PYTHONPATH": str(SRC)}
    ok = subprocess.run(
        [sys.executable, "-m", "atomic_information_lab", "solve", "--horizon", "2"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert ok.returncode == 0, ok.stderr
    assert json.loads(ok.stdout)["first_action"] == "reference"
    bad = subprocess.run(
        [sys.executable, "-m", "atomic_information_lab", "solve", "--horizon", "7"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    assert bad.returncode == 2
    assert bad.stdout == ""


def test_retention_limit_cli(capsys):
    assert main(["solve", "--horizon", "3", "--retention-limit", "0.05", "--retention-mode", "pathwise"]) == 0
    r = json.loads(capsys.readouterr().out)["retention"]
    assert r["mode"] == "pathwise" and r["value"] == pytest.approx(0.25, abs=1e-12)
    assert r["unconstrained_p_violation"] == pytest.approx(1.0)
    assert main(["solve", "--horizon", "3", "--retention-limit", "0.05", "--retention-mode", "expected"]) == 0
    r = json.loads(capsys.readouterr().out)["retention"]
    assert r["p_changed"] == pytest.approx(0.05, abs=1e-12) and r["duality_gap"] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize(
    "argv",
    [
        ["solve", "--horizon", "3", "--retention-limit", "1.5"],
        ["solve", "--horizon", "3", "--retention-limit", "-0.1"],
        ["solve", "--horizon", "3", "--retention-limit", "0.1", "--forced"],
        ["solve", "--horizon", "3", "--retention-limit", "0.1", "--self-audit"],
    ],
)
def test_retention_limit_cli_rejects(argv, capsys):
    assert main(argv) == 2
    assert "error:" in capsys.readouterr().err


def test_reproduce_retention_table(capsys):
    assert main(["reproduce", "--table", "retention", "--max-horizon", "2"]) == 0
    rows = json.loads(capsys.readouterr().out)["retention"]
    assert [r["horizon"] for r in rows] == [2] * 7

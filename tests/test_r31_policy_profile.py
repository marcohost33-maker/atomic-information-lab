"""Roadmap R29-02: separate outputs of a policy, checked against the independent oracle.

``metrics.policy_profile`` reports Bayes risk, its three ingredients, the
mutual information about the written bit, energy and steps on their own. The
oracle side (``_oracle.Model``, exact rational forward messages, CTMC by
uniformisation) shares no code with the package; only the *decisions* of the
policy are taken from the package, as a table history -> action, so that both
sides evaluate the same tree.

Every negative control has a positive control next to it.
"""

import json
import math
import os
import pathlib
import subprocess
import sys
from fractions import Fraction as Fr
from itertools import product

import _oracle
import pytest

from atomic_information_lab import (
    STOP,
    TWO_HYPOTHESES,
    Hypothesis,
    Objective,
    Problem,
    best_open_loop,
    evaluate_policy,
    greedy_policy,
    optimal_policy,
    policy_profile,
)
from atomic_information_lab.metrics import PROFILE_SCOPE, PolicyProfile, binary_entropy_bits
from atomic_information_lab.reproduce import profile_table

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
ENV = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"}
NAMES = ("fast", "precise", "reference")
TOL = 1e-12


def fixed(seq):
    return lambda _b, hist, _h: seq[len(hist)] if len(hist) < len(seq) else STOP


def decision_table(problem, policy, horizon):
    """history -> action name or STOP, by walking the package policy once."""
    table = {}

    def rec(b, hist, h):
        choice = policy(b, hist, h) if h > 0 else STOP
        table[hist] = choice
        if choice != STOP:
            a = problem.action(choice)
            for y, _p, post in problem.branches(b, a):
                rec(post, hist + ((a.name, y),), h - 1)

    rec(problem.initial_belief(), (), horizon)
    return table


def h2(p):
    p = float(p)
    return 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def oracle_profile(components, table, lam=0.012, alpha=0.5, beta=0.0):
    """The same quantities from exact unnormalised forward messages (no package code)."""
    model = _oracle.Model(components, lam=lam, alpha=alpha, beta=beta, exact=True)
    acts = {a[0]: a for a in model.actions}
    out = {k: Fr(0) for k in ("b0", "bt", "chg", "energy", "steps")}
    ent = {"h0": 0.0, "ht": 0.0, "hm": 0.0}
    dist = {}

    def rec(m, hist, energy):
        tot = sum(m)
        if tot == 0:
            return
        choice = table[hist]
        if choice == STOP:
            m0 = sum(v for v, s in zip(m, model.states, strict=True) if s[1] == 1)
            mt = sum(v for v, s in zip(m, model.states, strict=True) if s[2] == 1)
            out["b0"] += m0 * (tot - m0) / tot
            out["bt"] += mt * (tot - mt) / tot
            out["chg"] += sum(v for v, s in zip(m, model.states, strict=True) if s[1] != s[2])
            out["energy"] += tot * energy
            out["steps"] += tot * len(hist)
            ent["h0"] += float(tot) * h2(m0 / tot)
            ent["ht"] += float(tot) * h2(mt / tot)
            for th in range(len(components)):
                w = sum(v for v, s in zip(m, model.states, strict=True) if s[0] == th) / tot
                if w > 0:
                    ent["hm"] -= float(tot) * float(w) * math.log2(float(w))
            dist[len(hist)] = dist.get(len(hist), Fr(0)) + tot
            return
        a = acts[choice]
        for y in (0, 1):
            rec(model.step(m, a, y), hist + ((choice, y),), energy + Fr(a[3]))

    rec(model.root(), (), Fr(0))
    risk = Fr(alpha) * out["b0"] + (1 - Fr(alpha)) * out["bt"] + Fr(beta) * out["chg"]
    prior_model = -sum(w * math.log2(w) for w, *_ in components if w > 0)
    return {
        "bayes_risk": float(risk),
        "value": float(risk + Fr(lam) * out["energy"]),
        "brier_initial": float(out["b0"]),
        "brier_current": float(out["bt"]),
        "p_changed": float(out["chg"]),
        "information_initial_bits": 1.0 - ent["h0"],
        "information_model_bits": prior_model - ent["hm"],
        "entropy_current_bits": ent["ht"],
        "expected_energy": float(out["energy"]),
        "expected_steps": float(out["steps"]),
        "dist": {k: float(v) for k, v in dist.items()},
    }


def assert_matches(p: PolicyProfile, ref: dict):
    for key in (
        "value",
        "bayes_risk",
        "brier_initial",
        "brier_current",
        "p_changed",
        "information_initial_bits",
        "information_model_bits",
        "entropy_current_bits",
        "expected_energy",
        "expected_steps",
    ):
        assert getattr(p, key) == pytest.approx(ref[key], abs=TOL), key
    for k, mass in enumerate(p.steps_distribution):
        assert mass == pytest.approx(ref["dist"].get(k, 0.0), abs=TOL), f"P(T={k})"


SINGLE = ((1.0, 0.07, 0.045, 0.025),)
TWO = ((0.5, 0.07, 0.045, 0.025), (0.5, 0.23, 0.16, 0.10))


# ---------------------------------------------------------------- oracle agreement
@pytest.mark.parametrize("h", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("kind", ["optimal", "forced", "greedy"])
def test_single_model_profile_matches_exact_oracle(h, kind):
    pr = Problem()
    pol = {"optimal": optimal_policy(pr), "forced": optimal_policy(pr, allow_stop=False), "greedy": greedy_policy(pr)}[
        kind
    ]
    assert_matches(policy_profile(pr, pol, h), oracle_profile(SINGLE, decision_table(pr, pol, h)))


@pytest.mark.parametrize("h", [1, 2, 3])
def test_two_model_profile_matches_exact_oracle_with_other_weights(h):
    obj = Objective(alpha=0.3, energy_lambda=0.02, retention_weight=0.4)
    pr = Problem(TWO_HYPOTHESES, (0.5, 0.5), objective=obj)
    for pol in (optimal_policy(pr), fixed(("fast", "reference", "precise")[:h])):
        ref = oracle_profile(TWO, decision_table(pr, pol, h), lam=0.02, alpha=0.3, beta=0.4)
        assert_matches(policy_profile(pr, pol, h), ref)


def test_every_fixed_sequence_up_to_length_three_matches_exact_oracle():
    pr = Problem()
    seen = 0
    for n in range(1, 4):
        for seq in product(NAMES, repeat=n):
            assert_matches(policy_profile(pr, fixed(seq), n), oracle_profile(SINGLE, decision_table(pr, fixed(seq), n)))
            seen += 1
    assert seen == 39


@pytest.mark.parametrize("seq", [("reference", "precise"), ("fast", "reference", "fast")])
def test_brier_parts_and_retention_match_hidden_path_enumeration(seq):
    """Second method: every hidden path explicitly, squared error of the posterior-mean forecast."""
    acts = tuple(next(a for a in _oracle.DEFAULT_ACTIONS if a[0] == n) for n in seq)
    table, brier = _oracle.hidden_path_table(SINGLE, _oracle.DEFAULT_ACTIONS, acts)
    p = policy_profile(Problem(), fixed(seq), len(seq))
    assert p.brier_initial == pytest.approx(sum(b[0] for b in brier.values()), abs=1e-12)
    assert p.brier_current == pytest.approx(sum(b[1] for b in brier.values()), abs=1e-12)
    assert p.p_changed == pytest.approx(sum(e[3] for e in table.values()), abs=1e-12)
    # information from the path table: H(X0) - sum_y P(y) H(X0 | y)
    info = 1.0 - sum(e[0] * h2(e[1] / e[0]) for e in table.values())
    assert p.information_initial_bits == pytest.approx(info, abs=1e-12)


# ---------------------------------------------------------------- identities
@pytest.mark.parametrize("h", [0, 2, 4])
def test_profile_agrees_with_evaluate_policy_and_its_own_decomposition(h):
    obj = Objective(alpha=0.3, energy_lambda=0.02, retention_weight=0.4)
    pr = Problem(objective=obj)
    pol = optimal_policy(pr, allow_stop=False)
    p, e = policy_profile(pr, pol, h), evaluate_policy(pr, pol, h)
    assert p.value == pytest.approx(e.value, abs=TOL)
    assert p.bayes_risk == pytest.approx(e.expected_terminal_loss, abs=TOL)
    assert p.expected_energy == pytest.approx(e.expected_energy, abs=TOL)
    assert p.expected_steps == pytest.approx(e.expected_steps, abs=TOL)
    assert p.value == pytest.approx(p.bayes_risk + p.energy_cost, abs=TOL)
    assert p.energy_cost == pytest.approx(0.02 * p.expected_energy, abs=TOL)
    assert p.bayes_risk == pytest.approx(0.3 * p.brier_initial + 0.7 * p.brier_current + 0.4 * p.p_changed, abs=TOL)
    assert sum(p.steps_distribution) == pytest.approx(1.0, abs=TOL)
    assert p.expected_steps == pytest.approx(sum(k * q for k, q in enumerate(p.steps_distribution)), abs=TOL)
    assert p.steps_distribution[h] == pytest.approx(1.0, abs=TOL)  # forced: always exactly h readings
    assert p.scope == PROFILE_SCOPE and PROFILE_SCOPE in p.notes[0]


def test_stop_policy_reports_nothing_learned_and_nothing_spent():
    p = policy_profile(Problem(), lambda *_: STOP, 5)
    assert p.information_initial_bits == 0.0 and p.information_model_bits == 0.0
    assert p.expected_steps == 0.0 and p.expected_energy == 0.0 and p.p_changed == 0.0
    assert p.bayes_risk == p.value == 0.25 and p.entropy_current_bits == 1.0
    assert p.steps_distribution == (1.0, 0.0, 0.0, 0.0, 0.0, 0.0)


# ---------------------------------------------------------------- properties
def test_information_never_falls_when_a_reading_is_appended_and_never_exceeds_the_prior():
    """I(X0; Y_1..n) <= I(X0; Y_1..n+1) <= H(X0); the Bayes risk has no such monotonicity (control)."""
    pr = Problem()
    info = {(): 0.0}
    risk = {(): 0.25}
    for n in range(1, 5):
        for seq in product(NAMES, repeat=n):
            p = policy_profile(pr, fixed(seq), n)
            info[seq], risk[seq] = p.information_initial_bits, p.bayes_risk
            assert info[seq[:-1]] - 1e-12 <= info[seq] <= 1.0
    assert any(risk[s] > risk[s[:-1]] + 1e-6 for s in risk if s)  # more readings CAN raise the risk


def test_label_swap_leaves_every_output_unchanged_and_a_real_change_does_not():
    """x <-> 1-x with k01 <-> k10 (and b <-> 1-b) is a relabelling under the symmetric prior."""
    seq = ("fast", "reference", "precise", "fast")
    a = policy_profile(Problem((Hypothesis(0.07, 0.045, 0.025),)), fixed(seq), 4)
    b = policy_profile(Problem((Hypothesis(0.045, 0.07, 0.025),)), fixed(seq), 4)
    c = policy_profile(Problem((Hypothesis(0.09, 0.045, 0.025),)), fixed(seq), 4)
    for key in ("bayes_risk", "p_changed", "information_initial_bits", "entropy_current_bits"):
        assert getattr(a, key) == pytest.approx(getattr(b, key), abs=1e-13), key
        assert abs(getattr(a, key) - getattr(c, key)) > 1e-4, key  # control: the comparison can fail


def test_start_belief_with_less_prior_entropy_bounds_the_information():
    pr = Problem()
    start = next(post for y, _p, post in pr.branches(pr.initial_belief(), pr.action("reference")) if y == 0)
    start = next(post for y, _p, post in pr.branches(start, pr.action("precise")) if y == 1)
    p = policy_profile(pr, fixed(("fast", "precise")), 2, start)
    prior = binary_entropy_bits(pr.marginals(start)["p_initial_one"])
    assert 0.0 < prior < 0.9 and p.entropy_initial_prior_bits == prior
    assert 0.0 < p.information_initial_bits < prior


def test_model_information_is_positive_only_with_more_than_one_hypothesis():
    seq = ("precise",) * 4
    two = policy_profile(Problem(TWO_HYPOTHESES, (0.5, 0.5)), fixed(seq), 4)
    one = policy_profile(Problem(), fixed(seq), 4)
    assert 1e-3 < two.information_model_bits < 1.0 and one.information_model_bits == 0.0


# ---------------------------------------------------------------- findings (pinned)
def test_n2_risk_falls_while_no_information_about_the_written_bit_arrives():
    """bias_flip = 0.5 re-draws the inversion before every reading: readings are noise about x0."""
    blind = Problem((Hypothesis(0.07, 0.045, 0.5),), objective=Objective(energy_lambda=0.0))
    for h in (1, 2, 3):
        p = policy_profile(blind, optimal_policy(blind), h)
        assert p.information_initial_bits <= 1e-14
        assert p.brier_initial == pytest.approx(0.25, abs=1e-14)
    assert p.bayes_risk < 0.2495 and p.brier_current < 0.249 and p.p_changed > 0.1
    # positive control: the same code sees information when the inversion persists
    seeing = Problem(objective=Objective(energy_lambda=0.0))
    assert policy_profile(seeing, optimal_policy(seeing), 2).information_initial_bits > 0.2


def test_n32_the_value_optimal_sequence_is_not_the_most_informative_one():
    pr = Problem()
    rows = {s: policy_profile(pr, fixed(s), 2) for s in product(NAMES, repeat=2)}
    best_value = min(rows, key=lambda s: rows[s].value)
    best_info = max(rows, key=lambda s: rows[s].information_initial_bits)
    assert best_value == ("reference", "precise") == best_open_loop(pr, 2)[1]
    assert best_info == ("fast", "reference")
    v, i = rows[best_value], rows[best_info]
    assert v.information_initial_bits == pytest.approx(0.2119612, abs=1e-6)
    assert i.information_initial_bits == pytest.approx(0.2897201, abs=1e-6)
    assert i.expected_energy == pytest.approx(1.8) and v.expected_energy == pytest.approx(3.6)
    assert i.p_changed == pytest.approx(0.081634, abs=1e-5) and v.p_changed == pytest.approx(0.186168, abs=1e-5)
    assert i.value - v.value == pytest.approx(0.002714, abs=1e-5)  # ... and costs 0.0027 in value


def test_golden_profile_of_the_optimal_policy_at_horizon_five():
    p = policy_profile(Problem(), optimal_policy(Problem()), 5)
    assert p.value == pytest.approx(0.1418447352, abs=1e-10)  # README single-model table
    assert p.bayes_risk == pytest.approx(0.0968473566, abs=1e-9)
    assert p.information_initial_bits == pytest.approx(0.3866141413, abs=1e-9)
    assert p.p_changed == pytest.approx(0.1511485985, abs=1e-9)
    assert p.expected_energy == pytest.approx(3.7497815512, abs=1e-9)
    assert p.expected_steps == pytest.approx(3.5633874233, abs=1e-9)
    # the policy stops after three readings or uses all five, never four
    assert p.steps_distribution[3] == pytest.approx(0.7183062884, abs=1e-9)
    assert p.steps_distribution[4] == 0.0 and p.steps_distribution[5] == pytest.approx(0.2816937116, abs=1e-9)


def test_profile_table_value_column_equals_the_single_model_table():
    from atomic_information_lab.reproduce import single_model_table

    rows = profile_table(3)
    assert len(rows) == 12 and {r["policy"] for r in rows} == {
        "optimal_with_stop",
        "forced_exactly_h",
        "best_fixed_sequence",
    }
    ref = {r["horizon"]: r for r in single_model_table(3)}
    for r in rows:
        key = {
            "optimal_with_stop": "optimal_with_stop",
            "forced_exactly_h": "forced_exactly_h",
            "best_fixed_sequence": "best_open_loop",
        }[r["policy"]]
        assert r["value"] == pytest.approx(ref[r["h"]][key], abs=TOL)
        assert r["scope"] == PROFILE_SCOPE


# ---------------------------------------------------------------- fail-closed
@pytest.mark.parametrize("bad", [-1, 6, 2.0, True, None])
def test_bad_horizon_is_refused(bad):
    with pytest.raises(ValueError, match="horizon"):
        policy_profile(Problem(), lambda *_: STOP, bad)


def test_bad_belief_and_unknown_action_are_refused():
    pr = Problem()
    with pytest.raises(ValueError, match="belief"):
        policy_profile(pr, lambda *_: STOP, 1, (0.5,) * 8)
    with pytest.raises(ValueError, match="unknown action"):
        policy_profile(pr, lambda *_: "nope", 1)


@pytest.mark.parametrize("bad", [-1e-9, 1.0000001, float("nan"), float("inf"), True, "0.5", None])
def test_binary_entropy_refuses_non_probabilities(bad):
    with pytest.raises(ValueError):
        binary_entropy_bits(bad)


def test_binary_entropy_reference_values():
    assert binary_entropy_bits(0.0) == binary_entropy_bits(1.0) == 0.0
    assert binary_entropy_bits(0.5) == 1.0
    assert binary_entropy_bits(0.11) == pytest.approx(0.4999157, abs=1e-6)  # capacity of BSC(0.11) is ~0.5


# ---------------------------------------------------------------- command line
def _run(*args):
    return subprocess.run(
        [sys.executable, "-m", "atomic_information_lab", *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=ENV,
        timeout=600,
    )


def test_cli_solve_profile_and_its_refusals():
    p = _run("solve", "--horizon", "2", "--profile")
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    prof = out["profile"]
    assert prof["value"] == pytest.approx(out["value"], abs=TOL)
    assert prof["scope"] == PROFILE_SCOPE and prof["information_initial_bits"] == pytest.approx(0.2119612, abs=1e-6)
    plain = json.loads(_run("solve", "--horizon", "2").stdout)
    assert "profile" not in plain and plain["value"] == out["value"]
    forced = json.loads(_run("solve", "--horizon", "1", "--forced", "--profile").stdout)
    assert forced["profile"]["expected_steps"] == 1.0 and forced["profile"]["information_initial_bits"] <= 1e-14
    bad = _run("solve", "--horizon", "2", "--profile", "--retention-limit", "0.1")
    assert bad.returncode == 2 and bad.stdout == "" and "--profile" in bad.stderr


def test_cli_reproduce_profile_table():
    p = _run("reproduce", "--table", "profile", "--max-horizon", "1")
    assert p.returncode == 0, p.stderr
    rows = json.loads(p.stdout)["profile"]
    assert [(r["h"], r["policy"]) for r in rows][:2] == [(0, "optimal_with_stop"), (0, "forced_exactly_h")]
    forced = next(r for r in rows if r["h"] == 1 and r["policy"] == "forced_exactly_h")
    assert forced["value"] == pytest.approx(0.2595778083, abs=1e-10)  # README: forcing raises the risk
    assert forced["information_initial_bits"] <= 1e-14  # ... and buys no information about x0

"""The package's DECISIONS, scored by the independent oracle (cross-evaluation).

Value tests compare numbers: package optimum vs oracle optimum. They do not
show that the package acts optimally -- a decision rule can be wrong at a node
whose value is computed correctly elsewhere. Here the complete package
pipeline (its belief filter and its Bellman choice) is replayed inside the
oracle's dynamics and scored there:

* method A: oracle policy evaluation vs oracle optimal value (exact for h <= 3),
* method B: vs the minimum over ALL deterministic policies (brute force, h <= 3).

The wrapper is the one the research gate uses, so positives and negatives go
through the same check.
"""

import importlib.util
from pathlib import Path

import _oracle as O
import pytest
from test_oracle_agreement import CASES, make

from atomic_information_lab import greedy_policy, optimal_policy

SPEC = importlib.util.spec_from_file_location("gate", Path(__file__).parents[1] / "scripts" / "research_gate.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)

TOL = 1e-11  # >= horizon * TIE_TOL; observed gaps are 0 or a few ulp


def oracle_score(pkg, model, choose, h):
    value, _ = model.policy_value(model.root(), h, gate.oracle_side_policy(pkg, model, choose))
    return float(value)


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("allow_stop", [True, False])
def test_package_policy_is_optimal_under_the_oracle_exact(name, allow_stop):
    pkg, orc = make(**CASES[name])
    model = orc(True)
    choose = optimal_policy(pkg, allow_stop=allow_stop)
    for h in range(1, 4):
        best = float(model.value(model.root(), h, allow_stop=allow_stop))
        assert oracle_score(pkg, model, choose, h) == pytest.approx(best, abs=TOL)


@pytest.mark.parametrize("name", ["default", "two_models", "retention", "fast_dynamics"])
def test_package_policy_is_optimal_under_the_oracle_h4(name):
    pkg, orc = make(**CASES[name])
    model = orc(False)
    best = model.value(model.root(), 4)
    assert oracle_score(pkg, model, optimal_policy(pkg), 4) == pytest.approx(best, abs=1e-10)


@pytest.mark.parametrize("name", ["default", "lambda0", "two_models", "flip_half", "deterministic_sensors"])
@pytest.mark.parametrize("allow_stop", [True, False])
def test_package_policy_matches_brute_force_over_all_policies(name, allow_stop):
    kw = {"hyps": ((1.0, 0.07, 0.045, 0.025),), "actions": O.DEFAULT_ACTIONS, "lam": 0.012, "alpha": 0.5}
    kw.update(CASES[name])
    pkg, orc = make(**kw)
    model = orc(False)
    choose = optimal_policy(pkg, allow_stop=allow_stop)
    for h in range(1, 4):
        brute, _ = O.brute_force_all_policies(
            kw["hyps"], kw["actions"], h, kw["lam"], kw["alpha"], kw.get("beta", 0.0), allow_stop=allow_stop
        )
        assert oracle_score(pkg, model, choose, h) == pytest.approx(brute, abs=1e-10)


# ---------------------------------------------------------------- negative controls
@pytest.mark.parametrize(
    "rule, h, min_gap",
    [
        (lambda pkg: optimal_policy(pkg, allow_stop=False), 4, 1e-2),  # 0.1564 vs 0.1462
        (greedy_policy, 4, 1e-1),  # greedy never measures: 0.25 vs 0.1462
        (lambda pkg: optimal_policy(pkg, allow_stop=False), 1, 9e-3),  # forced read costs 0.0096
    ],
    ids=["forced_h4", "greedy_h4", "forced_h1"],
)
def test_suboptimal_package_rules_are_detected(rule, h, min_gap):
    pkg, orc = make()
    model = orc(h <= 3)
    gap = oracle_score(pkg, model, rule(pkg), h) - float(model.value(model.root(), h))
    assert gap > min_gap  # far outside TOL: the check discriminates


def _depth2_nodes_on_path(pkg, choose, horizon):
    """Histories of length 2 that the policy reaches with positive probability and acts at."""
    found = []

    def walk(belief, hist):
        h = horizon - len(hist)
        name = choose(belief, hist, h)
        if name == "stop":
            return
        if len(hist) == 2:
            found.append(hist)
            return
        for y, _, post in pkg.branches(belief, pkg.action(name)):
            walk(post, hist + ((name, y),))

    walk(pkg.initial_belief(), ())
    return found


def test_single_wrong_deep_decision_is_detected():
    """Optimal everywhere except at one reachable node at depth 2 -- invisible to every value test."""
    pkg, orc = make()
    model = orc(True)
    opt = optimal_policy(pkg)
    nodes = _depth2_nodes_on_path(pkg, opt, 3)
    assert nodes, "control: the optimal h=3 policy must act at depth 2"
    target = nodes[0]

    def wrong(belief, history, h):
        best = opt(belief, history, h)
        if history == target:
            return "precise" if best != "precise" else "fast"
        return best

    best = float(model.value(model.root(), 3))
    assert oracle_score(pkg, model, opt, 3) == pytest.approx(best, abs=TOL)
    assert oracle_score(pkg, model, wrong, 3) - best > 1e-6

"""Metamorphic relations: exact symmetries of the model, checked without any oracle.

The oracle is a second transcription of the same specification, so a mistake
in the specification itself would be shared. Each relation below follows from
the model definition alone (relabelling or re-parametrising something that
cannot matter), so a violation exposes an error in the package regardless of
what the oracle says. Every relation is paired with a control showing that the
compared quantities are not trivially equal.
"""

from itertools import product

import pytest

from atomic_information_lab import (
    DEFAULT_ACTIONS,
    DEFAULT_HYPOTHESIS,
    TWO_HYPOTHESES,
    Action,
    Hypothesis,
    Objective,
    Problem,
    solve,
)

TOL = 1e-13
H1, H2 = TWO_HYPOTHESES
H3 = Hypothesis(0.0, 0.3, 0.2)


def posterior(pr, history):
    b = pr.initial_belief()
    for name, y in history:
        match = [post for yy, _, post in pr.branches(b, pr.action(name)) if yy == y]
        if not match:
            return None
        b = match[0]
    return b


def histories(pr, depth):
    names = [a.name for a in pr.actions]
    for n in range(1, depth + 1):
        for seq in product(names, repeat=n):
            for ys in product((0, 1), repeat=n):
                yield tuple(zip(seq, ys, strict=True))


def values(pr, horizons=range(5)):
    return [solve(pr, horizon=h).value for h in horizons]


# ---- relabel the sensor inversion b -> 1-b: every reading flips, nothing about x changes
@pytest.mark.parametrize("pr", [Problem(), Problem(TWO_HYPOTHESES, (0.3, 0.7))], ids=["one", "two"])
def test_flipping_every_reading_leaves_the_bit_posterior_unchanged(pr):
    checked = 0
    for hist in histories(pr, 3):
        a, b = posterior(pr, hist), posterior(pr, tuple((n, 1 - y) for n, y in hist))
        assert (a is None) == (b is None)
        if a is None:
            continue
        ma, mb = pr.marginals(a), pr.marginals(b)
        for key in ("p_initial_one", "p_current_one", "p_changed"):
            assert ma[key] == pytest.approx(mb[key], abs=TOL)
        assert pr.model_posterior(a) == pytest.approx(pr.model_posterior(b), abs=TOL)
        checked += 1
    assert checked > 200


# ---- relabel the bit x -> 1-x: swap k01 <-> k10, target readings flip, reference readings do not
def _mirror(h):
    return Hypothesis(h.k10, h.k01, h.bias_flip)


@pytest.mark.parametrize("hyps", [(DEFAULT_HYPOTHESIS,), TWO_HYPOTHESES], ids=["one", "two"])
def test_relabelling_the_bit_mirrors_the_posterior(hyps):
    pr = Problem(hyps)
    mirrored = Problem(tuple(_mirror(h) for h in hyps))
    ref = {a.name for a in pr.actions if a.reads_reference}
    moved = 0
    for hist in histories(pr, 3):
        a = posterior(pr, hist)
        b = posterior(mirrored, tuple((n, y if n in ref else 1 - y) for n, y in hist))
        if a is None:
            assert b is None
            continue
        ma, mb = pr.marginals(a), mirrored.marginals(b)
        assert mb["p_initial_one"] == pytest.approx(1 - ma["p_initial_one"], abs=TOL)
        assert mb["p_current_one"] == pytest.approx(1 - ma["p_current_one"], abs=TOL)
        assert mb["p_changed"] == pytest.approx(ma["p_changed"], abs=TOL)
        moved += abs(ma["p_current_one"] - 0.5) > 1e-3
    assert moved > 100  # control: the posteriors are not all stuck at 1/2
    assert values(pr) == pytest.approx(values(mirrored), abs=TOL)


def test_mirror_control_without_relabelling_the_readings_differs():
    """Control for the relation above: mirror the rates but keep the readings -> posteriors differ."""
    pr, mirrored = Problem(), Problem((_mirror(DEFAULT_HYPOTHESIS),))
    hist = (("reference", 0), ("precise", 1), ("precise", 1))
    a, b = pr.marginals(posterior(pr, hist)), mirrored.marginals(posterior(mirrored, hist))
    assert abs(b["p_current_one"] - (1 - a["p_current_one"])) > 0.1


# ---- the order of hypotheses and actions is a labelling, not a parameter
def test_permuting_hypotheses_changes_nothing():
    base = Problem((H1, H2, H3), (0.2, 0.3, 0.5))
    perm = Problem((H3, H1, H2), (0.5, 0.2, 0.3))
    assert values(base) == pytest.approx(values(perm), abs=TOL)
    for hist in histories(base, 2):
        a, b = posterior(base, hist), posterior(perm, hist)
        ma, mb = base.marginals(a), perm.marginals(b)
        assert ma == pytest.approx(mb, abs=TOL)
        pa, pb = base.model_posterior(a), perm.model_posterior(b)
        assert (pb[1], pb[2], pb[0]) == pytest.approx(pa, abs=TOL)


@pytest.mark.parametrize("order", [(2, 0, 1), (1, 2, 0), (2, 1, 0)])
def test_permuting_actions_keeps_the_values(order):
    pr = Problem()
    perm = Problem(actions=tuple(DEFAULT_ACTIONS[i] for i in order))
    assert values(pr) == pytest.approx(values(perm), abs=TOL)


# ---- splitting a hypothesis into identical copies is not new information
@pytest.mark.parametrize("w", [0.5, 0.3, 0.9])
def test_splitting_a_hypothesis_into_copies_changes_nothing(w):
    single = Problem((DEFAULT_HYPOTHESIS,))
    split = Problem((DEFAULT_HYPOTHESIS, DEFAULT_HYPOTHESIS), (w, 1 - w))
    assert values(single) == pytest.approx(values(split), abs=TOL)
    for hist in histories(single, 2):
        assert single.marginals(posterior(single, hist)) == pytest.approx(
            split.marginals(posterior(split, hist)), abs=TOL
        )
        # the copies cannot be told apart: their posterior weights keep the prior ratio
        pw = split.model_posterior(posterior(split, hist))
        assert pw[0] == pytest.approx(w, abs=TOL)


def test_split_control_with_distinct_hypotheses_learns():
    mixed = Problem(TWO_HYPOTHESES, (0.5, 0.5))
    shifts = [abs(mixed.model_posterior(posterior(mixed, h))[0] - 0.5) for h in histories(mixed, 2)]
    assert max(shifts) > 1e-3


# ---- only the product lambda * E is a cost
@pytest.mark.parametrize("c", [0.5, 4.0])
def test_cost_depends_only_on_lambda_times_energy(c):
    base = Problem()
    scaled = Problem(
        actions=tuple(Action(a.name, a.backaction, a.error, a.energy / c, a.reads_reference) for a in DEFAULT_ACTIONS),
        objective=Objective(0.5, 0.012 * c),
    )
    assert values(base) == pytest.approx(values(scaled), abs=TOL)


def test_more_expensive_energy_never_lowers_the_value():
    lams = [0.0, 0.003, 0.012, 0.03, 0.1]
    rows = [values(Problem(objective=Objective(0.5, lam))) for lam in lams]
    for lo, hi in zip(rows, rows[1:], strict=False):
        assert all(a <= b + TOL for a, b in zip(lo, hi, strict=True))
    assert rows[0][4] < rows[-1][4] - 1e-3  # control: the price matters


# ---- a strictly dominated copy of an action is never worth having
def test_adding_a_dominated_action_changes_nothing():
    pr = Problem()
    dear = Action("fast_dear", 1.0, 0.12, 2.0)  # same physics as "fast", twice the energy
    more = Problem(actions=DEFAULT_ACTIONS + (dear,))
    assert values(pr) == pytest.approx(values(more), abs=TOL)
    for h in range(1, 5):
        assert solve(more, horizon=h).action != "fast_dear"


def test_dominance_control_a_cheaper_copy_helps():
    cheap = Action("fast_cheap", 1.0, 0.12, 0.2)
    more = Problem(actions=DEFAULT_ACTIONS + (cheap,))
    assert solve(more, horizon=4).value < solve(Problem(), horizon=4).value - 1e-4

"""Structural properties, negative controls (with positive controls) and documented findings."""

import itertools

import pytest

from atomic_information_lab import (
    DEFAULT_ACTIONS,
    DEFAULT_HYPOTHESIS,
    STOP,
    TWO_HYPOTHESES,
    Hypothesis,
    Objective,
    Problem,
    best_open_loop,
    evaluate_policy,
    greedy_policy,
    solve,
)

TOL = 1e-12
GRID = [
    Problem(),
    Problem(objective=Objective(energy_lambda=0.0)),
    Problem(objective=Objective(alpha=0.0)),
    Problem(objective=Objective(alpha=1.0, retention_weight=0.2)),
    Problem((Hypothesis(0.3, 0.01, 0.2),)),
    Problem(TWO_HYPOTHESES, (0.3, 0.7)),
]


@pytest.mark.parametrize("pr", GRID)
def test_value_with_stop_is_monotone_in_horizon(pr):
    vals = [solve(pr, horizon=h).value for h in range(5)]
    assert all(b <= a + TOL for a, b in zip(vals, vals[1:], strict=False))


@pytest.mark.parametrize("pr", GRID)
def test_dominance_chain(pr):
    """V_h <= best open loop <= L(b0); V_h <= forced; V_h <= greedy."""
    for h in range(4):
        v = solve(pr, horizon=h).value
        fixed, _ = best_open_loop(pr, h)
        assert v <= fixed + TOL
        assert fixed <= pr.terminal_loss(pr.initial_belief()) + TOL
        assert v <= solve(pr, horizon=h, allow_stop=False).value + TOL
        assert v <= evaluate_policy(pr, greedy_policy(pr), h).value + TOL


def test_forced_horizon_is_not_monotone():
    """Why STOP matters (historic F-01): forcing more measurements can increase the risk."""
    pr = Problem()
    forced = [solve(pr, horizon=h, allow_stop=False).value for h in range(6)]
    assert forced[1] > forced[0]
    assert forced[5] > forced[4]
    assert solve(pr, horizon=1).action == STOP


def test_beliefs_stay_normalised_over_the_whole_tree():
    pr = Problem(TWO_HYPOTHESES, (0.5, 0.5))
    worst = 0.0
    frontier = [pr.initial_belief()]
    for _ in range(4):
        nxt = []
        for b in frontier:
            for a in pr.actions:
                br = pr.branches(b, a)
                worst = max(worst, abs(sum(p for _, p, _ in br) - 1.0))
                for _, _, post in br:
                    worst = max(worst, abs(sum(post) - 1.0))
                    nxt.append(post)
        frontier = nxt
    assert len(frontier) == 6**4
    assert worst < 1e-12


def test_brier_is_strictly_proper():
    """argmin_q E_p[(q - X)^2] = p, which is what makes p(1-p) the Bayes Brier risk."""
    grid = [i / 200 for i in range(201)]
    for p in [0.0, 0.1, 0.3, 0.5, 0.77, 1.0]:
        best = min(grid, key=lambda q: p * (1 - q) ** 2 + (1 - p) * q**2)
        assert best == pytest.approx(p, abs=1e-12)
        assert p * (1 - p) == pytest.approx(min(p * (1 - q) ** 2 + (1 - p) * q**2 for q in grid), abs=1e-4)


# ---------------------------------------------------------------- controls
def _posterior_x0_after(pr, seq_obs):
    b = pr.initial_belief()
    for name, y in seq_obs:
        b = next(post for yy, _, post in pr.branches(b, pr.action(name)) if yy == y)
    return pr.marginals(b)["p_initial_one"]


def test_negative_control_bias_flip_half_carries_no_information():
    """With bias_flip=0.5 every reading is independent of x0: the posterior must stay at 0.5."""
    pr = Problem((Hypothesis(0.07, 0.045, 0.5),))
    for seq in itertools.product(["fast", "precise", "reference"], repeat=2):
        for ys in itertools.product((0, 1), repeat=2):
            assert _posterior_x0_after(pr, list(zip(seq, ys, strict=True))) == pytest.approx(0.5, abs=1e-15)
    assert [solve(pr, horizon=h).value for h in range(5)] == [0.25] * 5
    # positive control: same model with a persistent sensor bias does learn about x0
    informative = Problem((Hypothesis(0.07, 0.045, 0.0),))
    assert abs(_posterior_x0_after(informative, [("reference", 0), ("precise", 1)]) - 0.5) > 0.2  # measured 0.759


def test_finding_risk_drops_without_information():
    """FINDING N-2: at energy_lambda=0 the optimal risk falls below 0.25 although no reading
    carries information -- the gain comes from back-action pushing x towards the
    stationary distribution (current state becomes predictable). Risk reduction != information."""
    pr = Problem((Hypothesis(0.07, 0.045, 0.5),), objective=Objective(energy_lambda=0.0))
    values = [solve(pr, horizon=h).value for h in range(5)]
    assert values[1] < 0.25 - 1e-4
    assert _posterior_x0_after(pr, [("precise", 1)]) == pytest.approx(0.5, abs=1e-15)
    # the initial-bit component alone gains nothing
    pr0 = Problem((Hypothesis(0.07, 0.045, 0.5),), objective=Objective(alpha=1.0, energy_lambda=0.0))
    assert [solve(pr0, horizon=h).value for h in range(5)] == pytest.approx([0.25] * 5, abs=1e-15)


def test_negative_control_symmetric_without_reference():
    """Symmetric rates + no reference read: (x, b) -> (1-x, 1-b) is an exact symmetry,
    so nothing can be learned even for free measurements."""
    acts = DEFAULT_ACTIONS[:2]
    pr = Problem((Hypothesis(0.05, 0.05, 0.025),), actions=acts, objective=Objective(energy_lambda=0.0))
    assert [solve(pr, horizon=h).value for h in range(5)] == pytest.approx([0.25] * 5, abs=1e-15)
    # positive control: adding the reference read breaks the symmetry
    pr_ref = Problem((Hypothesis(0.05, 0.05, 0.025),), objective=Objective(energy_lambda=0.0))
    assert solve(pr_ref, horizon=2).value < 0.2


def test_reference_is_what_makes_the_default_model_informative():
    with_ref = Problem()
    no_ref = Problem(actions=DEFAULT_ACTIONS[:2])
    assert [solve(no_ref, horizon=h).value for h in range(6)] == pytest.approx([0.25] * 6, abs=1e-15)
    assert solve(with_ref, horizon=5).value < 0.15


def test_single_hypothesis_mixtures_reduce_to_single_model():
    single = Problem((DEFAULT_HYPOTHESIS,))
    dup = Problem((DEFAULT_HYPOTHESIS, DEFAULT_HYPOTHESIS), (0.5, 0.5))
    lopsided = Problem((DEFAULT_HYPOTHESIS, TWO_HYPOTHESES[1]), (1.0, 0.0))
    for h in range(5):
        v = solve(single, horizon=h).value
        assert solve(dup, horizon=h).value == pytest.approx(v, abs=TOL)
        assert solve(lopsided, horizon=h).value == pytest.approx(v, abs=TOL)


def test_model_posterior_learns_the_right_hypothesis():
    pr = Problem(TWO_HYPOTHESES, (0.5, 0.5))
    b = pr.initial_belief()
    assert pr.model_posterior(b) == pytest.approx((0.5, 0.5))
    for y in (0, 1, 0, 1):  # alternating readings favour the faster-flipping hypothesis
        b = next(post for yy, _, post in pr.branches(b, pr.action("fast")) if yy == y)
    post = pr.model_posterior(b)
    assert sum(post) == pytest.approx(1.0, abs=1e-12)
    assert post[1] > 0.5


def test_determinism_and_no_shared_state():
    a = solve(Problem(TWO_HYPOTHESES), horizon=3)
    Problem((Hypothesis(1.0, 1.0, 0.5),))
    b = solve(Problem(TWO_HYPOTHESES), horizon=3)
    assert a == b

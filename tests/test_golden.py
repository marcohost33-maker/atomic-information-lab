"""Golden values: the historic R11/R12 result files (committed data set, 2026-10-17).

These pin the consolidated code to the numbers published in the R11/R12
bundles. Agreement with an *independent* oracle is tested separately
(test_oracle_agreement.py); golden values alone would only show consistency
with the old code.
"""

import pytest

from atomic_information_lab.reproduce import single_model_table, two_model_sensitivity, two_model_table

TOL = 1e-12

# horizon: (optimal_with_stop, forced_exactly_h, best_open_loop, greedy_with_stop, expected_steps)
R11 = {
    0: (0.25, 0.25, 0.25, 0.25, 0.0),
    1: (0.25, 0.2595778082749113, 0.25, 0.25, 0.0),
    2: (0.17021731780078578, 0.17021731780078578, 0.17021731780078578, 0.25, 2.0),
    3: (0.1569144983839333, 0.1569144983839333, 0.1569144983839333, 0.25, 3.0),
    4: (0.1462466036277991, 0.15640229342200573, 0.1569144983839333, 0.25, 3.281693711644314),
    5: (0.141844735228861, 0.16191611166702546, 0.1569144983839333, 0.25, 3.563387423288628),
}
# horizon: (adaptive, best_open_loop, nominal single-model policy)
R12 = {
    0: (0.25, 0.25, 0.25),
    1: (0.25, 0.25, 0.25),
    2: (0.1967940160651204, 0.1967940160651204, 0.20399125613461255),
    3: (0.1878507466883349, 0.1878507466883349, 0.1878507466883349),
    4: (0.1840526516771559, 0.1878507466883349, 0.1840526516771559),
}
# (retention_weight, reference_backaction): (adaptive, best_open_loop)
R12_SENS = {
    (0.0, 0.0): (0.17913961843095866, 0.18374790155022797),
    (0.0, 0.55): (0.1840526516771559, 0.1878507466883349),
    (0.0, 1.0): (0.18718885678030361, 0.19032765042727223),
    (0.1, 0.0): (0.19410436251122157, 0.19602085032389763),
    (0.1, 0.55): (0.20704340762238077, 0.20995714641426588),
    (0.1, 1.0): (0.2123212165520687, 0.21486296805894012),
    (0.2, 0.0): (0.20627278649909248, 0.20681027486377274),
    (0.2, 0.55): (0.22763997875717654, 0.22763997875717654),
    (0.2, 1.0): (0.2374535763238338, 0.23829834686374743),
    (0.4, 0.0): (0.22837386667554405, 0.2283891239435229),
    (0.4, 0.55): (0.25, 0.25),
    (0.4, 1.0): (0.25, 0.25),
}


@pytest.fixture(scope="module")
def single():
    return {r["horizon"]: r for r in single_model_table(5)}


@pytest.fixture(scope="module")
def two():
    return {r["horizon"]: r for r in two_model_table(4)}


@pytest.mark.parametrize("h", sorted(R11))
def test_single_model_table(single, h):
    opt, forced, fixed, greedy, steps = R11[h]
    r = single[h]
    assert r["optimal_with_stop"] == pytest.approx(opt, abs=TOL)
    assert r["forced_exactly_h"] == pytest.approx(forced, abs=TOL)
    assert r["best_open_loop"] == pytest.approx(fixed, abs=TOL)
    assert r["greedy_with_stop"] == pytest.approx(greedy, abs=TOL)
    assert r["expected_steps"] == pytest.approx(steps, abs=1e-9)


def test_single_model_headline_ratios(single):
    """Reported in the R11 audit: 12.40 % vs forced, 9.60 % vs best fixed (h=5)."""
    r = single[5]
    vs_forced = (r["forced_exactly_h"] - r["optimal_with_stop"]) / r["forced_exactly_h"]
    vs_fixed = (r["best_open_loop"] - r["optimal_with_stop"]) / r["best_open_loop"]
    assert round(100 * vs_forced, 2) == 12.40
    assert round(100 * vs_fixed, 2) == 9.60
    assert single[1]["first_action"] == "stop"
    assert single[5]["best_open_loop_sequence"] == ["fast", "reference", "fast"]


@pytest.mark.parametrize("h", sorted(R12))
def test_two_model_table(two, h):
    adaptive, fixed, nominal = R12[h]
    r = two[h]
    assert r["adaptive_with_stop"] == pytest.approx(adaptive, abs=TOL)
    assert r["best_open_loop"] == pytest.approx(fixed, abs=TOL)
    assert r["nominal_single_model_policy"] == pytest.approx(nominal, abs=TOL)


def test_two_model_headline_ratio(two):
    r = two[4]
    assert round(100 * (r["best_open_loop"] - r["adaptive_with_stop"]) / r["best_open_loop"], 2) == 2.02


def test_two_model_sensitivity():
    rows = two_model_sensitivity(4)
    assert len(rows) == 12
    for r in rows:
        adaptive, fixed = R12_SENS[(r["retention_weight"], r["reference_backaction"])]
        assert r["adaptive_with_stop"] == pytest.approx(adaptive, abs=TOL)
        assert r["best_open_loop"] == pytest.approx(fixed, abs=TOL)
        assert r["gap"] >= -TOL
    stop_cases = {(r["retention_weight"], r["reference_backaction"]) for r in rows if r["first_action"] == "stop"}
    assert stop_cases == {(0.4, 0.55), (0.4, 1.0)}

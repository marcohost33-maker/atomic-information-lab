"""Roadmap R29-03 and the cross-family gate of 2026-11-15: Pareto list of designs, Fisher spectrum.

``design_pareto`` keeps the designs no other design beats on every criterion;
``IdentifiabilityReport`` and ``DesignScore`` carry the eigenvalues of the
scale-free Fisher matrix and its condition number. Checks here use a second,
differently written front (sort and sweep), exact rational linear algebra for
determinant and extreme eigenvalues, and hand-made scores for the edge cases.
"""

import dataclasses
import json
import math
from fractions import Fraction as Fr
from itertools import permutations

import pytest
from test_r28_consumer_contract import TOKEN, _check_contract, _promises, _run

from atomic_information_lab import Calibration, Design, design_pareto, design_search, identifiability
from atomic_information_lab.identifiability import (
    PARETO_CRITERIA,
    RANK_TOL,
    SESSIONS_SCOPE,
    DesignScore,
    IdentifiabilityReport,
    parameter_names,
)
from atomic_information_lab.reproduce import PARETO_LENGTH, design_pareto_table

CAL = Calibration(prep_error=0.02)
ANCHORED = tuple(n for n in parameter_names(CAL) if n != "backaction:fast")
D4 = Design(("fast", "precise", "reference", "precise"), True)


@pytest.fixture(scope="module")
def scores4():
    return design_search(CAL, ANCHORED, 4)


@pytest.fixture(scope="module")
def scores5():
    return design_search(CAL, ANCHORED, 5)


def crit(s):
    if s.log_det is None or s.worst_sessions is None:
        return None
    return ({"GO": 0, "AMBIGUOUS": 1}[s.verdict], -s.log_det, s.worst_sessions, s.energy_per_session)


def sweep_front(scores):
    """Independent front: sort lexicographically, keep a point unless a kept or equal-sorted earlier point beats it."""
    pts = sorted((c, i) for i, s in enumerate(scores) if (c := crit(s)) is not None)
    keep = []
    for c, i in pts:  # a dominating point sorts strictly earlier, so comparing with all earlier points suffices
        if not any(d != c and all(x <= y for x, y in zip(d, c, strict=True)) for d, _ in pts[: pts.index((c, i))]):
            keep.append(i)
    return sorted(keep)


# ---------------------------------------------------------------- the front
@pytest.mark.parametrize("which", ["scores4", "scores5"])
def test_front_equals_an_independent_sweep_and_is_sound_and_complete(which, request):
    scores = request.getfixturevalue(which)
    front = design_pareto(scores)
    idx = [scores.index(s) for s in front]
    assert idx == sweep_front(scores)  # same set, and in the order of design_search
    members = [crit(s) for s in front]
    for c in members:  # sound: nobody in the front is beaten
        assert not any(d != c and all(x <= y for x, y in zip(d, c, strict=True)) for d in members)
    for i, s in enumerate(scores):  # complete: everybody outside is beaten by a member, or has no criteria
        c = crit(s)
        if i not in idx and c is not None:
            assert any(d != c and all(x <= y for x, y in zip(d, c, strict=True)) for d in members), s.design
    assert all(s.verdict != "NO-GO" for s in front)


@pytest.mark.parametrize("which", ["scores4", "scores5"])
def test_extreme_designs_are_always_members(which, request):
    scores = request.getfixturevalue(which)
    front = design_pareto(scores)
    cand = [s for s in scores if crit(s) is not None and s.verdict == "GO"]
    assert cand  # positive control
    assert scores[0] in front  # D-optimal (first of design_search)
    for key in (
        lambda s: (s.worst_sessions, s.energy_per_session, -s.log_det),
        lambda s: (s.energy_per_session, s.worst_sessions, -s.log_det),
        lambda s: (s.campaign_energy_crlb, s.worst_sessions, s.energy_per_session, -s.log_det),
    ):
        assert min(cand, key=key) in front


def test_n33_the_d_optimal_design_is_neither_the_smallest_bound_nor_the_cheapest_campaign(scores5):
    front = design_pareto(scores5)
    assert len(scores5) == 486 and sum(s.log_det is not None for s in scores5) == 84
    assert [s.design.sequence for s in front] == [
        ("fast", "precise", "reference", "reference", "precise"),
        ("fast", "precise", "reference", "reference", "fast"),
        ("fast", "reference", "precise", "precise", "reference"),
        ("fast", "reference", "precise", "reference", "fast"),
        ("fast", "fast", "reference", "precise", "precise"),
    ]
    assert all(s.design.known_write and s.verdict == "GO" for s in front)
    d_opt = front[0]
    fewest = min(front, key=lambda s: s.worst_sessions)
    cheapest = min(front, key=lambda s: s.campaign_energy_crlb)
    assert d_opt.worst_sessions == 1_517_144 and fewest.worst_sessions == 1_364_627
    assert fewest.design.sequence == ("fast", "fast", "reference", "precise", "precise")
    assert cheapest.design.sequence == ("fast", "reference", "precise", "reference", "fast")
    assert cheapest.campaign_energy_crlb / d_opt.campaign_energy_crlb == pytest.approx(0.8473, abs=1e-3)
    # the design with the smallest bound is also by far the best conditioned one
    assert d_opt.condition_number == pytest.approx(2385.12, rel=1e-4)
    assert fewest.condition_number == pytest.approx(168, rel=1e-2)
    assert fewest.log_det < d_opt.log_det - 4  # ... and far from D-optimal


def test_length_four_front(scores4):
    front = design_pareto(scores4)
    assert [s.design.sequence for s in front] == [
        ("fast", "precise", "reference", "precise"),
        ("fast", "precise", "reference", "fast"),
        ("fast", "reference", "fast", "precise"),
    ]
    assert front[0].worst_sessions == 6_687_058  # README identifiability table
    assert min(front, key=lambda s: s.campaign_energy_crlb) is front[2]


# ---------------------------------------------------------------- edge cases on hand-made scores
def mk(seq, verdict="GO", log_det=0.0, worst=100, energy=1.0):
    return DesignScore(Design(seq, True), verdict, "x", log_det, worst, SESSIONS_SCOPE, energy)


def test_identical_criteria_do_not_dominate_each_other():
    a, b, c = mk(("fast",)), mk(("precise",)), mk(("reference",), worst=101)
    assert design_pareto([a, b, c]) == [a, b]  # c is beaten, the twins stay


def test_each_criterion_alone_keeps_a_design_in_the_list():
    base = mk(("fast",))
    assert design_pareto([base, mk(("precise",), log_det=1.0, worst=200, energy=2.0)])[1].log_det == 1.0
    assert len(design_pareto([base, mk(("precise",), log_det=-1.0, worst=99, energy=2.0)])) == 2
    assert len(design_pareto([base, mk(("precise",), log_det=-1.0, worst=200, energy=0.5)])) == 2
    assert design_pareto([base, mk(("precise",), log_det=-1.0, worst=200, energy=2.0)]) == [base]  # control


def test_an_ambiguous_design_never_removes_a_go_design():
    go = mk(("fast",), "GO", log_det=-5.0, worst=1000, energy=9.0)
    amb = mk(("precise",), "AMBIGUOUS", log_det=5.0, worst=10, energy=1.0)
    assert design_pareto([go, amb]) == [go, amb]
    better_go = mk(("reference",), "GO", log_det=6.0, worst=9, energy=0.5)
    assert design_pareto([go, amb, better_go]) == [better_go]


def test_rank_deficient_designs_are_left_out_and_order_is_kept():
    nogo = DesignScore(Design(("fast",)), "NO-GO", "x", None, None, SESSIONS_SCOPE, 0.1)
    a, b = mk(("precise",), energy=2.0, worst=5), mk(("reference",), energy=1.0, worst=9)
    assert design_pareto([nogo, a, b]) == [a, b] and design_pareto([b, nogo, a]) == [b, a]
    assert design_pareto([nogo]) == []


@pytest.mark.parametrize("bad", [[], "GO", None, [1], [("a",)], 7])
def test_bad_input_is_refused(bad):
    with pytest.raises(ValueError, match="DesignScore"):
        design_pareto(bad)


@pytest.mark.parametrize("field", ["log_det", "energy"])
def test_non_finite_criterion_fails_closed(field):
    with pytest.raises(ArithmeticError, match="non-finite"):
        design_pareto([mk(("fast",)), mk(("precise",), **{field: float("nan")})])


def test_positional_construction_of_the_old_fields_is_unchanged():
    s = DesignScore(D4, "GO", "x", 1.0, 5, SESSIONS_SCOPE, 2.0)
    assert (s.rank, s.min_normalised_eigenvalue, s.condition_number, s.campaign_energy_crlb) == (None,) * 4
    old = [f.name for f in dataclasses.fields(IdentifiabilityReport)][:18]
    assert old[-2:] == ["energy_per_session", "notes"]
    assert [f.name for f in dataclasses.fields(IdentifiabilityReport)][18:] == [
        "normalised_eigenvalues",
        "condition_number",
    ]
    assert "sessions_scope" in {f.name for f in dataclasses.fields(DesignScore)}  # campaign_energy_crlb has a scope


# ---------------------------------------------------------------- Fisher spectrum
def frac_matrix(rows):
    return [[Fr(x) for x in r] for r in rows]


def exact_det(rows):
    """Determinant of the float matrix, exact in rationals (fraction-free of rounding)."""
    a = frac_matrix(rows)
    n, det = len(a), Fr(1)
    for c in range(n):
        piv = next((r for r in range(c, n) if a[r][c] != 0), None)
        if piv is None:
            return Fr(0)
        if piv != c:
            a[c], a[piv] = a[piv], a[c]
            det = -det
        det *= a[c][c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            a[r] = [x - f * y for x, y in zip(a[r], a[c], strict=True)]
    return det


def exact_inverse(rows):
    a = frac_matrix(rows)
    n = len(a)
    aug = [a[i] + [Fr(int(i == j)) for j in range(n)] for i in range(n)]
    for c in range(n):
        piv = next(r for r in range(c, n) if aug[r][c] != 0)
        aug[c], aug[piv] = aug[piv], aug[c]
        aug[c] = [x / aug[c][c] for x in aug[c]]
        for r in range(n):
            if r != c and aug[r][c] != 0:
                aug[r] = [x - aug[r][c] * y for x, y in zip(aug[r], aug[c], strict=True)]
    return [[float(x) for x in row[n:]] for row in aug]


def power_iteration(m, steps=4000):
    n = len(m)
    v = [1.0 + 0.1 * i for i in range(n)]
    lam = 0.0
    for _ in range(steps):
        w = [sum(m[i][j] * v[j] for j in range(n)) for i in range(n)]
        lam = math.sqrt(sum(x * x for x in w))
        v = [x / lam for x in w]
    return lam


def normalised(fisher):
    d = [math.sqrt(fisher[i][i]) for i in range(len(fisher))]
    return [[fisher[i][j] / (d[i] * d[j]) for j in range(len(d))] for i in range(len(d))]


@pytest.mark.parametrize(
    "design",
    [
        D4,
        Design(("fast", "fast", "reference", "precise", "precise"), True),
        Design(("fast", "reference", "precise", "reference", "fast"), True),
    ],
)
def test_spectrum_against_exact_rational_linear_algebra(design):
    r = identifiability(CAL, design, ANCHORED)
    assert r.verdict == "GO"
    lam = r.normalised_eigenvalues
    n = len(ANCHORED)
    assert len(lam) == n and list(lam) == sorted(lam)
    assert sum(lam) == pytest.approx(n, abs=1e-10)  # trace of a unit-diagonal matrix
    assert lam[0] == r.min_normalised_eigenvalue
    assert r.condition_number == pytest.approx(lam[-1] / lam[0], rel=1e-12)
    norm = normalised(r.fisher)
    assert power_iteration(norm) == pytest.approx(lam[-1], rel=1e-9)
    assert 1.0 / power_iteration(exact_inverse(norm)) == pytest.approx(lam[0], rel=1e-7)
    assert math.prod(lam) == pytest.approx(float(exact_det(norm)), rel=1e-7)
    # the D-criterion of design_search is the determinant of the same matrix in original coordinates
    score = next(s for s in design_search(CAL, ANCHORED, len(design.sequence), (True,)) if s.design == design)
    assert score.log_det == pytest.approx(math.log(float(exact_det(r.fisher))), abs=1e-6)
    assert (score.rank, score.condition_number) == (r.rank, r.condition_number)
    assert score.min_normalised_eigenvalue == r.min_normalised_eigenvalue
    assert score.campaign_energy_crlb == score.worst_sessions * score.energy_per_session


def test_condition_number_is_withheld_exactly_when_the_rank_is_deficient():
    every = parameter_names(CAL)
    nogo = identifiability(CAL, Design(("fast", "reference", "precise") * 2, True), every)
    assert nogo.verdict == "NO-GO" and nogo.condition_number is None
    assert nogo.normalised_eigenvalues[0] <= RANK_TOL < nogo.normalised_eigenvalues[-1]
    dead = identifiability(CAL, Design(("fast", "precise") * 2), ("k01", "prep_error"))
    assert dead.zero_information == ("prep_error",) and dead.condition_number is None
    # one informed parameter: F / (sqrt(F) * sqrt(F)) is 1 up to rounding (0.9999999999999999 on Windows)
    assert dead.min_normalised_eigenvalue == 0.0 and dead.normalised_eigenvalues == pytest.approx((1.0,), abs=1e-12)
    one = identifiability(CAL, D4, ("error:fast",))
    assert one.condition_number == 1.0  # control: defined when full rank (max / min of one eigenvalue)
    assert one.normalised_eigenvalues == pytest.approx((1.0,), abs=1e-12)


def test_condition_number_as_a_function_of_the_design(scores4, scores5):
    """Full rank is not the whole story: among full-rank designs the condition number spans a factor of 46 to 97."""
    c4 = sorted(s.condition_number for s in scores4 if s.condition_number is not None)
    c5 = sorted(s.condition_number for s in scores5 if s.condition_number is not None)
    assert (len(c4), len(c5)) == (12, 84)
    assert c4[0] == pytest.approx(763.27, rel=1e-4) and c4[-1] == pytest.approx(35241.5, rel=1e-4)
    assert c5[0] == pytest.approx(147.05, rel=1e-4) and c5[-1] == pytest.approx(14336.6, rel=1e-4)
    assert scores4[0].condition_number == pytest.approx(4074.03, rel=1e-4)  # D-optimal, length 4
    assert scores5[0].condition_number == pytest.approx(2385.12, rel=1e-4)  # D-optimal, length 5
    # the D-optimal design is never the best conditioned one
    assert scores4[0].condition_number > 5 * c4[0] and scores5[0].condition_number > 16 * c5[0]
    assert all((s.condition_number is None) == (s.log_det is None) for s in scores4 + scores5)


def test_energy_does_not_depend_on_the_order_of_the_readings():
    """A criterion of the Pareto list: the same readings in another order must cost exactly the same.

    The package uses ``math.fsum`` (correctly rounded, hence order-independent by construction).
    Control: uncompensated left-to-right addition does depend on the order (0.1 + 0.2 + 0.3).
    """
    from functools import reduce
    from operator import add

    from atomic_information_lab import Action

    acts = (Action("a", 1.0, 0.1, 0.1), Action("b", 2.0, 0.05, 0.2), Action("c", 0.5, 0.02, 0.3, reads_reference=True))
    cal = Calibration(actions=acts)
    orders = list(permutations(("a", "b", "c")))
    seen = {identifiability(cal, Design(p, True), ("error:a",)).energy_per_session for p in orders}
    assert seen == {math.fsum((0.1, 0.2, 0.3))}
    naive = {reduce(add, (cal.action(n).energy for n in p)) for p in orders}
    assert len(naive) > 1  # control: plain addition is order-dependent for these energies


# ---------------------------------------------------------------- consumers
def test_designs_cli_pareto_contract():
    p = _run("designs", "--length", "4", "--pareto")
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    d = out["designs"]
    assert (
        d["pareto"] is True and d["criteria"] == list(PARETO_CRITERIA) and d["scored"] == 162 and d["full_rank"] == 12
    )
    assert [i["design"] for i in d["items"]][0] == ["fast", "precise", "reference", "precise"] and len(d["items"]) == 3
    assert all(i["sessions_scope"] == TOKEN and i["condition_number"] > 1 for i in d["items"])
    assert _check_contract(out, True) == 3  # every count travels with the lower-bound words
    assert not _promises(p.stderr)


def test_designs_cli_plain_list_and_refusals():
    p = _run("designs", "--length", "3", "--known-write", "yes", "--top", "0")
    assert p.returncode == 0, p.stderr
    d = json.loads(p.stdout)["designs"]
    assert d["pareto"] is False and d["criteria"] is None and len(d["items"]) == 27 == d["scored"]
    assert d["full_rank"] == 0 and all(i["worst_sessions"] is None and "notes" not in i for i in d["items"])
    for args in (
        ("--length", "9"),
        ("--length", "0"),
        ("--length", "2", "--top", "-1"),
        ("--length", "2", "--free", "x"),
    ):
        bad = _run("designs", *args)
        assert bad.returncode == 2 and bad.stdout == "" and not _promises(bad.stderr), args
    text = " ".join(_run("designs", "--help").stdout.split())
    assert TOKEN in text and "lower bound" in text and not _promises(text)


def test_reproduce_design_pareto_table_and_identifiability_spectrum():
    table = design_pareto_table()
    assert table["length"] == PARETO_LENGTH == 5 and len(table["pareto"]) == 5
    assert table["d_optimal"] == table["pareto"][0] and table["sessions_scope"] == TOKEN
    assert _check_contract({"design_pareto": table}, True) == 6
    p = _run("reproduce", "--table", "identifiability")
    rows = json.loads(p.stdout)["identifiability"]
    assert [r["condition_number"] is None for r in rows] == [r["verdict"] == "NO-GO" for r in rows]
    assert rows[-1]["condition_number"] == pytest.approx(4074.03, rel=1e-4)

"""Identifiability of the calibration model (roadmap R13/P0) against an independent reference.

* Distributions: the package's forward recursion vs explicit hidden-path
  enumeration (tests/_ident_oracle.py, uniformisation instead of the closed
  form), and vs the planning model's own ``Problem.branches``.
* Fisher information: exact forward-mode derivatives vs central differences
  of the ORACLE distribution.
* Structural witnesses: re-derived in exact rational arithmetic where the
  claim is exact (hidden preparation error, label swap, back-action windows),
  each with a control that must break it.
* Rank decisions: every continuous witness lies in the reported null space;
  positive controls are full rank.
"""

import json
import math
import random
from fractions import Fraction as Fr

import pytest
from _ident_oracle import session_distribution, spec_of

from atomic_information_lab import DEFAULT_ACTIONS, TWO_HYPOTHESES, Action, Hypothesis, Problem
from atomic_information_lab.__main__ import main
from atomic_information_lab.identifiability import (
    MAX_DESIGN_LENGTH,
    RANK_TOL,
    ZERO_INFO_TOL,
    Calibration,
    Design,
    _eigh,
    design_search,
    discrimination,
    fisher_information,
    identifiability,
    outcome_distribution,
    parameter_names,
)
from atomic_information_lab.reproduce import identifiability_table

CAL = Calibration(prep_error=0.02)
EVERY = parameter_names(CAL)
ANCHORED = tuple(f for f in EVERY if f != "backaction:fast")
D6 = ("fast", "reference", "precise", "reference", "fast", "precise")


def random_calibration(rng):
    acts = tuple(
        Action(a.name, rng.uniform(0.2, 4.0), rng.uniform(0.01, 0.3), a.energy, a.reads_reference)
        for a in DEFAULT_ACTIONS
    )
    hyp = Hypothesis(rng.uniform(0.01, 0.4), rng.uniform(0.01, 0.4), rng.uniform(0.0, 0.3))
    return Calibration(hyp, acts, rng.uniform(0.0, 0.3), rng.uniform(0.1, 0.9))


def random_design(rng, n):
    return Design(tuple(rng.choice(("fast", "precise", "reference")) for _ in range(n)), rng.random() < 0.5)


def max_diff(p, q):
    return max(abs(p.get(o, 0) - q.get(o, 0)) for o in set(p) | set(q))


# ---------------------------------------------------------------- distributions
@pytest.mark.parametrize("seed", range(12))
def test_distribution_matches_path_enumeration(seed):
    rng = random.Random(seed)
    cal = random_calibration(rng)
    design = random_design(rng, rng.randint(1, 4))
    pkg = outcome_distribution(cal, design)
    ora = session_distribution(spec_of(cal), design.sequence, design.known_write)
    assert set(pkg) == set(ora)
    assert max_diff(pkg, ora) <= 1e-13
    assert math.fsum(pkg.values()) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("seq", [("fast",), ("reference", "precise"), ("fast", "reference", "precise", "fast")])
def test_distribution_matches_planning_model(seq):
    """prep_error 0 and bias_prior 1/2 is the planning model: chain Problem.branches along the sequence."""
    pr = Problem((CAL.hypothesis,), actions=DEFAULT_ACTIONS)
    leaves = {(): (1.0, pr.initial_belief())}
    for name in seq:
        nxt = {}
        for ys, (w, b) in leaves.items():
            for y, p, post in pr.branches(b, pr.action(name)):
                nxt[ys + (y,)] = (w * p, post)
        leaves = nxt
    planning = {ys: w for ys, (w, _) in leaves.items()}
    pkg = outcome_distribution(Calibration(), Design(seq))
    assert max_diff(planning, pkg) <= 1e-14


# ---------------------------------------------------------------- Fisher information
def oracle_fisher(cal, design, free, rel=1e-6):
    """Central differences of the oracle distribution; F = sum dp dp^T / p."""
    base = session_distribution(spec_of(cal), design.sequence, design.known_write)
    grads = []
    for name in free:
        v = cal.parameters()[name]
        h = rel * max(1.0, abs(v))
        up = session_distribution(spec_of(cal.with_values({name: v + h})), design.sequence, design.known_write)
        dn = session_distribution(spec_of(cal.with_values({name: v - h})), design.sequence, design.known_write)
        grads.append({o: (up.get(o, 0) - dn.get(o, 0)) / (2 * h) for o in base})
    return [[sum(gi[o] * gj[o] / base[o] for o in base if base[o] > 0) for gj in grads] for gi in grads]


@pytest.mark.parametrize("seed", range(4))
def test_fisher_matches_finite_differences_of_oracle(seed):
    rng = random.Random(100 + seed)
    cal = random_calibration(rng).with_values({"prep_error": 0.1, "bias_flip": 0.1})  # interior point
    design = random_design(rng, 3)
    pkg = fisher_information(cal, design, EVERY)
    ora = oracle_fisher(cal, design, EVERY)
    scale = max(abs(x) for row in ora for x in row)
    assert max(abs(a - b) for ra, rb in zip(pkg, ora, strict=True) for a, b in zip(ra, rb, strict=True)) <= 1e-6 * scale


def test_fisher_is_symmetric_and_positive_semidefinite():
    fim = fisher_information(CAL, Design(D6, True), EVERY)
    n = len(fim)
    assert all(fim[i][j] == fim[j][i] for i in range(n) for j in range(n))
    lam, _ = _eigh(fim)
    assert min(lam) >= -1e-12 * max(lam)


def test_jacobi_reconstructs_the_matrix():
    rng = random.Random(7)
    for n in (1, 2, 5, 11):
        a = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                a[i][j] = a[j][i] = rng.uniform(-1, 1)
        lam, v = _eigh(a)
        rec = [[sum(v[i][k] * lam[k] * v[j][k] for k in range(n)) for j in range(n)] for i in range(n)]
        assert max(abs(rec[i][j] - a[i][j]) for i in range(n) for j in range(n)) <= 1e-12
        ortho = [[sum(v[k][i] * v[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        assert max(abs(ortho[i][j] - (i == j)) for i in range(n) for j in range(n)) <= 1e-12


# ---------------------------------------------------------------- exact witnesses (rational arithmetic)
def exact_spec(eps=Fr(0), pb=Fr(1, 2)):
    return {
        "k01": None,
        "k10": None,
        "f": Fr(1, 40),
        "eps": eps,
        "pb": pb,
        "actions": {
            "fast": (None, Fr(3, 25), False),
            "precise": (None, Fr(7, 200), False),
            "reference": (None, Fr(1, 40), True),
        },
    }


def two_state_steps(pi1, lam):
    """Exact step probabilities of exp(beta*Q) with stationary P(1) = pi1 and lam = exp(-beta*(k01+k10))."""
    return ((1 - lam) * pi1, (1 - lam) * (1 - pi1))


STEPS = {"fast": two_state_steps(Fr(7, 12), Fr(9, 10)), "precise": two_state_steps(Fr(7, 12), Fr(2, 3)),
         "reference": two_state_steps(Fr(7, 12), Fr(19, 20))}  # fmt: skip


def test_exact_prep_error_hidden_without_known_writes():
    seq = ("fast", "reference", "precise")
    a = session_distribution(exact_spec(eps=Fr(0)), seq, False, STEPS)
    b = session_distribution(exact_spec(eps=Fr(3, 10)), seq, False, STEPS)
    assert a == b  # exact equality of rationals
    ka = session_distribution(exact_spec(eps=Fr(0)), seq, True, STEPS)
    kb = session_distribution(exact_spec(eps=Fr(3, 10)), seq, True, STEPS)
    assert ka != kb  # control: recording the written bit reveals the preparation error


def test_exact_label_swap_without_reference():
    swapped = {a: (p10, p01) for a, (p01, p10) in STEPS.items()}
    seq = ("fast", "precise", "fast")
    assert session_distribution(exact_spec(), seq, False, STEPS) == session_distribution(
        exact_spec(), seq, False, swapped
    )
    ref = ("fast", "reference", "fast")
    assert session_distribution(exact_spec(), ref, False, STEPS) != session_distribution(
        exact_spec(), ref, False, swapped
    )
    # with a biased start the swap needs pb -> 1 - pb
    sp, sq = exact_spec(pb=Fr(1, 5)), exact_spec(pb=Fr(4, 5))
    assert session_distribution(sp, seq, False, STEPS) == session_distribution(sq, seq, False, swapped)


def test_exact_backaction_window():
    """Reference then precise: only lam_ref * lam_prec (= exp(-(b_ref + b_prec) s)) is visible."""
    moved = dict(STEPS)
    moved["reference"] = two_state_steps(Fr(7, 12), Fr(5, 6))
    moved["precise"] = two_state_steps(Fr(7, 12), Fr(19, 20) * Fr(2, 3) / Fr(5, 6))  # same product
    win = ("fast", "reference", "precise", "fast")
    assert session_distribution(exact_spec(), win, True, STEPS) == session_distribution(exact_spec(), win, True, moved)
    ctrl = ("fast", "reference", "fast", "precise")  # reference now shares its window with the anchor
    assert session_distribution(exact_spec(), ctrl, True, STEPS) != session_distribution(
        exact_spec(), ctrl, True, moved
    )


# ---------------------------------------------------------------- reports: witnesses vs rank
def witness_direction(rep, w):
    """Tangent of the witness family at the reported point (the scale family is a curve, the others lines)."""
    vals = dict(zip(rep.free, rep.values, strict=True))
    other = dict(w.other)
    if w.name == "drift_backaction_scale":  # (c k, beta / c): tangent (k, -beta) at c = 1
        return [vals[f] if f in ("k01", "k10") else -vals[f] if f in other else 0.0 for f in rep.free]
    return [other.get(f, vals[f]) - vals[f] for f in rep.free]


def in_null_span(rep, d, tol=1e-6):
    """Residual of d after projection on span(null_directions), relative to |d|."""
    basis = []
    for u in rep.null_directions:
        u = list(u)
        for b in basis:
            c = sum(x * y for x, y in zip(u, b, strict=True))
            u = [x - c * y for x, y in zip(u, b, strict=True)]
        nu = math.sqrt(sum(x * x for x in u))
        if nu > 1e-12:
            basis.append([x / nu for x in u])
    r = list(d)
    for b in basis:
        c = sum(x * y for x, y in zip(r, b, strict=True))
        r = [x - c * y for x, y in zip(r, b, strict=True)]
    return math.sqrt(sum(x * x for x in r)) <= tol * math.sqrt(sum(x * x for x in d))


@pytest.mark.parametrize(
    "design, free, expected",
    [
        (Design(D6), EVERY, {"drift_backaction_scale", "prep_error_hidden"}),
        (Design(D6, True), EVERY, {"drift_backaction_scale"}),
        (Design(D6), ANCHORED, {"prep_error_hidden"}),
        (Design(("fast", "reference", "precise") * 2, True), ANCHORED, {"backaction_window"}),
    ],
)
def test_no_go_witnesses_are_null_directions(design, free, expected):
    rep = identifiability(CAL, design, free)
    assert rep.verdict == "NO-GO" and not rep.locally_identifiable
    assert {w.name for w in rep.witnesses} == expected
    for w in rep.witnesses:
        assert w.max_abs_difference <= 1e-13
        assert in_null_span(rep, witness_direction(rep, w)), w.name
    assert all(x is None for x in rep.sessions_for_target)


def test_go_positive_controls_full_rank():
    for design in (Design(D6, True), Design(("fast", "precise", "reference", "precise"), True)):
        rep = identifiability(CAL, design, ANCHORED)
        assert rep.verdict == "GO" and rep.rank == len(ANCHORED) and not rep.witnesses
        assert rep.min_normalised_eigenvalue > 1e3 * RANK_TOL
        assert all(isinstance(s, int) and s >= 1 for s in rep.sessions_for_target)


def test_label_swap_is_global_not_local():
    free = ("k01", "k10", "bias_flip", "error:fast", "error:precise", "backaction:precise")
    rep = identifiability(CAL, Design(("fast", "precise") * 2), free)
    assert rep.locally_identifiable and rep.verdict == "AMBIGUOUS"
    assert [w.name for w in rep.witnesses] == ["label_swap"]
    # control: one reference read removes the ambiguity
    assert identifiability(CAL, Design(("fast", "reference", "precise", "fast")), free).verdict == "GO"


def test_zero_information_for_unused_action_and_trailing_reference():
    rep = identifiability(CAL, Design(("fast", "fast"), True), ("k01", "backaction:precise"))
    assert rep.zero_information == ("backaction:precise",)
    rep = identifiability(CAL, Design(("fast", "reference"), True), ("k01", "backaction:reference"))
    assert rep.zero_information == ("backaction:reference",)  # nothing reads x after it
    assert {w.name for w in rep.witnesses} == {"backaction_window"}


def test_relative_crlb_is_the_inverse_fisher_diagonal():
    rep = identifiability(CAL, Design(D6, True), ANCHORED)
    assert rep.verdict == "GO"  # fail by assertion, not inside the elimination
    n = len(rep.free)
    a = [[Fr(x) for x in row] + [Fr(int(i == j)) for j in range(n)] for i, row in enumerate(rep.fisher)]
    for c in range(n):  # exact Gauss-Jordan on the float matrix
        p = max(range(c, n), key=lambda r: abs(a[r][c]))
        a[c], a[p] = a[p], a[c]
        a[c] = [x / a[c][c] for x in a[c]]
        for r in range(n):
            if r != c and a[r][c]:
                a[r] = [x - a[r][c] * y for x, y in zip(a[r], a[c], strict=True)]
    for i in range(n):
        want = math.sqrt(float(a[i][n + i])) / abs(rep.values[i])
        assert rep.relative_crlb[i] == pytest.approx(want, rel=1e-7)
        assert rep.sessions_for_target[i] == max(1, math.ceil((rep.relative_crlb[i] / 0.1) ** 2))


def test_crlb_scales_with_target():
    a = identifiability(CAL, Design(D6, True), ANCHORED, target_rse=0.1)
    b = identifiability(CAL, Design(D6, True), ANCHORED, target_rse=0.2)
    assert a.verdict == b.verdict == "GO"
    for sa, sb in zip(a.sessions_for_target, b.sessions_for_target, strict=True):
        assert sb <= sa and sb >= sa // 4 - 1


# ---------------------------------------------------------------- discrimination and design search
def test_discrimination_controls():
    same = discrimination(CAL, CAL, Design(D6))
    assert same.kl_ab == 0 and same.chernoff == 0 and same.sessions_for_error is None
    a, b = Calibration(TWO_HYPOTHESES[0]), Calibration(TWO_HYPOTHESES[1])
    one = discrimination(a, b, Design(("precise",)))
    assert one.chernoff <= 1e-15 and one.sessions_for_error is None  # one reading is uniform under both
    four = discrimination(a, b, Design(("precise",) * 4))
    assert 0 < four.chernoff <= min(four.kl_ab, four.kl_ba)
    p, q = outcome_distribution(a, four.design), outcome_distribution(b, four.design)
    grid = min(math.log(sum(p[o] ** t * q[o] ** (1 - t) for o in p)) for t in [i / 2000 for i in range(1, 2000)])
    assert four.chernoff == pytest.approx(-grid, rel=1e-6)
    assert four.sessions_for_error == math.ceil(math.log(1 / (2 * 0.01)) / four.chernoff)
    assert four.sessions_for_error == 68  # README


def test_discrimination_label_swap_pair():
    swapped = CAL.with_values({"k01": CAL.hypothesis.k10, "k10": CAL.hypothesis.k01})
    assert discrimination(CAL, swapped, Design(("fast", "precise", "fast"))).chernoff <= 1e-15
    assert discrimination(CAL, swapped, Design(("fast", "reference", "fast"))).chernoff > 1e-6


def test_design_search_order_and_minimum_length():
    assert not [s for s in design_search(CAL, ANCHORED, 3) if s.verdict == "GO"]  # 3 steps never suffice
    res = design_search(CAL, ANCHORED, 4)
    assert len(res) == 3**4 * 2
    go = [s for s in res if s.verdict == "GO"]
    assert go and res[: len(go)] == go and all(s.design.known_write for s in go)
    assert len(go) == 12  # FINDINGS N-20
    assert res[0].design == Design(("fast", "precise", "reference", "precise"), True)
    assert [s.log_det for s in go] == sorted((s.log_det for s in go), reverse=True)


def test_design_search_refuses_explosion():
    with pytest.raises(ValueError, match="MAX_SEARCH_DESIGNS"):
        design_search(CAL, ANCHORED, 7)


# ---------------------------------------------------------------- fail-closed
@pytest.mark.parametrize(
    "call",
    [
        lambda: identifiability(CAL, Design(D6), ("nope",)),
        lambda: identifiability(CAL, Design(D6), "k01"),
        lambda: identifiability(CAL, Design(D6), ("k01", "k01")),
        lambda: identifiability(CAL, Design(D6), ()),
        lambda: identifiability(CAL, Design(D6), ("k01",), target_rse=0),
        lambda: identifiability(CAL, Design(("fast", "bogus")), ("k01",)),
        lambda: Design(()),
        lambda: Design("fast"),
        lambda: Design(("fast",) * (MAX_DESIGN_LENGTH + 1)),
        lambda: Design(("fast",), known_write=1),
        lambda: Calibration(prep_error=0.6),
        lambda: Calibration(prep_error=float("nan")),
        lambda: Calibration(bias_prior=1.5),
        lambda: Calibration(bias_prior=True),
        lambda: Calibration(actions=()),
        lambda: CAL.with_values({"nope": 1.0}),
        lambda: CAL.with_values({"error:fast": 0.7}),
        lambda: identifiability(CAL.with_values({"k01": 1e308}), Design(("precise",)), ("k01",)),
        lambda: discrimination(CAL, CAL, Design(D6), alpha=0.5),
        lambda: discrimination(CAL, Calibration(actions=DEFAULT_ACTIONS[:2]), Design(("fast",))),
        lambda: design_search(CAL, ANCHORED, 0),
        lambda: design_search(CAL, ANCHORED, 2, known_write=(True, True)),
    ],
)
def test_fail_closed(call):
    with pytest.raises(ValueError):
        call()


def test_boundary_error_zero_has_no_fisher_information():
    cal = CAL.with_values({"error:reference": 0.0, "bias_flip": 0.0})
    with pytest.raises(ArithmeticError, match="boundary"):
        fisher_information(cal, Design(("reference", "reference")), ("error:reference",))


# ---------------------------------------------------------------- reproduction, README, CLI
README_ROWS = [  # verdict, rank, free, witnesses, worst parameter and sessions for 10 % relative SE
    ("NO-GO", 9, 11, ["drift_backaction_scale", "prep_error_hidden"], None),
    ("NO-GO", 10, 11, ["drift_backaction_scale"], None),
    ("NO-GO", 9, 10, ["prep_error_hidden"], None),
    ("NO-GO", 9, 10, ["backaction_window"], None),
    ("GO", 10, 10, [], ["prep_error", 373950]),
    ("AMBIGUOUS", 6, 6, ["label_swap"], ["bias_flip", 1293272944]),
    ("GO", 10, 10, [], ["prep_error", 6687058]),
]


def test_identifiability_table_pinned():
    rows = identifiability_table()
    got = [(r["verdict"], r["rank"], r["free"], r["witnesses"], r["worst_parameter_sessions_for_10pct"]) for r in rows]
    assert got == README_ROWS


def test_readme_identifiability_table_matches_code():
    text = (__import__("pathlib").Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    for row in identifiability_table():
        worst = row["worst_parameter_sessions_for_10pct"]
        cell = "—" if worst is None else f"{worst[1]:,}".replace(",", " ") + f" (`{worst[0]}`)"
        line = f"| {row['scenario']} | {row['verdict']} | {row['rank']}/{row['free']} |"
        assert line in text, line
        assert cell in text, cell


def test_cli_identify(capsys):
    assert main(["identify", "--design", "fast,precise,reference,precise", "--known-write"]) == 0
    rep = json.loads(capsys.readouterr().out)["identifiability"]
    assert rep["verdict"] == "GO" and rep["rank"] == 10
    assert main(["identify", "--design", ",".join(D6), "--free", "all"]) == 0
    assert json.loads(capsys.readouterr().out)["identifiability"]["verdict"] == "NO-GO"


@pytest.mark.parametrize(
    "argv",
    [
        ["identify", "--design", "fast,bogus"],
        ["identify", "--design", "fast", "--free", "k01,nope"],
        ["identify", "--design", "fast", "--prep-error", "0.9"],
        ["identify", "--design", "fast", "--target-rse", "-1"],
    ],
)
def test_cli_identify_fail_closed(argv, capsys):
    assert main(argv) == 2
    assert "horizon" not in capsys.readouterr().err


def test_zero_information_threshold_has_a_wide_gap():
    """The claim next to ZERO_INFO_TOL: all designs of length 1..5, scaled diagonal far from the threshold."""
    from itertools import product

    v = CAL.parameters()
    dead, live = 0.0, math.inf
    for n in range(1, 6):
        for kw in (False, True):
            for seq in product(("fast", "precise", "reference"), repeat=n):
                fim = fisher_information(CAL, Design(seq, kw), EVERY)
                for i, f in enumerate(EVERY):
                    x = fim[i][i] * (v[f] or 1.0) ** 2
                    if x < ZERO_INFO_TOL:
                        dead = max(dead, x)
                    else:
                        live = min(live, x)
    assert dead < 1e-32 and live > 1e-7

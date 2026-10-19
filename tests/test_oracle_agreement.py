"""Package vs independent oracle (tests/_oracle.py shares no code with the package).

Oracle A: exact rational DP over unnormalised forward messages, CTMC step by
uniformisation. Oracle B: brute force over every deterministic policy with
explicit hidden-path enumeration.
"""

import math
import pathlib
import re
from itertools import product

import _oracle as O
import pytest

from atomic_information_lab import (
    DEFAULT_ACTIONS,
    TWO_HYPOTHESES,
    Action,
    Hypothesis,
    Objective,
    Problem,
    evaluate_policy,
    greedy_policy,
    nominal_policy,
    open_loop_value,
    optimal_policy,
    solve,
)

TOL = 1e-12


def make(hyps=((1.0, 0.07, 0.045, 0.025),), actions=O.DEFAULT_ACTIONS, lam=0.012, alpha=0.5, beta=0.0):
    """Build the package problem and the oracle model from the same raw numbers."""
    pkg = Problem(
        tuple(Hypothesis(k01, k10, f) for _, k01, k10, f in hyps),
        tuple(w for w, *_ in hyps),
        tuple(Action(n, ba, e, en, ref) for n, ba, e, en, ref in actions),
        Objective(alpha, lam, beta),
    )
    return pkg, (lambda exact: O.Model(hyps, actions, lam, alpha, beta, exact=exact))


DET = tuple((n, ba, 0.0, en, ref) for n, ba, _, en, ref in O.DEFAULT_ACTIONS)
CASES = {
    "default": {},
    "lambda0": {"lam": 0.0},
    "alpha0": {"alpha": 0.0},
    "alpha1": {"alpha": 1.0},
    "flip0": {"hyps": ((1.0, 0.07, 0.045, 0.0),)},
    "flip_half": {"hyps": ((1.0, 0.07, 0.045, 0.5),)},
    "static": {"hyps": ((1.0, 0.0, 0.0, 0.025),)},
    "symmetric": {"hyps": ((1.0, 0.05, 0.05, 0.025),)},
    "fast_dynamics": {"hyps": ((1.0, 3.0, 2.0, 0.025),)},
    "deterministic_sensors": {"actions": DET},
    "retention": {"beta": 0.3},
    "two_models": {"hyps": ((0.5, 0.07, 0.045, 0.025), (0.5, 0.23, 0.16, 0.10))},
    "three_models": {"hyps": ((0.2, 0.07, 0.045, 0.025), (0.3, 0.23, 0.16, 0.10), (0.5, 0.0, 0.3, 0.2))},
}


@pytest.mark.parametrize("name", sorted(CASES))
@pytest.mark.parametrize("allow_stop", [True, False])
def test_exact_rational_oracle_h_le_2(name, allow_stop):
    pkg, orc = make(**CASES[name])
    m = orc(True)
    for h in range(3):
        expected = float(m.value(m.root(), h, allow_stop=allow_stop))
        assert solve(pkg, horizon=h, allow_stop=allow_stop).value == pytest.approx(expected, abs=TOL)


@pytest.mark.parametrize("name", ["default", "lambda0", "deterministic_sensors", "two_models"])
def test_exact_rational_oracle_h3(name):
    pkg, orc = make(**CASES[name])
    m = orc(True)
    assert solve(pkg, horizon=3).value == pytest.approx(float(m.value(m.root(), 3)), abs=TOL)


@pytest.mark.parametrize("name", sorted(CASES))
def test_float_oracle_h4(name):
    pkg, orc = make(**CASES[name])
    m = orc(False)
    for allow_stop in (True, False):
        assert solve(pkg, horizon=4, allow_stop=allow_stop).value == pytest.approx(
            m.value(m.root(), 4, allow_stop=allow_stop), abs=TOL
        )


def test_float_oracle_h5_default():
    pkg, orc = make()
    m = orc(False)
    assert solve(pkg, horizon=5).value == pytest.approx(m.value(m.root(), 5), abs=TOL)
    assert solve(pkg, horizon=5, allow_stop=False).value == pytest.approx(
        m.value(m.root(), 5, allow_stop=False), abs=TOL
    )


@pytest.mark.parametrize("name", ["default", "deterministic_sensors", "two_models", "flip_half"])
@pytest.mark.parametrize("allow_stop", [True, False])
def test_brute_force_over_all_policies(name, allow_stop):
    """Bellman minimum == minimum over every enumerated deterministic policy (h <= 3)."""
    case = {"hyps": ((1.0, 0.07, 0.045, 0.025),), "actions": O.DEFAULT_ACTIONS, "lam": 0.012, "alpha": 0.5}
    case.update(CASES[name])
    pkg, _ = make(**case)
    expected_counts = {0: 1, 1: 4, 2: 49, 3: 7204} if allow_stop else {0: 1, 1: 3, 2: 27, 3: 2187}
    for h in range(4):
        best, n = O.brute_force_all_policies(
            case["hyps"], case["actions"], h, case["lam"], case["alpha"], allow_stop=allow_stop
        )
        assert n == expected_counts[h]
        assert solve(pkg, horizon=h, allow_stop=allow_stop).value == pytest.approx(best, abs=TOL)


@pytest.mark.parametrize("name", ["default", "two_models", "retention"])
def test_open_loop_sequences_exact(name):
    pkg, orc = make(**CASES[name])
    m = orc(True)
    names = [a[0] for a in m.actions]
    for n in range(4):
        for seq in product(names, repeat=n):
            acts = [next(a for a in m.actions if a[0] == s) for s in seq]
            assert open_loop_value(pkg, seq) == pytest.approx(float(m.open_loop(acts)), abs=TOL)


@pytest.mark.parametrize("name", ["default", "lambda0", "two_models", "deterministic_sensors"])
def test_greedy_and_expected_steps(name):
    pkg, orc = make(**CASES[name])
    m = orc(False)
    for h in range(5):
        g, _ = m.policy_value(m.root(), h, m.greedy_policy())
        assert evaluate_policy(pkg, greedy_policy(pkg), h).value == pytest.approx(g, abs=TOL)
        v, steps = m.policy_value(m.root(), h, m.optimal_policy())
        stats = evaluate_policy(pkg, optimal_policy(pkg), h)
        assert stats.value == pytest.approx(v, abs=1e-10)
        assert stats.expected_steps == pytest.approx(steps, abs=1e-9)


def test_nominal_policy_matches_oracle():
    hyps = ((0.5, 0.07, 0.045, 0.025), (0.5, 0.23, 0.16, 0.10))
    pkg, orc = make(hyps=hyps)
    nominal_pkg, nominal_orc = make(hyps=((1.0, 0.07, 0.045, 0.025),))
    E, N = orc(True), nominal_orc(True)
    opt = N.optimal_policy()

    def pol(m, hist, h):
        nm = N.root()
        for name, y in hist:
            nm = N.step(nm, next(a for a in N.actions if a[0] == name), y)
        return opt(nm, (), h)

    for h in range(4):
        expected, _ = E.policy_value(E.root(), h, pol)
        assert evaluate_policy(pkg, nominal_policy(nominal_pkg), h).value == pytest.approx(float(expected), abs=TOL)


def test_terminal_loss_is_expected_brier_of_posterior_forecasts():
    """L(b) = alpha*p0(1-p0)+(1-alpha)*pt(1-pt) equals the expected Brier score of the
    posterior-mean forecasts against the TRUE hidden values (path enumeration)."""
    pkg, _ = make(lam=0.0)
    for seq in [("fast",), ("reference", "precise"), ("fast", "reference", "fast")]:
        acts = [next(a for a in O.DEFAULT_ACTIONS if a[0] == s) for s in seq]
        _, brier = O.hidden_path_table(((1.0, 0.07, 0.045, 0.025),), O.DEFAULT_ACTIONS, acts)
        path_brier = sum(0.5 * b0 + 0.5 * bt for b0, bt in brier.values())
        assert open_loop_value(pkg, seq) == pytest.approx(path_brier, abs=TOL)


# ----------------------------------------------------------- oracle self-checks
def test_oracle_uniformisation_matches_closed_form():
    for k01, k10 in [(0.07, 0.045), (0.245, 0.1575), (2.0, 0.0), (0.0, 0.3), (5.0, 7.0)]:
        p01, p10 = O.ctmc_uniformised(k01, k10)
        s = k01 + k10
        assert p01 == pytest.approx(k01 / s * (1 - math.exp(-s)), abs=1e-14)
        assert p10 == pytest.approx(k10 / s * (1 - math.exp(-s)), abs=1e-14)


def test_oracle_refuses_its_own_underflow_domain():
    O.ctmc_uniformised(100.0, 50.0)
    with pytest.raises(ValueError):
        O.ctmc_uniformised(1000.0, 1.0)


def test_oracle_does_not_import_the_package():
    src = pathlib.Path(O.__file__).read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+atomic_information_lab", src, re.M)
    assert re.search(r"^\s*from fractions import", src, re.M)  # positive control: the scan sees imports


def test_package_defaults_match_oracle_transcription():
    assert (
        tuple((a.name, a.backaction, a.error, a.energy, a.reads_reference) for a in DEFAULT_ACTIONS)
        == O.DEFAULT_ACTIONS
    )
    assert tuple((h.k01, h.k10, h.bias_flip) for h in TWO_HYPOTHESES) == ((0.07, 0.045, 0.025), (0.23, 0.16, 0.10))


@pytest.mark.parametrize("name", ["default", "fast_dynamics", "two_models", "retention"])
def test_posterior_marginals_match_oracle(name):
    """Values are invariant under relabelling x <-> 1-x (uniform prior, symmetric loss), so a
    value-only comparison cannot see swapped rates. Compare the posteriors themselves."""
    pkg, orc = make(**CASES[name])
    m = orc(True)
    for n in range(3):
        for seq in product([a[0] for a in m.actions], repeat=n):
            for ys in product((0, 1), repeat=n):
                fm, b = m.root(), pkg.initial_belief()
                for s_, y in zip(seq, ys, strict=True):
                    fm = m.step(fm, next(a for a in m.actions if a[0] == s_), y)
                    match = [post for yy, _, post in pkg.branches(b, pkg.action(s_)) if yy == y]
                    b = match[0] if match else None
                    if b is None:
                        break
                tot = sum(fm)
                if b is None or tot == 0:
                    assert b is None and tot == 0
                    continue
                got = pkg.marginals(b)
                p0 = sum(v for v, st in zip(fm, m.states, strict=True) if st[1] == 1) / tot
                pt = sum(v for v, st in zip(fm, m.states, strict=True) if st[2] == 1) / tot
                assert got["p_initial_one"] == pytest.approx(float(p0), abs=TOL)
                assert got["p_current_one"] == pytest.approx(float(pt), abs=TOL)
                theta = [
                    sum(v for v, st in zip(fm, m.states, strict=True) if st[0] == k) / tot
                    for k in range(len(m.components))
                ]
                assert pkg.model_posterior(b) == pytest.approx([float(t) for t in theta], abs=TOL)

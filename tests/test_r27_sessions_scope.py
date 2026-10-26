"""AIL-R27-01: every session count from the Fisher information is a lower bound, and says so.

The Cramér–Rao bound limits the variance of unbiased estimators under regularity conditions.
No estimator is shown to reach it, so ``sessions_for_target`` is necessary, never sufficient,
for the target precision. The scope travels with the count through the report, the design
search, the CLI JSON and the reproduce table.
"""

import json
from fractions import Fraction as Fr

from test_identifiability import ANCHORED, CAL

from atomic_information_lab.__main__ import main
from atomic_information_lab.identifiability import SESSIONS_SCOPE, Design, design_search, identifiability
from atomic_information_lab.reproduce import identifiability_table


def test_sessions_scope_names_a_lower_bound():
    assert SESSIONS_SCOPE == "CRLB_NECESSARY_LOWER_BOUND_ONLY"


def test_report_and_design_search_carry_sessions_scope():
    rep = identifiability(CAL, Design(("fast", "precise", "reference", "precise"), True), ANCHORED)
    assert rep.verdict == "GO" and None not in rep.sessions_for_target
    assert rep.sessions_scope == SESSIONS_SCOPE
    scores = design_search(CAL, ANCHORED, 4)
    assert any(s.worst_sessions is not None for s in scores)
    assert all(s.sessions_scope == SESSIONS_SCOPE for s in scores)


def test_cli_json_and_table_carry_sessions_scope(capsys):
    assert main(["identify", "--design", "fast,precise,reference,precise", "--known-write"]) == 0
    rep = json.loads(capsys.readouterr().out)["identifiability"]
    assert rep.get("sessions_scope") == SESSIONS_SCOPE
    rows = identifiability_table()
    assert rows and all(r.get("sessions_scope") == SESSIONS_SCOPE for r in rows)


def test_crlb_is_not_reached_by_a_natural_estimator_exact():
    """Independent oracle (R27): the bound is necessary, not sufficient.

    Two Bernoulli(p) readings, p = 1/2, target theta = p^2, plug-in estimator (k/2)^2. Exact
    rational arithmetic, no use of the module. Its mean squared error exceeds the Cramér–Rao
    bound, so a session count computed from the bound would promise a precision this estimator
    does not deliver.
    """
    p, n = Fr(1, 2), 2
    law = {0: (1 - p) ** 2, 1: 2 * p * (1 - p), 2: p**2}
    theta = p**2
    mse = sum(w * ((Fr(k, n)) ** 2 - theta) ** 2 for k, w in law.items())
    fisher = n / (p * (1 - p))
    crlb = (2 * p) ** 2 / fisher
    assert sum(law.values()) == 1
    assert (mse, crlb) == (Fr(5, 32), Fr(1, 8))
    assert mse > crlb
    bias = sum(w * (Fr(k, n)) ** 2 for k, w in law.items()) - theta
    assert bias == Fr(1, 8)  # control: the estimator is biased, the bound's precondition fails

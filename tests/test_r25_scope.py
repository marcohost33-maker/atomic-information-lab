"""AIL-R25-01: every identifiability verdict carries a machine-readable scope.

A full Fisher rank supports local identifiability only (Rothenberg 1971); the witness list is
not exhaustive. A consumer that reads ``"GO"`` alone can mistake it for global identifiability,
so the scope travels with the verdict through the report, the design search, the CLI JSON and
the README table.
"""

import json
from fractions import Fraction as Fr

from test_identifiability import ANCHORED, CAL, D6

from atomic_information_lab.__main__ import main
from atomic_information_lab.identifiability import SCOPE, Design, design_search, identifiability
from atomic_information_lab.reproduce import identifiability_table

LABEL_SWAP_FREE = ("k01", "k10", "bias_flip", "error:fast", "error:precise", "backaction:precise")


def test_scope_table_is_exhaustive_and_never_claims_global():
    assert set(SCOPE) == {"GO", "AMBIGUOUS", "NO-GO"}
    assert len(set(SCOPE.values())) == 3
    for scope in SCOPE.values():
        assert "GLOBAL_UNPROVEN" in scope or "GLOBAL_REFUTED" in scope or scope.startswith("NOT_LOCALLY")


def test_report_scope_follows_verdict():
    go = identifiability(CAL, Design(("fast", "reference", "precise", "fast")), LABEL_SWAP_FREE)
    amb = identifiability(CAL, Design(("fast", "precise") * 2), LABEL_SWAP_FREE)
    nogo = identifiability(CAL, Design(D6), [n for n in CAL.parameters()])
    assert (go.verdict, go.scope) == ("GO", "LOCAL_ONLY_GLOBAL_UNPROVEN")
    assert (amb.verdict, amb.scope) == ("AMBIGUOUS", "LOCAL_ONLY_GLOBAL_REFUTED_BY_WITNESS")
    assert (nogo.verdict, nogo.scope) == ("NO-GO", "NOT_LOCALLY_IDENTIFIABLE")


def test_design_search_scores_carry_scope():
    scores = design_search(CAL, ANCHORED, 4)
    assert {s.verdict for s in scores} >= {"GO"}
    assert all(s.scope == SCOPE[s.verdict] for s in scores)


def test_cli_json_and_table_carry_scope(capsys):
    assert main(["identify", "--design", "fast,precise,reference,precise", "--known-write"]) == 0
    rep = json.loads(capsys.readouterr().out)["identifiability"]
    assert rep["verdict"] == "GO" and rep.get("scope") == "LOCAL_ONLY_GLOBAL_UNPROVEN"
    rows = identifiability_table()
    assert rows and all(r.get("scope") == SCOPE[r["verdict"]] for r in rows)


def test_product_only_model_is_globally_not_identifiable_exact():
    """Independent oracle (R25): observations that depend on k*beta only cannot separate k and beta.

    Exact rational arithmetic, no use of the module: two distinct parameter pairs with the same
    product give identical outcome laws, although each one alone has a non-zero derivative --
    the textbook case where local reasoning per coordinate says nothing about global
    identifiability.
    """

    def law(k, beta):
        p = k * beta
        return (p, 1 - p)

    a, b = (Fr(1, 2), Fr(1, 3)), (Fr(1, 3), Fr(1, 2))
    assert a != b and law(*a) == law(*b)
    assert law(Fr(1, 2), Fr(1, 3)) != law(Fr(1, 2), Fr(1, 4))  # control: the law does depend on beta

"""AIL-R28 Q2: the CRLB scope reaches every consumer of a session count, end to end.

R27 put ``sessions_scope = CRLB_NECESSARY_LOWER_BOUND_ONLY`` on the report, the design search, the
CLI JSON and the reproduce table, checked by reading the field name once. R28 asks for more: a
field name read back is not the consumer chain. These tests run the installed entry point as a
real process (``python -m atomic_information_lab``), parse what a user or a downstream program
actually receives, and check two invariants:

* **structure** -- every JSON object that carries a session count (any key containing
  ``sessions``, or ``crlb``) also carries ``sessions_scope`` with the literal lower-bound token;
* **text** -- every object with at least one numeric count also says in words that it is a lower
  bound, necessary and not sufficient; no string anywhere in the output, the help text or the
  error stream promises that a number of sessions is required, needed, enough or guaranteed.

Positive controls make sure no check is vacuous: the walker must find sites, and at least one of
them must hold a real number.
"""

import json
import os
import pathlib
import re
import subprocess
import sys

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
ENV = {**os.environ, "PYTHONPATH": str(SRC), "PYTHONIOENCODING": "utf-8"}
# Literal on purpose: comparing against the imported constant would follow a mutation of it.
TOKEN = "CRLB_NECESSARY_LOWER_BOUND_ONLY"
COUNT_KEY = re.compile(r"sessions|crlb")
# Promise phrasing about a number of sessions (English and German). "required" alone is not on the
# list: "sessions with known writes are required" is a true statement about the design, not a count.
PROMISE = [
    re.compile(p, re.IGNORECASE)
    for p in (
        r"\bsessions?\s+(?:are\s+|is\s+)?(?:required|needed|suffice|sufficient|enough)\b",
        r"\b(?:required|needed|enough|sufficient)\s+(?:number\s+of\s+)?sessions?\b",
        r"\bsessions?\s+(?:will\s+)?(?:guarantee|achieve|reach|deliver)",
        r"\bguarantee",
        r"erforderlich",
        r"ben[oö]tig",
        r"\breich(?:t|en)\b",
        r"garantier",
    )
]
LOWER_BOUND = (re.compile(r"lower bound", re.IGNORECASE), re.compile(r"necessary, not sufficient", re.IGNORECASE))


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "atomic_information_lab", *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=ENV,
        timeout=600,
    )


def _sites(obj, path="$"):
    """Yield (path, dict) for every JSON object holding a session-count key."""
    if isinstance(obj, dict):
        if any(COUNT_KEY.search(k) and k != "sessions_scope" for k in obj):
            yield path, obj
        for k, v in obj.items():
            yield from _sites(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _sites(v, f"{path}[{i}]")


def _numbers(v) -> list:
    if isinstance(v, bool) or v is None or isinstance(v, str):
        return []
    if isinstance(v, int | float):
        return [v]
    if isinstance(v, dict):
        v = list(v.values())
    if isinstance(v, list):
        return [x for e in v for x in _numbers(e)]
    return []


def _strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for k, v in obj.items() for s in (k, *_strings(v))]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def _promises(text: str) -> list[str]:
    return [m.group(0) for p in PROMISE for m in p.finditer(text)]


def _check_contract(out: dict, expect_counts: bool) -> int:
    sites = list(_sites(out))
    assert sites, "positive control: no session count in the output at all"
    counted = 0
    for path, site in sites:
        scope = site.get("sessions_scope")
        assert scope == TOKEN, f"{path}: sessions_scope missing or wrong: {scope!r}"
        has_count = any(_numbers(v) for k, v in site.items() if COUNT_KEY.search(k) and k != "sessions_scope")
        if has_count:
            counted += 1
            words = " ".join(_strings(site.get("notes", [])))
            for pat in LOWER_BOUND:
                assert pat.search(words), f"{path}: a session count without the words {pat.pattern!r} in notes"
    if expect_counts:
        assert counted, "positive control: no site with a numeric session count"
    found = [hit for s in _strings(out) for hit in _promises(s)]
    assert not found, f"promise phrasing in the output: {found}"
    return counted


TARGET_ONLY = "k01,k10,bias_flip,error:fast,error:precise,backaction:precise"


@pytest.mark.parametrize(
    ("design", "known", "free", "verdict", "counts"),
    [
        ("fast,precise,reference,precise", True, "anchored", "GO", True),
        ("fast,precise,fast,precise", False, TARGET_ONLY, "AMBIGUOUS", True),
        ("fast,reference,precise,fast,reference,precise", True, "anchored", "NO-GO", False),
    ],
)
def test_identify_cli_json_contract(design, known, free, verdict, counts):
    p = _run("identify", "--design", design, "--free", free, *(["--known-write"] if known else []))
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    assert out["identifiability"]["verdict"] == verdict
    assert _check_contract(out, counts) == (1 if counts else 0)
    assert not _promises(p.stderr)


@pytest.mark.parametrize("table", ["identifiability", "all"])
def test_reproduce_cli_json_contract(table):
    args = ("reproduce", "--table", table) + (("--max-horizon", "1") if table == "all" else ())
    p = _run(*args)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout)
    rows = out["identifiability"]
    assert len(rows) == 7
    # three scenarios of the README table have a numeric worst-parameter count
    assert _check_contract(out, True) == sum(r["worst_parameter_sessions_for_10pct"] is not None for r in rows) == 3


def test_identify_help_text_names_the_lower_bound():
    p = _run("identify", "--help")
    assert p.returncode == 0, p.stderr
    text = " ".join(p.stdout.split())  # argparse wraps lines
    assert TOKEN in text
    for pat in LOWER_BOUND:
        assert pat.search(text), pat.pattern
    assert not _promises(text), _promises(text)


@pytest.mark.parametrize("rse", ["0", "-0.1", "nan"])
def test_identify_rejects_bad_target_without_a_count(rse):
    """Silent-failure edge: an invalid precision target yields exit 2 and no count on stdout."""
    p = _run("identify", "--design", "fast,precise,reference,precise", "--known-write", "--target-rse", rse)
    assert p.returncode == 2
    assert p.stdout == ""
    assert not _promises(p.stderr)


def test_walker_and_wordlist_catch_what_they_are_for():
    """Self-check of the contract helpers on hand-made output (the tool against its own class)."""
    note = "lower bound; necessary, not sufficient"
    good = {"a": {"sessions_for_target": [5, None], "sessions_scope": TOKEN, "notes": [note]}}
    assert _check_contract(good, True) == 1
    with pytest.raises(AssertionError, match="sessions_scope"):
        _check_contract({"a": {"worst_sessions": 5}}, True)
    with pytest.raises(AssertionError, match="notes"):
        _check_contract({"a": {"worst_sessions": 5, "sessions_scope": TOKEN}}, True)
    with pytest.raises(AssertionError, match="promise"):
        bad = {**good["a"], "text": "374 000 sessions are required for 10 %"}
        _check_contract({"a": bad}, True)
    with pytest.raises(AssertionError, match="positive control"):
        _check_contract({"a": {"sessions_for_target": [None], "sessions_scope": TOKEN}}, True)
    for promise in ("68 sessions needed", "Sitzungen erforderlich", "das reicht", "enough sessions", "guaranteed"):
        assert _promises(promise), promise
    for fine in ("sessions with known writes are required", "at least 374 000 sessions (lower bound)"):
        assert not _promises(fine), fine

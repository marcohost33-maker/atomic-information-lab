"""Check the evidence chain of a pre-registered run from git history alone (R30, AIL-03 / AIL4-04).

Usage:
    python scripts/verify_preregistration.py [--repo DIR] [--prereg COMMIT] [--head REV]

Defaults are the R29 P1 run: protocol ``docs/ESTIMATOR_PROTOCOL.md`` registered in
``1097c2d``, results ``docs/ESTIMATOR_CALIBRATION_RESULTS.json``. Prints a JSON report
and exits 0 only if every check passes.

What git can show, and is checked here:

* C1 the protocol exists in the registration commit and the results file does not;
* C2 the results file was first added in a later commit that descends from the
  registration commit (content-addressed ancestry, not a timestamp);
* C3 the part of the protocol written before the run -- everything above the
  ``## Deviations`` heading -- is byte-identical at the registration commit, at the
  results commit and at ``--head``; later text may only be appended below it;
* C4 every file of the registration commit (protocol, runner, estimator) is
  byte-identical at the results commit; other package files that changed in between
  are listed, because they could change numbers (a reproduction at the registration
  commit settles that, see FINDINGS);
* C5 the fixed setting parsed from the protocol (sessions, replications, seeds,
  start offset, design) equals the setting recorded in the results.

What git cannot show, and is stated in ``limits`` instead of being checked: commit dates
are written by the committer and prove no wall-clock order; a results file can be
produced before the protocol commit and committed after it. Only an external anchor
(a push event, a timestamp authority, a third party holding the hash) closes that gap.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

PROTOCOL = "docs/ESTIMATOR_PROTOCOL.md"
RESULTS = "docs/ESTIMATOR_CALIBRATION_RESULTS.json"
PREREG = "1097c2d"
FROZEN_END = "\n## Deviations"
LIMITS = (
    "commit dates are self-declared by the committer; git proves content and ancestry, not wall-clock order",
    "a result computed before the registration commit and committed after it passes C1-C5; "
    "only an external time anchor (push event, timestamp authority, hash held by a third party) excludes that",
    "C4 lists changed package files; whether they change numbers is settled by re-running at the registration "
    "commit, not by this script",
)


class GitError(RuntimeError):
    pass


def _git(repo: Path, *args: str, check: bool = True) -> str | None:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, timeout=60)
    if r.returncode != 0:
        if check:
            raise GitError(f"git {' '.join(args)}: {r.stderr.decode(errors='replace').strip()}")
        return None
    return r.stdout.decode("utf-8")


def _show(repo: Path, rev: str, path: str) -> str | None:
    return _git(repo, "show", f"{rev}:{path}", check=False)


def _frozen(text: str) -> str | None:
    i = text.find(FROZEN_END)
    return None if i < 0 else text[:i]


def _int(s: str) -> int:
    return int(re.sub(r"[\s   ']", "", s))


def protocol_setting(text: str) -> dict:
    """The fixed setting written in the protocol (parsed, so a silent edit of either side is seen)."""
    frozen = _frozen(text)
    if frozen is None:
        raise ValueError("protocol has no '## Deviations' heading")

    def one(pattern: str) -> str:
        m = re.search(pattern, frozen)
        if not m:
            raise ValueError(f"protocol setting not found: {pattern}")
        return m.group(1)

    sessions = [_int(x) for x in one(r"\* Sessions: N ∈ \{([^}]*)\}").split(",")]
    design = tuple(x.strip() for x in one(r"\* Design: `([^`]*)`").split(","))
    return {
        "sessions": sessions,
        "replications": _int(one(r"Replications: R = ([\d\s  ]+)\b")),
        "seed_base": _int(one(r"Seed: `(\d+) \+ index of N`")),
        "start_offset": round(float(one(r"started at `true · ([\d.]+)`")) - 1, 12),
        "design": design,
        "known_write": "`known_write=True`" in frozen,
    }


def results_setting(results: dict) -> dict:
    runs = results["runs"]
    return {
        "sessions": [r["sessions"] for r in runs],
        "replications": sorted({r["replications"] for r in runs}),
        "seeds": [r["seed"] for r in runs],
        "start_offset": sorted({round(r["start_offset"], 12) for r in runs}),
        "design": sorted({tuple(r["design"]["sequence"]) for r in runs}),
        "known_write": sorted({r["design"]["known_write"] for r in runs}),
    }


def verify(repo: Path, prereg: str, head: str = "HEAD", protocol: str = PROTOCOL, results: str = RESULTS) -> dict:
    checks: dict[str, dict] = {}
    pre = _git(repo, "rev-parse", "--verify", f"{prereg}^{{commit}}").strip()
    head_sha = _git(repo, "rev-parse", "--verify", f"{head}^{{commit}}").strip()

    proto_pre = _show(repo, pre, protocol)
    checks["C1_protocol_registered_before_results"] = {
        "pass": proto_pre is not None and _show(repo, pre, results) is None,
        "protocol_in_registration_commit": proto_pre is not None,
        "results_in_registration_commit": _show(repo, pre, results) is not None,
    }

    added = _git(repo, "log", "--diff-filter=A", "--format=%H", head_sha, "--", results).split()
    res_commit = added[-1] if added else None
    descends = (
        res_commit is not None
        and res_commit != pre
        and _git(repo, "merge-base", "--is-ancestor", pre, res_commit, check=False) is not None
    )
    checks["C2_results_added_in_a_descendant"] = {"pass": descends, "results_commit": res_commit}

    frozen = {"registration": proto_pre, "head": _show(repo, head_sha, protocol)}
    if res_commit:
        frozen["results"] = _show(repo, res_commit, protocol)
    parts = {k: (None if v is None else _frozen(v)) for k, v in frozen.items()}
    checks["C3_frozen_protocol_unchanged"] = {
        "pass": parts["registration"] is not None and len({*parts.values()}) == 1,
        "identical": {k: v == parts["registration"] for k, v in parts.items()},
        "frozen_bytes": None if parts["registration"] is None else len(parts["registration"].encode()),
    }

    registered = _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", "--root", pre).split()
    changed_registered, changed_other = [], []
    if res_commit:
        changed_registered = [
            f for f in registered if f != protocol and _show(repo, pre, f) != _show(repo, res_commit, f)
        ]
        between = _git(repo, "diff", "--name-only", pre, res_commit).split()
        changed_other = [f for f in between if f not in registered and f != results]
    checks["C4_registered_code_unchanged_at_run"] = {
        "pass": res_commit is not None and not changed_registered,
        "registered_files": registered,
        "registered_files_changed": changed_registered,
        "other_files_changed_before_results": changed_other,
    }

    c5 = {"pass": False}
    try:
        want = protocol_setting(proto_pre or "")
        got = results_setting(json.loads(_show(repo, res_commit, results) or "{}")) if res_commit else None
        if got is not None:
            n = len(want["sessions"])
            c5 = {
                "pass": got["sessions"] == want["sessions"]
                and got["replications"] == [want["replications"]]
                and got["seeds"] == [want["seed_base"] + i for i in range(n)]
                and got["start_offset"] == [want["start_offset"]]
                and got["design"] == [want["design"]]
                and got["known_write"] == [want["known_write"]],
                "protocol": {**want, "design": list(want["design"])},
                "results": {**got, "design": [list(d) for d in got["design"]]},
            }
    except (ValueError, KeyError, TypeError) as exc:
        c5 = {"pass": False, "error": str(exc)}
    checks["C5_setting_matches_protocol"] = c5

    return {
        "schema": "ail-preregistration-check-v1",
        "registration_commit": pre,
        "head": head_sha,
        "protocol": protocol,
        "results": results,
        "checks": checks,
        "all_pass": all(c["pass"] for c in checks.values()),
        "limits": list(LIMITS),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--prereg", default=PREREG)
    ap.add_argument("--head", default="HEAD")
    args = ap.parse_args(argv)
    try:
        report = verify(Path(args.repo), args.prereg, args.head)
    except GitError as exc:
        print(json.dumps({"schema": "ail-preregistration-check-v1", "all_pass": False, "error": str(exc)}))
        return 2
    print(json.dumps(report, indent=1, ensure_ascii=False))
    return 0 if report["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())

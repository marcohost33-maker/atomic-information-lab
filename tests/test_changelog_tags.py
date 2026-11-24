"""Release metadata stays in step with the git tags (R34b; Lagebild 2026-11-24, A6).

Defect found: tag ``v0.1.0`` (-> 84cb661, 2026-11-22) existed while CHANGELOG.md still said
``## [0.1.0] - not released`` and listed everything since under ``[Unreleased]``.
Rule checked here: every tag ``vX.Y.Z`` has a heading ``## [X.Y.Z] - <date of the tagged commit>``,
no version heading says "not released", and CITATION.cff / pyproject / ``__version__`` agree.
Without git or without tags (e.g. an unpacked sdist) the tag part is skipped with a reason.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HEADING = re.compile(r"^## \[(\d+\.\d+\.\d+)\] - (\S.*)$", re.MULTILINE)


def changelog_versions(text: str) -> dict[str, str]:
    return {v: d.strip() for v, d in HEADING.findall(text)}


def _tags() -> dict[str, str]:
    if shutil.which("git") is None:
        return {}
    r = subprocess.run(
        ["git", "-C", str(ROOT), "for-each-ref", "--format=%(refname:short) %(*objectname)%(objectname)", "refs/tags"],
        capture_output=True,
        text=True,
    )
    out = {}
    for line in r.stdout.split("\n") if r.returncode == 0 else []:
        m = re.fullmatch(r"v(\d+\.\d+\.\d+) ([0-9a-f]+)", line.strip())
        if m:
            date = subprocess.run(
                ["git", "-C", str(ROOT), "log", "-1", "--format=%cs", f"v{m.group(1)}^{{commit}}"],
                capture_output=True,
                text=True,
            ).stdout.strip()
            out[m.group(1)] = date
    return out


def test_parser_positive_and_negative_control():
    text = "# C\n\n## [Unreleased]\n\n## [1.2.3] - 2026-01-02\n\n## [0.9.0] - not released\n"
    assert changelog_versions(text) == {"1.2.3": "2026-01-02", "0.9.0": "not released"}


def test_no_version_heading_says_not_released():
    versions = changelog_versions((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", d) for d in versions.values()), versions


def test_every_tag_has_a_dated_heading():
    tags = _tags()
    if not tags:
        pytest.skip("no git or no version tags (sdist or clone without tags)")
    versions = changelog_versions((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    for version, date in tags.items():
        assert versions.get(version) == date, (version, date, versions.get(version))


def test_citation_pyproject_and_package_version_agree():
    from atomic_information_lab import __version__

    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    py = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    cff_version = re.search(r"^version: (\S+)$", cff, re.MULTILINE).group(1)
    py_version = re.search(r'^version = "([^"]+)"$', py, re.MULTILINE).group(1)
    assert cff_version == py_version == __version__
    released = re.search(r'^date-released: "(\d{4}-\d{2}-\d{2})"$', cff, re.MULTILINE)
    versions = changelog_versions((ROOT / "CHANGELOG.md").read_text(encoding="utf-8"))
    if released:
        assert versions.get(cff_version) == released.group(1)
    tags = _tags()
    if cff_version in tags:
        assert released and released.group(1) == tags[cff_version]

"""Repository hygiene tests.

Validates that:
  * `requirements.txt` pins a version for every dependency entry
    (Requirement 10.1).
  * `README.md` contains all five required section headings: Setup,
    CLI Usage, MCP Connection, Architecture, and Model Justification
    (Requirement 10.2).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_PATH = REPO_ROOT / "requirements.txt"
README_PATH = REPO_ROOT / "README.md"


# A line counts as "pinned" if it contains any of the standard pip version
# specifiers: ==, ~=, >=, <=, ===, !=. We accept ranged constraints like
# `pydantic>=2,<3` because they still pin the major version. A bare package
# name with no operator at all is rejected.
_VERSION_OPERATOR_RE = re.compile(r"(==|~=|>=|<=|===|!=|<|>)")


def _iter_requirement_lines(text: str):
    """Yield non-blank, non-comment lines from a requirements.txt body."""
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Strip inline comments (anything after a '#') so they don't
        # interfere with operator detection.
        if "#" in line:
            line = line.split("#", 1)[0].strip()
        if not line:
            continue
        yield line


def test_requirements_file_exists():
    """The requirements.txt file must exist at the repo root."""
    assert REQUIREMENTS_PATH.is_file(), (
        f"requirements.txt not found at {REQUIREMENTS_PATH}"
    )


def test_every_requirement_has_pinned_version():
    """Every dependency entry in requirements.txt must pin a version.

    Validates: Requirements 10.1
    """
    text = REQUIREMENTS_PATH.read_text(encoding="utf-8")
    entries = list(_iter_requirement_lines(text))
    assert entries, "requirements.txt must list at least one dependency"

    unpinned: list[str] = []
    for entry in entries:
        if not _VERSION_OPERATOR_RE.search(entry):
            unpinned.append(entry)

    assert not unpinned, (
        "All requirements.txt entries must pin a version. "
        f"Unpinned entries: {unpinned}"
    )


@pytest.mark.parametrize(
    "heading",
    [
        "Setup",
        "CLI Usage",
        "MCP Connection",
        "Architecture",
        "Model Justification",
    ],
)
def test_readme_contains_required_heading(heading: str):
    """README.md must contain each required `## <heading>` section.

    Validates: Requirements 10.2
    """
    assert README_PATH.is_file(), f"README.md not found at {README_PATH}"
    text = README_PATH.read_text(encoding="utf-8")

    # Match a line that begins with `## ` followed by the heading text.
    # We allow trailing whitespace but require the heading to be the full
    # line content (no extra words), to avoid accidental matches inside
    # paragraphs or sub-headings of a different level.
    pattern = re.compile(
        rf"^##\s+{re.escape(heading)}\s*$",
        re.MULTILINE,
    )
    assert pattern.search(text), (
        f"README.md is missing required heading '## {heading}'"
    )

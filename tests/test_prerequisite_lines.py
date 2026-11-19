"""Each extension lesson page carries one hand-written Prerequisites line that mirrors the syllabus (#321)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from infrastructure.extension import route as study_route

ROOT = Path(__file__).resolve().parents[1]
ROUTE = study_route.load(ROOT)
EXTENSION = sorted(ROUTE.prerequisites, key=lambda n: tuple(int(x) for x in n.split(".")))
LINE = re.compile(r"(?m)^\*\*Prerequisites:\*\*(.*(?:\n(?!\n).*)*)")
ANY_PREREQUISITES_PARAGRAPH = re.compile(r"(?im)^(?:\*\*)?prerequisites\b")
REFERENCE = re.compile(r"\[(\d+(?:\.\d+)+)\]")
OBJECTIVES = "## Content & Learning Objectives"


def opening(number: str) -> str:
    """The page text before its objectives heading, where the Prerequisites line lives."""
    text = ROUTE.pages[number].read_text(encoding="utf-8")
    assert OBJECTIVES in text, f"{number}: page has no '{OBJECTIVES}' heading"
    return text.split(OBJECTIVES, 1)[0]


def test_every_extension_section_is_checked():
    assert len(EXTENSION) == 86
    assert {"0.6", "1.6", "5.1", "18.2"} <= set(EXTENSION)


@pytest.mark.parametrize("number", EXTENSION)
def test_page_has_one_prerequisites_line_matching_the_syllabus(number):
    text = opening(number)
    lines = LINE.findall(text)
    assert len(lines) == 1, (
        f"{number}: expected one '**Prerequisites:**' paragraph before the objectives"
    )
    assert len(ANY_PREREQUISITES_PARAGRAPH.findall(text)) == 1, (
        f"{number}: another paragraph also starts with 'Prerequisites'; keep only the syllabus line"
    )
    line = lines[0]
    named = REFERENCE.findall(line)
    assert sorted(named) == sorted(ROUTE.prerequisites[number]), (
        f"{number}: the line names {named}, the syllabus says {list(ROUTE.prerequisites[number])}"
    )
    for prerequisite in named:
        title = ROUTE.first_stop(prerequisite).title
        assert f"[{prerequisite}] {title}" in " ".join(line.split()), (
            f"{number}: write [{prerequisite}] followed by its title, {title!r}"
        )

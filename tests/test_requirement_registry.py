"""Consistency checks for the requirement registry, the syllabus mapping and the companions list.

Status lives only in docs/roadmap_requirement_registry.yml; the syllabus `requirements:` lists are the
only mapping from requirements to sections. Counts are deliberately not asserted: they move as
sections are finished.
"""

import re
from collections import Counter
from pathlib import Path

import pytest
import yaml

from infrastructure.extension.sync_config import (
    REGISTRY,
    SYLLABUS,
    validate_companions,
    validate_requirements,
)

STATUSES = ["open", "met", "deferred"]
DEFERRED_REASONS = {"hardware", "upstream", "optional"}
ROW_FIELDS = {"id", "topic", "roadmap_refs", "status", "note"}
DEFERRED_FIELDS = {"deferred_reason", "deferred_until"}

# Original-ARENA rows: taught by chapters 0-4, so no extension section maps them.
ORIGINAL_ROWS = {
    "foundations.original_arena_preservation",
    "foundations.original_mechinterp_spine",
    "representation.linear_and_nonlinear_probes",
    "circuits.path_patching",
    "world.othello_gpt",
    "organisms.greater_than",
}
# Cross-course rows: met by every extension section together, so no single section maps them.
CROSS_COURSE_ROWS = {"foundations.hard_exercise_test_ladders"}

REGISTRY_DATA = yaml.safe_load(REGISTRY.read_text())
ROWS = REGISTRY_DATA["requirements"]
SYLLABUS_DATA = yaml.safe_load(SYLLABUS.read_text())


def _sections() -> list[dict]:
    return [s for chapter in SYLLABUS_DATA["chapters"].values() for s in chapter["sections"]]


def _mapped() -> list[tuple[str, dict]]:
    return [(req, s) for s in _sections() for req in s.get("requirements", [])]


def test_allowed_statuses_and_row_statuses():
    assert REGISTRY_DATA["allowed_statuses"] == STATUSES
    bad = [r["id"] for r in ROWS if r["status"] not in STATUSES]
    assert not bad, f"invalid status on {bad}"


def test_rows_have_only_allowed_fields():
    for row in ROWS:
        allowed = ROW_FIELDS | (DEFERRED_FIELDS if row["status"] == "deferred" else set())
        assert set(row) == allowed, f"{row['id']}: fields {sorted(set(row) ^ allowed)} differ"
        assert (
            isinstance(row["note"], str) and row["note"].strip() and "\n" not in row["note"].strip()
        )


RETIRED_GATES = re.compile(
    r"independent|editorial|reproduc|release registration|acceptance|publication|receipt|evidence",
    re.IGNORECASE,
)


def test_notes_do_not_mention_retired_gates():
    bad = [r["id"] for r in ROWS if RETIRED_GATES.search(r["note"])]
    assert not bad, bad


def test_no_duplicate_ids():
    dupes = [i for i, n in Counter(r["id"] for r in ROWS).items() if n > 1]
    assert not dupes, dupes


def test_deferred_rows_have_reason_and_trigger_and_others_have_neither():
    for row in ROWS:
        if row["status"] == "deferred":
            assert row["deferred_reason"] in DEFERRED_REASONS, row["id"]
            assert str(row["deferred_until"]).strip(), row["id"]
        else:
            assert "deferred_reason" not in row and "deferred_until" not in row, row["id"]


def test_syllabus_names_no_unknown_ids():
    known = {r["id"] for r in ROWS}
    unknown = sorted({req for req, _ in _mapped()} - known)
    assert not unknown, unknown


def test_every_id_is_mapped_exactly_once_unless_original_or_cross_course():
    mapped = Counter(req for req, _ in _mapped())
    exempt = ORIGINAL_ROWS | CROSS_COURSE_ROWS
    ids = {r["id"] for r in ROWS}
    assert exempt <= ids
    for rid in ids - exempt:
        assert mapped[rid] == 1, f"{rid} is mapped {mapped[rid]} times"


def test_met_rows_map_to_a_finished_section_or_are_original():
    sections_of = {}
    for req, section in _mapped():
        sections_of.setdefault(req, []).append(section)
    for row in ROWS:
        if row["status"] != "met" or row["id"] in ORIGINAL_ROWS:
            continue
        sections = sections_of.get(row["id"], [])
        assert sections, f"{row['id']} is met but mapped to no section"
        assert all(s.get("status") == "finished" for s in sections), row["id"]


def test_validate_requirements_rejects_unknown_and_duplicate_ids():
    with pytest.raises(ValueError, match="unknown"):
        validate_requirements({"nope.row": "5.1"}, {"a.b"})
    validate_requirements({"a.b": "5.1"}, {"a.b", "c.d"})


GOOD = [
    {"unit": "u01", "title": "Basics", "dir": None, "status": "planned"},
    {"unit": "u02", "title": "Hyperparameter scaling", "dir": "present", "status": "written"},
]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "present").mkdir()
    return tmp_path


def test_real_companions_list_is_valid():
    validate_companions(SYLLABUS_DATA["companions"])
    assert [c["unit"] for c in SYLLABUS_DATA["companions"]] == [f"u0{i}" for i in range(1, 8)]


def test_validate_companions_accepts_a_good_list(root):
    validate_companions(GOOD, root=root)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda c: c[1].update(unit="u01"), "duplicate"),
        (lambda c: c[0].update(unit="unit1"), "unit"),
        (lambda c: c[0].update(status="done"), "status"),
        (lambda c: c[1].update(dir="missing"), "does not exist"),
        (lambda c: c[0].update(status="finished"), "requires a dir"),
        (lambda c: c[0].pop("title"), "title"),
    ],
)
def test_validate_companions_rejects_bad_lists(root, mutate, message):
    companions = [dict(c) for c in GOOD]
    mutate(companions)
    with pytest.raises(ValueError, match=message):
        validate_companions(companions, root=root)

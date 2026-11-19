"""The syllabus `route:` block: what `sync_config` accepts, and what the study guide shows (#315)."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from infrastructure.extension import route as study_route
from infrastructure.extension import sync_config

ROOT = Path(__file__).resolve().parents[1]
SYLLABUS = yaml.safe_load((ROOT / "infrastructure/extension/syllabus.yaml").read_text(encoding="utf-8"))
ORIGINALS = study_route.original_sections(ROOT)


def small_syllabus() -> dict:
    """Two originals, two extension sections, one companion, one planned lesson."""
    return {
        "chapters": {
            "chapter5_fixture": {
                "number": 5,
                "sections": [
                    {"number": "5.1", "title": "First", "dir": "part1_first", "prerequisites": ["1.1"]},
                    {"number": "5.2", "title": "Second", "dir": "part2_second", "prerequisites": ["5.1"]},
                ],
            }
        },
        "companions": [{"unit": "u01", "title": "Basics", "dir": None, "status": "planned"}],
        "route": [
            {"stage": "Start", "stops": [{"section": "0.0", "core": True}, {"companion": "u01"}]},
            {"stage": "Build", "stops": [
                {"section": "1.1", "core": True},
                {"section": "5.1", "core": True},
                {"section": "5.2"},
                {"planned": "5.3", "title": "Third", "days": 1},
            ]},
        ],
    }


SMALL_ORIGINALS = {"0.0": "Prerequisites", "1.1": "Transformers from Scratch"}


def check(doc: dict) -> None:
    study_route.validate_route(doc, SMALL_ORIGINALS)


def test_sync_config_accepts_the_real_route():
    study_route.validate_route(SYLLABUS, ORIGINALS)


def test_a_small_valid_route_passes():
    check(small_syllabus())


@pytest.mark.parametrize(
    "edit, message",
    [
        (lambda d: d["route"][1]["stops"].append({"section": "9.9"}), "9.9"),
        (lambda d: d["chapters"]["chapter5_fixture"]["sections"][1].update(prerequisites=["7.7"]), "7.7"),
        (lambda d: d["route"][0]["stops"].append({"companion": "u05"}), "u05"),
    ],
    ids=["unknown section", "unknown prerequisite", "unknown companion"],
)
def test_unknown_ids_are_rejected(edit, message):
    doc = small_syllabus()
    edit(doc)
    with pytest.raises(ValueError, match=message):
        check(doc)


def test_prerequisite_after_the_stop_that_needs_it_is_rejected():
    doc = small_syllabus()
    stops = doc["route"][1]["stops"]
    stops[1], stops[2] = stops[2], stops[1]  # 5.2 now comes before its prerequisite 5.1
    with pytest.raises(ValueError, match=r"5\.2.*5\.1"):
        check(doc)


def test_prerequisite_at_the_same_stop_is_rejected():
    doc = small_syllabus()
    doc["chapters"]["chapter5_fixture"]["sections"][0]["prerequisites"] = ["5.1"]
    with pytest.raises(ValueError, match=r"5\.1"):
        check(doc)


def test_core_stop_cannot_need_an_optional_one():
    doc = small_syllabus()
    doc["route"][1]["stops"][0]["core"] = False  # 1.1 optional, but core 5.1 needs it
    with pytest.raises(ValueError, match=r"core.*5\.1.*1\.1"):
        check(doc)


def test_every_configured_section_must_be_on_the_route():
    doc = small_syllabus()
    del doc["route"][1]["stops"][2]  # 5.2 is configured but nowhere on the route
    with pytest.raises(ValueError, match=r"5\.2"):
        check(doc)


def test_every_listed_companion_must_have_a_stop():
    doc = small_syllabus()
    doc["companions"].append({"unit": "u02", "title": "Scaling", "dir": None, "status": "planned"})
    with pytest.raises(ValueError, match="u02"):
        check(doc)


def test_planned_label_must_not_shadow_a_configured_section():
    doc = small_syllabus()
    doc["route"][1]["stops"][3]["planned"] = "5.2"
    with pytest.raises(ValueError, match=r"5\.2"):
        check(doc)


def test_a_section_appears_once_unless_split_into_parts():
    doc = small_syllabus()
    doc["route"][1]["stops"].append({"section": "5.1"})
    with pytest.raises(ValueError, match=r"5\.1"):
        check(doc)
    doc = small_syllabus()
    doc["route"][1]["stops"][1].update(parts="1-2", days=1)
    doc["route"][1]["stops"].append({"section": "5.1", "parts": "3-", "days": 0.5})
    check(doc)


def test_sync_config_runs_the_route_check(monkeypatch, tmp_path):
    doc = copy.deepcopy(SYLLABUS)
    doc["route"][0]["stops"].append({"section": "99.9"})
    syllabus = tmp_path / "syllabus.yaml"
    syllabus.write_text(yaml.safe_dump(doc), encoding="utf-8")
    monkeypatch.setattr(sync_config, "SYLLABUS", syllabus)
    monkeypatch.setattr(sync_config, "CONFIG", tmp_path / "config_extension.yaml")
    with pytest.raises(ValueError, match=r"99\.9"):
        sync_config.main()


@pytest.mark.parametrize("kind", ["split", "planned"])
def test_stops_without_time_lines_need_days(kind):
    doc = small_syllabus()
    if kind == "planned":
        del doc["route"][1]["stops"][3]["days"]
        message = r"5\.3"
    else:
        doc["route"][1]["stops"][1].update(parts="1-2", days=1)
        doc["route"][1]["stops"].append({"section": "5.1", "parts": "3-"})
        message = r"5\.1"
    with pytest.raises(ValueError, match=message):
        check(doc)


# ---------- the real route (#315) ----------

SPEC_STAGES = [
    "Start", "Fundamentals", "Transformers", "Is it real?", "Modern architectures", "Features", "Circuits",
    "Reading activations", "Geometry", "Research Practice I", "RL", "Evals", "Alignment", "Multimodal",
    "World models and convergence", "Attribution", "Dynamics and intrinsic", "Finale",
]
# Spec #39's "Core days" column. Stage 11 was 3 before the Batch 0 sync split upstream 2.2 into 2.2.1 and 2.2.2.
SPEC_CORE_DAYS = [1, 4.5, 2, 1, 2, 5, 3, 1, 3, 1.5, 4, 3, 5.5, 3, 2.5, 2, 3, 3]


@pytest.fixture(scope="module")
def real_route():
    return study_route.load(ROOT)


def test_the_route_has_the_approved_eighteen_stages(real_route):
    assert [title for title, _ in real_route.stages] == SPEC_STAGES


def test_stage_core_days_follow_the_time_lines(real_route):
    assert [real_route.stage_days(n)[0] for n in range(1, 19)] == SPEC_CORE_DAYS
    core, everything = real_route.totals()
    assert core == sum(SPEC_CORE_DAYS)
    assert core < everything


@pytest.mark.parametrize(
    "minutes, days", [(0, 1), (110, 1), (275, 1), (330, 1), (495, 1.5), (515, 1.5), (870, 2.5), (1300, 3.5)]
)
def test_a_day_is_six_hours_in_half_days_with_a_minimum_of_one(minutes, days):
    assert study_route.days_from_minutes(minutes) == days


def stops_of(route, number):
    return [stop for stop in route.stops if stop.number == number]


def test_split_stops_sit_where_the_spec_puts_them(real_route):
    first, last = stops_of(real_route, "10.1")
    assert (first.stage, first.core, first.first_part, first.last_part) == (10, True, 1, 4)
    assert (last.stage, last.core, last.first_part, last.last_part) == (18, True, 5, None)
    core_part, rest = stops_of(real_route, "11.2")
    assert (core_part.stage, core_part.core, core_part.first_part, core_part.last_part) == (9, True, 1, 2)
    assert (rest.stage, rest.core) == (9, False)
    for number in ("14.4", "19.3"):
        (stop,) = stops_of(real_route, number)
        assert (stop.stage, stop.core) == (15, False)


def test_upstream_split_sections_use_their_own_identities(real_route):
    numbers = {stop.number for stop in real_route.stops}
    assert {"0.5.1", "0.5.2", "2.2.1", "2.2.2", "2.3.2", "2.4.1", "2.4.2", "2.4.3"} <= numbers
    assert not {"0.5", "2.2", "2.4", "1.3", "1.4", "1.5"} & numbers
    (dqn,) = stops_of(real_route, "2.2.1")
    assert dqn.title == "DQN" and dqn.kind == "original"


def test_original_stops_open_on_learn_arena_education(real_route):
    (ioi,) = stops_of(real_route, "1.4.1")
    assert real_route.original_url(ioi) == "https://learn.arena.education/chapter1_transformer_interp/21_ioi/"
    (vpg,) = stops_of(real_route, "2.2.2")
    assert real_route.original_url(vpg) == "https://learn.arena.education/chapter2_rl/22_vpg/"


def test_approved_future_lessons_are_plain_labels(real_route):
    planned = [stop for stop in real_route.stops if stop.kind == "planned"]
    assert [stop.number for stop in planned] == ["0.7", "10.2", "10.3", "19.1", "19.2", "19.3"]
    assert all(real_route.original_url(stop) is None for stop in planned)

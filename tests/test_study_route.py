"""The syllabus `route:` block: what `sync_config` accepts, and what the study guide shows (#315)."""

from __future__ import annotations

import copy
import dataclasses
import re
from pathlib import Path
from urllib.parse import unquote

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


# ---------- the study guide (#315, #322) ----------

GUIDE = ROOT / "mastery_hub/STUDY-GUIDE.md"
STATUS_WORDS = re.compile(
    r"(?i)\b(status|verified|verification|evidence|provenance|readiness|acceptance|pending|historical|"
    r"not (?:yet )?(?:written|available|finished)|work in progress|draft)\b"
)
MARKDOWN_LINK = re.compile(r"\]\(<?([^)>\s]+)>?\)")


def guide_text() -> str:
    return GUIDE.read_text(encoding="utf-8")


def test_guide_route_block_is_generated_from_the_syllabus(real_route):
    text = guide_text()
    start, end = study_route.GUIDE_MARKERS
    block = text.split(start, 1)[1].split(end, 1)[0]
    assert block.strip() == study_route.render_guide(real_route, ROOT).strip(), (
        "the study guide is out of date: run `uv run python infrastructure/extension/route.py --write`"
    )


def test_guide_shows_every_stage_with_days_and_totals(real_route):
    text = guide_text()
    table = [line for line in text.splitlines() if re.match(r"^\| \d+ \|", line)]
    assert len(table) == 18
    for stage, line in enumerate(table, start=1):
        core, everything = real_route.stage_days(stage)
        assert f"| {core:g} | {everything:g} |" in line, line
    core, everything = real_route.totals()
    assert f"about {core:g} days" in text and f"about {everything:g} days" in text


def test_guide_starts_at_original_zero_with_the_graduate_shortcut():
    text = guide_text()
    assert "https://learn.arena.education/chapter0_fundamentals/00_prereqs/" in text
    assert re.search(r"Done ARENA chapters 0–1\?.*0\.6.*stage 4", text)


def test_guide_has_no_status_or_provenance_wording():
    found = STATUS_WORDS.findall(guide_text())
    assert not found, f"learner copy carries status wording: {found}"


def test_guide_links_resolve():
    for target in MARKDOWN_LINK.findall(guide_text()):
        if target.startswith("http"):
            assert target.startswith(("https://learn.arena.education/", "https://deep-learning-alchemy.github.io/",
                                      "http://127.0.0.1:18505/")), target
            continue
        path = (GUIDE.parent / unquote(target.split("#", 1)[0])).resolve()
        assert path.exists(), f"dead link {target}"


def test_planned_lessons_are_unlinked_labels(real_route):
    text = guide_text()
    for stop in real_route.stops:
        if stop.kind == "planned":
            assert f"[{stop.number}] {stop.title}" in text
            assert not re.search(rf"\[\[{re.escape(stop.number)}\][^\]]*\]\(", text)


def test_companions_are_optional_side_trips_at_their_stops(real_route):
    stops = real_route.stops
    after = {stop.number: stops[i - 1].number for i, stop in enumerate(stops) if stop.kind == "companion"}
    assert after == {"u01": "0.3", "u02": "u01", "u03": "u02", "u04": "1.1", "u05": "5.2", "u06": "16.9", "u07": "10.1"}
    assert all(not stop.core for stop in stops if stop.kind == "companion")
    text = guide_text()
    for unit in after:
        assert f"Alchemy {unit}" in text
    u02 = text.split("Alchemy u02", 1)[1].split("\n", 1)[0]
    assert "optimizer lab" in u02 and "mastery_hub/app.py" in u02


def test_side_trips_are_unsized_and_never_core(real_route):
    trips = [stop for stop in real_route.stops if stop.kind == "companion"]
    assert len(trips) == 7
    assert all(stop.days is None and not stop.core for stop in trips)


def test_companion_materials_link_only_what_exists(real_route, tmp_path):
    for unit in real_route.companions:
        for _, target in study_route.companion_materials(real_route, unit, ROOT):
            assert target.startswith("https://") or (ROOT / target).exists(), (unit, target)
    lab = dict(study_route.companion_materials(real_route, "u02", ROOT))
    assert lab.get("optimizer lab") == "mastery_hub/app.py"
    # Without the companion page or the lab, nothing dangles.
    assert study_route.companion_materials(real_route, "u02", tmp_path) == []


# ---------- chapter homes (#321) ----------


def test_chapter_tldr_names_preparation_start_optional_and_onward_route(real_route):
    text = study_route.chapter_tldr(real_route, "chapter8_automated_circuits")
    assert "stage 7" in text and "Circuits" in text
    assert re.search(r"Before you start.*\[1\.4\.1\] Indirect Object Identification", text)
    assert re.search(r"Start with.*\[8\.1\] Activation Patching Refresher", text)
    assert re.search(r"(?s)Core.*\[8\.1\].*\[8\.2\]", text)
    optional = text.split("Optional", 1)[1]
    assert "[8.3] AtP* from Scratch" in optional
    assert re.search(r"\[8\.7\][^\n]*after[^\n]*\[6\.4\]", optional)
    assert re.search(r"Then.*stage 8.*\[7\.1\]", text)


def test_split_chapters_give_directions_per_stage(real_route):
    text = study_route.chapter_tldr(real_route, "chapter10_capstone_research_sprint")
    stage10, stage18 = text.split("Stage 18", 1)
    assert "parts 1️⃣–4️⃣" in stage10 and "[10.2] Replicate and Critique a Fresh Claim" in stage10
    assert "[10.3]" in stage10 and re.search(r"Then.*stage 11.*\[2\.1\] Intro to RL", stage10)
    assert "part 5️⃣" in stage18 or "parts 5️⃣" in stage18
    assert "end of the route" in stage18
    peft = study_route.chapter_tldr(real_route, "chapter15_peft_misalignment")
    assert "Stage 13" in peft and "Stage 14" in peft
    assert re.search(r"(?s)Stage 14.*\[15\.5\] VLM LoRA.*Then.*\[13\.1\]", peft)


def test_chapter_tldr_links_use_the_given_destinations(real_route):
    seen = []

    def ref(stop):
        seen.append(stop)
        return f"<{stop.number}>"

    text = study_route.chapter_tldr(real_route, "chapter8_automated_circuits", ref)
    assert "<8.1>" in text and "<1.4.1>" in text and "<7.1>" in text
    assert {stop.kind for stop in seen} <= {"original", "extension", "planned", "companion"}


def test_every_extension_chapter_has_a_tldr(real_route):
    for chapter in SYLLABUS["chapters"]:
        if SYLLABUS["chapters"][chapter]["number"] >= 5:
            assert "Start with" in study_route.chapter_tldr(real_route, chapter) or "Optional" in study_route.chapter_tldr(real_route, chapter)


def test_the_finale_links_u07_once_its_companion_exists(real_route, tmp_path):
    (tmp_path / "mastery_hub/units/u07_exotic").mkdir(parents=True)
    companions = dict(real_route.companions)
    companions["u07"] = {**companions["u07"], "dir": "mastery_hub/units/u07_exotic", "status": "written"}
    released = dataclasses.replace(real_route, companions=companions)
    assert study_route.companion_materials(released, "u07", tmp_path) == [
        ("companion notebook", "mastery_hub/units/u07_exotic")
    ]
    assert study_route.companion_materials(real_route, "u07", ROOT) == [] or all(
        label == "companion list" for label, _ in study_route.companion_materials(real_route, "u07", ROOT)
    )

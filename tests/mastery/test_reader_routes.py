"""Public reader routes and local notebook handoff, without executing lesson code."""
from html import unescape
from pathlib import Path
import sys
import pytest
from streamlit.testing.v1 import AppTest

from mastery_hub import read as reader

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rows():
    return reader.inventory(ROOT)


def test_bookmarks_are_one_based_and_resolve_only_configured_sections(rows):
    row, part, error = reader.resolve_route({"section": "12.3", "part": "2"}, rows)
    assert error is None
    assert row["number"] == "12.3"
    assert part == 1
    assert reader.lesson_url("12.3", 1) == "?section=12.3&part=2"
    assert reader.resolve_route({}, rows) == (None, 0, None)


@pytest.mark.parametrize("query, message", [
    ({"section": "../../private.txt"}, "configured section"),
    ({"section": "999.1"}, "configured section"),
    ({"part": "2"}, "section"),
    ({"section": "7.1", "part": "0"}, "Part"),
    ({"section": "7.1", "part": "-1"}, "Part"),
    ({"section": "7.1", "part": "1.5"}, "Part"),
    ({"section": "7.1", "part": "99999"}, "Part"),
    ({"section": ["7.1", "12.1"]}, "one"),
    ({"section": "7.1", "part": ["1", "2"]}, "one"),
])
def test_invalid_routes_are_actionable_without_lesson_fallback(rows, query, message):
    row, _, error = reader.resolve_route(query, rows)
    assert row is None
    assert message.lower() in error.lower()
    assert "choose" in error.lower() or "use" in error.lower()


def app(query=None):
    at = AppTest.from_file(str(ROOT / "mastery_hub/read.py"), default_timeout=30)
    if query:
        at.query_params.update(query)
    return at.run()


def markdown_text(at):
    return unescape("\n".join(element.value for element in at.markdown))


def _param(at, key: str) -> str:
    """One query parameter as a string (Streamlit's AppTest returned a list before 1.67)."""
    value = at.query_params[key]
    return value[-1] if isinstance(value, list) else value

def test_reader_starts_at_bookmark_and_manual_selectors_update_resume():
    at = app({"section": "12.3", "part": "2"})
    assert not at.exception
    assert at.selectbox[0].value == "chapter12_vlm_interpretability"
    assert at.selectbox[1].value["number"] == "12.3"
    assert at.selectbox[2].value == 1
    text = markdown_text(at)
    assert "?section=12.3&part=2" in text
    assert "Previous part" in text and "Next part" in text
    assert "?section=12.3&part=1" in text
    assert "?section=12.3&part=3" in text
    assert 'target="_self"' in text
    at.selectbox[2].select_index(2).run()
    assert not at.exception
    assert _param(at, "section") == "12.3"
    assert _param(at, "part") == "3"
    at.selectbox[1].select_index(0).run()
    assert not at.exception
    assert at.selectbox[1].value["number"] == "12.1"
    assert _param(at, "section") == "12.1"
    assert _param(at, "part") == "1"
    at.selectbox[0].select("chapter7_activation_to_language").run()
    assert not at.exception
    assert at.selectbox[1].value["number"] == "7.1"
    assert _param(at, "section") == "7.1"
    at.run()
    assert at.selectbox[0].value == "chapter7_activation_to_language"
    assert at.selectbox[1].value["number"] == "7.1"
    # A newly followed URL overrides prior session selections, including the part.
    at.query_params.update({"section": "16.1", "part": "2"})
    at.run()
    assert not at.exception
    assert at.selectbox[1].value["number"] == "16.1"
    assert at.selectbox[2].value == 1


def test_start_routes_and_exact_jupyter_handoff_are_visible(rows):
    at = app({"section": "7.1", "part": "1"})
    assert not at.exception
    text = markdown_text(at)
    assert "https://learn.arena.education/chapter0_fundamentals/00_prereqs/" in text
    assert "?section=0.6&part=1" in text and "stage 4" in text
    for retired in ("Small CPU route", "Read internal answers", "Match images and text", "Intervene on a direction"):
        assert retired not in text
    row = next(row for row in rows if row["number"] == "7.1")
    assert str(row["exercises"].parent) in text
    assert row["exercises"].name in text
    assert "PUBLIC Practice" not in text and "blind evaluation" not in text
    assert "jupyter lab" in text
    assert "Kernel" in text and str(Path(sys.executable).absolute()) in text
    assert "Restart Kernel" in text and "Run All" in text
    assert "https://learn.arena.education/" in text
    assert "Previous part" not in text
    at.selectbox[2].select_index(len(reader.parts(row["page"])) - 1).run()
    assert not at.exception
    assert "Next part" not in markdown_text(at)
    assert "Previous part" in markdown_text(at)


@pytest.mark.parametrize("query", [{"section": "7.1", "part": "-1"}, {"section": "999.1"}])
def test_invalid_url_does_not_render_an_unrelated_lesson_or_download(query):
    at = app(query)
    assert not at.exception
    assert at.info
    assert not at.get("download_button")
    assert "choose" in "\n".join(element.value for element in at.info).lower()


def test_missing_configured_route_is_actionable_before_reading_page(rows, tmp_path):
    unavailable = dict(next(row for row in rows if row["number"] == "18.1"))
    missing_page = tmp_path / "missing-configured-page.md"
    unavailable.update(available=False, page=missing_page,
                       status="Lesson unavailable. Restore the configured fixture page.")
    row, _, error = reader.resolve_route({"section": "18.1", "part": "1"}, [unavailable])
    assert row is None
    assert "not available" in error and "Restore" in error and "Choose" in error
    assert not missing_page.exists()


def test_acdc_route_points_to_prerequisites():
    at = app({"section": "8.4", "part": "1"})
    assert not at.exception
    text = markdown_text(at)
    for phrase in ("1.2", "8.1", "8.2"):
        assert phrase in text


def test_mini_vlm_route_has_no_retired_memory_handoff():
    paths = [str(path.relative_to(ROOT)) for _, path in reader.supplements("12.3", ROOT)]
    assert paths == ["mastery_hub/STUDY-GUIDE.md"]
    at = app({"section": "12.3", "part": "1"})
    assert not at.exception
    assert "Before the optional Step 5 Qwen bridge" not in markdown_text(at)


# ---------- the study route in the reader (#320, #322) ----------

from infrastructure.extension import route as study_route

ROUTE = study_route.load(ROOT)


def sidebar_text(at):
    return unescape("\n".join(element.value for element in at.sidebar.markdown))


def expander_labels(at):
    return [element.label for element in at.sidebar.expander]


def last_part(section):
    row = next(row for row in reader.inventory(ROOT) if row["number"] == section)
    return len(reader.parts(row["page"]))


def test_sidebar_lists_the_eighteen_stages_and_marks_the_current_one():
    at = app({"section": "8.2", "part": "2"})
    assert not at.exception
    labels = expander_labels(at)
    stage_labels = [label for label in labels if label.lstrip("📍 ").split(".", 1)[0].isdigit()]
    assert len(stage_labels) == 18
    assert [label.split(". ", 1)[1] for label in stage_labels] == [title for title, _ in ROUTE.stages]
    assert [label for label in stage_labels if label.startswith("📍")] == ["📍 7. Circuits"]
    text = sidebar_text(at)
    assert "?section=8.1&part=1" in text  # core stops of the stage link to the reader
    assert "https://learn.arena.education/chapter1_transformer_interp/21_ioi/" in text
    assert "optional" in text.lower()


def test_next_on_the_route_appears_at_the_end_of_a_lesson_and_crosses_stages():
    at = app({"section": "8.2", "part": "2"})
    assert "Next on the route" not in markdown_text(at)
    at = app({"section": "8.2", "part": str(last_part("8.2"))})
    assert not at.exception
    text = markdown_text(at)
    assert "Next on the route" in text
    assert "?section=7.1&part=1" in text.split("Next on the route", 1)[1]


def test_split_ten_one_leads_to_research_practice_then_the_finale():
    at = app({"section": "10.1", "part": "5"})  # part 4️⃣, the end of stage 10's stop
    text = markdown_text(at)
    after = text.split("Next on the route", 1)[1]
    assert "[10.2] Replicate and Critique a Fresh Claim" in after  # planned: a plain label
    assert "?section=10.2" not in after
    assert 'href="https://learn.arena.education/chapter2_rl/01_intro_rl/"' in after
    assert [label for label in expander_labels(at) if label.startswith("📍")] == ["📍 10. Research Practice I"]
    at = app({"section": "10.1", "part": "6"})  # part 5️⃣, the finale
    assert [label for label in expander_labels(at) if label.startswith("📍")] == ["📍 18. Finale"]
    assert "end of the route" in markdown_text(at)


def test_core_parts_of_eleven_two_lead_on_to_stage_ten():
    at = app({"section": "11.2", "part": "3"})  # part 2️⃣ ends the core stop
    text = markdown_text(at)
    assert "?section=10.1&part=1" in text.split("Next on the route", 1)[1]


def test_next_on_the_route_skips_targets_the_reader_cannot_open(rows):
    rows = [dict(row) for row in rows]
    for row in rows:
        if row["number"] == "7.1":
            row["available"] = False
    stop = ROUTE.locate("8.2", last_part("8.2") - 1, last_part("8.2"))
    waiting, target = reader.next_on_route(ROUTE, rows, stop)
    assert [s.number for s in waiting] == ["7.1"]
    assert target.number == "1.3.1" and ROUTE.original_url(target)


def test_side_trips_show_at_their_stops_in_the_reader():
    at = app({"section": "5.2", "part": "1"})
    text = sidebar_text(at)
    assert "Alchemy u05" in text and "side trip" in text
    at = app({"section": "0.6", "part": "1"})
    text = sidebar_text(at)
    lab = text.split("Alchemy u02", 1)[1].split("Alchemy u03", 1)[0]
    assert "optimizer lab" in lab and "mastery_hub/app.py" in lab


def test_chapter_directions_come_from_the_route():
    at = app({"section": "8.1", "part": "1"})
    text = markdown_text(at)
    assert "Before you start" in text and "Start with" in text
    assert "?section=7.1&part=1" in text.split("Then", 1)[1]


def test_route_sidebar_stays_when_the_url_is_invalid():
    at = app({"section": "999.1"})
    assert not at.exception
    labels = expander_labels(at)
    assert sum(1 for label in labels if label.split(".", 1)[0].isdigit()) == 18


def test_addenda_in_original_chapters_get_no_chapter_directions():
    text = markdown_text(app({"section": "0.6", "part": "1"}))
    assert "before, during and after" not in text and "Start with" not in text

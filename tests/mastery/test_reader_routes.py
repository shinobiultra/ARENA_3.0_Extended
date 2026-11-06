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
    assert at.query_params["section"] == ["12.3"]
    assert at.query_params["part"] == ["3"]
    at.selectbox[1].select_index(0).run()
    assert not at.exception
    assert at.selectbox[1].value["number"] == "12.1"
    assert at.query_params["section"] == ["12.1"]
    assert at.query_params["part"] == ["1"]
    at.selectbox[0].select("chapter7_activation_to_language").run()
    assert not at.exception
    assert at.selectbox[1].value["number"] == "7.1"
    assert at.query_params["section"] == ["7.1"]
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
    for number in ("16.1", "16.2", "7.1", "12.1", "12.3", "9.1"):
        assert f"?section={number}&part=1" in text
    row = next(row for row in rows if row["number"] == "7.1")
    assert str(row["exercises"].parent) in text
    assert "PUBLIC Practice" in text and "worked answers" in text
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


def test_mini_vlm_route_exposes_memory_handoff_before_optional_bridge():
    # The finite allowlist is checked without importing or running model code.
    paths = [str(path.relative_to(ROOT)) for _, path in reader.supplements("12.3", ROOT)]
    assert "mastery_hub/practice_addons/mini_vlm_memory_handoff/README.md" in paths
    at = app({"section": "12.3", "part": "1"})
    assert not at.exception
    text = markdown_text(at)
    assert "Before the optional Step 5 Qwen bridge" in text
    assert "keeps earlier models resident" in text
    assert "6GB" not in text

"""Reader-visible annotations only; no lesson imports, kernels or model execution."""
from __future__ import annotations

import base64
import importlib
import importlib.util
from pathlib import Path
import re

import pytest

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
PAGES = {
    "7.1": "chapter7_activation_to_language/instructions/pages/01_[7.1]_Logit_Lens_Tuned_Lens_and_Patchscopes.md",
    "12.3": "chapter12_vlm_interpretability/instructions/pages/03_[12.3]_Mini_VLM_from_Scratch.md",
}
NOTE_BLOCK = re.compile(r"\n<!-- pilot-reader-note:([\w-]+):start -->\n.*?\n<!-- pilot-reader-note:\1:end -->\n", re.S)


def test_reader_composer_is_available():
    assert importlib.util.find_spec("infrastructure.extension.reader_notes") is not None, "Reader annotations are not implemented"


@pytest.fixture
def reader():
    # Fail by assertion while the candidate feature is absent, without collection errors.
    spec = importlib.util.find_spec("infrastructure.extension.reader_notes")
    assert spec is not None, "The pure reader-note composer has not been implemented"
    return importlib.import_module("infrastructure.extension.reader_notes")


@pytest.fixture
def notes(reader):
    path = Path(reader.__file__).resolve().parents[2] / "docs/mastery/pilot_reader_notes_20261023.md"
    assert path.is_file(), "The dated notes source is missing"
    return path.read_text(encoding="utf-8")


def page(section):
    path = ROOT / PAGES[section]
    return path, path.read_text(encoding="utf-8")


@pytest.mark.parametrize("section,expected", [("7.1", 3), ("12.3", 1)])
def test_annotations_leave_every_original_character_in_order(reader, notes, section, expected):
    path, source = page(section)
    rendered = reader.add_reader_notes(source, path, notes)
    assert len(NOTE_BLOCK.findall(rendered)) == expected
    assert NOTE_BLOCK.sub("", rendered) == source
    assert reader.add_reader_notes(rendered, path, notes) == rendered


def test_7_1_qualifications_precede_both_native_necessity_claims(reader, notes):
    path, source = page("7.1")
    rendered = reader.add_reader_notes(source, path, notes)
    grid = rendered.index("<summary>Help - how should I interpret this grid?</summary>")
    grid_note = rendered.index("<!-- pilot-reader-note:7-1-interpretation:start -->")
    grid_claim = rendered.index("The grid is effectively a map")
    summary = rendered.index("## Summary of 3️⃣")
    summary_note = rendered.index("<!-- pilot-reader-note:7-1-summary:start -->")
    summary_claim = rendered.index("The target-layer cutoff tells us")
    assert grid < grid_note < grid_claim < summary < summary_note < summary_claim
    assert rendered.count("does not establish necessity") >= 2


def test_reflection_allows_the_recorded_zero_success_case(reader, notes):
    path, source = page("7.1")
    rendered = reader.add_reader_notes(source, path, notes)
    block = re.search(r"<!-- pilot-reader-note:7-1-controls-reflection:start -->(.*?)<!-- pilot-reader-note:7-1-controls-reflection:end -->", rendered, re.S)[1]
    assert "no control successes" in block
    assert "historical" in block.lower()
    assert "raw notebooks" in block
    assert rendered.index("### Exercise - controls for the Patchscope") < rendered.index(block) < rendered.index("## Try it yourself - ask the hidden state")


def test_recovery_guard_is_visible_before_exercise_and_links_exact_addon(reader, notes):
    path, source = page("12.3")
    rendered = reader.add_reader_notes(source, path, notes)
    block = re.search(r"<!-- pilot-reader-note:12-3-recovery:start -->(.*?)<!-- pilot-reader-note:12-3-recovery:end -->", rendered, re.S)[1]
    assert rendered.index(block) < rendered.index("### Exercise - the recovery metric")
    assert "| 7 | 3 | -1 | 4 |" in block
    assert "| 9 | 0 | 0 | 0 |" in block
    assert "whole cohort" in block and "before averaging" in block
    href = "../../../mastery_hub/practice_addons/mini_vlm_checked_recovery/exercises.ipynb"
    assert href in block
    assert (path.parent / href).resolve().is_file()


@pytest.mark.parametrize("filename", [
    "chapter0_fundamentals/instructions/pages/01_[0.1]_Ray_Tracing.md",
    "chapter1_transformer_interp/instructions/pages/21_[1.4.1]_Indirect_Object_Identification.md",
    "chapter7_activation_to_language/instructions/pages/01_[7.1]_different.md",
    "chapter12_vlm_interpretability/instructions/pages/03_[12.3]_Mini_VLM_from_Scratch.py",
])
def test_originals_and_near_matches_are_exact_noops(reader, filename):
    text = "Original bytes stay here.\n## Summary of 3️⃣\n"
    assert reader.add_reader_notes(text, ROOT / filename, "not even valid notes") == text


@pytest.mark.parametrize("section,anchor", [
    ("7.1", "<summary>Help - how should I interpret this grid?</summary>"),
    ("7.1", "## Summary of 3️⃣"),
    ("7.1", "### Exercise - controls for the Patchscope"),
    ("12.3", "### Exercise - the recovery metric"),
])
@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_duplicate_anchor_skips_that_note_instead_of_misplacing_it(reader, notes, section, anchor, count):
    path, source = page(section)
    broken = source.replace(anchor, "heading moved") if count == 0 else source + "\n" + anchor + "\n"
    rendered = reader.add_reader_notes(broken, path, notes)
    expected_notes = len(reader._TARGETS[reader._key(path)]) - 1
    assert rendered.count("<!-- pilot-reader-note:") == 2 * expected_notes
    assert all(line in rendered for line in broken.splitlines())


def test_partial_annotation_is_an_actionable_error(reader, notes):
    path, source = page("7.1")
    with pytest.raises(ValueError, match="partial.*review"):
        reader.add_reader_notes(source + "\n<!-- pilot-reader-note:7-1-interpretation:start -->\n", path, notes)


def test_missing_note_section_does_not_silently_omit_qualification(reader, notes):
    path, source = page("7.1")
    broken = notes.replace("<!-- reader-note:7-1-summary:start -->", "<!-- removed -->")
    with pytest.raises(ValueError, match="note section.*expected once.*review"):
        reader.add_reader_notes(source, path, broken)



@pytest.mark.parametrize("section,expected", [("7.1", 3), ("12.3", 1)])
def test_public_renderer_includes_annotations_downloads_and_figures(tmp_path, reader, notes, section, expected):
    from infrastructure.extension.render_media import inline_page_media
    path, source = page(section)
    before = path.read_bytes()
    # Build a tiny deployment tree containing exact page/figure/download bytes.
    staged_page = tmp_path / PAGES[section]
    staged_page.parent.mkdir(parents=True)
    staged_page.write_bytes(before)
    files = {"docs/mastery/pilot_reader_notes_20261023.md": notes.encode()}
    if section == "12.3":
        rel = "mastery_hub/practice_addons/mini_vlm_checked_recovery/exercises.ipynb"
        files[rel] = (ROOT / rel).read_bytes()
    for href in re.findall(r"src=[\"'](\.\./\.\./exercises/[^\"']+)[\"']", source):
        original = (path.parent / href).resolve()
        files[str(original.relative_to(ROOT))] = original.read_bytes()
    for rel, payload in files.items():
        destination = tmp_path / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    rendered = inline_page_media(source, staged_page)
    assert len(NOTE_BLOCK.findall(rendered)) == expected
    assert "data:image/png;base64," in rendered
    downloads = re.findall(r'<a download="([^"]+)" href="data:[^;]+;base64,([^"]+)"', rendered)
    assert downloads
    payloads = {filename: base64.b64decode(encoded) for filename, encoded in downloads}
    assert payloads["pilot_reader_notes_20261023.md"] == notes.encode()
    if section == "12.3":
        assert payloads["exercises.ipynb"] == files["mastery_hub/practice_addons/mini_vlm_checked_recovery/exercises.ipynb"]
    assert path.read_bytes() == staged_page.read_bytes() == before


def test_reading_note_markdown_renders_inside_original_help_details(reader, notes):
    markdown_it = pytest.importorskip("markdown_it")
    path, source = page("7.1")
    annotated = reader.add_reader_notes(source, path, notes)
    html = markdown_it.MarkdownIt("commonmark", {"html": True}).render(annotated)
    summary = "<summary>Help - how should I interpret this grid?</summary>"
    details_body = html.split(summary, 1)[1].split("</details>", 1)[0]
    assert "<strong>Reading note — 23 October 2026.</strong>" in details_body
    assert "**Reading note — 23 October 2026.**" not in details_body
    # The same parsing check still leaves every original lesson character intact.
    assert NOTE_BLOCK.sub("", annotated) == source

"""Reader-visible annotations only; no lesson imports, kernels or model execution."""
from __future__ import annotations

import base64
import hashlib
import importlib
import importlib.util
import json
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
def test_missing_or_duplicate_anchor_stops_instead_of_misplacing_note(reader, notes, section, anchor, count):
    path, source = page(section)
    broken = source.replace(anchor, "heading moved") if count == 0 else source + "\n" + anchor + "\n"
    with pytest.raises(ValueError, match="anchor.*expected once.*review"):
        reader.add_reader_notes(broken, path, notes)


def test_partial_annotation_is_an_actionable_error(reader, notes):
    path, source = page("7.1")
    with pytest.raises(ValueError, match="partial.*review"):
        reader.add_reader_notes(source + "\n<!-- pilot-reader-note:7-1-interpretation:start -->\n", path, notes)


def test_missing_note_section_does_not_silently_omit_qualification(reader, notes):
    path, source = page("7.1")
    broken = notes.replace("<!-- reader-note:7-1-summary:start -->", "<!-- removed -->")
    with pytest.raises(ValueError, match="note section.*expected once.*review"):
        reader.add_reader_notes(source, path, broken)


def test_original_source_pins_and_addon_payloads_still_match_exact_bytes():
    manifest = json.loads((ROOT / "mastery_hub/units/pilot_practice/original_source_manifest.json").read_text())
    for rel, record in manifest["files"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == record["sha256"], rel
    addon = ROOT / "mastery_hub/practice_addons/mini_vlm_checked_recovery"
    manifest = json.loads((addon / "source_manifest.json").read_text())
    for rel, expected in manifest["addon_files"].items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected, rel


def test_annotation_cannot_mutate_current_addon_inputs_or_pin_manifests(reader, notes):
    addon_manifest = ROOT / "mastery_hub/practice_addons/mini_vlm_checked_recovery/source_manifest.json"
    manifest = json.loads(addon_manifest.read_text())
    # This list is a historical preflight inventory; the ROME registry transition is
    # recorded in combined_reviewed53_20261021. Check current byte preservation,
    # not that an older global registry hash describes the current repository.
    paths = [ROOT / row["path"] for row in manifest["unchanged_canonical_inputs"]]
    paths += [addon_manifest, ROOT / "mastery_hub/units/pilot_practice/original_source_manifest.json"]
    before = {path: path.read_bytes() for path in paths}
    for section in PAGES:
        path, source = page(section)
        reader.add_reader_notes(source, path, notes)
    assert {path: path.read_bytes() for path in paths} == before


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


def test_dated_note_wrapping_and_download_pins_preserve_archived_source(reader):
    import tarfile

    from markdown_it import MarkdownIt
    from infrastructure.extension import page_links
    from scripts.audit_no_minified_files import minified_file_blockers

    notes_path = Path(reader.__file__).resolve().parents[2] / "docs/mastery/pilot_reader_notes_20261023.md"
    helper_path = Path(page_links.__file__).resolve()
    archive = ROOT / "docs/mastery/evidence/continuation_20261023/source_routes_v1/finished_v1.tar.gz"
    prefix = "payloads/routes9/"
    with tarfile.open(archive, "r:gz") as frozen:
        original_notes = frozen.extractfile(prefix + "docs/mastery/pilot_reader_notes_20261023.md").read()
        original_helper = frozen.extractfile(prefix + "infrastructure/extension/page_links.py").read()
    old_sha = "ee0c27e7573c549f3669433b3d67249b363324ecff42a68150ad59a3c86e9dec"
    assert hashlib.sha256(original_notes).hexdigest() == old_sha
    assert hashlib.sha256(original_helper).hexdigest() == "6fb30a3fbb4d75cfc602278c528589c71a891eb741ec515c2c44edf98fdee101"
    current_notes = notes_path.read_bytes()
    assert current_notes.split() == original_notes.split()
    assert minified_file_blockers([notes_path, helper_path]) == []
    new_sha = hashlib.sha256(current_notes).hexdigest()
    assert original_helper.count(old_sha.encode()) == 2
    assert helper_path.read_bytes() == original_helper.replace(old_sha.encode(), new_sha.encode())
    parser = MarkdownIt("commonmark", {"html": True})
    normalize = lambda html: re.sub(r"\s+", " ", html).strip()
    assert normalize(parser.render(current_notes.decode())) == normalize(parser.render(original_notes.decode()))

"""Fail closed at the public reader boundary; never execute lesson notebooks."""
from pathlib import Path
import hashlib
import json

import pytest
from streamlit.testing.v1 import AppTest
import yaml

from infrastructure.extension import blank_practice
from infrastructure.extension import teaching_editions as editions
from mastery_hub import read as reader

ROOT = Path(__file__).resolve().parents[2]


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


@pytest.fixture
def delivery_lane(tmp_path):
    """A complete inert configured lesson, including real blank-download authority."""
    paths = {
        "page": "chapter5/instructions/pages/01_[5.1]_Fixture.md",
        "exercises": "chapter5/exercises/part1_fixture/5.1_Fixture_exercises.ipynb",
        "reference": "chapter5/exercises/part1_fixture/5.1_Fixture_solutions.ipynb",
    }
    notebook = {
        "nbformat": 4, "nbformat_minor": 5, "metadata": {"historical": "retained"},
        "cells": [
            {"cell_type": "markdown", "id": "explanation", "metadata": {},
             "source": ["Original fixture explanation.\n"]},
            {"cell_type": "code", "id": "core", "metadata": {"historical": True},
             "source": ["raise RuntimeError('inert: never execute this lesson')\n"],
             "execution_count": 7,
             "outputs": [{"output_type": "stream", "name": "stdout", "text": ["historical result\n"]}]},
        ],
    }
    raw = {"page": b"# Fixture signature\nOriginal fixture explanation.\n", 
           "exercises": encoded(notebook), "reference": encoded(notebook)}
    for kind, relative in paths.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw[kind])
    files = {
        "infrastructure/extension/syllabus.yaml": yaml.safe_dump({"chapters": {
            "chapter5": {"number": 5, "title": "Fixture chapter", "sections": [
                {"number": "5.1", "title": "Fixture lesson"}]}}}),
        "infrastructure/core/config_extension.yaml": yaml.safe_dump({"conversion_map": {
            "5.1": {"streamlit_page": "01_[5.1]_Fixture", "exercise_dir": "part1_fixture"}}}),
        "infrastructure/chapters/chapter5/master_5_1.py": "# Inert authoring fixture\n",
        "mastery_hub/STUDY-GUIDE.md": "Fixture study guide.\n",
    }
    for relative, text in files.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    manifest = {
        "schema_version": 1, "section": "5.1",
        "sources": {kind: {"path": relative, "sha256": hashlib.sha256(raw[kind]).hexdigest()}
                    for kind, relative in paths.items()},
        "replacements": [{"old": "Original fixture explanation.", "new": "Revised fixture explanation.",
                          "counts": {"page": 1, "exercises": 1, "reference": 1}}],
    }
    manifest_path = tmp_path / "mastery_hub/teaching_editions/5_1.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_bytes(encoded(manifest))
    return tmp_path, reader.inventory(tmp_path)[0], raw, manifest, manifest_path


def fixture_app(root):
    # The production rendering function, selectors and URL route run in Streamlit.
    at = AppTest.from_string(
        "import streamlit as st\nfrom pathlib import Path\n"
        "from mastery_hub.read import render\n"
        f"render(st, Path({str(root)!r}))\n", default_timeout=30,
    )
    at.query_params.update({"section": "5.1", "part": "1"})
    return at.run()


def assert_unavailable(at, detail):
    assert not at.exception
    info = "\n".join(element.value for element in at.info)
    assert "regeneration" in info
    assert "report" in info and "preserved source files" in info
    assert detail in info
    assert not at.get("download_button"), "No partial or mismatched download set may escape"
    assert len(at.selectbox) == 2, "Failure must precede the part selector"
    text = "\n".join(element.value for element in at.markdown)
    assert "Start with blank Practice" not in text
    assert "Fixture signature" not in text
    assert "Original fixture explanation." not in text
    assert "Revised fixture explanation." not in text
    assert "Resume / bookmark this part" not in text


@pytest.mark.parametrize("failure", ["malformed", "stale-page", "stale-exercises", "stale-reference"])
def test_actual_edition_load_failure_is_actionable_without_fallback(delivery_lane, failure):
    root, row, raw, _, manifest_path = delivery_lane
    if failure == "malformed":
        manifest_path.write_bytes(b'{"schema_version":')
        detail = "Malformed UTF-8 teaching-edition JSON"
    else:
        kind = failure.removeprefix("stale-")
        row[kind].write_bytes(raw[kind] + b" ")
        detail = f"Stale teaching-edition {kind} source pin"
    assert_unavailable(fixture_app(root), detail)


@pytest.mark.parametrize("kind", ["page", "exercises", "reference"])
def test_actual_preimage_failure_is_actionable_before_any_download(delivery_lane, kind):
    root, _, _, manifest, manifest_path = delivery_lane
    # Manifest remains valid and pinned; its finite count is impossible only for
    # the chosen projection, so earlier stages really succeed before this fails.
    manifest["replacements"][0]["counts"][kind] = 2
    manifest_path.write_bytes(encoded(manifest))
    assert_unavailable(fixture_app(root), f"{kind} preimage count: expected 2, found 1")


@pytest.mark.parametrize("stage", ["load", "blank", "exercises", "reference", "page"])
def test_public_script_route_catches_every_delivery_valueerror(monkeypatch, stage):
    """Exercise the production __main__ path, including real URL routing."""
    message = f"inert injected {stage} delivery failure"
    calls = []
    if stage == "load":
        def failing_load(*args, **kwargs):
            calls.append("load")
            raise ValueError(message)
        monkeypatch.setattr(editions, "load_edition", failing_load)
    elif stage == "blank":
        def failing_blank(*args, **kwargs):
            calls.append("blank")
            raise ValueError(message)
        monkeypatch.setattr(blank_practice, "download", failing_blank)
    elif stage == "page":
        def failing_page(*args, **kwargs):
            calls.append("page")
            raise ValueError(message)
        monkeypatch.setattr(editions, "apply_page", failing_page)
    else:
        project = editions.project_notebook
        def failing_project(spec, raw, kind, *args, **kwargs):
            calls.append(kind)
            if kind == stage:
                raise ValueError(message)
            return project(spec, raw, kind, *args, **kwargs)
        monkeypatch.setattr(editions, "project_notebook", failing_project)
    at = AppTest.from_file(str(ROOT / "mastery_hub/read.py"), default_timeout=30)
    at.query_params.update({"section": "7.1", "part": "1"})
    assert_unavailable(at.run(), message)
    assert calls == (["exercises", "reference"] if stage == "reference" else [stage])
    assert at.selectbox[1].value["number"] == "7.1"


class DownloadUI:
    """Capture the exact public download payloads without reaching private APIs."""
    def __init__(self, row):
        self.row = row
        self.sidebar = self
        self.query_params = {"section": row["number"], "part": "1"}
        self.session_state = {}
        self.downloads = []
        self.infos = []
    def set_page_config(self, **kwargs): pass
    def title(self, text): pass
    def markdown(self, text, **kwargs): pass
    def info(self, text): self.infos.append(text)
    def selectbox(self, label, options, **kwargs):
        return self.row["chapter"] if label == "Chapter" else self.row if label == "Lesson" else 0
    def download_button(self, label, data, file_name, **kwargs):
        self.downloads.append((label, data, file_name))


def test_valid_delivery_keeps_exact_originals_revised_payloads_and_both_inverses(delivery_lane):
    root, row, raw, _, _ = delivery_lane
    before = {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    ui = DownloadUI(row)
    reader.render(ui, root)
    assert not ui.infos
    downloads = {label: (data, name) for label, data, name in ui.downloads}
    assert list(downloads)[:2] == ["Download revised Study Practice", "Download revised worked reference"]
    spec = editions.load_edition(root, row)
    blank, blank_name = downloads["Download blank Practice exercises"]
    assert (blank, blank_name) == reader.blank_practice_download(root, "5.1")
    for kind, label in (("exercises", "Download exercises"), ("reference", "Download worked reference")):
        assert downloads[label] == (raw[kind], row[kind].name)
        revised_label = "Download revised Study Practice" if kind == "exercises" else "Download revised worked reference"
        payload, filename = downloads[revised_label]
        parent = blank if kind == "exercises" else None
        assert payload == editions.project_notebook(spec, raw[kind], kind, base_bytes=parent)
        assert "_Study_" in filename
        assert editions.restore_notebook(spec, payload, raw[kind], kind, base_bytes=parent) == (blank if parent is not None else raw[kind])
        assert json.loads(payload)["metadata"]["arena_teaching_edition"]["execution_claim"] == "none"
    restored = reader.restore_blank_practice(blank, raw["exercises"], row["exercises"].relative_to(root).as_posix(), reader.blank_config_identity(root))
    assert restored == json.loads(raw["exercises"])
    assert before == {p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    at = fixture_app(root)
    assert not at.exception and not at.info
    assert len(at.get("download_button")) == len(ui.downloads)
    assert "Revised fixture explanation." in "\n".join(element.value for element in at.markdown)

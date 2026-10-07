"""Structural checks for extension sections built from master files.

These are cheap: they don't run any models. The real check that a section works is
`infrastructure/extension/build.py N.M --execute`, which runs the solutions notebook top to bottom
and saves its outputs; here we check that the committed files are what the master file generates,
that the solutions notebook was executed without errors, and that the section directory looks like
an original ARENA section.
"""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import nbformat
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infrastructure" / "extension"))
sys.path.insert(0, str(ROOT / "infrastructure" / "core"))

import build  # noqa: E402

SYLLABUS = yaml.safe_load((ROOT / "infrastructure" / "extension" / "syllabus.yaml").read_text())["chapters"]
CONFIG = yaml.safe_load((ROOT / "infrastructure" / "core" / "config.yaml").read_text())

ALLOWED_FILES = {"solutions.py", "tests.py", "utils.py"}
ALLOWED_DIRS = {"media", "data", "__pycache__"}


def built_sections() -> list[tuple[str, str, dict]]:
    out = []
    for chapter_name, chapter in SYLLABUS.items():
        for section in chapter["sections"]:
            name = f"master_{section['number'].replace('.', '_')}.py"
            if list((ROOT / "infrastructure" / "chapters").rglob(name)):
                out.append((chapter_name, section["number"], section))
    return out


SECTIONS = built_sections()
IDS = [number for _, number, _ in SECTIONS]


def section_paths(chapter_name: str, number: str) -> tuple[Path, str]:
    mapping = CONFIG["conversion_map"][number]
    exercise_dir = ROOT / chapter_name / "exercises" / mapping["exercise_dir"]
    colab_name = mapping["streamlit_page"].split("_[")[-1].replace("]", "")
    return exercise_dir, colab_name


def normalised_cells(path: Path) -> list[tuple[str, str]]:
    nb = nbformat.read(path, as_version=4)
    return [(cell.cell_type, cell.source) for cell in nb.cells]


@pytest.mark.parametrize("chapter_name,number,section", SECTIONS, ids=IDS)
def test_generated_files_match_master(tmp_path, chapter_name, number, section):
    """The committed notebooks and solutions.py must be exactly what the master file generates."""
    exercise_dir, colab_name = section_paths(chapter_name, number)
    master = build.find_master(number)
    tmp_master_dir = tmp_path / "masters"
    tmp_master_dir.mkdir()
    shutil.copy(master, tmp_master_dir / master.name)

    mapping = CONFIG["conversion_map"][number]
    chapter_num = int(number.split(".")[0])
    tmp_ipynb = tmp_master_dir / master.with_suffix(".ipynb").name
    tmp_ipynb.write_text('{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}')
    generated = build.ExtensionMasterFile(
        master_path=tmp_ipynb,
        chapter_dir=tmp_path / chapter_name,
        chapter_name_long=CONFIG["chapter_names_long"][chapter_num],
        exercise_dir_name=mapping["exercise_dir"],
        streamlit_page_name=mapping["streamlit_page"],
    )
    generated.exercises_dir.mkdir(parents=True)
    generated.instructions_dir.mkdir(parents=True)
    generated.master_py_to_ipynb(overwrite=True)
    generated.generate_files(overwrite=True, verbose=False)

    for suffix in ["_exercises.ipynb", "_solutions.ipynb"]:
        committed = normalised_cells(exercise_dir / f"{colab_name}{suffix}")
        fresh = normalised_cells(generated.exercises_dir / f"{colab_name}{suffix}")
        assert committed == fresh, f"{colab_name}{suffix} is out of date: rebuild with build.py {number}"
    assert (exercise_dir / "solutions.py").read_text() == (generated.exercises_dir / "solutions.py").read_text(), (
        f"solutions.py is out of date: rebuild with build.py {number}"
    )


@pytest.mark.parametrize("chapter_name,number,section", SECTIONS, ids=IDS)
def test_solutions_notebook_was_executed(chapter_name, number, section):
    exercise_dir, colab_name = section_paths(chapter_name, number)
    nb = nbformat.read(exercise_dir / f"{colab_name}_solutions.ipynb", as_version=4)
    code_cells = [cell for cell in nb.cells if cell.cell_type == "code"]
    executed = [cell for cell in code_cells if cell.get("execution_count")]
    assert len(executed) >= 0.9 * len(code_cells), f"Solutions notebook for {number} hasn't been executed (build.py {number} --execute)"
    errors = [out for cell in code_cells for out in cell.get("outputs", []) if out.get("output_type") == "error"]
    assert not errors, f"Solutions notebook for {number} contains errors: {errors[0].get('ename')}"


@pytest.mark.parametrize("chapter_name,number,section", SECTIONS, ids=IDS)
def test_exercise_notebook_shape(chapter_name, number, section):
    exercise_dir, colab_name = section_paths(chapter_name, number)
    nb = nbformat.read(exercise_dir / f"{colab_name}_exercises.ipynb", as_version=4)
    markdown = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
    code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    assert "## Content & Learning Objectives" in markdown
    assert re.search(r"^# 1️⃣ ", markdown, flags=re.M), "Sections should be split into numbered parts (# 1️⃣ ...)"
    assert len(re.findall(r"^#+ Exercise", markdown, flags=re.M)) >= 5, "Expected at least 5 exercises"
    assert "NotImplementedError" in code, "Exercise stubs should raise NotImplementedError"
    assert "arena_ext" not in code, "Sections must not depend on the old arena_ext prototype library"
    assert "to_str_tokens" not in code, "model.to_str_tokens is broken with transformers 5; use model.to_string per token"


@pytest.mark.parametrize("chapter_name,number,section", SECTIONS, ids=IDS)
def test_section_directory_is_clean(chapter_name, number, section):
    exercise_dir, colab_name = section_paths(chapter_name, number)
    unexpected = []
    for path in exercise_dir.iterdir():
        if path.is_dir():
            if path.name not in ALLOWED_DIRS:
                unexpected.append(path.name + "/")
        elif path.name not in ALLOWED_FILES and not path.name.startswith(colab_name) and path.suffix not in {".csv", ".json", ".txt", ".npz", ".pt", ".png"}:
            unexpected.append(path.name)
    assert not unexpected, f"Unexpected files in {exercise_dir.relative_to(ROOT)}: {unexpected}"
    assert not list(exercise_dir.glob("*.FAILED.ipynb")), "Leftover failed-execution notebook"


@pytest.mark.parametrize("chapter_name,number,section", SECTIONS, ids=IDS)
def test_expected_figures_exist(chapter_name, number, section):
    exercise_dir, _ = section_paths(chapter_name, number)
    master_text = build.find_master(number).read_text()
    names = set(build.MEDIA_PATTERN.findall(master_text))
    missing = [name for name in names if not (exercise_dir / "media" / f"{name}.png").exists()]
    assert not missing, f"Figures referenced by the master but not generated: {missing}"


def test_every_syllabus_section_is_in_the_config():
    for chapter in SYLLABUS.values():
        for section in chapter["sections"]:
            assert section["number"] in CONFIG["conversion_map"], f"Run infrastructure/extension/sync_config.py ({section['number']} missing)"

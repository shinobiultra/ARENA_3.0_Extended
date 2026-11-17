"""Extension lessons are run locally: no Colab links and no Colab setup cells (#311)."""

import sys
from pathlib import Path

import nbformat
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infrastructure" / "extension"))
sys.path.insert(0, str(ROOT / "infrastructure" / "core"))

import build

CONFIG = yaml.safe_load(
    (ROOT / "infrastructure" / "core" / "config.yaml").read_text(encoding="utf-8")
)
EXTENSION = [
    s
    for s in CONFIG["conversion_map"]
    if int(s.split(".")[0]) >= build.FIRST_EXTENSION_CHAPTER or s in ("0.6", "1.6")
]
COLAB_MARKERS = ("colab.research.google.com", "google.colab", "IN_COLAB")


def _lesson_files(section: str) -> list[Path]:
    mapping = CONFIG["conversion_map"][section]
    chapter_dir = ROOT / CONFIG["chapter_names"][int(section.split(".")[0])]
    exercise_dir = chapter_dir / "exercises" / mapping["exercise_dir"]
    page = chapter_dir / "instructions" / "pages" / f"{mapping['streamlit_page']}.md"
    return [page, *sorted(exercise_dir.glob("*.ipynb"))]


def test_page_links_are_local(tmp_path):
    master_path = tmp_path / "master_7_1.ipynb"
    master_path.write_text("{}")
    master_path.with_suffix(".py").write_text("")
    master = build.ExtensionMasterFile(
        master_path=master_path,
        chapter_dir=ROOT / "chapter7_activation_to_language",
        chapter_name_long="Chapter 7 - Activation to Language",
        exercise_dir_name="part1_lenses_patchscopes",
        streamlit_page_name="01_[7.1]_Logit_Lens_Tuned_Lens_and_Patchscopes",
    )
    streamlit, notebooks = master.get_page_links()
    text = "\n".join(streamlit + notebooks)
    assert "colab" not in text.lower()
    assert "7.1_Logit_Lens_Tuned_Lens_and_Patchscopes_exercises.ipynb" in text


@pytest.mark.parametrize("section", EXTENSION)
def test_master_has_no_colab_setup(section):
    master = build.find_master(section).read_text(encoding="utf-8")
    assert "IN_COLAB" not in master and "google.colab" not in master


@pytest.mark.parametrize("section", EXTENSION)
def test_shipped_lesson_has_no_colab(section):
    offenders = []
    for path in _lesson_files(section):
        if path.suffix == ".ipynb":
            nb = nbformat.read(path, as_version=4)
            text = "\n".join(cell.source for cell in nb.cells)
        else:
            text = path.read_text(encoding="utf-8")
        offenders += [f"{path.relative_to(ROOT)}: {m}" for m in COLAB_MARKERS if m in text]
    assert not offenders, offenders

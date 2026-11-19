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
# Small data files a lesson reads may sit beside the notebooks.
DATA_SUFFIXES = {".csv", ".json", ".txt", ".npz", ".pt", ".png"}
# Helper modules the notebook imports (Spec #39, Lesson files: "imported helpers").
SECTION_HELPER_FILES = {
    "5.1": {"gemma_reference.py", "legacy_solutions.py"},
    "5.5": {"mamba_backbone.py"},
    "5.8": {"tensor_plumbing.py", "text_data.py", "text_diffusion_methods.py", "text_diffusion_tests.py",
            "text_evaluation.py", "text_evaluation_tests.py", "text_model.py", "text_model_tests.py",
            "text_plotting.py", "text_training.py", "text_training_tests.py"},
    "6.1": {"sae_plumbing.py"},
    "6.5": {"crosscoder_plumbing.py"},
    "9.4": {"validation_reference.py"},
    "9.5": {"validation_reference.py"},
    "9.6": {"rome_metadata.py"},
    "11.1": {"geometry_data.py", "geometry_methods.py", "geometry_methods_tests.py", "geometry_plotting.py"},
    "11.2": {"alignment_ioi_data.py", "alignment_methods.py", "alignment_methods_tests.py"},
    "11.3": {"das_ioi_data.py", "das_methods.py", "das_methods_tests.py", "das_native.py", "das_native_tests.py",
             "das_organism.py", "das_protocol.py", "das_training_methods.py", "das_training_tests.py",
             "run_das_screen.py", "run_native_development.py"},
    "11.4": {"erasure_data.py", "erasure_methods.py", "erasure_methods_tests.py", "erasure_native.py",
             "erasure_protocol.py", "reference_parity.py"},
    "11.5": {"ccs_data.py", "ccs_methods.py", "ccs_methods_tests.py", "ccs_native.py", "ccs_native_methods.py",
             "ccs_native_methods_tests.py", "reference_parity.py"},
    "13.1": {"lesson_reference.py"},
    "13.2": {"legacy_reference.py"},
    "14.1": {"ijepa_plumbing.py"},
    "14.2": {"controls_legacy_solutions.py"},
    "15.3": {"legacy_reference.py"},
    "16.3": {f"{part}_{name}" for part, name in (("tree", "treeshap"), ("deep", "deepshap"), ("gradient", "gradientshap"))
             for name in (f"{name}_data.py", f"{name}_methods_tests.py", f"{name}_plots.py")},
    "16.8": {"agreement_tests.py", "agreement_utils.py", "relu_tests.py"},
    "17.1": {"checkpoint_reports.py"},
}
# Data folders some lessons read.
SECTION_DATA_DIRECTORIES = {"5.8": {"artifacts"}, "8.6": {"assets"}, "12.1": {"fixtures"}}
# Retired verification and custody files (#249 CC5): never ship in a lesson folder.
RETIRED_NAME = re.compile(
    r"verification_report|artifacts\.lock|README|historical|expected_outputs|development_evidence"
    r"|review-bundle|own_execution_contract|native_protocol|^lesson\.md$|^publish_|^execute_|^build_notebooks"
)


def _uses_arena_ext_namespace(code: str) -> bool:
    """Find exact prototype names in source and statically constant string additions.

    This bounded source check does not resolve variables or runtime expressions.
    """
    import ast

    def constant_string(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left = constant_string(node.left)
            right = constant_string(node.right)
            if left is not None and right is not None:
                return left + right
        return None

    candidates = [code]
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    if tree is not None:
        for node in ast.walk(tree):
            value = constant_string(node)
            if value is not None:
                candidates.append(value)
    return any(re.search(r"\barena_ext\b", candidate) is not None for candidate in candidates)

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
    counts = [cell.get("execution_count") for cell in code_cells]
    # A fresh-kernel run executes every code cell once, in order.
    assert counts == list(range(1, len(code_cells) + 1)), (
        f"Solutions notebook for {number} is not one complete fresh-kernel run "
        f"(build.py {number} --execute): execution counts {counts[:8]}..."
    )
    errors = [out for cell in code_cells for out in cell.get("outputs", []) if out.get("output_type") == "error"]
    assert not errors, f"Solutions notebook for {number} contains errors: {errors[0].get('ename')}"


@pytest.mark.parametrize("chapter_name,number,section", SECTIONS, ids=IDS)
def test_exercise_notebook_shape(chapter_name, number, section):
    exercise_dir, colab_name = section_paths(chapter_name, number)
    nb = nbformat.read(exercise_dir / f"{colab_name}_exercises.ipynb", as_version=4)
    assert all(cell.execution_count is None for cell in nb.cells if cell.cell_type == "code"), (
        "Exercise stubs must be unexecuted; preserve reference execution counts in solutions only"
    )
    markdown = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "markdown")
    code = "\n".join(cell.source for cell in nb.cells if cell.cell_type == "code")
    assert "## Content & Learning Objectives" in markdown
    assert re.search(r"^# 1️⃣ ", markdown, flags=re.M), "Sections should be split into numbered parts (# 1️⃣ ...)"
    assert len(re.findall(r"^#+ Exercise", markdown, flags=re.M)) >= 5, "Expected at least 5 exercises"
    assert "NotImplementedError" in code, "Exercise stubs should raise NotImplementedError"
    assert not _uses_arena_ext_namespace(code), "Sections must not depend on the old arena_ext prototype library"
    assert "to_str_tokens" not in code, "model.to_str_tokens is broken with transformers 5; use model.to_string per token"


@pytest.mark.parametrize("chapter_name,number,section", SECTIONS, ids=IDS)
def test_section_directory_is_clean(chapter_name, number, section):
    exercise_dir, colab_name = section_paths(chapter_name, number)
    """The folder ships the original ARENA set (#249 CC5): notebooks, solutions/tests/utils, the helper
    modules the notebook imports, media, and small data files; no retired verification files."""
    data_dirs = SECTION_DATA_DIRECTORIES.get(number, set())
    unexpected = []
    for path in exercise_dir.iterdir():
        if path.is_dir():
            if path.name not in ALLOWED_DIRS | data_dirs:
                unexpected.append(path.name + "/")
        elif not (
            path.name in ALLOWED_FILES | SECTION_HELPER_FILES.get(number, set())
            or (path.name.startswith(colab_name) and path.suffix == ".ipynb")
            or path.suffix in DATA_SUFFIXES
        ):
            unexpected.append(path.name)
    for path in exercise_dir.rglob("*"):
        relative = path.relative_to(exercise_dir)
        if relative.parts[0] in data_dirs or "__pycache__" in relative.parts:
            continue
        if RETIRED_NAME.search(path.name):
            unexpected.append(str(relative))
    unexpected = sorted(set(unexpected))
    assert not unexpected, f"Unexpected files in {exercise_dir.relative_to(ROOT)}: {unexpected}"
    assert not list(exercise_dir.glob("*.FAILED.ipynb")), "Leftover failed-execution notebook"


@pytest.mark.parametrize("chapter_name,number,section", SECTIONS, ids=IDS)
def test_expected_figures_exist(chapter_name, number, section):
    exercise_dir, _ = section_paths(chapter_name, number)
    master_text = build.find_master(number).read_text()
    names = set(build.MEDIA_PATTERN.findall(master_text))
    missing = [name for name in names if not (exercise_dir / "media" / f"{name}.png").exists()]
    assert not missing, f"Figures referenced by the master but not generated: {missing}"


_LINK = re.compile(r"""!?\[[^\]]*\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)|\b(?:src|href)=["']([^"']+)["']""")
_FENCE = re.compile(r"```.*?```", re.S)


def local_references(markdown: str) -> list[str]:
    """Relative link and image targets in markdown, ignoring code fences, URLs and anchors."""
    from urllib.parse import unquote

    targets = []
    for match in _LINK.finditer(_FENCE.sub("", markdown)):
        target = match.group(1) or match.group(2)
        if re.match(r"^(?:[a-z][a-z0-9+.-]*:|#|//)", target, re.I):
            continue
        target = unquote(target.split("#")[0].split("?")[0])
        if target:
            targets.append(target)
    return targets


@pytest.mark.parametrize("chapter_name,number,section", SECTIONS, ids=IDS)
def test_referenced_files_exist(chapter_name, number, section):
    """Every local file the lesson links to or shows exists, seen from its exercises notebook.

    Masters write relative links from the exercise folder (where Jupyter resolves them);
    the build rewrites figure paths for the instruction page.
    """
    exercise_dir, colab_name = section_paths(chapter_name, number)
    notebook = nbformat.read(exercise_dir / f"{colab_name}_exercises.ipynb", as_version=4)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    missing = sorted({target for target in local_references(markdown) if not (exercise_dir / target).exists()})
    assert not missing, f"{colab_name}_exercises.ipynb references missing files: {missing}"


def test_every_syllabus_section_is_in_the_config():
    for chapter in SYLLABUS.values():
        for section in chapter["sections"]:
            assert section["number"] in CONFIG["conversion_map"], f"Run infrastructure/extension/sync_config.py ({section['number']} missing)"

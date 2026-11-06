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
# Extra files tolerated in lesson folders until the lesson-folder cleanup slice
# removes them (verification reports, lock files, READMEs, historical copies).
# They are allowed, not required; new arbitrary files elsewhere still fail.
RETAINED_HISTORICAL_FILES = {
    "5.1": {"README.md", "artifacts.lock.yml", "legacy_solutions.py"},
    "5.3": {"README.md", "artifacts.lock.yml", "historical_release.py"},
    "5.5": {"README.md", "artifacts.lock.yml"},
    "5.8": {"README.md", "artifacts.lock.yml", "text_model.py", "text_plotting.py",
            "text_diffusion_methods.py", "text_evaluation_tests.py", "study_evidence.py",
            "text_evaluation.py", "study_evidence_tests.py", "text_diffusion_tests.py",
            "text_training.py", "text_model_tests.py", "study_release.py", "text_data.py",
            "text_data_tests.py", "text_training_tests.py"},
    "5.4": {"artifacts.lock.yml"},
    "6.4": {
        "README.md", "historical_artifacts.lock.yml",
        "historical_verification_report.json", "historical_verification_report.schema.json",
    },
    "6.5": {
        "README.md", "historical_artifacts.lock.yml",
        "historical_verification_report.json", "historical_verification_report.schema.json",
    },
    "9.3": {
        "README.md", "historical_9_2_artifacts.lock.yml",
        "historical_9_2_verification_report.json",
    },
    "11.1": {
        "geometry_study.py", "execute_learner.py", "README.md", "lesson.md",
        "geometry_cuda.py", "geometry_methods_tests.py", "geometry_reproduction_cache.py",
        "build_notebooks.py", "publish_study.py", "geometry_methods.py",
        "geometry_plotting.py", "geometry_evidence.py", "artifacts.lock.yml", "geometry_data.py",
    },
    "11.2": {
        "README.md", "alignment_data.py", "alignment_evidence.py", "alignment_ioi_data.py",
        "alignment_methods.py", "alignment_methods_tests.py", "alignment_model.py",
        "alignment_reproduce.py", "alignment_study.py", "artifacts.lock.yml",
        "build_notebooks.py", "execute_learner.py", "lesson.md", "publish_study.py",
        "run_alignment_development.py",
    },
    "11.3": {
        "README.md", "artifacts.lock.yml", "build_notebooks.py", "das_ioi_data.py",
        "das_methods.py", "das_methods_tests.py", "das_native.py", "das_native_tests.py",
        "das_organism.py", "das_protocol.py", "das_reproduction.py", "das_reproduction_tests.py",
        "das_training_methods.py", "das_training_tests.py", "execute_learner.py", "lesson.md",
        "native_protocol.md", "publish_study.py", "run_das_screen.py",
        "run_native_development.py", "run_organism_development.py",
    },
    "11.4": {
        "README.md", "artifacts.lock.yml", "build_notebooks.py", "erasure_data.py",
        "erasure_methods.py", "erasure_methods_tests.py", "erasure_native.py",
        "erasure_protocol.py", "erasure_reproduction.py", "erasure_reproduction_tests.py",
        "execute_learner.py", "lesson.md", "native_protocol.md", "publish_study.py",
        "reference_parity.py", "run_development.py",
    },
    "11.5": {
        "ccs_native_methods.py", "ccs_study.py", "run_development.py", "execute_learner.py",
        "ccs_methods.py", "README.md", "lesson.md", "reference_parity.py", "native_protocol.md",
        "ccs_methods_tests.py", "build_notebooks.py", "ccs_data.py", "publish_study.py",
        "ccs_native.py", "artifacts.lock.yml", "ccs_reproduction.py",
        "ccs_reproduction_tests.py", "ccs_native_methods_tests.py",
        "11.5_Contrast_Consistent_Search_exercises.ipynb",
        "11.5_Contrast_Consistent_Search_solutions.ipynb",
    },
    "17.1": {
        "17.1_Checkpoint_Archaeology_and_Mechanism_Emergence_exercises.ipynb",
        "17.1_Checkpoint_Archaeology_and_Mechanism_Emergence_solutions.ipynb",
        "README.md", "artifacts.lock.yml", "legacy_solutions.py",
    },
    "13.2": {"README.md", "artifacts.lock.yml", "legacy_reference.py"},
    "15.1": {"artifacts.lock.yml"},
    "15.2": {
        "README.md", "artifacts.lock.yml",
        "15.2_LoRA_vs_Full_Finetuning_exercises.ipynb",
        "15.2_LoRA_vs_Full_Finetuning_solutions.ipynb",
    },
    "15.3": {
        "README.md", "artifacts.lock.yml", "build_notebooks.py", "execute_learner.py",
        "generate_cpu_fixtures.py", "legacy_reference.py",
    },
    "16.1": {"artifacts.lock.yml"},
}
RETAINED_HISTORICAL_DIRS = {
    number: {"expected_outputs"} for number in ("5.1", "5.3", "5.4", "5.5", "11.1", "11.2", "11.3", "11.4", "11.5", "13.2", "15.1", "15.2", "15.3", "16.1", "17.1")
}
RETAINED_HISTORICAL_DIRS["11.5"].add("development_evidence")
RETAINED_HISTORICAL_DIRS["5.8"] = {"expected_outputs", "artifacts"}
RETAINED_HISTORICAL_DIRS["6.4"] = {"expected_outputs"}
RETAINED_HISTORICAL_DIRS["6.5"] = {"expected_outputs"}
RETAINED_HISTORICAL_DIRS["9.3"] = {"provided"}
# Lesson helper modules beside the standard ARENA files.
SECTION_PLUMBING_FILES = {
    "5.1": {"gemma_reference.py"},
    "5.5": {"mamba_backbone.py"},
    "5.8": {"tensor_plumbing.py"},
    "6.5": {"crosscoder_plumbing.py"},
    "15.1": {"historical_release.py"},
    "17.1": {"checkpoint_reports.py"},
}
SECTION_DOCUMENTATION_FILES = {"15.1": {"README.md"}, "16.1": {"README.md"}}
SECTION_DOCUMENTATION_FILES["12.3"] = {"README.md"}
SECTION_DOCUMENTATION_FILES["16.11"] = {"README.md"}
SECTION_DOCUMENTATION_FILES["16.5"] = {"README.md"}
# The 12.1 CIFAR image fixture directory is data, not method code.
SECTION_DATA_DIRECTORIES = {"12.1": {"fixtures"}}
SECTION_DOCUMENTATION_FILES.update({
    number: {"README.md"} for number in (
        "5.2", "5.4", "7.1", "7.2", "7.3", "7.4", "7.5", "8.1", "8.2",
        "8.3", "8.4", "8.5", "8.8", "9.1", "10.1", "12.1", "16.2", "16.4",
        "16.6", "16.9", "16.10",
    )
})

SECTION_PLUMBING_FILES["5.2"] = {"execute_learner.py"}
SECTION_DOCUMENTATION_FILES["5.2"].add("lesson.md")
SECTION_PLUMBING_FILES["9.3"] = {"execute_learner.py"}
SECTION_DOCUMENTATION_FILES["9.3"] = {"lesson.md"}

RETAINED_HISTORICAL_FILES.update({
    '8.6': {'historical_verification_report.schema.json', 'README.md', 'historical_verification_report.json', 'historical_artifacts.lock.yml'},
    '8.7': {'historical_verification_report.schema.json', 'README.md', 'historical_verification_report.json', 'historical_artifacts.lock.yml'},
    '9.4': {'README.md', 'historical_README.md', 'historical_verification_report.json', 'historical_verification_report.schema.json', 'historical_artifacts.lock.yml'},
    '9.5': {'README.md', 'historical_README.md', 'historical_verification_report.json', 'historical_verification_report.schema.json', 'historical_artifacts.lock.yml'},
    '13.1': {'README.md', 'historical_README.md', 'historical_verification_report.json', 'historical_verification_report.schema.json', 'historical_artifacts.lock.yml'},
    '14.1': {'README.md', 'historical_tests.py', 'historical_README.md', 'historical_solutions.py', 'historical_verification_report.json', 'historical_verification_report.schema.json', 'historical_artifacts.lock.yml'},
    '14.5': {'historical_lesson.md', 'README.md', 'historical_README.md', 'historical_verification_report.schema.json', 'historical_artifacts.lock.yml'},
})
RETAINED_HISTORICAL_DIRS.update({
    '8.6': {'assets', 'expected_outputs'},
    '8.7': {'expected_outputs'},
    '9.4': {'expected_outputs'},
    '9.5': {'expected_outputs'},
    '13.1': {'expected_outputs'},
    '14.1': {'historical_expected_outputs'},
    '14.5': {'expected_outputs'},
})
SECTION_PLUMBING_FILES.update({
    '9.4': {'notebook_checks.py', 'validation_reference.py'},
    '9.5': {'validation_reference.py'},
    '13.1': {'lesson_reference.py'},
    '14.1': {'ijepa_plumbing.py'},
})


RETAINED_HISTORICAL_FILES["5.9"] = {
    "README.md", "historical_README.md", "historical_artifacts.lock.yml",
    "historical_solutions.py", "historical_tests.py", "historical_utils.py",
    "historical_instruction_page.md", "historical_verification_report.json",
    "historical_verification_report.schema.json",
}
RETAINED_HISTORICAL_DIRS["5.9"] = {"historical_expected_outputs"}


RETAINED_HISTORICAL_FILES['6.1'] = {
    'README.md',
    'artifacts.lock.yml',
    'verification_report.json',
    'verification_report.schema.json',
}
RETAINED_HISTORICAL_FILES['9.6'] = {
    'README.md',
    'historical_README.md',
    'historical_artifacts.lock.yml',
    'historical_build_notebooks.py',
    'historical_release_report.py',
    'historical_solutions.py',
    'historical_tests.py',
    'historical_verification_report.schema.json',
}
RETAINED_HISTORICAL_FILES['12.5'] = {
    'README.md',
    'historical_artifacts.lock.yml',
    'historical_solutions.py',
    'historical_verification_report.json',
    'legacy_README.md',
    'legacy_tests.py',
    'verification_report.schema.json',
}
RETAINED_HISTORICAL_DIRS['6.1'] = {
    'expected_outputs',
}
RETAINED_HISTORICAL_DIRS['9.6'] = {
    'historical_expected_outputs',
}
RETAINED_HISTORICAL_DIRS['12.5'] = {
    'historical_expected_outputs',
}
SECTION_PLUMBING_FILES['6.1'] = {
    'sae_plumbing.py',
}
SECTION_PLUMBING_FILES['9.6'] = {
    'rome_metadata.py',
}
SECTION_PLUMBING_FILES['12.5'] = set()

RETAINED_HISTORICAL_FILES['6.2'] = {'README.md'}
RETAINED_HISTORICAL_FILES['14.2'] = {'controls_legacy_utils.py', 'scratch_legacy_tests.py', 'controls_legacy_solutions.py', 'README.md', 'scratch_legacy_solutions.py', 'controls_legacy_tests.py'}
RETAINED_HISTORICAL_FILES['16.7'] = {'region_tests.py', 'region_utils.py', 'README.md', 'token_utils.py', 'token_tests.py'}
RETAINED_HISTORICAL_FILES['16.8'] = {'agreement_utils.py', 'agreement_tests.py', 'relu_utils.py', 'README.md', 'relu_tests.py'}
RETAINED_HISTORICAL_DIRS['6.2'] = set()
RETAINED_HISTORICAL_DIRS['14.2'] = set()
RETAINED_HISTORICAL_DIRS['16.7'] = set()
RETAINED_HISTORICAL_DIRS['16.8'] = set()
SECTION_PLUMBING_FILES['6.2'] = set()
SECTION_PLUMBING_FILES['14.2'] = set()
SECTION_PLUMBING_FILES['16.7'] = set()
SECTION_PLUMBING_FILES['16.8'] = set()

RETAINED_HISTORICAL_FILES['0.6'] = {'artifacts.lock.yml', '0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_exercises.ipynb', 'verification_report.json', 'verification_report.schema.json', 'README.md', '0.6_How_to_Know_When_an_Interpretability_Result_Is_Fake_solutions.ipynb'}
RETAINED_HISTORICAL_FILES['1.6'] = {'artifacts.lock.yml', '1.6_Local_Frontier_ML_Infrastructure_exercises.ipynb', '1.6_Local_Frontier_ML_Infrastructure_solutions.ipynb', 'verification_report.schema.json', 'README.md', 'verification_report.json'}
RETAINED_HISTORICAL_FILES['16.3'] = {'gradient_gradientshap_plots.py', '16.3_TreeSHAP_DeepSHAP_and_GradientSHAP_exercises.ipynb', 'deep_deepshap_plots.py', 'gradient_gradientshap_methods_tests.py', '16.3_TreeSHAP_DeepSHAP_and_GradientSHAP_solutions.ipynb', 'gradient_gradientshap_data.py', 'deep_deepshap_data.py', 'deep_deepshap_methods_tests.py', 'tree_treeshap_data.py', 'tree_treeshap_methods_tests.py', 'tree_treeshap_plots.py', 'README.md'}
RETAINED_HISTORICAL_DIRS['0.6'] = {'expected_outputs'}
RETAINED_HISTORICAL_DIRS['1.6'] = {'expected_outputs'}
RETAINED_HISTORICAL_DIRS['16.3'] = set()
SECTION_PLUMBING_FILES['0.6'] = set()
SECTION_PLUMBING_FILES['1.6'] = set()
SECTION_PLUMBING_FILES['16.3'] = set()


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
    unexpected = []
    for path in exercise_dir.iterdir():
        if path.is_dir():
            if path.name not in ALLOWED_DIRS | RETAINED_HISTORICAL_DIRS.get(number, set()) | SECTION_DATA_DIRECTORIES.get(number, set()):
                unexpected.append(path.name + "/")
        elif path.name not in ALLOWED_FILES | RETAINED_HISTORICAL_FILES.get(number, set()) | SECTION_PLUMBING_FILES.get(number, set()) | SECTION_DOCUMENTATION_FILES.get(number, set()) and not path.name.startswith(colab_name) and path.suffix not in {".csv", ".json", ".txt", ".npz", ".pt", ".png"}:
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

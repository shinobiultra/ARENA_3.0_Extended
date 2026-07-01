"""Audit the learner-facing course surface for extension chapters."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CONFIG_PATH = ROOT / "infrastructure/core/config.yaml"
EXTENSION_ORIGINAL_SECTIONS = {"0.6", "1.6"}
REQUIRED_STREAMLIT_THEME_KEYS = {
    "primaryColor",
    "backgroundColor",
    "secondaryBackgroundColor",
    "textColor",
}
REQUIRED_INSTRUCTION_REQUIREMENTS = {
    "torch",
    "streamlit==1.45.0",
    "pyyaml",
}
REQUIRED_NOTEBOOK_VERIFICATION_STRINGS = (
    "def run_gpu_test(max_vram_gb: float = 24.0)",
    "def run_full_experiment(max_vram_gb: float = 24.0)",
    "verification_report.json",
)


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def is_extension_section(number: str) -> bool:
    if number in EXTENSION_ORIGINAL_SECTIONS:
        return True
    try:
        return int(number.split(".", maxsplit=1)[0]) >= 5
    except ValueError:
        return False


def extension_chapter_names(config: dict[str, Any]) -> list[str]:
    chapter_names = config["chapter_names"]
    return [chapter_names[index] for index in sorted(chapter_names) if int(index) >= 5]


def _chapter_number_from_name(config: dict[str, Any], chapter_name: str) -> int:
    for number, name in config["chapter_names"].items():
        if name == chapter_name:
            return int(number)
    raise KeyError(chapter_name)


def _requirement_lines(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }


def _page_has_arena_shape(text: str, number: str, title: str) -> bool:
    has_title = f"# [{number}] {title}" in text
    has_intro = "# Introduction" in text or "## Core Question" in text
    has_learning_objectives = (
        "## Content & Learning Objectives" in text
        or "## Learning Objectives" in text
        or "##### Learning Objectives" in text
    )
    has_exercise = "Exercise" in text
    has_visible_test = "tests." in text or "run_smoke_test" in text
    has_gpu_contract = "run_gpu_test" in text or "CUDA" in text or "GPU Contract" in text
    return all(
        [
            has_title,
            has_intro,
            has_learning_objectives,
            has_exercise,
            has_visible_test,
            has_gpu_contract,
        ]
    )


def _exercise_notebook_source(path: Path) -> str:
    notebook = json.loads(path.read_text())
    return "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))


def _exercise_notebook_declares_verification_contract(path: Path) -> bool:
    source = _exercise_notebook_source(path)
    return all(required in source for required in REQUIRED_NOTEBOOK_VERIFICATION_STRINGS)


def chapter_surface_blockers(config: dict[str, Any], chapter_name: str) -> list[str]:
    chapter_dir = ROOT / chapter_name
    instructions_dir = chapter_dir / "instructions"
    blockers: list[str] = []

    home_path = instructions_dir / "Home.py"
    if not home_path.exists():
        blockers.append(f"{chapter_name}: missing instructions/Home.py")
    elif "render_extension_home(__file__)" not in home_path.read_text():
        blockers.append(f"{chapter_name}: Home.py does not use shared extension renderer")

    config_path = instructions_dir / ".streamlit/config.toml"
    if not config_path.exists():
        blockers.append(f"{chapter_name}: missing instructions/.streamlit/config.toml")
    else:
        config_text = config_path.read_text()
        if "[theme]" not in config_text:
            blockers.append(f"{chapter_name}: streamlit config missing [theme]")
        for key in REQUIRED_STREAMLIT_THEME_KEYS:
            if key not in config_text:
                blockers.append(f"{chapter_name}: streamlit config missing {key}")

    requirements_path = instructions_dir / "requirements.txt"
    if not requirements_path.exists():
        blockers.append(f"{chapter_name}: missing instructions/requirements.txt")
    else:
        missing = REQUIRED_INSTRUCTION_REQUIREMENTS - _requirement_lines(requirements_path)
        if missing:
            blockers.append(f"{chapter_name}: instruction requirements missing {sorted(missing)}")

    return blockers


def section_surface_blockers(
    config: dict[str, Any],
    chapter_name: str,
    section: dict[str, Any],
) -> list[str]:
    chapter_dir = ROOT / chapter_name
    number = str(section.get("number", ""))
    title = str(section.get("title", ""))
    blockers: list[str] = []

    conversion = config["conversion_map"].get(number)
    if conversion is None:
        blockers.append(f"{number}: missing conversion_map entry")
        return blockers
    if conversion.get("exercise_dir") != section.get("exercise_dir"):
        blockers.append(f"{number}: conversion_map exercise_dir mismatch")
    expected_page = f"{conversion.get('streamlit_page')}.md"
    if expected_page != section.get("page_file"):
        blockers.append(f"{number}: conversion_map streamlit_page does not match page_file")

    page_path = chapter_dir / "instructions/pages" / str(section.get("page_file", ""))
    if not page_path.exists():
        blockers.append(f"{number}: missing instruction page {page_path.relative_to(ROOT)}")
    elif not _page_has_arena_shape(page_path.read_text(), number, title):
        blockers.append(f"{number}: instruction page lacks ARENA-style learner surface")

    exercise_dir = chapter_dir / "exercises" / str(section.get("exercise_dir", ""))
    if not exercise_dir.exists():
        blockers.append(f"{number}: missing exercise dir {exercise_dir.relative_to(ROOT)}")
    for required in ("README.md", "solutions.py", "tests.py", "verification_report.json"):
        required_path = exercise_dir / required
        if not required_path.exists():
            blockers.append(f"{number}: missing {required_path.relative_to(ROOT)}")

    solutions_path = exercise_dir / "solutions.py"
    if solutions_path.exists() and "def run_gpu_test" in solutions_path.read_text():
        exercise_notebooks = sorted(exercise_dir.glob("*_exercises.ipynb"))
        if not exercise_notebooks:
            blockers.append(f"{number}: missing exercise notebook with verification contract")
        else:
            for notebook_path in exercise_notebooks:
                try:
                    has_contract = _exercise_notebook_declares_verification_contract(notebook_path)
                except json.JSONDecodeError as exc:
                    blockers.append(
                        f"{number}: invalid notebook JSON in {notebook_path.relative_to(ROOT)}: {exc}"
                    )
                    continue
                if not has_contract:
                    blockers.append(
                        f"{number}: exercise notebook lacks report-backed GPU/full verification surface"
                    )

    return blockers


def course_surface_blockers(config: dict[str, Any] | None = None) -> list[str]:
    config = config or load_config()
    blockers: list[str] = []

    for chapter_name in extension_chapter_names(config):
        blockers.extend(chapter_surface_blockers(config, chapter_name))

    for chapter_name, chapter in config["chapters"].items():
        for section in chapter.get("sections", []):
            number = str(section.get("number", ""))
            if is_extension_section(number):
                blockers.extend(section_surface_blockers(config, chapter_name, section))

    return blockers


def main() -> None:
    blockers = course_surface_blockers()
    print(f"extension_chapters_checked={len(extension_chapter_names(load_config()))}")
    print(f"course_surface_blockers={len(blockers)}")
    if blockers:
        print("COURSE_SURFACE=FAIL")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(1)
    print("COURSE_SURFACE=PASS")


if __name__ == "__main__":
    main()

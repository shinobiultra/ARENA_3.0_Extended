"""Audit the learner-facing course surface for extension chapters.

Checks the chapter scaffold (Home page, Streamlit theme, instruction
requirements), that each section's config entry, page and exercise folder
exist and agree, and that each page has the ARENA shape: a numbered title,
an introduction, learning objectives, exercises and visible test calls.
"""

from __future__ import annotations

import ast
import re
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


# Historical published headings reviewed against these exact registry identities.
# Do not accept arbitrary suffixes or keywords from a different method's title.
REVIEWED_PAGE_TITLE_ALIASES = {
    ("8.3", "AtP* from Scratch"): {"AtP*: Fixing Attribution Patching"},
    ("15.2", "LoRA vs Full Fine-Tuning"): {
        "LoRA vs Full Finetuning on a Real Transformer"
    },
}


def _normalized_page_title(title: str) -> str:
    """Ignore conjunction punctuation while retaining every method/title word."""
    title = re.sub(r"\s*&\s*", " and ", title)
    title = re.sub(r",\s+and\b", " and", title)
    return " ".join(title.split())


def _page_has_title(text: str, number: str, title: str) -> bool:
    accepted = {title} | REVIEWED_PAGE_TITLE_ALIASES.get((number, title), set())
    accepted = {_normalized_page_title(candidate) for candidate in accepted}
    headings = re.findall(r"^#\s+\[([^\]\n]+)\]\s+([^\n]+)$", text, re.M)
    return any(
        actual_number == number and _normalized_page_title(actual_title) in accepted
        for actual_number, actual_title in headings
    )


def _page_has_arena_shape(text: str, number: str, title: str) -> bool:
    has_title = _page_has_title(text, number, title)
    has_intro = re.search(r"^#{1,2}\s+(?:Introduction|Core Question)\s*$", text, re.M | re.I) is not None
    has_learning_objectives = re.search(r"^#{2,5}\s+(?:Content (?:&|and) )?Learning Objectives\s*$", text, re.M | re.I) is not None
    has_exercise = "Exercise" in text
    has_visible_test = "tests." in text or "run_smoke_test" in text
    return all([has_title, has_intro, has_learning_objectives, has_exercise, has_visible_test])


def _home_uses_shared_renderer(source: str) -> bool:
    """Recognize a direct chapter-home call, without trusting comments or wrappers.

    The current renderer resolves the chapter from __file__, uses the configured
    chapter/section content through st_dependencies, and displays the chapter's
    nonempty introductory Markdown. Only this known import contract is accepted.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    renderers: set[str] = set()
    introductions: dict[str, str] = {}
    original_file_binding = True
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            for item in node.names:
                name = item.asname or item.name
                introductions.pop(name, None)
                if name == "*":
                    renderers.clear()
                    introductions.clear()
                    original_file_binding = False
                if name == "__file__":
                    original_file_binding = False
                if (node.level == 0 and node.module == "infrastructure.extension.home"
                        and item.name == "render_chapter_home"):
                    renderers.add(name)
                else:
                    renderers.discard(name)
            continue
        # Imports/definitions/assignments that shadow the known imported renderer
        # remove its authority. An unrelated function with that name is no proof.
        bound: set[str] = set()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            bound.update(item.id for target in targets for item in ast.walk(target)
                         if isinstance(item, ast.Name) and isinstance(item.ctx, ast.Store))
            for name in bound:
                introductions.pop(name, None)
                if (isinstance(node, (ast.Assign, ast.AnnAssign))
                        and isinstance(node.value, ast.Constant)
                        and isinstance(node.value.value, str) and node.value.value.strip()):
                    introductions[name] = node.value.value
        elif isinstance(node, ast.Import):
            bound.update(item.asname or item.name.split(".")[0] for item in node.names)
        else:
            # Conditional assignments/imports are not a known direct renderer
            # contract, even when the happy-path source text looks convincing.
            for item in ast.walk(node):
                if isinstance(item, ast.Name) and isinstance(item.ctx, (ast.Store, ast.Del)):
                    bound.add(item.id)
                elif isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    bound.add(item.name)
                elif isinstance(item, (ast.Import, ast.ImportFrom)):
                    bound.update(alias.asname or alias.name.split(".")[0] for alias in item.names)
            for name in bound:
                introductions.pop(name, None)
        if "*" in bound:
            renderers.clear()
            introductions.clear()
            original_file_binding = False
        if "__file__" in bound:
            original_file_binding = False
        renderers.difference_update(bound)
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            for name in bound:
                introductions.pop(name, None)
        if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
            continue
        call = node.value
        if (not original_file_binding or not isinstance(call.func, ast.Name) or call.func.id not in renderers
                or len(call.args) != 2 or call.keywords
                or not isinstance(call.args[0], ast.Name) or call.args[0].id != "__file__"):
            continue
        intro = call.args[1]
        if ((isinstance(intro, ast.Constant) and isinstance(intro.value, str) and intro.value.strip())
                or (isinstance(intro, ast.Name) and intro.id in introductions)):
            return True
    return False


def chapter_surface_blockers(config: dict[str, Any], chapter_name: str) -> list[str]:
    chapter_dir = ROOT / chapter_name
    instructions_dir = chapter_dir / "instructions"
    blockers: list[str] = []

    home_path = instructions_dir / "Home.py"
    if not home_path.exists():
        blockers.append(f"{chapter_name}: missing instructions/Home.py")
    elif not _home_uses_shared_renderer(home_path.read_text()):
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
    for required in ("solutions.py", "tests.py"):
        required_path = exercise_dir / required
        if not required_path.exists():
            blockers.append(f"{number}: missing {required_path.relative_to(ROOT)}")

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

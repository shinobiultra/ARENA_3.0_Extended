"""Check links to this repository from learner-facing documents (Spec #39, P9).

Covered: extension lesson pages and chapter homes (learner pages), the README, the study guide and
the Alchemy companion page. A link to this repository must be relative, or a GitHub `blob/main/` or
`tree/main/` link, and its target must exist. A lesson page's Markdown also ships in its notebooks,
so its relative links may resolve from the page's folder or the lesson's exercises folder. Code
spans and blocks are not links. Links to other branches (which get deleted), Colab
links, and other GitHub views are rejected. Release and repository-home links are GitHub
navigation and pass. Learner pages may not link issues or pull requests.

Usage (from the repository root):

    uv run python scripts/check_repo_links.py
"""

from __future__ import annotations

import functools
import re
import sys
from pathlib import Path
from urllib.parse import unquote

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = "shinobiultra/ARENA_3.0_Extended"
FIRST_EXTENSION_CHAPTER = 5
EXTENSION_SECTIONS_IN_ORIGINAL_CHAPTERS = ("0.6", "1.6")
DOCUMENTS = ("README.md", "ALCHEMY.md", "mastery_hub/STUDY-GUIDE.md")

_REPO_RE = re.escape(REPO)
REPO_URL = re.compile(
    r"https?://(?:www\.)?(?:github\.com/|raw\.githubusercontent\.com/|colab\.research\.google\.com/github/)"
    + _REPO_RE
    + r"""[^\s)"'<>\]`]*"""
)
MARKDOWN_TARGET = re.compile(r"\]\(\s*<?([^)\s>]+)")
# Fenced blocks, and inline spans (which may wrap onto the next line but not past a blank line).
CODE = re.compile(r"```.*?```|`(?:[^`\n]|\n(?!\s*\n))*`", re.DOTALL)
ATTRIBUTE_TARGET = re.compile(r"""\b(?:href|src)=["']([^"']+)["']""")


@functools.cache
def _lesson_folders(root: Path) -> dict[str, Path]:
    """Map each lesson page (repo-relative) to its exercises folder, where its notebooks live."""

    config_path = root / "infrastructure/core/config.yaml"
    if not config_path.exists():
        return {}
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    folders = {}
    for number, mapping in config["conversion_map"].items():
        chapter_dir = config["chapter_names"].get(int(str(number).split(".")[0]))
        if chapter_dir and "streamlit_page" in mapping and "exercise_dir" in mapping:
            page = f"{chapter_dir}/instructions/pages/{mapping['streamlit_page']}.md"
            folders[page] = root / chapter_dir / "exercises" / mapping["exercise_dir"]
    return folders


def covered_files(root: Path = ROOT) -> list[tuple[Path, bool]]:
    """Return (path, is_learner_page) for every document the check covers."""

    config = yaml.safe_load((root / "infrastructure/core/config.yaml").read_text(encoding="utf-8"))
    files: list[tuple[Path, bool]] = [
        (root / name, False) for name in DOCUMENTS if (root / name).exists()
    ]
    for number, chapter_dir in sorted(config["chapter_names"].items()):
        instructions = root / chapter_dir / "instructions"
        if number >= FIRST_EXTENSION_CHAPTER:
            files += (
                [(instructions / "Home.py", True)] if (instructions / "Home.py").exists() else []
            )
            files += [(page, True) for page in sorted((instructions / "pages").glob("*.md"))]
    for section in EXTENSION_SECTIONS_IN_ORIGINAL_CHAPTERS:
        mapping = config["conversion_map"][section]
        chapter_dir = config["chapter_names"][int(section.split(".")[0])]
        files.append(
            (root / chapter_dir / "instructions/pages" / f"{mapping['streamlit_page']}.md", True)
        )
    return files


def _exists(root: Path, relative: str) -> bool:
    relative = unquote(relative.split("#", 1)[0].split("?", 1)[0]).strip("/")
    return bool(relative) and (root / relative).exists()


def _repo_url_problem(url: str, root: Path, learner: bool) -> str | None:
    if "colab.research.google.com" in url:
        return "is not a relative or blob/main link"
    rest = url.split(REPO, 1)[1].split("#", 1)[0].split("?", 1)[0].strip("/")
    if "raw.githubusercontent.com" in url:
        ref, _, path = rest.partition("/")
        if ref != "main":
            return f"points at {ref!r}, not main"
        return None if _exists(root, path) else f"target {path!r} does not exist"
    view, _, tail = rest.partition("/")
    if view in ("", ".git", "releases"):
        return None
    if view in ("issues", "pull", "pulls"):
        return "links an issue or pull request from a learner page" if learner else None
    if view in ("blob", "tree", "raw"):
        ref, _, path = tail.partition("/")
        if ref != "main":
            return f"points at {ref!r}, not main"
        return None if _exists(root, path) else f"target {path!r} does not exist"
    return "is not a relative or blob/main link"


def _relative_problem(target: str, bases: list[Path], root: Path) -> str | None:
    if "://" in target or target.startswith(("#", "/", "?", "mailto:", "data:")) or "{" in target:
        return None
    path = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not path:
        return None
    for base in bases:
        resolved = (base / path).resolve()
        if resolved.is_relative_to(root.resolve()) and resolved.exists():
            return None
    return f"target {path!r} does not exist"


def link_problems(path: Path, root: Path = ROOT, learner: bool = False) -> list[str]:
    """Return one message per broken or disallowed link to this repository in `path`."""

    text = path.read_text(encoding="utf-8")
    label = path.relative_to(root).as_posix()
    # The page's Markdown also ships in its notebooks, so its links may resolve from either folder.
    bases = [path.parent] + ([folder] if (folder := _lesson_folders(root).get(label)) else [])
    problems = []
    for url in dict.fromkeys(REPO_URL.findall(text)):
        if problem := _repo_url_problem(url, root, learner):
            problems.append(f"{label}: {url} {problem}")
    prose = CODE.sub("", text)
    targets = MARKDOWN_TARGET.findall(prose) + ATTRIBUTE_TARGET.findall(prose)
    for target in dict.fromkeys(targets):
        if problem := _relative_problem(target, bases, root):
            problems.append(f"{label}: {target} {problem}")
    return problems


def main() -> None:
    files = covered_files()
    problems = [
        problem for path, learner in files for problem in link_problems(path, learner=learner)
    ]
    print(f"files_checked={len(files)}")
    print(f"link_problems={len(problems)}")
    for problem in problems:
        print(f"- {problem}")
    if problems:
        raise SystemExit(1)


if __name__ == "__main__":
    sys.exit(main())

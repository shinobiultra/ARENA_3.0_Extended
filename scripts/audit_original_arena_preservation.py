"""Audit that original ARENA chapters 0-4 are unchanged.

Every file that the pinned original ARENA commit ships under chapters 0-4, and
under their master-file folders, must still exist with identical bytes. New
files beside them (the 0.6 and 1.6 extension lessons) are fine. The original
entries in `infrastructure/core/config.yaml` must be unchanged (the config is
append-only), and `requirements-original.txt` must equal the original
`requirements.txt`.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ARTIFACT_REGISTRY_LOCK_PATH = ROOT / "docs/artifact_registry.lock.yml"
ORIGINAL_CHAPTERS = (
    "chapter0_fundamentals",
    "chapter1_transformer_interp",
    "chapter2_rl",
    "chapter3_llm_evals",
    "chapter4_alignment_science",
)
ORIGINAL_PREFIXES = tuple(f"{chapter}/" for chapter in ORIGINAL_CHAPTERS) + tuple(
    f"infrastructure/chapters/{chapter}/" for chapter in ORIGINAL_CHAPTERS
)


def _git_lines(*args: str) -> list[str]:
    result = subprocess.run(["git", *args], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE)
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _git_bytes(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, stdout=subprocess.PIPE).stdout


def original_base_revision() -> str:
    """Return the pinned upstream ARENA commit from the artifact registry."""

    lock = yaml.safe_load(ARTIFACT_REGISTRY_LOCK_PATH.read_text())
    for artifact in lock.get("artifacts", []):
        if artifact.get("name") == "ARENA 3.0 source course":
            return str(artifact["revision"])
    raise RuntimeError("ARENA 3.0 source course missing from artifact registry lock")


def _base_text(revision: str, relative_path: str) -> str:
    return _git_bytes("show", f"{revision}:{relative_path}").decode()


def original_paths(revision: str) -> list[str]:
    return sorted(set(_git_lines("ls-tree", "-r", "--name-only", revision, "--", *ORIGINAL_PREFIXES)))


def original_chapter_file_blockers(revision: str | None = None) -> list[str]:
    """Compare every original chapter and master file byte-for-byte against the base."""

    revision = revision or original_base_revision()
    blockers: list[str] = []
    for relative in original_paths(revision):
        current_path = ROOT / relative
        if not current_path.exists():
            blockers.append(f"{relative}: original file deleted")
        elif current_path.read_bytes() != _git_bytes("show", f"{revision}:{relative}"):
            blockers.append(f"{relative}: differs from pinned original ARENA base")
    return blockers


def _section_map(chapter: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(section.get("number")): section for section in chapter.get("sections", [])}


def config_append_only_blockers(revision: str | None = None) -> list[str]:
    """Check that original config metadata is unchanged while allowing new sections."""

    revision = revision or original_base_revision()
    current = yaml.safe_load((ROOT / "infrastructure/core/config.yaml").read_text())
    base = yaml.safe_load(_base_text(revision, "infrastructure/core/config.yaml"))
    blockers: list[str] = []

    for section_number, base_mapping in base["conversion_map"].items():
        if current["conversion_map"].get(section_number) != base_mapping:
            blockers.append(f"config conversion_map[{section_number!r}] changed")

    for key in ("chapter_names", "chapter_names_long"):
        for chapter_index, base_value in base[key].items():
            if current[key].get(chapter_index) != base_value:
                blockers.append(f"config {key}[{chapter_index!r}] changed")

    for chapter_name, base_chapter in base["chapters"].items():
        current_chapter = current["chapters"].get(chapter_name)
        if current_chapter is None:
            blockers.append(f"config chapters[{chapter_name!r}] missing")
            continue
        base_metadata = {key: value for key, value in base_chapter.items() if key != "sections"}
        current_metadata = {key: value for key, value in current_chapter.items() if key != "sections"}
        if current_metadata != base_metadata:
            blockers.append(f"config chapters[{chapter_name!r}] metadata changed")
        current_sections = _section_map(current_chapter)
        for section_number, base_section in _section_map(base_chapter).items():
            if current_sections.get(section_number) != base_section:
                blockers.append(f"config chapters[{chapter_name!r}] section {section_number!r} changed")
    return blockers


def requirements_split_blockers(revision: str | None = None) -> list[str]:
    """Require the original requirements snapshot to remain exact."""

    revision = revision or original_base_revision()
    snapshot_path = ROOT / "requirements-original.txt"
    if not snapshot_path.exists():
        return ["requirements-original.txt: missing exact original requirements snapshot"]
    if snapshot_path.read_text().splitlines() != _base_text(revision, "requirements.txt").splitlines():
        return ["requirements-original.txt: differs from pinned original ARENA requirements"]
    return []


def split_config_blockers(revision: str | None = None) -> list[str]:
    """Require config.yaml to equal config_original.yaml merged with config_extension.yaml."""

    revision = revision or original_base_revision()
    original_config_path = ROOT / "infrastructure/core/config_original.yaml"
    extension_config_path = ROOT / "infrastructure/core/config_extension.yaml"
    merged_config_path = ROOT / "infrastructure/core/config.yaml"

    blockers: list[str] = []
    if not original_config_path.exists():
        blockers.append("infrastructure/core/config_original.yaml: missing")
    elif original_config_path.read_text() != _base_text(revision, "infrastructure/core/config.yaml"):
        blockers.append("infrastructure/core/config_original.yaml: differs from pinned original config")
    if not extension_config_path.exists():
        blockers.append("infrastructure/core/config_extension.yaml: missing")
    if blockers:
        return blockers

    from scripts.build_merged_config import load_yaml, merge_config

    try:
        expected = merge_config(load_yaml(original_config_path), load_yaml(extension_config_path))
    except Exception as exc:
        return [f"infrastructure/core/config_extension.yaml: invalid overlay ({exc})"]
    if yaml.safe_load(merged_config_path.read_text()) != expected:
        blockers.append("infrastructure/core/config.yaml: differs from config_original + config_extension")
    return blockers


def original_preservation_blockers() -> list[str]:
    """Return all blockers for the original-course preservation check."""

    revision = original_base_revision()
    return (
        original_chapter_file_blockers(revision)
        + config_append_only_blockers(revision)
        + requirements_split_blockers(revision)
        + split_config_blockers(revision)
    )


def main() -> None:
    revision = original_base_revision()
    blockers = original_preservation_blockers()
    print(f"original_files_checked={len(original_paths(revision))}")
    print(f"preservation_blockers={len(blockers)}")
    if blockers:
        print("ORIGINAL_ARENA_PRESERVATION=FAIL")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(1)
    print("ORIGINAL_ARENA_PRESERVATION=PASS")


if __name__ == "__main__":
    main()

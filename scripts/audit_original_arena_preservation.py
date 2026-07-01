"""Audit that the original ARENA course surface remains append-only.

The roadmap allows compatibility patches and additive extension sections, but
it explicitly says the original ARENA material should be preserved. This script
checks the current git working tree against that contract.
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

COMPATIBILITY_PATCH_RATIONALES = {
    ".github/workflows/extension-quality.yml": "real CI gates for extension audits and GPU reports",
    ".gitignore": "cache, model-weight, and generated-artifact hygiene for the extension",
    ".python-version": "uv-managed Python version pin for the CUDA 13 environment",
    "Extension-Roadmap.md": "user-authored extension specification",
    "guidance_2-0.md": "review guidance for the ARENA-style rewrite",
    "README.md": "entrypoint documentation for the extended course",
    "install.sh": "original installer redirected to the pinned original requirements split",
    "infrastructure/core/config.yaml": "append-only registration of extension chapters and sections",
    "infrastructure/core/config_original.yaml": "frozen upstream ARENA config snapshot",
    "infrastructure/core/config_extension.yaml": "extension-only config overlay",
    "pyproject.toml": "project metadata, pytest import mode, and CI marker registration",
    "requirements-ci-cpu.txt": "minimal hosted-CI dependencies for audit tests",
    "requirements-legacy-rl.txt": "isolated legacy RL dependency stack kept out of the CUDA 13 env",
    "requirements-original.txt": "exact upstream requirements snapshot for the original installer",
    "requirements.txt": "default uv CUDA 13 dependency stack",
    "uv.lock": "uv resolution metadata for the managed environment",
}
COMPATIBILITY_CONTRACT_PATH = ROOT / "docs/original_preservation_contract.md"
ARTIFACT_REGISTRY_LOCK_PATH = ROOT / "docs/artifact_registry.lock.yml"
ORIGINAL_CHAPTER_PREFIXES = (
    "chapter0_fundamentals/",
    "chapter1_transformer_interp/",
    "chapter2_rl/",
    "chapter3_llm_evals/",
    "chapter4_alignment_science/",
)

ADDITIVE_EXTENSION_PREFIXES = (
    "FOR_REVIEW/",
    "arena_ext/",
    "chapter5_modern_architectures/",
    "chapter6_sparse_feature_methods/",
    "chapter7_activation_to_language/",
    "chapter8_automated_circuits/",
    "chapter9_alignment_interpretability/",
    "chapter10_capstone_research_sprint/",
    "chapter11_representation_geometry/",
    "chapter12_vlm_interpretability/",
    "chapter13_image_generation_interpretability/",
    "chapter14_jepa_world_models/",
    "chapter15_peft_misalignment/",
    "chapter16_shapley_attribution_baselines/",
    "chapter17_training_dynamics/",
    "data/generated/refusal_proxy_prompts_v1/",
    "docs/",
    "research_projects/",
    "scripts/",
    "tests/",
)

ADDITIVE_ORIGINAL_CHAPTER_PATHS = (
    "chapter0_fundamentals/exercises/part6_fake_interpretability_results/",
    "chapter0_fundamentals/instructions/pages/"
    "06_[0.6]_How_to_Know_When_an_Interpretability_Result_Is_Fake.md",
    "chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure/",
    "chapter1_transformer_interp/instructions/pages/40_[1.6]_Local_Frontier_ML_Infrastructure.md",
)


def _git_lines(*args: str) -> list[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _git_bytes(*args: str) -> bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return result.stdout


def changed_paths() -> dict[str, list[str]]:
    """Return changed paths grouped by git source."""

    unstaged = _git_lines("diff", "--name-only")
    staged = _git_lines("diff", "--cached", "--name-only")
    untracked = _git_lines("ls-files", "--others", "--exclude-standard")
    return {
        "tracked": sorted(set(unstaged + staged)),
        "untracked": sorted(set(untracked)),
    }


def _is_prefix_match(path: str, prefixes: tuple[str, ...]) -> bool:
    return any(path == prefix.rstrip("/") or path.startswith(prefix) for prefix in prefixes)


def is_allowed_preservation_change(path: str) -> bool:
    """Return whether a changed path is compatible with append-only extension work."""

    normalized = path.strip()
    if normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized in COMPATIBILITY_PATCH_RATIONALES:
        return True
    if _is_prefix_match(normalized, ADDITIVE_EXTENSION_PREFIXES):
        return True
    return _is_prefix_match(normalized, ADDITIVE_ORIGINAL_CHAPTER_PATHS)


def preservation_blockers(paths_by_source: dict[str, list[str]]) -> list[str]:
    blockers: list[str] = []
    for source, paths in sorted(paths_by_source.items()):
        for path in paths:
            if not is_allowed_preservation_change(path):
                blockers.append(f"{source}: {path}")
    return blockers


def compatibility_contract_blockers() -> list[str]:
    """Require every global compatibility patch to be documented with a rationale."""

    blockers: list[str] = []
    if not COMPATIBILITY_CONTRACT_PATH.exists():
        return [f"missing {COMPATIBILITY_CONTRACT_PATH.relative_to(ROOT)}"]

    text = COMPATIBILITY_CONTRACT_PATH.read_text()
    for path, rationale in sorted(COMPATIBILITY_PATCH_RATIONALES.items()):
        if not rationale.strip():
            blockers.append(f"{path}: empty compatibility-patch rationale")
        if path not in text:
            blockers.append(
                f"{path}: missing from {COMPATIBILITY_CONTRACT_PATH.relative_to(ROOT)}"
            )
    return blockers


def original_base_revision() -> str:
    """Return the pinned upstream ARENA commit from the artifact registry."""

    lock = yaml.safe_load(ARTIFACT_REGISTRY_LOCK_PATH.read_text())
    for artifact in lock.get("artifacts", []):
        if artifact.get("name") == "ARENA 3.0 source course":
            return str(artifact["revision"])
    raise RuntimeError("ARENA 3.0 source course missing from artifact registry lock")


def _base_text(revision: str, relative_path: str) -> str:
    return _git_bytes("show", f"{revision}:{relative_path}").decode()


def _base_paths(revision: str, prefixes: tuple[str, ...]) -> list[str]:
    paths: list[str] = []
    for prefix in prefixes:
        paths.extend(_git_lines("ls-tree", "-r", "--name-only", revision, "--", prefix))
    return sorted(set(paths))


def _section_map(chapter: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(section.get("number")): section for section in chapter.get("sections", [])}


def original_chapter_file_blockers(revision: str | None = None) -> list[str]:
    """Compare original chapter files byte-for-byte against the pinned base."""

    revision = revision or original_base_revision()
    blockers: list[str] = []
    for relative in _base_paths(revision, ORIGINAL_CHAPTER_PREFIXES):
        if is_allowed_preservation_change(relative):
            continue
        current_path = ROOT / relative
        if not current_path.exists():
            blockers.append(f"{relative}: original file deleted")
            continue
        base_bytes = _git_bytes("show", f"{revision}:{relative}")
        if current_path.read_bytes() != base_bytes:
            blockers.append(f"{relative}: differs from pinned original ARENA base")
    return blockers


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
        current_metadata = {
            key: value for key, value in current_chapter.items() if key != "sections"
        }
        if current_metadata != base_metadata:
            blockers.append(f"config chapters[{chapter_name!r}] metadata changed")

        current_sections = _section_map(current_chapter)
        for section_number, base_section in _section_map(base_chapter).items():
            if current_sections.get(section_number) != base_section:
                blockers.append(
                    f"config chapters[{chapter_name!r}] section {section_number!r} changed"
                )
    return blockers


def requirements_split_blockers(revision: str | None = None) -> list[str]:
    """Require the original requirements snapshot to remain exact and discoverable."""

    revision = revision or original_base_revision()
    base_requirements = _base_text(revision, "requirements.txt").splitlines()
    snapshot_path = ROOT / "requirements-original.txt"
    if not snapshot_path.exists():
        return ["requirements-original.txt: missing exact original requirements snapshot"]
    if snapshot_path.read_text().splitlines() != base_requirements:
        return ["requirements-original.txt: differs from pinned original ARENA requirements"]
    return []


def split_config_blockers(revision: str | None = None) -> list[str]:
    """Require config.yaml to be generated-equivalent to original + extension config."""

    revision = revision or original_base_revision()
    blockers: list[str] = []
    original_config_path = ROOT / "infrastructure/core/config_original.yaml"
    extension_config_path = ROOT / "infrastructure/core/config_extension.yaml"
    merged_config_path = ROOT / "infrastructure/core/config.yaml"

    if not original_config_path.exists():
        blockers.append("infrastructure/core/config_original.yaml: missing")
    elif original_config_path.read_text() != _base_text(
        revision,
        "infrastructure/core/config.yaml",
    ):
        blockers.append(
            "infrastructure/core/config_original.yaml: differs from pinned original config"
        )

    if not extension_config_path.exists():
        blockers.append("infrastructure/core/config_extension.yaml: missing")
    if blockers:
        return blockers

    from scripts.build_merged_config import load_yaml, merge_config

    try:
        expected = merge_config(
            load_yaml(original_config_path),
            load_yaml(extension_config_path),
        )
    except Exception as exc:
        return [f"infrastructure/core/config_extension.yaml: invalid overlay ({exc})"]

    current = yaml.safe_load(merged_config_path.read_text())
    if current != expected:
        blockers.append(
            "infrastructure/core/config.yaml: differs from config_original + config_extension"
        )
    return blockers


def original_preservation_blockers() -> list[str]:
    """Return all blockers for the original-course preservation contract."""

    revision = original_base_revision()
    return (
        preservation_blockers(changed_paths())
        + compatibility_contract_blockers()
        + original_chapter_file_blockers(revision)
        + config_append_only_blockers(revision)
        + requirements_split_blockers(revision)
        + split_config_blockers(revision)
    )


def main() -> None:
    paths_by_source = changed_paths()
    blockers = original_preservation_blockers()
    changed_count = sum(len(paths) for paths in paths_by_source.values())

    print(f"changed_paths_checked={changed_count}")
    print(f"tracked_paths={len(paths_by_source['tracked'])}")
    print(f"untracked_paths={len(paths_by_source['untracked'])}")
    print(f"preservation_blockers={len(blockers)}")

    if blockers:
        print("ORIGINAL_ARENA_PRESERVATION=FAIL")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(1)

    print("ORIGINAL_ARENA_PRESERVATION=PASS")


if __name__ == "__main__":
    main()

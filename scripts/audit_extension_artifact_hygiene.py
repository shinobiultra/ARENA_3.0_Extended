"""Audit extension files for raw weights, oversized files and unsafe prompt data.

Lessons ship small files only: no committed model weights or archives, nothing
over 5 MB, prompt fixtures that follow their data contract, and a .gitignore
that keeps caches and weights out. Only files git sees (tracked, or untracked
and not ignored) under extension-owned paths are checked, so legacy ARENA
assets and local ignored caches are not reclassified.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_DATA_CONTRACTS_SPEC = importlib.util.spec_from_file_location(
    "arena_ext_data_contracts_helpers",
    ROOT / "arena_ext" / "data_contracts.py",
)
if _DATA_CONTRACTS_SPEC is None or _DATA_CONTRACTS_SPEC.loader is None:
    raise RuntimeError("could not load arena_ext/data_contracts.py")
_DATA_CONTRACTS_MODULE = importlib.util.module_from_spec(_DATA_CONTRACTS_SPEC)
sys.modules[_DATA_CONTRACTS_SPEC.name] = _DATA_CONTRACTS_MODULE
_DATA_CONTRACTS_SPEC.loader.exec_module(_DATA_CONTRACTS_MODULE)
load_jsonl_records = _DATA_CONTRACTS_MODULE.load_jsonl_records
validate_dataset_manifest = _DATA_CONTRACTS_MODULE.validate_dataset_manifest
validate_prompt_records = _DATA_CONTRACTS_MODULE.validate_prompt_records


CONFIG_PATH = ROOT / "infrastructure/core/config.yaml"
MAX_EXTENSION_FILE_BYTES = 5 * 1024 * 1024
FORBIDDEN_ARTIFACT_SUFFIXES = (
    ".safetensors",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".pth",
    ".pt",
    ".npz",
    ".npy",
    ".bin",
    ".zip",
    ".tar",
    ".tar.gz",
    ".mp4",
    ".joblib",
    ".pickle",
    ".pkl",
)
REQUIRED_TRACKED_FIXTURE_PATHS = (
    "data/generated/refusal_proxy_prompts_v1/README.md",
    "data/generated/refusal_proxy_prompts_v1/manifest.yml",
    "data/generated/refusal_proxy_prompts_v1/prompt_schema.json",
    "data/generated/refusal_proxy_prompts_v1/prompts.jsonl",
)
ALLOWED_BINARY_ARTIFACTS = {
    "chapter16_shapley_attribution_baselines/exercises/"
    "part8_shapley_mechinterp_agreement/artifacts/deletion_curves.png",
    "chapter16_shapley_attribution_baselines/exercises/"
    "part8_shapley_mechinterp_agreement/artifacts/insertion_curves.png",
    "chapter16_shapley_attribution_baselines/exercises/"
    "part8_shapley_mechinterp_agreement/artifacts/topk_overlap_heatmap.png",
}
STATIC_EXTENSION_ROOTS = (
    "arena_ext",
    "docs",
    "research_projects",
    "scripts",
    "tests",
    "data/generated",
    "chapter0_fundamentals/exercises/part6_fake_interpretability_results",
    "chapter0_fundamentals/exercises/part6_fake_interpretability_results_practice",
    "chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure",
    "chapter1_transformer_interp/exercises/part6_frontier_ml_infrastructure_practice",
)
# Extension pages that live inside original chapters.
EXTENSION_PAGE_MARKERS = ("_[0.6]_", "_[1.6]_")


def load_config(path: Path = CONFIG_PATH) -> dict:
    return yaml.safe_load(path.read_text())


def extension_root_paths(config: dict | None = None) -> list[Path]:
    config = config or load_config()
    chapter_names = config["chapter_names"]
    dynamic_roots = [
        ROOT / chapter_names[index]
        for index in sorted(chapter_names)
        if int(index) >= 5
    ]
    static_roots = [ROOT / path for path in STATIC_EXTENSION_ROOTS]
    roots: list[Path] = []
    seen: set[Path] = set()
    for path in [*static_roots, *dynamic_roots]:
        if path.exists() and path not in seen:
            roots.append(path)
            seen.add(path)
    return roots


def _is_allowed_artifact(path: Path) -> bool:
    try:
        relative = path.relative_to(ROOT).as_posix()
    except ValueError:
        return False
    return relative in ALLOWED_BINARY_ARTIFACTS


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def _git_visible_files() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
        cwd=ROOT, check=True, capture_output=True,
    )
    return [path for path in result.stdout.decode().split("\0") if path]


def iter_extension_files(roots: Iterable[Path] | None = None) -> list[Path]:
    """Extension-owned files that git sees (tracked, or untracked and not ignored).

    With the default roots, extension pages inside chapters 0 and 1 are included too.
    """
    include_pages = roots is None
    roots = list(roots or extension_root_paths())
    prefixes = tuple(_display_path(root).rstrip("/") + "/" for root in roots)
    files: list[Path] = []
    for relative in _git_visible_files():
        in_root = relative.startswith(prefixes) or relative + "/" in prefixes
        extension_page = (include_pages
                          and relative.startswith(("chapter0_fundamentals/instructions/pages/",
                                                   "chapter1_transformer_interp/instructions/pages/"))
                          and any(marker in relative for marker in EXTENSION_PAGE_MARKERS))
        path = ROOT / relative
        if (in_root or extension_page) and path.is_file():
            files.append(path)
    return sorted(set(files))


def forbidden_artifact_blockers(files: Iterable[Path]) -> list[str]:
    blockers: list[str] = []
    for path in files:
        name = path.name.lower()
        relative = _display_path(path)
        if _is_allowed_artifact(path):
            continue
        if any(name.endswith(suffix) for suffix in FORBIDDEN_ARTIFACT_SUFFIXES):
            blockers.append(f"raw artifact forbidden in extension path: {relative}")
    return blockers


def large_file_blockers(files: Iterable[Path]) -> list[str]:
    blockers: list[str] = []
    for path in files:
        if _is_allowed_artifact(path):
            continue
        size = path.stat().st_size
        if size > MAX_EXTENSION_FILE_BYTES:
            relative = _display_path(path)
            blockers.append(f"oversized extension file: {relative} ({size} bytes)")
    return blockers


def prompt_jsonl_blockers(path: Path) -> list[str]:
    records = load_jsonl_records(path)
    if not records or not any("prompt" in record for record in records):
        return []

    report = validate_prompt_records(records)
    relative = _display_path(path)
    blockers: list[str] = []
    blockers.extend(f"{relative}: missing field {item}" for item in report.missing_fields)
    blockers.extend(
        f"{relative}: missing metadata field {item}"
        for item in report.missing_metadata_fields
    )
    blockers.extend(f"{relative}: duplicate id {item}" for item in report.duplicate_ids)
    blockers.extend(
        f"{relative}: missing counterfactual for {item}"
        for item in report.missing_counterfactuals
    )
    blockers.extend(
        f"{relative}: unredacted sensitive prompt {item}"
        for item in report.unsafe_display_violations
    )
    blockers.extend(
        f"{relative}: missing prompt_hash for sensitive prompt {item}"
        for item in report.missing_prompt_hashes
    )
    return blockers


def dataset_manifest_blockers(path: Path) -> list[str]:
    manifest = yaml.safe_load(path.read_text())
    if not isinstance(manifest, dict) or not manifest.get("records_path"):
        return []
    report = validate_dataset_manifest(manifest)
    relative = _display_path(path)
    blockers: list[str] = []
    if not report.has_seed:
        blockers.append(f"{relative}: missing seed")
    if not report.has_schema_path:
        blockers.append(f"{relative}: missing schema_path")
    if not report.has_split_policy:
        blockers.append(f"{relative}: missing train/validation/test split policy")
    if not report.has_ood_split:
        blockers.append(f"{relative}: missing ood_split")
    if not report.has_counterfactual_mapping:
        blockers.append(f"{relative}: missing counterfactual_mapping")
    if not report.has_label_function:
        blockers.append(f"{relative}: missing label_function")
    if not report.has_example_preview:
        blockers.append(f"{relative}: missing example_preview")
    for control in report.missing_controls:
        blockers.append(f"{relative}: missing required control {control}")

    records_path = ROOT / str(manifest["records_path"])
    if not records_path.exists():
        blockers.append(f"{relative}: records_path does not exist")
    elif records_path.suffix == ".jsonl":
        blockers.extend(prompt_jsonl_blockers(records_path))
    return blockers


def prompt_artifact_blockers(files: Iterable[Path]) -> list[str]:
    blockers: list[str] = []
    for path in files:
        if path.suffix == ".jsonl":
            blockers.extend(prompt_jsonl_blockers(path))
        elif path.name == "manifest.yml":
            blockers.extend(dataset_manifest_blockers(path))
    return blockers


def ignored_fixture_blockers(paths: Iterable[str] = REQUIRED_TRACKED_FIXTURE_PATHS) -> list[str]:
    blockers: list[str] = []
    for relative in paths:
        path = ROOT / relative
        if not path.exists():
            blockers.append(f"required fixture missing: {relative}")
            continue
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative],
            cwd=ROOT,
            check=False,
        )
        if result.returncode == 0:
            blockers.append(f"required fixture is ignored by git: {relative}")
    return blockers


def required_gitignore_blockers() -> list[str]:
    required_patterns = {
        ".venv",
        "__pycache__/",
        ".env",
        "models/",
        "external/",
        ".arena_artifacts/",
        "dictionaries/",
        "!data/generated/refusal_proxy_prompts_v1/**",
    }
    lines = {
        line.strip()
        for line in (ROOT / ".gitignore").read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    return [f".gitignore missing {pattern}" for pattern in sorted(required_patterns - lines)]


def artifact_hygiene_blockers() -> list[str]:
    files = iter_extension_files()
    return (
        forbidden_artifact_blockers(files)
        + large_file_blockers(files)
        + prompt_artifact_blockers(files)
        + ignored_fixture_blockers()
        + required_gitignore_blockers()
    )


def main() -> None:
    files = iter_extension_files()
    blockers = artifact_hygiene_blockers()
    prompt_artifact_count = len(
        [path for path in files if path.suffix == ".jsonl" or path.name == "manifest.yml"]
    )
    print(f"extension_files_checked={len(files)}")
    print(f"prompt_artifacts_checked={prompt_artifact_count}")
    print(f"artifact_hygiene_blockers={len(blockers)}")
    if blockers:
        print("EXTENSION_ARTIFACT_HYGIENE=FAIL")
        for blocker in blockers:
            print(f"- {blocker}")
        raise SystemExit(1)
    print("EXTENSION_ARTIFACT_HYGIENE=PASS")


if __name__ == "__main__":
    main()

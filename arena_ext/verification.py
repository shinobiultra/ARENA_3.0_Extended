"""Verification report helpers for ARENA extension notebooks."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


def _json_safe(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _json_safe(value.detach().cpu().tolist())
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return repr(value)


def current_git_commit(root: Path | None = None) -> str:
    """Return the current git commit, or `unknown` outside a git checkout."""

    cwd = root if root is not None else Path.cwd()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip() or "unknown"


def file_sha256(path: Path) -> str:
    """Return the SHA256 digest for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_input_path(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        return resolved_path.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"report input is outside root: {path}") from exc


def _combined_manifest_sha256(files: list[dict[str, Any]]) -> str:
    payload = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def build_report_input_manifest(root: Path, paths: Iterable[Path]) -> dict[str, Any]:
    """Build a deterministic manifest for files that produced a report."""

    root = root.resolve()
    entries = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        if not resolved.exists() or not resolved.is_file():
            continue
        relative = _relative_input_path(root, resolved)
        if relative in seen:
            continue
        seen.add(relative)
        entries.append(
            {
                "path": relative,
                "sha256": file_sha256(resolved),
                "size_bytes": resolved.stat().st_size,
            }
        )
    entries.sort(key=lambda item: item["path"])
    return {
        "schema_version": 1,
        "algorithm": "sha256",
        "combined_sha256": _combined_manifest_sha256(entries),
        "files": entries,
    }


def verify_report_input_manifest(root: Path, manifest: dict[str, Any]) -> list[str]:
    """Return freshness blockers for a report input manifest."""

    blockers: list[str] = []
    if not isinstance(manifest, dict):
        return ["report_inputs is missing or not an object"]
    if manifest.get("schema_version") != 1:
        blockers.append("report_inputs.schema_version must be 1")
    if manifest.get("algorithm") != "sha256":
        blockers.append("report_inputs.algorithm must be 'sha256'")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        blockers.append("report_inputs.files must be a nonempty list")
        return blockers

    current_entries: list[dict[str, Any]] = []
    for index, entry in enumerate(files):
        if not isinstance(entry, dict):
            blockers.append(f"report_inputs.files[{index}] is not an object")
            continue
        relative = entry.get("path")
        expected_sha = entry.get("sha256")
        expected_size = entry.get("size_bytes")
        if not isinstance(relative, str) or not relative:
            blockers.append(f"report_inputs.files[{index}].path is missing")
            continue
        path = root / relative
        if not path.exists() or not path.is_file():
            blockers.append(f"report input missing: {relative}")
            continue
        actual_sha = file_sha256(path)
        actual_size = path.stat().st_size
        if actual_sha != expected_sha:
            blockers.append(f"report input hash changed: {relative}")
        if actual_size != expected_size:
            blockers.append(f"report input size changed: {relative}")
        current_entries.append(
            {
                "path": relative,
                "sha256": actual_sha,
                "size_bytes": actual_size,
            }
        )

    current_entries.sort(key=lambda item: item["path"])
    combined = _combined_manifest_sha256(current_entries)
    if combined != manifest.get("combined_sha256"):
        blockers.append("report_inputs.combined_sha256 does not match current inputs")
    return blockers


def cuda_environment() -> dict[str, Any]:
    """Return compact CUDA metadata without requiring torch at import time."""

    try:
        import torch
    except ImportError:
        return {
            "gpu_name": "unavailable",
            "peak_vram_gb": 0.0,
            "torch": "not_installed",
            "cuda_available": False,
        }
    if not torch.cuda.is_available():
        return {
            "gpu_name": "unavailable",
            "peak_vram_gb": 0.0,
            "torch": torch.__version__,
            "cuda_available": False,
            "cuda_version": torch.version.cuda,
        }
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 6),
        "torch": torch.__version__,
        "cuda_available": True,
        "cuda_version": torch.version.cuda,
    }


@dataclass(frozen=True)
class VerificationReport:
    notebook_id: str
    date_run: str
    git_commit: str
    report_inputs: dict[str, Any]
    gt_tier: str
    evidence_level: str
    claim_scope: str
    gpu_name: str
    peak_vram_gb: float
    wall_clock_seconds: float
    models: list[Any]
    datasets: list[Any]
    tests_passed: bool
    metrics: dict[str, Any]
    baselines: dict[str, Any]
    negative_controls: dict[str, Any]
    ood_tests: dict[str, Any]
    known_failures: list[str]
    safety_notes: list[str]
    accepted: bool

    def as_dict(self) -> dict[str, Any]:
        return _json_safe(dataclasses.asdict(self))


def build_verification_report(
    artifact_lock: dict[str, Any],
    *,
    wall_clock_seconds: float,
    metrics: dict[str, Any],
    baselines: dict[str, Any] | None = None,
    negative_controls: dict[str, Any] | None = None,
    ood_tests: dict[str, Any] | None = None,
    known_failures: list[str] | None = None,
    tests_passed: bool = True,
    accepted: bool | None = None,
    root: Path | None = None,
    date_run: str | None = None,
    report_inputs: dict[str, Any] | None = None,
    peak_vram_gb: float | None = None,
) -> VerificationReport:
    """Build a universal verification report from an artifact lock."""

    env = cuda_environment()
    failures = known_failures or []
    if accepted is None:
        accepted = tests_passed and not failures
    return VerificationReport(
        notebook_id=str(artifact_lock["notebook_id"]),
        date_run=date_run or datetime.now(UTC).isoformat(timespec="seconds"),
        git_commit=current_git_commit(root),
        report_inputs=_json_safe(report_inputs or {}),
        gt_tier=str(artifact_lock["gt_tier"]),
        evidence_level=str(artifact_lock.get("evidence_level", "notebook_contract")),
        claim_scope=str(
            artifact_lock.get(
                "claim_scope",
                "Deterministic notebook contract only unless exact real-model "
                "artifacts and controls are declared.",
            )
        ),
        gpu_name=str(env["gpu_name"]),
        peak_vram_gb=(
            float(env["peak_vram_gb"]) if peak_vram_gb is None else float(peak_vram_gb)
        ),
        wall_clock_seconds=round(float(wall_clock_seconds), 6),
        models=list(artifact_lock.get("models", [])),
        datasets=list(artifact_lock.get("datasets", [])),
        tests_passed=bool(tests_passed),
        metrics=_json_safe(metrics),
        baselines=_json_safe(baselines or {}),
        negative_controls=_json_safe(negative_controls or {}),
        ood_tests=_json_safe(ood_tests or {}),
        known_failures=list(failures),
        safety_notes=list(artifact_lock.get("safety_notes", [])),
        accepted=bool(accepted),
    )


def write_verification_report(report: VerificationReport, path: Path) -> None:
    """Write `verification_report.json` with stable formatting."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.as_dict(), indent=2, sort_keys=True) + "\n")

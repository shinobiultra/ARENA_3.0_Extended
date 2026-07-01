"""Run extension notebook contracts and write verification_report.json files."""

from __future__ import annotations

import argparse
import ast
import gc
import importlib.util
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_VERIFICATION_SPEC = importlib.util.spec_from_file_location(
    "arena_ext_verification_helpers",
    ROOT / "arena_ext" / "verification.py",
)
if _VERIFICATION_SPEC is None or _VERIFICATION_SPEC.loader is None:
    raise RuntimeError("could not load arena_ext/verification.py")
_VERIFICATION_MODULE = importlib.util.module_from_spec(_VERIFICATION_SPEC)
sys.modules[_VERIFICATION_SPEC.name] = _VERIFICATION_MODULE
_VERIFICATION_SPEC.loader.exec_module(_VERIFICATION_MODULE)
build_report_input_manifest = _VERIFICATION_MODULE.build_report_input_manifest
build_verification_report = _VERIFICATION_MODULE.build_verification_report
write_verification_report = _VERIFICATION_MODULE.write_verification_report


def is_extension_section(number: str) -> bool:
    if number in {"0.6", "1.6"}:
        return True
    try:
        return int(number.split(".", maxsplit=1)[0]) >= 5
    except ValueError:
        return False


def extension_records(section_filter: set[str] | None = None) -> list[dict[str, Any]]:
    config = yaml.safe_load((ROOT / "infrastructure/core/config.yaml").read_text())
    records = []
    for chapter_name, chapter in config["chapters"].items():
        chapter_dir = ROOT / chapter_name
        for section in chapter.get("sections", []):
            number = str(section.get("number", ""))
            if not is_extension_section(number):
                continue
            if section_filter is not None and number not in section_filter:
                continue
            exercise_dir = chapter_dir / "exercises" / section["exercise_dir"]
            records.append(
                {
                    "number": number,
                    "title": section["title"],
                    "exercise_dir": exercise_dir,
                    "instruction_page": chapter_dir / "instructions" / "pages" / section["page_file"],
                    "solutions_path": exercise_dir / "solutions.py",
                    "lock_path": exercise_dir / "artifacts.lock.yml",
                    "report_path": exercise_dir / "verification_report.json",
                }
            )
    return records


def load_solution(path: Path, index: int) -> Any:
    spec = importlib.util.spec_from_file_location(f"arena_extension_solution_{index}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _module_path(module_name: str) -> Path | None:
    if not module_name.startswith("arena_ext"):
        return None
    relative = Path(*module_name.split("."))
    module_file = ROOT / f"{relative}.py"
    if module_file.exists():
        return module_file
    package_file = ROOT / relative / "__init__.py"
    if package_file.exists():
        return package_file
    return None


def _arena_ext_import_paths(path: Path, seen: set[Path] | None = None) -> set[Path]:
    seen = seen or set()
    if path in seen or not path.exists():
        return set()
    seen.add(path)
    imported: set[Path] = set()
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return imported

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_path = _module_path(alias.name)
                if module_path is not None:
                    imported.add(module_path)
        elif isinstance(node, ast.ImportFrom) and node.module:
            base_path = _module_path(node.module)
            if base_path is not None:
                imported.add(base_path)
            if node.module == "arena_ext":
                imported.add(ROOT / "arena_ext/__init__.py")
                for alias in node.names:
                    child_path = _module_path(f"arena_ext.{alias.name}")
                    if child_path is not None:
                        imported.add(child_path)

    for imported_path in list(imported):
        imported.update(_arena_ext_import_paths(imported_path, seen))
    return imported


def report_input_paths(record: dict[str, Any], lock: dict[str, Any] | None = None) -> list[Path]:
    """Return CPU-checkable files that produced a report."""

    exercise_dir = Path(record["exercise_dir"])
    paths = [
        ROOT / "scripts/run_extension_verification_reports.py",
        ROOT / "arena_ext/verification.py",
        record.get("solutions_path"),
        exercise_dir / "README.md",
        exercise_dir / "tests.py",
        exercise_dir / "utils.py",
        record.get("lock_path"),
        record.get("instruction_page"),
        exercise_dir / "verification_report.schema.json",
        exercise_dir / "expected_outputs/README.md",
        exercise_dir / "expected_outputs/smoke_test.json",
        exercise_dir / "expected_outputs/reference_metrics.json",
    ]
    paths.extend(sorted(exercise_dir.glob("*_exercises.ipynb")))
    paths.extend(sorted(exercise_dir.glob("*_solutions.ipynb")))
    solutions_path = record.get("solutions_path")
    if isinstance(solutions_path, Path):
        paths.extend(sorted(_arena_ext_import_paths(solutions_path)))

    for extra in (lock or {}).get("freshness_inputs", []):
        extra_path = ROOT / str(extra)
        paths.append(extra_path)

    return [path for path in paths if isinstance(path, Path)]


def cleanup_cuda_allocations() -> None:
    try:
        import torch
    except ImportError:
        return
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def reset_cuda_peak() -> None:
    cleanup_cuda_allocations()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def runtime_cuda_metadata() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "torch_version": None,
            "cuda_version": None,
            "gpu_total_memory_gb": None,
        }

    total_memory_gb = None
    if torch.cuda.is_available():
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    return {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_total_memory_gb": total_memory_gb,
    }


def stamp_runtime_cuda_metadata(gpu_result: Any) -> Any:
    if not isinstance(gpu_result, dict):
        return gpu_result
    stamped = dict(gpu_result)
    for key, value in runtime_cuda_metadata().items():
        stamped.setdefault(key, value)
    return stamped


def contract_failures(contract: Any) -> list[str]:
    if not isinstance(contract, dict):
        return []
    failures = []
    for key in ("contract_passed", "tests_passed", "accepted"):
        if key in contract and contract[key] is False:
            failures.append(f"{key} was false")
    return failures


def flatten_metric_values(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested metric dicts/lists while retaining leaf and container values."""

    values: dict[str, Any] = {}
    if prefix:
        values[prefix] = obj
    if isinstance(obj, dict):
        for key, value in obj.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            values.update(flatten_metric_values(value, child_prefix))
    elif isinstance(obj, list):
        for index, value in enumerate(obj):
            child_prefix = f"{prefix}.{index}" if prefix else str(index)
            values.update(flatten_metric_values(value, child_prefix))
    return values


def _metric_path_priority(path: str, base_key: str) -> tuple[int, int, str]:
    if path == f"gpu_test.{base_key}":
        return (0, path.count("."), path)
    if path == f"notebook_contract.{base_key}":
        return (1, path.count("."), path)
    if path.startswith("gpu_test."):
        return (2, path.count("."), path)
    if path.startswith("notebook_contract."):
        return (3, path.count("."), path)
    return (4, path.count("."), path)


def find_metric_value(metrics: dict[str, Any], key: str) -> tuple[str, Any] | None:
    flat = flatten_metric_values(metrics)
    candidates = [path for path in flat if path.rsplit(".", maxsplit=1)[-1] == key]
    if not candidates:
        return None
    candidates.sort(key=lambda path: _metric_path_priority(path, key))
    best_path = candidates[0]
    return best_path, flat[best_path]


def _is_number(value: Any) -> bool:
    return isinstance(value, float | int) and not isinstance(value, bool)


def _values_match(actual: Any, expected: Any) -> bool:
    if _is_number(expected) and _is_number(actual):
        return abs(float(actual) - float(expected)) <= 1e-6
    return actual == expected


def expected_metric_failures(expected_metrics: dict[str, Any], metrics: dict[str, Any]) -> list[str]:
    """Validate artifact-lock expected metrics against actual report metrics."""

    failures: list[str] = []
    gpu_test = metrics.get("gpu_test", {})
    for key, expected in expected_metrics.items():
        if key in {"tests_passed", "accepted"}:
            continue
        if key == "max_allowed_gpu_gb":
            actual = gpu_test.get("peak_vram_gb") if isinstance(gpu_test, dict) else None
            if actual is None:
                failures.append("expected max_allowed_gpu_gb but gpu_test.peak_vram_gb is missing")
            elif float(actual) > float(expected):
                failures.append(
                    f"expected gpu_test.peak_vram_gb <= {expected}, got {actual}"
                )
            continue

        comparison = "exact"
        base_key = key
        exact_found = find_metric_value(metrics, key)
        if exact_found is not None:
            path, actual = exact_found
            use_suffix_threshold = (
                key.endswith(("_min", "_max"))
                and _is_number(expected)
                and isinstance(actual, bool)
            )
            if not use_suffix_threshold and not _values_match(actual, expected):
                failures.append(f"expected {path} == {expected!r}, got {actual!r}")
            if not use_suffix_threshold:
                continue

        if key.endswith("_min"):
            comparison = "min"
            base_key = key[:-4]
        elif key.endswith("_max"):
            comparison = "max"
            base_key = key[:-4]

        found = find_metric_value(metrics, base_key)
        if found is None:
            failures.append(f"expected metric {key!r} but no actual metric {base_key!r} was found")
            continue

        path, actual = found
        if comparison == "min":
            if not _is_number(actual) or float(actual) < float(expected):
                failures.append(f"expected {path} >= {expected}, got {actual!r}")
        elif comparison == "max":
            if not _is_number(actual) or float(actual) > float(expected):
                failures.append(f"expected {path} <= {expected}, got {actual!r}")
        elif not _values_match(actual, expected):
            failures.append(f"expected {path} == {expected!r}, got {actual!r}")
    return failures


def gpu_evidence_summary(gpu_result: Any) -> dict[str, Any]:
    """Classify GPU evidence without equating placeholders with CUDA runs."""

    if not isinstance(gpu_result, dict):
        return {
            "category": "missing",
            "uses_cuda": False,
            "placeholder_only": True,
            "section_specific_metric_keys": [],
        }

    generic_keys = {
        "cuda_available",
        "device",
        "full_path",
        "gpu_name",
        "gpu_total_memory_gb",
        "peak_vram_gb",
        "smoke_test_available",
        "torch_version",
        "cuda_version",
        "within_vram_budget",
    }
    section_specific_keys = sorted(key for key in gpu_result if key not in generic_keys)
    placeholder_only = (
        gpu_result.get("smoke_test_available") is True
        and not section_specific_keys
        and gpu_result.get("cuda_available") is not True
    )
    uses_cuda = gpu_result.get("cuda_available") is True
    if uses_cuda and section_specific_keys:
        category = "cuda_section_metric"
    elif uses_cuda:
        category = "cuda_environment_or_budget"
    elif placeholder_only:
        category = "placeholder_only"
    elif section_specific_keys:
        category = "cpu_or_budget_metric"
    else:
        category = "missing"

    return {
        "category": category,
        "uses_cuda": uses_cuda,
        "placeholder_only": placeholder_only,
        "section_specific_metric_keys": section_specific_keys,
    }


def run_record(record: dict[str, Any], index: int, max_vram_gb: float) -> bool:
    failures: list[str] = []
    metrics: dict[str, Any] = {}
    start = time.perf_counter()
    reset_cuda_peak()
    try:
        module = load_solution(record["solutions_path"], index)
        smoke = getattr(module, "run_smoke_test")
        gpu = getattr(module, "run_gpu_test")
        metrics["notebook_contract"] = smoke(cpu=False)
        failures.extend(contract_failures(metrics["notebook_contract"]))
        metrics["gpu_test"] = stamp_runtime_cuda_metadata(gpu(max_vram_gb=max_vram_gb))
        metrics["gpu_evidence"] = gpu_evidence_summary(metrics["gpu_test"])
    except Exception as exc:  # noqa: BLE001 - report should capture failures.
        failures.append(repr(exc))
    elapsed = time.perf_counter() - start

    artifact_lock = yaml.safe_load(record["lock_path"].read_text())
    failures.extend(
        expected_metric_failures(artifact_lock.get("expected_metrics", {}), metrics)
    )
    report = build_verification_report(
        artifact_lock,
        root=ROOT,
        report_inputs=build_report_input_manifest(
            ROOT,
            report_input_paths(record, artifact_lock),
        ),
        wall_clock_seconds=elapsed,
        metrics=metrics,
        baselines={
            "declared_controls": artifact_lock.get("controls", []),
            "expected_metrics": artifact_lock.get("expected_metrics", {}),
        },
        negative_controls={
            "declared_controls": [
                control
                for control in artifact_lock.get("controls", [])
                if "negative" in str(control).lower() or "random" in str(control).lower()
            ]
        },
        ood_tests={"declared": "required_where_applicable"},
        known_failures=failures,
        tests_passed=not failures,
        accepted=not failures,
        peak_vram_gb=(
            metrics.get("gpu_test", {}).get("peak_vram_gb")
            if isinstance(metrics.get("gpu_test"), dict)
            else None
        ),
    )
    write_verification_report(report, record["report_path"])
    status = "PASS" if report.accepted else "FAIL"
    print(f"{status} {record['number']} {record['title']}")
    cleanup_cuda_allocations()
    return report.accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", action="append", help="Run only one section number.")
    parser.add_argument("--max-vram-gb", type=float, default=24.0)
    args = parser.parse_args()

    section_filter = set(args.section) if args.section else None
    records = extension_records(section_filter)
    if not records:
        raise SystemExit("No extension sections matched.")

    passed = 0
    for index, record in enumerate(records):
        if run_record(record, index, args.max_vram_gb):
            passed += 1
    print(f"Wrote {len(records)} verification reports; {passed} accepted.")
    if passed != len(records):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

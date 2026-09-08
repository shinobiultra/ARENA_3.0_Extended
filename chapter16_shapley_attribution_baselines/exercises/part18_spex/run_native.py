"""Cold CPU execution and source-only reproduction for the fixed joint SPEX study.

Importing this module never imports a model, starts a study, or reads model
weights. Native execution is an explicit CLI/API operation after protocol freeze.
The supervisor retains failed children and reference outcomes without changing
the scientific design or treating poor predictive performance as a gate failure.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import psutil


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SPEX_MODULE = "chapter16_shapley_attribution_baselines.exercises.part18_spex.spex_methods"
PROXY_MODULE = "chapter16_shapley_attribution_baselines.exercises.part19_proxyspex.proxy_methods"
STUDY_MODULE = "chapter16_shapley_attribution_baselines.exercises.part18_spex.native_study"
REFERENCE_STAGE_IDS = tuple(f"{variant}_b{bits}" for bits in (3, 4, 5)
                            for variant in ("author_spex", "author_proxy", "maintained_proxy"))
CPU_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
    "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4", "BLIS_NUM_THREADS": "4",
    "VECLIB_MAXIMUM_THREADS": "4", "OMP_DYNAMIC": "FALSE", "TOKENIZERS_PARALLELISM": "false",
}


@dataclass(frozen=True)
class ResourceLimits:
    wall_seconds: float = 10800.0
    rss_bytes: int = 8 * 1024**3
    output_bytes: int = 8 * 1024**3
    sample_seconds: float = 0.05


NATIVE_LIMITS = ResourceLimits()
NATIVE_TORCH_THREADS = 8
NATIVE_INTEROP_THREADS = 4
REFERENCE_WALL_SECONDS = 300.0
REFERENCE_RSS_BYTES = 8 * 1024**3
REFERENCE_OUTPUT_BYTES = 256 * 1024**2
LOGIT_TOLERANCE = {"atol": 1e-5, "rtol": 1e-6}
ESTIMATOR_TOLERANCE = {"atol": 1e-8, "rtol": 1e-7}


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _utc_now():
    return datetime.now(timezone.utc).isoformat()


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _file_inventory(directory):
    """Logical file bytes, including sparse files; never follow an output symlink."""
    directory = Path(directory)
    records = {}
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Output must not contain a symlink: {path}")
        if path.is_file():
            records[path.relative_to(directory).as_posix()] = path.stat().st_size
    return records


def _source_records(paths):
    result = {}
    for path in sorted({Path(p).resolve() for p in paths}):
        name = str(path)
        try:
            result[name] = {"sha256": digest(path), "size_bytes": path.stat().st_size}
        except OSError as error:
            result[name] = {"error": type(error).__name__ + ": " + str(error)}
    return result


def _live_processes(process, seen):
    try:
        for item in [process, *process.children(recursive=True)]:
            seen[item.pid] = item
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    live = []
    for item in list(seen.values()):
        try:
            if item.is_running() and item.status() != psutil.STATUS_ZOMBIE:
                live.append(item)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return live


def _stop_tree(child, process, seen):
    """Only stop this supervisor's new process group and observed descendants."""
    live = _live_processes(process, seen)
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for item in reversed(live):
        try:
            item.terminate()
        except psutil.NoSuchProcess:
            pass
    # Popen alone must reap its own child, or its actual signal/exit code can be lost.
    _, survivors = psutil.wait_procs([item for item in live if item.pid != child.pid], timeout=0.5)
    for item in survivors:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        child.wait(timeout=2)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def _finalize_receipt(output, record, cap):
    """Include the receipt itself and converge its self-reported byte inventory."""
    target = Path(output) / "execution.json"
    for _ in range(8):
        _write_json(target, record)
        files = _file_inventory(output)
        size = sum(files.values())
        stable = record.get("output_files") == files and record.get("new_output_bytes") == size
        record["output_files"], record["new_output_bytes"] = files, size
        if size > cap:
            if "output_bytes" not in record["resource_violations"]:
                record["resource_violations"].append("output_bytes")
                stable = False
            record["status"] = "FAIL"
        if stable:
            return
    _write_json(target, record)


def supervise(command, output, *, limits=NATIVE_LIMITS, source_paths=(), cwd=None,
              env=None, manifest_extra=None):
    """Run a new child and retain an actual receipt, even on timeout or exception.

Custom limits are for synthetic tests and reference stages. ``run_native`` always
uses the fixed NATIVE_LIMITS. This lower-level function never invents a success
from a log line and never deletes partial output.
"""
    output = Path(output).resolve()
    if (not math.isfinite(limits.wall_seconds) or limits.wall_seconds < 0
            or limits.rss_bytes < 1 or limits.output_bytes < 1 or limits.sample_seconds <= 0):
        raise ValueError("Resource limits must be finite and nonnegative, with positive byte/sample bounds.")
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    environment = os.environ.copy()
    environment.update(env or {})
    environment.update(CPU_ENVIRONMENT)
    record = {
        "schema_version": 1, "status": "RUNNING", "started_at_utc": _utc_now(),
        "command": list(map(str, command)), "cwd": str(Path(cwd or ROOT).resolve()),
        "limits": asdict(limits), "cpu_environment": CPU_ENVIRONMENT,
        "source_inputs": {}, "changed_sources": [], "returncode": None,
        "sampled_peak_rss_bytes": 0, "rss_samples": 0, "observed_process_ids": [],
        "resource_violations": [], "failures": [], "output_directory": str(output),
        **(manifest_extra or {}),
    }
    child = process = None
    seen = {}
    try:
        before = _source_records(source_paths)
        record["source_inputs"] = before
        if any("error" in value for value in before.values()):
            raise ValueError("Required source/input file is unreadable before execution.")
        for index, (name, metadata) in enumerate(before.items()):
            source = Path(name)
            # Model snapshots are hash-bound inputs, never copied into learner output.
            if source.suffix in {".py", ".md", ".toml", ".lock", ".json", ".yaml", ".yml"}:
                try:
                    relative = source.relative_to(ROOT)
                except ValueError:
                    relative = Path("external_inputs") / f"{index:04d}_{source.name}"
                destination = output / "source_snapshot" / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, destination)
                if digest(destination) != metadata["sha256"]:
                    raise ValueError(f"Source changed while taking its snapshot: {name}")
        _write_json(output / "execution.json", record)
        if time.monotonic() - start > limits.wall_seconds:
            record["resource_violations"].append("wall_seconds")
            raise TimeoutError("Wall cap reached before starting the child.")
        if sum(_file_inventory(output).values()) > limits.output_bytes:
            record["resource_violations"].append("output_bytes")
            raise ValueError("Output cap reached before starting the child.")
        with (output / "child.log").open("w") as log:
            child = subprocess.Popen(record["command"], cwd=record["cwd"], env=environment,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            process = psutil.Process(child.pid)
            while True:
                live = _live_processes(process, seen)
                rss = 0
                for item in live:
                    try:
                        rss += item.memory_info().rss
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                record["sampled_peak_rss_bytes"] = max(record["sampled_peak_rss_bytes"], rss)
                record["rss_samples"] += 1
                elapsed = time.monotonic() - start
                size = sum(_file_inventory(output).values())
                for name, measured, cap in (("wall_seconds", elapsed, limits.wall_seconds),
                                             ("rss_bytes", rss, limits.rss_bytes),
                                             ("output_bytes", size, limits.output_bytes)):
                    if measured > cap and name not in record["resource_violations"]:
                        record["resource_violations"].append(name)
                if record["resource_violations"]:
                    _stop_tree(child, process, seen)
                    break
                if child.poll() is not None:
                    descendants = [item.pid for item in live if item.pid != child.pid]
                    if descendants:
                        record["failures"].append({"kind": "live_descendants_after_child_exit", "pids": descendants})
                        _stop_tree(child, process, seen)
                    break
                time.sleep(limits.sample_seconds)
            record["returncode"] = child.wait()
    except BaseException as error:
        record["failures"].append({"kind": type(error).__name__, "message": str(error),
                                    "traceback": traceback.format_exc()})
        if child is not None and process is not None:
            _stop_tree(child, process, seen)
            record["returncode"] = child.returncode
    finally:
        after = _source_records(source_paths)
        before = record["source_inputs"]
        record["source_inputs_after"] = after
        record["changed_sources"] = sorted(name for name in set(before) | set(after)
                                             if before.get(name) != after.get(name))
        record["source_inputs_combined_sha256"] = hashlib.sha256(
            json.dumps(before, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        record["observed_process_ids"] = sorted(seen)
        record["elapsed_seconds"] = time.monotonic() - start
        record["finished_at_utc"] = _utc_now()
        try:
            record["output_files"] = _file_inventory(output)
            record["new_output_bytes"] = sum(record["output_files"].values())
        except ValueError as error:
            record["failures"].append({"kind": "output_inventory", "message": str(error)})
            record["new_output_bytes"] = None
        for name, measured, cap in (("wall_seconds", record["elapsed_seconds"], limits.wall_seconds),
                                    ("output_bytes", record["new_output_bytes"], limits.output_bytes)):
            if measured is not None and measured > cap and name not in record["resource_violations"]:
                record["resource_violations"].append(name)
        record["status"] = ("COMPLETE" if record["returncode"] == 0 and not record["failures"]
                             and not record["resource_violations"] and not record["changed_sources"] else "FAIL")
        if record["new_output_bytes"] is None:
            _write_json(output / "execution.json", record)
        else:
            _finalize_receipt(output, record, limits.output_bytes)
    return record


def run_reference_stage(command, output, *, stage_id, cwd=None, env=None):
    """One of nine frozen reference stages, inside the inherited joint deadline.

Failure and timeout are source outcomes. The caller must keep the returned
receipt and must not substitute another estimator or silently retry the stage.
"""
    stage_ids = json.loads(os.environ.get("ARENA_SPEX_REFERENCE_STAGE_IDS", "[]"))
    if stage_ids != list(REFERENCE_STAGE_IDS) or stage_id not in stage_ids:
        raise ValueError("Reference stage is not one of the nine frozen IDs.")
    deadline = float(os.environ["ARENA_SPEX_JOINT_DEADLINE_UNIX"])
    registry_path = Path(os.environ["ARENA_SPEX_REFERENCE_REGISTRY"])
    if not Path(output).resolve().is_relative_to(registry_path.parent.resolve()):
        raise ValueError("Reference output must remain inside the supervised joint output directory.")
    with registry_path.open("a+") as registry:
        fcntl.flock(registry, fcntl.LOCK_EX)
        registry.seek(0)
        previous = [json.loads(line) for line in registry if line.strip()]
        if any(item["stage_id"] == stage_id for item in previous):
            raise ValueError("A reference stage cannot be retried inside a fixed run.")
        registry.seek(0, os.SEEK_END)
        registry.write(json.dumps({"stage_id": stage_id, "output_directory": str(Path(output).resolve()),
                                   "started_at_utc": _utc_now()}) + "\n")
        registry.flush()
    remaining = max(0.0, deadline - time.time())
    limits = ResourceLimits(wall_seconds=min(REFERENCE_WALL_SECONDS, remaining),
                            rss_bytes=REFERENCE_RSS_BYTES, output_bytes=REFERENCE_OUTPUT_BYTES)
    record = supervise(command, output, limits=limits, cwd=cwd, env=env,
                       manifest_extra={"reference_stage_id": stage_id, "reference_stage_cap_seconds": 300.0,
                                       "joint_deadline_unix": deadline, "is_source_outcome": True})
    return record


def _native_source_paths():
    """Only sources executed by this native run; lessons/builders/tests bind later."""
    return {
        *(HERE / name for name in (
            "native_data.py", "native_query_plan.py", "native_oracle.py", "native_study.py",
            "run_native.py", "spex_methods.py", "spex_coding.py", "reference_bridge.py", "native_protocol.md",
        )),
        HERE.parent / "part19_proxyspex/proxy_methods.py",
        HERE / "reference_environment/pyproject.toml", HERE / "reference_environment/uv.lock",
        ROOT / "pyproject.toml", ROOT / "uv.lock",
    }


def _execution_sources(freeze, spex_module, proxy_module, extra_paths):
    paths = _native_source_paths()
    for source in paths:
        relative = source.resolve().relative_to(ROOT).as_posix()
        if freeze["source_inputs"].get(relative) != digest(source):
            raise ValueError(f"Actual native input is missing from the freeze or changed: {relative}")
    for name, expected in freeze["source_inputs"].items():
        source = (ROOT / name).resolve()
        source.relative_to(ROOT)
        if digest(source) != expected:
            raise ValueError(f"Frozen native source changed: {name}")
        paths.add(source)
    for name in (spex_module, proxy_module, STUDY_MODULE):
        spec = importlib.util.find_spec(name)
        if spec is None or spec.origin is None:
            raise ValueError(f"Missing source module: {name}")
        source = Path(spec.origin).resolve()
        relative = source.relative_to(ROOT).as_posix()
        if freeze["source_inputs"].get(relative) != digest(source):
            raise ValueError(f"Module is not bound by the native freeze: {name}")
        paths.add(source)
    paths.update(Path(path).resolve() for path in (extra_paths or ()))
    return sorted(paths)


def _run_native_impl(output, *, freeze_path, plan_dir, snapshot_path, spex_module=SPEX_MODULE,
                     proxy_module=PROXY_MODULE, source_paths=None):
    """Run only fixed design inputs and source modules in a cold supervised child."""
    start = time.time()
    output, freeze_path, plan_dir, snapshot_path = map(
        lambda path: Path(path).resolve(), (output, freeze_path, plan_dir, snapshot_path))
    if output.exists():
        raise ValueError("A native execution requires a new, nonexistent output directory.")
    freeze = json.loads(freeze_path.read_text())
    if freeze.get("reference_stage_ids") != list(REFERENCE_STAGE_IDS):
        raise ValueError("Freeze must bind all nine ordered reference stage IDs.")
    sources = _execution_sources(freeze, spex_module, proxy_module, source_paths)
    sources += [freeze_path, *sorted(path for path in plan_dir.rglob("*") if path.is_file()),
                *sorted(path for path in snapshot_path.rglob("*") if path.is_file())]
    command = [sys.executable, str(Path(__file__).resolve()), "--child-output", str(output),
               "--freeze-path", str(freeze_path), "--plan-dir", str(plan_dir),
               "--snapshot-path", str(snapshot_path), "--spex-module", spex_module,
               "--proxy-module", proxy_module]
    environment = {
        "ARENA_SPEX_SUPERVISED_CHILD": "1", "ARENA_SPEX_JOINT_DEADLINE_UNIX": str(start + 10800),
        "ARENA_SPEX_REFERENCE_STAGE_IDS": json.dumps(list(REFERENCE_STAGE_IDS)),
        "ARENA_SPEX_REFERENCE_REGISTRY": str(output / "reference_stages.jsonl"),
    }
    preflight_elapsed = time.time() - start
    remaining_limits = ResourceLimits(wall_seconds=max(0.0, 10800 - preflight_elapsed))
    record = supervise(command, output, source_paths=sources, env=environment, limits=remaining_limits,
                       manifest_extra={"execution_kind": "cold_source_native", "model_device": "cpu",
                                       "native_torch_threads": NATIVE_TORCH_THREADS,
                                       "native_interop_threads": NATIVE_INTEROP_THREADS,
                                       "numerical_threads": 4,
                                       "child_received_learner_payloads": False,
                                       "fixed_inputs": {"freeze_path": str(freeze_path), "plan_dir": str(plan_dir),
                                                        "snapshot_path": str(snapshot_path)},
                                       "method_modules": {"spex": spex_module, "proxy": proxy_module},
                                       "reference_stage_ids": list(REFERENCE_STAGE_IDS),
                                       "joint_limits": asdict(NATIVE_LIMITS),
                                       "preflight_elapsed_seconds": preflight_elapsed})
    record["joint_elapsed_seconds"] = time.time() - start
    if record["joint_elapsed_seconds"] > 10800:
        record["status"] = "FAIL"
        if "wall_seconds" not in record["resource_violations"]:
            record["resource_violations"].append("wall_seconds")
    if record["status"] == "COMPLETE":
        registry = output / "reference_stages.jsonl"
        stages = [json.loads(line) for line in registry.read_text().splitlines()] if registry.exists() else []
        if [item["stage_id"] for item in stages] != list(REFERENCE_STAGE_IDS):
            record["status"] = "FAIL"
            record["failures"].append({"kind": "reference_stage_inventory", "observed": stages})
        record["reference_stage_receipts"] = [str(Path(item["output_directory"]) / "execution.json") for item in stages]
    _finalize_receipt(output, record, NATIVE_LIMITS.output_bytes)
    return record


def run_native(output, *, freeze_path, plan_dir, snapshot_path, spex_module=SPEX_MODULE,
               proxy_module=PROXY_MODULE, source_paths=None):
    """Cold execution, including a retained failure receipt for rejected preflight."""
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("A native execution requires a new, nonexistent output directory.")
    started = time.monotonic()
    try:
        return _run_native_impl(output, freeze_path=freeze_path, plan_dir=plan_dir,
                                snapshot_path=snapshot_path, spex_module=spex_module,
                                proxy_module=proxy_module, source_paths=source_paths)
    except BaseException as error:
        failure = {"kind": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        output.mkdir(parents=True, exist_ok=True)
        target = output / "execution.json"
        record = json.loads(target.read_text()) if target.is_file() else {
            "schema_version": 1, "execution_kind": "rejected_native_preflight", "returncode": None,
            "sampled_peak_rss_bytes": 0, "rss_samples": 0, "observed_process_ids": [],
            "source_inputs": _source_records([freeze_path, *(source_paths or ())]),
            "changed_sources": [], "resource_violations": [], "failures": [],
            "child_received_learner_payloads": False, "cpu_environment": CPU_ENVIRONMENT,
            "native_torch_threads": NATIVE_TORCH_THREADS,
            "native_interop_threads": NATIVE_INTEROP_THREADS, "numerical_threads": 4,
            "joint_limits": asdict(NATIVE_LIMITS), "output_directory": str(output),
        }
        record["status"] = "FAIL"
        record["failures"].append(failure)
        record["elapsed_seconds"] = time.monotonic() - started
        record["finished_at_utc"] = _utc_now()
        try:
            _finalize_receipt(output, record, NATIVE_LIMITS.output_bytes)
        except ValueError:
            _write_json(target, record)
        return record


def _child_run(args):
    if os.environ.get("ARENA_SPEX_SUPERVISED_CHILD") != "1":
        raise ValueError("Native child requires the resource supervisor.")
    if any(os.environ.get(key) != value for key, value in CPU_ENVIRONMENT.items()):
        raise ValueError("CPU-only four-thread numerical environment changed.")
    import torch

    # Prospective root amendment: eight GPT2 inference threads; numerical and
    # reference-fitting environments remain four, as does Torch interop.
    torch.set_num_threads(NATIVE_TORCH_THREADS)
    torch.set_num_interop_threads(NATIVE_INTEROP_THREADS)
    study = importlib.import_module(STUDY_MODULE)
    study.run_study(Path(args.child_output) / "study",
                    spex_methods=importlib.import_module(args.spex_module),
                    proxy_methods=importlib.import_module(args.proxy_module),
                    freeze_path=args.freeze_path, plan_dir=args.plan_dir, snapshot_path=args.snapshot_path)


# These are operational measurements, not fitted values. Every excluded value is
# retained verbatim in the comparison receipt. Unknown fields remain strict.
_OPERATIONAL_KEYS = {
    "fit_seconds", "elapsed_seconds", "wall_seconds", "started_at_utc", "finished_at_utc",
    "timestamp", "sampled_peak_rss_bytes", "rss_samples", "observed_process_ids",
    "output_directory", "cache_absolute_path", "database_path", "database_sha256",
    "database_bytes", "physical_database_sha256", "physical_database_bytes",
    "child_log_path", "receipt_path", "execution_receipt", "command", "cwd",
    "new_output_bytes", "output_files", "joint_deadline_unix",
}


def _exclusion_reason(path):
    if path in (("prediction_boundary_hashes",), ("oracle_value_hashes",)):
        return "Physical record link is validated by load_study; decoded scientific values are compared separately."
    if path == ("oracle_inventory", "database"):
        return "Physical SQLite path/hash/size include raw timing receipts; byte identity is verified separately and exported rows are compared."
    if len(path) == 3 and path[0] == "audit_records" and path[-1] == "hashes":
        return "Physical audit record link is validated by load_study; all decoded audit values are compared."
    if path and path[0] == "references" and path[-1] == "training_values_float64_sha256":
        return "Reference-input byte digest is retained; underlying physical query values are compared under the frozen logit tolerance."
    if path and path[0] == "references" and path[-1] == "operational":
        return "Explicit reference-bridge operational receipt; deterministic source outcome and fitted values remain compared."
    if len(path) == 3 and path[0] == "conditions" and path[-1] == "hashes":
        return "Physical saved-record hashes include measured fit_seconds; decoded values are compared in results."
    if len(path) == 3 and path[0] == "results" and path[-1] == "fit_seconds":
        return "Measured fit duration; retained in the original and fresh receipts."
    if path and path[0] in {"oracle_inventory", "oracle_receipts", "references"} and path[-1] in _OPERATIONAL_KEYS:
        return "Explicit operational timing/process/cache field; scientific values, statuses and counts remain compared."
    return None


def _path_text(path):
    result = "root"
    for key in path:
        result += "." + key if isinstance(key, str) and key.isidentifier() else "[" + repr(key) + "]"
    return result


def _json_value(value):
    if isinstance(value, np.ndarray):
        return {"dtype": value.dtype.str, "shape": list(value.shape), "values": _json_value(value.tolist())}
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": repr(value)}
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {"unsupported_type": type(value).__name__, "repr": repr(value)}


def _inventory(value, *, comparable_only=False):
    counts = {"arrays": 0, "array_elements": 0, "scalars": 0, "containers": 0}
    def visit(item, path):
        if comparable_only and _exclusion_reason(path):
            return
        if isinstance(item, np.ndarray):
            counts["arrays"] += 1
            counts["array_elements"] += int(item.size)
        elif isinstance(item, dict):
            counts["containers"] += 1
            for key, child in item.items():
                visit(child, (*path, key))
        elif isinstance(item, (tuple, list)):
            counts["containers"] += 1
            for index, child in enumerate(item):
                visit(child, (*path, index))
        else:
            counts["scalars"] += 1
    visit(value, ())
    return counts


def _tolerance(path):
    exact_parameters = {"selected_parameters", "best_params", "best_params_", "selected_alpha",
                        "selected_index", "requested_energy_fraction"}
    if any(key in exact_parameters for key in path if isinstance(key, str)):
        return {"atol": 0.0, "rtol": 0.0}
    text = ".".join(map(str, path)).lower()
    if (path and path[0] == "oracle_values") or any(
        token in text for token in ("logit", "audit_values", "train_values", "unmasked", "counterfactual")
    ):
        return LOGIT_TOLERANCE
    return ESTIMATOR_TOLERANCE


def compare_payloads(original, fresh):
    """Compare every leaf, reporting all changed numerical coordinates.

Keys (including tuple supports), container types/order, array shape/dtype, integer
IDs, counts, selections and statuses are exact. Floats may meet the prospective
logit/estimator tolerances but never become an exact match. A tiny logit change
that changes a discrete tree/support/CV choice is therefore a real failure.
"""
    report = {
        "schema_version": 1, "status": "PASS", "comparison": "structural_exact_with_disclosed_float_tolerances",
        "tolerances": {"logits": LOGIT_TOLERANCE, "estimators": ESTIMATOR_TOLERANCE},
        "counts": {"original": _inventory(original, comparable_only=True),
                   "fresh": _inventory(fresh, comparable_only=True)},
        "full_payload_counts": {"original": _inventory(original), "fresh": _inventory(fresh)},
        "exact_matches": [], "numerical_differences": [], "mismatches": [], "excluded_metadata": [],
    }
    if report["counts"]["original"] != report["counts"]["fresh"]:
        report["mismatches"].append({"path": "root", "kind": "inventory_counts", **report["counts"]})

    def mismatch(path, kind, left, right):
        report["mismatches"].append({"path": _path_text(path), "kind": kind,
                                     "original": _json_value(left), "fresh": _json_value(right)})

    def compare_arrays(left, right, path):
        name = _path_text(path)
        if left.shape != right.shape or left.dtype != right.dtype:
            mismatch(path, "array_shape_or_dtype", {"shape": left.shape, "dtype": str(left.dtype)},
                     {"shape": right.shape, "dtype": str(right.dtype)})
            return
        if left.dtype.kind not in "biufUS":
            mismatch(path, "unsupported_array_dtype", str(left.dtype), str(right.dtype))
            return
        a, b = np.ascontiguousarray(left), np.ascontiguousarray(right)
        a_hash = hashlib.sha256(a.tobytes()).hexdigest()
        b_hash = hashlib.sha256(b.tobytes()).hexdigest()
        metadata = {"path": name, "shape": list(left.shape), "dtype": a.dtype.str,
                    "elements": int(a.size), "original_sha256": a_hash, "fresh_sha256": b_hash}
        if a.dtype.kind not in "US" and (not np.isfinite(a).all() or not np.isfinite(b).all()):
            mismatch(path, "nonfinite_numerical_values", a, b)
            return
        if a_hash == b_hash:
            report["exact_matches"].append(metadata)
            return
        if a.dtype.kind in "US":
            indices = np.flatnonzero(a.reshape(-1) != b.reshape(-1))
            mismatch(path, "exact_string_array", {
                "flat_indices": indices.tolist(), "values": a.reshape(-1)[indices].tolist(),
            }, {"flat_indices": indices.tolist(), "values": b.reshape(-1)[indices].tolist()})
            return
        # Byte differences retain signed-zero distinctions too; every differing
        # coordinate is listed rather than hiding a sample behind a maximum error.
        different = np.any(a.reshape(-1).view(np.uint8).reshape(-1, a.dtype.itemsize)
                           != b.reshape(-1).view(np.uint8).reshape(-1, b.dtype.itemsize), axis=1)
        indices = np.flatnonzero(different)
        left_values, right_values = a.reshape(-1)[indices], b.reshape(-1)[indices]
        tolerance = _tolerance(path) if a.dtype.kind == "f" else {"atol": 0.0, "rtol": 0.0}
        within = bool(np.allclose(a, b, **tolerance)) if a.dtype.kind == "f" else False
        if a.dtype.kind == "f":
            delta = right_values.astype(np.float64) - left_values.astype(np.float64)
            delta_values = _json_value(delta.tolist())
            maximum = _json_value(float(np.max(np.abs(delta)))) if delta.size else 0.0
        else:
            delta_values = [int(right) - int(left) for left, right in zip(left_values, right_values)]
            maximum = max(map(abs, delta_values), default=0)
        detail = {**metadata, "tolerance": tolerance, "within_tolerance": within,
                  "different_elements": int(len(indices)), "flat_indices": indices.tolist(),
                  "original_values": _json_value(left_values.tolist()),
                  "fresh_values": _json_value(right_values.tolist()), "fresh_minus_original": delta_values,
                  "max_absolute_error": maximum}
        report["numerical_differences"].append(detail)
        if not within:
            report["mismatches"].append({"path": name, "kind": "numerical_tolerance",
                                         "difference_record": len(report["numerical_differences"]) - 1})

    def visit(left, right, path):
        reason = _exclusion_reason(path)
        if reason:
            report["excluded_metadata"].append({"path": _path_text(path), "reason": reason,
                                                  "original": _json_value(left), "fresh": _json_value(right)})
            return
        if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
            if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
                mismatch(path, "array_type", type(left).__name__, type(right).__name__)
            else:
                compare_arrays(left, right, path)
            return
        if type(left) is not type(right):
            mismatch(path, "structural_type", type(left).__name__, type(right).__name__)
            return
        if isinstance(left, dict):
            left_keys, right_keys = list(left), list(right)
            if len(left_keys) != len(right_keys) or any(type(a) is not type(b) or a != b for a, b in zip(left_keys, right_keys)):
                mismatch(path, "mapping_keys_or_order", left_keys, right_keys)
            for key in left_keys:
                if key in right and any(type(key) is type(other) and key == other for other in right_keys):
                    visit(left[key], right[key], (*path, key))
            return
        if isinstance(left, (list, tuple)):
            if len(left) != len(right):
                mismatch(path, "sequence_length", len(left), len(right))
            for index, (a, b) in enumerate(zip(left, right)):
                visit(a, b, (*path, index))
            return
        if isinstance(left, (float, np.floating)):
            a, b = np.asarray(left), np.asarray(right)
            compare_arrays(a, b, path)
            return
        if left is None or isinstance(left, (str, int, bool, np.integer, np.bool_)):
            if left != right:
                mismatch(path, "exact_scalar", left, right)
            else:
                report["exact_matches"].append({"path": _path_text(path), "type": type(left).__name__})
            return
        mismatch(path, "unsupported_payload_type", type(left).__name__, type(right).__name__)

    visit(original, fresh, ())
    report["status"] = "FAIL" if report["mismatches"] else "PASS"
    report["bitwise_exact"] = not report["mismatches"] and not report["numerical_differences"]
    report["exact_leaf_count"] = len(report["exact_matches"])
    report["different_numerical_leaf_count"] = len(report["numerical_differences"])
    return report


def _completed_run(directory):
    directory = Path(directory).resolve()
    receipt = json.loads((directory / "execution.json").read_text())
    if (receipt.get("status") != "COMPLETE" or receipt.get("resource_violations")
            or receipt.get("changed_sources") or receipt.get("child_received_learner_payloads") is not False):
        raise ValueError("A completed, unchanged-source, cold native execution is required.")
    return receipt


def _verify_physical_receipts(payload, study_directory):
    """Operational DB hashes may differ between runs, but each must remain valid."""
    database = payload.get("oracle_inventory", {}).get("database")
    if database is None:
        return
    path = Path(database["path"]).resolve()
    expected = (Path(study_directory) / "oracle/oracle.sqlite3").resolve()
    if path != expected or not path.is_file():
        raise ValueError("Oracle SQLite receipt must identify this completed study's actual database.")
    if path.stat().st_size != database["size_bytes"] or digest(path) != database["sha256"]:
        raise ValueError("Physical oracle SQLite bytes changed after the study receipt.")


def _comparison_child(args):
    if os.environ.get("ARENA_SPEX_COMPARISON_CHILD") != "1":
        raise ValueError("Reproduction comparison requires the resource supervisor.")
    study = importlib.import_module(STUDY_MODULE)
    original = study.load_study(args.learner_study)
    fresh = study.load_study(args.fresh_study)
    _verify_physical_receipts(original, args.learner_study)
    _verify_physical_receipts(fresh, args.fresh_study)
    comparison = compare_payloads(original, fresh)
    directory = Path(args.comparison_child_output)
    _write_json(directory / "comparison.json", comparison)
    # The parent reads this bounded summary, not the potentially large difference
    # arrays. All payload loading and full comparison remain inside sampled RSS.
    _write_json(directory / "comparison_summary.json", {
        "status": comparison["status"], "counts": comparison["counts"],
        "bitwise_exact": comparison["bitwise_exact"],
        "numerical_difference_leaves": len(comparison["numerical_differences"]),
        "mismatch_count": len(comparison["mismatches"]),
        "comparison_sha256": digest(directory / "comparison.json"),
    })
    if comparison["status"] != "PASS":
        raise SystemExit(1)


def reproduce(learner_run, output, **fixed_design):
    """A fresh child gets only the fixed design; payload comparison happens afterward."""
    started = time.monotonic()
    learner_run, output = Path(learner_run).resolve(), Path(output).resolve()
    before = _completed_run(learner_run)
    original_receipt_hash = digest(learner_run / "execution.json")
    fresh = run_native(output, **fixed_design)
    record = {"schema_version": 1, "status": "FAIL", "learner_directory": str(learner_run),
              "fresh_directory": str(output), "learner_execution_sha256": original_receipt_hash,
              "fresh_execution_sha256": digest(output / "execution.json"),
              "native_child_received_learner_payloads": False,
              "cached_learner_models_or_values_used_for_native_execution": False,
              "comparison_phase": "Supervised completed evidence comparison after cold native execution; no model fitting.",
              "fresh_execution_status": fresh["status"], "failures": []}
    try:
        if fresh["status"] != "COMPLETE":
            raise ValueError("Fresh source-only execution did not complete; its full evidence is retained.")
        if before["source_inputs"] != fresh["source_inputs"]:
            raise ValueError("Learner and fresh runs used different source/frozen input bytes.")
        if before.get("method_modules") != fresh.get("method_modules"):
            raise ValueError("Learner and fresh runs used different method modules.")
        if digest(learner_run / "execution.json") != original_receipt_hash:
            raise ValueError("Learner receipt changed during the fresh execution.")
        comparison_output = output / "comparison_process"
        command = [sys.executable, str(Path(__file__).resolve()), "--comparison-child-output", str(comparison_output),
                   "--learner-study", str(learner_run / "study"), "--fresh-study", str(output / "study")]
        remaining = ResourceLimits(
            wall_seconds=max(0.0, NATIVE_LIMITS.wall_seconds - (time.monotonic() - started)),
            output_bytes=max(1, NATIVE_LIMITS.output_bytes - sum(_file_inventory(output).values())),
        )
        comparison_execution = supervise(command, comparison_output, limits=remaining,
                                           source_paths=list(fresh["source_inputs"]),
                                           env={"ARENA_SPEX_COMPARISON_CHILD": "1"},
                                           manifest_extra={"execution_kind": "completed_payload_comparison_only",
                                                           "native_model_queries_or_fits": 0})
        record["comparison_execution"] = {
            "path": str(comparison_output / "execution.json"), "sha256": digest(comparison_output / "execution.json"),
            "status": comparison_execution["status"], "returncode": comparison_execution["returncode"],
            "resource_violations": comparison_execution["resource_violations"],
        }
        record["sampled_peak_rss_bytes"] = max(fresh.get("sampled_peak_rss_bytes", 0),
                                                comparison_execution["sampled_peak_rss_bytes"])
        comparison_path = comparison_output / "comparison.json"
        if not comparison_path.is_file():
            raise ValueError("Comparison child did not finish; its failure log and partial output are retained.")
        comparison = json.loads((comparison_output / "comparison_summary.json").read_text())
        record["comparison_path"] = str(comparison_path)
        record["comparison_sha256"] = digest(comparison_path)
        if record["comparison_sha256"] != comparison["comparison_sha256"]:
            raise ValueError("Comparison bytes changed after the supervised summary was written.")
        record["comparison_counts"] = comparison["counts"]
        record["bitwise_exact"] = comparison["bitwise_exact"]
        record["numerical_difference_leaves"] = comparison["numerical_difference_leaves"]
        record["mismatch_count"] = comparison["mismatch_count"]
        record["status"] = comparison["status"] if comparison_execution["status"] == "COMPLETE" else "FAIL"
    except BaseException as error:
        record["failures"].append({"kind": type(error).__name__, "message": str(error),
                                    "traceback": traceback.format_exc()})
    record["elapsed_seconds"] = time.monotonic() - started
    if record["elapsed_seconds"] > NATIVE_LIMITS.wall_seconds:
        record["status"] = "FAIL"
        record["failures"].append({"kind": "wall_seconds", "message": "Source execution plus comparison exceeded the joint wall cap."})
    # The final reproduction receipt is retained evidence and consumes the same
    # joint output budget as the child logs and the complete difference report.
    _write_json(output / "reproduction.json", record)
    if sum(_file_inventory(output).values()) > NATIVE_LIMITS.output_bytes:
        record["status"] = "FAIL"
        record["failures"].append({"kind": "output_bytes", "message": "Comparison evidence exceeded the joint retained-output cap."})
        _write_json(output / "reproduction.json", record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child-output", type=Path)
    parser.add_argument("--comparison-child-output", type=Path)
    parser.add_argument("--learner-study", type=Path)
    parser.add_argument("--fresh-study", type=Path)
    parser.add_argument("--reproduce-from", type=Path)
    parser.add_argument("--freeze-path", type=Path)
    parser.add_argument("--plan-dir", type=Path)
    parser.add_argument("--snapshot-path", type=Path)
    parser.add_argument("--spex-module", default=SPEX_MODULE)
    parser.add_argument("--proxy-module", default=PROXY_MODULE)
    args = parser.parse_args()
    if args.comparison_child_output:
        if args.learner_study is None or args.fresh_study is None:
            parser.error("Comparison requires --learner-study and --fresh-study")
        _comparison_child(args)
        return
    if any(getattr(args, key) is None for key in ("freeze_path", "plan_dir", "snapshot_path")):
        parser.error("Native execution requires --freeze-path, --plan-dir and --snapshot-path")
    if args.child_output:
        _child_run(args)
        return
    if args.output is None:
        parser.error("--output is required")
    kwargs = {name: getattr(args, name) for name in ("freeze_path", "plan_dir", "snapshot_path",
                                                   "spex_module", "proxy_module")}
    record = reproduce(args.reproduce_from, args.output, **kwargs) if args.reproduce_from else run_native(args.output, **kwargs)
    print(json.dumps({key: record[key] for key in ("status",) if key in record}, indent=2))
    if record["status"] not in {"COMPLETE", "PASS"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

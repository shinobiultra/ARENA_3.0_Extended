"""Synthetic approval-chain and resource-amendment tests; no model execution."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from . import native_study as study
from . import run_native as runner

SCOPE = "SYNTHETIC_TEST_NOT_NATIVE"
RELATIVE = "chapter16_shapley_attribution_baselines/exercises/part18_spex/"
ENVELOPE = {"status", "candidate_manifest_path", "candidate_manifest_sha256", "independent_review_path",
            "independent_review_sha256", "approved_envelope_written_at_utc"}
UNITS = "completed, saved method-condition fits; lower bound on attempted work, not an exact attempted fit count"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    return path


def file_record(path):
    return {"path": str(path), "sha256": sha(path), "size_bytes": path.stat().st_size}


def synthetic_approval(record, directory, status):
    """A deliberately synthetic review chain, confined to pytest's temporary tree."""
    candidate = write_json(directory / "candidate.json", {key: value for key, value in record.items() if key not in ENVELOPE})
    review = write_json(directory / "review.json", {"evidence_scope": SCOPE, "status": status,
                                                    "candidate_manifest_sha256": sha(candidate)})
    record.update(status="APPROVED", candidate_manifest_path=str(candidate), candidate_manifest_sha256=sha(candidate),
                  independent_review_path=str(review), independent_review_sha256=sha(review),
                  approved_envelope_written_at_utc=record["created_at_utc"])
    return write_json(directory / "freeze.json", record)


@pytest.fixture
def amendment(tmp_path, monkeypatch):
    root, prior = tmp_path / "repository", tmp_path / "interrupted_synthetic_run"
    root.mkdir()
    original_protocol = "SYNTHETIC_TEST_NOT_NATIVE original scientific protocol.\n"
    current_runner = Path(runner.__file__).read_text()
    old_runner = current_runner.replace("10800", "5400")
    current_study = Path(study.__file__).read_text()
    old_tree = ast.parse(current_study)
    next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == "validate_freeze").body = [ast.Pass()]
    sources = {
        RELATIVE + "run_native.py": old_runner,
        RELATIVE + "native_study.py": ast.unparse(old_tree),
        RELATIVE + "native_protocol.md": original_protocol,
        RELATIVE + "spex_methods.py": "SCIENTIFIC_VALUE = 7\n",
        RELATIVE + "native_query_plan.py": "FROZEN_SEED = 7719\n",
        RELATIVE + "native_oracle.py": "INFERENCE_THREADS = 8\n",
    }
    for name, source in sources.items():
        for path in (root / name, prior / "source_snapshot" / name):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
    environment = write_json(tmp_path / "environment.json", study.environment_snapshot())
    original = {
        "scope": SCOPE, "created_at_utc": "2026-09-08T00:00:00+00:00",
        "native_execution_started": False, "native_queries_before_freeze": 0, "native_fits_before_freeze": 0,
        "source_inputs": {name: sha(root / name) for name in sources},
        "resources": {"wall_seconds": 5400, "rss_bytes": 8 * 1024**3, "output_bytes": 8 * 1024**3,
                      "reference_stage_seconds": 300, "reference_stage_output_bytes": 256 * 1024**2,
                      "torch_threads": 8, "torch_interop_threads": 4, "numerical_threads": 4},
        "main_environment_path": str(environment), "main_environment_sha256": sha(environment),
        "query_plan_directory": str(tmp_path / "frozen_plan"), "query_plan_manifest_sha256": "1" * 64,
        "model_snapshot_path": str(tmp_path / "NO_MODEL_WEIGHTS"), "oracle_identity_sha256": "2" * 64,
        "spex_defaults": {"max_iterations": 100}, "tolerances": {"native_logits": {"atol": 1e-5, "rtol": 1e-6}},
        "publication_authorized": False,
    }
    original_path = synthetic_approval(original, tmp_path / "original", "APPROVE_NATIVE_PROTOCOL")
    (root / RELATIVE / "run_native.py").write_text(current_runner)
    (root / RELATIVE / "native_study.py").write_text(current_study)
    (root / RELATIVE / "native_protocol.md").write_text(
        original_protocol + "\n\n## Resource-only execution amendment: 180-minute joint cap\nSynthetic explanation only.\n")
    db = prior / "study/oracle/oracle.sqlite3"
    db.parent.mkdir(parents=True)
    with sqlite3.connect(db) as connection:
        connection.executescript("CREATE TABLE calls(receipt TEXT); CREATE TABLE batches(status TEXT,identity TEXT);"
                                 "CREATE TABLE physical_rows(id INTEGER); CREATE TABLE cache(id INTEGER);")
        connection.executemany("INSERT INTO calls VALUES (?)", [(json.dumps({"status": status, "stage": stage}),)
                               for status, stage in (("PASS", "invariance"), ("FAIL", "training"))])
        connection.executemany("INSERT INTO batches VALUES (?,?)", [("PASS", '{"rows":3}'), ("FAIL", '{"rows":1}')])
        connection.executemany("INSERT INTO physical_rows VALUES (?)", [(0,), (1,), (2,)])
        connection.execute("INSERT INTO cache VALUES (0)")
    fit = {"prompt_index": 0, "method": "spex", "seed": 0, "budget": 456, "condition": "real",
           "fit_accounting": {"spex_recoveries": 1, "oracle_calls_by_methods": 0},
           "baseline_fit": {"fits": 2}, "fit_seconds": 1.25, "evidence_scope": SCOPE,
           "predictions": {"final": np.zeros(512)}}
    condition = prior / "study/conditions/p00_synthetic"
    fit_hashes = study.save_record(condition, fit)
    execution = {"status": "FAIL", "returncode": -2, "changed_sources": [], "resource_violations": [],
                 "evidence_scope": SCOPE, "finished_at_utc": "2026-09-08T00:02:00+00:00",
                 "fixed_inputs": {"freeze_path": str(original_path), "plan_dir": original["query_plan_directory"],
                                  "snapshot_path": original["model_snapshot_path"]},
                 "joint_limits": {"wall_seconds": 5400},
                 "source_inputs": {str(root / name): {"sha256": value} for name, value in original["source_inputs"].items()}}
    execution["source_inputs"][str(original_path)] = {"sha256": sha(original_path)}
    execution_path = write_json(prior / "execution.json", execution)
    child_log = prior / "child.log"
    child_log.write_text(SCOPE + "\n")
    runtime = write_json(tmp_path / "runtime.json", {"evidence_scope": SCOPE})
    inventory = {
        "status": "INTERRUPTED_RESOURCE_FEASIBILITY", "recorded_at_utc": "2026-09-08T00:03:00+00:00",
        "execution": file_record(execution_path), "original_scientific_freeze": file_record(original_path),
        "oracle_database": file_record(db), "child_log": file_record(child_log), "runtime_basis": file_record(runtime),
        "partial_values_or_fits_will_be_reused_for_new_cold_execution": False, "proposed_joint_wall_seconds": 10800,
        "table_counts": {"calls": 2, "batches": 2, "physical_rows": 3, "cache": 1},
        "batch_status_counts": {"PASS": 1, "FAIL": 1}, "call_status_counts": {"PASS": 1, "FAIL": 1},
        "call_stages": {"invariance": 1, "training": 1}, "attempted_physical_rows_from_started_batches": 4,
        "completed_physical_rows": 3, "completed_cached_rows": 1, "audits_saved": 0, "reference_stage_records": 0,
        "native_fits_before_amendment": 1, "native_fits_before_amendment_units": UNITS,
        "saved_conditions": [{"path": str(condition.relative_to(prior)), "identity": [0, "spex", 0, 456, "real"],
                              "hashes": fit_hashes}],
        "saved_fit_accounting": {"oracle_calls_by_methods": 0, "ensemble_fits": 0, "residual_tree_fits": 0,
                                 "ridge_candidate_fits": 0, "ridge_matrix_decompositions": 0,
                                 "spex_recoveries": 1, "baseline_fits": 2},
        "saved_fit_seconds": 1.25,
    }
    inventory_path = write_json(tmp_path / "interruption.json", inventory)
    record = deepcopy(original)
    record.update(freeze_kind="RESOURCE_ONLY_AMENDMENT", counts_scope="original_scientific_freeze",
                  original_scientific_freeze_path=str(original_path), original_scientific_freeze_sha256=sha(original_path),
                  prior_execution_path=str(execution_path), prior_execution_sha256=sha(execution_path),
                  prior_interruption_inventory_path=str(inventory_path), prior_interruption_inventory_sha256=sha(inventory_path),
                  native_queries_before_amendment=3, native_fits_before_amendment=1,
                  native_fits_before_amendment_units=UNITS, native_execution_started=True,
                  created_at_utc="2026-09-08T00:04:00+00:00")
    record["source_inputs"] = {name: sha(root / name) for name in sources}
    record["resources"]["wall_seconds"] = 10800
    path = synthetic_approval(record, tmp_path / "amendment", "APPROVE_RESOURCE_ONLY_AMENDMENT")
    monkeypatch.setattr(study, "ROOT", root)
    return SimpleNamespace(root=root, prior=prior, sources=sources, original=original, original_path=original_path,
                           record=record, path=path, inventory=inventory, inventory_path=inventory_path,
                           execution=execution, execution_path=execution_path)


def reapprove(rig):
    return synthetic_approval(rig.record, rig.path.parent, "APPROVE_RESOURCE_ONLY_AMENDMENT")


def test_original_scientific_freeze_format_and_explicit_resource_amendment(amendment):
    rig = amendment
    assert study.validate_freeze(rig.path)["native_queries_before_amendment"] == 3
    assert study.validate_freeze(rig.path)["native_fits_before_amendment_units"] == UNITS
    for name, source in rig.sources.items():
        (rig.root / name).write_text(source)
    assert study.validate_freeze(rig.original_path) == rig.original
    print("Synthetic original approval and resource-only amendment validate with distinct historical/current counts.")


@pytest.mark.parametrize("mutation", ["tolerance", "query_plan", "grid", "oracle_identity", "reference_cap", "rss_cap",
                                      "threads", "wall_cap", "zero_queries", "zero_fits", "count_units", "count_scope",
                                      "execution_started", "creation_time", "extra_field"])
def test_even_reapproved_amendment_cannot_change_science_or_hide_prior_work(amendment, mutation):
    rig = amendment
    actions = {
        "tolerance": lambda: rig.record["tolerances"]["native_logits"].update(atol=1e-3),
        "query_plan": lambda: rig.record.update(query_plan_manifest_sha256="3" * 64),
        "grid": lambda: rig.record["spex_defaults"].update(max_iterations=99),
        "oracle_identity": lambda: rig.record.update(oracle_identity_sha256="3" * 64),
        "reference_cap": lambda: rig.record["resources"].update(reference_stage_seconds=600),
        "rss_cap": lambda: rig.record["resources"].update(rss_bytes=16 * 1024**3),
        "threads": lambda: rig.record["resources"].update(torch_threads=16),
        "wall_cap": lambda: rig.record["resources"].update(wall_seconds=21600),
        "zero_queries": lambda: rig.record.update(native_queries_before_amendment=0),
        "zero_fits": lambda: rig.record.update(native_fits_before_amendment=0),
        "count_units": lambda: rig.record.update(native_fits_before_amendment_units="exact attempted fits"),
        "count_scope": lambda: rig.record.update(counts_scope="new_envelope"),
        "execution_started": lambda: rig.record.update(native_execution_started=False),
        "creation_time": lambda: rig.record.update(created_at_utc="2026-09-08T00:01:00+00:00"),
        "extra_field": lambda: rig.record.update(unapproved_scientific_override=True),
    }
    actions[mutation]()
    with pytest.raises(ValueError):
        study.validate_freeze(reapprove(rig))


@pytest.mark.parametrize("name,old,new", [
    ("spex_methods.py", "SCIENTIFIC_VALUE = 7", "SCIENTIFIC_VALUE = 8"),
    ("native_query_plan.py", "FROZEN_SEED = 7719", "FROZEN_SEED = 7720"),
    ("native_oracle.py", "INFERENCE_THREADS = 8", "INFERENCE_THREADS = 16"),
    ("run_native.py", "REFERENCE_WALL_SECONDS = 300.0", "REFERENCE_WALL_SECONDS = 301.0"),
    ("native_study.py", "rcond=None", "rcond=1e-12"),
    ("native_protocol.md", "original scientific protocol", "changed scientific protocol"),
])
def test_rehashed_source_change_outside_narrow_amendment_is_rejected(amendment, name, old, new):
    rig = amendment
    path = rig.root / RELATIVE / name
    text = path.read_text()
    assert old in text
    path.write_text(text.replace(old, new, 1))
    rig.record["source_inputs"][RELATIVE + name] = sha(path)
    with pytest.raises(ValueError, match="Only the supervisor|Supervisor AST|Study AST|exact prefix"):
        study.validate_freeze(reapprove(rig))


@pytest.mark.parametrize("target", ["original_freeze", "original_review", "original_candidate", "inventory",
                                    "database", "fit_array", "new_review", "unbound_new_count"])
def test_unbound_or_changed_historical_evidence_is_rejected(amendment, target):
    rig = amendment
    files = {
        "original_freeze": rig.original_path,
        "original_review": Path(rig.original["independent_review_path"]),
        "original_candidate": Path(rig.original["candidate_manifest_path"]),
        "inventory": rig.inventory_path,
        "database": Path(rig.inventory["oracle_database"]["path"]),
        "fit_array": rig.prior / rig.inventory["saved_conditions"][0]["path"] / "arrays.npz",
        "new_review": Path(rig.record["independent_review_path"]),
    }
    if target == "unbound_new_count":
        changed = deepcopy(rig.record)
        changed["native_queries_before_amendment"] += 1
        write_json(rig.path, changed)
    else:
        with files[target].open("ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError):
        study.validate_freeze(rig.path)


@pytest.mark.parametrize("mutation", ["queries", "fits", "attempts", "audit", "reference", "reused_values", "execution_status"])
def test_locally_rehashed_interruption_inventory_must_match_physical_receipts(amendment, mutation):
    rig = amendment
    if mutation == "execution_status":
        rig.execution["returncode"] = 0
        write_json(rig.execution_path, rig.execution)
        rig.inventory["execution"] = file_record(rig.execution_path)
        rig.record["prior_execution_sha256"] = sha(rig.execution_path)
    else:
        field, value = {
            "queries": ("completed_physical_rows", 0), "fits": ("native_fits_before_amendment", 0),
            "attempts": ("attempted_physical_rows_from_started_batches", 99), "audit": ("audits_saved", 1),
            "reference": ("reference_stage_records", 1),
            "reused_values": ("partial_values_or_fits_will_be_reused_for_new_cold_execution", True),
        }[mutation]
        rig.inventory[field] = value
    write_json(rig.inventory_path, rig.inventory)
    rig.record["prior_interruption_inventory_sha256"] = sha(rig.inventory_path)
    with pytest.raises(ValueError):
        study.validate_freeze(reapprove(rig))


def test_joint_cap_is_10800_and_reference_limits_threads_tolerances_are_unchanged():
    assert runner.ResourceLimits().wall_seconds == runner.NATIVE_LIMITS.wall_seconds == 10800
    assert runner.NATIVE_LIMITS.rss_bytes == runner.NATIVE_LIMITS.output_bytes == 8 * 1024**3
    assert runner.REFERENCE_WALL_SECONDS == 300
    assert runner.REFERENCE_RSS_BYTES == 8 * 1024**3 and runner.REFERENCE_OUTPUT_BYTES == 256 * 1024**2
    assert runner.NATIVE_TORCH_THREADS == 8 and runner.NATIVE_INTEROP_THREADS == 4
    assert runner.LOGIT_TOLERANCE == {"atol": 1e-5, "rtol": 1e-6}
    assert runner.ESTIMATOR_TOLERANCE == {"atol": 1e-8, "rtol": 1e-7}
    print("Only the complete native/reproduction wall cap is 10,800 seconds; all other resource and numerical limits stay fixed.")

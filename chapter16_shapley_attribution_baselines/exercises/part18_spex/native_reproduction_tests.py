"""Synthetic-only resource and comparison tests. No model, oracle or native fit."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import psutil
import pytest

from chapter16_shapley_attribution_baselines.exercises.part18_spex import run_native as runner


def _command(code):
    return [sys.executable, "-c", code]


def _limits(**changes):
    values = dict(wall_seconds=4.0, rss_bytes=128 * 1024**2, output_bytes=4 * 1024**2, sample_seconds=0.01)
    values.update(changes)
    return runner.ResourceLimits(**values)


def test_dummy_child_cpu_environment_exit_and_retained_inventory(tmp_path):
    output = tmp_path / "run"
    target = output / "environment.json"
    code = ("import json,os,pathlib; "
            f"pathlib.Path({str(target)!r}).write_text(json.dumps(dict(os.environ)))")
    record = runner.supervise(_command(code), output, limits=_limits(),
                              env={"CUDA_VISIBLE_DEVICES": "0", "OMP_NUM_THREADS": "99"})
    actual = json.loads(target.read_text())
    assert all(actual[key] == value for key, value in runner.CPU_ENVIRONMENT.items())
    assert record["status"] == "COMPLETE" and record["returncode"] == 0
    assert record["rss_samples"] > 0 and record["sampled_peak_rss_bytes"] > 0
    assert json.loads((output / "execution.json").read_text()) == record
    assert record["new_output_bytes"] == sum(p.stat().st_size for p in output.rglob("*") if p.is_file())
    assert record["output_files"]["execution.json"] == (output / "execution.json").stat().st_size


def test_dummy_child_failure_keeps_actual_exit_and_log(tmp_path):
    output = tmp_path / "failed"
    record = runner.supervise(_command("import sys; print('actual source failure',flush=True); sys.exit(7)"),
                              output, limits=_limits())
    assert record["status"] == "FAIL" and record["returncode"] == 7
    assert "actual source failure" in (output / "child.log").read_text()
    assert record["resource_violations"] == []


def test_wall_timeout_stops_child_and_descendant_and_retains_signal_exit(tmp_path):
    output = tmp_path / "timeout"
    pid_path = output / "descendant.pid"
    inner = "import time; time.sleep(10)"
    code = ("import pathlib,subprocess,sys,time; "
            f"p=subprocess.Popen([sys.executable,'-c',{inner!r}]); "
            f"pathlib.Path({str(pid_path)!r}).write_text(str(p.pid)); time.sleep(10)")
    record = runner.supervise(_command(code), output, limits=_limits(wall_seconds=0.25))
    assert record["status"] == "FAIL" and "wall_seconds" in record["resource_violations"]
    assert record["returncode"] < 0
    assert len(record["observed_process_ids"]) >= 2
    pid = int(pid_path.read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


def test_process_tree_rss_cap_counts_descendant_allocation(tmp_path):
    output = tmp_path / "memory"
    inner = "import time; x=bytearray(32*1024*1024); time.sleep(5)"
    code = ("import subprocess,sys,time; "
            f"p=subprocess.Popen([sys.executable,'-c',{inner!r}]); time.sleep(5)")
    cap = 36 * 1024**2
    record = runner.supervise(_command(code), output, limits=_limits(rss_bytes=cap))
    assert record["status"] == "FAIL" and "rss_bytes" in record["resource_violations"]
    assert record["sampled_peak_rss_bytes"] > cap
    assert len(record["observed_process_ids"]) >= 2


def test_output_cap_retains_oversized_partial_output(tmp_path):
    output = tmp_path / "storage"
    target = output / "partial.dat"
    code = f"import pathlib,time; pathlib.Path({str(target)!r}).write_bytes(b'x'*200000); time.sleep(2)"
    record = runner.supervise(_command(code), output, limits=_limits(output_bytes=50000))
    assert record["status"] == "FAIL" and "output_bytes" in record["resource_violations"]
    assert target.stat().st_size == 200000
    assert (output / "execution.json").is_file()


def test_source_drift_preserves_before_snapshot_and_actual_after_hash(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("VALUE = 1\n")
    old_hash = runner.digest(source)
    output = tmp_path / "drift"
    code = f"import pathlib; pathlib.Path({str(source)!r}).write_text('VALUE = 2\\n')"
    record = runner.supervise(_command(code), output, limits=_limits(), source_paths=[source])
    assert record["status"] == "FAIL" and record["returncode"] == 0
    assert record["changed_sources"] == [str(source)]
    assert record["source_inputs"][str(source)]["sha256"] == old_hash
    assert record["source_inputs_after"][str(source)]["sha256"] == runner.digest(source)
    snapshots = list((output / "source_snapshot").rglob("*.py"))
    assert len(snapshots) == 1 and runner.digest(snapshots[0]) == old_hash


def test_missing_source_rejects_before_child_but_keeps_receipt(tmp_path):
    output = tmp_path / "missing"
    record = runner.supervise(_command("raise AssertionError('must not launch')"), output,
                              limits=_limits(), source_paths=[tmp_path / "missing.py"])
    assert record["status"] == "FAIL" and record["returncode"] is None
    assert record["failures"] and (output / "execution.json").is_file()


def test_existing_output_is_never_reused_or_deleted(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    sentinel = output / "earlier_failure.json"
    sentinel.write_text('{"status":"FAIL"}')
    with pytest.raises(FileExistsError):
        runner.supervise(_command("pass"), output, limits=_limits())
    assert sentinel.read_text() == '{"status":"FAIL"}'


def _reference_environment(monkeypatch, directory, *, remaining=10):
    directory.mkdir(exist_ok=True)
    monkeypatch.setenv("ARENA_SPEX_JOINT_DEADLINE_UNIX", str(time.time() + remaining))
    monkeypatch.setenv("ARENA_SPEX_REFERENCE_STAGE_IDS", json.dumps(list(runner.REFERENCE_STAGE_IDS)))
    monkeypatch.setenv("ARENA_SPEX_REFERENCE_REGISTRY", str(directory / "reference_stages.jsonl"))


def test_reference_failure_is_explicit_source_outcome_without_retry(tmp_path, monkeypatch):
    _reference_environment(monkeypatch, tmp_path)
    record = runner.run_reference_stage(_command("import sys;sys.exit(9)"), tmp_path / "stage", stage_id="author_spex_b3")
    assert record["status"] == "FAIL" and record["returncode"] == 9 and record["is_source_outcome"]
    assert record["resource_violations"] == []
    assert record["limits"]["rss_bytes"] == 8 * 1024**3
    assert record["limits"]["output_bytes"] == 268435456
    with pytest.raises(ValueError, match="cannot be retried"):
        runner.run_reference_stage(_command("pass"), tmp_path / "retry", stage_id="author_spex_b3")
    with pytest.raises(ValueError, match="nine frozen IDs"):
        runner.run_reference_stage(_command("pass"), tmp_path / "new", stage_id="undeclared_b3")


def test_reference_stage_has_own_cap_inside_joint_deadline(tmp_path, monkeypatch):
    _reference_environment(monkeypatch, tmp_path, remaining=10)
    monkeypatch.setattr(runner, "REFERENCE_WALL_SECONDS", 0.1)
    record = runner.run_reference_stage(_command("import time;time.sleep(3)"), tmp_path / "stage", stage_id="author_proxy_b3")
    assert record["status"] == "FAIL" and record["is_source_outcome"]
    assert "wall_seconds" in record["resource_violations"]
    assert record["limits"]["wall_seconds"] == 0.1


def test_expired_joint_deadline_starts_no_reference_child(tmp_path, monkeypatch):
    _reference_environment(monkeypatch, tmp_path, remaining=-1)
    record = runner.run_reference_stage(_command("raise AssertionError('must not launch')"),
                                        tmp_path / "stage", stage_id="maintained_proxy_b3")
    assert record["status"] == "FAIL" and record["returncode"] is None
    assert "wall_seconds" in record["resource_violations"]


def test_reference_stage_output_cap_retains_partial_source_outcome(tmp_path, monkeypatch):
    _reference_environment(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "REFERENCE_OUTPUT_BYTES", 50000)
    output = tmp_path / "stage"
    partial = output / "partial.bin"
    command = _command(f"from pathlib import Path; Path({str(partial)!r}).write_bytes(b'x' * 200000)")
    record = runner.run_reference_stage(command, output, stage_id="maintained_proxy_b3")
    assert record["status"] == "FAIL" and record["is_source_outcome"]
    assert record["limits"]["output_bytes"] == 50000
    assert "output_bytes" in record["resource_violations"]
    assert partial.stat().st_size == 200000


def _payload():
    return {
        "status": "COMPLETE", "counts": {"physical_rows": 3, "fits": 2},
        "oracle_values": {"mask_ids": np.array([1, 3, 7], dtype=np.uint64),
                          "logits": np.array([[0.5, -0.5], [1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                          "game": np.array(["primary", "primary", "counterfactual"])},
        "results": [{"spectrum": {(0, 1): -0.25}, "predictions": np.array([0.1, 0.2, 0.3]),
                     "selected_parameters": {"learning_rate": 0.1, "max_depth": 2},
                     "selected_alpha": 0.01, "fit_seconds": 1.0}],
        "references": [{"stage_id": "author_spex_b3", "status": "FAIL", "operational": {"elapsed_seconds": 1.0}}],
    }


def test_identical_synthetic_payload_records_exact_counts_and_matches():
    original = _payload()
    result = runner.compare_payloads(original, deepcopy(original))
    assert result["status"] == "PASS" and result["bitwise_exact"]
    assert result["counts"]["original"] == result["counts"]["fresh"]
    assert result["exact_leaf_count"] > 0 and result["numerical_differences"] == []


def test_toleranced_logits_and_predictions_keep_every_difference_separate():
    original, fresh = _payload(), _payload()
    fresh["oracle_values"]["logits"] += np.float32(2e-6)
    fresh["results"][0]["predictions"] += 1e-9
    result = runner.compare_payloads(original, fresh)
    assert result["status"] == "PASS" and not result["bitwise_exact"]
    differences = result["numerical_differences"]
    assert [row["different_elements"] for row in differences] == [6, 3]
    assert [row["flat_indices"] for row in differences] == [list(range(6)), list(range(3))]
    assert all(row["within_tolerance"] for row in differences)
    assert differences[0]["tolerance"] == runner.LOGIT_TOLERANCE
    assert differences[1]["tolerance"] == runner.ESTIMATOR_TOLERANCE


@pytest.mark.parametrize("mutation", ["counts", "ids", "support", "selection", "alpha", "shape", "dtype",
                                     "keys", "order", "container", "reference_outcome", "nonfinite", "large_prediction"])
def test_reproduction_rejects_changed_structure_choices_or_out_of_tolerance_values(mutation):
    original, fresh = _payload(), _payload()
    if mutation == "counts":
        fresh["counts"]["fits"] += 1
    elif mutation == "ids":
        fresh["oracle_values"]["mask_ids"][0] += 1
    elif mutation == "support":
        fresh["results"][0]["spectrum"] = {(0, 2): -0.25}
    elif mutation == "selection":
        fresh["results"][0]["selected_parameters"]["learning_rate"] += 1e-12
    elif mutation == "alpha":
        fresh["results"][0]["selected_alpha"] += 1e-12
    elif mutation == "shape":
        fresh["results"][0]["predictions"] = np.array([[0.1, 0.2, 0.3]])
    elif mutation == "dtype":
        fresh["results"][0]["predictions"] = fresh["results"][0]["predictions"].astype(np.float32)
    elif mutation == "keys":
        fresh["results"][0]["unexpected"] = 1
    elif mutation == "order":
        fresh["counts"] = dict(reversed(list(fresh["counts"].items())))
    elif mutation == "container":
        fresh["results"] = tuple(fresh["results"])
    elif mutation == "reference_outcome":
        fresh["references"][0]["status"] = "PASS"
    elif mutation == "nonfinite":
        fresh["results"][0]["predictions"][0] = np.nan
    else:
        fresh["results"][0]["predictions"][0] += 1e-3
    result = runner.compare_payloads(original, fresh)
    assert result["status"] == "FAIL" and result["mismatches"]
    json.dumps(result, allow_nan=False)


def test_operational_differences_are_retained_and_do_not_hide_scientific_status():
    original, fresh = _payload(), _payload()
    fresh["results"][0]["fit_seconds"] = 20.0
    fresh["references"][0]["operational"] = {"elapsed_seconds": 12.0, "pids": [10, 11], "result_path": "/fresh/result.json"}
    result = runner.compare_payloads(original, fresh)
    assert result["status"] == "PASS" and result["bitwise_exact"]
    assert result["full_payload_counts"] != result["counts"]
    operational = next(row for row in result["excluded_metadata"] if row["path"].endswith("operational"))
    assert operational["fresh"] == fresh["references"][0]["operational"]


def test_unknown_runtime_looking_field_is_not_silently_excluded():
    original, fresh = _payload(), _payload()
    original["results"][0]["unknown_runtime"] = 1.0
    fresh["results"][0]["unknown_runtime"] = 2.0
    assert runner.compare_payloads(original, fresh)["status"] == "FAIL"


def test_string_arrays_and_large_integer_ids_remain_exact():
    original, fresh = _payload(), _payload()
    fresh["oracle_values"]["game"][0] = "other"
    assert runner.compare_payloads(original, fresh)["status"] == "FAIL"
    original = {"ids": np.array([2**63 + 10], dtype=np.uint64)}
    fresh = {"ids": np.array([2**63 + 11], dtype=np.uint64)}
    result = runner.compare_payloads(original, fresh)
    assert result["status"] == "FAIL"
    assert result["numerical_differences"][0]["fresh_minus_original"] == [1]


def test_physical_hash_exclusions_keep_full_metadata_and_decoded_differences():
    original, fresh = _payload(), _payload()
    for payload, label in [(original, "original"), (fresh, "fresh")]:
        payload["prediction_boundary_hashes"] = {"record.json": label}
        payload["oracle_value_hashes"] = {"arrays.npz": label}
        payload["conditions"] = [{"path": "conditions/0", "hashes": {"record.json": label}}]
        payload["audit_records"] = [{"path": "audit/0", "hashes": {"arrays.npz": label}}]
        payload["oracle_inventory"] = {"physical_rows": 3, "database": {"path": "/" + label, "sha256": label, "size_bytes": 100}}
        payload["references"][0]["training_values_float64_sha256"] = label
    fresh["oracle_values"]["logits"] += np.float32(2e-6)
    result = runner.compare_payloads(original, fresh)
    assert result["status"] == "PASS" and not result["bitwise_exact"]
    assert len(result["excluded_metadata"]) == 8
    fresh["oracle_inventory"]["physical_rows"] += 1
    assert runner.compare_payloads(original, fresh)["status"] == "FAIL"


def test_each_physical_sqlite_receipt_is_verified_before_ignoring_cross_run_hash(tmp_path):
    database = tmp_path / "study/oracle/oracle.sqlite3"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"synthetic retained database bytes")
    record = {"oracle_inventory": {"database": {"path": str(database), "sha256": runner.digest(database),
                                                 "size_bytes": database.stat().st_size}}}
    runner._verify_physical_receipts(record, tmp_path / "study")
    database.write_bytes(b"changed bytes")
    with pytest.raises(ValueError, match="SQLite bytes changed"):
        runner._verify_physical_receipts(record, tmp_path / "study")


def test_rejected_native_preflight_has_failure_receipt_without_launch(tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text('{"reference_stage_ids":[]}')
    output = tmp_path / "rejected"
    record = runner.run_native(output, freeze_path=freeze, plan_dir=tmp_path / "plan", snapshot_path=tmp_path / "model")
    assert record["status"] == "FAIL" and record["returncode"] is None
    assert record["execution_kind"] == "rejected_native_preflight"
    assert record["observed_process_ids"] == []
    assert (output / "execution.json").is_file()


def test_cli_rejected_preflight_executes_runtime_and_exits_nonzero(tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text('{"reference_stage_ids":[]}')
    output = tmp_path / "cli"
    child = subprocess.run([sys.executable, str(Path(runner.__file__).resolve()), "--output", str(output),
                            "--freeze-path", str(freeze), "--plan-dir", str(tmp_path / "plan"),
                            "--snapshot-path", str(tmp_path / "model")], capture_output=True, text=True, timeout=10)
    assert child.returncode == 1
    record = json.loads((output / "execution.json").read_text())
    assert record["status"] == "FAIL" and record["returncode"] is None
    assert "native" not in record.get("observed_process_ids", [])


def test_native_source_inventory_excludes_unexecuted_lesson_builders_and_tests():
    paths = runner._native_source_paths()
    names = {path.name for path in paths}
    assert {"native_data.py", "native_query_plan.py", "native_oracle.py", "native_study.py", "run_native.py",
            "spex_methods.py", "spex_coding.py", "proxy_methods.py", "reference_bridge.py", "native_protocol.md",
            "pyproject.toml", "uv.lock"} <= names
    assert not any(name.endswith("_tests.py") or "lesson" in name or "build" in name or "plot" in name for name in names)
    assert len(paths) == 14


def test_missing_frozen_actual_dependency_rejects_before_module_import(tmp_path, monkeypatch):
    actual = tmp_path / "native_core.py"
    actual.write_text("VALUE = 1\n")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "_native_source_paths", lambda: {actual})
    with pytest.raises(ValueError, match="missing from the freeze"):
        runner._execution_sources({"source_inputs": {}}, "unimported.spex", "unimported.proxy", None)


def test_native_child_guard_and_prospective_eight_thread_torch_configuration(tmp_path, monkeypatch):
    args = SimpleNamespace(child_output=tmp_path, freeze_path=tmp_path / "freeze.json",
                           plan_dir=tmp_path / "plan", snapshot_path=tmp_path / "model",
                           spex_module=runner.SPEX_MODULE, proxy_module=runner.PROXY_MODULE)
    monkeypatch.delenv("ARENA_SPEX_SUPERVISED_CHILD", raising=False)
    with pytest.raises(ValueError, match="resource supervisor"):
        runner._child_run(args)
    monkeypatch.setenv("ARENA_SPEX_SUPERVISED_CHILD", "1")
    for name, value in runner.CPU_ENVIRONMENT.items():
        monkeypatch.setenv(name, value)
    calls = []
    fake_torch = SimpleNamespace(set_num_threads=lambda value: calls.append(("torch_threads", value)),
                                 set_num_interop_threads=lambda value: calls.append(("interop_threads", value)))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    spex, proxy = SimpleNamespace(name="spex"), SimpleNamespace(name="proxy")
    study = SimpleNamespace(run_study=lambda output, **kwargs: calls.append(("study", output, kwargs)))
    monkeypatch.setattr(runner.importlib, "import_module",
                        lambda name: {runner.STUDY_MODULE: study, runner.SPEX_MODULE: spex, runner.PROXY_MODULE: proxy}[name])
    runner._child_run(args)
    assert calls[:2] == [("torch_threads", 8), ("interop_threads", 4)]
    assert calls[2] == ("study", tmp_path / "study", {
        "spex_methods": spex, "proxy_methods": proxy, "freeze_path": args.freeze_path,
        "plan_dir": args.plan_dir, "snapshot_path": args.snapshot_path,
    })
    assert os.environ["OMP_NUM_THREADS"] == os.environ["OPENBLAS_NUM_THREADS"] == "4"


@pytest.mark.parametrize("receipt_exceeds_cap", [False, True])
def test_source_only_reproduction_launches_before_loading_payloads(tmp_path, monkeypatch, receipt_exceeds_cap):
    learner, fresh = tmp_path / "learner", tmp_path / "fresh"
    learner.mkdir()
    receipt = {"status": "COMPLETE", "resource_violations": [], "changed_sources": [],
               "child_received_learner_payloads": False, "source_inputs": {"source.py": {"sha256": "a" * 64}},
               "method_modules": {"spex": "frozen.spex", "proxy": "frozen.proxy"}}
    (learner / "execution.json").write_text(json.dumps(receipt))
    (learner / "fitted.pt").write_bytes(b"This fitted learner payload must never reach the child")
    events = []
    fixed = {"freeze_path": tmp_path / "freeze.json", "plan_dir": tmp_path / "plan", "snapshot_path": tmp_path / "model"}

    def fake_run(output, **kwargs):
        assert kwargs == fixed
        assert all(str(learner) not in str(value) for value in kwargs.values())
        events.append("cold_child")
        output.mkdir()
        (output / "execution.json").write_text(json.dumps(receipt))
        return deepcopy(receipt)

    def fake_load(path):
        assert events[0] == "cold_child"
        events.append("load:" + str(path))
        return _payload()

    def fake_supervise(command, output, **kwargs):
        assert events == ["cold_child"]
        assert "--comparison-child-output" in command
        assert str(learner / "study") in command and str(fresh / "study") in command
        assert kwargs["limits"].wall_seconds <= runner.NATIVE_LIMITS.wall_seconds
        assert kwargs["limits"].output_bytes < runner.NATIVE_LIMITS.output_bytes
        output.mkdir()
        with monkeypatch.context() as scoped:
            scoped.setenv("ARENA_SPEX_COMPARISON_CHILD", "1")
            runner._comparison_child(SimpleNamespace(comparison_child_output=output,
                                                       learner_study=learner / "study", fresh_study=fresh / "study"))
        result = {"status": "COMPLETE", "returncode": 0, "resource_violations": [], "sampled_peak_rss_bytes": 100}
        (output / "execution.json").write_text(json.dumps(result))
        if receipt_exceeds_cap:
            # Existing completed evidence fits; only the final reproduction
            # receipt crosses the cap. It must still be retained and fail.
            cap = sum(runner._file_inventory(fresh).values()) + 1
            monkeypatch.setattr(runner, "NATIVE_LIMITS", runner.ResourceLimits(output_bytes=cap))
        return result

    monkeypatch.setattr(runner, "run_native", fake_run)
    monkeypatch.setattr(runner, "supervise", fake_supervise)
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: SimpleNamespace(load_study=fake_load))
    result = runner.reproduce(learner, fresh, **fixed)
    assert result["status"] == ("FAIL" if receipt_exceeds_cap else "PASS")
    assert result["bitwise_exact"]
    if receipt_exceeds_cap:
        assert any(item["kind"] == "output_bytes" for item in result["failures"])
        assert json.loads((fresh / "reproduction.json").read_text())["status"] == "FAIL"
    assert events == ["cold_child", "load:" + str(learner / "study"), "load:" + str(fresh / "study")]
    assert result["cached_learner_models_or_values_used_for_native_execution"] is False
    assert result["native_child_received_learner_payloads"] is False


def test_failed_source_only_child_retains_failure_without_loading_learner(tmp_path, monkeypatch):
    learner = tmp_path / "learner"
    learner.mkdir()
    receipt = {"status": "COMPLETE", "resource_violations": [], "changed_sources": [],
               "child_received_learner_payloads": False, "source_inputs": {}}
    (learner / "execution.json").write_text(json.dumps(receipt))

    def fake_run(output, **kwargs):
        output.mkdir()
        failed = dict(receipt, status="FAIL", resource_violations=["wall_seconds"])
        (output / "execution.json").write_text(json.dumps(failed))
        return failed

    monkeypatch.setattr(runner, "run_native", fake_run)
    monkeypatch.setattr(runner.importlib, "import_module", lambda name: pytest.fail("No payload import after failed child"))
    result = runner.reproduce(learner, tmp_path / "fresh", freeze_path="fixed", plan_dir="fixed", snapshot_path="fixed")
    assert result["status"] == "FAIL" and result["failures"]
    assert (tmp_path / "fresh/reproduction.json").is_file()

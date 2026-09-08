"""Synthetic portable-data contracts; no actual model scores or native fits."""
import copy
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.evidence_bundles import seal_json_data
from . import native_data, native_learner as learner, native_query_plan, native_study, spex_methods


@pytest.fixture(scope="module")
def synthetic_fixture():
    # Only the prospective masks are read. No completed native data is accessed.
    freeze = json.loads((learner.HERE / "native_freeze.json").read_text())
    plan, arrays = native_query_plan.load_plan(freeze["query_plan_directory"])
    union = arrays["training_union_ids"]
    training = np.stack([np.sin(union.astype(float) * 1e-6) + index / 7 for index in range(32)])
    prompts = [{**row, "feature_names": row["words"]} for row in native_data.prompt_records()]
    audit = [{"prompt_index": index, "audit_values": np.linspace(-1, 1, 512) + index / 7,
              "unmasked": np.array([1.0]), "counterfactual": np.array([-1.0])} for index in range(32)]
    rows = []
    for index in range(32):
        for condition in plan["conditions"]:
            for label in ("real", "shuffled"):
                actual = audit[index]["audit_values"]
                stages = ("final",) if condition["method"] == "spex" else ("proxy", "extracted", "truncated", "refined")
                row = {"prompt_index": index, "prompt_id": prompts[index]["id"], "condition": label,
                       **{key: condition[key] for key in ("method", "seed", "budget")},
                       "query_receipt": {"raw_requested": condition["budget"],
                                         "unique_model_masks": condition["unique_count"]},
                       "audit_values": actual, "predictions": {stage: .5 * actual for stage in stages},
                       "baseline_predictions": {"constant": np.zeros(512), "additive": .25 * actual},
                       "spectrum": {(): .25, (0, 2): -.75}}
                rows.append(learner._compact_result(row))
    encoded = {name: {"dtype": str(value.dtype), "shape": list(value.shape), "values": value.tolist()}
               for name, value in arrays.items() if not name.endswith(("_masks", "_matrices", "_delays"))}
    return learner._plain({
        "schema": learner.SCHEMA, "status": "COMPLETE_NATIVE_SCORES", "prompts": prompts,
        "query_plan": plan, "plan_arrays": encoded, "training_values": training, "audit": audit,
        "results": rows, "references": [
            {"stage_id": f"{variant}_b{bits}", "status": "SOURCE_FAILURE", "metrics": {},
             "source_exception_type_message": "Synthetic contract fixture; no reference execution"}
            for bits in (3, 4, 5) for variant in ("author_spex", "author_proxy", "maintained_proxy")],
        "source_inputs": freeze["source_inputs"],
        "inventory": {"synthetic_test_only": True},
        "binding": {"native_execution_sha256": "0" * 64, "fresh_reproduction_sha256": "1" * 64,
                    "query_plan_directory": "/nonexistent/original/cache", "model_snapshot_path": "/missing/model"},
    })


def _seal_fixture(directory, fixture):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "native_query_fixture.json.gz"
    source = directory / "synthetic_source.json"
    source.write_text('{"scope":"synthetic contract test, zero native observations"}\n')
    fixture = copy.deepcopy(fixture)
    fixture["binding"]["native_execution_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    path.write_bytes(gzip.compress(json.dumps(fixture, separators=(",", ":"), allow_nan=False).encode(), mtime=0))
    seal_json_data(path, provenance={
        "kind": "native_execution_packaging", "measurement_claim": "Synthetic schema test only; no native evidence.",
        "source_execution": {"absolute_path": str(source.resolve()),
                             "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                             "size_bytes": source.stat().st_size}})
    return path, source


def test_portable_fixture_roundtrip_and_relocation(synthetic_fixture, tmp_path):
    original, source = _seal_fixture(tmp_path / "old_cache", synthetic_fixture)
    relocated = tmp_path / "empty_checkout/native_query_fixture.json.gz"
    relocated.parent.mkdir()
    relocated.write_bytes(original.read_bytes())
    Path(str(relocated) + ".data-contract.json").write_bytes(Path(str(original) + ".data-contract.json").read_bytes())
    # Loading only needs the portable bytes and contract, never original provenance paths.
    original.unlink(); source.unlink(); Path(str(original) + ".data-contract.json").unlink()
    evidence = learner.load_native_evidence(relocated)
    np.testing.assert_array_equal(evidence["training_values"], np.asarray(synthetic_fixture["training_values"]))
    assert len(evidence["payload"]["results"]) == 1152
    assert evidence["payload"]["results"][0]["spectrum"][(0, 2)] == -.75
    assert evidence["evidence_kind"] == "portable_exact_query_scores"
    assert not original.exists()


def test_public_loader_rejects_changed_compressed_bytes(synthetic_fixture, tmp_path):
    path, _ = _seal_fixture(tmp_path, synthetic_fixture)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError):
        learner.load_native_evidence(path)


def test_public_loader_rejects_wrong_binding(synthetic_fixture, tmp_path):
    path, _ = _seal_fixture(tmp_path, synthetic_fixture)
    with pytest.raises(ValueError, match="explicit binding"):
        learner.load_native_evidence(path, binding={"fixture_sha256": "f" * 64})


@pytest.mark.parametrize("mutation", ["missing_case", "duplicate_case", "changed_mask", "missing_score",
                                      "nonfinite_score", "missing_reference", "changed_comparison_vector"])
def test_semantic_fixture_mutations_fail(synthetic_fixture, mutation):
    fixture = copy.deepcopy(synthetic_fixture)
    if mutation == "missing_case":
        fixture["results"].pop()
    elif mutation == "duplicate_case":
        fixture["results"][-1] = fixture["results"][0]
    elif mutation == "changed_mask":
        fixture["plan_arrays"]["spex_s0_k456_ids"]["values"][0] ^= 1
    elif mutation == "missing_score":
        fixture["training_values"][0].pop()
    elif mutation == "nonfinite_score":
        fixture["training_values"][0][0] = float("nan")
    elif mutation == "missing_reference":
        fixture["references"].pop()
    else:
        fixture["results"][0]["replay_reference_predictions"]["final"][0] += .01
    with pytest.raises(ValueError):
        learner._decode_fixture(fixture)


def test_ordered_values_preserve_duplicates_and_exclude_audit(synthetic_fixture):
    evidence = learner._decode_fixture(copy.deepcopy(synthetic_fixture))
    condition = learner._condition(evidence["plan"], "spex", 0, 456)
    labels = learner.training_values_for(evidence, condition, 7)
    ids = evidence["arrays"][condition["key"] + "_ids"]
    np.testing.assert_array_equal(labels, np.sin(ids.astype(float) * 1e-6) + 1)
    first = evidence["arrays"][condition["key"] + "_unique_first"]
    inverse = evidence["arrays"][condition["key"] + "_inverse"]
    assert len(first) < len(labels)
    np.testing.assert_array_equal(labels, labels[first][inverse])
    assert not np.isin(ids, evidence["arrays"]["audit_ids"]).any()


def test_relocated_fixture_calls_student_functions_without_an_oracle(synthetic_fixture, tmp_path, monkeypatch):
    path, source = _seal_fixture(tmp_path, synthetic_fixture)
    source.unlink()
    evidence = learner.load_native_evidence(path)
    observed = []

    def recover(design, labels):
        observed.append(labels.copy())
        return SimpleNamespace(spectrum={(): float(labels.mean())}, trace=[],
                               initial_bins=np.zeros((3, 19, 8)), residual_bins=np.zeros((3, 19, 8)),
                               stop_reason="synthetic student sentinel", residual_energy=0.0)

    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must never call a native study or oracle")

    monkeypatch.setattr(native_study, "run_study", forbidden)
    namespace = SimpleNamespace(make_query_design=spex_methods.make_query_design,
                                recover_spectrum=recover, predict_spectrum=spex_methods.predict_spectrum)
    replay = learner.replay_condition(evidence, method="spex", methods=namespace, budget=456)
    assert len(observed) == 2 and all(values.shape == (456,) for values in observed)
    assert not np.array_equal(observed[0], observed[1])
    assert replay["replay"]["new_model_queries"] == 0
    assert replay["replay"]["estimator_refits"] == 2
    assert replay["replay"]["audit_labels_received_by_fit"] is False
    assert len(replay["recorded_comparators"]) == 2
    assert all(row["recovery"]["stop_reason"] == "synthetic student sentinel" for row in replay["results"])
    assert all(row["predictions"]["final"].shape == (512,) for row in replay["results"])
    assert all("final" in row["prediction_allclose"] for row in replay["replay"]["comparisons"])
    assert not replay["replay"]["comparisons"][0]["prediction_allclose"]["final"]
    assert replay["replay"]["comparisons"][0]["maximum_prediction_absolute_delta"]["final"] > 0


def test_fit_boundary_never_receives_audit_targets(synthetic_fixture, monkeypatch):
    evidence = learner._decode_fixture(copy.deepcopy(synthetic_fixture))
    observed = {}
    namespace = SimpleNamespace(student_owned=True)

    def sentinel(condition, arrays, labels, *, spex_methods, proxy_methods):
        observed.update(arrays=arrays, labels=labels, spex=spex_methods, proxy=proxy_methods)
        assert proxy_methods is namespace and spex_methods is None
        assert not any("audit_values" in key or "target" in key for key in arrays)
        raise RuntimeError("stop before synthetic proxy fit")

    monkeypatch.setattr(native_study, "fit_condition", sentinel)
    with pytest.raises(RuntimeError, match="stop before"):
        learner.replay_condition(evidence, method="proxyspex", methods=namespace, budget=456)
    assert observed["labels"].shape == (456,)
    assert observed["arrays"]["audit_masks"].shape == (512, 28)


def test_reference_failure_and_stage_errors_remain_explicit():
    record = {"stage_id": "author_proxy_b3", "status": "SOURCE_FAILURE", "predictions": {},
              "spectra": {}, "source_exception_type_message": "IndexError: retained source failure"}
    compact = learner._compact_reference(record, np.ones(512))
    assert compact["status"] == "SOURCE_FAILURE" and compact["metrics"] == {}
    assert "IndexError" in compact["source_exception_type_message"]


def test_native_lineage_must_match_data_contract(synthetic_fixture, tmp_path):
    path, _ = _seal_fixture(tmp_path, synthetic_fixture)
    contract = json.loads(Path(str(path) + ".data-contract.json").read_text())
    changed = json.loads(gzip.decompress(path.read_bytes()))
    changed["binding"]["native_execution_sha256"] = "f" * 64
    path.write_bytes(gzip.compress(json.dumps(changed).encode(), mtime=0))
    seal_json_data(path, provenance=contract["provenance"])
    with pytest.raises(ValueError, match="lineage"):
        learner.load_native_evidence(path)


def test_training_export_uses_primary_scores_and_frozen_union_order():
    physical = {
        "game": np.array(["invariance/probe", "primary", "primary", "counterfactual", "primary", "primary"]),
        "prompt_index": np.array([0, 0, 1, 0, 0, 1]),
        "mask_id": np.array([2, 9, 2, 2, 2, 9]),
        "difference_float64": np.array([99., -.25, .75, -88., .125, -.625]),
    }
    payload = {"prompts": [{}, {}], "oracle_values": {"physical": physical}}
    actual = learner._training_score_table(payload, np.array([2, 9], dtype=np.uint64))
    np.testing.assert_array_equal(actual, [[.125, -.25], [.75, -.625]])


def test_full_loader_accepts_actual_supervisor_status_and_checks_bound_bytes(tmp_path, monkeypatch):
    freeze = json.loads((learner.HERE / "native_freeze.json").read_text())
    plan_directory = Path(freeze["query_plan_directory"])
    plan, arrays = native_query_plan.load_plan(plan_directory)
    execution = {"status": "COMPLETE", "returncode": 0, "changed_sources": [],
                 "resource_violations": [], "failures": []}
    path = tmp_path / "execution.json"
    path.write_text(json.dumps(execution))
    database = tmp_path / "study/oracle/oracle.sqlite3"
    database.parent.mkdir(parents=True)
    database.write_bytes(b"synthetic physical receipt bytes; never queried")
    payload = {"freeze_sha256": "f" * 64, "query_plan": plan,
               "oracle_inventory": {"database": {"sha256": learner._sha(database),
                                                   "size_bytes": database.stat().st_size}}}
    binding = {"native_execution_sha256": learner._sha(path), "summary_hashes": {"synthetic": True},
               "freeze_sha256": "f" * 64, "query_plan_directory": str(plan_directory),
               "query_plan_manifest_sha256": learner._sha(plan_directory / "query_plan.json")}
    monkeypatch.setattr(native_study, "load_record", lambda *args, **kwargs: {})
    monkeypatch.setattr(native_study, "load_study", lambda *args: payload)
    monkeypatch.setattr(learner, "_training_score_table", lambda *args: np.zeros((32, len(arrays["training_union_ids"]))))
    evidence = learner.load_native_evidence(tmp_path, binding=binding)
    assert evidence["evidence_kind"] == "complete_external_study"
    path.write_text(json.dumps(dict(execution, status="FAIL")))
    with pytest.raises(ValueError, match="receipt differs"):
        learner.load_native_evidence(tmp_path, binding=binding)

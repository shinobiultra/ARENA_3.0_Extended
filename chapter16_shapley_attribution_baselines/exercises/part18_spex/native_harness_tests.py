"""Full orchestration checks with synthetic labels, never native model outcomes.

The actual query plan, study loop, baselines, persistence, inventory validator and
loader run here. Tokenization, the oracle, method fits and reference processes
are analytical stubs. Their accounting exercises the protocol's complete grid;
it is not evidence that GPT-2, SPEX or any proxy/reference model was executed.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from . import native_data, native_oracle, native_query_plan, native_study, spex_methods
from ..part19_proxyspex import proxy_methods

SCOPE = "SYNTHETIC_TEST_NOT_NATIVE"


def analytical_logits(prompt_index, masks, *, counterfactual=False):
    """A directly evaluated polynomial, with exactly representable binary values."""
    masks = np.asarray(masks, dtype=np.uint8)
    score = (prompt_index / 16 + masks @ (np.arange(1, 29) / 64)
             + (1 - 2 * masks[:, 0].astype(np.int64)) * (1 - 2 * masks[:, 1].astype(np.int64)) / 8)
    if counterfactual:
        score = -score
    return np.column_stack((score + 2, np.full(len(masks), 2))).astype(np.float32)


def analytical_values(prompt_index, masks, *, counterfactual=False):
    logits = analytical_logits(prompt_index, masks, counterfactual=counterfactual)
    return logits[:, 0].astype(np.float64) - logits[:, 1].astype(np.float64)


class AnalyticalOracle:
    """Track actual requested IDs and cache misses without any model backend."""

    def __init__(self, output_dir, tokenized_records, snapshot_path, *, student_mask_inputs, freeze_path, state):
        assert len(tokenized_records) == 32
        assert all(len(row["feature_names"]) == 28 and "counterfactual" in row for row in tokenized_records)
        assert not Path(snapshot_path).exists(), "Synthetic tests must not point at model weights."
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir()
        self.state = state
        self.cache = set()
        self.receipts = []
        self.batches = []
        self.chunks = []
        self.physical_count = 0
        self.closed = False
        state.oracle = self

    def _request(self, prompt_index, masks, *, stage, method, seed, budget, game="primary", force=False):
        masks = np.asarray(masks)
        assert masks.shape == (budget, 28) and np.isin(masks, (0, 1)).all()
        ids = native_data.mask_ids(masks)
        keys = [(prompt_index, game, int(identity)) for identity in ids]
        prior_hits = sum(key in self.cache for key in keys)
        selected, seen = [], set()
        for index, key in enumerate(keys):
            if force or (key not in self.cache and key not in seen):
                selected.append(index)
                seen.add(key)
        if not force:
            self.cache.update(seen)
        call_id = len(self.receipts) + 1
        logits = analytical_logits(prompt_index, masks, counterfactual=game == "counterfactual")
        for start in range(0, len(selected), 256):
            indices = np.asarray(selected[start:start + 256])
            count = len(indices)
            batch_id = len(self.batches) + 1
            self.batches.append({"evidence_scope": SCOPE, "status": "PASS", "call_id": call_id,
                                 "batch_id": batch_id, "rows": count, "stage": stage,
                                 "prompt_index": prompt_index, "mask_ids": ids[indices].tolist()})
            pair = logits[indices]
            self.chunks.append({
                "physical_id": np.arange(self.physical_count + 1, self.physical_count + count + 1, dtype=np.int64),
                "batch_id": np.full(count, batch_id, dtype=np.int64),
                "row_in_batch": np.arange(count, dtype=np.int64),
                "prompt_index": np.full(count, prompt_index, dtype=np.int64),
                "mask_id": ids[indices].astype(np.int64),
                "target_logits_float32": pair,
                "difference_float64": pair[:, 0].astype(np.float64) - pair[:, 1].astype(np.float64),
                "stage": np.full(count, stage), "game": np.full(count, game), "method": np.full(count, method),
            })
            self.physical_count += count
        receipt = {"evidence_scope": SCOPE, "call_id": call_id, "prompt_index": prompt_index, "game": game,
                   "stage": stage, "method": method, "seed": seed, "budget": budget,
                   "mask_ids": ids.tolist(), "raw_requested": len(ids), "unique_requested": len(set(map(int, ids))),
                   "cache_hits_before_call": prior_hits, "cache_hit_count": len(ids) - len(selected),
                   "new_physical_evaluations": len(selected), "forced_probe": force}
        self.receipts.append(receipt)
        return logits[:, 0].astype(np.float64) - logits[:, 1].astype(np.float64)

    def query(self, prompt_index, masks, *, stage, method, seed, budget):
        if stage == "held_out_audit":
            self.state.check_prediction_boundary()
            self.state.audit_started = True
            np.testing.assert_array_equal(masks, self.state.arrays["audit_masks"])
        elif stage == "training":
            assert not self.state.audit_started, "Training resumed after audit labels became available."
            expected = self.state.plan["conditions"][self.state.fit_calls % 18]
            assert (prompt_index, method, seed, budget) == (
                self.state.fit_calls // 18, expected["method"], expected["seed"], expected["budget"])
            np.testing.assert_array_equal(masks, self.state.arrays[expected["key"] + "_masks"])
        elif stage == "primary_sign":
            assert self.state.audit_started
            np.testing.assert_array_equal(masks, np.ones((1, 28), dtype=np.uint8))
        else:
            raise AssertionError(f"Unplanned synthetic query stage: {stage}")
        return self._request(prompt_index, masks, stage=stage, method=method, seed=seed, budget=budget)

    def validate_invariance(self):
        assert not self.receipts
        probes = np.stack((np.zeros(28, dtype=np.uint8), np.ones(28, dtype=np.uint8),
                           (np.arange(28) % 2 == 0).astype(np.uint8)))
        table = [("invariance_reduced", probes), ("invariance_repeat", probes)]
        table.extend((f"invariance_singleton_{i}", probes[i:i + 1]) for i in range(3))
        table.extend((("invariance_replacement", probes), ("invariance_full_head", probes)))
        for stage, masks in table:
            self._request(0, masks, stage=stage, method="probe", seed=0, budget=len(masks), force=True)
        assert self.physical_count == 15 and not self.cache and len(self.batches) == 7
        return {"evidence_scope": SCOPE, "status": "PASS", "synthetic_probe_rows": 15}

    def counterfactual(self, prompt_index):
        assert self.state.audit_started
        return self._request(prompt_index, np.ones((1, 28), dtype=np.uint8), stage="counterfactual",
                             method="control", seed=0, budget=1, game="counterfactual")

    def inventory(self):
        return {"evidence_scope": SCOPE, "physical_evaluations": self.physical_count,
                "attempted_physical_evaluations": self.physical_count, "cached_unique_masks": len(self.cache),
                "batches": self.batches, "actual_native_model_evaluations": 0}

    def export_values(self):
        return {"evidence_scope": SCOPE,
                "physical": {name: np.concatenate([chunk[name] for chunk in self.chunks]) for name in self.chunks[0]},
                "batches": self.batches, "requests": self.receipts}

    def close(self):
        self.closed = True


@pytest.fixture(scope="module")
def synthetic_study(tmp_path_factory):
    """Run all 1,152 conditions once; every saved receipt is explicitly synthetic."""
    root = tmp_path_factory.mktemp(SCOPE)
    plan_dir, output = root / "query_plan", root / "synthetic_study"
    native_query_plan.write_plan(plan_dir, spex_methods, proxy_methods)
    plan, arrays = native_query_plan.load_plan(plan_dir)
    state = SimpleNamespace(root=root, output=output, plan=plan, arrays=arrays, fit_calls=0,
                            baseline_calls=0, audit_started=False, boundary_checks=0, reference_calls=0,
                            oracle=None, tokenization_calls=0, freeze_checks=0)
    original_save = native_study.save_record

    def marked_save(path, record):
        record["evidence_scope"] = SCOPE
        return original_save(path, record)

    def check_prediction_boundary():
        assert state.fit_calls == 32 * 18 and state.baseline_calls == 2304
        assert state.reference_calls == 9
        if state.boundary_checks == 0:
            boundary = native_study.load_record(output / "prediction_boundary")
            assert boundary["audit_model_queries_so_far"] == 0
            assert len(boundary["condition_records"]) == 1152 and len(boundary["references"]) == 9
            for entry in boundary["condition_records"]:
                row = native_study.load_record(output / entry["path"], expected_hashes=entry["hashes"])
                assert row["evidence_scope"] == SCOPE and "audit_values" not in row
                assert all(value.shape == (512,) for value in row["predictions"].values())
            for row in boundary["references"]:
                assert Path(row["operational"]["predictions_path"]).is_file()
                assert len(row["predictions"]["final"]) == 512
            assert not any(row["stage"] == "held_out_audit" for row in state.oracle.receipts)
        state.boundary_checks += 1

    state.check_prediction_boundary = check_prediction_boundary

    def lightweight_fit(condition, supplied_arrays, train_values, *, spex_methods, proxy_methods):
        assert not state.audit_started
        assert supplied_arrays.keys() == arrays.keys() and "audit_values" not in supplied_arrays
        prompt = state.fit_calls // 18
        assert state.oracle.receipts[-1]["stage"] == "training"
        masks = arrays[condition["key"] + "_masks"]
        np.testing.assert_array_equal(train_values, analytical_values(prompt, masks))
        np.testing.assert_array_equal(state.oracle.receipts[-1]["mask_ids"], arrays[condition["key"] + "_ids"])
        state.fit_calls += 1
        first, inverse = (arrays[condition["key"] + suffix] for suffix in ("_unique_first", "_inverse"))
        shuffled = train_values[first][arrays[condition["key"] + "_shuffle"]][inverse]
        records = []
        for control, labels in (("real", train_values), ("shuffled", shuffled)):
            baselines, baseline_fit = native_study.baseline_predictions(masks, labels, arrays["audit_masks"])
            state.baseline_calls += 2
            # These simple predictions exercise plumbing; no method model is fitted.
            prediction = baselines["constant"].copy()
            row = {"evidence_scope": SCOPE, "method": condition["method"], "seed": condition["seed"],
                   "budget": condition["budget"], "condition": control, "baseline_predictions": baselines,
                   "baseline_fit": {**baseline_fit, "evidence_scope": SCOPE},
                   "spectrum": {(): float(prediction[0]), (0, 1): -0.125},
                   "training_values": labels.copy(), "fit_seconds": 0.0}
            if condition["method"] == "spex":
                row.update(predictions={"final": prediction},
                           fit_accounting={"evidence_scope": SCOPE, "spex_recoveries": 1, "oracle_calls_by_methods": 0})
            else:
                row.update(predictions={key: prediction.copy() for key in ("proxy", "extracted", "truncated", "refined")},
                           fit={"evidence_scope": SCOPE, "proxy_cv": {"selected_parameters": {"n_estimators": 128}}},
                           fit_accounting={"evidence_scope": SCOPE, "oracle_calls_by_methods": 0, "ensemble_fits": 13,
                                           "ridge_candidate_fits": 16, "ridge_matrix_decompositions": 4,
                                           "residual_tree_fits": 3968})
            records.append(row)
        return records

    def lightweight_references(directory, inputs, supplied_arrays, freeze):
        assert not state.audit_started and state.fit_calls == 576
        assert len(list((output / "conditions").glob("*/hashes.json"))) == 1152
        assert len(inputs) == 6 and freeze["evidence_scope"] == SCOPE
        assert supplied_arrays.keys() == arrays.keys() and "audit_values" not in supplied_arrays
        for condition, values in inputs:
            assert condition["seed"] == 0
            np.testing.assert_array_equal(values, analytical_values(0, arrays[condition["key"] + "_masks"]))
        receipts = []
        for bits in (3, 4, 5):
            for variant in ("author_spex", "author_proxy", "maintained_proxy"):
                stage_id = f"{variant}_b{bits}"
                stage_dir = Path(directory) / stage_id
                stage_dir.mkdir(parents=True)
                prediction = np.full(512, bits / 8, dtype=np.float64)
                path = stage_dir / "audit_predictions.npz"
                np.savez_compressed(path, final=prediction, evidence_scope=np.asarray(SCOPE))
                row = {"evidence_scope": SCOPE, "stage_id": stage_id, "variant": variant, "b": bits,
                       "status": "SYNTHETIC_COMPLETED", "audit_values_received": False, "native_oracle_calls": 0,
                       "fallback_used": False, "predictions": {"final": prediction.tolist()},
                       "operational": {"predictions_path": str(path)}, "spectra": {"final": []}}
                (stage_dir / "bridge_stage.json").write_text(json.dumps(row))
                receipts.append(row)
                state.reference_calls += 1
        return receipts

    def tokenize(record, tokenizer):
        assert tokenizer.evidence_scope == SCOPE
        state.tokenization_calls += 1
        return {"input_ids": np.arange(29, dtype=np.int64), "token_feature": np.r_[np.arange(28), -1],
                "target_ids": np.array([10, 11]), "offset_mapping": np.column_stack((np.arange(29), np.arange(29) + 1))}

    freeze_path = root / "synthetic_fixture.json"
    freeze_path.write_text(json.dumps({"evidence_scope": SCOPE, "approval": "NONE; validator stubbed only inside pytest"}))
    freeze = {"evidence_scope": SCOPE, "query_plan_directory": str(plan_dir), "source_inputs": {},
              "model_snapshot_path": str(root / "NO_MODEL_SNAPSHOT"),
              "query_plan_manifest_sha256": hashlib.sha256((plan_dir / "query_plan.json").read_bytes()).hexdigest()}

    def validate_fixture(path):
        assert Path(path) == freeze_path
        state.freeze_checks += 1
        return freeze

    def forbid_native(*args, **kwargs):
        pytest.fail("SYNTHETIC_TEST_NOT_NATIVE must never load model weights or execute scientific method fits.")

    transformers_stub = ModuleType("transformers")
    transformers_stub.AutoTokenizer = SimpleNamespace(
        from_pretrained=lambda *args, **kwargs: SimpleNamespace(evidence_scope=SCOPE))
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("ARENA_SPEX_SUPERVISED_CHILD", "1")
        patch.setitem(sys.modules, "transformers", transformers_stub)
        patch.setattr(native_oracle, "Oracle", lambda *args, **kwargs: AnalyticalOracle(*args, **kwargs, state=state))
        patch.setattr(native_oracle._GPT2Backend, "__init__", forbid_native)
        patch.setattr(spex_methods, "recover_spectrum", forbid_native)
        patch.setattr(proxy_methods, "fit_proxy", forbid_native)
        patch.setattr(native_study, "validate_freeze", validate_fixture)
        patch.setattr(native_data, "tokenize_record", tokenize)
        patch.setattr(native_study, "fit_condition", lightweight_fit)
        patch.setattr(native_study, "run_references", lightweight_references)
        patch.setattr(native_study, "save_record", marked_save)
        state.loaded = native_study.run_study(output, spex_methods=spex_methods,
                                             proxy_methods=proxy_methods, freeze_path=freeze_path)
    (root / "SYNTHETIC_TEST_NOT_NATIVE.json").write_text(json.dumps({
        "evidence_scope": SCOPE, "actual_native_model_evaluations": 0, "actual_proxy_or_reference_fits": 0,
        "synthetic_condition_records": len(state.loaded["results"]),
        "synthetic_oracle_rows": state.oracle.physical_count,
        "purpose": "Complete harness ordering, accounting, serialization and binding checks only.",
    }, indent=2) + "\n")
    return state


def test_complete_synthetic_harness_grid_and_audit_boundary(synthetic_study):
    state = synthetic_study
    loaded = state.loaded
    assert loaded["evidence_scope"] == SCOPE
    assert state.tokenization_calls == 64 and state.freeze_checks == 2 and state.oracle.closed
    assert state.fit_calls == 576 and state.baseline_calls == 2304 and state.boundary_checks == 32
    assert len(loaded["results"]) == 1152 and len(loaded["audit"]) == 32 and len(loaded["references"]) == 9
    observed = loaded["observed_inventory"]
    assert observed["baseline_fits"] == 2304
    assert observed["spex_recoveries"] == observed["proxy_pipelines"] == 576
    assert observed["proxy_ensemble_fits"] == 7488 and observed["ridge_candidate_fits"] == 9216
    assert observed["ridge_matrix_decompositions"] == 2304 and observed["proxy_residual_tree_fits"] == 2285568
    expected = [(prompt, condition["method"], condition["seed"], condition["budget"], control)
                for prompt in range(32) for condition in state.plan["conditions"] for control in ("real", "shuffled")]
    assert [(row["prompt_index"], row["method"], row["seed"], row["budget"], row["condition"])
            for row in loaded["results"]] == expected
    for row in loaded["results"]:
        assert row["evidence_scope"] == SCOPE and (0, 1) in row["spectrum"]
        np.testing.assert_array_equal(row["audit_values"], analytical_values(row["prompt_index"], state.arrays["audit_masks"]))
        assert row["audit_values"].dtype == np.float64 and row["audit_values"].shape == (512,)
        assert set(row["metrics"]) == set(row["predictions"]) | {"constant", "additive"}
    print("Full synthetic harness preserved the 1,152-condition order and withheld all audit labels until predictions were saved.")


def test_synthetic_physical_export_retains_all_requested_ids_values_and_counts(synthetic_study):
    state = synthetic_study
    exported = state.loaded["oracle_values"]
    physical = exported["physical"]
    receipts = state.loaded["oracle_receipts"]
    assert exported["evidence_scope"] == SCOPE
    assert len(physical["physical_id"]) == state.oracle.physical_count == 481999
    assert len(state.oracle.cache) == state.loaded["oracle_inventory"]["cached_unique_masks"] == 481984
    np.testing.assert_array_equal(physical["physical_id"], np.arange(1, 482000))
    np.testing.assert_array_equal(physical["difference_float64"],
                                  physical["target_logits_float32"][:, 0].astype(np.float64)
                                  - physical["target_logits_float32"][:, 1].astype(np.float64))
    assert physical["target_logits_float32"].dtype == np.float32
    assert physical["difference_float64"].dtype == np.float64
    stage_calls = Counter(row["stage"] for row in receipts)
    assert stage_calls["training"] == 576 and stage_calls["held_out_audit"] == 32
    assert stage_calls["primary_sign"] == stage_calls["counterfactual"] == 32
    assert sum(count for stage, count in stage_calls.items() if stage.startswith("invariance_")) == 7
    assert sum(row["new_physical_evaluations"] for row in receipts) == 481999
    assert all(row["new_physical_evaluations"] == 0 for row in receipts if row["stage"] == "primary_sign")
    assert all(0 < batch["rows"] <= 256 and batch["status"] == "PASS" for batch in exported["batches"])
    for prompt in range(32):
        training = physical["mask_id"][(physical["prompt_index"] == prompt) & (physical["stage"] == "training")]
        np.testing.assert_array_equal(np.sort(training), state.arrays["training_union_ids"])
        held_out = physical["mask_id"][(physical["prompt_index"] == prompt) & (physical["stage"] == "held_out_audit")]
        np.testing.assert_array_equal(held_out, state.arrays["audit_ids"])
        audit = state.loaded["audit"][prompt]
        np.testing.assert_array_equal(audit["counterfactual"],
                                      analytical_values(prompt, np.ones((1, 28), dtype=np.uint8), counterfactual=True))
    # Compare the exported rows independently from their serialization source.
    ids = physical["mask_id"].astype(np.uint64)
    for start in range(0, len(ids), 8192):
        stop = start + 8192
        masks = ((ids[start:stop, None] >> np.arange(27, -1, -1, dtype=np.uint64)) & 1).astype(np.uint8)
        expected = analytical_logits(physical["prompt_index"][start:stop], masks)
        changed = physical["game"][start:stop] == "counterfactual"
        expected[changed] = analytical_logits(physical["prompt_index"][start:stop][changed], masks[changed], counterfactual=True)
        np.testing.assert_array_equal(physical["target_logits_float32"][start:stop], expected)
    print("Synthetic row export retained all 481,999 analytical evaluations, including 15 probes and 32 counterfactuals; native evaluations = 0.")


@pytest.mark.parametrize("target", ["condition", "reference", "audit", "oracle_values", "prediction_boundary"])
def test_locally_rehashed_saved_payload_cannot_cross_ancestor_binding(synthetic_study, tmp_path, target):
    state = synthetic_study
    if target == "condition":
        path = state.output / state.loaded["conditions"][0]["path"]
    elif target == "audit":
        path = state.output / state.loaded["audit_records"][0]["path"]
    elif target == "reference":
        path = state.output / "summary"
    else:
        path = state.output / target
    originals = {name: (path / name).read_bytes() for name in ("arrays.npz", "record.json", "hashes.json")}
    record = native_study.load_record(path)
    if target == "condition":
        record["predictions"]["final"][0] += 1
    elif target == "reference":
        record["references"][0]["predictions"]["final"][0] += 1
    elif target == "audit":
        record["audit_values"][0] += 1
    elif target == "oracle_values":
        record["physical"]["target_logits_float32"][0, 0] += 1
    else:
        record["audit_model_queries_so_far"] = 1
    replacement = tmp_path / "locally_rehashed_synthetic_payload"
    native_study.save_record(replacement, record)
    try:
        for name in originals:
            (path / name).write_bytes((replacement / name).read_bytes())
        # Local integrity is valid; rejection must come from an earlier binding.
        native_study.load_record(path)
        with pytest.raises(ValueError, match="pre-audit or summary binding|Reference predictions changed"):
            native_study.load_study(state.output)
    finally:
        for name, data in originals.items():
            (path / name).write_bytes(data)
    print("Locally rehashed synthetic payload was rejected by its saved ancestor binding:", target)

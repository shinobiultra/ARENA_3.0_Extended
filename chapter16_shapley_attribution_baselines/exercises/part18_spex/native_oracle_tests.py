"""Independent analytical fake-model checks; never load or execute native GPT-2."""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from . import native_data
from . import native_oracle as module
from .native_oracle import Oracle


class AnalyticalBackend:
    """Literal position-weighted retained-token sums with two affine target heads."""

    kind = "analytical_fake_model_no_native_weights"

    def __init__(self, *, broken_head=False, wrong_dtype=False):
        self.calls = []
        self.broken_head = broken_head
        self.wrong_dtype = wrong_dtype

    def predict(self, inputs, target_ids, *, full_head=False):
        self.calls.append({name: value.copy() for name, value in inputs.items()})
        assert np.array_equal(target_ids, [10, 11])
        assert len(inputs["input_ids"]) <= 256
        products = inputs["input_ids"] * (inputs["position_ids"] + 1) * inputs["attention_mask"]
        total = products.sum(axis=1).astype(np.float32)
        result = np.stack((total / np.float32(128) + np.float32(0.25),
                           total / np.float32(256) - np.float32(0.5)), axis=1)
        if self.broken_head and full_head:
            result[:, 0] += np.float32(0.1)
        return result.astype(np.float64) if self.wrong_dtype else result


def records():
    ids = np.arange(1, 30, dtype=np.int64)
    ids[20] = 11
    ids[-1] = 99
    primary = {"id": "analytical/prompt0", "input_ids": ids,
               "token_feature": np.r_[np.arange(28), -1].astype(np.int64),
               "target_ids": np.array([10, 11], dtype=np.int64),
               "subject_word_index": 20}
    counterfactual = {name: value.copy() for name, value in primary.items() if isinstance(value, np.ndarray)}
    counterfactual["input_ids"][20] = 10
    primary["counterfactual"] = counterfactual
    return [primary]


def snapshot(path):
    path.mkdir()
    for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"):
        (path / name).write_text("Analytical test bytes, not native model weights: " + name)
    return path


def approve(oracle, path):
    sources = [Path(module.__file__).resolve(), Path(native_data.__file__).resolve()]
    freeze = {"status": "APPROVED", "oracle_identity_sha256": oracle.identity_sha256,
              "native_queries_before_freeze": 0, "native_fits_before_freeze": 0,
              "source_inputs": {str(source.relative_to(module.ROOT)): hashlib.sha256(source.read_bytes()).hexdigest() for source in sources}}
    path.write_text(json.dumps(freeze))
    oracle.authorize(path)
    return path


@pytest.fixture
def rig(tmp_path, monkeypatch):
    backend = AnalyticalBackend()
    load_calls = []

    def factory(self):
        load_calls.append(self.snapshot_path)
        return backend

    monkeypatch.setattr(Oracle, "_make_backend", factory)
    oracle = Oracle(tmp_path / "oracle", records(), snapshot(tmp_path / "snapshot"))
    yield oracle, backend, load_calls, tmp_path
    oracle.close()


def call(oracle, masks, **kwargs):
    metadata = {"stage": "training", "method": "analytical_fixture", "seed": 0, "budget": len(masks)}
    metadata.update(kwargs)
    return oracle.query(0, masks, **metadata)


def masks_for_ids(ids):
    return ((np.asarray(ids, dtype=np.uint64)[:, None] >> np.arange(27, -1, -1, dtype=np.uint64)) & 1).astype(np.int64)


def independent_expected(masks, *, counterfactual=False):
    # Deliberately do not call the masker, oracle or fake backend to form the oracle.
    token_values = list(range(1, 29))
    token_values[20] = 10 if counterfactual else 11
    values = []
    for row in masks:
        total = np.float32(99 * 29 + sum(int(keep) * token * (position + 1)
                                      for position, (keep, token) in enumerate(zip(row, token_values, strict=True))))
        first = np.float32(total / np.float32(128) + np.float32(0.25))
        second = np.float32(total / np.float32(256) - np.float32(0.5))
        values.append(np.float64(first) - np.float64(second))
    return np.asarray(values, dtype=np.float64)


def test_construction_never_loads_weights_and_queries_require_approved_freeze(rig):
    oracle, backend, loads, directory = rig
    assert not loads and not backend.calls
    assert oracle.inventory()["physical_evaluations"] == 0
    with pytest.raises(PermissionError, match="approved freeze"):
        call(oracle, np.ones((1, 28), dtype=np.int64))
    assert not loads and not backend.calls
    approve(oracle, directory / "freeze.json")
    assert not loads
    np.testing.assert_array_equal(call(oracle, np.ones((1, 28), dtype=np.int64)), independent_expected(np.ones((1, 28))))
    assert len(loads) == 1
    print("Oracle construction and approval perform no model forward; the first approved query loads only its backend.")


def test_eight_thread_engine_contract_keeps_blas_four_and_all_other_settings(rig, monkeypatch):
    """Exercise constructor configuration against module stubs, with no weights."""
    oracle, _, loads, _ = rig
    assert oracle.identity["engine"] == {
        "dtype": "float32", "difference_dtype": "float64", "device": "cpu",
        "attention": "eager_causal", "eval": True, "threads": 8,
        "max_batch": 256, "new_mask_order": "first_occurrence", "replacement_id": 50256,
    }
    calls = []
    torch_stub = ModuleType("torch")
    torch_stub.float32 = "SYNTHETIC_FLOAT32_TYPE"
    torch_stub.set_num_threads = lambda count: calls.append(("threads", count))
    torch_stub.use_deterministic_algorithms = lambda enabled: calls.append(("deterministic", enabled))

    class SyntheticModel:
        config = SimpleNamespace(model_type="gpt2", n_layer=12, n_embd=768,
                                 vocab_size=50257, _attn_implementation="eager", use_cache=True)

        def cpu(self):
            calls.append(("device", "cpu"))
            return self

        def float(self):
            calls.append(("dtype", "float32"))
            return self

        def eval(self):
            calls.append(("eval", True))
            return self

        def requires_grad_(self, enabled):
            calls.append(("requires_grad", enabled))

    def synthetic_loader(path, **kwargs):
        calls.append(("synthetic_loader", path, kwargs))
        return SyntheticModel()

    transformers_stub = ModuleType("transformers")
    transformers_stub.GPT2LMHeadModel = SimpleNamespace(from_pretrained=synthetic_loader)
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "transformers", transformers_stub)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "4")
    backend = module._GPT2Backend(oracle.snapshot_path)
    assert calls == [
        ("threads", 8), ("deterministic", True),
        ("synthetic_loader", str(oracle.snapshot_path),
         {"local_files_only": True, "dtype": "SYNTHETIC_FLOAT32_TYPE", "attn_implementation": "eager"}),
        ("device", "cpu"), ("dtype", "float32"), ("eval", True), ("requires_grad", False),
    ]
    assert not backend.model.config.use_cache
    assert not loads and oracle.inventory()["physical_evaluations"] == 0
    calls.clear()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "8")
        with pytest.raises(RuntimeError, match=name + "=4"):
            module._GPT2Backend(oracle.snapshot_path)
        monkeypatch.setenv(name, "4")
    assert not calls, "Incorrect BLAS settings must fail before even a synthetic loader is called."
    print("SYNTHETIC_TEST_NOT_NATIVE: eight PyTorch inference threads are bound; CPU float32, eager attention and four BLAS threads remain fixed.")


def test_request_order_duplicate_accounting_and_exact_promoted_values(rig):
    oracle, backend, _, directory = rig
    approve(oracle, directory / "freeze.json")
    masks = masks_for_ids([0, 2**28 - 1, 6, 0, 6, 7])
    result = call(oracle, masks)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, independent_expected(masks))
    receipt = oracle.receipts[-1]
    assert (receipt["raw_count"], receipt["unique_count"], receipt["new_physical_count"], receipt["cache_hit_count"]) == (6, 4, 4, 2)
    assert receipt["cached_unique_before_call"] == receipt["cache_hits_before_call"] == 0
    np.testing.assert_array_equal(call(oracle, masks[[4, 0, 4]]), result[[4, 0, 4]])
    assert oracle.receipts[-1]["new_physical_count"] == 0
    assert oracle.receipts[-1]["cache_hits_before_call"] == 3
    assert len(backend.calls) == 1
    inventory = list(oracle.physical_rows())
    assert [row["mask_id"] for row in inventory] == [0, 2**28 - 1, 6, 7]
    print("Logical duplicates, pre-call cache hits and new physical rows are distinct; output order and float64 differences are exact.")


def test_fixed_256_row_batches_and_only_requested_masks_are_evaluated(rig):
    oracle, backend, _, directory = rig
    approve(oracle, directory / "freeze.json")
    masks = masks_for_ids(np.arange(513)[::-1])
    np.testing.assert_array_equal(call(oracle, masks), independent_expected(masks))
    assert [len(batch["input_ids"]) for batch in backend.calls] == [256, 256, 1]
    exported = oracle.export_values()
    np.testing.assert_array_equal(exported["physical"]["mask_id"], np.arange(513)[::-1])
    assert oracle.inventory()["physical_evaluations"] == oracle.inventory()["cached_unique_masks"] == 513
    assert exported["physical"]["target_logits_float32"].dtype == np.float32
    print("Only 513 requested masks are evaluated in first-occurrence order, in batches 256/256/1.")


def test_disk_reopen_uses_exact_identity_and_never_loads_for_cached_rows(rig, monkeypatch):
    oracle, _, _, directory = rig
    freeze = approve(oracle, directory / "freeze.json")
    masks = masks_for_ids([0, 17, 123])
    expected = call(oracle, masks)
    with Oracle(oracle.output_dir, records(), oracle.snapshot_path, freeze_path=freeze) as reopened:
        monkeypatch.setattr(reopened, "_make_backend", lambda: pytest.fail("Cached rows must not load a model."))
        np.testing.assert_array_equal(call(reopened, masks[::-1]), expected[::-1])
    altered = records()
    altered[0]["input_ids"][0] = 123
    with pytest.raises(ValueError, match="different exact identity"):
        Oracle(oracle.output_dir, altered, oracle.snapshot_path)
    (oracle.snapshot_path / "merges.txt").write_text("Changed tokenizer bytes")
    with pytest.raises(PermissionError, match="model/tokenizer"):
        call(oracle, masks)
    print("Reopening exact cached inputs performs zero forwards; token or physical snapshot drift is rejected.")


def test_freeze_or_source_changes_revoke_even_cached_queries(rig):
    oracle, backend, _, directory = rig
    freeze_path = approve(oracle, directory / "freeze.json")
    call(oracle, masks_for_ids([0]))
    freeze = json.loads(freeze_path.read_text())
    freeze["native_queries_before_freeze"] = 1
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(PermissionError, match="unchanged"):
        call(oracle, masks_for_ids([0]))
    with pytest.raises(PermissionError, match="precede"):
        oracle.authorize(freeze_path)
    freeze["native_queries_before_freeze"] = 0
    freeze["source_inputs"][str(Path(module.__file__).resolve().relative_to(module.ROOT))] = "0" * 64
    freeze_path.write_text(json.dumps(freeze))
    with pytest.raises(PermissionError, match="source changed"):
        oracle.authorize(freeze_path)
    assert len(backend.calls) == 1
    print("Modified approval, post-query freeze claims and changed bound source revoke both cache access and inference.")


def test_malformed_queries_and_masking_implementation_fail_before_forward(rig, monkeypatch):
    oracle, backend, _, directory = rig
    approve(oracle, directory / "freeze.json")
    for masks in (np.zeros((1, 27)), np.full((1, 28), 2), np.full((1, 28), np.nan)):
        with pytest.raises(ValueError, match="binary"):
            call(oracle, masks)
    with pytest.raises(ValueError, match="prompt_index"):
        oracle.query(1, np.ones((1, 28)), stage="x", method="y", seed=0, budget=1)
    with pytest.raises(ValueError, match="labels"):
        call(oracle, np.ones((1, 28)), stage="")
    # Inject an invalid result at the mechanical interface without claiming it is a native masker.
    original = oracle.student_mask_inputs
    monkeypatch.setattr(oracle, "_approval", lambda: None)

    def broken(*args, **kwargs):
        result = original(*args, **kwargs)
        result["position_ids"][:, -1] = 0
        return result

    oracle.student_mask_inputs = broken
    with pytest.raises(ValueError, match="position_ids"):
        call(oracle, np.zeros((1, 28)))
    assert not backend.calls
    print("Invalid masks, labels, prompt IDs and student position changes are rejected before inference.")


def test_invariance_probes_include_all_fifteen_physical_rows_and_full_head(rig):
    oracle, _, _, directory = rig
    approve(oracle, directory / "freeze.json")
    result = oracle.validate_invariance()
    assert result["status"] == "PASS"
    inventory = oracle.inventory()
    assert inventory["physical_evaluations"] == inventory["attempted_physical_evaluations"] == 15
    assert inventory["cached_unique_masks"] == 0
    assert len(inventory["batches"]) == 7
    assert inventory["batches"][-1]["identity"]["head"] == "full"
    assert all(call["forced_probe"] for call in oracle.receipts)
    assert all(call["cache_hit_count"] == 0 for call in oracle.receipts)
    with pytest.raises(ValueError, match="fixed prospectively"):
        oracle.validate_invariance(masks=np.zeros((3, 28)))
    print("All repeats, singleton batches, changed replacement IDs and full-head controls remain in the fifteen-row physical inventory.")


def test_failed_invariance_retains_all_outcomes_without_changing_tolerances(rig):
    oracle, backend, _, directory = rig
    backend.broken_head = True
    approve(oracle, directory / "freeze.json")
    with pytest.raises(ValueError, match="invariance check failed"):
        oracle.validate_invariance()
    saved = json.loads((oracle.output_dir / "invariance_000.json").read_text())
    assert saved["status"] == "FAIL" and not saved["comparisons"]["full_head"]["passed"]
    assert saved["plan"]["atol"] == 1e-5 and saved["plan"]["rtol"] == 1e-6
    assert oracle.inventory()["physical_evaluations"] == 15
    with pytest.raises(RuntimeError, match="silently resume"):
        call(oracle, masks_for_ids([0]))
    print("A deliberately incorrect full head fails at the prospective tolerance and preserves every probe result.")


def test_subject_counterfactual_has_separate_cache_and_physical_inventory(rig):
    oracle, _, _, directory = rig
    approve(oracle, directory / "freeze.json")
    full = np.ones((1, 28), dtype=np.int64)
    primary = call(oracle, full)
    counterfactual = oracle.counterfactual(0)
    np.testing.assert_array_equal(counterfactual, independent_expected(full, counterfactual=True))
    assert not np.array_equal(primary, counterfactual)
    assert oracle.inventory()["physical_evaluations"] == 2
    np.testing.assert_array_equal(oracle.counterfactual(0), counterfactual)
    assert oracle.receipts[-1]["new_physical_count"] == 0
    assert {row["game"] for row in oracle.physical_rows()} == {"primary", "counterfactual"}
    wrong = copy.deepcopy(oracle.records[0]["counterfactual"])
    wrong["input_ids"][0] = 123
    with pytest.raises(ValueError, match="frozen record"):
        oracle.counterfactual(0, wrong)
    print("The subject-only counterfactual keeps target orientation, adds one physical row, and has a distinct cache identity.")


def test_corrupted_cached_logits_are_rejected(rig):
    oracle, _, _, directory = rig
    approve(oracle, directory / "freeze.json")
    call(oracle, masks_for_ids([0]))
    with oracle._db:
        oracle._db.execute("UPDATE physical_rows SET value=value+1")
    with pytest.raises(ValueError, match="checksum"):
        call(oracle, masks_for_ids([0]))
    with pytest.raises(ValueError, match="checksum"):
        oracle.export_values()
    print("Changed stored numerical values fail checksum validation on both cache reads and comparison exports.")


def test_failed_backend_batch_is_retained_and_cannot_silently_resume(rig):
    oracle, backend, _, directory = rig
    backend.wrong_dtype = True
    approve(oracle, directory / "freeze.json")
    with pytest.raises(ValueError, match="float32"):
        call(oracle, masks_for_ids([0, 1]))
    inventory = oracle.inventory()
    assert inventory["physical_evaluations"] == 0
    assert inventory["attempted_physical_evaluations"] == 2
    assert inventory["batches"][0]["status"] == oracle.receipts[0]["status"] == "FAIL"
    with pytest.raises(RuntimeError, match="silently resume"):
        call(oracle, masks_for_ids([0]))
    print("A failed backend preserves its attempted physical cost and blocks an unrecorded continuation.")


def test_same_masker_compiled_in_notebook_has_location_independent_identity():
    namespace = {"np": np, "N_FEATURES": 28}
    source = "\n" * 40 + inspect.getsource(native_data.masked_word_inputs)
    exec(compile(source, "<completed learner cell>", "exec", dont_inherit=True), namespace)
    assert module.masking_identity(namespace["masked_word_inputs"]) == module.masking_identity(native_data.masked_word_inputs)
    print("The same visible masking definition has identical semantic code identity across notebook filenames and line positions.")


def test_two_fresh_analytical_runs_export_identical_values_and_identifiers(tmp_path, monkeypatch):
    monkeypatch.setattr(Oracle, "_make_backend", lambda self: AnalyticalBackend())
    model_snapshot = snapshot(tmp_path / "snapshot")
    exports = []
    for index in (0, 1):
        with Oracle(tmp_path / f"run{index}", records(), model_snapshot) as oracle:
            approve(oracle, tmp_path / f"freeze{index}.json")
            oracle.validate_invariance()
            call(oracle, masks_for_ids([7, 0, 2**28 - 1, 7, 3]))
            oracle.counterfactual(0)
            exports.append(oracle.export_values())
    first, second = exports
    assert module._sha(first) == module._sha(second)
    assert len(first["physical"]["physical_id"]) == 20
    assert first["physical"]["target_logits_float32"].shape == (20, 2)
    assert sum(request["raw_count"] for request in first["requests"]) == 21
    assert not any("started_at_utc" in request for request in first["requests"])
    print("Independent cold analytical runs export identical physical values, logical requests, mask IDs, stages and batches.")

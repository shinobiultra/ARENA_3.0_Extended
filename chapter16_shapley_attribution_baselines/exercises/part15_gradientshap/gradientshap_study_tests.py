"""No-fit semantic tests and an explicit post-freeze native fit gate."""

from __future__ import annotations

import inspect
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest
import torch as t

from chapter16_shapley_attribution_baselines.exercises.part14_deepshap import (
    deepshap_methods as deep_methods,
)
from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap import (
    gradientshap_methods as methods,
)
from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap import (
    gradientshap_study as study,
)


def _active_mlp(scale: float) -> t.nn.Sequential:
    model = t.nn.Sequential(
        t.nn.Linear(8, 32, dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(32, 32, dtype=t.float64),
        t.nn.ReLU(),
        t.nn.Linear(32, 1, dtype=t.float64),
    )
    with t.no_grad():
        model[0].weight.copy_(
            t.arange(32 * 8, dtype=t.float64).reshape(32, 8) * scale / 100_000
        )
        model[0].bias.fill_(1.0)
        model[2].weight.copy_(t.eye(32, dtype=t.float64) * (0.2 + scale / 50))
        model[2].bias.fill_(0.5)
        model[4].weight.copy_(
            t.linspace(-0.05, 0.08, 32, dtype=t.float64)[None, :] * scale
        )
        model[4].bias.fill_(20.0 + scale)
    return model


def _record(model: t.nn.Sequential, key: str, train_x: t.Tensor, train_y: t.Tensor) -> dict:
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    index = study.MODEL_KEYS.index(key)
    permutation = t.arange(len(train_x), dtype=t.int64)
    shuffle_seed = None
    if key.startswith("shuffled_"):
        shuffle_seed = study.SHUFFLE_SEEDS[index - 3]
        permutation = t.from_numpy(
            np.random.default_rng(shuffle_seed).permutation(len(train_x))
        )
    kind = "ridge" if key == "ridge" else "mlp"
    initial = {} if kind == "ridge" else {name: value.clone() for name, value in state.items()}
    hashes = {
        "initial_state": study.tensor_mapping_sha256(initial),
        "standardized_state": study.tensor_mapping_sha256(state),
        "raw_folded_state": study.tensor_mapping_sha256(state),
    }
    with t.no_grad():
        prediction = model(train_x).squeeze(-1)
    return {
        "kind": kind,
        "control": "ridge" if key == "ridge" else key.rsplit("_", 1)[0],
        "seed": 0 if key == "ridge" else study.MODEL_SEEDS[index % 3],
        "initial_state": initial,
        "standardized_state": {name: value.clone() for name, value in state.items()},
        "state": state,
        "loss_trace": t.zeros(
            1 if kind == "ridge" else study.TRAINING_STEPS, dtype=t.float64
        ),
        "target_permutation": permutation,
        "training_targets_raw": train_y[permutation],
        "training_prediction": prediction,
        "folding_max_abs_error": 0.0,
        "parameters": {"shuffle_seed": shuffle_seed},
        "state_hashes": hashes,
    }


def _fixture(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, dict, dict, dict]:
    monkeypatch.setattr(study, "BACKGROUND_SPECS", {
        "primary64": (161460, 4),
        "nested16": (161460, 2),
        "nested128": (161460, 6),
        "independent64_a": (161461, 4),
        "independent64_b": (161462, 4),
    })
    monkeypatch.setattr(study, "BUDGETS", (2, 4))
    monkeypatch.setattr(study, "QUADRATURE_NODES", (2, 4))
    monkeypatch.setattr(study, "REPLICATES", 2)
    monkeypatch.setattr(study, "PATH_DISTANCE_SAMPLES", 2)
    monkeypatch.setattr(study, "COALITION_DISTANCE_DONORS", 2)

    grid = t.arange(12 * 8, dtype=t.float64).reshape(12, 8)
    train_x = (grid.remainder(17) / 17).contiguous()
    train_y = 20 + train_x @ t.linspace(0.1, 0.8, 8, dtype=t.float64)
    train = {
        "x": train_x.numpy().copy(),
        "y": train_y.numpy().copy(),
        "ids": np.arange(12, dtype=np.int64),
    }
    explain_x = train_x[[2, 5]] + 0.02
    audit_x = train_x[[7, 9]] + 0.04
    explain = {
        "x": explain_x.numpy().copy(),
        "y": (20 + explain_x.sum(1)).numpy().copy(),
        "ids": np.array([20, 21], dtype=np.int64),
    }
    audit = {
        "x": audit_x.numpy().copy(),
        "y": (20 + audit_x.sum(1)).numpy().copy(),
        "ids": np.array([30, 31], dtype=np.int64),
    }
    models = {
        key: _record(_active_mlp(1 + index / 10), key, train_x, train_y)
        for index, key in enumerate(study.MODEL_KEYS[:-1])
    }
    ridge = t.nn.Sequential(t.nn.Linear(8, 1, dtype=t.float64))
    with t.no_grad():
        ridge[0].weight.copy_(t.linspace(-0.3, 0.4, 8, dtype=t.float64)[None, :])
        ridge[0].bias.fill_(10.0)
    models["ridge"] = _record(ridge, "ridge", train_x, train_y)
    normalization = {
        "x_mean": train_x.mean(0),
        "x_std": train_x.std(0, correction=0),
        "y_mean": train_y.mean(),
        "y_std": train_y.std(correction=0),
    }
    fitted = {
        "models": models,
        "normalization": normalization,
        "train_ids": t.from_numpy(train["ids"]).clone(),
        "train_x_raw": train_x.clone(),
        "train_y_raw": train_y.clone(),
        "units": dict(study.UNITS),
        "feature_names": study.FEATURE_NAMES,
        "model_labels": dict(study.MODEL_LABELS),
        "phase": "fitted_before_attribution",
    }
    background_manifest = {}
    for name, (seed, size) in study.BACKGROUND_SPECS.items():
        indices = study.background_order(train["ids"], seed)[:size]
        background_manifest[name] = {
            "seed": seed,
            "rows": size,
            "train_indices": indices.tolist(),
            "source_ids": train["ids"][indices].tolist(),
        }
    replicate_seeds = [study._replicate_seed(index) for index in range(study.REPLICATES)]
    manifest = {
        "splits": {
            "train": train["ids"].tolist(),
            "explain": explain["ids"].tolist(),
            "audit": audit["ids"].tolist(),
        },
        "backgrounds": background_manifest,
        "known_16_14_evidence": {
            "fitted_artifact_sha256": "fixture",
            "model_state_hashes": {
                key: record["state_hashes"] for key, record in models.items()
            },
        },
        "model_recipe": {
            "model_keys": list(study.MODEL_KEYS),
            "model_seeds": list(study.MODEL_SEEDS),
            "shuffle_seeds": list(study.SHUFFLE_SEEDS),
            "training_steps": study.TRAINING_STEPS,
            "learning_rate": study.LEARNING_RATE,
            "weight_decay": study.WEIGHT_DECAY,
            "ridge_penalty": study.RIDGE_PENALTY,
        },
        "sampling_design_candidate": {"replicate_seeds": replicate_seeds},
        "replicate_diagnostic_candidate": {
            "explanation_source_ids": [20],
            "audit_source_ids": [30],
        },
    }
    monkeypatch.setattr(study, "load_manifest", lambda: manifest)
    return fitted, train, explain, audit, manifest


def test_protocol_constants_and_public_signatures_are_fixed() -> None:
    assert study.MODEL_KEYS == (
        "clean_0", "clean_1", "clean_2", "shuffled_0", "shuffled_1",
        "shuffled_2", "ridge",
    ), "The seven-model order changed."
    assert (study.BUDGETS, study.QUADRATURE_NODES, study.NOISE_SIGMAS) == (
        (8, 32, 128, 512, 2048), (256, 512), (0.0, 0.02, 0.10)
    ), "The frozen estimator budgets changed."
    assert tuple(inspect.signature(study.fit_models).parameters) == ("train",), (
        "fit_models must not accept audit data or a training-step override."
    )
    assert tuple(inspect.signature(study.prepare_study).parameters) == (
        "fitted", "train", "explain", "methods", "deep_methods"
    ), "Preparation must receive explanation data without an audit argument."


def test_frozen_replicate_seeds_and_noise_stream_match_literal_digest_rules() -> None:
    manifest = study.load_manifest()
    declared = manifest["sampling_design_candidate"]["replicate_seeds"]
    expected = [
        int.from_bytes(
            hashlib.sha256(
                f"arena16.15/gradientshap/v1/161511/replicate/{index}".encode()
            ).digest()[:8],
            "little",
        )
        for index in range(64)
    ]
    assert declared == expected, "The 64 frozen replicate seeds changed."
    assert [study._replicate_seed(index) for index in range(64)] == expected, (
        "The study no longer derives the frozen replicate seeds literally."
    )
    source_ids = t.tensor([1, 30], dtype=t.int64)
    observed = study._standard_normal_schedule(source_ids, 3)
    replay = []
    for source_id in source_ids.tolist():
        digest = hashlib.sha256(
            f"arena16.15/gradientshap/v1/161512/{source_id}/standard_normal".encode()
        ).digest()
        rng = np.random.Generator(
            np.random.PCG64(int.from_bytes(digest[:16], "little"))
        )
        replay.append(rng.standard_normal((3, 8)))
    assert t.equal(observed, t.tensor(np.stack(replay), dtype=t.float64)), (
        "Noise draws changed namespace, tag, byte order or PCG64 call."
    )


def test_state_folding_preserves_raw_predictions_without_mutation() -> None:
    t.manual_seed(161590)
    model = _active_mlp(1.2)
    state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    normalization = {
        "x_mean": t.linspace(-2, 2, 8, dtype=t.float64),
        "x_std": t.linspace(0.5, 1.5, 8, dtype=t.float64),
        "y_mean": t.tensor(31.0, dtype=t.float64),
        "y_std": t.tensor(7.0, dtype=t.float64),
    }
    raw_state = study._fold_mlp_state(state, normalization)
    raw_model = study._model_from_state({"kind": "mlp", "state": raw_state})
    x_raw = t.randn(9, 8, dtype=t.float64)
    expected = model(
        (x_raw - normalization["x_mean"]) / normalization["x_std"]
    ).squeeze(-1) * normalization["y_std"] + normalization["y_mean"]
    assert t.allclose(raw_model(x_raw).squeeze(-1), expected, atol=1e-12, rtol=1e-12), (
        "MLP folding changed raw-unit predictions."
    )
    assert all(t.equal(state[name], model.state_dict()[name]) for name in state), (
        "MLP folding mutated the caller's standardized state."
    )


def test_reproduction_gate_checks_all_states_and_bound_fit_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted, _, _, _, manifest = _fixture(monkeypatch)
    result = study.verify_16_14_reproduction(fitted, manifest)
    assert result["status"] == "PASS" and result["mismatches"] == (), (
        "The exact fixture should pass the pre-attribution reproduction gate."
    )
    changed = {**fitted, "models": dict(fitted["models"])}
    changed_record = dict(changed["models"]["clean_0"])
    changed_state = {
        name: value.clone() for name, value in changed_record["state"].items()
    }
    changed_state["4.bias"][0] += 1e-12
    changed_record["state"] = changed_state
    changed["models"]["clean_0"] = changed_record
    with pytest.raises(AssertionError, match="clean_0.raw_folded_state"):
        study.verify_16_14_reproduction(changed, manifest)


def test_prepare_and_evaluate_retain_all_tiny_analogs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted, train, explain, audit, _ = _fixture(monkeypatch)
    prepared = study.prepare_study(
        fitted, train, explain, methods=methods, deep_methods=deep_methods
    )
    assert prepared["phase"] == "prepared_before_audit", (
        "Preparation did not stop at the audit boundary."
    )
    assert prepared["audit_used_for_selection"] is False, (
        "Preparation must explicitly exclude audit selection."
    )
    assert set(prepared["rankings"]) == {
        "exact", "deep", "gradient", "eg_k2048", "quadrature_512"
    }, "Every explanation-only method order must be frozen."
    for key in study.MODEL_KEYS:
        item = prepared["primary"][key]
        assert item["mc"]["samples"].shape == (2, 4, 8), (
            f"{key} lost complete primary contribution draws."
        )
        assert item["quadrature"]["pair_4"].shape == (2, 4, 8), (
            f"{key} lost donor-wise high-node quadrature."
        )
        assert item["baselines"]["hybrid_prediction"].shape == (2, 256, 4), (
            f"{key} lost complete coalition/donor hybrid predictions."
        )
        model = study._model_from_state(fitted["models"][key])
        expected_game = deep_methods.coalition_values(
            model,
            t.from_numpy(explain["x"]),
            prepared["backgrounds"]["primary64"]["x_raw"],
            deep_methods.coalition_masks(8),
        )
        assert t.equal(item["baselines"]["coalition_values"], expected_game), (
            f"{key} retained hybrids do not average to the declared empirical game."
        )

    evaluated = study.evaluate_study(
        fitted,
        prepared,
        train,
        explain,
        audit,
        methods=methods,
        deep_methods=deep_methods,
    )
    assert evaluated["phase"] == "evaluated_after_audit", (
        "Evaluation lost the explicit post-audit phase."
    )
    assert evaluated["query_source_ids"].tolist() == [20, 21, 30, 31], (
        "All-row order must be explanation followed by audit."
    )
    assert evaluated["all_rows"]["clean_0"]["mc"]["samples"].shape == (4, 4, 8), (
        "All-row Expected Gradients samples were summarized or dropped."
    )
    assert set(evaluated["background_variants"]) == set(study.BACKGROUND_SPECS), (
        "All five finite background designs must be retained."
    )
    assert evaluated["noise_variants"]["sigma_0"]["samples"].data_ptr() == (
        evaluated["all_rows"]["clean_0"]["mc"]["samples"].data_ptr()
    ), "Sigma zero must alias the clean primary sample tensor."
    replicate = evaluated["sampling_replicates"]
    assert replicate["samples"].shape == (2, 3, 2, 4, 8), (
        "Replicate samples lost a replicate, model, row, draw or feature axis."
    )
    assert replicate["baseline_index"].shape == (2, 2, 4), (
        "Replicate schedules must be shared across the three models."
    )
    assert t.equal(replicate["row_split"], t.tensor([0, 1])), (
        "Replicate source IDs must retain their explanation/audit split boundary."
    )
    distance = evaluated["distance_diagnostic"]
    assert len(distance["path"]["distance"]) == 4 * 2, (
        "The tiny path-distance analog lost points."
    )
    assert len(distance["coalition"]["distance"]) == 4 * 2 * 256, (
        "The tiny coalition-distance analog lost hybrids."
    )
    random = evaluated["replacement_curves"]["random"]
    assert random["insertion"].shape == (7, 2, 100, 9), (
        "Every model, audit row and random replacement order must be retained."
    )


def test_evaluation_rejects_tampered_preparation_and_play_recomputes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fitted, train, explain, audit, _ = _fixture(monkeypatch)
    prepared = study.prepare_study(
        fitted, train, explain, methods=methods, deep_methods=deep_methods
    )
    changed = {**prepared, "explain_ids": prepared["explain_ids"].flip(0)}
    with pytest.raises(ValueError, match="Explanation identities"):
        study.evaluate_study(
            fitted, changed, train, explain, audit,
            methods=methods, deep_methods=deep_methods,
        )
    evaluated = study.evaluate_study(
        fitted, prepared, train, explain, audit,
        methods=methods, deep_methods=deep_methods,
    )
    play = study.native_play(
        fitted,
        prepared,
        evaluated,
        train,
        explain,
        audit,
        methods=methods,
        deep_methods=deep_methods,
        model_key="ridge",
        split="audit",
        row_index=0,
        background_key="nested16",
        samples=4,
        sigma=0.02,
    )
    assert play["selection"]["source_id"] == 30, "The play lost the chosen real row."
    assert play["mc_samples"].shape == (4, 8), "The play must retain every editable draw."
    assert set(play["local_replacement_curves"]) == {
        "eg", "quadrature_512", "exact", "deep", "gradient"
    }, "The play must compare all five local attribution orders."


def test_frozen_concrete_fit_reproduces_all_21_states() -> None:
    """Explicit post-freeze gate; excluded from the default no-fit semantic suite."""
    import os

    if os.environ.get("ARENA_GRADIENTSHAP_RUN_NATIVE_FIT") != "1":
        pytest.skip("set ARENA_GRADIENTSHAP_RUN_NATIVE_FIT=1 after protocol freeze")
    from chapter16_shapley_attribution_baselines.exercises.part15_gradientshap.gradientshap_data import (
        load_split,
    )

    manifest = study.load_manifest()
    fitted = study.fit_models(load_split("train"))
    result = study.verify_16_14_reproduction(fitted, manifest)
    assert result["status"] == "PASS" and len(result["model_state_hashes"]) == 7, (
        "The frozen fresh fit did not reproduce all seven known predictors."
    )

"""No-fit semantic tests for the complete interaction study."""

from __future__ import annotations

import inspect
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch as t

from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap import interaction_data as data
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap import interaction_methods as methods
from chapter16_shapley_attribution_baselines.exercises.part16_shapley_taylor_faith_shap import interaction_study as study


def test_manifest_freezes_scope_rng_and_ten_physical_shards() -> None:
    manifest = data.load_manifest()
    assert manifest["status"] == "PRE_FREEZE_NATIVE_UNRUN"
    assert manifest["native_fits_at_manifest"] == 0
    assert len(manifest["splits"]["explain"]) == 153
    assert len(manifest["splits"]["audit"]) == 119
    assert manifest["interaction_design_candidate"]["method_axis"] == [
        "sii",
        "stii",
        "fsii",
    ]
    rng = manifest["rng_contract"]
    assert rng["encoding"] == "UTF-8"
    assert rng["message_has_trailing_newline"] is False
    assert rng["seed_bytes"] == "first 16 digest bytes"
    assert len(rng["random_exact"]["games"]) == 15
    assert rng["within_size_shuffle"]["shape"] == [128, 119, 256]
    assert len(rng["random_pair_order"]["derived_seeds_hex"]) == 256
    shards = manifest["physical_hybrid_shards"]
    assert shards["count"] == len(shards["shards"]) == 10
    assert sum(item["stage"] == "prepared_before_audit" for item in shards["shards"]) == 5
    assert sum(item["stage"] == "evaluated_after_audit" for item in shards["shards"]) == 5
    assert len(shards["aliases"]) == 2
    assert shards["descriptor_schema"] == [
        "kind",
        "artifact",
        "manifest",
        "payload_path",
        "array_kind",
        "shape",
        "dtype",
        "elements",
        "value_sha256",
        "artifact_sha256",
        "size_bytes",
    ]
    assert shards["view_descriptor_schema"] == shards["descriptor_schema"] + [
        "selector",
        "target_payload_path",
    ]
    assert len(shards["logical_split_mappings"]) == 6
    label = manifest["fixed_exact_games"]["control_labels"]["mixed_degree2"]
    assert "not an additive zero-pair game" in label


def test_manifest_prospectively_maps_every_retained_payload_family() -> None:
    manifest = data.load_manifest()
    contract = manifest["retained_payload_contract"]
    assert contract["status"] == "PROSPECTIVE_PRE_NATIVE"
    assert contract["reference_result_instance_count"] == 22
    assert contract["reference_tensor_leaves_per_instance"] == 30
    paths = {item["path"] for item in contract["tensor_leaves"]}
    required = {
        "prepared.primary64.fsii_closed_form",
        "prepared.primary64.context_synergy",
        "evaluated.primary64_audit.residual",
        "evaluated.all_rows.primary64.kernel_mse",
        "evaluated.direct_full_context.lookup_difference",
        "evaluated.random_controls.within_size.permutation",
        "evaluated.random_controls.within_size.null.zero_total",
        "evaluated.background_sensitivity.independent64_b.pair_sign_change",
        "evaluated.support.nearest_train_source_id",
        "native_play.fixed_row.stii_coefficients",
    }
    assert required <= paths
    template = {item["path_suffix"] for item in contract["reference_result_tensor_template"]}
    assert {"inputs.game_values", "inputs.centered_game_scale"} <= template
    assert manifest["reference_scope_candidate"]["full_order_boundary_checks"] == 5
    assert manifest["reference_scope_candidate"]["n3_order3_boundary_rechecks_grid_case"] is True


def test_recipe_group_ids_are_stable_and_join_duplicate_recipes(monkeypatch) -> None:
    table = np.zeros((5, 9), dtype=np.float64)
    table[:, :7] = np.arange(5)[:, None]
    table[3, :7] = table[1, :7]
    table[:, 7] = np.arange(5)
    monkeypatch.setattr(data, "source_data", lambda: table.copy())
    groups = data.recipe_group_ids(np.array([3, 1, 4], dtype=np.int64))
    assert groups.dtype == np.int64
    assert groups[0] == groups[1]
    assert groups[2] != groups[1]


def test_background_order_is_deterministic_and_identity_sensitive() -> None:
    ids = np.array([40, 2, 17, 99], dtype=np.int64)
    first = data.background_order(ids, 161460)
    second = data.background_order(ids.copy(), 161460)
    assert np.array_equal(first, second)
    assert sorted(first.tolist()) == list(range(len(ids)))
    with pytest.raises(ValueError, match="unique"):
        data.background_order(np.array([1, 1], dtype=np.int64), 161460)


def test_fixed_games_match_dividends_and_exact_high_order_fractions() -> None:
    fixed = study.fixed_exact_games()
    explained = study.explain_complete_games(
        fixed["game_values"], methods=methods, n_players=4, order=2
    )
    t.testing.assert_close(explained["mobius"], fixed["mobius_oracle"], atol=1e-12, rtol=0)
    triple, four = 4, 5
    singletons = t.tensor([1, 2, 4, 8])
    t.testing.assert_close(
        explained["stii_coefficients"][triple, singletons],
        t.zeros(4, dtype=t.float64),
        atol=1e-12,
        rtol=0,
    )
    t.testing.assert_close(
        explained["fsii_coefficients"][triple, singletons[:3]],
        t.full((3,), -1 / 6, dtype=t.float64),
        atol=1e-12,
        rtol=0,
    )
    t.testing.assert_close(
        explained["stii_coefficients"][four, explained["pair_masks"]],
        t.full((6,), 1 / 6, dtype=t.float64),
        atol=1e-12,
        rtol=0,
    )
    t.testing.assert_close(
        explained["fsii_coefficients"][four, explained["pair_masks"]],
        t.full((6,), 3 / 10, dtype=t.float64),
        atol=1e-12,
        rtol=0,
    )
    t.testing.assert_close(
        explained["sii_pairs"][four],
        t.full((6,), 1 / 3, dtype=t.float64),
        atol=1e-12,
        rtol=0,
    )


def test_exact_suite_retains_five_separate_full_order_boundaries() -> None:
    suite = study._exact_suite(methods, reference=None)
    assert suite["counts"] == {
        "fixed_unique_games": 8,
        "random_unique_games": 15,
        "random_game_order_cases": 45,
        "random_stii_plus_fsii_cases": 90,
        "full_order_boundary_checks": 5,
        "boundary_checks_distinct_from_grid": 4,
    }
    for n_players in range(3, 8):
        boundary = suite["random"][n_players]["boundary_order_n"]
        assert boundary["replicate"] == 0
        assert boundary["order"] == n_players
        assert boundary["reuses_grid_case"] is (n_players == 3)
        t.testing.assert_close(
            boundary["explained"]["stii_coefficients"],
            boundary["explained"]["mobius"],
            atol=1e-10,
            rtol=1e-10,
        )


def test_reference_adapter_receives_flattened_leading_game_axes() -> None:
    seen: dict[str, tuple[int, ...]] = {}

    def reference_check(values, stii, fsii, sii, **kwargs):
        seen.update(
            values=tuple(values.shape),
            stii=tuple(stii.shape),
            fsii=tuple(fsii.shape),
            sii=tuple(sii.shape),
        )
        assert kwargs == {"n_players": 8, "order": 2, "expected_game_count": 6}
        return {"status": "PASS"}

    values = t.randn(2, 3, 256, dtype=t.float64)
    explained = study.explain_complete_games(
        values, methods=methods, include_contexts=False
    )
    result = study._reference_result(
        SimpleNamespace(reference_check=reference_check),
        explained,
        n_players=8,
        order=2,
    )
    assert result["status"] == "PASS"
    assert seen == {
        "values": (6, 256),
        "stii": (6, 256),
        "fsii": (6, 256),
        "sii": (6, 28),
    }


def test_random_exact_games_and_pair_orders_match_manifest_without_global_rng() -> None:
    np.random.seed(904)
    before = np.random.get_state()
    games = study.random_exact_games()
    orders = study.fixed_random_pair_orders()
    after = np.random.get_state()
    assert all(np.array_equal(left, right) for left, right in zip(before[1:], after[1:]))
    assert sum(len(item["game_values"]) for item in games.values()) == 15
    assert orders["orders"].shape == (256, 28)
    assert all(sorted(row.tolist()) == list(range(28)) for row in orders["orders"])


def test_pairwise_sii_degree_two_agrees_and_is_not_an_efficient_joint_vector() -> None:
    fixed = study.fixed_exact_games()
    degree_two = fixed["game_values"][6]
    sii = study.pairwise_sii(degree_two, n_players=4)
    pair_masks = t.tensor([3, 5, 9, 6, 10, 12])
    t.testing.assert_close(sii, fixed["mobius_oracle"][6, pair_masks], atol=1e-12, rtol=0)
    four = fixed["game_values"][5]
    shapley = t.full((4,), 1 / 4, dtype=t.float64)
    assert not t.isclose(shapley.sum() + study.pairwise_sii(four, n_players=4).sum(), four[-1])


def test_donor_game_retains_every_single_donor_prediction() -> None:
    model = t.nn.Sequential(t.nn.Linear(8, 1, dtype=t.float64))
    with t.no_grad():
        model[0].weight.copy_(t.arange(1, 9, dtype=t.float64)[None])
        model[0].bias.fill_(0.25)
    queries = t.tensor([[1.0] * 8, [2.0] * 8], dtype=t.float64)
    donors = t.tensor([[0.0] * 8, [-1.0] * 8, [3.0] * 8], dtype=t.float64)
    block = study.donor_replacement_games(
        model,
        queries,
        donors,
        methods=methods,
        donor_source_ids=t.tensor([10, 20, 30], dtype=t.int64),
    )
    assert block["hybrid_prediction"].shape == (2, 256, 3)
    t.testing.assert_close(
        block["coalition_values"], block["hybrid_prediction"].mean(-1), atol=0, rtol=0
    )
    t.testing.assert_close(
        block["hybrid_prediction"][:, -1],
        model(queries).squeeze(-1)[:, None].expand(-1, 3),
        atol=1e-12,
        rtol=0,
    )


def test_complete_game_evaluation_retains_shapes_and_additive_zero_pairs() -> None:
    model = t.nn.Sequential(t.nn.Linear(8, 1, dtype=t.float64))
    queries = t.randn(3, 8, dtype=t.float64)
    donors = t.randn(4, 8, dtype=t.float64)
    result = study.evaluate_model_games(
        model,
        queries,
        donors,
        methods=methods,
        donor_source_ids=t.arange(4, dtype=t.int64),
    )
    assert result["pair_values"].shape == (3, 3, 28)
    assert result["context_synergy"].shape == (3, 28, 64)
    assert result["reconstruction"].shape == (2, 3, 256)
    assert float(result["pair_values"].abs().max()) < 1e-10
    assert bool((result["kernel_mse"][1] <= result["kernel_mse"][0] + 1e-12).all())


def test_explanation_ranking_group_weights_and_keeps_all_pairs() -> None:
    values = t.zeros((7, 3, 3, 28), dtype=t.float64)
    values[:, 0, :, 0] = 100.0
    values[:, 1, :, 0] = 100.0
    values[:, 2, :, 1] = 150.0
    groups = t.tensor([0, 0, 1], dtype=t.int64)
    ranking = study.explanation_pair_rankings(values, groups)
    assert ranking["pair_order"].shape == (7, 3, 28)
    assert bool((ranking["pair_order"][..., 0] == 1).all())
    assert all(sorted(row.tolist()) == list(range(28)) for row in ranking["pair_order"].reshape(-1, 28))


def test_capture_curves_retain_random_distribution_and_zero_flags() -> None:
    orders = t.arange(28, dtype=t.int64).expand(7, 3, 28).clone()
    synergy = t.zeros((7, 4, 28), dtype=t.float64)
    synergy[0, :, 0] = 1.0
    random = study.fixed_random_pair_orders()["orders"]
    result = study.pair_capture_curves(
        orders, synergy, t.tensor([0, 1, 2, 3], dtype=t.int64), random
    )
    assert result["capture_curves"].shape == (7, 3, 29)
    assert result["random_capture_curves"].shape == (7, 256, 29)
    assert not bool(result["zero_denominator"][0])
    assert bool(result["zero_denominator"][1:].all())
    assert result["capture_curves"][0, 0, 1] == 1
    assert bool((result["capture_curves"][1:] == 0).all())


def test_within_size_shuffle_preserves_endpoints_and_cardinality_multisets() -> None:
    values = t.stack(
        [t.arange(256, dtype=t.float64), t.arange(256, dtype=t.float64).square()]
    )
    observed = study.explain_complete_games(values, methods=methods)["pair_values"]
    result = study.within_size_shuffle_control(
        values,
        observed,
        t.tensor([9001, 9002], dtype=t.int64),
        t.tensor([0, 1], dtype=t.int64),
        methods=methods,
    )
    permutation = result["permutation"]
    assert permutation.shape == (128, 2, 256)
    assert bool((permutation[..., 0] == 0).all())
    assert bool((permutation[..., -1] == 255).all())
    sizes = study.coalition_masks().sum(-1)
    for size in range(1, 8):
        positions = t.nonzero(sizes == size).flatten().sort().values
        shuffled_positions = permutation[0, 0, positions].to(t.int64).sort().values
        assert t.equal(positions, shuffled_positions)
    assert result["pair_values"].shape == (128, 2, 3, 28)
    assert "not expected to have zero" in result["interpretation"]


def test_full_within_size_schedule_matches_manifest_hash_and_nested_seeds() -> None:
    manifest = data.load_manifest()
    audit_ids = t.tensor(manifest["splits"]["audit"], dtype=t.int64)
    schedule = study._within_size_permutations(audit_ids)
    assert schedule["permutation"].shape == (128, 119, 256)
    assert len(schedule["derived_seeds_hex"]) == 128
    assert all(len(row) == 119 for row in schedule["derived_seeds_hex"])


def test_nearest_training_tie_breaks_by_smallest_source_id() -> None:
    train = t.stack([t.arange(8, dtype=t.float64) + index for index in range(10)])
    train[1] = train[0]
    train_ids = t.tensor([5, 3, 20, 21, 22, 23, 24, 25, 26, 27], dtype=t.int64)
    mean = train.mean(0)
    std = train.std(0, correction=0)
    donors = train[:8]
    result = study.nearest_training_diagnostics(
        train,
        train_ids,
        train[0:1],
        donors,
        {"x_mean": mean, "x_std": std},
    )
    assert result["nearest_train_source_id"].shape == (1, 256, 8)
    assert int(result["nearest_train_source_id"][0, -1, 0]) == 3
    assert float(result["nearest_train_distance"][0, -1, 0]) == 0.0


def test_nearest_training_accumulates_squared_distance_in_feature_order() -> None:
    # The first square is 1e16, so each subsequent unit is rounded away by
    # feature-order addition. A tensor reduction groups the seven units and
    # returns a different, still representable square root on this fixture.
    query = t.tensor([[1e8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]], dtype=t.float64)
    assert float(query.square().sum(-1).sqrt()[0]) != 1e8
    result = study.nearest_training_diagnostics(
        t.zeros((1, 8), dtype=t.float64),
        t.tensor([17], dtype=t.int64),
        query,
        t.zeros((8, 8), dtype=t.float64),
        {"x_mean": t.zeros(8, dtype=t.float64), "x_std": t.ones(8, dtype=t.float64)},
    )
    assert t.equal(result["nearest_train_distance"][0, -1], t.full((8,), 1e8, dtype=t.float64))
    assert t.equal(result["nearest_train_source_id"][0, -1], t.full((8,), 17, dtype=t.int64))


def test_split_hybrid_combination_retains_physical_tensors_without_concat() -> None:
    shared = {
        "donor_source_ids": t.tensor([10, 11], dtype=t.int64),
        "pair_indices": study.pair_indices(),
        "pair_masks": (1 << study.pair_indices()[:, 0]) | (1 << study.pair_indices()[:, 1]),
        "context_masks": t.zeros((28, 64), dtype=t.int64),
    }
    explanation = {
        **shared,
        "hybrid_prediction": t.zeros((7, 2, 256, 2), dtype=t.float64),
        "coalition_values": t.zeros((7, 2, 256), dtype=t.float64),
        "reconstruction": t.zeros((2, 7, 2, 256), dtype=t.float64),
        "model_keys": study.MODEL_KEYS,
    }
    audit = {
        **shared,
        "hybrid_prediction": t.ones((7, 1, 256, 2), dtype=t.float64),
        "coalition_values": t.ones((7, 1, 256), dtype=t.float64),
        "reconstruction": t.ones((2, 7, 1, 256), dtype=t.float64),
        "model_keys": study.MODEL_KEYS,
    }
    combined = study._combine_game_blocks(explanation, audit, model_axis=True)
    assert combined["hybrid_prediction"]["logical_shape"] == (7, 3, 256, 2)
    assert combined["hybrid_prediction"]["explain"] is explanation["hybrid_prediction"]
    assert combined["hybrid_prediction"]["audit"] is audit["hybrid_prediction"]
    assert combined["coalition_values"].shape == (7, 3, 256)
    assert combined["reconstruction"].shape == (2, 7, 3, 256)


def test_exact_and_empirical_plays_recompute_visible_methods() -> None:
    exact = study.exact_dividend_play(methods=methods)
    assert exact["kind"] == "exact_dividend_play"
    assert exact["explained"]["pair_values"].shape == (3, 6)

    model = t.nn.Sequential(t.nn.Linear(8, 1, dtype=t.float64))
    donors = t.randn(2, 8, dtype=t.float64)
    source_ids = t.tensor([0, 1], dtype=t.int64)
    donor_ids = t.tensor([173, 174], dtype=t.int64)
    one = study.evaluate_model_games(
        model, t.randn(1, 8, dtype=t.float64), donors, methods=methods, donor_source_ids=donor_ids
    )
    blocks = {key: one for key in study.MODEL_KEYS}
    primary_explain = study._stack_model_blocks(blocks)
    primary_audit = study._stack_model_blocks(blocks)
    primary = study._combine_game_blocks(primary_explain, primary_audit, model_axis=True)
    evaluated = {
        "phase": "evaluated_after_audit",
        "feature_names": study.FEATURE_NAMES,
        "feature_units": study.FEATURE_UNITS,
        "target_name": study.TARGET_NAME,
        "target_unit": study.TARGET_UNIT,
        "model_keys": study.MODEL_KEYS,
        "method_keys": study.METHOD_KEYS,
        "source_ids": source_ids,
        "row_split": t.tensor([0, 1], dtype=t.int64),
        "all_rows": {"primary64": primary, "background_variants": {}},
        "performance": {},
        "rankings": {},
        "pair_capture": {},
        "background_sensitivity": {},
        "direct_full_context": {},
    }
    play = study.empirical_context_play(evaluated, methods)
    assert play["source_id"] == 0
    assert play["four_values_by_donor"].shape == (4, 2)
    assert play["pair_values"].shape == (3,)
    assert play["target_unit"] == "MPa"
    assert play["interprets_model_input_intervention_only"] is True
    native = study.native_play(evaluated, methods=methods)
    assert native["fixed_row"]["model_key"] == "clean_0"
    assert native["fixed_row"]["stii_coefficients"].shape == (256,)
    assert native["fixed_row"]["fsii_coefficients"].shape == (256,)
    assert native["fixed_row"]["sii_pairs"].shape == (28,)
    assert native["fixed_row"]["ridge_pair_max_abs"].ndim == 0
    # A retained 256-value play must not keep the entire model/row/donor bank alive.
    for name in ("empirical_background_average", "empirical_fixed_donor"):
        game = native[name]["game_values"]
        assert game.untyped_storage().nbytes() == game.numel() * game.element_size()
    t.testing.assert_close(native["exact_dividend"]["mobius"], exact["mobius_oracle"], rtol=0, atol=0)


def test_prepare_signature_excludes_audit_and_phase_is_literal() -> None:
    signature = inspect.signature(study.prepare_study)
    assert "audit" not in signature.parameters
    assert list(signature.parameters)[:3] == ["fitted", "train", "explain"]
    source = inspect.getsource(study.prepare_study)
    assert '"prepared_before_audit"' in source
    assert '"audit_used_for_selection": False' in source


@pytest.mark.skipif(
    os.environ.get("ARENA_RUN_16_16_NATIVE_FIT") != "1",
    reason="Explicit post-freeze full-fit validation only.",
)
def test_optional_post_freeze_full_known_fit_and_21_hash_gate() -> None:
    train = data.load_split("train")
    fitted = study.fit_known_models(train)
    evidence = study.verify_known_reproduction(fitted, data.load_manifest())
    assert evidence["status"] == "PASS"
    assert evidence["state_comparisons"] == 21
    assert evidence["mismatches"] == ()


def test_every_frozen_exact_case_passes_the_complete_reference_validator():
    """Exercise the actual 58-case plan, including n=7 full-order interpolation."""
    from . import interaction_reference as reference

    exact = study._exact_suite(methods, reference)
    checked = reference.validate_exact_reference_suite({
        "phase": "prepared_before_audit", "exact_games": exact
    })
    assert checked["status"] == "PASS"

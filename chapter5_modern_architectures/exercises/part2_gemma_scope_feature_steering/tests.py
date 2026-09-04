from collections.abc import Callable

import torch as t

from arena_ext import features as reference


def _solutions():
    from chapter5_modern_architectures.exercises.part2_gemma_scope_feature_steering import (
        solutions,
    )

    return solutions


def test_feature_density_l0_and_dead_fraction_match_reference(
    feature_density: Callable | None = None,
    l0: Callable | None = None,
    dead_feature_fraction: Callable | None = None,
):
    solutions = _solutions()
    feature_density = feature_density or solutions.feature_density
    l0 = l0 or solutions.l0
    dead_feature_fraction = dead_feature_fraction or solutions.dead_feature_fraction
    feature_acts = t.tensor(
        [
            [[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]],
            [[4.0, 0.0, 0.0], [0.0, 0.0, 5.0]],
        ]
    )
    t.testing.assert_close(
        feature_density(feature_acts),
        reference.feature_density(feature_acts),
        msg="Feature density should be the per-feature firing rate over batch/position axes.",
    )
    assert l0(feature_acts) == reference.l0(feature_acts), (
        "L0 should be the average number of active sparse features per activation vector."
    )
    assert dead_feature_fraction(feature_acts) == reference.dead_feature_fraction(feature_acts), (
        "Dead-feature fraction should count features that never fire above threshold."
    )
    print("All tests in `test_feature_density_l0_and_dead_fraction_match_reference` passed!")


def test_compute_sae_reconstruction_metrics_matches_manual_and_reference(
    compute_sae_reconstruction_metrics: Callable | None = None,
):
    solutions = _solutions()
    compute_sae_reconstruction_metrics = (
        compute_sae_reconstruction_metrics or solutions.compute_sae_reconstruction_metrics
    )
    activations = t.tensor([[[1.0, -1.0], [2.0, 0.0]]])
    reconstructed = activations + t.tensor([[[0.5, -0.5], [0.0, 1.0]]])
    feature_acts = t.tensor([[[1.0, 0.0, 2.0], [0.0, 0.0, 3.0]]])
    reference_logits = t.tensor([[[2.0, 0.0, -1.0]]])
    reconstructed_logits = t.tensor([[[1.5, 0.1, -0.7]]])
    actual = compute_sae_reconstruction_metrics(
        activations=activations,
        reconstructed_activations=reconstructed,
        feature_acts=feature_acts,
        reference_logits=reference_logits,
        reconstructed_logits=reconstructed_logits,
        clean_loss=1.0,
        reconstructed_loss=1.25,
        zero_ablation_loss=2.0,
    )
    expected = reference.compute_sae_reconstruction_metrics(
        activations=activations,
        reconstructed_activations=reconstructed,
        feature_acts=feature_acts,
        reference_logits=reference_logits,
        reconstructed_logits=reconstructed_logits,
        clean_loss=1.0,
        reconstructed_loss=1.25,
        zero_ablation_loss=2.0,
    )
    manual_mse = ((reconstructed - activations) ** 2).mean().item()
    assert actual.reconstruction_mse == manual_mse, (
        "Reconstruction MSE should be the mean squared error over activation elements."
    )
    assert actual.loss_recovered == 0.75, (
        "Loss recovered should be (zero_ablation_loss - reconstructed_loss) / "
        "(zero_ablation_loss - clean_loss)."
    )
    t.testing.assert_close(
        t.tensor(list(actual.__dict__.values())),
        t.tensor(list(expected.__dict__.values())),
        msg="All SAE reconstruction metrics should match the independent reference.",
    )
    print("All tests in `test_compute_sae_reconstruction_metrics_matches_manual_and_reference` passed!")


def test_direct_logit_attribution_projects_decoder_vectors(
    direct_logit_attribution: Callable | None = None,
):
    solutions = _solutions()
    direct_logit_attribution = direct_logit_attribution or solutions.direct_logit_attribution
    decoder_vectors = t.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0]])
    unembedding = t.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    actual = direct_logit_attribution(decoder_vectors, unembedding, token_ids=[0, 2])
    expected = t.tensor([[1.0, 3.0], [4.0, 6.0], [-2.0, 0.0]])
    t.testing.assert_close(
        actual,
        expected,
        msg="DLA should compute decoder_vector @ W_U and then select requested tokens.",
    )
    print("All tests in `test_direct_logit_attribution_projects_decoder_vectors` passed!")


def test_feature_detection_report_handles_heldout_controls(
    feature_detection_report: Callable | None = None,
    roc_auc_binary: Callable | None = None,
):
    solutions = _solutions()
    feature_detection_report = feature_detection_report or solutions.feature_detection_report
    roc_auc_binary = roc_auc_binary or solutions.roc_auc_binary
    scores = t.tensor([0.9, 0.8, 0.2, 0.2])
    labels = t.tensor([1, 1, 0, 0], dtype=t.bool)
    report = feature_detection_report(scores, labels)
    assert report.auc == 1.0, (
        f"Perfectly separated positives and negatives should have AUC 1.0, got {report.auc}."
    )
    assert report.threshold_accuracy == 1.0, (
        "Default midpoint threshold should classify this held-out control perfectly."
    )
    tied_auc = roc_auc_binary(
        t.tensor([0.5, 0.5, 0.0, 1.0]),
        t.tensor([1, 0, 0, 1], dtype=t.bool),
    )
    assert tied_auc == reference.roc_auc_binary(
        t.tensor([0.5, 0.5, 0.0, 1.0]),
        t.tensor([1, 0, 0, 1], dtype=t.bool),
    ), "AUC should use average ranks for tied scores."
    print("All tests in `test_feature_detection_report_handles_heldout_controls` passed!")


def test_top_activating_examples_and_ablation_match_reference(
    top_activating_examples: Callable | None = None,
    ablate_features: Callable | None = None,
):
    solutions = _solutions()
    top_activating_examples = top_activating_examples or solutions.top_activating_examples
    ablate_features = ablate_features or solutions.ablate_features
    _, _, feature_acts = solutions.synthetic_feature_batch()
    indices, values = top_activating_examples(feature_acts, feature_id=3, k=3)
    reference_indices, reference_values = reference.top_activating_examples(
        feature_acts,
        feature_id=3,
        k=3,
    )
    t.testing.assert_close(
        indices,
        reference_indices,
        msg="Top activating example indices should match flattened top-k positions.",
    )
    t.testing.assert_close(
        values,
        reference_values,
        msg="Top activating example values should match flattened top-k scores.",
    )
    zeroed = ablate_features(feature_acts, [3], replacement="zero")
    meaned = ablate_features(feature_acts, [3], replacement="mean")
    assert bool(zeroed[..., 3].eq(0).all().item()), (
        "Zero ablation should set every selected feature activation to zero."
    )
    t.testing.assert_close(
        meaned[..., 3],
        reference.ablate_features(feature_acts, [3], replacement="mean")[..., 3],
        msg="Mean ablation should replace selected features with their global mean.",
    )
    print("All tests in `test_top_activating_examples_and_ablation_match_reference` passed!")


def test_decoder_steering_and_random_control_report(
    apply_decoder_steering: Callable | None = None,
    steering_comparison_report: Callable | None = None,
):
    solutions = _solutions()
    apply_decoder_steering = apply_decoder_steering or solutions.apply_decoder_steering
    steering_comparison_report = (
        steering_comparison_report or solutions.steering_comparison_report
    )
    activations = t.zeros(2, 3, 4)
    decoder_vectors = t.eye(4)
    steered_last = apply_decoder_steering(activations, decoder_vectors, [1], 2.0)
    steered_all = apply_decoder_steering(
        activations,
        decoder_vectors,
        [0, 2],
        [1.0, -1.0],
        positions="all",
    )
    assert bool(steered_last[:, -1, 1].eq(2.0).all().item()), (
        "Last-position steering should add the selected decoder direction only at the last token."
    )
    assert bool(steered_last[:, :-1, :].eq(0).all().item()), (
        "Last-position steering should leave earlier positions unchanged."
    )
    assert bool(steered_all[..., 0].eq(1.0).all().item()), (
        "All-position steering should broadcast the first selected direction to every token."
    )
    assert bool(steered_all[..., 2].eq(-1.0).all().item()), (
        "All-position steering should apply the signed coefficient for each feature."
    )
    report = steering_comparison_report(
        baseline_scores=t.tensor([0.1, 0.2, 0.3]),
        steered_scores=t.tensor([0.5, 0.6, 0.7]),
        random_control_scores=t.tensor([0.2, 0.2, 0.3]),
    )
    assert report.passes_control, (
        "Feature steering should pass only when its score delta beats the random-control delta."
    )
    print("All tests in `test_decoder_steering_and_random_control_report` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    if run_smoke_test is None:
        run_smoke_test = _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["metrics"]["l0"] > 0, "Smoke metrics should report nonzero sparse activity."
    assert result["feature_validation"]["auc"] == 1.0, (
        "Smoke validation should include a perfectly separated held-out feature control."
    )
    assert result["logit_attribution"] == [[1.0, 3.0], [4.0, 6.0]], (
        "Smoke DLA should return the expected selected-token effects."
    )
    assert result["steering"]["passes_control"], (
        "Smoke steering should beat the random-feature control."
    )
    assert result["ablation"]["feature_3_zeroed"], (
        "Smoke ablation should zero the selected feature."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_jumprelu_encode_decode(
    jumprelu_encode: Callable,
    sae_decode: Callable,
):
    """Check the exact JumpReLU equations, including the strict threshold boundary."""

    w_dec = t.eye(3)
    w_enc = w_dec.T
    b_dec = t.tensor([0.25, -0.5, 1.0])
    b_enc = t.zeros(3)
    threshold = t.tensor([0.5, 0.5, 0.5])
    feature_acts = t.tensor([[1.0, 0.0, 2.0], [0.0, 1.5, 0.0]])
    residual = feature_acts @ w_dec + b_dec

    encoded = jumprelu_encode(
        residual,
        w_enc=w_enc,
        b_enc=b_enc,
        b_dec=b_dec,
        threshold=threshold,
    )
    reconstructed = sae_decode(encoded, w_dec=w_dec, b_dec=b_dec)
    t.testing.assert_close(encoded, feature_acts)
    t.testing.assert_close(reconstructed, residual)

    boundary = b_dec + t.tensor([0.5, 0.0, 0.0])
    boundary_encoded = jumprelu_encode(
        boundary,
        w_enc=w_enc,
        b_enc=b_enc,
        b_dec=b_dec,
        threshold=threshold,
    )
    assert boundary_encoded[0].item() == 0.0, (
        "JumpReLU must use pre_activation > threshold, so equality stays inactive."
    )
    print("All tests in `test_jumprelu_encode_decode` passed!")


def test_feature_discovery_and_heldout_scoring(
    select_feature_by_mean_difference: Callable,
    binary_roc_auc: Callable,
    topk_example_indices: Callable,
):
    """Check that selection uses discovery rows and evaluation handles ties."""

    scores = t.tensor(
        [
            [1.2, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.2, 1.0],
            [0.0, 1.0, 0.0],
            [1.3, 0.0, 1.0],
            [1.1, 0.0, 0.0],
            [0.0, 1.3, 1.0],
            [0.0, 1.1, 0.0],
        ]
    )
    labels = t.tensor([1, 1, 0, 0, 1, 1, 0, 0], dtype=t.bool)
    discovery = t.tensor([1, 1, 1, 1, 0, 0, 0, 0], dtype=t.bool)
    selected = select_feature_by_mean_difference(scores, labels, discovery)
    assert selected == 0, "The target feature must be selected from discovery rows only."
    target_auc = binary_roc_auc(scores[~discovery, selected], labels[~discovery])
    nuisance_auc = binary_roc_auc(scores[~discovery, 2], labels[~discovery])
    top_indices = topk_example_indices(scores[~discovery, selected], k=2).tolist()
    assert target_auc == 1.0, (
        "The discovery-selected feature should rank every positive held-out example above "
        f"every negative example; got AUC {target_auc:.3f}."
    )
    assert nuisance_auc == 0.5, (
        "The tied nuisance feature should score at chance under tie-aware AUC; "
        f"got {nuisance_auc:.3f}."
    )
    assert top_indices == [0, 1], (
        "Top-k inspection should return the two highest-scoring held-out rows in descending "
        f"order; got {top_indices}."
    )
    print("All tests in `test_feature_discovery_and_heldout_scoring` passed!")


def test_feature_steering_and_ablation(
    steer_residual: Callable,
    ablate_feature: Callable,
    direction_score: Callable,
):
    """Check additive steering and contribution-preserving progressive ablation."""

    decoder = t.eye(4)
    residual = t.tensor([[1.5, 0.5, 0.0, 0.0], [2.0, 0.0, 1.0, 0.0]])
    feature_acts = residual.clone()
    target_direction = decoder[0]

    steered = steer_residual(residual, decoder[0], coefficient=2.0)
    t.testing.assert_close(
        direction_score(steered, target_direction) - direction_score(residual, target_direction),
        t.full((2,), 2.0),
    )
    target_ablated = ablate_feature(
        residual,
        feature_acts,
        decoder,
        feature_id=0,
        fraction=1.0,
    )
    random_ablated = ablate_feature(
        residual,
        feature_acts,
        decoder,
        feature_id=1,
        fraction=1.0,
    )
    t.testing.assert_close(direction_score(target_ablated, target_direction), t.zeros(2))
    t.testing.assert_close(
        direction_score(random_ablated, target_direction),
        direction_score(residual, target_direction),
    )
    print("All tests in `test_feature_steering_and_ablation` passed!")


def test_released_gemma_scope_artifact(
    config: dict,
    state: dict[str, t.Tensor],
    jumprelu_encode: Callable,
    sae_decode: Callable,
):
    """Validate the pinned released artifact and one genuine CPU encode/decode pass."""

    assert config["model_name"] == "google/gemma-3-1b-it", (
        "The SAE must be paired with its declared base model, google/gemma-3-1b-it; "
        f"got {config.get('model_name')!r}."
    )
    assert config["hf_hook_point_in"] == "model.layers.13.output", (
        "The SAE expects layer-13 residual outputs; steering another hook point would make "
        f"the feature coordinates invalid. Got {config.get('hf_hook_point_in')!r}."
    )
    assert config["architecture"] == "jump_relu", (
        "This lesson implements the JumpReLU threshold contract, so the released artifact "
        f"must declare architecture='jump_relu'; got {config.get('architecture')!r}."
    )
    assert config["width"] == 16384, (
        "The pinned SAE should expose 16,384 features; a different width means the config "
        f"and tensor checkpoint do not match. Got {config.get('width')!r}."
    )
    expected_shapes = {
        "w_enc": (1152, 16384),
        "w_dec": (16384, 1152),
        "b_enc": (16384,),
        "b_dec": (1152,),
        "threshold": (16384,),
    }
    missing_tensors = sorted(set(expected_shapes) - set(state))
    assert not missing_tensors, (
        "The released SAE state is missing tensors required for encode/decode: "
        f"{missing_tensors}."
    )
    for name, shape in expected_shapes.items():
        assert tuple(state[name].shape) == shape, (
            f"SAE tensor {name!r} should have shape {shape}; got {tuple(state[name].shape)}."
        )
        assert bool(t.isfinite(state[name]).all().item()), (
            f"SAE tensor {name!r} contains NaN or infinite values, so feature scores are invalid."
        )

    candidate_mask = (state["threshold"] > state["b_enc"]) & (
        state["w_enc"].square().sum(dim=0) > 1e-8
    )
    feature_id = int(t.nonzero(candidate_mask, as_tuple=False)[0].item())
    column = state["w_enc"][:, feature_id]
    scale = (
        state["threshold"][feature_id] - state["b_enc"][feature_id] + 1.0
    ) / column.square().sum().clamp_min(1e-8)
    residual = state["b_dec"] + scale * column
    feature_acts = jumprelu_encode(
        residual[None],
        w_enc=state["w_enc"],
        b_enc=state["b_enc"],
        b_dec=state["b_dec"],
        threshold=state["threshold"],
    )
    reconstructed = sae_decode(feature_acts, w_dec=state["w_dec"], b_dec=state["b_dec"])
    assert feature_acts[0, feature_id].item() > 0, (
        "The constructed residual should cross the chosen feature's JumpReLU threshold; "
        "check the encoder orientation, centering term, and strict threshold comparison."
    )
    assert bool(t.isfinite(reconstructed).all().item()), (
        "A finite SAE input should decode to a finite residual; inspect decoder orientation "
        "and bias broadcasting."
    )
    print("All tests in `test_released_gemma_scope_artifact` passed!")


# These learner-facing checks receive notebook functions as arguments; they are
# invoked directly from notebook cells rather than collected as pytest fixtures.
test_jumprelu_encode_decode.__test__ = False
test_feature_discovery_and_heldout_scoring.__test__ = False
test_feature_steering_and_ablation.__test__ = False
test_released_gemma_scope_artifact.__test__ = False

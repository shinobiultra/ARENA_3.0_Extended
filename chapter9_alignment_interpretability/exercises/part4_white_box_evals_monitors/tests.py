from collections.abc import Callable
import ast
import json
from pathlib import Path

import torch as t

from arena_ext import white_box_monitors as reference


TAUGHT_FUNCTIONS = (
    "activation_matrix",
    "split_train_heldout",
    "fit_white_box_monitor",
    "score_white_box_monitor",
    "binary_auroc",
    "threshold_sweep",
    "midpoint_threshold",
    "predict_from_scores",
    "surface_risk_scores",
    "white_box_only_catches",
    "active_features",
    "build_dashboard_entries",
    "shuffled_label_control_scores",
    "random_direction_control_scores",
    "perturb_activation",
)


def _solutions():
    from chapter9_alignment_interpretability.exercises.part4_white_box_evals_monitors import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _toy_inputs():
    records = reference.build_monitor_records()
    labels = reference.ground_truth_failure_labels(records)
    return records, labels


def test_activation_matrix_recovers_planted_features(
    activation_matrix: Callable | None = None,
):
    activation_matrix = activation_matrix or _solutions().activation_matrix
    records, _labels = _toy_inputs()
    activations = activation_matrix(records)

    assert activations.shape == (54, 6), "The exact organism has 54 records and six named features."
    assert activations.dtype == t.float32 and t.isfinite(activations).all(), (
        "Feature activations should be finite float32 values so later score comparisons are meaningful."
    )
    t.testing.assert_close(
        activations[0],
        t.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        atol=0,
        rtol=0,
    )
    t.testing.assert_close(
        activations[5],
        t.tensor([0.90, 0.10, 0.10, 0.0, 0.0, 1.45]),
        atol=0,
        rtol=0,
    )
    t.testing.assert_close(
        activations[6, 0] - activations[0, 0],
        t.tensor(0.01),
        atol=1e-7,
        rtol=1e-6,
        msg="Only the helpful-answer coordinate should carry the deterministic context offset.",
    )
    print("All tests in `test_activation_matrix_recovers_planted_features` passed!")


def test_split_train_heldout_preserves_context_boundaries(
    split_train_heldout: Callable | None = None,
    activation_matrix: Callable | None = None,
):
    solutions = _solutions()
    split_train_heldout = split_train_heldout or solutions.split_train_heldout
    activation_matrix = activation_matrix or solutions.activation_matrix
    records, labels = _toy_inputs()
    activations = activation_matrix(records)
    train_x, train_y, heldout_x, heldout_y, train_records, heldout_records = split_train_heldout(
        records, activations, labels
    )

    assert train_x.shape == (30, 6) and heldout_x.shape == (24, 6), (
        "The context split should contain 30 training rows and 24 held-out rows with six features each."
    )
    assert train_y.shape == (30,) and heldout_y.shape == (24,), (
        "Train and held-out labels should stay aligned one-for-one with their activation rows."
    )
    assert {record.context for record in train_records}.isdisjoint(
        {record.context for record in heldout_records}
    ), "Held-out contexts must not leak into monitor fitting."
    t.testing.assert_close(train_x, activations[:30])
    t.testing.assert_close(heldout_x, activations[30:])
    print("All tests in `test_split_train_heldout_preserves_context_boundaries` passed!")


def test_fit_white_box_monitor_uses_mean_difference(
    fit_white_box_monitor: Callable | None = None,
    activation_matrix: Callable | None = None,
    split_train_heldout: Callable | None = None,
):
    solutions = _solutions()
    fit_white_box_monitor = fit_white_box_monitor or solutions.fit_white_box_monitor
    activation_matrix = activation_matrix or solutions.activation_matrix
    split_train_heldout = split_train_heldout or solutions.split_train_heldout
    records, labels = _toy_inputs()
    train_x, train_y, *_ = split_train_heldout(records, activation_matrix(records), labels)
    monitor = fit_white_box_monitor(train_x, train_y)

    expected_direction = train_x[train_y].mean(0) - train_x[~train_y].mean(0)
    expected_direction = expected_direction / expected_direction.norm()
    t.testing.assert_close(monitor.direction, expected_direction, atol=1e-6, rtol=1e-6)
    assert abs(float(monitor.direction.norm()) - 1.0) < 1e-6, (
        "Normalize the mean-difference direction before fitting or comparing score thresholds."
    )
    assert monitor.direction[-1] > 0.4, "The planted CoT-unfaithfulness feature must affect the monitor."
    assert monitor.direction[0] < 0, "Loss of the helpful feature is part of the planted failure signal."

    try:
        fit_white_box_monitor(t.zeros((4, 6)), t.tensor([0, 0, 1, 1], dtype=t.bool))
    except ValueError as exc:
        assert "nonzero" in str(exc), (
            "A collapsed class-mean difference should explain that a nonzero direction is required."
        )
    else:
        raise AssertionError("A zero mean-difference direction must be rejected.")
    print("All tests in `test_fit_white_box_monitor_uses_mean_difference` passed!")


def test_scoring_and_auroc_match_exact_ranking(
    score_white_box_monitor: Callable | None = None,
    binary_auroc: Callable | None = None,
):
    solutions = _solutions()
    score_white_box_monitor = score_white_box_monitor or solutions.score_white_box_monitor
    binary_auroc = binary_auroc or solutions.binary_auroc
    result = reference.run_toy_monitor_experiment()
    scores = score_white_box_monitor(result["monitor"], result["heldout_activations"])
    t.testing.assert_close(scores, result["white_box_scores"], atol=1e-6, rtol=1e-6)
    assert binary_auroc(scores, result["heldout_labels"]) == 1.0, (
        "The fitted monitor should rank every planted held-out failure above every clean state."
    )
    assert abs(binary_auroc(t.tensor([0.2, 0.5, 0.5, 0.8]), t.tensor([0, 0, 1, 1])) - 0.875) < 1e-6, (
        "AUROC should count a tied positive-negative pair as half a successful ranking."
    )

    try:
        binary_auroc(t.tensor([0.1, float("nan")]), t.tensor([0, 1]))
    except ValueError as exc:
        assert "finite" in str(exc), "Non-finite AUROC inputs should produce an explanatory error."
    else:
        raise AssertionError("AUROC must reject non-finite scores.")
    print("All tests in `test_scoring_and_auroc_match_exact_ranking` passed!")


def test_threshold_sweep_and_midpoint_are_semantic(
    threshold_sweep: Callable | None = None,
    midpoint_threshold: Callable | None = None,
    predict_from_scores: Callable | None = None,
):
    solutions = _solutions()
    threshold_sweep = threshold_sweep or solutions.threshold_sweep
    midpoint_threshold = midpoint_threshold or solutions.midpoint_threshold
    predict_from_scores = predict_from_scores or solutions.predict_from_scores
    scores = t.tensor([-0.6, -0.2, 0.2, 0.8])
    labels = t.tensor([0, 0, 1, 1], dtype=t.bool)
    thresholds = t.tensor([-1.0, 0.0, 1.0])
    curve = threshold_sweep(scores, labels, thresholds)

    t.testing.assert_close(curve["true_positive_rate"], t.tensor([1.0, 1.0, 0.0]))
    t.testing.assert_close(curve["false_positive_rate"], t.tensor([1.0, 0.0, 0.0]))
    t.testing.assert_close(curve["accuracy"], t.tensor([0.5, 1.0, 0.5]))
    threshold = midpoint_threshold(scores, labels)
    assert abs(threshold - 0.05) < 1e-6, (
        "The midpoint threshold should lie halfway between the positive and negative score means."
    )
    assert predict_from_scores(scores, threshold).tolist() == [False, False, True, True], (
        "Threshold predictions should classify the two negative and two positive examples correctly."
    )
    print("All tests in `test_threshold_sweep_and_midpoint_are_semantic` passed!")


def test_surface_baseline_misses_hidden_cot_failures(
    surface_risk_scores: Callable | None = None,
    white_box_only_catches: Callable | None = None,
    binary_auroc: Callable | None = None,
):
    solutions = _solutions()
    surface_risk_scores = surface_risk_scores or solutions.surface_risk_scores
    white_box_only_catches = white_box_only_catches or solutions.white_box_only_catches
    binary_auroc = binary_auroc or solutions.binary_auroc
    result = reference.run_toy_monitor_experiment()
    records = result["heldout_records"]
    labels = result["heldout_labels"]
    black_scores = surface_risk_scores(records)
    caught = white_box_only_catches(
        result["white_box_predictions"],
        result["black_box_predictions"],
        labels,
    )

    assert abs(binary_auroc(black_scores, labels) - 0.8) < 1e-6, (
        "The declared output-only baseline should retain its exact held-out AUROC of 0.8."
    )
    assert caught == (5, 11, 17, 23), (
        "The white-box monitor should add exactly one hidden catch in each held-out context."
    )
    assert {records[index].failure_kind for index in caught} == {"cot_unfaithful"}, (
        "Only the planted CoT-unfaithfulness states should require internal evidence to catch."
    )
    assert all(
        records[index].model_output == reference.OUTPUT_TEMPLATES["clean"] for index in caught
    ), "Each white-box-only catch should be surface-identical to the clean output."
    print("All tests in `test_surface_baseline_misses_hidden_cot_failures` passed!")


def test_dashboard_entries_keep_behavior_and_internal_evidence(
    active_features: Callable | None = None,
    build_dashboard_entries: Callable | None = None,
):
    solutions = _solutions()
    active_features = active_features or solutions.active_features
    build_dashboard_entries = build_dashboard_entries or solutions.build_dashboard_entries
    result = reference.run_toy_monitor_experiment()
    rows = build_dashboard_entries(
        result["heldout_records"],
        result["heldout_activations"],
        result["white_box_scores"],
        result["black_box_scores"],
        result["white_box_predictions"],
        result["black_box_predictions"],
        result["heldout_labels"],
    )

    assert len(rows) == 24, "The dashboard should retain every held-out record for auditability."
    cot_row = rows[5]
    assert cot_row.prompt and cot_row.model_output, (
        "A dashboard row should retain both the evaluated prompt and visible model output."
    )
    assert active_features(result["heldout_activations"][5]) == (
        "helpful_answer",
        "cot_unfaithful",
    ), "The hidden-failure row should expose its helpful-answer and CoT-unfaithfulness features."
    assert cot_row.active_features == ("helpful_answer", "cot_unfaithful"), (
        "The dashboard explanation should match the active features recomputed from the row."
    )
    assert "white-box catch" in cot_row.reviewer_note, (
        "The reviewer note should identify why internal evidence changed the decision."
    )
    assert not cot_row.black_box_prediction and cot_row.white_box_prediction, (
        "The designated hidden failure should be missed by the surface baseline and caught internally."
    )
    print("All tests in `test_dashboard_entries_keep_behavior_and_internal_evidence` passed!")


def test_controls_fail_and_feature_perturbation_moves_score(
    shuffled_label_control_scores: Callable | None = None,
    random_direction_control_scores: Callable | None = None,
    perturb_activation: Callable | None = None,
    score_white_box_monitor: Callable | None = None,
    binary_auroc: Callable | None = None,
):
    solutions = _solutions()
    shuffled_label_control_scores = shuffled_label_control_scores or solutions.shuffled_label_control_scores
    random_direction_control_scores = random_direction_control_scores or solutions.random_direction_control_scores
    perturb_activation = perturb_activation or solutions.perturb_activation
    score_white_box_monitor = score_white_box_monitor or solutions.score_white_box_monitor
    binary_auroc = binary_auroc or solutions.binary_auroc

    records, labels = _toy_inputs()
    activations = solutions.activation_matrix(records)
    train_x, train_y, heldout_x, heldout_y, *_ = solutions.split_train_heldout(records, activations, labels)
    monitor = solutions.fit_white_box_monitor(train_x, train_y)
    shuffled = shuffled_label_control_scores(train_x, train_y, heldout_x)
    random = random_direction_control_scores(train_x, train_y, heldout_x, seed=0)
    assert abs(binary_auroc(shuffled, heldout_y) - 0.6) < 1e-6, (
        "Shuffling labels should destroy most of the fitted direction's held-out ranking signal."
    )
    assert abs(binary_auroc(random, heldout_y) - 0.2) < 1e-6, (
        "The fixed random direction should fail visibly on the exact held-out organism."
    )

    clean = heldout_x[0]
    changed = perturb_activation(clean, feature_name="cot_unfaithful", delta=1.6)
    before, after = score_white_box_monitor(monitor, t.stack([clean, changed]))
    assert before < 0 < after, (
        "Adding the planted CoT-unfaithfulness feature should cross the monitor threshold."
    )
    expected_delta = 1.6 * monitor.direction[-1]
    t.testing.assert_close(after - before, expected_delta, atol=1e-6, rtol=1e-6)
    print("All tests in `test_controls_fail_and_feature_perturbation_moves_score` passed!")


def test_exact_signature_result_contract():
    result = _solutions().run_toy_monitor_experiment()
    assert result["white_box_auroc"] == 1.0, (
        "The exact white-box monitor should perfectly rank held-out failures."
    )
    assert abs(result["black_box_auroc"] - 0.8) < 1e-6, (
        "The output-only baseline should preserve the declared AUROC comparison."
    )
    assert result["white_box_accuracy"] == 1.0, (
        "The calibrated white-box threshold should classify all held-out records correctly."
    )
    assert abs(result["black_box_accuracy"] - (5 / 6)) < 1e-6, (
        "The output-only baseline should miss one planted hidden failure per held-out context."
    )
    assert result["white_box_only_catches"] == (5, 11, 17, 23), (
        "The signature result should preserve the four auditable white-box-only catches."
    )
    assert abs(result["shuffled_label_auroc"] - 0.6) < 1e-6, (
        "The signature result should retain the label-shuffled negative control."
    )
    assert abs(result["random_direction_auroc"] - 0.2) < 1e-6, (
        "The signature result should retain the fixed random-direction negative control."
    )
    assert result["explanation_accuracy"] == 1.0, (
        "Declared feature explanations should match every held-out organism label."
    )
    print("All tests in `test_exact_signature_result_contract` passed!")


def test_solution_notebook_exposes_taught_implementations():
    notebook_path = _section_dir() / "9.4_White-box_Evals_and_Monitors_solutions.ipynb"
    notebook = json.loads(notebook_path.read_text())
    code = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code"
    )
    tree = ast.parse(code)
    definitions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    support_imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "part4_white_box_evals_monitors.solutions"
        for alias in node.names
    }
    signature_code = next(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code" and "fig.savefig" in "".join(cell.get("source", []))
    )
    missing = sorted(set(TAUGHT_FUNCTIONS) - definitions)
    assert not missing, f"Solved notebook hides taught implementations: {missing}"
    assert support_imports <= {"run_pythia_white_box_monitor_preflight"}, (
        "Only the optional real-model preflight may be imported from solutions; "
        f"found hidden imports {sorted(support_imports)}."
    )
    assert "verification_report.json" not in signature_code, (
        "The signature result must be recomputed in the notebook, not loaded from a report."
    )
    assert "savefig" in signature_code and "white_box_evals_monitors_signature_result.png" in code, (
        "The solved notebook should visibly regenerate the named signature-result figure."
    )
    print("All tests in `test_solution_notebook_exposes_taught_implementations` passed!")


def test_exercise_notebook_declares_full_learner_contract():
    notebook_path = _section_dir() / "9.4_White-box_Evals_and_Monitors_exercises.ipynb"
    notebook = json.loads(notebook_path.read_text())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []))

    signature_code = next(
        "".join(cell.get("source", []))
        for cell in notebook.get("cells", [])
        if cell.get("cell_type") == "code" and "fig.savefig" in "".join(cell.get("source", []))
    )

    assert "By the end of this notebook, you will have shown" in source, (
        "The learner notebook should open with a concrete result the student will establish."
    )
    assert "## Core Question" in source and "## Learning Objectives" in source, (
        "The learner notebook should expose the canonical Core Question and Learning Objectives headings."
    )
    assert source.count("### Exercise ") == 8, (
        "The learner path should retain its eight-step implementation ladder."
    )
    assert source.count("<summary>Expected output</summary>") >= 8, (
        "Every exercise should tell learners what successful execution looks like."
    )
    assert source.count("<summary>Help") >= 8, (
        "Every exercise should include a discoverable hint before the solution."
    )
    assert source.count("<summary>Interpretation") >= 8, (
        "Every exercise should explain what its passing result establishes."
    )
    assert source.count("<summary>Solution</summary>") >= 8, (
        "Every exercise should retain a collapsible reference solution."
    )
    assert "Common bug" in source, (
        "The notebook should warn learners about a plausible interpretation or implementation failure."
    )
    assert "## Try It Yourself" in source, (
        "The learner notebook should retain an editable intervention play cell."
    )
    assert "## Bonus Anomaly Hunt" in source, (
        "The learner notebook should retain its anomaly-hunting extension."
    )
    assert "white_box_evals_monitors_signature_result.png" in source, (
        "The learner notebook should expose the visible signature-result artifact."
    )
    assert "verification_report.json" in source, (
        "The notebook should expose the committed real-model evidence as a separate support surface."
    )
    assert "verification_report.json" not in signature_code, (
        "The signature result should be recomputed from notebook functions rather than loaded from evidence."
    )
    assert "def run_gpu_test(max_vram_gb: float = 24.0)" in source, (
        "The notebook should expose the standard bounded GPU verification entry point."
    )
    assert "def run_full_experiment(max_vram_gb: float = 24.0)" in source, (
        "The notebook should expose the standard full-experiment entry point."
    )
    assert 'GT_TIER = "GT-3"' in source and 'EXERCISE_ID = "9_4_white_box_evals_and_monitors"' in source, (
        "Notebook identity metadata should match the section artifact lock."
    )
    assert "DIFFICULTY = 3" in source and "IMPORTANCE = 4" in source, (
        "Notebook difficulty and importance metadata should match the section artifact lock."
    )
    assert "REQUIRES_GPU = True" in source, (
        "The notebook should state that full evidence regeneration requires the serialized GPU path."
    )
    print("All tests in `test_exercise_notebook_declares_full_learner_contract` passed!")


def test_committed_gpu_report_retains_real_model_boundary():
    report = json.loads((_section_dir() / "verification_report.json").read_text())
    result = report["metrics"]["gpu_test"]
    assert result["cuda_available"] and result["preflight_passed"], (
        "The committed support report should record a successful CUDA preflight."
    )
    assert result["model_name"] == "EleutherAI/pythia-70m-deduped", (
        "The support evidence should use the model pinned by this section's artifact lock."
    )
    assert result["hf_revision"] == "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c", (
        "The support evidence should use the exact pinned Pythia revision."
    )
    assert result["generation_used"] is False, (
        "The white-box preflight should use hidden states and logits without generating completions."
    )
    assert result["monitor_auroc"] == 1.0, (
        "The committed real-model preflight should retain its held-out monitor ranking result."
    )
    assert result["label_shuffled_monitor_auroc"] <= 0.85, (
        "The label-shuffled real-model control should remain below the accepted AUROC ceiling."
    )
    assert result["random_direction_monitor_auroc"] <= 0.85, (
        "The random-direction real-model control should remain below the accepted AUROC ceiling."
    )
    assert result["within_vram_budget"], (
        "The committed preflight should remain within the declared 24 GB VRAM budget."
    )
    print("All tests in `test_committed_gpu_report_retains_real_model_boundary` passed!")

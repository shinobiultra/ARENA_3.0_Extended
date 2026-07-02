import csv
import json
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch as t


def _solutions():
    from chapter16_shapley_attribution_baselines.exercises.part8_shapley_mechinterp_agreement import (
        solutions,
    )

    return solutions


def test_rank_desc_toy_oracle(rank_desc: Callable | None = None):
    rank_desc = rank_desc or _solutions()._rank_desc
    scores = t.tensor([1.0, 3.0, -2.0, 3.0], dtype=t.float64)
    assert rank_desc(scores) == [1, 3, 0, 2], (
        "Ranking should sort descending and keep a deterministic order for ties."
    )
    print("All tests in `test_rank_desc_toy_oracle` passed!")


def test_analytic_neural_game_mechanistic_scores_toy_oracle(
    analytic_neural_game_mechanistic_scores: Callable | None = None,
):
    analytic_neural_game_mechanistic_scores = (
        analytic_neural_game_mechanistic_scores
        or _solutions().analytic_neural_game_mechanistic_scores
    )
    scores = analytic_neural_game_mechanistic_scores()
    expected = t.tensor([2.3, -1.45, 2.7, 0.15], dtype=t.float64)
    assert t.allclose(scores, expected), (
        "Analytic mechanism scores should come from the generated rule decomposition: "
        "linear terms plus half of each pair interaction."
    )
    assert int(scores.argmax().item()) == 2, "Feature 2 should be the top analytic feature."
    assert int(scores.argmin().item()) == 1, "Feature 1 should carry negative analytic credit."
    print("All tests in `test_analytic_neural_game_mechanistic_scores_toy_oracle` passed!")


def test_curve_from_rank_deletion_and_insertion(curve_from_rank: Callable | None = None):
    curve_from_rank = curve_from_rank or _solutions()._curve_from_rank
    values = _solutions().additive_game(t.tensor([1.0, 2.0, 0.5, -0.25]))
    rank = [1, 0, 2, 3]

    deletion = curve_from_rank(values, rank, "deletion")
    assert [point["value"] for point in deletion] == [3.25, 1.25, 0.25, -0.25, 0.0], (
        "Deletion should start at the full coalition and remove players in rank order."
    )
    assert [point["player"] for point in deletion] == ["start", 1, 0, 2, 3], (
        "Deletion points should record which player was removed at each step."
    )

    insertion = curve_from_rank(values, rank, "insertion")
    assert [point["value"] for point in insertion] == [0.0, 2.0, 3.0, 3.5, 3.25], (
        "Insertion should start empty and add players in rank order."
    )
    assert [point["player"] for point in insertion] == ["start", 1, 0, 2, 3], (
        "Insertion points should record which player was added at each step."
    )
    print("All tests in `test_curve_from_rank_deletion_and_insertion` passed!")


def test_additive_agreement_smoke_test(
    additive_agreement_smoke_test: Callable | None = None,
):
    additive_agreement_smoke_test = (
        additive_agreement_smoke_test or _solutions().additive_agreement_smoke_test
    )
    result = additive_agreement_smoke_test()
    assert result["agrees_with_mechanistic"], (
        "Additive Shapley, patching, and mechanistic scores should agree in the positive control."
    )
    assert result["topk_overlap"] == 1.0, (
        "The top-k feature set should fully overlap with the mechanistic ground truth."
    )
    assert result["spearman_correlation"] > 0.99, (
        "Attribution ranking should have near-perfect Spearman correlation with the mechanism."
    )
    assert result["deletion_drop"] > result["random_baseline_drop"], (
        "Deleting the top attributed feature should damage behavior more than the non-top baseline."
    )
    print("All tests in `test_additive_agreement_smoke_test` passed!")


def test_xor_disagreement_smoke_test(
    xor_disagreement_smoke_test: Callable | None = None,
):
    xor_disagreement_smoke_test = (
        xor_disagreement_smoke_test or _solutions().xor_disagreement_smoke_test
    )
    result = xor_disagreement_smoke_test()
    assert result["ordinary_shapley_misses"], (
        "Ordinary single-feature Shapley should miss the XOR pair mechanism."
    )
    assert result["interaction_recovers_pair"], (
        "Pairwise Shapley interactions should recover the XOR causal pair."
    )
    assert result["recovered_pair_interaction"] == 2.0, (
        "The recovered XOR pair interaction should have value 2.0."
    )
    assert result["max_single_feature_value"] == 0.0, (
        "Single-feature values should be zero in this XOR fixture."
    )
    print("All tests in `test_xor_disagreement_smoke_test` passed!")


def test_write_agreement_artifacts_contract(tmp_path: Path | None = None):
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp(prefix="arena-16-8-artifacts-"))
    solutions = _solutions()
    device = t.device("cpu")
    true_values = solutions.coalition_table_from_true_game(device)
    mechanistic_scores = solutions.analytic_neural_game_mechanistic_scores()
    agreement = solutions.attribution_agreement_report(
        true_values,
        mechanistic_scores=mechanistic_scores,
        num_players=solutions.NEURAL_GAME_NUM_PLAYERS,
        topk=2,
    )
    shuffled_values = solutions.additive_game(
        t.tensor([-2.0, 1.5, -1.0, 0.25], dtype=t.float64)
    )
    shuffled_agreement = solutions.attribution_agreement_report(
        shuffled_values,
        mechanistic_scores=mechanistic_scores,
        num_players=solutions.NEURAL_GAME_NUM_PLAYERS,
        topk=2,
    )
    interactions = solutions.pairwise_shapley_interactions(
        true_values,
        num_players=solutions.NEURAL_GAME_NUM_PLAYERS,
    )

    summary = solutions.write_agreement_artifacts(
        output_dir=tmp_path,
        model_values=true_values,
        true_values=true_values,
        shuffled_values=shuffled_values,
        agreement=agreement,
        shuffled_agreement=shuffled_agreement,
        model_interactions=interactions,
        true_interactions=interactions,
    )

    assert summary["agreement_artifacts_written"], (
        "Artifact writer should materialize every declared agreement artifact."
    )
    assert summary["agreement_artifact_count"] == 5, (
        "The artifact contract should include matrix, two curves, heatmap, and examples."
    )
    matrix_path = tmp_path / "agreement_matrix.csv"
    with matrix_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 7, "The agreement matrix should keep all seven required rows."
    assert {row["task"] for row in rows} >= {
        "additive_control",
        "neural_coalition_game",
        "shuffled_label_control",
        "xor_control",
    }, "Agreement artifacts should include positive, negative, and disagreement cases."
    for artifact_name in [
        "deletion_curves.png",
        "insertion_curves.png",
        "topk_overlap_heatmap.png",
        "method_disagreement_examples.md",
    ]:
        artifact = tmp_path / artifact_name
        assert artifact.exists() and artifact.stat().st_size > 0, (
            f"{artifact_name} should be written and non-empty."
        )
    print("All tests in `test_write_agreement_artifacts_contract` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["additive_agreement"]["agrees_with_mechanistic"], (
        "The notebook contract should include the additive agreement control."
    )
    assert result["xor_disagreement"]["interaction_recovers_pair"], (
        "The notebook contract should include the XOR interaction disagreement control."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_gpu_report_records_agreement_and_controls():
    report_path = Path(__file__).with_name("verification_report.json")
    report = json.loads(report_path.read_text())
    assert report["accepted"], "The committed 16.8 report should be accepted."
    assert report["gt_tier"] == "GT-0", "16.8 should remain a finite-game GT-0 section."

    gpu = report["metrics"]["gpu_test"]
    assert gpu["preflight_passed"], "The CUDA mechanistic-agreement preflight should pass."
    assert gpu["model_family"] == "cuda_trained_neural_coalition_game_mlp", (
        "The report should use the generated nonlinear neural coalition game."
    )
    assert gpu["coalition_count"] == 16, "The report should evaluate the complete 2**4 table."
    assert gpu["fit_mse"] <= 1e-8, "The trained model should fit the finite table."
    assert gpu["spearman_correlation"] >= 0.99, (
        "Shapley values should recover the analytic mechanistic ranking."
    )
    assert gpu["topk_overlap"] == 1.0, "Top-k overlap should match the analytic mechanism."
    assert gpu["deletion_drop"] > gpu["random_baseline_drop"], (
        "Deleting the top Shapley feature should beat the non-top deletion baseline."
    )
    assert gpu["interaction_max_abs_error"] <= 1e-4, (
        "Pair interactions should recover the planted interaction table."
    )
    assert gpu["top_interaction_pair"] == [0, 2], (
        "The positive planted interaction pair should be recovered."
    )
    assert gpu["second_interaction_pair"] == [1, 3], (
        "The negative planted interaction pair should be recovered as the second strongest pair."
    )
    assert gpu["shuffled_control_rejected"], (
        "The shuffled-label trained model should fail mechanistic agreement."
    )
    assert gpu["shuffled_control_topk_overlap"] == 0.0, (
        "The shuffled-label control should not recover the top mechanistic features."
    )
    assert gpu["agreement_artifacts_written"], "The CUDA report should write visible artifacts."
    assert gpu["agreement_artifact_count"] == 5, "The report should list all five artifacts."
    assert gpu["agreement_matrix_rows"] >= 7, "The agreement matrix should include all cases."
    assert gpu["peak_vram_gb"] < 1.0, "The finite model organism should stay under 1GB VRAM."
    assert gpu["within_vram_budget"], "The committed report should satisfy the VRAM budget."
    print("All tests in `test_committed_gpu_report_records_agreement_and_controls` passed!")

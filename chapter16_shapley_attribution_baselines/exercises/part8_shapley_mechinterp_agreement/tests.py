import ast
import csv
import json
import tempfile
from collections.abc import Callable
from pathlib import Path

import torch as t


SECTION_DIR = Path(__file__).resolve().parent


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


def test_enumerate_coalitions_toy_oracle(enumerate_coalitions: Callable | None = None):
    enumerate_coalitions = enumerate_coalitions or _solutions().enumerate_coalitions
    coalitions = enumerate_coalitions(3)
    assert len(coalitions) == 8, "Three binary players should have 2**3 coalitions."
    assert coalitions[0] == frozenset(), "The empty coalition should appear first."
    assert coalitions[-1] == frozenset({0, 1, 2}), "The full coalition should appear last."
    assert len(set(coalitions)) == len(coalitions), "Coalitions should be unique."
    print("All tests in `test_enumerate_coalitions_toy_oracle` passed!")


def test_exact_shapley_from_table_additive_oracle(
    exact_shapley_from_table: Callable | None = None,
):
    exact_shapley_from_table = exact_shapley_from_table or _solutions().exact_shapley_from_table
    values = _solutions().additive_game(t.tensor([1.0, 2.0, 0.5], dtype=t.float64))
    shapley = exact_shapley_from_table(values, num_players=3)
    assert t.allclose(shapley, t.tensor([1.0, 2.0, 0.5], dtype=t.float64)), (
        "Exact Shapley should recover the weights in an additive game."
    )
    assert abs(float(shapley.sum().item()) - 3.5) < 1e-9, (
        "Shapley values should satisfy efficiency: they sum to full minus empty value."
    )
    print("All tests in `test_exact_shapley_from_table_additive_oracle` passed!")


def test_finite_circuit_table_toy_oracle(
    finite_circuit_table: Callable | None = None,
    mechanistic_endpoint_scores: Callable | None = None,
    mechanistic_pair_matrix: Callable | None = None,
):
    solutions = _solutions()
    finite_circuit_table = finite_circuit_table or solutions.finite_circuit_table
    mechanistic_endpoint_scores = mechanistic_endpoint_scores or solutions.mechanistic_endpoint_scores
    mechanistic_pair_matrix = mechanistic_pair_matrix or solutions.mechanistic_pair_matrix

    values = finite_circuit_table()
    assert len(values) == 16, "The four-feature circuit should expose the complete 2**4 table."
    assert abs(values[frozenset()] - 0.25) < 1e-9, "The empty coalition should equal the intercept."
    assert abs(values[frozenset({0, 2})] - 5.25) < 1e-9, (
        "The {0, 2} coalition should include the positive x0*x2 edge."
    )
    scores = mechanistic_endpoint_scores()
    assert t.allclose(scores, t.tensor([2.3, -1.45, 2.7, 0.15], dtype=t.float64)), (
        "Endpoint scores should allocate each pair edge equally to its two endpoint features."
    )
    pair_matrix = mechanistic_pair_matrix()
    assert pair_matrix[0, 2] == pair_matrix[2, 0] == 2.2, (
        "The positive planted pair edge should be represented symmetrically."
    )
    assert pair_matrix[1, 3] == pair_matrix[3, 1] == -1.5, (
        "The negative planted pair edge should be represented symmetrically."
    )
    print("All tests in `test_finite_circuit_table_toy_oracle` passed!")


def test_finite_circuit_agreement_case_matches_ground_truth(
    finite_circuit_table: Callable | None = None,
    exact_shapley_from_table: Callable | None = None,
    mechanistic_endpoint_scores: Callable | None = None,
):
    solutions = _solutions()
    finite_circuit_table = finite_circuit_table or solutions.finite_circuit_table
    exact_shapley_from_table = exact_shapley_from_table or solutions.exact_shapley_from_table
    mechanistic_endpoint_scores = mechanistic_endpoint_scores or solutions.mechanistic_endpoint_scores

    values = finite_circuit_table()
    shapley = exact_shapley_from_table(values, num_players=solutions.NEURAL_GAME_NUM_PLAYERS)
    mechanism = mechanistic_endpoint_scores()
    assert t.allclose(shapley, mechanism), (
        "The planted finite circuit should be an exact agreement case: Shapley equals "
        "linear wires plus half of each pair edge."
    )
    assert int(shapley.argmax().item()) == 2, "The answer-slot feature should be top-ranked."
    print("All tests in `test_finite_circuit_agreement_case_matches_ground_truth` passed!")


def test_causal_patching_effects_expose_interaction_consequences(
    causal_patching_effects: Callable | None = None,
):
    solutions = _solutions()
    causal_patching_effects = causal_patching_effects or solutions.causal_patching_effects
    values = solutions.finite_circuit_table()
    patching = causal_patching_effects(values, num_players=solutions.NEURAL_GAME_NUM_PLAYERS)
    shapley = solutions.exact_shapley_from_table(values, num_players=solutions.NEURAL_GAME_NUM_PLAYERS)
    assert int(patching.argmax().item()) == int(shapley.argmax().item()) == 2, (
        "Causal patching and Shapley should identify the same top feature in the agreement case."
    )
    assert patching[0] > shapley[0] and patching[2] > shapley[2], (
        "Full-minus-ablated patching should expose downstream interaction consequences, "
        "so it need not equal single-feature Shapley in an interaction-bearing circuit."
    )
    print("All tests in `test_causal_patching_effects_expose_interaction_consequences` passed!")


def test_pairwise_interactions_recover_planted_edges(
    pairwise_interactions_from_table: Callable | None = None,
):
    solutions = _solutions()
    pairwise_interactions_from_table = (
        pairwise_interactions_from_table or solutions.pairwise_interactions_from_table
    )
    interactions = pairwise_interactions_from_table(
        solutions.finite_circuit_table(),
        num_players=solutions.NEURAL_GAME_NUM_PLAYERS,
    )
    expected = solutions.mechanistic_pair_matrix()
    assert t.allclose(interactions, expected, atol=1e-9), (
        "Pairwise Shapley interactions should recover the planted positive and negative edges."
    )
    print("All tests in `test_pairwise_interactions_recover_planted_edges` passed!")


def test_shuffled_mechanistic_control_rejected(shuffled_mechanistic_control: Callable | None = None):
    shuffled_mechanistic_control = (
        shuffled_mechanistic_control or _solutions().shuffled_mechanistic_control
    )
    result = shuffled_mechanistic_control()
    assert result["true_top2_overlap"] == 1.0, "The true mechanism should pass top-2 overlap."
    assert result["shuffled_top2_overlap"] < 1.0, (
        "A shuffled mechanism-label control should fail the same top-k agreement check."
    )
    assert result["control_rejected"], "The shuffled control should be explicitly rejected."
    print("All tests in `test_shuffled_mechanistic_control_rejected` passed!")


def test_data_player_bridge_matches_exact_data_shapley(
    data_player_bridge_report: Callable | None = None,
):
    data_player_bridge_report = data_player_bridge_report or _solutions().data_player_bridge_report
    result = data_player_bridge_report()
    assert result["pearson_correlation"] > 0.99, (
        "The one-run gradient-dot proxy should correlate with exact Data Shapley on this "
        "one-step linear bridge problem."
    )
    assert result["identifies_harmful"], "The mislabeled example should be the harmful one."
    assert result["identifies_helpful_tie"], (
        "At least one helpful example should be top-scored, with ties handled honestly."
    )
    print("All tests in `test_data_player_bridge_matches_exact_data_shapley` passed!")


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


def test_write_agreement_artifacts_contract(
    tmp_path: Path | None = None,
    write_agreement_artifacts: Callable | None = None,
):
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp(prefix="arena-16-8-artifacts-"))
    solutions = _solutions()
    write_agreement_artifacts = write_agreement_artifacts or solutions.write_agreement_artifacts
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

    summary = write_agreement_artifacts(
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
    assert summary["agreement_artifact_count"] >= 5, (
        "The artifact contract should include at least matrix, two curves, heatmap, and examples."
    )
    matrix_path = tmp_path / "agreement_matrix.csv"
    with matrix_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 7, "The agreement matrix should keep at least the seven required rows."
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


def test_write_signature_panel_contract(
    tmp_path: Path | None = None,
    write_signature_panel: Callable | None = None,
):
    if tmp_path is None:
        tmp_path = Path(tempfile.mkdtemp(prefix="arena-16-8-signature-"))
    write_signature_panel = write_signature_panel or _solutions().write_signature_panel
    path = tmp_path / "signature.png"
    summary = write_signature_panel(path)
    assert summary["signature_panel_written"], "The signature panel should be written."
    assert path.exists() and path.stat().st_size > 10_000, (
        "The signature result should be a nontrivial visual artifact, not an empty placeholder."
    )
    assert summary["top2_overlap"] == 1.0, "The agreement panel should include a true agreement case."
    assert summary["xor_pair_interaction_abs"] == 2.0, (
        "The panel should be backed by a tested XOR disagreement diagnosis."
    )
    print("All tests in `test_write_signature_panel_contract` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["additive_agreement"]["agrees_with_mechanistic"], (
        "The notebook contract should include the additive agreement control."
    )
    assert result["xor_disagreement"]["interaction_recovers_pair"], (
        "The notebook contract should include the XOR interaction disagreement control."
    )
    assert result["data_player_bridge"]["pearson_correlation"] > 0.99, (
        "The notebook contract should include the one-step Data Shapley bridge."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_notebooks_have_arena_learner_surface():
    required_phrases = [
        "By the end of this notebook",
        "Core Question",
        "Learning Objectives",
        "Signature Result",
        "Try It Yourself",
        "Bonus Anomaly Hunt",
        "Limitations",
        "<summary>Expected output</summary>",
        "<summary>Help",
        "<summary>Interpreting the result</summary>",
        "<summary>Solution</summary>",
    ]
    for notebook_name in [
        "16.8_Do_SHAPley_and_Mechanistic_Interpretability_Agree_exercises.ipynb",
        "16.8_Do_SHAPley_and_Mechanistic_Interpretability_Agree_solutions.ipynb",
    ]:
        text = (SECTION_DIR / notebook_name).read_text(encoding="utf-8")
        for phrase in required_phrases:
            assert phrase in text, f"{notebook_name} is missing {phrase!r}."
        assert text.count("Exercise -") >= 6, f"{notebook_name} should expose at least six exercises."
        assert "verification_report.json" not in text[text.find("Signature Result") : text.find("Limitations")], (
            f"{notebook_name} should not make the signature result a verification-report wrapper."
        )
    print("All tests in `test_notebooks_have_arena_learner_surface` passed!")


def test_solution_notebook_exposes_taught_implementations():
    notebook_path = SECTION_DIR / (
        "16.8_Do_SHAPley_and_Mechanistic_Interpretability_Agree_solutions.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    function_names: set[str] = set()
    for cell in notebook["cells"]:
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        assert "NotImplementedError" not in source, (
            "The solution notebook must not retain learner stubs."
        )
        tree = ast.parse(source)
        function_names.update(
            node.name for node in tree.body if isinstance(node, ast.FunctionDef)
        )

    required = {
        "enumerate_coalitions",
        "finite_circuit_value",
        "exact_shapley_from_table",
        "causal_patching_effects",
        "agreement_summary",
        "pairwise_interactions_from_table",
        "shuffled_mechanistic_control",
        "data_player_bridge_report",
        "write_signature_panel",
        "run_smoke_test",
    }
    missing = sorted(required - function_names)
    assert not missing, (
        "The solved notebook must expose every taught method inline; "
        f"missing {missing}."
    )


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

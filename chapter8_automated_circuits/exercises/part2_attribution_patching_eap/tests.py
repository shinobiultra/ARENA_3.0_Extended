"""Semantic and learner-surface tests for section 8.2."""

from __future__ import annotations

import ast
from collections.abc import Callable
from pathlib import Path

import nbformat
import torch as t


SECTION_DIR = Path(__file__).resolve().parent
EXERCISE_NOTEBOOK = SECTION_DIR / "8.2_Attribution_Patching_and_EAP_exercises.ipynb"
SOLUTION_NOTEBOOK = SECTION_DIR / "8.2_Attribution_Patching_and_EAP_solutions.ipynb"
PAGE = SECTION_DIR.parents[1] / "instructions" / "pages" / "02_[8.2]_Attribution_Patching_and_EAP.md"


def _solutions():
    from chapter8_automated_circuits.exercises.part2_attribution_patching_eap import solutions

    return solutions


def _assert_close(actual: t.Tensor, expected: t.Tensor, *, atol: float = 1e-4) -> None:
    assert t.allclose(actual.to(dtype=t.float64), expected.to(dtype=t.float64), atol=atol), (
        f"Expected {expected.tolist()}, got {actual.tolist()}."
    )


def test_make_planted_graph_has_exact_ground_truth(
    make_graph: Callable | None = None,
    metric: Callable | None = None,
) -> None:
    make_graph = make_graph or _solutions().make_planted_attribution_graph
    metric = metric or _solutions().toy_metric
    graph = make_graph()
    assert graph.weights.shape == (4, 4), (
        "The planted graph should expose one weight for every receiver-sender pair."
    )
    assert graph.sender_labels == ("name copy", "relation gate", "context", "decoy"), (
        "Sender labels should keep the four learner-visible routes in their declared order."
    )
    assert graph.receiver_labels[1] == "threshold gate", (
        "Receiver index 1 should identify the nonlinear threshold gate used in the failure case."
    )
    assert graph.weights[3, :3].abs().sum().item() == 0.0, (
        "The unused receiver should remain an exact zero-effect control."
    )
    corrupt = metric(graph, graph.corrupt_sender)
    clean = metric(graph, graph.clean_sender)
    assert abs(float(corrupt.item())) < 1e-12, (
        "The corrupt sender state should define the zero metric endpoint."
    )
    assert abs(float(clean.item()) - 4.399326441019204) < 1e-9, (
        "The clean planted graph should reproduce the declared exact metric."
    )
    print("Planted graph ground truth passed: clean-corrupt metric = 4.399326.")


def test_exact_node_patching_recovers_known_effects(
    exact_node_patch_effects: Callable | None = None,
) -> None:
    exact_node_patch_effects = exact_node_patch_effects or _solutions().exact_node_patch_effects
    graph = _solutions().make_planted_attribution_graph()
    actual = exact_node_patch_effects(graph)
    expected = t.tensor([1.4391066, 1.4, 0.8312297, 0.0], dtype=t.float64)
    _assert_close(actual, expected)
    assert int(actual.argmax().item()) == 0, (
        "Exact node patching should rank the name-copy sender first."
    )
    assert actual[1] > 1.0, "The gated relation route must be causally important."
    assert actual[3] == 0.0, "The decoy must be an exact zero-effect control."
    print("Exact node patching passed: the gated route is important and the decoy is null.")


def test_first_order_attribution_exposes_nonlinear_false_negative(
    first_order_node_attribution: Callable | None = None,
) -> None:
    first_order_node_attribution = first_order_node_attribution or _solutions().first_order_node_attribution
    graph = _solutions().make_planted_attribution_graph()
    actual = first_order_node_attribution(graph)
    expected = t.tensor([1.44, 0.0, 0.94, 0.0], dtype=t.float64)
    _assert_close(actual, expected)
    assert actual[1] == 0.0, (
        "Corrupt-point attribution should miss the inactive threshold-gated sender."
    )
    assert _solutions().exact_node_patch_effects(graph)[1] > 1.0, (
        "The missed threshold-gated sender must still have a large exact causal effect."
    )
    print("First-order attribution passed: it visibly misses the threshold-gated route.")


def test_integrated_gradients_recovers_ranking_and_completeness(
    sender_path_gradients: Callable | None = None,
    integrated_gradient_node_scores: Callable | None = None,
) -> None:
    sender_path_gradients = sender_path_gradients or _solutions().sender_path_gradients
    integrated_gradient_node_scores = integrated_gradient_node_scores or _solutions().integrated_gradient_node_scores
    graph = _solutions().make_planted_attribution_graph()
    gradients = sender_path_gradients(graph, steps=64)
    assert gradients.shape == (64, 4), (
        "Path integration should record one four-sender gradient at every interpolation step."
    )
    actual = integrated_gradient_node_scores(graph, steps=64)
    expected = t.tensor([1.4134, 1.9441, 1.0414, 0.0], dtype=t.float64)
    _assert_close(actual, expected, atol=2e-3)
    full_delta = _solutions().toy_metric(graph, graph.clean_sender) - _solutions().toy_metric(
        graph, graph.corrupt_sender
    )
    assert abs(float(actual.sum().item() - full_delta.item())) < 5e-4, (
        "Integrated node scores should conserve the full clean-corrupt metric change."
    )
    exact = _solutions().exact_node_patch_effects(graph)
    ap = _solutions().first_order_node_attribution(graph)
    ig_corr = _solutions().agreement_metrics(exact, actual, top_k=2).correlation
    ap_corr = _solutions().agreement_metrics(exact, ap, top_k=2).correlation
    assert ig_corr > 0.95 and ig_corr > ap_corr + 0.45, (
        "Integrated gradients should recover the exact ranking and clearly outperform endpoint attribution."
    )
    print("Integrated gradients passed: top-2 ranking recovered and completeness error < 5e-4.")


def test_exact_edge_patching_matches_planted_matrix(
    planted_edge_deltas: Callable | None = None,
    exact_edge_patch_effects: Callable | None = None,
) -> None:
    planted_edge_deltas = planted_edge_deltas or _solutions().planted_edge_deltas
    exact_edge_patch_effects = exact_edge_patch_effects or _solutions().exact_edge_patch_effects
    graph = _solutions().make_planted_attribution_graph()
    edge_deltas = planted_edge_deltas(graph)
    assert edge_deltas.shape == (4, 4), (
        "Edge deltas should retain the receiver-by-sender shape of the planted graph."
    )
    actual = exact_edge_patch_effects(graph)
    expected = t.tensor(
        [
            [1.32, 0.0, 0.1191066, 0.0],
            [0.0, 1.4, 0.0, 0.0],
            [0.3, 0.0, 0.5312297, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=t.float64,
    )
    _assert_close(actual, expected)
    assert int(actual.abs().flatten().topk(3).indices[0].item()) == 5, (
        "The relation-gate to threshold-gate edge should have the largest exact effect."
    )
    print("Exact edge patching passed: relation gate -> threshold gate is the top edge.")


def test_eap_and_eap_ig_separate_speed_from_faithfulness(
    receiver_path_gradients: Callable | None = None,
    eap_edge_scores: Callable | None = None,
    corrupt_receiver_gradients: Callable | None = None,
) -> None:
    receiver_path_gradients = receiver_path_gradients or _solutions().receiver_path_gradients
    eap_edge_scores = eap_edge_scores or _solutions().eap_edge_scores
    corrupt_receiver_gradients = corrupt_receiver_gradients or _solutions().corrupt_receiver_gradients
    graph = _solutions().make_planted_attribution_graph()
    deltas = _solutions().planted_edge_deltas(graph)
    corrupt_grad = corrupt_receiver_gradients(graph)
    eap = eap_edge_scores(deltas, corrupt_grad)
    eap_ig = eap_edge_scores(deltas, receiver_path_gradients(graph, steps=64).mean(dim=0))
    assert eap.shape == eap_ig.shape == (4, 4), (
        "EAP and EAP-IG should score the same receiver-by-sender edge set."
    )
    assert eap[1, 1] == 0.0, "Corrupt-point EAP must miss the inactive threshold edge."
    assert eap_ig[1, 1] > 1.9, "EAP-IG must recover the threshold edge along the path."
    exact = _solutions().exact_edge_patch_effects(graph)
    eap_metrics = _solutions().agreement_metrics(exact, eap, top_k=3)
    eap_ig_metrics = _solutions().agreement_metrics(exact, eap_ig, top_k=3)
    assert eap_metrics.topk_overlap == 2 / 3, (
        "Endpoint EAP should visibly miss one of the three strongest exact edges."
    )
    assert eap_ig_metrics.topk_overlap == 1.0, (
        "EAP-IG should recover all three strongest exact edges."
    )
    assert eap_ig_metrics.correlation > 0.97 > eap_metrics.correlation, (
        "Path integration should improve edge-score correlation beyond endpoint EAP."
    )
    print("EAP comparison passed: IG raises top-3 overlap from 2/3 to 1.0.")


def test_agreement_metrics_rejects_pretty_but_wrong_rankings(
    agreement_metrics: Callable | None = None,
) -> None:
    agreement_metrics = agreement_metrics or _solutions().agreement_metrics
    exact = t.tensor([3.0, 2.0, 1.0, 0.0])
    wrong = t.tensor([0.0, 0.1, 4.0, 3.0])
    report = agreement_metrics(exact, wrong, top_k=2, false_negative_threshold=1.5)
    assert report.topk_overlap == 0.0, (
        "The deliberately wrong ranking should have no top-2 overlap with the reference."
    )
    assert report.false_negative_indices == (0, 1), (
        "Agreement diagnostics should name both high-effect items omitted by the wrong ranking."
    )
    assert report.mean_absolute_error > 0.0, (
        "A wrong score vector should have nonzero absolute error."
    )
    try:
        agreement_metrics(exact, wrong[:3], top_k=2)
    except ValueError as exc:
        assert "match" in str(exc), (
            "Mismatched score shapes should produce a clear matching-shape error."
        )
    else:
        raise AssertionError("Mismatched score tensors must fail.")
    print("Agreement diagnostics passed: wrong top-k rankings cannot hide behind a plot.")


def test_signature_result_metrics_and_controls() -> None:
    result = _solutions().run_planted_signature_result()
    assert abs(result.full_metric_delta - 4.399326441019204) < 1e-9, (
        "The signature result should retain the planted clean-corrupt metric delta."
    )
    assert result.node_attribution.correlation < 0.47, (
        "Endpoint node attribution should expose the intended nonlinear failure."
    )
    assert result.node_integrated.correlation > 0.95, (
        "Integrated node attribution should closely match exact patch effects."
    )
    assert result.node_attribution.topk_overlap == 0.5, (
        "Endpoint attribution should recover only one of the exact top-2 nodes."
    )
    assert result.node_integrated.topk_overlap == 1.0, (
        "Integrated attribution should recover both exact top-2 nodes."
    )
    assert result.edge_eap.correlation < 0.67, (
        "Endpoint EAP should remain visibly misaligned with exact edge effects."
    )
    assert result.edge_eap_ig.correlation > 0.97, (
        "EAP-IG should closely track the exact edge-effect ranking."
    )
    assert result.edge_eap.topk_overlap == 2 / 3, (
        "Endpoint EAP should miss one exact top-3 edge."
    )
    assert result.edge_eap_ig.topk_overlap == 1.0, (
        "EAP-IG should recover the complete exact top-3 edge set."
    )
    assert result.node_ig_conservation_error < 5e-4, (
        "Integrated node scores should conserve the path metric within tolerance."
    )
    assert result.edge_ig_conservation_error < 5e-4, (
        "Integrated edge scores should conserve the path metric within tolerance."
    )
    assert result.linear_control_max_error < 1e-12, (
        "Endpoint attribution should be exact on the linear positive control."
    )
    assert result.shuffled_edge_correlation < 0.0, (
        "Shuffling edge deltas should destroy score correlation with exact effects."
    )
    assert result.shuffled_edge_topk_overlap == 0.0, (
        "The shuffled-delta control should recover none of the exact top-3 edges."
    )
    print("Signature result passed: nonlinear recovery, linear control, and shuffled control all separate.")


def test_solution_smoke_contract_contains_signature_metrics() -> None:
    signature = _solutions().run_smoke_test(cpu=True)["toy_signature"]
    assert signature["node_integrated_correlation"] > 0.95, (
        "The smoke contract should retain the integrated-node agreement result."
    )
    assert signature["edge_eap_ig_top3_overlap"] == 1.0, (
        "The smoke contract should retain exact top-3 edge recovery under EAP-IG."
    )
    assert signature["linear_control_max_error"] < 1e-12, (
        "The smoke contract should retain the linear positive-control parity check."
    )
    assert signature["shuffled_edge_top3_overlap"] == 0.0, (
        "The smoke contract should retain the failing shuffled-delta control."
    )


def _notebook_sources(path: Path) -> tuple[str, str]:
    notebook = nbformat.read(path, as_version=4)
    markdown = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "markdown")
    code = "\n".join(cell.source for cell in notebook.cells if cell.cell_type == "code")
    return markdown, code


def test_notebook_code_cells_parse() -> None:
    for path in (EXERCISE_NOTEBOOK, SOLUTION_NOTEBOOK):
        _, code = _notebook_sources(path)
        ast.parse(code)


def test_solution_notebook_exposes_taught_implementations() -> None:
    _, code = _notebook_sources(SOLUTION_NOTEBOOK)
    taught = (
        "make_planted_attribution_graph",
        "toy_metric",
        "exact_node_patch_effects",
        "first_order_node_attribution",
        "sender_path_gradients",
        "integrated_gradient_node_scores",
        "planted_edge_deltas",
        "exact_edge_patch_effects",
        "receiver_path_gradients",
        "eap_edge_scores",
        "corrupt_receiver_gradients",
        "agreement_metrics",
    )
    for name in taught:
        assert f"def {name}(" in code, f"Solved notebook must define {name} inline."
    assert "NotImplementedError" not in code, (
        "The solution notebook should contain complete implementations for every exercise."
    )
    assert "solutions." not in code, "The solved learner path must not call hidden taught methods."


def test_learner_surface_contains_full_arena_progression() -> None:
    exercise_markdown, exercise_code = _notebook_sources(EXERCISE_NOTEBOOK)
    solution_markdown, _ = _notebook_sources(SOLUTION_NOTEBOOK)
    for markdown in (exercise_markdown, solution_markdown):
        assert markdown.count("### Exercise ") == 7, (
            "Each notebook should retain the seven-step exact-to-approximation progression."
        )
        for marker in (
            "By the end of this notebook",
            "Learning Objectives",
            "Expected output",
            "Help",
            "Interpretation",
            "Solution",
            "Try It Yourself",
            "Bonus Anomaly Hunt",
            "Limitations",
            "Reading",
        ):
            assert marker in markdown, (
                f"Each notebook should expose the learner-surface marker {marker!r}."
            )
    assert "verification_report.json" not in exercise_code, (
        "The exercise implementation should compute its evidence instead of loading a report."
    )
    assert "raise NotImplementedError" in exercise_code, (
        "The exercise notebook should retain visible student implementation stubs."
    )


def test_instruction_page_teaches_the_same_claim() -> None:
    page = PAGE.read_text()
    for marker in (
        "corrupt-point attribution misses",
        "Exact node patching",
        "EAP-IG",
        "Try It Yourself",
        "Bonus Anomaly Hunt",
        "linear positive control",
        "shuffled-delta control",
        "not an IOI-scale",
        "attribution_patching_eap_signature_result.png",
    ):
        assert marker in page, (
            f"The instruction page should teach the same scoped claim via {marker!r}."
        )


def test_original_arena_preservation_boundary() -> None:
    relative_paths = [
        str(path.relative_to(SECTION_DIR.parents[2]))
        for path in SECTION_DIR.rglob("*")
        if path.is_file()
    ]
    assert all(not path.startswith("chapter1_transformer_interp/") for path in relative_paths), (
        "Section-local changes must not cross the original ARENA preservation boundary."
    )

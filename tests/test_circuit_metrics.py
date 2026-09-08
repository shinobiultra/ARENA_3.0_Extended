import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.circuit_metrics import (
        acdc_pruning_report,
        circuit_method_comparison_report,
        circuit_completeness_report,
        circuit_faithfulness_report,
        circuit_minimality_report,
        ood_template_report,
        random_circuit_baseline_report,
    )


def test_acdc_pruning_report_keeps_edges_above_threshold():
    scores = t.tensor([0.9, 0.2, 0.7])
    names = ["name-mover", "backup", "negative"]

    report = acdc_pruning_report(scores, names, threshold=0.5)

    assert report.kept_edges == ("name-mover", "negative")
    assert report.removed_edges == ("backup",)
    assert report.num_kept == 2


def test_acdc_pruning_report_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="nonempty"):
        acdc_pruning_report(t.empty(0), [], threshold=0.5)
    with pytest.raises(ValueError, match="edge_names"):
        acdc_pruning_report(t.tensor([0.9]), [""], threshold=0.5)
    with pytest.raises(ValueError, match="finite"):
        acdc_pruning_report(t.tensor([float("nan")]), ["name-mover"], threshold=0.5)
    with pytest.raises(ValueError, match="finite"):
        acdc_pruning_report(t.tensor([0.9]), ["name-mover"], threshold=float("nan"))


def test_circuit_faithfulness_report_measures_preserved_fraction():
    report = circuit_faithfulness_report(
        full_metric=3.0,
        corrupt_metric=-1.0,
        circuit_metric=2.2,
        min_preserved_fraction=0.75,
    )

    assert report.preserved_fraction == pytest.approx(0.8)
    assert report.passes_faithfulness


def test_circuit_metric_reports_reject_invalid_thresholds():
    with pytest.raises(ValueError, match="nonnegative"):
        circuit_faithfulness_report(
            full_metric=3.0,
            corrupt_metric=-1.0,
            circuit_metric=2.2,
            min_preserved_fraction=-0.1,
        )
    with pytest.raises(ValueError, match="finite"):
        circuit_minimality_report(
            circuit_metric=2.2,
            ablated_metric=0.5,
            min_metric_damage=float("nan"),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        circuit_completeness_report(
            circuit_metric=2.2,
            expanded_metric=2.35,
            max_omitted_node_gain=-0.1,
        )
    with pytest.raises(ValueError, match="finite"):
        random_circuit_baseline_report(
            circuit_metric=2.2,
            random_metric=float("inf"),
            min_margin=1.0,
        )


def test_circuit_minimality_report_requires_damage_from_ablation():
    report = circuit_minimality_report(
        circuit_metric=2.2,
        ablated_metric=0.5,
        min_metric_damage=1.0,
    )

    assert report.metric_damage == pytest.approx(1.7)
    assert report.passes_minimality


def test_circuit_completeness_report_requires_small_omitted_gain():
    report = circuit_completeness_report(
        circuit_metric=2.2,
        expanded_metric=2.35,
        max_omitted_node_gain=0.2,
    )

    assert report.omitted_node_gain == pytest.approx(0.15)
    assert report.passes_completeness


def test_random_circuit_baseline_report_requires_margin():
    report = random_circuit_baseline_report(
        circuit_metric=2.2,
        random_metric=0.8,
        min_margin=1.0,
    )

    assert report.margin == pytest.approx(1.4)
    assert report.circuit_beats_random


def test_ood_template_report_requires_every_template_to_pass():
    logits = t.tensor(
        [
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    answer_ids = t.tensor([0, 1, 0, 1])
    template_ids = t.tensor([0, 0, 1, 1])

    report = ood_template_report(logits, answer_ids, template_ids, min_accuracy=1.0)

    assert report.per_template_accuracy == {0: 1.0, 1: 1.0}
    assert report.worst_template_accuracy == 1.0
    assert report.passes_ood


def test_ood_template_report_rejects_degenerate_inputs():
    logits = t.tensor([[2.0, 0.0], [0.0, 2.0]])
    answer_ids = t.tensor([0, 1])
    template_ids = t.tensor([0, 1])

    with pytest.raises(ValueError, match="between 0 and 1"):
        ood_template_report(logits, answer_ids, template_ids, min_accuracy=1.2)
    with pytest.raises(ValueError, match="valid vocabulary"):
        ood_template_report(logits, t.tensor([0, 2]), template_ids)
    bad_logits = logits.clone()
    bad_logits[0, 0] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        ood_template_report(bad_logits, answer_ids, template_ids)


def test_circuit_method_comparison_report_compares_approximate_methods_to_exact():
    exact = t.tensor([0.95, 0.8, 0.15, 0.05])
    eap_ig = t.tensor([0.9, 0.7, 0.2, 0.1])
    weak_eap = t.tensor([0.2, 0.1, 0.9, 0.8])
    names = ["name-mover", "backup-name-mover", "mlp-noise", "wrong-position"]

    report = circuit_method_comparison_report(
        exact,
        {"eap_ig": eap_ig, "weak_eap": weak_eap},
        names,
        top_k=2,
        min_topk_overlap=0.5,
        min_score_correlation=0.5,
    )

    assert report.exact_top_edges == ("name-mover", "backup-name-mover")
    assert report.method_top_edges["eap_ig"] == ("name-mover", "backup-name-mover")
    assert report.method_top_edges["weak_eap"] == ("mlp-noise", "wrong-position")
    assert report.topk_overlap == {"eap_ig": 1.0, "weak_eap": 0.0}
    assert report.score_correlations["eap_ig"] > 0.95
    assert report.score_correlations["weak_eap"] < 0.0
    assert report.circuit_sizes == {"exact": 2, "eap_ig": 2, "weak_eap": 2}
    assert report.best_matching_method == "eap_ig"
    assert not report.passes_comparison


def test_circuit_method_comparison_report_rejects_bad_inputs():
    exact = t.tensor([0.95, 0.8, 0.15, 0.05])
    eap_ig = t.tensor([0.9, 0.7, 0.2, 0.1])
    names = ["name-mover", "backup-name-mover", "mlp-noise", "wrong-position"]

    with pytest.raises(ValueError, match="top_k"):
        circuit_method_comparison_report(exact, {"eap_ig": eap_ig}, names, top_k=0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        circuit_method_comparison_report(
            exact,
            {"eap_ig": eap_ig},
            names,
            top_k=2,
            min_topk_overlap=1.2,
        )
    bad_exact = exact.clone()
    bad_exact[0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        circuit_method_comparison_report(bad_exact, {"eap_ig": eap_ig}, names, top_k=2)
    with pytest.raises(ValueError, match="match exact_scores shape"):
        circuit_method_comparison_report(exact, {"eap_ig": t.empty(0)}, names, top_k=2)

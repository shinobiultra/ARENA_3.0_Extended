import importlib.util

import pytest

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    import torch as t

    from arena_ext.sparse_feature_circuits import (
        eap_ig_comparison_report,
        exact_feature_edge_patching_report,
        exact_feature_node_patching_report,
        expected_pythia_sfc_dictionary_paths,
        official_sparse_feature_circuit_artifact_report,
        official_sparse_feature_artifact_readiness_report,
        random_feature_graph_control_report,
        residual_feature_preflight_report,
        shift_style_sparse_feature_editing_report,
        sparse_autoencoder_state_dict_smoke_report,
        sparse_feature_attribution_smoke_report,
        summarize_official_sparse_feature_circuit_artifact,
        threshold_feature_graph_report,
        toy_shift_sparse_feature_editing_fixture,
    )


def test_exact_feature_node_patching_reports_recovered_fraction():
    contributions = t.tensor([0.7, 0.2, 0.05, 0.05])

    report = exact_feature_node_patching_report(contributions, [0, 1])

    assert report.graph_logit_diff == pytest.approx(0.9)
    assert report.full_logit_diff == pytest.approx(1.0)
    assert report.recovered_fraction == pytest.approx(0.9)
    assert report.passes_recovery


def test_exact_feature_edge_patching_uses_edge_oracle():
    edge_scores = t.tensor([[0.05, 0.8], [0.05, 0.1]])

    report = exact_feature_edge_patching_report(edge_scores, [(0, 1)])

    assert report.full_edge_score == pytest.approx(1.0)
    assert report.graph_edge_score == pytest.approx(0.8)
    assert report.recovered_fraction == pytest.approx(0.8)
    assert report.passes_recovery


def test_eap_ig_comparison_requires_improvement_over_plain_eap():
    exact = t.tensor([0.7, 0.2, 0.05, 0.05])
    eap = t.tensor([0.5, 0.35, 0.1, 0.05])
    eap_ig = t.tensor([0.69, 0.21, 0.04, 0.06])

    report = eap_ig_comparison_report(exact, eap, eap_ig, max_eap_ig_error=0.02)

    assert report.eap_error > report.eap_ig_error
    assert report.eap_ig_improves


def test_threshold_feature_graph_preserves_metric():
    report = threshold_feature_graph_report(
        t.tensor([0.7, 0.2, 0.05, 0.05]),
        threshold=0.2,
    )

    assert report.selected_feature_ids == (0, 1)
    assert report.recovered_fraction == pytest.approx(0.9)
    assert report.passes_threshold


def test_random_feature_graph_control_fails_on_low_value_features():
    report = random_feature_graph_control_report(
        t.tensor([0.7, 0.2, 0.05, 0.05]),
        target_feature_ids=[0, 1],
        random_feature_ids=[2, 3],
        min_margin=0.5,
    )

    assert report.target_recovered_fraction == pytest.approx(0.9)
    assert report.random_recovered_fraction == pytest.approx(0.1)
    assert report.random_graph_fails


def test_shift_style_sparse_feature_editing_removes_spurious_reliance():
    fixture = toy_shift_sparse_feature_editing_fixture()

    report = shift_style_sparse_feature_editing_report(
        fixture["train_features"],
        fixture["train_labels"],
        fixture["ood_features"],
        fixture["ood_labels"],
        fixture["classifier_weights"],
        target_feature_ids=[0],
        spurious_feature_ids=[1],
        random_feature_ids=[2],
        suppression=0.0,
        max_target_accuracy_drop=0.05,
        min_ood_improvement=0.5,
        min_random_edit_gap=0.5,
    )

    assert report.baseline_train_accuracy == pytest.approx(1.0)
    assert report.baseline_ood_accuracy == pytest.approx(0.0)
    assert report.edited_train_accuracy == pytest.approx(1.0)
    assert report.edited_ood_accuracy == pytest.approx(1.0)
    assert report.random_edit_ood_accuracy == pytest.approx(0.0)
    assert report.spurious_reliance_after < report.spurious_reliance_before
    assert report.target_accuracy_drop == pytest.approx(0.0)
    assert report.ood_improvement == pytest.approx(1.0)
    assert report.random_edit_control_fails
    assert report.editing_passes


def test_residual_feature_preflight_decomposes_real_model_bridge_metric():
    clean_hidden = t.tensor([2.0, 1.0, 0.5, 0.0])
    corrupt_hidden = t.zeros(4)
    unembedding = t.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )
    clean_logits = unembedding @ clean_hidden
    corrupt_logits = unembedding @ corrupt_hidden

    report = residual_feature_preflight_report(
        clean_hidden,
        corrupt_hidden,
        unembedding,
        target_token_id=0,
        distractor_token_id=1,
        clean_logits=clean_logits,
        corrupt_logits=corrupt_logits,
        top_k=2,
        random_seed=0,
        min_recovered_fraction=0.9,
        min_random_margin=0.5,
    )

    assert report.clean_logit_diff == pytest.approx(2.5)
    assert report.corrupt_logit_diff == pytest.approx(0.0)
    assert report.total_effect == pytest.approx(2.5)
    assert report.linearization_error == pytest.approx(0.0)
    assert report.selected_feature_ids == (0, 1)
    assert report.recovered_fraction == pytest.approx(1.2)
    assert report.random_control_fails


def test_expected_pythia_sfc_dictionary_paths_cover_all_submodules():
    paths = expected_pythia_sfc_dictionary_paths(num_layers=6)

    assert len(paths) == 19
    assert paths[0] == "embed/10_32768/ae.pt"
    assert "attn_out_layer5/10_32768/ae.pt" in paths
    assert "mlp_out_layer5/10_32768/ae.pt" in paths
    assert "resid_out_layer5/10_32768/ae.pt" in paths


def test_official_sparse_feature_artifact_readiness_reports_missing_dictionaries():
    repo_files = {
        "README.md",
        "attribution.py",
        "activation_utils.py",
        "ablation.py",
        "circuit.py",
        "circuit_plotting.py",
        "dictionary_loading_utils.py",
        "loading_utils.py",
        "data/simple_train.json",
        "data/simple_test.json",
        "data/rc_train.json",
        "data/rc_test.json",
        "annotations/pythia-70m-deduped.jsonl",
        "scripts/get_circuit.sh",
        "scripts/evaluate_circuit.sh",
    }

    report = official_sparse_feature_artifact_readiness_report(
        official_repo_files=repo_files,
        hf_repo_files={"dictionaries_pythia-70m-deduped_10.zip"},
        local_dictionary_files=(),
        official_repo_commit="7fbd82b",
        hf_commit_hash="50a4344",
        hf_zip_size_bytes=2_369_891_306,
    )

    assert report.remote_manifest_passed
    assert report.hf_zip_present
    assert report.expected_dictionary_count == 19
    assert len(report.missing_dictionary_paths) == 19
    assert not report.local_dictionaries_ready
    assert not report.ready_for_gt2_replication


def test_official_sparse_feature_artifact_readiness_accepts_complete_manifest():
    repo_files = {
        "README.md",
        "attribution.py",
        "activation_utils.py",
        "ablation.py",
        "circuit.py",
        "circuit_plotting.py",
        "dictionary_loading_utils.py",
        "loading_utils.py",
        "data/simple_train.json",
        "data/simple_test.json",
        "data/rc_train.json",
        "data/rc_test.json",
        "annotations/pythia-70m-deduped.jsonl",
        "scripts/get_circuit.sh",
        "scripts/evaluate_circuit.sh",
    }

    report = official_sparse_feature_artifact_readiness_report(
        official_repo_files=repo_files,
        hf_repo_files={"dictionaries_pythia-70m-deduped_10.zip"},
        local_dictionary_files=expected_pythia_sfc_dictionary_paths(),
        official_repo_commit="7fbd82b",
        hf_commit_hash="50a4344",
        hf_zip_size_bytes=2_369_891_306,
    )

    assert report.remote_manifest_passed
    assert report.local_dictionaries_ready
    assert report.ready_for_gt2_replication


def test_sparse_autoencoder_state_dict_smoke_report_encodes_and_decodes():
    state_dict = {
        "bias": t.zeros(2),
        "encoder.weight": t.eye(2),
        "encoder.bias": t.zeros(2),
        "decoder.weight": t.eye(2),
    }

    report = sparse_autoencoder_state_dict_smoke_report(state_dict, t.tensor([1.0, 2.0]))

    assert report.activation_dim == 2
    assert report.dict_size == 2
    assert report.feature_l0 == 2
    assert report.reconstruction_mse == pytest.approx(0.0)
    assert report.relative_l2_error == pytest.approx(0.0)
    assert report.passes_smoke


def test_sparse_feature_attribution_smoke_report_uses_decoder_contributions():
    state_dict = {
        "bias": t.zeros(4),
        "encoder.weight": t.eye(4),
        "encoder.bias": t.zeros(4),
        "decoder.weight": t.eye(4),
    }
    unembedding = t.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
    )

    report = sparse_feature_attribution_smoke_report(
        t.tensor([2.0, 1.0, 0.5, 0.0]),
        t.zeros(4),
        state_dict,
        unembedding,
        target_token_id=0,
        distractor_token_id=1,
        top_k=2,
        random_seed=0,
        min_random_margin=0.5,
    )

    assert report.total_residual_effect == pytest.approx(2.5)
    assert report.decoded_feature_effect == pytest.approx(2.5)
    assert report.sae_error_effect == pytest.approx(0.0)
    assert report.selected_feature_ids == (0, 1)
    assert report.selected_effect == pytest.approx(3.0)
    assert report.recovered_fraction == pytest.approx(1.2)
    assert report.random_control_fails
    assert report.passes_smoke


def test_summarize_official_sparse_feature_circuit_artifact_counts_saved_graph_values():
    class SparseActLike:
        def __init__(self, act, resc):
            self.act = act
            self.resc = resc

    with t.sparse.check_sparse_tensor_invariants():
        edge = t.sparse_coo_tensor(
            indices=t.tensor([[0, 2, 3], [1, 0, 2]]),
            values=t.tensor([0.2, -0.05, 0.12]),
            size=(4, 4),
        )
    artifact = {
        "examples": [{"clean": "The cats"}, {"clean": "The dogs"}],
        "nodes": {
            "embed": SparseActLike(t.tensor([0.3, 0.1]), t.tensor([0.4])),
            "resid_0": SparseActLike(t.tensor([0.05, -0.21]), t.tensor([0.0])),
        },
        "edges": {"embed": {"resid_0": edge}},
    }

    report = summarize_official_sparse_feature_circuit_artifact(
        artifact,
        circuit_path="circuit.pt",
        figure_path="graph.png",
        node_threshold=0.2,
        edge_threshold=0.1,
        expected_node_submodule_count=2,
    )

    assert report.examples == 2
    assert report.node_submodule_count == 2
    assert report.edge_group_count == 1
    assert report.thresholded_node_count == 3
    assert report.thresholded_edge_count == 2
    assert report.max_abs_node_effect == pytest.approx(0.4)
    assert report.max_abs_edge_effect == pytest.approx(0.2)
    assert report.passes_smoke


def test_official_sparse_feature_circuit_artifact_report_handles_missing_artifact(tmp_path):
    report = official_sparse_feature_circuit_artifact_report(
        tmp_path / "missing.pt",
        figure_path=tmp_path / "missing.png",
    )

    assert not report.circuit_present
    assert not report.figure_present
    assert not report.passes_smoke

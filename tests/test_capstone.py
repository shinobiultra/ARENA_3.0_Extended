import importlib.util
import sys
from pathlib import Path

import pytest

root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "arena_ext").exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch is not installed")

if TORCH_AVAILABLE:
    from arena_ext.capstone import (
        ActivationOracleCapstoneConfig,
        baseline_suite_report,
        build_activation_oracle_capstone_batch,
        build_capstone_plan,
        capstone_readiness_report,
        causal_validation_suite_report,
        reproducibility_report,
        run_activation_oracle_capstone_experiment,
        summarize_activation_oracle_capstone,
    )


def test_build_capstone_plan_strips_and_records_fields():
    plan = build_capstone_plan(
        research_question=" Do mini Activation Oracles beat probes? ",
        benchmark="held-out activation questions",
        baselines=["probe", "text_only", "random_control"],
        mechanistic_claim="question conditioning uses latent state features",
        causal_validations=["ablation", "patching", "random_control", "ood"],
        reproducible_scripts=["scripts/run_capstone.py"],
        writeup_path="reports/capstone.md",
    )

    assert plan.research_question == "Do mini Activation Oracles beat probes?"
    assert plan.baselines == ("probe", "text_only", "random_control")
    assert plan.writeup_path == "reports/capstone.md"


def test_baseline_suite_report_finds_missing_baselines():
    report = baseline_suite_report(["probe", "random_control"])

    assert report.missing_baselines == ("text_only",)
    assert not report.complete


def test_causal_validation_suite_report_requires_core_validations():
    report = causal_validation_suite_report(
        ["ablation", "patching", "random_control", "ood"]
    )

    assert report.has_ablation
    assert report.has_patching
    assert report.has_random_control
    assert report.has_ood
    assert report.complete


def test_reproducibility_report_requires_scripts_seeds_and_artifacts(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "scripts/run_capstone.py").write_text("print('ok')\n")
    (tmp_path / "results/metrics.json").write_text("{}\n")

    report = reproducibility_report(
        script_paths=["scripts/run_capstone.py"],
        seeds=[0, 1, 2],
        artifact_paths=["results/metrics.json"],
        root=tmp_path,
    )

    assert report.reproducible
    assert report.seeds == (0, 1, 2)

    missing = reproducibility_report(
        script_paths=["scripts/missing.py"],
        seeds=[0],
        artifact_paths=["results/metrics.json"],
        root=tmp_path,
    )
    assert not missing.reproducible


def test_capstone_readiness_report_requires_all_contract_pieces(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "results").mkdir()
    (tmp_path / "scripts/run_capstone.py").write_text("print('ok')\n")
    (tmp_path / "results/metrics.json").write_text("{}\n")

    plan = build_capstone_plan(
        research_question="Do mini Activation Oracles beat probes?",
        benchmark="held-out activation questions",
        baselines=["probe", "text_only", "random_control"],
        mechanistic_claim="question conditioning uses latent state features",
        causal_validations=["ablation", "patching", "random_control", "ood"],
        reproducible_scripts=["scripts/run_capstone.py"],
        writeup_path="reports/capstone.md",
    )
    baselines = baseline_suite_report(list(plan.baselines))
    validations = causal_validation_suite_report(list(plan.causal_validations))
    reproducibility = reproducibility_report(
        script_paths=list(plan.reproducible_scripts),
        seeds=[0, 1, 2],
        artifact_paths=["results/metrics.json"],
        root=tmp_path,
    )

    report = capstone_readiness_report(plan, baselines, validations, reproducibility)

    assert report.has_research_question
    assert report.baseline_suite_complete
    assert report.causal_validation_complete
    assert report.reproducibility_complete
    assert report.ready


def test_activation_oracle_capstone_batch_is_balanced():
    config = ActivationOracleCapstoneConfig(examples_per_template=16)
    batch = build_activation_oracle_capstone_batch(
        seed=0,
        template_ids=(0, 1),
        config=config,
    )

    assert batch.activations.shape == (16 * 2 * config.n_questions, config.d_model)
    assert batch.question_ids.bincount().tolist() == [32, 32, 32, 32]
    assert batch.answer_ids.float().mean().item() == 0.5
    xor_mask = batch.question_ids.eq(3)
    expected_xor = batch.latent_bits[xor_mask, 0].bitwise_xor(batch.latent_bits[xor_mask, 1])
    assert batch.answer_ids[xor_mask].equal(expected_xor)


def test_activation_oracle_capstone_runs_live_training_on_small_config():
    config = ActivationOracleCapstoneConfig(
        examples_per_template=16,
        hidden_size=32,
        oracle_steps=80,
        text_only_steps=30,
        probe_steps=50,
        label_shuffle_steps=50,
    )
    result = run_activation_oracle_capstone_experiment(
        seeds=(0,),
        config=config,
        device="cpu",
    )
    summary = result["summary"]

    assert summary["oracle_accuracy_mean"] >= 0.90
    assert summary["text_only_accuracy_mean"] == 0.5
    assert (
        summary["oracle_compositional_accuracy_mean"]
        > summary["linear_probe_compositional_accuracy_mean"] + 0.2
    )
    assert summary["ablation_drop_mean"] > 0.2
    assert summary["counterfactual_patch_target_accuracy_mean"] >= 0.90
    assert summary["random_patch_change_rate_mean"] <= 0.15
    assert summary["label_shuffle_accuracy_mean"] <= 0.65


def test_capstone_summary_rejects_missing_controls():
    seed_report = {
        "seed": 0,
        "train_example_count": 128,
        "iid_example_count": 128,
        "heldout_template_example_count": 64,
        "oracle_accuracy": 1.0,
        "oracle_compositional_accuracy": 1.0,
        "text_only_accuracy": 0.5,
        "linear_probe_bank_accuracy": 0.75,
        "linear_probe_compositional_accuracy": 0.5,
        "heldout_template_accuracy": 1.0,
        "ablation_drop": 0.0,
        "counterfactual_patch_change_rate": 0.0,
        "counterfactual_patch_target_accuracy": 0.0,
        "random_patch_change_rate": 0.0,
        "random_activation_mean_confidence": 0.5,
        "random_activation_accuracy": 0.5,
        "label_shuffle_accuracy": 0.5,
    }

    summary = summarize_activation_oracle_capstone([seed_report] * 3)

    assert summary["oracle_beats_text_only"]
    assert not summary["causal_controls_passed"]
    assert not summary["preflight_passed"]

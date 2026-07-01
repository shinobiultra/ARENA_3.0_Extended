from collections.abc import Callable
import json
from pathlib import Path

from arena_ext import capstone as reference


def _solutions():
    from chapter10_capstone_research_sprint.exercises.part1_capstone_research_sprint import (
        solutions,
    )

    return solutions


def _section_dir() -> Path:
    return Path(__file__).resolve().parent


def _gpu_report() -> dict:
    report = json.loads((_section_dir() / "verification_report.json").read_text())
    return report["metrics"]["gpu_test"]


def _assert_report_matches_reference(actual: object, expected: object, *, msg: str) -> None:
    actual_dict = actual.__dict__
    expected_dict = expected.__dict__
    assert actual_dict.keys() == expected_dict.keys(), (
        f"{msg} should expose the same fields as the independent reference."
    )
    assert actual_dict == expected_dict, (
        f"{msg} should match the independent reference exactly.\n"
        f"Expected: {expected_dict}\nGot: {actual_dict}"
    )


def test_build_capstone_plan_normalizes_blank_fields(
    build_capstone_plan: Callable | None = None,
):
    build_capstone_plan = build_capstone_plan or _solutions().build_capstone_plan
    actual = build_capstone_plan(
        research_question="  Do mini Activation Oracles beat probes?  ",
        benchmark=" held-out activation questions ",
        baselines=[" probe ", "", "text_only", "random_control"],
        mechanistic_claim=" question conditioning uses latent state features ",
        causal_validations=["ablation", "", "patching", "random_control", "ood"],
        reproducible_scripts=[" scripts/run_capstone.py ", ""],
        writeup_path=" reports/capstone.md ",
    )
    expected = reference.build_capstone_plan(
        research_question="  Do mini Activation Oracles beat probes?  ",
        benchmark=" held-out activation questions ",
        baselines=[" probe ", "", "text_only", "random_control"],
        mechanistic_claim=" question conditioning uses latent state features ",
        causal_validations=["ablation", "", "patching", "random_control", "ood"],
        reproducible_scripts=[" scripts/run_capstone.py ", ""],
        writeup_path=" reports/capstone.md ",
    )
    _assert_report_matches_reference(actual, expected, msg="Capstone plan")
    assert actual.baselines == ("probe", "text_only", "random_control"), (
        "Blank baselines should be dropped and nonblank baselines should be stripped."
    )
    assert actual.reproducible_scripts == ("scripts/run_capstone.py",), (
        "The readiness gate should record clean script paths, not raw user input."
    )
    print("All tests in `test_build_capstone_plan_normalizes_blank_fields` passed!")


def test_baseline_suite_report_identifies_missing_required_baseline(
    baseline_suite_report: Callable | None = None,
):
    baseline_suite_report = baseline_suite_report or _solutions().baseline_suite_report
    actual = baseline_suite_report(["probe", "random_control"])
    expected = reference.baseline_suite_report(["probe", "random_control"])
    _assert_report_matches_reference(actual, expected, msg="Baseline suite report")
    assert actual.missing_baselines == ("text_only",), (
        "A capstone without the text-only baseline should not pass the baseline gate."
    )
    assert not actual.complete, (
        "Baseline completeness should be false until every required baseline is present."
    )
    print(
        "All tests in `test_baseline_suite_report_identifies_missing_required_baseline` passed!"
    )


def test_baseline_smoke_test_has_required_controls(
    baseline_smoke_test: Callable | None = None,
):
    baseline_smoke_test = baseline_smoke_test or _solutions().baseline_smoke_test
    result = baseline_smoke_test()
    assert result["required_baselines"] == ("probe", "text_only", "random_control"), (
        "The example capstone should require probe, text-only, and random-control baselines."
    )
    assert result["present_baselines"] == ("probe", "text_only", "random_control"), (
        "The example capstone should include every required baseline."
    )
    assert result["missing_baselines"] == (), (
        "The example capstone should have no missing baseline controls."
    )
    assert result["complete"], "The example baseline suite should pass."
    print("All tests in `test_baseline_smoke_test_has_required_controls` passed!")


def test_causal_validation_suite_report_accepts_equivalent_names(
    causal_validation_suite_report: Callable | None = None,
):
    causal_validation_suite_report = (
        causal_validation_suite_report or _solutions().causal_validation_suite_report
    )
    actual = causal_validation_suite_report(
        ["ablation", "counterfactual_patching", "random_control", "heldout_templates"],
    )
    expected = reference.causal_validation_suite_report(
        ["ablation", "counterfactual_patching", "random_control", "heldout_templates"],
    )
    _assert_report_matches_reference(actual, expected, msg="Causal validation suite report")
    assert actual.has_patching and actual.has_ood and actual.complete, (
        "Counterfactual patching and held-out templates should satisfy the patching/OOD gates."
    )
    missing_ood = causal_validation_suite_report(["ablation", "patching", "random_control"])
    assert not missing_ood.complete and not missing_ood.has_ood, (
        "A causal validation suite without an OOD or held-out-template check should fail."
    )
    print(
        "All tests in `test_causal_validation_suite_report_accepts_equivalent_names` passed!"
    )


def test_reproducibility_report_requires_scripts_seeds_and_artifacts(
    reproducibility_report: Callable | None = None,
):
    reproducibility_report = reproducibility_report or _solutions().reproducibility_report
    root = _section_dir()
    actual = reproducibility_report(
        script_paths=[" scripts/run_capstone.py "],
        seeds=[0, 1, 2],
        artifact_paths=[" results/metrics.json "],
        root=root,
    )
    expected = reference.reproducibility_report(
        script_paths=[" scripts/run_capstone.py "],
        seeds=[0, 1, 2],
        artifact_paths=[" results/metrics.json "],
        root=root,
    )
    _assert_report_matches_reference(actual, expected, msg="Reproducibility report")
    assert actual.reproducible, (
        "Scripts, seeds, and existing artifact paths should pass the reproducibility gate."
    )
    no_artifacts = reproducibility_report(
        script_paths=["scripts/run_capstone.py"],
        seeds=[0],
        artifact_paths=[],
        root=root,
    )
    assert not no_artifacts.reproducible, (
        "A run with no declared artifacts should fail the reproducibility gate."
    )
    missing_script = reproducibility_report(
        script_paths=["scripts/missing_capstone.py"],
        seeds=[0],
        artifact_paths=["results/metrics.json"],
        root=root,
    )
    assert not missing_script.reproducible, (
        "A declared script path must exist; string metadata alone is not reproducibility."
    )
    print(
        "All tests in `test_reproducibility_report_requires_scripts_seeds_and_artifacts` passed!"
    )


def test_capstone_readiness_report_requires_every_gate(
    build_capstone_plan: Callable | None = None,
    baseline_suite_report: Callable | None = None,
    causal_validation_suite_report: Callable | None = None,
    reproducibility_report: Callable | None = None,
    capstone_readiness_report: Callable | None = None,
):
    solutions = _solutions()
    build_capstone_plan = build_capstone_plan or solutions.build_capstone_plan
    baseline_suite_report = baseline_suite_report or solutions.baseline_suite_report
    causal_validation_suite_report = (
        causal_validation_suite_report or solutions.causal_validation_suite_report
    )
    reproducibility_report = reproducibility_report or solutions.reproducibility_report
    capstone_readiness_report = capstone_readiness_report or solutions.capstone_readiness_report

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
        root=_section_dir(),
    )
    actual = capstone_readiness_report(plan, baselines, validations, reproducibility)
    expected = reference.capstone_readiness_report(
        reference.build_capstone_plan(
            research_question="Do mini Activation Oracles beat probes?",
            benchmark="held-out activation questions",
            baselines=["probe", "text_only", "random_control"],
            mechanistic_claim="question conditioning uses latent state features",
            causal_validations=["ablation", "patching", "random_control", "ood"],
            reproducible_scripts=["scripts/run_capstone.py"],
            writeup_path="reports/capstone.md",
        ),
        reference.baseline_suite_report(["probe", "text_only", "random_control"]),
        reference.causal_validation_suite_report(["ablation", "patching", "random_control", "ood"]),
        reference.reproducibility_report(
            script_paths=["scripts/run_capstone.py"],
            seeds=[0, 1, 2],
            artifact_paths=["results/metrics.json"],
            root=_section_dir(),
        ),
    )
    _assert_report_matches_reference(actual, expected, msg="Capstone readiness report")
    assert actual.ready, "A complete plan should pass the capstone readiness gate."

    incomplete_baselines = baseline_suite_report(["probe", "random_control"])
    blocked = capstone_readiness_report(plan, incomplete_baselines, validations, reproducibility)
    assert not blocked.ready and not blocked.baseline_suite_complete, (
        "Readiness should fail when a required baseline is missing."
    )
    print("All tests in `test_capstone_readiness_report_requires_every_gate` passed!")


def test_notebook_contract(run_smoke_test: Callable | None = None):
    run_smoke_test = run_smoke_test or _solutions().run_smoke_test
    result = run_smoke_test(cpu=True)
    assert result["plan"]["research_question"] == "Do mini Activation Oracles beat probes?", (
        "The notebook contract should keep the example research question fixed."
    )
    assert result["plan"]["benchmark"] == "held-out activation questions", (
        "The notebook contract should expose the benchmark used in the verification report."
    )
    assert result["baselines"]["complete"], (
        "The notebook contract should include a complete baseline suite."
    )
    assert result["validations"]["complete"], (
        "The notebook contract should include ablation, patching, random control, and OOD checks."
    )
    assert result["reproducibility"]["reproducible"], (
        "The notebook contract should record scripts, seeds, and output artifacts."
    )
    assert result["readiness"]["ready"], (
        "The notebook contract should pass the capstone readiness gate."
    )
    print("All tests in `test_notebook_contract` passed!")


def test_committed_gpu_report_matches_artifact_contract():
    gpu = _gpu_report()
    assert gpu["cuda_available"], "The committed report should come from a CUDA run."
    assert gpu["baseline_suite_complete"], (
        "The CUDA report should show the baseline suite was complete."
    )
    assert gpu["causal_validation_complete"], (
        "The CUDA report should show the causal validation suite was complete."
    )
    assert gpu["reproducible"], (
        "The CUDA report should show scripts, seeds, and artifacts were recorded."
    )
    assert gpu["script_paths_exist"] and gpu["artifact_paths_exist"], (
        "The CUDA report should prove declared scripts and artifacts exist on disk."
    )
    assert gpu["capstone_pipeline_executed"], (
        "The CUDA report should execute the capstone pipeline."
    )
    assert gpu["live_training_executed"], (
        "The CUDA report should come from a trained activation-oracle experiment."
    )
    assert gpu["metrics_file_valid"] and gpu["writeup_file_valid"], (
        "The CUDA report should validate the generated metrics and writeup artifacts."
    )
    assert gpu["metrics_by_seed_file_valid"] and gpu["seed_count"] == 3, (
        "The CUDA report should record per-seed metrics for three seeds."
    )
    assert gpu["oracle_accuracy_mean"] >= 0.9, (
        "The activation oracle should solve the IID held-out questions."
    )
    assert gpu["oracle_beats_text_only"], (
        "The oracle should beat the text-only question-prior baseline."
    )
    assert gpu["oracle_beats_linear_probe_bank"], (
        "The oracle should beat the bank of activation-only linear probes overall."
    )
    assert gpu["compositional_oracle_beats_linear_probe"], (
        "The question-conditioned oracle should beat linear probes on the XOR question."
    )
    assert gpu["heldout_template_accuracy_mean"] >= 0.9 and gpu["ood_passed"], (
        "The capstone report should include a passing held-out-template OOD split."
    )
    assert gpu["ablation_drop_mean"] > 0.2 and gpu["causal_controls_passed"], (
        "Ablating relevant latent dimensions should damage oracle accuracy."
    )
    assert gpu["counterfactual_patch_target_accuracy_mean"] >= 0.9, (
        "Counterfactual latent patching should usually recover the donor answer."
    )
    assert gpu["random_patch_change_rate_mean"] <= 0.15, (
        "Random-dimension patching should be weaker than targeted patching."
    )
    assert gpu["random_activation_control_passed"], (
        "The random-activation control should not solve the benchmark."
    )
    assert gpu["label_shuffle_control_passed"], (
        "The label-shuffle control should not solve the benchmark."
    )
    assert gpu["ready"] and gpu["preflight_passed"], (
        "The CUDA report should pass the live capstone preflight."
    )
    assert gpu["peak_vram_gb"] <= 1.0 and gpu["within_vram_budget"], (
        "The 10.1 live capstone preflight should stay under the 1 GB artifact budget."
    )
    print("All tests in `test_committed_gpu_report_matches_artifact_contract` passed!")

# %%
"""Reference solutions for [10.1] Capstone Research Sprint."""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t


MAIN = __name__ == "__main__"
CAPSTONE_DIR = Path(__file__).resolve().parent


# %%
@dataclass(frozen=True)
class CapstonePlan:
    research_question: str
    benchmark: str
    baselines: tuple[str, ...]
    mechanistic_claim: str
    causal_validations: tuple[str, ...]
    reproducible_scripts: tuple[str, ...]
    writeup_path: str


@dataclass(frozen=True)
class BaselineSuiteReport:
    required_baselines: tuple[str, ...]
    present_baselines: tuple[str, ...]
    missing_baselines: tuple[str, ...]
    complete: bool


@dataclass(frozen=True)
class CausalValidationSuiteReport:
    validations: tuple[str, ...]
    has_ablation: bool
    has_patching: bool
    has_random_control: bool
    has_ood: bool
    complete: bool


@dataclass(frozen=True)
class ReproducibilityReport:
    script_paths: tuple[str, ...]
    seeds: tuple[int, ...]
    artifact_paths: tuple[str, ...]
    reproducible: bool


@dataclass(frozen=True)
class CapstoneReadinessReport:
    has_research_question: bool
    has_benchmark: bool
    has_mechanistic_claim: bool
    baseline_suite_complete: bool
    causal_validation_complete: bool
    reproducibility_complete: bool
    has_writeup_path: bool
    ready: bool


# %%
def build_capstone_plan(
    *,
    research_question: str,
    benchmark: str,
    baselines: list[str],
    mechanistic_claim: str,
    causal_validations: list[str],
    reproducible_scripts: list[str],
    writeup_path: str,
) -> CapstonePlan:
    """Bundle a paper-style capstone plan after normalizing blank fields."""

    return CapstonePlan(
        research_question=research_question.strip(),
        benchmark=benchmark.strip(),
        baselines=tuple(baseline.strip() for baseline in baselines if baseline.strip()),
        mechanistic_claim=mechanistic_claim.strip(),
        causal_validations=tuple(
            validation.strip() for validation in causal_validations if validation.strip()
        ),
        reproducible_scripts=tuple(
            script.strip() for script in reproducible_scripts if script.strip()
        ),
        writeup_path=writeup_path.strip(),
    )


def baseline_suite_report(
    present_baselines: list[str],
    *,
    required_baselines: tuple[str, ...] = ("probe", "text_only", "random_control"),
) -> BaselineSuiteReport:
    """Check whether the capstone includes each required baseline exactly once."""

    present = tuple(baseline.strip() for baseline in present_baselines if baseline.strip())
    present_set = set(present)
    required_set = set(required_baselines)
    missing = tuple(baseline for baseline in required_baselines if baseline not in present_set)
    has_duplicates = len(present_set) != len(present)
    has_unknown = not present_set <= required_set
    return BaselineSuiteReport(
        required_baselines=required_baselines,
        present_baselines=present,
        missing_baselines=missing,
        complete=len(missing) == 0 and not has_duplicates and not has_unknown,
    )


def causal_validation_suite_report(
    validations: list[str],
) -> CausalValidationSuiteReport:
    """Check whether causal, random-control, and OOD validations are present."""

    normalized = tuple(
        validation.strip().lower() for validation in validations if validation.strip()
    )
    validation_set = set(normalized)
    has_ablation = "ablation" in validation_set
    has_patching = "patching" in validation_set or "counterfactual_patching" in validation_set
    has_random_control = "random_control" in validation_set
    has_ood = "ood" in validation_set or "heldout_templates" in validation_set
    complete = has_ablation and has_patching and has_random_control and has_ood
    return CausalValidationSuiteReport(
        validations=normalized,
        has_ablation=has_ablation,
        has_patching=has_patching,
        has_random_control=has_random_control,
        has_ood=has_ood,
        complete=complete,
    )


def reproducibility_report(
    *,
    script_paths: list[str],
    seeds: list[int],
    artifact_paths: list[str],
    root: str | Path | None = None,
) -> ReproducibilityReport:
    """Check whether runnable scripts, seeds, and output artifacts exist."""

    scripts = tuple(path.strip() for path in script_paths if path.strip())
    artifacts = tuple(path.strip() for path in artifact_paths if path.strip())
    seed_tuple = tuple(
        seed for seed in seeds if isinstance(seed, int) and not isinstance(seed, bool)
    )
    seeds_valid = len(seed_tuple) == len(seeds) and len(set(seed_tuple)) == len(seed_tuple)
    root_path = Path.cwd() if root is None else Path(root)
    paths_are_relative = all(
        not Path(path).is_absolute() and ".." not in Path(path).parts
        for path in (*scripts, *artifacts)
    )
    scripts_exist = paths_are_relative and all((root_path / script).is_file() for script in scripts)
    artifacts_exist = paths_are_relative and all(
        (root_path / artifact).is_file() for artifact in artifacts
    )
    return ReproducibilityReport(
        script_paths=scripts,
        seeds=seed_tuple,
        artifact_paths=artifacts,
        reproducible=bool(
            scripts
            and seed_tuple
            and artifacts
            and seeds_valid
            and scripts_exist
            and artifacts_exist
        ),
    )


def capstone_readiness_report(
    plan: CapstonePlan,
    baselines: BaselineSuiteReport,
    validations: CausalValidationSuiteReport,
    reproducibility: ReproducibilityReport,
) -> CapstoneReadinessReport:
    """Gate a capstone plan before treating it as paper-style evidence."""

    has_question = bool(plan.research_question)
    has_benchmark = bool(plan.benchmark)
    has_claim = bool(plan.mechanistic_claim)
    has_writeup = bool(plan.writeup_path)
    ready = (
        has_question
        and has_benchmark
        and has_claim
        and baselines.complete
        and validations.complete
        and reproducibility.reproducible
        and has_writeup
    )
    return CapstoneReadinessReport(
        has_research_question=has_question,
        has_benchmark=has_benchmark,
        has_mechanistic_claim=has_claim,
        baseline_suite_complete=baselines.complete,
        causal_validation_complete=validations.complete,
        reproducibility_complete=reproducibility.reproducible,
        has_writeup_path=has_writeup,
        ready=ready,
    )


# %%
def _example_plan() -> CapstonePlan:
    return build_capstone_plan(
        research_question="Do mini Activation Oracles beat probes?",
        benchmark="held-out activation questions",
        baselines=["probe", "text_only", "random_control"],
        mechanistic_claim="question conditioning uses latent state features",
        causal_validations=["ablation", "patching", "random_control", "ood"],
        reproducible_scripts=["scripts/run_capstone.py"],
        writeup_path="reports/capstone.md",
    )


def plan_smoke_test() -> dict:
    return _example_plan().__dict__


def baseline_smoke_test() -> dict:
    plan = _example_plan()
    return baseline_suite_report(list(plan.baselines)).__dict__


def validation_smoke_test() -> dict:
    plan = _example_plan()
    return causal_validation_suite_report(list(plan.causal_validations)).__dict__


def reproducibility_smoke_test() -> dict:
    plan = _example_plan()
    return reproducibility_report(
        script_paths=list(plan.reproducible_scripts),
        seeds=[0, 1, 2],
        artifact_paths=["results/metrics.json"],
        root=CAPSTONE_DIR,
    ).__dict__


def readiness_smoke_test() -> dict:
    plan = _example_plan()
    baselines = baseline_suite_report(list(plan.baselines))
    validations = causal_validation_suite_report(list(plan.causal_validations))
    reproducibility = reproducibility_report(
        script_paths=list(plan.reproducible_scripts),
        seeds=[0, 1, 2],
        artifact_paths=["results/metrics.json"],
        root=CAPSTONE_DIR,
    )
    return capstone_readiness_report(
        plan,
        baselines,
        validations,
        reproducibility,
    ).__dict__


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "plan": plan_smoke_test(),
        "baselines": baseline_smoke_test(),
        "validations": validation_smoke_test(),
        "reproducibility": reproducibility_smoke_test(),
        "readiness": readiness_smoke_test(),
    }


def _mean_metric(by_seed: list[dict], key: str) -> float:
    return sum(float(seed_report[key]) for seed_report in by_seed) / len(by_seed)


def _close_enough(left: float, right: float, *, atol: float = 1e-6) -> bool:
    return abs(float(left) - float(right)) <= atol


def _metrics_by_seed_file_valid(by_seed: list[dict], metrics: dict) -> bool:
    required_keys = {
        "seed",
        "oracle_accuracy",
        "text_only_accuracy",
        "linear_probe_bank_accuracy",
        "linear_probe_compositional_accuracy",
        "heldout_template_accuracy",
        "ablation_drop",
        "counterfactual_patch_change_rate",
        "counterfactual_patch_target_accuracy",
        "random_patch_change_rate",
        "random_activation_accuracy",
        "label_shuffle_accuracy",
    }
    if not isinstance(by_seed, list) or len(by_seed) != metrics.get("seed_count"):
        return False
    if [int(seed_report.get("seed", -1)) for seed_report in by_seed] != metrics.get("seeds"):
        return False
    if not all(required_keys <= set(seed_report) for seed_report in by_seed):
        return False
    summary_pairs = {
        "oracle_accuracy_mean": "oracle_accuracy",
        "text_only_accuracy_mean": "text_only_accuracy",
        "linear_probe_bank_accuracy_mean": "linear_probe_bank_accuracy",
        "linear_probe_compositional_accuracy_mean": "linear_probe_compositional_accuracy",
        "heldout_template_accuracy_mean": "heldout_template_accuracy",
        "ablation_drop_mean": "ablation_drop",
        "counterfactual_patch_change_rate_mean": "counterfactual_patch_change_rate",
        "counterfactual_patch_target_accuracy_mean": "counterfactual_patch_target_accuracy",
        "random_patch_change_rate_mean": "random_patch_change_rate",
        "random_activation_accuracy_mean": "random_activation_accuracy",
        "label_shuffle_accuracy_mean": "label_shuffle_accuracy",
    }
    return all(
        _close_enough(metrics[summary_key], _mean_metric(by_seed, seed_key))
        for summary_key, seed_key in summary_pairs.items()
    )


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("CUDA is required for the 10.1 capstone experiment preflight.")

    t.cuda.reset_peak_memory_stats()
    subprocess.run(
        [
            sys.executable,
            str(CAPSTONE_DIR / "scripts" / "run_capstone.py"),
            "--output-dir",
            str(CAPSTONE_DIR),
            "--device",
            "cuda",
            "--max-vram-gb",
            str(max_vram_gb),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    plan = _example_plan()
    baselines = baseline_suite_report(list(plan.baselines))
    validations = causal_validation_suite_report(list(plan.causal_validations))
    reproducibility = reproducibility_report(
        script_paths=list(plan.reproducible_scripts),
        seeds=[0, 1, 2],
        artifact_paths=["results/metrics.json"],
        root=CAPSTONE_DIR,
    )
    readiness = capstone_readiness_report(plan, baselines, validations, reproducibility)
    metrics_path = CAPSTONE_DIR / "results" / "metrics.json"
    by_seed_path = CAPSTONE_DIR / "results" / "metrics_by_seed.json"
    failure_cases_path = CAPSTONE_DIR / "results" / "failure_cases.jsonl"
    writeup_path = CAPSTONE_DIR / "reports" / "capstone.md"
    metrics = json.loads(metrics_path.read_text())
    by_seed = json.loads(by_seed_path.read_text())
    metrics_file_valid = bool(
        metrics.get("preflight_passed")
        and metrics.get("seed_count") == 3
        and metrics.get("oracle_accuracy_mean", 0.0) >= 0.90
        and metrics.get("oracle_beats_text_only")
        and metrics.get("oracle_beats_linear_probe_bank")
        and metrics.get("compositional_oracle_beats_linear_probe")
        and metrics.get("ood_passed")
        and metrics.get("causal_controls_passed")
        and metrics.get("random_activation_control_passed")
        and metrics.get("label_shuffle_control_passed")
    )
    writeup = writeup_path.read_text()
    writeup_file_valid = (
        "Mini Activation-Oracle Capstone Report" in writeup
        and "Counterfactual patch" in writeup
        and "Limitations" in writeup
    )
    t.cuda.synchronize()
    peak_vram_gb = max(
        t.cuda.max_memory_allocated() / 1024**3,
        float(metrics.get("peak_vram_gb", 0.0)),
    )
    failure_cases_exist = failure_cases_path.is_file()
    metrics_by_seed_file_valid = _metrics_by_seed_file_valid(by_seed, metrics)
    return {
        **metrics,
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "baseline_suite_complete": baselines.complete,
        "causal_validation_complete": validations.complete,
        "reproducible": reproducibility.reproducible,
        "script_paths_exist": all(
            (CAPSTONE_DIR / script).is_file() for script in plan.reproducible_scripts
        ),
        "artifact_paths_exist": all(
            (CAPSTONE_DIR / artifact).is_file() for artifact in reproducibility.artifact_paths
        )
        and by_seed_path.is_file()
        and failure_cases_exist
        and writeup_path.is_file(),
        "capstone_pipeline_executed": True,
        "live_training_executed": True,
        "metrics_file_valid": metrics_file_valid,
        "writeup_file_valid": writeup_file_valid,
        "metrics_by_seed_file_valid": metrics_by_seed_file_valid,
        "failure_cases_file_exists": failure_cases_exist,
        "ready": readiness.ready,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "preflight_passed": (
            readiness.ready
            and metrics_file_valid
            and writeup_file_valid
            and metrics_by_seed_file_valid
            and len(by_seed) == 3
            and peak_vram_gb <= max_vram_gb
        ),
        "full_path": (
            "Train a question-conditioned activation oracle on CUDA, compare it "
            "against text-only and linear-probe baselines, then run OOD, ablation, "
            "counterfactual patching, random-patch, random-activation, and "
            "label-shuffle controls."
        ),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())

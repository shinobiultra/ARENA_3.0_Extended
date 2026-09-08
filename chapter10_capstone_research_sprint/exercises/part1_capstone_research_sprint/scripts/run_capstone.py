"""Run the exact XOR-direction capstone study and write compact artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


SECTION_DIR = Path(__file__).resolve().parents[1]


def _load_solutions() -> ModuleType:
    path = SECTION_DIR / "solutions.py"
    spec = importlib.util.spec_from_file_location("capstone_runner_solutions", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_report(path: Path, metrics: dict[str, object]) -> None:
    noise_sigmas = metrics["noise_sigmas"]
    noise_accuracies = metrics["noise_accuracies"]
    lines = [
        "# Exact XOR-Direction Capstone Result",
        "",
        "## Preregistered claim",
        "",
        (
            "A ridge direction fitted on balanced train templates will recover the exact "
            "distributed XOR mediator, generalize to held-out templates, and causally "
            "transfer counterfactual donor answers beyond matched controls."
        ),
        "",
        "## Result",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Held-out activation accuracy | {metrics['heldout_accuracy']:.3f} |",
        f"| Raw-bit linear baseline | {metrics['raw_bits_accuracy']:.3f} |",
        f"| Template-only baseline | {metrics['template_only_accuracy']:.3f} |",
        f"| Shuffled-label mean | {metrics['label_shuffle_accuracy_mean']:.3f} |",
        f"| Exact-direction cosine | {metrics['direction_cosine']:.3f} |",
        f"| Paired accuracy delta | {metrics['paired_accuracy_delta']:+.3f} |",
        (
            "| Paired bootstrap 95% interval | "
            f"[{metrics['paired_accuracy_delta_ci_low']:+.3f}, "
            f"{metrics['paired_accuracy_delta_ci_high']:+.3f}] |"
        ),
        f"| Learned patch donor-target accuracy | {metrics['learned_patch_target_accuracy']:.3f} |",
        f"| Random-direction patch mean | {metrics['random_patch_target_accuracy_mean']:.3f} |",
        f"| Accuracy after direction ablation | {metrics['ablation_accuracy']:.3f} |",
        "",
        "## Controls and failure analysis",
        "",
        (
            "Raw-bit and template-only probes remain at chance, and shuffled-label probes "
            "average near chance. The learned intervention is compared with 256 isotropic "
            "directions of the same dimensionality."
        ),
        "",
        (
            "The strongest random direction is an instructive anomaly: it reaches "
            f"{metrics['best_random_patch_target_accuracy']:.3f} donor-target accuracy "
            "because its absolute cosine with the exact direction is "
            f"{metrics['best_random_direction_alignment']:.3f}."
        ),
        "",
        "The activation-noise stress curve is:",
        "",
        "| Sigma | Accuracy |",
        "| ---: | ---: |",
    ]
    lines.extend(
        f"| {float(sigma):.2f} | {float(accuracy):.3f} |"
        for sigma, accuracy in zip(noise_sigmas, noise_accuracies)
    )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            (
                "The organism computes parity exactly and applies a fixed orthogonal mix. "
                "This supports a method-validation claim, not a discovery about a released "
                "transformer. Held-out templates vary nuisance features but preserve the "
                "task rule. Real-model representations may be nonlinear, contextual, and "
                "unstable across inputs or layers."
            ),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(output_dir: Path, *, device: str) -> dict[str, str]:
    solutions = _load_solutions()
    metrics = solutions.run_study(device=device)
    results_dir = output_dir / "results"
    reports_dir = output_dir / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "metrics.json"
    failures_path = results_dir / "failure_cases.jsonl"
    report_path = reports_dir / "capstone.md"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    anomaly = {
        "type": "random_direction_alignment",
        "patch_target_accuracy": metrics["best_random_patch_target_accuracy"],
        "absolute_exact_direction_cosine": metrics["best_random_direction_alignment"],
        "interpretation": "the strongest isotropic null overlaps the true mechanism",
    }
    noise_failure = {
        "type": "activation_noise_boundary",
        "sigma": 1.0,
        "accuracy": metrics["noise_accuracies"][3],
        "interpretation": "clean exact recovery is not robust to unit-scale measurement noise",
    }
    failures_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in [anomaly, noise_failure]) + "\n",
        encoding="utf-8",
    )
    _write_report(report_path, metrics)
    return {
        "metrics_path": str(metrics_path),
        "failure_cases_path": str(failures_path),
        "report_path": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=SECTION_DIR)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    outputs = write_outputs(args.output_dir, device=args.device)
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

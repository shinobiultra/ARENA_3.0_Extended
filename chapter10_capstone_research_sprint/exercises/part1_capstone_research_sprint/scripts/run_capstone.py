"""Run the 10.1 mini activation-oracle capstone experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch as t

ROOT = next(
    path
    for path in [Path(__file__).resolve(), *Path(__file__).resolve().parents]
    if (path / "arena_ext").exists()
)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arena_ext.capstone import (  # noqa: E402
    QUESTION_NAMES,
    ActivationOracleCapstoneConfig,
    run_activation_oracle_capstone_experiment,
)


def _fmt(value: float) -> str:
    return f"{float(value):.3f}"


def _mean_by_question(by_seed: list[dict[str, Any]], key: str) -> dict[str, float]:
    return {
        question: sum(float(seed_report[key][question]) for seed_report in by_seed)
        / len(by_seed)
        for question in QUESTION_NAMES
    }


def _write_markdown_report(
    *,
    report_path: Path,
    summary: dict[str, Any],
    by_seed: list[dict[str, Any]],
    failure_cases: list[dict[str, Any]],
    device: str,
) -> None:
    oracle_question_means = _mean_by_question(by_seed, "oracle_accuracy_by_question")
    probe_question_means = _mean_by_question(by_seed, "linear_probe_accuracy_by_question")
    lines = [
        "# Mini Activation-Oracle Capstone Report",
        "",
        "## Claim",
        "",
        (
            "A question-conditioned MLP activation oracle can recover a controlled "
            "latent state from synthetic residual-stream activations, including a "
            "nonlinear XOR question that a bank of linear probes does not solve. "
            "The claim is scoped to this generated model-organism benchmark."
        ),
        "",
        "## Setup",
        "",
        f"- Benchmark: `{summary['benchmark']}`",
        f"- Dataset: `{summary['dataset']}`",
        f"- Device: `{device}`",
        f"- Seeds: {summary['seeds']}",
        f"- Train examples per seed: {summary['train_example_count']}",
        f"- IID test examples per seed: {summary['iid_example_count']}",
        (
            "- Held-out template examples per seed: "
            f"{summary['heldout_template_example_count']}"
        ),
        f"- Questions: {', '.join(QUESTION_NAMES)}",
        "",
        "## Results",
        "",
        "| Metric | Mean |",
        "| --- | ---: |",
        f"| Oracle accuracy | {_fmt(summary['oracle_accuracy_mean'])} |",
        f"| Text-only accuracy | {_fmt(summary['text_only_accuracy_mean'])} |",
        (
            "| Linear-probe-bank accuracy | "
            f"{_fmt(summary['linear_probe_bank_accuracy_mean'])} |"
        ),
        (
            "| Oracle XOR-question accuracy | "
            f"{_fmt(summary['oracle_compositional_accuracy_mean'])} |"
        ),
        (
            "| Linear-probe XOR-question accuracy | "
            f"{_fmt(summary['linear_probe_compositional_accuracy_mean'])} |"
        ),
        (
            "| Held-out-template accuracy | "
            f"{_fmt(summary['heldout_template_accuracy_mean'])} |"
        ),
        f"| Relevant-dimension ablation drop | {_fmt(summary['ablation_drop_mean'])} |",
        (
            "| Counterfactual patch answer-change rate | "
            f"{_fmt(summary['counterfactual_patch_change_rate_mean'])} |"
        ),
        (
            "| Counterfactual patch target accuracy | "
            f"{_fmt(summary['counterfactual_patch_target_accuracy_mean'])} |"
        ),
        (
            "| Random-dimension patch change rate | "
            f"{_fmt(summary['random_patch_change_rate_mean'])} |"
        ),
        (
            "| Random-activation accuracy | "
            f"{_fmt(summary['random_activation_accuracy_mean'])} |"
        ),
        (
            "| Random-activation confidence | "
            f"{_fmt(summary['random_activation_mean_confidence_mean'])} |"
        ),
        (
            "| Label-shuffle oracle accuracy | "
            f"{_fmt(summary['label_shuffle_accuracy_mean'])} |"
        ),
        "",
        "## Causal Validation",
        "",
        (
            "Ablating the latent dimensions used by the asked question drops oracle "
            "accuracy, while patching those dimensions from a donor example usually "
            "changes the answer to the donor answer. Patching randomly sampled "
            "non-latent control dimensions has a much smaller effect."
        ),
        "",
        "## Per-Question Means",
        "",
        "| Question | Oracle accuracy | Linear probe accuracy |",
        "| --- | ---: | ---: |",
    ]
    for question in QUESTION_NAMES:
        lines.append(
            "| "
            f"{question} | "
            f"{_fmt(oracle_question_means[question])} | "
            f"{_fmt(probe_question_means[question])} |"
        )
    lines.extend(
        [
            "",
            "## Limitations",
            "",
            (
                "This is a generated model-organism sprint, not evidence about a "
                "released transformer. Random activations are scored by accuracy, "
                "not abstention, because the binary oracle can be confidently wrong "
                "off distribution. The high random-activation confidence is recorded "
                "as a calibration limitation for the next iteration. Random-patch "
                "controls sample non-latent dimensions, so they show that unrelated "
                "control coordinates do less than targeted latent patches."
            ),
            "",
            "## Failure Cases",
            "",
        ]
    )
    if failure_cases:
        lines.append("| Split | Question | Template | Target | Prediction |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for failure in failure_cases[:8]:
            lines.append(
                "| "
                f"{failure['split']} | {failure['question']} | "
                f"{failure['template_id']} | {failure['target']} | "
                f"{failure['prediction']} |"
            )
    else:
        lines.append("No held-out-template failures were observed in the committed run.")
    lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def write_outputs(
    output_dir: Path,
    *,
    seeds: tuple[int, ...],
    device: str,
    max_vram_gb: float,
) -> dict[str, str]:
    results_dir = output_dir / "results"
    reports_dir = output_dir / "reports"
    results_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    config = ActivationOracleCapstoneConfig()
    result = run_activation_oracle_capstone_experiment(
        seeds=seeds,
        config=config,
        device=device,
        max_vram_gb=max_vram_gb,
    )
    summary = result["summary"]
    by_seed = result["by_seed"]
    failure_cases = result["failure_cases"]

    metrics_path = results_dir / "metrics.json"
    by_seed_path = results_dir / "metrics_by_seed.json"
    failure_path = results_dir / "failure_cases.jsonl"
    report_path = reports_dir / "capstone.md"
    metrics_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    by_seed_path.write_text(json.dumps(by_seed, indent=2, sort_keys=True) + "\n")
    failure_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in failure_cases)
    )
    _write_markdown_report(
        report_path=report_path,
        summary=summary,
        by_seed=by_seed,
        failure_cases=failure_cases,
        device=device,
    )
    return {
        "metrics_path": str(metrics_path),
        "metrics_by_seed_path": str(by_seed_path),
        "failure_cases_path": str(failure_path),
        "report_path": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-vram-gb", type=float, default=24.0)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Use cuda when available by default.",
    )
    args = parser.parse_args()
    device = "cuda" if args.device == "auto" and t.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not t.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false.")
    print(
        json.dumps(
            write_outputs(
                args.output_dir,
                seeds=tuple(args.seeds),
                device=device,
                max_vram_gb=args.max_vram_gb,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

"""Run the extension's real CI quality gates.

This wrapper intentionally calls existing audits and tests. It does not create
extra mock smoke tests; the hosted lane is limited to checks that can run from
committed metadata without model downloads, while the GPU lane reruns real CUDA
verification reports.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

HOSTED_CPU_PYTEST_TARGETS = [
    "tests/test_original_arena_preservation_audit.py",
    "tests/test_course_surface_audit.py",
    "tests/test_hard_exercise_ladder_audit.py",
    "tests/test_report_evidence_contract_audit.py",
    "tests/test_extension_artifact_hygiene.py",
    "tests/test_no_minified_files_audit.py",
    "tests/test_extension_verification_assets.py",
    "tests/test_build_merged_config.py",
    "tests/test_arena_style_depth_audit.py",
    "tests/test_roadmap_final_completeness.py",
    "tests/test_strict_completion_audit.py",
]


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def python_command(script: str, *args: str) -> list[str]:
    return [sys.executable, script, *args]


def run_whitespace_gate() -> None:
    run(["git", "diff", "--check"])
    run(["git", "diff", "--cached", "--check"])


def run_hosted_cpu() -> None:
    run_whitespace_gate()
    run([sys.executable, "-m", "pytest", "-q", *HOSTED_CPU_PYTEST_TARGETS])
    run(python_command("scripts/build_merged_config.py", "--check"))
    run(python_command("scripts/audit_original_arena_preservation.py"))
    run(python_command("scripts/audit_course_surface.py"))
    run(python_command("scripts/audit_arena_style_depth.py"))
    run(python_command("scripts/audit_hard_exercise_ladders.py"))
    run(python_command("scripts/audit_report_evidence_contracts.py"))
    run(python_command("scripts/audit_extension_artifact_hygiene.py"))
    run(python_command("scripts/audit_no_minified_files.py"))
    run(
        python_command(
            "scripts/audit_gpu_verification_reports.py",
            "--require-cuda-section-metrics",
        )
    )


def run_local_unit() -> None:
    run_whitespace_gate()
    run([sys.executable, "-m", "pytest", "-q"])
    run_hosted_cpu()


def run_gpu(section_filters: list[str]) -> None:
    env = os.environ.copy()
    env.setdefault("BNB_CUDA_VERSION", "130")
    run(["nvidia-smi"], env=env)
    run(
        [
            sys.executable,
            "-c",
            (
                "import torch; "
                "assert torch.cuda.is_available(); "
                "print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
            ),
        ],
        env=env,
    )
    report_command = python_command("scripts/run_extension_verification_reports.py")
    for section in section_filters:
        report_command.extend(["--section", section])
    report_command.extend(["--max-vram-gb", "24"])
    run(report_command, env=env)
    run(
        python_command(
            "scripts/audit_gpu_verification_reports.py",
            "--require-cuda-section-metrics",
        ),
        env=env,
    )
    run(python_command("scripts/audit_report_evidence_contracts.py"), env=env)


def run_release() -> None:
    run_gpu([])
    run(python_command("scripts/audit_roadmap_final_completeness.py"))
    run(python_command("scripts/audit_extension_completion_strict.py"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("hosted-cpu", "local-unit", "gpu", "release"),
        default="hosted-cpu",
    )
    parser.add_argument(
        "--section",
        action="append",
        default=[],
        help="Extension section number to regenerate in --mode gpu; omit for all sections.",
    )
    args = parser.parse_args()

    if args.mode == "hosted-cpu":
        run_hosted_cpu()
    elif args.mode == "local-unit":
        run_local_unit()
    elif args.mode == "gpu":
        run_gpu(args.section)
    else:
        run_release()


if __name__ == "__main__":
    main()

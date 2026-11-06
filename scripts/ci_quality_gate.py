"""Run the extension's CI quality gates.

The hosted lane runs cheap checks that need no model downloads: the
chapters 0-4 preservation audit, the merged-config check, the course-surface,
artifact-hygiene and no-minified audits, and the batch-exit test. Fresh-kernel
lesson runs stay out of CI; they happen on the laptop or the Julia worker
(`infrastructure/extension/build.py <section> --execute`).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

HOSTED_CPU_PYTEST_TARGETS = [
    "tests/test_original_arena_preservation_audit.py",
    "tests/test_course_surface_audit.py",
    "tests/test_extension_artifact_hygiene.py",
    "tests/test_no_minified_files_audit.py",
    "tests/test_build_merged_config.py",
    "tests/test_extension_batch_exit.py",
]

# The study reader lane checks the learner app without running lesson code.
STUDY_DELIVERY_TARGETS = [
    "tests/mastery/test_reader_routes.py",
    "tests/mastery/test_reading_entrypoint.py",
    "tests/mastery/test_blank_practice_foundation_integration.py",
    "tests/mastery/test_extension_prediction_workbook.py",
    "tests/mastery/test_app.py",
    "tests/mastery/test_app_workspace.py",
    "tests/mastery/test_empirical_protocol.py",
]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


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
    run(python_command("scripts/audit_extension_artifact_hygiene.py"))
    run(python_command("scripts/audit_no_minified_files.py"))


def run_local_unit() -> None:
    run_whitespace_gate()
    run([sys.executable, "-m", "pytest", "-q"])
    run_hosted_cpu()


def run_study_delivery() -> None:
    run([sys.executable, "-m", "pytest", "-q", *STUDY_DELIVERY_TARGETS])
    for script in (
        "scripts/build_extension_prediction_workbook.py",
        "scripts/build_circuit_prediction_notebook.py",
        "scripts/build_shapley_prediction_notebook.py",
        "scripts/build_sage_prediction_notebook.py",
        "scripts/build_ssd_prediction_notebook.py",
        "scripts/build_graph_prediction_notebook.py",
        "scripts/build_sae_prediction_notebook.py",
    ):
        run(python_command(script, "--check"))
    run(python_command("scripts/build_merged_config.py", "--check"))
    run(python_command("scripts/audit_original_arena_preservation.py"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("hosted-cpu", "study-delivery", "local-unit"),
        default="hosted-cpu",
    )
    args = parser.parse_args()

    if args.mode == "hosted-cpu":
        run_hosted_cpu()
    elif args.mode == "study-delivery":
        run_study_delivery()
    else:
        run_local_unit()


if __name__ == "__main__":
    main()

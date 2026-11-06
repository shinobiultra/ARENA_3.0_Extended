"""Regenerate actual completed-learner evidence (CUDA by default when available).

Pass --cpu-only to explicitly omit the pretrained experiment. The resulting
reference then records CUDA as absent, never as a substitute CPU success.
"""

from __future__ import annotations

import json
import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chapter15_peft_misalignment.exercises.part3_qlora_from_scratch import solutions


HERE = Path(__file__).resolve().parent / "expected_outputs"


def _write(name: str, payload: dict) -> None:
    path = HERE / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    print(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-only", action="store_true")
    args = parser.parse_args()
    section = HERE.parent
    env = {**os.environ, "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"}
    env["ARENA_QLORA_CPU_ONLY"] = "1" if args.cpu_only else "0"
    subprocess.run([sys.executable, str(section / "execute_learner.py")], cwd=ROOT, env=env, check=True)
    measured = json.loads(Path("/tmp/arena-qlora-learner-metrics.json").read_text())
    completed = json.loads((section / "15.3_QLoRA_from_Scratch_solutions.ipynb").read_text())
    code_cells = [cell for cell in completed["cells"] if cell["cell_type"] == "code"]
    assert all(cell["execution_count"] is not None for cell in code_cells)
    assert not any(output.get("output_type") == "error" for cell in code_cells for output in cell["outputs"])
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(section / "tests.py"),
         "-k", "not reference_evidence_is_current"],
        cwd=ROOT, env=env, capture_output=True, text=True, check=True,
    )
    smoke = solutions.run_smoke_test(cpu=True)
    inputs = ["solutions.py", "tests.py", "build_notebooks.py", "execute_learner.py",
              "generate_cpu_fixtures.py", "artifacts.lock.yml",
              "15.3_QLoRA_from_Scratch_exercises.ipynb", "15.3_QLoRA_from_Scratch_solutions.ipynb"]
    assets = section.parents[1] / "instructions/assets"
    figures = [assets / "qlora_nf4_toy_signature.png"]
    if measured["cuda"] is not None:
        figures.append(assets / "qlora_real_signature_panel.png")
    reference = {
        "evidence": "actual_learner_completed_from_own_dropdowns_in_fresh_kernel",
        **measured,
        "completed_learner_code_cells": len(code_cells),
        "all_code_cells_executed_without_errors": True,
        "actual_test_run": {"returncode": tests.returncode, "stdout": tests.stdout, "stderr": tests.stderr},
        "source_sha256": {name: hashlib.sha256((section / name).read_bytes()).hexdigest() for name in inputs},
        "figure_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in figures},
        "course_release_accepted": False,
    }
    _write("smoke_test.json", smoke)
    _write("reference_metrics.json", reference)


if __name__ == "__main__":
    main()

"""Reference entry points for the complete real-model refusal-direction learner.

Core methods also appear, with executable answers, in the paired notebooks.
Historical proxy implementations are archived under expected_outputs/legacy_proxy.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch as t
from chapter9_alignment_interpretability.exercises.part1_refusal_directions_safe_steering.direction_methods import (
    fit_mean_direction, pairwise_auroc, evaluate_frozen_direction, direction_null_ensemble,
    direction_geometry, edit_residual, qwen_direction_hooks, cache_qwen_residuals,
    greedy_direction_generation, refusal_marker_flags,
)
from chapter9_alignment_interpretability.exercises.part1_refusal_directions_safe_steering.direction_evaluation import (
    paired_marker_change, select_development_direction,
)


def run_smoke_test(cpu: bool = True) -> dict:
    """Expose the evaluation-shift trap and exact projection geometry on CPU."""
    device = "cpu" if cpu else "cuda"
    train = t.tensor([[4.,1.],[6.,3.],[0.,1.],[2.,3.]],device=device)
    labels = t.tensor([True,True,False,False],device=device)
    fitted = fit_mean_direction(train,labels)
    shifted = evaluate_frozen_direction(train+t.tensor([10.,0.],device=device),labels,fitted)
    removed = edit_residual(train,fitted["difference"],mode="ablate",strength=1.)
    projection_error = float((removed@fitted["unit"]).abs().max())
    return {"accepted":float(shifted["accuracy"])==.5 and float(shifted["auroc"])==1. and projection_error<1e-6,
        "scope":"analytic_train_only_threshold_and_projection_smoke",
        "shifted_accuracy":float(shifted["accuracy"]),"shifted_auroc":float(shifted["auroc"]),
        "projection_error":projection_error,"real_model_experiment":False}


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    from chapter9_alignment_interpretability.exercises.part1_refusal_directions_safe_steering.direction_cuda import run_fresh_refusal_cuda_check
    return run_fresh_refusal_cuda_check(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Build paired surfaces, execute every own answer afresh, then verify on CUDA."""
    import subprocess,tempfile
    here=Path(__file__).resolve().parent
    output=Path(tempfile.mkdtemp(prefix="arena-refusal-full-"))/"execution"
    subprocess.run([sys.executable,str(here/"build_notebooks.py")],cwd=ROOT,check=True)
    subprocess.run([sys.executable,"-u",str(here/"execute_learner.py"),"--scope","full",
                    "--output-directory",str(output),"--publish"],cwd=ROOT,check=True)
    return run_gpu_test(max_vram_gb=max_vram_gb)


if __name__ == "__main__":
    print(run_smoke_test())

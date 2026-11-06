"""Record actual CPU/CUDA maze experiments and tests without claiming course release."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import nbformat
import torch as t

SECTION = Path(__file__).resolve().parent
ROOT = SECTION.parents[2]
sys.path.insert(0, str(ROOT))
from chapter14_jepa_world_models.exercises.part4_maze_transformer_world_model import solutions

CPU_CONFIG = {'seed': 0, 'train_maze_count': 260, 'heldout_maze_count': 70, 'epochs': 10, 'd_model': 56}
CUDA_CONFIG = {'seed': 0, 'train_maze_count': 900, 'heldout_maze_count': 180, 'epochs': 18, 'd_model': 72}


def run_case(device: str, config: dict) -> dict:
    if device == 'cuda' and not t.cuda.is_available():
        raise RuntimeError('The CUDA reference requires CUDA; CPU fallback is forbidden.')
    started = time.monotonic()
    artifacts = solutions.run_signature_experiment(device=device, **config)
    result = solutions._serializable_report(artifacts)
    weights_hash = hashlib.sha256()
    for name, value in artifacts.model.state_dict().items():
        weights_hash.update(name.encode())
        weights_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    try:
        solutions.validate_experiment(artifacts, require_cuda=device == 'cuda')
        result['contract_passed'] = True
        result['contract_failure'] = None
    except AssertionError as error:
        result['contract_passed'] = False
        result['contract_failure'] = str(error)
    result.update({
        'device': device, 'config': config, 'elapsed_seconds': time.monotonic() - started,
        'torch_version': t.__version__, 'cuda_version': t.version.cuda,
        'gpu_name': t.cuda.get_device_name() if device == 'cuda' else None,
        'model_weight_sha256': weights_hash.hexdigest(),
        'measured_metrics': artifacts.metrics,
        'deterministic_algorithms': t.are_deterministic_algorithms_enabled(),
        'float32_matmul_precision': t.get_float32_matmul_precision(),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'cohorts': {'rollout_layouts': min(60, config['heldout_maze_count']),
                    'selection': 'generator prefix; required detour layouts occur first',
                    'probe_states_per_split_max': 1200,
                    'patch_pairs': 'first up to 24 correctly classified different-action route pairs'},
    })
    print(device, 'contract passed:', result['contract_passed'], result['contract_failure'], flush=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--verify-cuda-repeatability', action='store_true')
    args = parser.parse_args()
    t.set_num_threads(8)
    solved = nbformat.read(next(SECTION.glob('*solutions.ipynb')), as_version=4)
    code_cells = [cell for cell in solved.cells if cell.cell_type == 'code']
    if not all(cell.execution_count is not None for cell in code_cells):
        raise RuntimeError('Execute all completed learner cells before recording references.')
    if any(output.output_type == 'error' for cell in code_cells for output in cell.outputs):
        raise RuntimeError('The completed learner contains an execution error.')
    previous = json.loads((SECTION / 'expected_outputs/reference_metrics.json').read_text())
    cpu = run_case('cpu', CPU_CONFIG)
    cuda = None if args.cpu_only else run_case('cuda', CUDA_CONFIG)
    repeatability = None
    if args.verify_cuda_repeatability:
        if cuda is None:
            raise ValueError('CUDA repeatability cannot be checked with --cpu-only.')
        repeated = run_case('cuda', CUDA_CONFIG)
        same_weights = repeated['model_weight_sha256'] == cuda['model_weight_sha256']
        same_metrics = all(
            value == repeated['measured_metrics'][name]
            for name, value in cuda['measured_metrics'].items() if name != 'peak_vram_gb'
        )
        assert same_weights and same_metrics, 'Same-process CUDA repeatability failed.'
        repeatability = {'repeats': 2, 'same_process': True, 'identical_weights': same_weights,
                         'identical_metrics_except_peak_vram': same_metrics,
                         'model_weight_sha256': repeated['model_weight_sha256']}
    test_run = subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', str(SECTION / 'tests.py'),
         '-k', 'not reference_evidence_is_current'], cwd=ROOT, text=True, capture_output=True,
        env={**os.environ, 'OMP_NUM_THREADS': '8', 'MKL_NUM_THREADS': '8'},
    )
    print(test_run.stdout, flush=True)
    if test_run.stderr:
        print(test_run.stderr, flush=True)
    payload = {
        'protocol_version': 'maze-14.4-policy-controls-v2',
        'date_run': datetime.now(timezone.utc).isoformat(),
        'claim': 'A supervised maze policy and controlled action-state interventions; internal search is unproven.',
        'thresholds': previous['thresholds'],
        'cpu': cpu, 'cuda': cuda, 'tests_passed': test_run.returncode == 0,
        'cuda_repeatability': repeatability,
        'test_output': test_run.stdout,
        'learner': {'executed_code_cells': len(code_cells), 'error_outputs': 0},
        'accepted': False,
        'limitations': [
            'CPU and CUDA configurations and prefix-cohort detour mixtures differ; this is not a scaling comparison.',
            'Same-action donors and Manhattan distance undermine route-specific and internal-search interpretations.',
            'The larger CUDA experiment fails the original patch effect and control thresholds.',
            'Independent review, maze-topology reconstruction, adjacency-head analysis, and latent planning remain pending.',
        ],
    }
    (SECTION / 'expected_outputs/smoke_test.json').write_text(json.dumps({
        **CPU_CONFIG, 'device': 'cpu', **{key: value for key, value in cpu.items()
                                       if key not in {'training', 'probes', 'patching'}},
    }, indent=2) + '\n')
    inputs = [path for path in SECTION.rglob('*') if path.is_file()
              and '__pycache__' not in path.parts and path.name != 'reference_metrics.json']
    inputs += [SECTION.parents[1] / 'instructions/pages/04_[14.4]_Maze_Transformers_and_Latent_Planning.md',
               SECTION.parents[1] / 'instructions/assets/maze_transformer_signature_result.png']
    payload['input_sha256'] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(inputs)
    }
    (SECTION / 'expected_outputs/reference_metrics.json').write_text(json.dumps(payload, indent=2) + '\n')
    if test_run.returncode:
        raise SystemExit(test_run.returncode)


if __name__ == '__main__':
    main()

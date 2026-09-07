"""Reference imports and release commands for the learner-owned geometry study.

Open the paired notebook to implement the method and run its visible experiment.
The command below independently regenerates a fixed batch of every control.
"""
from __future__ import annotations
import inspect
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.geometry_methods import *
from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls import geometry_methods_tests


def run_smoke_test(cpu=True):
    tested=[]
    for name,func in vars(geometry_methods_tests).items():
        if name.startswith('test_') and inspect.isfunction(func):func();tested.append(name)
    return {'tests_passed':True,'accepted':True,'test_count':len(tested),'claim_scope':'Numerical method oracles; real evidence is checked separately.'}


def run_gpu_test(max_vram_gb: float = 24.0):
    from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.geometry_cuda import reproduce
    here=Path(__file__).parent
    publication=json.loads((here/'expected_outputs/learner_execution.json').read_text())
    learner=Path(os.environ.get('ARENA_GEOMETRY_LEARNER_DIRECTORY',publication['learner_directory']))
    cached_manifest=os.environ.get('ARENA_GEOMETRY_REPRODUCTION_MANIFEST')
    if cached_manifest:
        from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.geometry_reproduction_cache import validate_cached_reproduction
        fresh,output=validate_cached_reproduction(learner,Path(cached_manifest))
    else:
        report_root=learner.parent / 'fresh_reproductions'
        report_root.mkdir(parents=True,exist_ok=True)
        output=Path(tempfile.mkdtemp(prefix='run-',dir=report_root))/'fresh_cuda'
        fresh=reproduce(learner,output)
    audit=fresh['learner_audit'];peak=max(fresh['peak_cuda_allocated_gib'],audit['runtime']['peak_cuda_allocated_gib'])
    if peak>max_vram_gb:raise ValueError(f'Actual CUDA peak {peak:.3f} GiB exceeds the requested budget.')
    return {'cuda_available':True,'gpu_name':fresh['gpu'],'peak_vram_gb':peak,'within_vram_budget':True,
            'actual_learner_passed':True,'own_answer_code_cells':audit['own_answer_code_cells'],'embedded_figures':audit['embedded_figures'],
            'final_prompts':audit['final_prompts'],'all_donor_pairs':audit['all_donor_pairs'],'retained_outcomes':audit['retained_outcomes'],
            'null_draws_per_kind':32,'projection_fits':audit['projection_fits'],'angle_play_rows':audit['angle_play_rows'],
            'fresh_compared_examples':fresh['compared_examples'],'fresh_condition_batches':fresh['condition_batches'],
            'fresh_max_logit_error':fresh['max_calendar_logit_error'],'fresh_top_token_mismatches':fresh['top_token_mismatches'],
            'fresh_elapsed_seconds':fresh['elapsed_seconds'],'fresh_output_directory':str(output),
            'reused_native_reproduction':bool(cached_manifest),
            'native_cuda_executed_this_report':not bool(cached_manifest),
            'reproduction_manifest_sha256':fresh.get('reproduction_manifest_sha256'),
            'reproduction_manifest_path':fresh.get('reproduction_manifest_path'),
            'reproduction_record_sha256':fresh.get('reproduction_record_sha256'),
            'reproduction_actual_rows_sha256':fresh.get('reproduction_actual_rows_sha256'),
            'final_used_for_selection':False,'tsne_is_transductive':True,'tasks':audit['tasks'],
            'paper_equivalent_full_replication':False}


def run_full_experiment(max_vram_gb: float = 24.0):
    return {'numerical':run_smoke_test(),'fresh_cuda':run_gpu_test(max_vram_gb)}

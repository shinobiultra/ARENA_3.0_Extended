"""Release/report orchestration; learner methods remain in the notebook itself."""
from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import torch as t

from . import text_data
from .study_evidence import validate_text_study

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def summarize_primary_tasks(primary):
    """Keep objectively scored tasks separate from reference-only story diagnostics."""
    metrics={'parentheses':'valid_grammar_rate','arithmetic':'exact_answer_rate','qa':'exact_answer_rate'}
    means={label:{task:statistics.mean(row[metric] for row in task_rows[task])
                  for task,metric in metrics.items()} for label,task_rows in primary.items()}
    stories={label:{metric:statistics.mean(row[metric] for row in task_rows['story'])
                    for metric in ('eos_rate','mean_reference_token_edit_distance','reference_token_match_rate')}
             for label,task_rows in primary.items()}
    return means,stories


def run_text_gpu_check(max_vram_gb=24.):
    temporary=Path(tempfile.mkdtemp(prefix='arena-diffusion-text-report-'))
    saved_path=HERE/'expected_outputs/text_learner_execution.json'
    saved=json.loads(saved_path.read_text()) if saved_path.exists() else {}
    learner_value=os.environ.get('ARENA_DIFFUSION_LEARNER_DIRECTORY',saved.get('output_directory_override'))
    data_value=os.environ.get('ARENA_DIFFUSION_DATA_DIRECTORY',saved.get('data_directory'))
    data_directory=Path(data_value) if data_value else temporary/'data'
    if not (data_directory/'provenance.json').exists():
        text_data.build_text_data(data_directory)
    learner=Path(learner_value) if learner_value else temporary/'own_learner'
    if not (learner/'execution.json').exists():
        subprocess.run([sys.executable,'-u',str(HERE/'execute_learner.py'),'--scope','full','--output-directory',str(learner)],
                       cwd=ROOT,env={**os.environ,'ARENA_DIFFUSION_DATA_DIRECTORY':str(data_directory)},check=True)
    audit=validate_text_study(learner,data_directory)
    fresh=temporary/'fresh_cuda'
    subprocess.run([sys.executable,'-u',str(HERE/'study_cuda.py'),'--learner-directory',str(learner),
                    '--data-directory',str(data_directory),'--output-directory',str(fresh)],cwd=ROOT,check=True)
    result=json.loads((fresh/'status.json').read_text())
    if result['status']!='PASS' or result['sources_changed_during_run']:
        raise ValueError('Fresh text CUDA execution did not complete with unchanged sources')
    if result['peak_vram_gb']>max_vram_gb:
        raise ValueError('The actual text experiment exceeded the GPU budget')
    means,stories=summarize_primary_tasks(audit['primary_task_metrics'])
    return {'peak_vram_gb':max(result['peak_vram_gb'],t.cuda.max_memory_allocated()/1024**3),
            'text_own_answer_code_cells':43,'text_training_runs':audit['training_runs'],
            'text_generation_conditions':audit['generation_conditions'],'text_generated_examples':audit['generated_examples'],
            'text_figures':audit['figures'],'text_timing_conditions':audit['timing_conditions'],
            'text_timing_repetitions':audit['timing_repetitions'],'text_play_conditions':audit['play_conditions'],
            'text_fresh_training_runs':result['fresh_training_runs'],
            'text_fresh_generation_conditions':result['fresh_generation_conditions'],
            'text_fresh_generated_examples':result['fresh_generated_examples'],
            'text_all_fresh_outcomes_match_learner':result['all_fresh_outcomes_match_learner'],
            'text_max_weight_error':result['max_weight_error'],'text_max_trace_numeric_error':result['max_trace_numeric_error'],
            'text_final_used_for_selection':False,'text_primary_task_means':means,
            'text_story_reference_diagnostics':stories,
            'text_natural_language_quality_established':False,'text_all_controls_retained':True,
            'text_peak_vram_gb':result['peak_vram_gb'],'text_fresh_cuda_directory':str(fresh),
            'text_fresh_raw_files_sha256':result['raw_files_sha256'],'text_fresh_elapsed_seconds':result['elapsed_seconds'],
            'text_learner_directory':str(learner),'text_data_directory':str(data_directory)}

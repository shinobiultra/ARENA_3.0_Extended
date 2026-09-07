"""Offline audit of actual text-study rows, controls and own-answer provenance.

This module is independent review code, not an implementation imported by the
learner. Recomputing saved scores is not described as fresh model generation.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import math
import statistics
from pathlib import Path

import torch as t
from tokenizers import Tokenizer

from . import text_data as data
from .text_diffusion_methods import stable_token_commitment
from .text_evaluation import score_generated_text, summarize_generated_rows

HERE=Path(__file__).resolve().parent
MODES=('diffusion','diffusion-shuffled','autoregressive','autoregressive-shuffled')
SEEDS=(0,1,2)
TASKS=('parentheses','arithmetic','qa','story')
PINNED_DATA_HASHES={
    'train.json':'c5ec06679f4f3129912a4bbb62756c6281e874c165c182ad1035962b87443077',
    'development.json':'ec989158afcc8ab3af6fafffddd42d84c11c449cba544cbb276b27a1c75e6307',
    'final.json':'2cfe1f26f2242ebf22acc3bf1aad39b09aa7e88a94c241257e9ce693da8d81f7',
    'tokenizer.json':'3e050b72d39c0e297b193e7ff4dff6eb603c3f9ef140c451c450ca0e7452252b',
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def unique_rows(rows,key,expected,message):
    keys=[key(row) for row in rows]
    if len(keys)!=len(expected) or set(keys)!=set(expected):
        raise ValueError(message)
    return dict(zip(keys,rows))


def expected_configs(kind):
    if kind=='autoregressive':
        return [dict(name='ar-greedy',temperatures=[0.,0.]),dict(name='ar-warm',temperatures=[.8,.8])]
    result=[dict(name=f'{sampler}-{steps}-greedy',sampler=sampler,steps=steps,
                 temperatures=[0.,0.],block_size=None)
            for sampler in ('ancestral','uniform','confidence') for steps in (1,4,8,16)]
    result += [dict(name='confidence-16-annealed',sampler='confidence',steps=16,temperatures=[1.,.2],block_size=None),
               dict(name='confidence-16-warm',sampler='confidence',steps=16,temperatures=[.8,.8],block_size=None)]
    result += [dict(name=f'confidence-block{width}-4',sampler='confidence',steps=4,temperatures=[0.,0.],block_size=width)
               for width in (4,8)]
    return result


def validate_training_records(fits,training_rows):
    expected={(seed,label) for seed in SEEDS for label in MODES}
    lookup=unique_rows(fits,lambda row:(row['seed'],row['kind']+('-shuffled' if row['shuffled'] else '')),
                       expected,'Retain all twelve training runs exactly once')
    tasks=[row['task'] for row in training_rows]
    for (seed,label),fit in lookup.items():
        if (fit['updates']!=12000 or fit['batch_size']!=32 or fit['learning_rate']!=.002
                or fit['weight_decay']!=.01 or fit['gradient_clip']!=1. or fit['parameter_count']!=203264
                or fit['total_diffusion_steps']!=16 or fit['development_every']!=1000
                or fit['development_noise_seed']!=1729 or fit['development_noise_repeats']!=3):
            raise ValueError('Training must match the frozen model, optimizer and development budget')
        if [row['step'] for row in fit['history']]!=list(range(1,12001)):
            raise ValueError('Keep every training update, including later overfitting')
        checkpoints=[row for row in fit['history'] if 'development_objective' in row]
        if [row['step'] for row in checkpoints]!=list(range(1000,12001,1000)):
            raise ValueError('Keep every prespecified development checkpoint')
        if any(not math.isfinite(float(row[key])) for row in fit['history'] for key in ('training_loss','gradient_norm')):
            raise ValueError('Nonfinite training values cannot be hidden')
        if any(not math.isfinite(float(row['development_objective'])) for row in checkpoints):
            raise ValueError('Development objectives must be finite')
        winner=min(checkpoints,key=lambda row:(row['development_objective'],row['step']))
        if fit['selected_step']!=winner['step'] or fit['selected_development_objective']!=winner['development_objective']:
            raise ValueError('Select the earliest minimum of the complete development history')
        permutation=fit['target_permutation']
        if sorted(permutation)!=list(range(len(training_rows))) or any(tasks[j]!=tasks[i] for i,j in enumerate(permutation)):
            raise ValueError('Keep full within-task permutations, including fixed points')
        if not fit['shuffled'] and permutation!=list(range(len(training_rows))):
            raise ValueError('Unshuffled training must keep each target attached to its context')
    for seed in SEEDS:
        rows=[lookup[seed,label] for label in MODES]
        for key in ('initial_parameters_sha256','batch_indices_sha256','development_steps_sha256','development_noisy_sha256'):
            if len({row[key] for row in rows})!=1:
                raise ValueError('Four modes must share initialization, inputs and independent development draws')
        if rows[1]['target_permutation']!=rows[3]['target_permutation']:
            raise ValueError('The two shuffled controls must share the same target permutation')
    return lookup


def validate_generated_condition(record,source_rows,tokenizer,trace_ids):
    result=record['result'];config=record['config'];kind=record['kind']
    if (result['kind']!=kind or result['seed']!=1000 or result['batch_size']!=32
            or result['temperatures']!=config['temperatures'] or result['block_size']!=config.get('block_size')):
        raise ValueError('Generation must use its frozen RNG, temperature, block and batch settings')
    if kind=='diffusion' and (result['sampler']!=config['sampler'] or result['steps']!=config['steps']):
        raise ValueError('Retain the actual declared reveal algorithm and step budget')
    expected={row['id']:row for row in source_rows}
    measured=unique_rows(result['rows'],lambda row:row['id'],expected,'Keep every final example exactly once')
    for identifier,row in measured.items():
        tokens=row['generated_ids']
        if len(tokens)!=24 or any(type(value) is not int or not 0<=value<768 for value in tokens):
            raise ValueError('Retain all 24 generated token IDs, including invalid special predictions')
        rescored=score_generated_text(expected[identifier],tokens,tokenizer)
        if rescored!=row:
            raise ValueError('Decoded text, grammar, EOS and exact scores must match actual token IDs')
    if summarize_generated_rows(list(measured.values()),source_rows)!=result['summary']:
        raise ValueError('Summary metrics must match every individual final outcome')
    traces=unique_rows(result['traces'],lambda row:row['id'],trace_ids,'Use all fixed source-order trace examples, including failures')
    for identifier,trace in traces.items():
        if trace['prompt']!=expected[identifier]['prompt'] or trace['outcome']!=measured[identifier]:
            raise ValueError('Trajectory prompts and final words must match the scored example')
        if kind!='diffusion':continue
        calls=config['steps']*math.ceil(24/(config['block_size'] or 24))
        states=t.tensor(trace['states'])
        if states.shape!=(calls+1,24) or not states[0].eq(data.MASK_ID).all():
            raise ValueError('Retain the all-MASK initial state and every actual diffusion call')
        if states[-1].tolist()!=measured[identifier]['generated_ids'] or len(trace['steps'])!=calls:
            raise ValueError('The trajectory must end at the actual scored generation')
        if ((states[:-1]!=data.MASK_ID)&(states[1:]!=states[:-1])).any():
            raise ValueError('Absorbing reveal rules must preserve previously visible tokens')
        commitment=stable_token_commitment(states[:,None],data.MASK_ID)[0].tolist()
        if commitment!=trace['stable_commitment']:
            raise ValueError('Commitment must be derived from the complete actual state sequence')
        for step_index,step in enumerate(trace['steps']):
            positions=t.tensor(step['active_positions'])-40
            if t.tensor(step['masked_before']).tolist()!=states[step_index,positions].eq(data.MASK_ID).tolist():
                raise ValueError('Entropy eligibility must match the actual pre-step mask')
            if len(step['entropy'])!=len(positions) or any(not math.isfinite(h) or h<0 or h>math.log(767)+1e-5 for h in step['entropy']):
                raise ValueError('Retain finite predictive entropy for each active block position')
    return result['summary']


def validate_text_study(run_directory,data_directory):
    run=Path(run_directory);directory=Path(data_directory)
    evidence=json.loads(gzip.decompress((run/'experiments/completed_text_study.json.gz').read_bytes()))
    execution=json.loads((run/'execution.json').read_text())
    from .execute_learner import LEARNER_INPUTS,REQUIRED_FIGURES
    if (execution['status']!='PASS' or execution['scope']!='full' or execution['code_cells_executed']!=43
            or execution['sources_changed_during_run'] or execution['completed_from_own_answers'] is not True):
        raise ValueError('Require a complete unchanged-source own-answer execution')
    if set(execution['input_files'])!=set(LEARNER_INPUTS):
        raise ValueError('The complete declared learner input inventory is required')
    for name,digest in execution['input_files'].items():
        if sha256(HERE/name)!=digest or sha256(run/'source_snapshot'/name)!=digest:
            raise ValueError('Current and snapshotted learner sources must match the actual execution')
    from .build_notebooks import lesson_cells,complete_learner
    completed=complete_learner({'cells':lesson_cells()})
    learner_definitions={node.name:ast.dump(node) for cell in completed['cells'] if cell['cell_type']=='code'
        for node in ast.parse(''.join(cell['source'])).body if isinstance(node,(ast.FunctionDef,ast.ClassDef))}
    expected_methods={
        'text_diffusion_methods.py':None,
        'text_model.py':None,
        'text_training.py':{'permute_targets_within_task','fixed_development_noise','development_objective','fit_text_model'},
        'text_evaluation.py':None,
    }
    for filename,names in expected_methods.items():
        for node in ast.parse((HERE/filename).read_text()).body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and (names is None or node.name in names):
                if learner_definitions.get(node.name)!=ast.dump(node):
                    raise ValueError('The independent checker must use methods identical to the visible learner answers')
    if set(execution['figures'])!=REQUIRED_FIGURES or any(sha256(run/'figures'/name)!=digest for name,digest in execution['figures'].items()):
        raise ValueError('All twelve figures must bind the actual completed learner')
    protocol=evidence['protocol'];protocol_path=run/'experiments/protocol.json'
    selection_path=run/'experiments/checkpoint_selection.json'
    if (json.loads(protocol_path.read_text())!=protocol or sha256(protocol_path)!=evidence['protocol_sha256']
            or sha256(selection_path)!=evidence['selection_sha256']):
        raise ValueError('Final evidence must bind the unchanged frozen protocol and checkpoint choices')
    if (protocol['seeds']!=list(SEEDS) or protocol['modes']!=list(MODES) or protocol['updates']!=12000
            or protocol['diffusion_configs']!=expected_configs('diffusion')
            or protocol['ar_configs']!=expected_configs('autoregressive')
            or protocol['final_loaded_when_frozen'] is not False or evidence['final_used_for_selection'] is not False
            or protocol['full_model_quality_or_paper_replication'] is not False):
        raise ValueError('Keep every frozen mode/configuration and the bounded claim scope')
    provenance=json.loads((directory/'provenance.json').read_text())
    if any(sha256(directory/name)!=digest for name,digest in PINNED_DATA_HASHES.items()):
        raise ValueError('Use the complete data and training-only vocabulary frozen before model development')
    if provenance!=evidence['data_provenance'] or sha256(directory/'provenance.json')!=protocol['data_provenance_sha256']:
        raise ValueError('The actual data must match the frozen source provenance')
    if (provenance['revision']!=data.DATASET_REVISION or provenance['source_files']!={key:list(value) for key,value in data.SOURCE_FILES.items()}
            or provenance['tokenizer_fit_split']!='training candidate pool only'):
        raise ValueError('Use pinned stories and a training-only tokenizer')
    rows={}
    for split in ('train','development','final'):
        if sha256(directory/f'{split}.json')!=provenance['inventory'][split]['sha256']:
            raise ValueError('Split bytes differ from the declared inventory')
        rows[split]=json.loads((directory/f'{split}.json').read_text())
    if sha256(directory/'tokenizer.json')!=provenance['tokenizer_sha256']:
        raise ValueError('Tokenizer bytes differ from their training-only inventory')
    tokenizer=Tokenizer.from_file(str(directory/'tokenizer.json'))
    if len(rows['train'])!=5632 or len(rows['development'])!=512 or len(rows['final'])!=512:
        raise ValueError('Keep the frozen dataset sizes')
    if len({tuple(row['context_ids']) for split in rows.values() for row in split})!=6656:
        raise ValueError('Actual tokenized input contexts must be unique across all splits')
    if evidence['final_ids']!=[row['id'] for row in rows['final']]:
        raise ValueError('The final example inventory must preserve source order')
    fits=validate_training_records(evidence['fits'],rows['train'])
    selection=json.loads(selection_path.read_text())
    if selection['final_loaded'] is not False or selection['protocol_sha256']!=evidence['protocol_sha256'] or selection['checkpoints']!=evidence['selection']:
        raise ValueError('Checkpoint choices must be frozen before final answers are loaded')
    choices=unique_rows(evidence['selection'],lambda row:(row['seed'],row['label']),fits,'Freeze all twelve checkpoint choices')
    for key,choice in choices.items():
        if choice['selected_step']!=fits[key]['selected_step']:
            raise ValueError('Final generation must use the actual development winner')
        weights=run/'experiments/weights'/f'seed{key[0]}-{key[1]}'
        if sha256(weights/'selected.pt')!=choice['selected_sha256'] or sha256(weights/'checkpoint-12000.pt')!=choice['endpoint_sha256']:
            raise ValueError('Selected and endpoint weight bytes must match the frozen choices')
    expected=set()
    for seed in SEEDS:
        for label in MODES:
            kind=label.removesuffix('-shuffled')
            primary='ancestral-16-greedy' if kind=='diffusion' else 'ar-greedy'
            expected.update((seed,label,'selected',c['name']) for c in expected_configs(kind))
            expected.add((seed,label,'endpoint',primary))
        expected.update([(seed,'diffusion','untrained','ancestral-16-greedy'),(seed,'autoregressive','untrained','ar-greedy')])
    generated=unique_rows(evidence['results'],lambda row:(row['seed'],row['label'],row['checkpoint'],row['config']['name']),
                          expected,'Keep all 126 declared final generation conditions')
    trace_ids=[row['id'] for task in TASKS for row in [r for r in rows['final'] if r['task']==task][:2]]
    for (_,label,_,name),record in generated.items():
        kind=label.removesuffix('-shuffled')
        if record['kind']!=kind or record['shuffled']!=label.endswith('-shuffled'):
            raise ValueError('Model/control labels must identify the actual condition')
        if record['config']!=next(c for c in expected_configs(kind) if c['name']==name):
            raise ValueError('Do not relabel a sampling configuration after observing final performance')
        validate_generated_condition(record,rows['final'],tokenizer,trace_ids)
    expected_timing={(seed,label,c['name']) for seed in SEEDS for label in MODES
        for c in expected_configs(label.removesuffix('-shuffled')) if c['temperatures']==[0.,0.] and c.get('block_size') is None}
    timings=unique_rows(evidence['timings'],lambda row:(row['seed'],row['label'],row['config']),expected_timing,
                        'Keep all 78 matched cached-AR/diffusion timing conditions')
    for (seed,label,name),row in timings.items():
        measured=row['measurement'];times=measured['seconds'];kind=label.removesuffix('-shuffled')
        if (len(times)!=5 or any(not math.isfinite(x) or x<=0 for x in times) or measured['warmups']!=2
                or measured['repeats']!=5 or measured['median_seconds']!=statistics.median(times)
                or measured['batch_size']!=32 or measured['output_slots']!=24
                or measured['cuda_synchronized'] is not True or measured['trajectory_storage'] is not False):
            raise ValueError('Timing must retain synchronized repeats, fixed batch/length and no trajectory overhead')
        calls=measured['forward_calls']
        if kind=='diffusion':
            steps=next(c['steps'] for c in expected_configs(kind) if c['name']==name)
            if calls!=steps or measured['input_tokens']!=32*64*steps:
                raise ValueError('Count the actual full-window diffusion workload')
        elif not 1<=calls<=24 or measured['input_tokens']!=32*40+32*(calls-1):
            raise ValueError('AR timing must include one prefill and cached new-token inputs')
    if len(evidence['plays'])!=9 or {p['play'] for p in evidence['plays']}!={'fact edit','schedule edit','story continuation'}:
        raise ValueError('Retain the three real play experiments and their matched controls')
    primary={label:{task:[] for task in TASKS} for label in MODES}
    for (seed,label,checkpoint,name),record in generated.items():
        if checkpoint=='selected' and name in ('ancestral-16-greedy','ar-greedy'):
            for task in TASKS:primary[label][task].append(record['result']['summary'][task])
    return {'accepted':True,'fresh_model_regeneration':False,'training_runs':12,'generation_conditions':126,
            'generated_examples':126*512,'timing_conditions':78,'timing_repetitions':390,'play_conditions':9,
            'figures':12,'all_controls_retained':True,'primary_task_metrics':primary}

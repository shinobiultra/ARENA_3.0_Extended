"""Check two actual notebook executions and assemble their CPU study evidence.

No models are fitted here. The two own-answer notebooks independently train every
model; this entrypoint compares their complete numeric results and retains timings.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
ID = '16_7_data_shapley_in_one_training_run'
SEEDS = ['16701', '16702', '16703']
CLAIM = ('Exact subset-retraining Data Shapley and a distinct fixed-trajectory local SGD game; '
         'first- and second-order approximations, affine ghost gradient products, and 128-step nonlinear '
         'digit training with held-out deletion tests over three fixed seeds. Exact local allocations '
         'telescope; first-order accounting can overallocate and low-credit deletion can harm test loss. '
         'No large-model throughput or universal data-pruning claim.')
LEVEL = 'native_cpu_multistep_inrun_shapley_with_exact_games_and_retraining_controls'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def read_numeric(path):
    with gzip.open(path, 'rb') as handle:
        raw = handle.read(64 * 1024**2 + 1)
    require(len(raw) <= 64 * 1024**2, 'Numeric payload exceeds bound')
    return json.loads(raw)


def array(value):
    require(isinstance(value, dict) and value.get('tensor') is True, 'Expected a recorded tensor')
    result = np.asarray(value['values'])
    require(list(result.shape) == value['shape'], 'Tensor shape differs from recorded shape')
    require(np.isfinite(result).all(), 'Nonfinite tensor')
    return result


def compare(left, right):
    counts = dict(tensors=0, tensor_elements=0, scalars=0, excluded_timing_fields=0, max_absolute_error=0.0)
    excluded = []
    def visit(a, b, path=()):
        # Only wall-clock measurements vary. Keep both original payloads intact.
        timing = ((len(path) == 4 and path[:1] == ('digit_runs',) and path[1] in SEEDS
                   and path[2:] == ('run', 'wall_seconds'))
                  or (len(path) == 3 and path[0] in ('timing_rows', 'training_times') and path[2] == 'seconds'))
        if timing:
            require(all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in (a, b)), 'Invalid timing')
            excluded.append('/'.join(map(str, path)))
            counts['excluded_timing_fields'] += 1
            return
        require(type(a) is type(b), f'Type changed at {path}')
        if isinstance(a, dict):
            require(list(a) == list(b), f'Keys/order changed at {path}')
            if a.get('tensor') is True:
                x, y = array(a), array(b)
                require(a['dtype'] == b['dtype'] and x.shape == y.shape and np.array_equal(x, y), f'Tensor changed at {path}')
                counts['tensors'] += 1
                counts['tensor_elements'] += int(x.size)
            else:
                for key in a:
                    visit(a[key], b[key], path+(key,))
        elif isinstance(a, list):
            require(len(a) == len(b), f'Length changed at {path}')
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, path+(index,))
        else:
            require(not isinstance(a, float) or math.isfinite(a), f'Nonfinite scalar at {path}')
            require(a == b, f'Scalar changed at {path}')
            counts['scalars'] += 1
    visit(left, right)
    require(counts['excluded_timing_fields'] == 23, 'Unexpected timing inventory')
    return counts, excluded


def validate_execution(directory):
    receipt = json.loads((directory/'learner_execution.json').read_text())
    require(receipt['status'] == 'PASS' and receipt['sources_unchanged'] is True, 'Learner did not pass')
    require(receipt['sources_before'] == receipt['sources_after'], 'Source changed during execution')
    for relative, digest in receipt['sources_before'].items():
        require(sha(ROOT/relative) == digest and sha(directory/'source_snapshot'/relative) == digest, 'Stale execution source: '+relative)
    require(receipt['code_cells'] == receipt['executed_code_cells'] == 33, 'Incomplete code-cell execution')
    require(len(receipt['own_answer_functions']) == len(set(receipt['own_answer_functions'])) == 26, 'Own-answer coverage incomplete')
    require(len(receipt['figures']) == 9, 'Figure inventory incomplete')
    for name, digest in receipt['figures'].items():
        require(sha(directory/'figures'/name) == digest, 'Figure changed: '+name)
    require(sha(directory/'figures/numeric_evidence.json.gz') == receipt['numeric_evidence_sha256'], 'Numeric payload changed')
    require(sha(receipt['executed_notebook']) == receipt['notebook_sha256'], 'Executed notebook changed')
    notebook = json.loads(Path(receipt['executed_notebook']).read_text())
    embedded = sum('image/png' in output.get('data', {}) for cell in notebook['cells'] for output in cell.get('outputs', []))
    require(embedded == 9, 'Figures were not embedded in the actual notebook')
    return receipt


def summarize(payload):
    from chapter16_shapley_attribution_baselines.exercises.part9_influence_and_tracin.attribution_data import load_study_data, data_manifest
    data = load_study_data()
    require(payload['data_manifest'] == data_manifest(data), 'Data bytes or split changed')
    x = data['final']['x'][:, :64].numpy()
    y = data['final']['y'].numpy()
    observed_labels = data['train']['corrupted_y'].numpy()
    def loss_for(params):
        w1, b1, w2, b2 = map(array, params)
        logits = (np.tanh(x @ w1 + b1) @ w2 + b2).reshape(-1)
        return float(np.mean(np.logaddexp(0,logits)-y*logits))
    runs = payload['digit_runs']
    require(list(runs) == SEEDS and list(payload['deletion_results']) == SEEDS, 'All three fixed seeds required')
    trajectory_count = 0
    def check_run(run):
        nonlocal trajectory_count
        require(array(run['allocations']).shape == (128, 128), 'Incomplete per-update allocations')
        require(array(run['losses']).shape == (129,), 'Incomplete loss trajectory')
        require(np.allclose(array(run['allocations']).sum(0), array(run['scores']), atol=1e-12, rtol=1e-10), 'Allocation sum differs')
        require([array(p).shape for p in run['params']] == [(64,16),(16,),(16,1),(1,)], 'Wrong model family')
        trajectory_count += 1
    outcomes = {}
    for seed, record in runs.items():
        schedule = np.stack([array(batch) for batch in record['schedule']])
        require(schedule.shape == (128,4), 'Incomplete training schedule')
        for epoch in schedule.reshape(4,128):
            require(np.array_equal(np.sort(epoch), np.arange(128)), 'Missing or repeated training row in epoch')
        check_run(record['run']); check_run(record['shuffled'])
        for p, q in zip(record['run']['params'], record['shuffled']['params']):
            require(np.array_equal(array(p), array(q)), 'Ranking-label control changed the training trajectory')
        observed = payload['deletion_results'][seed]
        require(abs(observed['full_loss']-loss_for(record['run']['params'])) < 1e-12, 'Full test loss differs from actual weights')
        conditions = observed['conditions']
        order = np.argsort(array(record['run']['scores']), kind='stable')
        selected = payload['deletion_choices'][seed]
        require(np.array_equal(array(selected['lowest']),order[:12]) and np.array_equal(array(selected['highest']),order[-12:]), 'Credit selection changed')
        require(np.array_equal(array(selected['shuffled']),np.argsort(array(record['shuffled']['scores']),kind='stable')[:12]), 'Shuffled selection changed')
        require(np.array_equal(array(selected['known_corruption']),np.flatnonzero(data['train']['flipped'].numpy())), 'Known corruption selection changed')
        require(set(conditions) == {'lowest','highest','known_corruption','shuffled'} | {f'random_{i}' for i in range(8)}, 'Deletion controls incomplete')
        for name, result in conditions.items():
            removed = array(result['removed'])
            require(removed.shape == (12,) and len(set(removed.tolist())) == 12 and removed.min() >= 0 and removed.max() < 128, 'Deletion set malformed')
            if name.startswith('random_'):
                require(np.array_equal(np.sort(observed_labels[removed]),np.sort(observed_labels[array(selected['lowest'])])), 'Random control label counts differ')
            require(abs(result['final_loss']-loss_for(result['run']['params'])) < 1e-12, 'Deletion test loss differs from actual weights')
            require(np.array_equal(removed, array(payload['deletion_choices'][seed][name])), 'Deletion choices changed')
            require(abs(observed['full_loss']-result['final_loss']-result['improvement']) < 1e-14, 'Deletion gain inconsistent')
            check_run(result['run'])
        outcomes[seed] = {'full_test_loss': observed['full_loss'], 'deletion_improvements': {k:v['improvement'] for k,v in conditions.items()},
                          'first_order_credit_sum': float(array(record['run']['scores']).sum()),
                          'ranking_loss_improvement': float(array(record['run']['losses'])[0]-array(record['run']['losses'])[-1])}
    primary = runs[SEEDS[0]]['run']
    exact = array(primary['exact_allocations'])
    losses = array(primary['losses'])
    exact_error = float(np.abs(exact.sum(1)-(losses[:-1]-losses[1:])).max())
    require(exact.shape == (128,128) and exact_error < 1e-12, 'Local exact accounting does not telescope')
    require(np.allclose(exact.sum(0), array(primary['exact_scores']), atol=1e-12, rtol=1e-10), 'Exact aggregate differs')
    check_run(payload['play']['run'])
    require(len(payload['training_times']) == 6, 'Training timings missing')
    for row in payload['training_times']:
        if row['mode'] == 'SGD + accounting':
            check_run(row['run'])
        else:
            require(row['mode'] == 'SGD only' and len(row['params']) == 4, 'Plain SGD timing incomplete')
            trajectory_count += 1
        params = row['run']['params'] if 'run' in row else row['params']
        require(all(np.array_equal(array(p),array(q)) for p,q in zip(params,primary['params'])), 'Timed route changed fitted weights')
    require(trajectory_count == 49, 'Native training inventory differs')
    require(len(payload['foundation']['coalition_values']) == 256, 'Exact retraining foundation missing')
    return dict(status='PASS', training_runs=trajectory_count, optimizer_updates=128*trajectory_count,
                primary_training_seeds=[int(s) for s in SEEDS], train_rows=128, ranking_rows=48, test_rows=64,
                exact_local_games=128, exact_local_coalitions=2048, fixed_batch_denominator=4,
                final_used_for_selection=False, native_results=outcomes,
                exact_local_efficiency_max_error=exact_error, data_manifest=payload['data_manifest'])


def has_native_evidence(report):
    m = report.get('metrics', {}).get('gpu_test', {})
    return (report.get('notebook_id') == ID and report.get('evidence_level') == LEVEL
            and m.get('device') == 'cpu' and m.get('native_model_retrained') is True
            and m.get('native_model_retrained_in_this_report') is False
            and m.get('report_assembly_model_fits') == 0
            and m.get('actual_learner_passed') is True and m.get('fresh_reproduction_passed') is True
            and m.get('reference_refits') == m.get('fresh_model_fits') == 49
            and m.get('optimizer_updates') == 6272 and m.get('own_answer_code_cells') == 33
            and m.get('embedded_figures') == 9 and m.get('training_seeds') == [16701,16702,16703]
            and m.get('exact_local_games') == 128 and m.get('final_query_count') == 64
            and m.get('final_used_for_selection') is False and m.get('fresh_max_error') == 0
            and m.get('fresh_compared_tensors', 0) > 300 and m.get('fresh_compared_elements', 0) > 100000
            and all(isinstance(m.get(k),str) and len(m[k]) == 64 for k in ('learner_evidence_sha256','fresh_evidence_sha256')))


def assemble(primary, fresh, output):
    require(primary.resolve() != fresh.resolve(), 'Fresh execution must be separate')
    receipts = [validate_execution(p) for p in (primary,fresh)]
    require(receipts[0]['sources_before'] == receipts[1]['sources_before'], 'Execution sources differ')
    payloads = [read_numeric(p/'figures/numeric_evidence.json.gz') for p in (primary,fresh)]
    comparison, excluded = compare(*payloads)
    summary = summarize(payloads[0])
    require(summary == summarize(payloads[1]), 'Summary changed')
    output.mkdir(parents=True, exist_ok=False)
    write(output/'native_summary.json', summary)
    learner = {**receipts[0], 'completed_from_own_answers': True, 'native_model_executed': True,
               'source_inputs': receipts[0]['sources_before'], 'changed_sources': [], 'embedded_figures': 9,
               'native_summary': summary, 'native_summary_sha256': sha(output/'native_summary.json')}
    write(output/'learner_execution.json', learner)
    reproduction = dict(status='PASS', native_model_retrained=True, cached_model_inputs_used=False,
                        changed_sources=[], source_inputs=receipts[1]['sources_before'], comparisons=comparison,
                        comparison='exact_recursive_dtype_shape_value_including_container_order_except_named_wall_times',
                        excluded_timing_fields=excluded, fresh_execution_receipt=receipts[1],
                        primary_numeric_sha256=receipts[0]['numeric_evidence_sha256'],
                        fresh_numeric_sha256=receipts[1]['numeric_evidence_sha256'])
    write(output/'fresh_reproduction.json', reproduction)
    measured = dict(device='cpu', native_model_retrained=True, native_model_retrained_in_this_report=False,
                    report_assembly_model_fits=0, actual_learner_passed=True, fresh_reproduction_passed=True,
                    reference_refits=49, fresh_model_fits=49, optimizer_updates=6272, own_answer_code_cells=33,
                    embedded_figures=9, training_seeds=[16701,16702,16703], exact_local_games=128,
                    final_query_count=64, final_used_for_selection=False, fresh_max_error=0,
                    fresh_compared_tensors=comparison['tensors'], fresh_compared_elements=comparison['tensor_elements'],
                    learner_evidence_sha256=sha(output/'learner_execution.json'), fresh_evidence_sha256=sha(output/'fresh_reproduction.json'))
    report = dict(notebook_id=ID, date_run=datetime.now(timezone.utc).isoformat(), gt_tier='GT-0', evidence_level=LEVEL,
                  claim_scope=CLAIM, tests_passed=True, accepted=True, gpu_name='CPU', peak_vram_gb=0,
                  wall_clock_seconds=sum(r['elapsed_seconds'] for r in receipts),
                  metrics={'gpu_test':measured,'gpu_evidence':{'category':'cpu_or_budget_metric','uses_cuda':False,'placeholder_only':False},'native_summary':summary},
                  baselines={'exact_local_game':'all 16 coalitions at all 128 steps','deletions':['highest','known_corruption']},
                  negative_controls={'deletions':['shuffled ranking labels','eight label-count-matched random sets per seed']},
                  ood_tests={'heldout':'64 official test images, unused during selection'},
                  known_failures=[],
                  scientific_limitations=['First-order allocations overestimate ranking-loss improvement.', 'Lowest-credit deletion harms held-out loss for seed 16702.'],
                  safety_notes=['Public handwritten digits; local CPU fits only.'])
    write(output/'report_body.json', report)
    print(json.dumps({'status':'PASS','training_runs_per_execution':49,'comparison':comparison},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--fresh', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    assemble(args.primary,args.fresh,args.output)

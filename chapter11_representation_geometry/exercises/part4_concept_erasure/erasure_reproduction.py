"""Rebuild every fitted condition and repeat all native outputs in a fresh process.

The saved learner tensors are comparison targets, never substituted for fresh
model inference. Full vocabularies are streamed one condition at a time.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import torch as t
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from chapter11_representation_geometry.exercises.part4_concept_erasure import (
    erasure_data as data,
    erasure_methods as m,
    erasure_protocol as protocol,
)
from chapter11_representation_geometry.exercises.part4_concept_erasure.build_notebooks import (
    complete_learner, lesson_cells,
)
from chapter11_representation_geometry.exercises.part4_concept_erasure.erasure_native import (
    capture_native, continue_native, full_forward_intervention, weight_digest,
)


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_conditions():
    return [(f'{layer}/{position}', name, f'conditions/site{site_index}_method{method_index:02d}.pt')
            for site_index, (layer, position) in enumerate(protocol.SITES)
            for method_index, name in enumerate(protocol.condition_names())]


def validate_summary(summary):
    require(summary.get('status') == 'PASS', 'The native learner did not pass.')
    require(summary.get('weights_unchanged') is True and summary.get('model_gradients_absent') is True,
            'Native weights or gradients changed.')
    require(summary.get('model_id') == data.MODEL_ID and summary.get('revision') == data.REVISION,
            'Wrong native model or revision.')
    require((summary.get('conditions'), summary.get('final_role_rows'), summary.get('final_object_rows'), summary.get('final_outcomes'))
            == (106, 288, 384, 71232), 'Incomplete native condition or row inventory.')
    expected = expected_conditions()
    actual = [(row['site'], row['method'], row['tensor_file']) for row in summary['final']]
    require(actual == expected, 'Missing, duplicated, reordered or relabeled final conditions.')
    require(set(summary['condition_tensor_sha256']) == {item[2] for item in expected}, 'Incomplete condition tensor inventory.')
    for row in summary['final']:
        for task, count in [('final', 288), ('object_final', 384)]:
            require(row[task]['rows'] == count, 'A condition omitted evaluation rows.')
            require(row[task]['native_hook_max_error'] <= 3e-4, 'Native hook parity failed.')


def validate_learner(directory):
    directory = Path(directory).resolve()
    execution = json.loads((directory / 'execution.json').read_text())
    require(execution.get('status') == 'PASS' and not execution.get('changed_sources'), 'Own-answer learner failed or changed while running.')
    require(execution.get('completed_from_own_answers') is True and execution.get('native_model_executed') is True,
            'Need the complete own-answer learner including the native model.')
    own = json.loads((directory / 'completed_learner.ipynb').read_text())
    current = complete_learner(dict(cells=lesson_cells()))
    require(len(own['cells']) == len(current['cells']) == execution['cells'], 'The learner cell inventory changed.')
    for index, (executed, cell) in enumerate(zip(own['cells'], current['cells'], strict=True)):
        require(executed['cell_type'] == cell['cell_type'], f'Cell kind changed: {index}')
        if cell['cell_type'] == 'code':
            require(executed.get('execution_count') is not None, f'Code cell {index} was not executed.')
            require(not any(output.get('output_type') == 'error' for output in executed.get('outputs', [])), f'Code cell {index} failed.')
            old_ast = ast.dump(ast.parse(''.join(executed['source'])), include_attributes=False)
            new_ast = ast.dump(ast.parse(''.join(cell['source'])), include_attributes=False)
            require(old_ast == new_ast, f'Executed own-answer code changed: cell {index}')
    for name, digest in execution['source_inputs'].items():
        require(file_digest(directory / 'source_snapshot' / name) == digest, 'Learner source snapshot changed: ' + name)
        if name != str((HERE / 'lesson.md').relative_to(ROOT)):
            require(file_digest(ROOT / name) == digest, 'Current learner scientific source changed: ' + name)
    assets = directory / 'figures'
    summary = json.loads((assets / 'native_summary.json').read_text())
    require(summary == execution['native_summary'], 'The execution and native summaries disagree.')
    validate_summary(summary)
    require(file_digest(assets / 'fitted_conditions.pt') == summary['fitted_conditions_sha256'], 'Fitted maps changed.')
    require(file_digest(assets / 'native_inputs.pt') == summary['native_inputs_sha256'], 'Native inputs changed.')
    for name, digest in execution['figures'].items():
        require(file_digest(assets / name) == digest, 'Executed figure changed: ' + name)
    # Per-condition tensor hashes are checked immediately before comparison.
    return execution, summary


def edit(x, condition):
    if condition['family'].startswith('Matched'):
        return m.matched_random_edit(x, condition['eraser']['mean'], condition['direction'], condition['eraser'])
    return m.apply_eraser(x, condition['eraser'])


def construct_conditions(train, reader_train, train_labels, reader_labels, site_index):
    z = F.one_hot(train_labels, 2).double()
    mean, width = train.mean(0), train.shape[1]
    old_reader = m.fit_binary_probe(train, train_labels, l2=protocol.PROBE_L2, steps=protocol.PROBE_STEPS)
    q = old_reader['coefficient'] / old_reader['coefficient'].norm()
    fitted = {
        'No intervention': dict(mean=mean, left=train.new_empty(width, 0), right=train.new_empty(0, width), rank=0),
        'Probe coefficient': dict(mean=mean, left=q[:, None], right=q[None], rank=1),
        'Orthogonal covariance': m.fit_orthogonal_eraser(train, z, svd_atol=protocol.SVD_ATOL),
        'LEACE empirical': m.fit_leace(train, z, svd_atol=protocol.SVD_ATOL),
        'LEACE shrinkage .01': m.fit_leace(train, z, shrinkage=protocol.SHRINKAGE, svd_atol=protocol.SVD_ATOL),
    }
    conditions = {name: dict(family=name, eraser=fit) for name, fit in fitted.items()}
    for draw in range(protocol.RANDOM_DRAWS):
        seed = protocol.RANDOM_SEED_STARTS[site_index] + draw
        q = t.randn(width, dtype=t.float64, generator=t.Generator().manual_seed(seed))
        q = q / q.norm()
        conditions[f'Random rank one/draw{draw:02d}'] = dict(
            family='Random rank one', seed=seed, eraser=dict(mean=mean, left=q[:, None], right=q[None], rank=1))
        for family, target in [('Matched empirical', 'LEACE empirical'), ('Matched shrinkage', 'LEACE shrinkage .01')]:
            conditions[f'{family}/draw{draw:02d}'] = dict(family=family, seed=seed, direction=q, eraser=fitted[target])
    for name in protocol.condition_names():
        current = conditions[name]
        current['old_reader'] = old_reader
        current['new_reader'] = m.fit_binary_probe(edit(reader_train, current), reader_labels,
                                                  l2=protocol.PROBE_L2, steps=protocol.PROBE_STEPS)
        if name in fitted:
            current['fitting_reader'] = m.fit_binary_probe(edit(train, current), train_labels,
                                                          l2=protocol.PROBE_L2, steps=protocol.PROBE_STEPS)
    return conditions


def compare_fitted_conditions(actual, expected):
    require(actual['family'] == expected['family'], 'A condition family changed.')
    t.testing.assert_close(actual['eraser']['mean'], expected['eraser']['mean'], atol=1e-9, rtol=1e-9)
    # SVD factors can change signs without changing the affine map.
    t.testing.assert_close(actual['eraser']['left'] @ actual['eraser']['right'],
                           expected['eraser']['left'] @ expected['eraser']['right'], atol=1e-9, rtol=1e-9)
    require(actual['eraser']['rank'] == expected['eraser']['rank'], 'Erasure rank changed.')
    if 'direction' in actual:
        t.testing.assert_close(actual['direction'], expected['direction'], atol=0, rtol=0)
    for field in ['old_reader', 'new_reader', 'fitting_reader']:
        if field in actual:
            for key in ['coefficient', 'intercept', 'training_mean', 'training_scale']:
                t.testing.assert_close(actual[field][key], expected[field][key], atol=1e-8, rtol=1e-8)


@t.no_grad()
def infer(model, cache, layer, position, condition):
    x = cache.activations(layer, position).double()
    logits, replacements = [], []
    for start in range(0, len(x), 24):
        indices = t.arange(start, min(start + 24, len(x)))
        replacement = edit(x[indices], condition).float()
        logits.append(continue_native(model, cache, layer, indices, replacement, position))
        replacements.append(replacement)
    return t.cat(logits), t.cat(replacements)


def run(learner, directory):
    learner, directory = Path(learner).resolve(), Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    state = dict(status='RUNNING', learner_directory=str(learner), source_inputs={}, conditions=[], actual_intervention_rows=0)

    def save():
        state['elapsed_seconds'] = time.perf_counter() - started
        (directory / 'reproduction.json').write_text(json.dumps(state, indent=2) + '\n')

    save()
    try:
        execution, summary = validate_learner(learner)
        state['learner_execution_sha256'] = file_digest(learner / 'execution.json')
        source_paths = [ROOT / name for name in execution['source_inputs']] + [Path(__file__).resolve()]
        for path in source_paths:
            name = str(path.relative_to(ROOT))
            copy = directory / 'source_snapshot' / name
            copy.parent.mkdir(parents=True, exist_ok=True)
            copy.write_bytes(path.read_bytes())
            state['source_inputs'][name] = file_digest(path)
        save()
        t.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(data.MODEL_ID, revision=data.REVISION, local_files_only=True)
        tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, 'left'
        model = AutoModelForCausalLM.from_pretrained(data.MODEL_ID, revision=data.REVISION, local_files_only=True,
                                                   dtype=t.float32, attn_implementation='eager').cpu().eval()
        model.requires_grad_(False)
        before = weight_digest(model)
        require(before == summary['weights_sha256'], 'A different checkpoint was loaded.')
        rows = {name: data.prompt_rows(name) for name in ['train', 'probe_train', 'development', 'final']}
        rows['object_development'] = data.prompt_rows('development', 'object')
        rows['object_final'] = data.prompt_rows('final', 'object')
        ids = data.answer_token_ids(tokenizer, rows['train'])
        caches = {name: capture_native(model, tokenizer, values, layers=(6, 9)) for name, values in rows.items()}
        old_inputs = t.load(learner / 'figures/native_inputs.pt', weights_only=True)
        require(rows == old_inputs['rows'], 'Actual prompt identities, order or labels changed.')
        baseline_errors = {}
        for split, cache in caches.items():
            error = float((cache.logits - old_inputs['baseline_logits'][split]).abs().max())
            t.testing.assert_close(cache.logits, old_inputs['baseline_logits'][split], atol=3e-4, rtol=0)
            baseline_errors[split] = error
            for layer, position in protocol.SITES:
                t.testing.assert_close(cache.activations(layer, position), old_inputs['activations'][f'{layer}/{position}'][split], atol=3e-5, rtol=0)
        del old_inputs
        expected_bank = t.load(learner / 'figures/fitted_conditions.pt', weights_only=True)
        fresh_bank = {}
        # Fit all maps/readers before using final labels for outcomes.
        for index, (layer, position) in enumerate(protocol.SITES):
            site = f'{layer}/{position}'
            fresh_bank[site] = construct_conditions(caches['train'].activations(layer, position).double(),
                caches['probe_train'].activations(layer, position).double(),
                t.tensor([row['role'] for row in rows['train']]), t.tensor([row['role'] for row in rows['probe_train']]), index)
            for name in protocol.condition_names():
                compare_fitted_conditions(fresh_bank[site][name], expected_bank[site][name])
            print(site, 'all 53 freshly fitted conditions agree', flush=True)
        t.save(fresh_bank, directory / 'fresh_fitted_conditions.pt')
        state['fitted_conditions_sha256'] = file_digest(directory / 'fresh_fitted_conditions.pt')
        state['baseline_max_errors'] = baseline_errors
        (directory / 'conditions').mkdir()
        for site, name, filename in expected_conditions():
            layer_string, position = site.split('/')
            layer = int(layer_string)
            expected_path = learner / 'figures' / filename
            require(file_digest(expected_path) == summary['condition_tensor_sha256'][filename], 'A saved final tensor changed: ' + filename)
            expected = t.load(expected_path, weights_only=True)
            condition = fresh_bank[site][name]
            actual, record = {}, dict(site=site, method=name)
            expected_record = next(row for row in summary['final'] if row['site'] == site and row['method'] == name)
            final_x = edit(caches['final'].activations(layer, position).double(), condition)
            labels = t.tensor([row['role'] for row in rows['final']])
            for key, reader_key in [('new_probe', 'new_reader'), ('old_probe', 'old_reader')]:
                actual[key] = m.evaluate_binary_probe(final_x, labels, condition[reader_key])
                t.testing.assert_close(actual[key]['logits'], expected[key]['logits'], atol=1e-8, rtol=1e-8)
                t.testing.assert_close(actual[key]['probabilities'], expected[key]['probabilities'], atol=1e-9, rtol=1e-9)
            for task in ['final', 'object_final']:
                cache = caches[task]
                logits, replacement = infer(model, cache, layer, position, condition)
                targets = t.tensor([ids[row['answer']] for row in rows[task]])
                require(t.equal(targets, expected[task]['targets']), 'Actual target IDs changed.')
                t.testing.assert_close(replacement, expected[task]['replacements'], atol=3e-5, rtol=0)
                t.testing.assert_close(logits, expected[task]['logits'], atol=3e-4, rtol=0)
                require(t.equal(logits.argmax(-1), expected[task]['logits'].argmax(-1)), 'Actual winning token changed.')
                with t.no_grad():
                    independent = full_forward_intervention(model, cache, layer, t.arange(2), replacement[:2], position)
                t.testing.assert_close(logits[:2], independent, atol=3e-4, rtol=0)
                record[task] = dict(rows=len(rows[task]), full_vocabulary_max_error=float((logits - expected[task]['logits']).abs().max()),
                                    replacement_max_error=float((replacement - expected[task]['replacements']).abs().max()),
                                    native_hook_max_error=float((independent - logits[:2]).abs().max()), all_winners_equal=True)
                measured = m.compare_model_outputs(logits, cache.logits, targets)
                for metric, value in measured.items():
                    require(math.isclose(value, expected_record[task][metric], abs_tol=1e-7, rel_tol=1e-6),
                            'Reported native metric disagrees with actual fresh outputs: ' + metric)
                record[task]['metrics'] = measured
                actual[task] = dict(logits=logits, replacements=replacement, targets=targets)
                state['actual_intervention_rows'] += len(rows[task])
            destination = directory / filename
            t.save(actual, destination)
            record['tensor_file'] = filename
            record['tensor_sha256'] = file_digest(destination)
            state['conditions'].append(record)
            with (directory / 'actual_comparisons.jsonl').open('a') as stream:
                stream.write(json.dumps(record) + '\n')
            save()
            if len(state['conditions']) % 10 == 0:
                print('Repeated', len(state['conditions']), '/ 106 complete native conditions', flush=True)
        require(len(state['conditions']) == 106 and state['actual_intervention_rows'] == 71232, 'Fresh evaluation was incomplete.')
        state.update(status='PASS', model_id=data.MODEL_ID, revision=data.REVISION, weights_sha256=before,
                     weights_unchanged=weight_digest(model) == before, model_gradients_absent=all(p.grad is None for p in model.parameters()),
                     actual_comparisons_sha256=file_digest(directory / 'actual_comparisons.jsonl'))
        require(state['weights_unchanged'] and state['model_gradients_absent'], 'The fresh model changed.')
    except BaseException as error:
        state.update(status='FAIL', error=str(error))
        raise
    finally:
        state['changed_sources'] = [name for name, digest in state['source_inputs'].items() if file_digest(ROOT / name) != digest]
        if state['changed_sources']:
            state['status'] = 'FAIL'
        save()
    require(state['status'] == 'PASS', 'Fresh source changed while running.')
    print('PASS: independently refitted and reproduced all 71,232 final outcomes.', flush=True)
    return state


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learner-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    arguments = parser.parse_args()
    run(arguments.learner_directory, arguments.output_directory)

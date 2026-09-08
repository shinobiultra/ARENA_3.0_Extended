"""Train/development-only native study. Never construct or evaluate final rows."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import torch as t
from transformers import AutoModelForCausalLM, AutoTokenizer
from chapter11_representation_geometry.exercises.part4_concept_erasure import erasure_methods as m, erasure_data as data
from chapter11_representation_geometry.exercises.part4_concept_erasure.erasure_native import (
    capture_native, native_with_eraser, native_metrics, full_forward_intervention, weight_digest,
)


def plain(value):
    if isinstance(value, t.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def run(directory):
    directory.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).parent
    sources = [here / name for name in ['erasure_methods.py', 'erasure_data.py', 'erasure_native.py', 'run_development.py']]
    sources += [ROOT / 'chapter11_representation_geometry/exercises/part3_causal_abstraction_das' / name
                for name in ['das_native.py', 'run_native_development.py']]
    snapshot = directory / 'source_snapshot'
    hashes = {}
    for source in sources:
        relative = str(source.relative_to(ROOT))
        destination = snapshot / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        hashes[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    state = dict(status='RUNNING', source_inputs=hashes, final_accessed=False, conditions=[])
    started = time.perf_counter()
    tensors = {}

    def save():
        state['elapsed_seconds'] = time.perf_counter() - started
        (directory / 'development.json').write_text(json.dumps(plain(state), indent=2) + '\n')

    save()
    try:
        t.set_num_threads(4)
        tokenizer = AutoTokenizer.from_pretrained(data.MODEL_ID, revision=data.REVISION, local_files_only=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(data.MODEL_ID, revision=data.REVISION,
                    local_files_only=True, dtype=t.float32, attn_implementation='eager').cpu().eval()
        model.requires_grad_(False)
        before = weight_digest(model)
        rows = {name: data.prompt_rows(name) for name in ['train', 'probe_train', 'development']}
        rows['object'] = data.prompt_rows('development', 'object')
        ids = data.answer_token_ids(tokenizer, rows['train'])
        caches = {name: capture_native(model, tokenizer, values, layers=(6, 9)) for name, values in rows.items()}
        tensors['rows'], tensors['baseline_logits'] = rows, {name: cache.logits for name, cache in caches.items()}
        state['baseline'] = {name: native_metrics(cache.logits, cache.logits, rows[name], ids) for name, cache in caches.items()}
        for layer, position in [(6, 'subject'), (9, 'end')]:
            train = caches['train'].activations(layer, position).double()
            audit = caches['probe_train'].activations(layer, position).double()
            dev = caches['development'].activations(layer, position).double()
            labels = {name: t.tensor([row['role'] for row in rows[name]]) for name in ['train', 'probe_train', 'development']}
            z = t.nn.functional.one_hot(labels['train'], 2).double()
            old_probe = m.fit_binary_probe(train, labels['train'])
            q = old_probe['coefficient'][:, None]
            q = q / q.norm()
            fits = {
                'No intervention': dict(mean=train.mean(0), left=t.empty(768, 0, dtype=t.float64), right=t.empty(0, 768, dtype=t.float64), rank=0),
                'Probe coefficient': dict(mean=train.mean(0), left=q, right=q.T, rank=1),
                'Orthogonal covariance': m.fit_orthogonal_eraser(train, z),
                'LEACE empirical': m.fit_leace(train, z),
                'LEACE shrinkage .01': m.fit_leace(train, z, shrinkage=.01),
            }
            for name, fit in fits.items():
                edited_train = m.apply_eraser(train, fit)
                audit_edited = m.apply_eraser(audit, fit)
                dev_edited = m.apply_eraser(dev, fit)
                new_probe = m.fit_binary_probe(audit_edited, labels['probe_train'])
                on_fitting = m.fit_binary_probe(edited_train, labels['train'])
                metrics = dict(layer=layer, position=position, method=name, rank=fit['rank'],
                    covariance_rank=fit.get('covariance_rank'),
                    fitting_covariance_norm=float(m.centered_statistics(edited_train, z)['cross_covariance'].norm()),
                    fitting_mse=float((edited_train - train).square().sum(1).mean()),
                    development_mse=float((dev_edited - dev).square().sum(1).mean()),
                    fitting_probe=m.evaluate_binary_probe(edited_train, labels['train'], on_fitting),
                    old_probe_on_development=m.evaluate_binary_probe(dev_edited, labels['development'], old_probe),
                    independent_probe_on_development=m.evaluate_binary_probe(dev_edited, labels['development'], new_probe))
                for task in ['development', 'object']:
                    logits, replacements = native_with_eraser(model, caches[task], layer, position, fit, m.apply_eraser)
                    metrics[task] = native_metrics(logits, caches[task].logits, rows[task], ids)
                    tensors[layer, position, name, task] = dict(logits=logits, replacements=replacements)
                    if task == 'development':
                        with t.no_grad():
                            oracle = full_forward_intervention(model, caches[task], layer, t.arange(8), replacements[:8], position)
                        error = float((oracle - logits[:8]).abs().max())
                        if error > 3e-4:
                            raise ValueError(f'Native full-forward mismatch at {layer}/{position}/{name}: {error}')
                        metrics['native_hook_max_error'] = error
                tensors[layer, position, name, 'fits'] = dict(eraser=fit, old_probe=old_probe, new_probe=new_probe, fitting_probe=on_fitting)
                state['conditions'].append(metrics)
                print(layer, position, name, 'probe CE', round(metrics['independent_probe_on_development']['log_loss'], 4),
                      'role', metrics['development']['accuracy'], 'object', metrics['object']['accuracy'], flush=True)
                save()
        state.update(status='PASS', model_id=data.MODEL_ID, revision=data.REVISION, weights_sha256=before,
                     weights_unchanged=weight_digest(model) == before, model_gradients_absent=all(p.grad is None for p in model.parameters()))
        if not state['weights_unchanged'] or not state['model_gradients_absent']:
            raise ValueError('The pretrained model changed.')
        t.save(tensors, directory / 'actual_development_tensors.pt')
    except BaseException as error:
        state.update(status='FAIL', error=str(error))
        raise
    finally:
        state['changed_sources'] = [name for name, digest in hashes.items()
                                   if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest]
        if state['changed_sources']:
            state['status'] = 'FAIL'
        save()
    print('Native development complete; final evaluation remains closed.', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', type=Path, required=True)
    run(parser.parse_args().output_directory)

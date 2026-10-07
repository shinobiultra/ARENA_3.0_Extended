"""Append the selected nonlinear accountant to 16.7 and execute its own answers."""
from __future__ import annotations

import argparse
import ast
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part7_data_shapley_in_one_training_run.inrun_cells import CELLS

STEM = '16.7_Data_Shapley_in_One_Training_Run'
TESTS = {
    'mlp_forward,mlp_factors,per_example_gradients': 'inrun_tests.test_forward_and_factors(mlp_forward, mlp_factors, per_example_gradients)',
    'ghost_dot_products': 'inrun_tests.test_ghost_products(ghost_dot_products)',
    'sgd_step,local_coalition_values': 'inrun_tests.test_fixed_denominator(sgd_step)\ninrun_tests.test_local_game(local_coalition_values)',
    'batch_shapley': 'inrun_tests.test_batch_shapley(batch_shapley)',
    'first_order_scores,second_order_scores': 'inrun_tests.test_taylor_allocations(first_order_scores, second_order_scores)',
    'run_accountant': 'inrun_tests.test_accounting_trajectory(run_accountant)',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cell(kind, source, **metadata):
    value = dict(cell_type=kind, source=source, metadata={'arena': metadata} if metadata else {},
                 id=hashlib.sha256((kind+'\n'+source).encode()).hexdigest()[:12])
    if kind == 'code':
        value.update(outputs=[], execution_count=None)
    return value


def clean_legacy_markdown(source):
    source = re.sub(r'(?m)^ {8}(```|</?details|<summary)', r'\1', source)
    source = re.sub(r'(```python\n) {8}(def )', r'\1\2', source)
    return source


def build():
    legacy = json.loads((HERE / 'retraining_foundation.json').read_text())
    # Keep the analytic learner; the reference GPU-report wrapper is not an exercise.
    cells = []
    for original in legacy['cells'][:34]:
        original = copy.deepcopy(original)
        original['source'] = ''.join(original['source'])
        if original['cell_type'] == 'markdown':
            original['source'] = clean_legacy_markdown(original['source'])
        else:
            original.update(outputs=[], execution_count=None)
        cells.append(original)
    cells[2]['source'] = cells[2]['source'].replace('REQUIRES_GPU = True', 'REQUIRES_GPU = False')
    cells[2]['source'] = cells[2]['source'].replace('seconds on toy contract; minutes on local real-model path', 'minutes on a CPU; no GPU required')
    cells[2]['source'] += "\nimport os\nINRUN_OUTPUT = Path(os.environ.get('ARENA_INRUN_OUTPUT', '/tmp/arena-inrun-play'))\nINRUN_OUTPUT.mkdir(parents=True, exist_ok=True)\n"
    old_save = cells[29]['source'].index('if SAVE_SIGNATURE:')
    cells[29]['source'] = cells[29]['source'][:old_save] + "figure.savefig(INRUN_OUTPUT / '167_01_exact_retraining.png', dpi=150, bbox_inches='tight')\nplt.show()\n"
    cells[30]['source'] = 'The figure above is generated from your current implementations and coalition retrainings.'
    cells[33]['source'] = cells[33]['source'].replace(
        '- The learner result is intentionally CPU-sized. The release contract below reruns an independent one-step training organism on CUDA; it does not turn this into a large-dataset valuation claim.',
        '- These first experiments are deliberately small. The next part tests a nonlinear minibatch trajectory on handwritten digits and distinguishes its accounting game from full subset retraining.')
    cells[0]['source'] += '\n\nAfter the exact-retraining foundation, the second half defines and implements a nonlinear SGD contribution accountant, ghost gradient products and real-digit deletion controls.'
    source = (HERE / 'inrun_methods.py').read_text()
    functions = {node.name: ast.get_source_segment(source, node) for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    lesson = (HERE / 'inrun_lesson.md').read_text()
    position = 0
    for match in re.finditer(r'<!-- (code|exercise|solution): ([\w,]+) -->', lesson):
        prose = lesson[position:match.start()].strip()
        if prose:
            cells.append(cell('markdown', prose))
        kind, key = match.groups()
        if kind == 'code':
            code = CELLS[key].strip()
            if key == 'setup':
                code += '\n\n' + functions['init_mlp'] + '\n\n' + functions['make_schedule']
            cells.append(cell('code', code, calling_cell=key))
        elif kind == 'exercise':
            stubs = []
            for name in key.split(','):
                node = ast.parse(functions[name]).body[0]
                doc = node.body[0] if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) else None
                node.body = ([doc] if doc else []) + [ast.Raise(exc=ast.Call(func=ast.Name(id='NotImplementedError', ctx=ast.Load()), args=[], keywords=[]), cause=None)]
                stubs.append(ast.unparse(ast.fix_missing_locations(node)))
            cells.append(cell('code', '\n\n'.join(stubs)+'\n\n'+TESTS[key], exercise_functions=key.split(',')))
        else:
            answer = '\n\n'.join(functions[name] for name in key.split(','))
            cells.append(cell('markdown', '<details>\n<summary>Solution</summary>\n\n```python\n'+answer+'\n```\n\n</details>', solution_for=key))
        position = match.end()
    tail = lesson[position:].strip()
    if tail:
        cells.append(cell('markdown', tail))
    # Save complete numeric evidence without using it as a teaching output.
    cells.append(cell('code', """from chapter16_shapley_attribution_baselines.exercises.part7_data_shapley_in_one_training_run.inrun_io import save_numeric_evidence
save_numeric_evidence(INRUN_OUTPUT / 'numeric_evidence.json.gz', {
    'foundation': foundation_evidence, 'data_manifest': data_manifest(digits), 'local_results': local_results,
    'digit_runs': digit_runs, 'deletion_choices': deletion_choices,
    'deletion_results': deletion_results, 'timing_rows': timing_rows, 'training_times': training_times,
    'play': {'removed_row': TRY_REMOVED_ROW, 'run': play_run},
})""", evidence_only=True))
    metadata = copy.deepcopy(legacy['metadata'])
    metadata['arena'] = {'section': '16.7', 'source': 'visible own answers', 'original_chapters_modified': False}
    return dict(nbformat=4, nbformat_minor=5, metadata=metadata, cells=cells)


def own_answers(student):
    answers = {}
    for c in student['cells']:
        if c['cell_type'] != 'markdown' or '<summary>Solution</summary>' not in c['source']:
            continue
        for block in re.findall(r'```python\s*\n(.*?)\n\s*```', c['source'], re.S):
            code = block.lstrip()
            for node in ast.parse(code).body:
                if isinstance(node, ast.FunctionDef):
                    answers[node.name] = node
    completed = copy.deepcopy(student)
    replaced = []
    for c in completed['cells']:
        if c['cell_type'] != 'code' or 'raise NotImplementedError' not in c['source']:
            continue
        tree = ast.parse(c['source'])
        for i, node in enumerate(tree.body):
            if isinstance(node, ast.FunctionDef) and any(isinstance(n, ast.Raise) for n in ast.walk(node)):
                if node.name not in answers:
                    raise ValueError('No visible solution for '+node.name)
                tree.body[i] = copy.deepcopy(answers[node.name])
                replaced.append(node.name)
        c['source'] = ast.unparse(tree)
    if len(replaced) != len(set(replaced)) or not replaced:
        raise ValueError('Missing or repeated own-answer replacements')
    return completed, replaced


def execute(output):
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    student = build()
    completed, names = own_answers(student)
    sources = [HERE/name for name in ('inrun_methods.py', 'inrun_tests.py', 'inrun_cells.py', 'inrun_lesson.md', 'build_inrun_notebooks.py', 'inrun_io.py', 'tests.py', 'retraining_foundation.json')]
    sources.append(HERE.parent / 'part9_influence_and_tracin/attribution_data.py')
    before = {str(p.relative_to(ROOT)): digest(p) for p in sources}
    for p in sources:
        dest = output / 'source_snapshot' / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    nbformat.write(nbformat.from_dict(student), output / f'{STEM}_exercises.ipynb')
    notebook = nbformat.from_dict(completed)
    started = time.perf_counter()
    failure = None
    try:
        NotebookClient(notebook, timeout=600, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute(
            env={**os.environ, 'PATH': str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH',''),
                 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                 'MKL_NUM_THREADS': '1', 'ARENA_INRUN_OUTPUT': str(output / 'figures')})
    except BaseException as exc:
        failure = repr(exc)
        raise
    finally:
        solved = output / f'{STEM}_solutions.ipynb'
        nbformat.write(notebook, solved)
        after = {str(p.relative_to(ROOT)): digest(p) for p in sources}
        receipt = dict(status='PASS' if failure is None and before == after else 'FAIL', failure=failure,
                       sources_before=before, sources_after=after, sources_unchanged=before == after,
                       elapsed_seconds=time.perf_counter()-started, own_answer_functions=names,
                       code_cells=sum(c.cell_type == 'code' for c in notebook.cells),
                       executed_code_cells=sum(c.cell_type == 'code' and c.execution_count is not None for c in notebook.cells),
                       notebook_sha256=digest(solved), executed_notebook=str(solved),
                       figures={p.name:digest(p) for p in sorted((output/'figures').glob('*.png'))},
                       numeric_evidence_sha256=digest(output/'figures/numeric_evidence.json.gz') if (output/'figures/numeric_evidence.json.gz').exists() else None)
        (output/'learner_execution.json').write_text(json.dumps(receipt, indent=2)+'\n')
    rendered_student = copy.deepcopy(student)
    for c, actual in zip(rendered_student['cells'], notebook.cells):
        if c['cell_type'] == 'code' and 'raise NotImplementedError' not in c['source']:
            c.update(outputs=copy.deepcopy(actual.outputs), execution_count=actual.execution_count)
    for label, value in [('exercises', nbformat.from_dict(rendered_student)), ('solutions', notebook)]:
        html, _ = HTMLExporter(template_name='lab').from_notebook_node(value)
        (output/f'{label}.html').write_text(html)
    blocks = []
    for c, actual in zip(student['cells'], notebook.cells):
        if c['cell_type'] == 'markdown':
            blocks.append(c['source'])
        elif not c.get('metadata', {}).get('arena', {}).get('evidence_only'):
            blocks.append('```python\n'+c['source']+'\n```')
            for name in re.findall(r"show_inrun\(fig, '([^']+)'\)", c['source']):
                blocks.append(f'![{name.replace("_", " ")}](../assets/{name}.png)')
            if "figure.savefig(INRUN_OUTPUT / '167_01_exact_retraining.png'" in c['source']:
                blocks.append('![Exact retraining and its controls](../assets/167_01_exact_retraining.png)')
            for observed in actual.get('outputs', []):
                if observed.get('output_type') == 'stream':
                    blocks.append('```text\n'+observed['text'].rstrip()+'\n```')
    (output/'07_[16.7]_Data_Shapley_in_One_Training_Run.md').write_text('\n\n'.join(blocks)+'\n')
    print(json.dumps({k:receipt[k] for k in ('status','elapsed_seconds','executed_code_cells','code_cells')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    execute(parser.parse_args().output)

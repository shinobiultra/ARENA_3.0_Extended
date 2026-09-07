"""Complete every stub from its own answer and run a fresh notebook kernel."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import nbformat
from nbclient import NotebookClient
from chapter11_representation_geometry.exercises.part4_concept_erasure.build_notebooks import lesson_cells, complete_learner


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def execute(directory):
    directory.mkdir(parents=True, exist_ok=False)
    source_names = [
        'lesson.md', 'native_protocol.md', 'erasure_methods.py', 'erasure_methods_tests.py',
        'erasure_native.py', 'erasure_data.py', 'erasure_protocol.py', 'reference_parity.py',
        'build_notebooks.py', 'execute_learner.py',
    ]
    source_paths = [HERE / name for name in source_names]
    shared = ROOT / 'chapter11_representation_geometry/exercises/part3_causal_abstraction_das'
    source_paths += [shared / 'das_native.py', shared / 'run_native_development.py']
    hashes = {}
    for source in source_paths:
        relative = str(source.relative_to(ROOT))
        snapshot = directory / 'source_snapshot' / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(source.read_bytes())
        hashes[relative] = digest(source)
    state = dict(status='RUNNING', source_inputs=hashes, completed_from_own_answers=True, native_model_executed=False)
    started = time.perf_counter()
    notebook = None

    def save():
        state['elapsed_seconds'] = time.perf_counter() - started
        (directory / 'execution.json').write_text(json.dumps(state, indent=2) + '\n')

    def finished(cell, cell_index, **kwargs):
        state['last_completed_cell'] = cell_index
        save()

    save()
    try:
        raw = dict(cells=lesson_cells(), metadata={'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}}, nbformat=4, nbformat_minor=5)
        notebook = nbformat.reads(json.dumps(complete_learner(raw)), as_version=4)
        NotebookClient(notebook, timeout=1800, kernel_name='python3', on_cell_executed=finished,
                       resources={'metadata': {'path': str(ROOT)}}).execute(
            env={**os.environ, 'ARENA_ERASURE_OUTPUT_DIRECTORY': str(directory / 'figures')})
        state.update(status='PASS', cells=len(notebook.cells), code_cells=sum(c.cell_type == 'code' for c in notebook.cells),
                     embedded_figures=sum('image/png' in output.get('data', {}) for cell in notebook.cells for output in cell.get('outputs', [])),
                     figures={path.name: digest(path) for path in sorted((directory / 'figures').glob('*.png'))})
        native = directory / 'figures/native_summary.json'
        if native.exists():
            state['native_summary'] = json.loads(native.read_text())
            state['native_model_executed'] = True
            if not state['native_summary']['weights_unchanged']:
                raise ValueError('The native model must stay frozen.')
    except BaseException as error:
        state.update(status='FAIL', error=str(error))
        raise
    finally:
        state['changed_sources'] = [name for name, value in hashes.items() if digest(ROOT / name) != value]
        if state['changed_sources']:
            state['status'] = 'FAIL'
        if notebook is not None:
            nbformat.write(notebook, directory / 'completed_learner.ipynb')
        save()
    if state['status'] != 'PASS':
        raise ValueError('Learner source changed during execution.')
    print(json.dumps(state, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-directory', required=True, type=Path)
    execute(parser.parse_args().output_directory.resolve())

#!/usr/bin/env python3
"""Build only the additive mastery pilot notebooks; --check never writes."""
from __future__ import annotations

import argparse
import copy
import importlib.util
from pathlib import Path
import sys

import nbformat

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / 'mastery_hub/units/u02_optimization'
OUTPUTS = {kind: UNIT / f'{kind}.ipynb' for kind in ('exercises', 'solutions')}


def make_notebook(kind):
    spec = importlib.util.spec_from_file_location('mastery_quadratic_lesson', UNIT / 'lesson.py')
    lesson = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lesson)
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.from_dict(cell) for cell in lesson.build_cells(kind)],
        metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python'},
                  'mastery_hub': {'evidence_class': 'analytic_toy', 'unit_id': 'u02_optimization',
                                  'simulator_version': 'quadratic-heavy-ball-v1'}})
    nbformat.validate(notebook)
    return notebook


def canonical_content(notebook):
    """Ignore genuine execution artifacts, preserving all canonical source/metadata."""
    notebook = copy.deepcopy(notebook)
    notebook.metadata.pop('widgets', None)
    # Kernel-added language details do not change the teaching source.
    notebook.metadata['language_info'] = {'name': 'python'}
    for cell in notebook.cells:
        cell.metadata.pop('execution', None)
        cell.metadata.pop('trusted', None)
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    return notebook


def synchronize(path, notebook, *, check):
    """Read-only on check; preserve bytes and outputs whenever source is unchanged."""
    if path.exists():
        try:
            old = nbformat.read(path, as_version=4)
            if canonical_content(old) == canonical_content(notebook):
                return True
        except (ValueError, nbformat.reader.NotJSONError):
            pass
    if check:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, path)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Detect drift without touching any file')
    parser.add_argument('--execute', action='store_true', help='Execute a fresh solution notebook on CPU')
    args = parser.parse_args(argv)
    if args.check and args.execute:
        parser.error('--check and --execute cannot be combined')
    success = True
    for kind, path in OUTPUTS.items():
        notebook = make_notebook(kind)
        if args.execute and kind == 'solutions':
            from nbclient import NotebookClient
            # Provision an ephemeral kernel pointing at this exact environment; no install needed.
            from jupyter_client.kernelspec import KernelSpecManager
            import tempfile
            import json
            with tempfile.TemporaryDirectory(prefix='mastery-kernel-') as temporary:
                kernel_dir = Path(temporary) / 'mastery-cpu'
                kernel_dir.mkdir()
                (kernel_dir / 'kernel.json').write_text(json.dumps({
                    'argv': [sys.executable, '-m', 'ipykernel_launcher', '-f', '{connection_file}'],
                    'display_name': 'Mastery CPU', 'language': 'python'}))
                manager = KernelSpecManager(kernel_dirs=[temporary])
                client = NotebookClient(notebook, timeout=120, startup_timeout=20, kernel_name='mastery-cpu',
                                        resources={'metadata': {'path': str(ROOT)}})
                from jupyter_client import AsyncKernelManager
                client.km = AsyncKernelManager(kernel_name='mastery-cpu', kernel_spec_manager=manager)
                client.execute()
            nbformat.validate(notebook)
            nbformat.write(notebook, path)
            print(f'Executed: {path.relative_to(ROOT)}')
        else:
            current = synchronize(path, notebook, check=args.check)
            success &= current
            print(f'{"OK" if current else "DRIFT"}: {path.relative_to(ROOT)}')
    return 0 if success else 1


if __name__ == '__main__':
    raise SystemExit(main())

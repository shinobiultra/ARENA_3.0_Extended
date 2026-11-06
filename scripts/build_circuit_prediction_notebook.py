#!/usr/bin/env python3
"""Build only the additive blind circuit prediction notebook; preserve learner work and --check is read-only."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / 'mastery_hub/units/circuit_prediction'
OUTPUT = UNIT / 'exercises.ipynb'
BASELINE_KEY = 'source_baseline_sha256'


def canonical_content(notebook):
    """Separate generated source/teaching metadata from execution artifacts."""
    notebook = deepcopy(notebook)
    notebook.metadata.pop('widgets', None)
    notebook.metadata['language_info'] = {'name': 'python'}
    notebook.metadata.get('circuit_prediction', {}).pop(BASELINE_KEY, None)
    for cell in notebook.cells:
        cell.metadata.pop('execution', None)
        cell.metadata.pop('trusted', None)
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    return notebook


def source_fingerprint(notebook):
    payload = json.dumps(canonical_content(notebook), sort_keys=True, ensure_ascii=False,
                         separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def make_notebook():
    spec = importlib.util.spec_from_file_location('circuit_prediction_lesson', UNIT / 'lesson.py')
    lesson = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lesson)
    notebook = nbformat.v4.new_notebook(cells=[nbformat.from_dict(c) for c in lesson.build_cells()],
        metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                  'language_info': {'name': 'python'},
                  'circuit_prediction': {'schema': lesson.SCHEMA, 'schema_version': lesson.VERSION,
                      'status': 'exploratory_blind_variant', 'case_count': 1,
                      'automatic_experiment_execution': False}})
    notebook.metadata['circuit_prediction'][BASELINE_KEY] = source_fingerprint(notebook)
    nbformat.validate(notebook)
    return notebook


def _has_execution(notebook):
    return any(cell.cell_type == 'code' and (cell.outputs or cell.execution_count is not None)
               for cell in notebook.cells)


def synchronize(path, expected, *, check):
    """Update only pristine generated source; never discard filled cells or records.

    A source baseline identifies an unedited older generation. If source is current,
    leave every byte (including records/runtime metadata) intact. If generation is
    stale, any execution evidence prevents rewriting it. A learner can preserve or
    move their notebook explicitly before requesting a new blank workbook.
    """
    path = Path(path)
    if path.exists():
        try:
            old = nbformat.read(path, as_version=4)
            nbformat.validate(old)
        except (ValueError, nbformat.ValidationError, nbformat.reader.NotJSONError):
            return False
        old_fingerprint = source_fingerprint(old)
        expected_fingerprint = source_fingerprint(expected)
        baseline = old.metadata.get('circuit_prediction', {}).get(BASELINE_KEY)
        if old_fingerprint == expected_fingerprint:
            # A legacy blank generation can acquire its baseline safely. Saved
            # outputs are always kept byte-for-byte, even without a baseline.
            if baseline is None and not _has_execution(old):
                if check:
                    return False
                old.metadata.setdefault('circuit_prediction', {})[BASELINE_KEY] = old_fingerprint
                nbformat.write(old, path)
            return True
        if check or baseline != old_fingerprint or _has_execution(old):
            return False
    elif check:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(expected, path)
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    # Fixed CLI output; callers cannot redirect generation into original chapters.
    current = synchronize(OUTPUT, make_notebook(), check=args.check)
    message = 'OK' if current else 'DRIFT / preserved: source changed, saved work, or unreadable artifact'
    print(f'{message}: {OUTPUT.relative_to(ROOT)}')
    return 0 if current else 1


if __name__ == '__main__':
    raise SystemExit(main())

#!/usr/bin/env python3
"""Generate one blank companion; --check is read-only and learner work is preserved."""
import argparse
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / 'mastery_hub/units/clip_prediction'
OUTPUT = UNIT / 'exercises.ipynb'
BASELINE = 'source_baseline_sha256'


def fingerprint(notebook):
    value = deepcopy(notebook)
    value.get('metadata', {}).get('clip_prediction', {}).pop(BASELINE, None)
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'),
                     ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def module():
    spec = importlib.util.spec_from_file_location('clip_notebook_source', UNIT / 'lesson.py')
    lesson = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lesson)
    return lesson


def make_notebook():
    lesson = module()
    notebook = {
        'nbformat': 4, 'nbformat_minor': 5, 'cells': lesson.build_cells(),
        'metadata': {
            'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
            'language_info': {'name': 'python'},
            'clip_prediction': {'status': 'unexecuted_source_candidate',
                                'automatic_experiment_execution': False},
        },
    }
    notebook['metadata']['clip_prediction'][BASELINE] = fingerprint(notebook)
    return notebook


def has_work(raw):
    return ('widgets' in raw.get('metadata', {}) or any(
        'execution' in cell.get('metadata', {}) or 'trusted' in cell.get('metadata', {})
        or cell.get('outputs') or cell.get('execution_count') is not None
        for cell in raw.get('cells', [])
    ))


def synchronize(path, expected, *, check):
    path = Path(path)
    if path.is_symlink():
        return False
    if path.exists():
        try:
            old = module().parse(path.read_text())
            if type(old) is not dict or type(old.get('cells')) is not list:
                return False
            if type(old.get('metadata')) is not dict:
                return False
            if not all(type(c) is dict and type(c.get('metadata', {})) is dict
                       for c in old['cells']):
                return False
            # Typed JSON equality prevents True from masquerading as integer 1.
            if json.dumps(old, sort_keys=True) == json.dumps(expected, sort_keys=True):
                return True
            # Conservative refusal also covers pristine but outdated templates.
            # A preflight fingerprint cannot authorize overwriting later work.
            return False
        except (ValueError, UnicodeError, AttributeError):
            return False
    elif check:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        return False  # late learner file or symlink; preserve it byte-for-byte
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(json.dumps(expected, indent=1, ensure_ascii=False) + '\n')
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    ok = synchronize(OUTPUT, make_notebook(), check=args.check)
    print(('OK' if ok else 'DRIFT / learner work preserved') + ': ' + str(OUTPUT))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())

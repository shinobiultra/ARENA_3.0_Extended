"""Render the actual exercise scaffold with recorded demonstration outputs for review."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import nbformat
from nbconvert import HTMLExporter

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.build_notebooks import build_notebooks, STEM
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.reproduce_native import validate_current
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.execute_learner import digest


def exercise_with_outputs(learner):
    """Keep exercise stubs; only copy demonstration outputs by matching cell ID/source."""
    validate_current(learner)
    student, solved = build_notebooks()
    executed = nbformat.read(Path(learner) / 'completed_learner.ipynb', as_version=4)
    assert [(c.cell_type, c.source) for c in executed.cells] == [(c['cell_type'], c['source']) for c in solved['cells']]
    by_id = {cell.id: cell for cell in executed.cells}
    for cell in student['cells']:
        if cell['cell_type'] == 'code' and not cell.get('metadata', {}).get('arena', {}).get('exercise_functions'):
            actual = by_id[cell['id']]
            assert cell['source'] == actual.source
            cell['outputs'] = copy.deepcopy(actual.outputs)
            cell['execution_count'] = actual.execution_count
    return nbformat.reads(json.dumps(student), as_version=4), executed


def render(learner):
    learner = Path(learner).resolve()
    directory = learner / 'html_review'
    directory.mkdir(exist_ok=False)
    student, solved = exercise_with_outputs(learner)
    outputs = {}
    for name, notebook in [('exercises', student), ('solutions', solved)]:
        nbformat.write(notebook, directory / f'DRAFT_{STEM}_{name}.ipynb')
        body, _ = HTMLExporter().from_notebook_node(notebook)
        path = directory / f'{name}.html'
        path.write_text(body)
        outputs[path.name] = digest(path)
    record = {'status': 'RENDERED_REQUIRES_VISUAL_REVIEW', 'learner_execution_sha256': digest(learner / 'execution.json'),
              'renderer_sha256': digest(Path(__file__)), 'html': outputs, 'demonstrations_from_recorded_execution': True,
              'student_exercise_stubs_preserved': True, 'independent_review': False}
    (directory / 'render.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learner', type=Path, required=True)
    render(parser.parse_args().learner)

"""Publish the executed, reproduced and author-reviewed SHAP-IQ learner."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shutil
import sys

import nbformat
import torch as t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.build_notebooks import STEM
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.execute_learner import digest
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.review_learner import exercise_with_outputs
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.reproduce_native import PAYLOADS, tree_inventory, validate_current

PAGE = HERE.parents[1] / 'instructions/pages/17_[16.17]_One_Query_Many_Interactions.md'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_fresh(learner, fresh):
    learner, fresh = Path(learner).resolve(), Path(fresh).resolve()
    execution = validate_current(learner)
    record = json.loads((fresh / 'reproduction.json').read_text())
    require(record['status'] == 'PASS' and not record['mismatches'], 'Fresh reproduction must pass.')
    require(record['learner_directory'] == str(learner), 'Wrong reproduced learner.')
    require(record['learner_execution_sha256'] == digest(learner / 'execution.json'), 'Execution receipt changed.')
    require(record['reproduction_script_sha256'] == digest(HERE / 'reproduce_native.py'), 'Reproducer changed.')
    require(record['source_inputs'] == execution['source_inputs'], 'Fresh run used different inputs.')
    require(record['native_models_refit'] and not record['cached_model_inputs_used'] and not record['child_received_learner_payloads'], 'Fresh fits are required.')
    for name, checksum in record['source_inputs'].items():
        require(digest(fresh / 'source_snapshot' / name) == checksum, 'Fresh source snapshot changed: ' + name)
    totals = dict.fromkeys(('arrays', 'array_elements', 'scalars', 'containers'), 0)
    summary = json.loads((fresh / 'fresh_native/summary.json').read_text())
    require(set(summary['artifacts']) == {name + '.pt' for name in PAYLOADS}, 'Incomplete fresh physical inventory.')
    for name in PAYLOADS:
        path = fresh / 'fresh_native' / (name + '.pt')
        physical = summary['artifacts'][path.name]
        require(path.stat().st_size == physical['bytes'] and digest(path) == physical['sha256'], 'Fresh physical artifact changed: ' + name)
        for directory, side in ((learner / 'figures/native', 'original'), (fresh / 'fresh_native', 'fresh')):
            value = t.load(directory / (name + '.pt'), map_location='cpu', weights_only=False)
            actual = tree_inventory(value)
            del value
            require(actual == record['inventories'][name][side], 'Retained deterministic inventory changed: ' + name + '/' + side)
            if side == 'original':
                for key, count in actual['counts'].items():
                    totals[key] += count
        require(record['inventories'][name]['original'] == record['inventories'][name]['fresh'], 'Nonidentical native payloads: ' + name)
    require(totals == record['counts'], 'Incorrect reproduction totals.')
    return execution, record


def validate_review(learner):
    learner = Path(learner).resolve()
    record = json.loads((learner / 'html_review/author_visual_review.json').read_text())
    require(record['status'] == 'PASS_AUTHOR_REVIEW' and record['independent_review'] is False, 'A truthful author review is required.')
    require(record['learner_execution_sha256'] == digest(learner / 'execution.json'), 'Review refers to another execution.')
    require(record['renderer_sha256'] == digest(HERE / 'review_learner.py'), 'Renderer changed after visual review.')
    for relative, checksum in record['reviewed_files'].items():
        path = (learner / relative).resolve()
        path.relative_to(learner)
        require(digest(path) == checksum, 'Reviewed evidence changed: ' + relative)
    execution = json.loads((learner / 'execution.json').read_text())
    require(record['figures'] == execution['figures'] and len(record['figures']) == 10, 'All ten actual figures must be reviewed.')
    for name, item in execution['figures'].items():
        require(digest(learner / 'figures' / name) == item['sha256'], 'Reviewed figure changed: ' + name)
    require(all(record['checks'].values()), 'Visual or play review failed.')
    return record


def _figure_names(source):
    names = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'show_figure' and len(node.args) >= 2:
            value = node.args[1]
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                names.append(value.value + '.png')
    return names


def publish(learner, fresh):
    learner, fresh = Path(learner).resolve(), Path(fresh).resolve()
    execution, reproduction = validate_fresh(learner, fresh)
    review = validate_review(learner)
    student, solved = exercise_with_outputs(learner)
    assets = PAGE.parent.parent / 'assets'
    assets.mkdir(exist_ok=True)
    pieces, shown, published_figures = [], [], {}
    for cell in student.cells:
        if cell.cell_type == 'markdown':
            pieces.append(cell.source)
        else:
            pieces.append('```python\n' + cell.source + '\n```')
            names = _figure_names(cell.source)
            image_count = sum('image/png' in output.get('data', {}) for output in cell.outputs)
            require(len(names) == image_count, 'Figure calls and actual output count differ.')
            for name in names:
                require(name in execution['figures'], 'Unrecorded figure: ' + name)
                target = assets / ('shapiq_' + name)
                shutil.copy2(learner / 'figures' / name, target)
                published_figures[target.name] = digest(target)
                pieces.append(f'![Actual learner result: {name[3:-4].replace("_", " ")}](../assets/{target.name})')
                shown.append(name)
    require(sorted(shown) == sorted(execution['figures']), 'Missing or repeated figure in the lesson.')
    for variant, notebook in [('exercises', student), ('solutions', solved)]:
        notebook.metadata.arena.status = 'published_author_reviewed'
        nbformat.write(notebook, HERE / f'{STEM}_{variant}.ipynb')
    PAGE.write_text('\n\n'.join(pieces) + '\n')
    expected = HERE / 'expected_outputs'
    expected.mkdir(exist_ok=True)
    for source, name in ((learner / 'execution.json', 'own_answer_execution.json'),
                         (fresh / 'reproduction.json', 'fresh_reproduction.json'),
                         (learner / 'html_review/author_visual_review.json', 'author_visual_review.json'),
                         (learner / 'figures/native/summary.json', 'native_summary.json')):
        shutil.copy2(source, expected / name)
    record = {**execution, 'learner_directory': str(learner), 'publication_fresh_directory': str(fresh),
              'published_exercises_sha256': digest(HERE / f'{STEM}_exercises.ipynb'),
              'published_solutions_sha256': digest(HERE / f'{STEM}_solutions.ipynb'),
              'published_page_sha256': digest(PAGE), 'published_figures': published_figures,
              'publication_review_scope': 'author_review_not_independent',
              'publication_review_sha256': digest(expected / 'author_visual_review.json'),
              'publisher_sha256': digest(Path(__file__))}
    (expected / 'learner_execution.json').write_text(json.dumps(record, indent=2) + '\n')
    print('Published the own-answer checked pair, instruction page and ten actual figures.')
    return record


def validate_published():
    path = HERE / 'expected_outputs/learner_execution.json'
    record = json.loads(path.read_text())
    for artifact, field in ((HERE / f'{STEM}_exercises.ipynb', 'published_exercises_sha256'),
                            (HERE / f'{STEM}_solutions.ipynb', 'published_solutions_sha256'),
                            (PAGE, 'published_page_sha256')):
        require(digest(artifact) == record[field], 'Published learner changed: ' + str(artifact))
    for name, checksum in record['published_figures'].items():
        require(name.startswith('shapiq_') and Path(name).name == name, 'Invalid published figure namespace.')
        require(digest(PAGE.parent.parent / 'assets' / name) == checksum, 'Published figure changed: ' + name)
    require(record['publisher_sha256'] == digest(Path(__file__)), 'Publisher changed since publication.')
    require(digest(HERE / 'expected_outputs/author_visual_review.json') == record['publication_review_sha256'], 'Published review receipt changed.')
    validate_review(record['learner_directory'])
    validate_current(record['learner_directory'])
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learner', type=Path, required=True)
    parser.add_argument('--fresh', type=Path, required=True)
    args = parser.parse_args()
    publish(args.learner, args.fresh)

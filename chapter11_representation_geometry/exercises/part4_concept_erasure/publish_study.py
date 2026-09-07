"""Publish only the executed, visually reviewed and freshly reproduced learner."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from chapter11_representation_geometry.exercises.part4_concept_erasure.build_notebooks import HERE, STEM, lesson_cells, complete_learner
from chapter11_representation_geometry.exercises.part4_concept_erasure.erasure_reproduction import (
    validate_learner, expected_conditions, file_digest, require,
)

FIGURES = (
    'erasure_four_point_geometry.png', 'erasure_constraint_and_distortion.png',
    'erasure_probe_refitting.png', 'erasure_nonlinear_counterexample.png',
    'erasure_covariance_play.png', 'erasure_native_development.png',
    'erasure_final_probe_recovery.png', 'erasure_covariance_transfer.png',
    'erasure_final_task_and_collateral.png', 'erasure_distortion_and_native_kl.png',
    'erasure_actual_token_gallery.png', 'erasure_native_strength_play.png',
)


def publish(learner, fresh):
    learner, fresh = Path(learner).resolve(), Path(fresh).resolve()
    execution, summary = validate_learner(learner)
    reproduction = json.loads((fresh / 'reproduction.json').read_text())
    require(reproduction['status'] == 'PASS' and not reproduction['changed_sources'], 'Fresh execution failed or changed.')
    require(reproduction['learner_directory'] == str(learner), 'The fresh run used another learner.')
    require(reproduction['learner_execution_sha256'] == file_digest(learner / 'execution.json'), 'Learner evidence changed after reproduction.')
    require(reproduction['actual_intervention_rows'] == 71232, 'Not all native outcomes were repeated.')
    require(reproduction['weights_unchanged'] and reproduction['model_gradients_absent'], 'Fresh model weights changed.')
    actual = [(row['site'], row['method'], row['tensor_file']) for row in reproduction['conditions']]
    require(actual == expected_conditions(), 'Missing or substituted fresh conditions.')
    for name, digest in reproduction['source_inputs'].items():
        require(file_digest(fresh / 'source_snapshot' / name) == digest, 'Fresh source snapshot changed: ' + name)
        if name != str((HERE / 'lesson.md').relative_to(ROOT)):
            require(file_digest(ROOT / name) == digest, 'Current fresh scientific source changed: ' + name)
    require(file_digest(fresh / 'actual_comparisons.jsonl') == reproduction['actual_comparisons_sha256'], 'Fresh comparisons changed.')
    require(file_digest(fresh / 'fresh_fitted_conditions.pt') == reproduction['fitted_conditions_sha256'], 'Fresh fitted maps changed.')
    for row in reproduction['conditions']:
        require(file_digest(fresh / row['tensor_file']) == row['tensor_sha256'], 'Fresh native tensors changed: ' + row['tensor_file'])
        require(file_digest(learner / 'figures' / row['tensor_file']) == summary['condition_tensor_sha256'][row['tensor_file']],
                'Learner native tensors changed: ' + row['tensor_file'])
        for task in ['final', 'object_final']:
            require(row[task]['all_winners_equal'] and row[task]['full_vocabulary_max_error'] <= 3e-4, 'Native predictions did not reproduce.')
    review = json.loads((learner / 'visual_review.json').read_text())
    require(review['status'] == 'PASS' and review['browser_math_and_dropdowns_checked'], 'Actual browser and figure review is required.')
    require(review['lesson_sha256'] == file_digest(HERE / 'lesson.md'), 'The reviewed lesson changed.')
    actual_figures = {name: file_digest(learner / 'figures' / name) for name in FIGURES}
    require(review['figures'] == actual_figures, 'Not every current figure was reviewed.')
    student = dict(cells=lesson_cells(), metadata={
        'kernelspec': dict(display_name='Python 3', language='python', name='python3'),
        'language_info': dict(name='python'),
    }, nbformat=4, nbformat_minor=5)
    solved = complete_learner(student)
    executed = json.loads((learner / 'completed_learner.ipynb').read_text())
    for exercise, answer, actual_cell in zip(student['cells'], solved['cells'], executed['cells'], strict=True):
        if exercise['cell_type'] == 'code':
            answer.update(outputs=copy.deepcopy(actual_cell['outputs']), execution_count=actual_cell['execution_count'])
            if 'raise NotImplementedError' not in ''.join(exercise['source']):
                exercise.update(outputs=copy.deepcopy(actual_cell['outputs']), execution_count=actual_cell['execution_count'])
    for suffix, notebook in [('exercises', student), ('solutions', solved)]:
        (HERE / f'{STEM}_{suffix}.ipynb').write_text(json.dumps(notebook, indent=1) + '\n')
    assets = HERE.parents[1] / 'instructions/assets'
    for name in FIGURES:
        (assets / name).write_bytes((learner / 'figures' / name).read_bytes())
    pieces, displayed = [], []
    for cell in student['cells']:
        source = ''.join(cell['source']).strip()
        if cell['cell_type'] == 'markdown':
            pieces.append(source)
        else:
            pieces.append('```python\n' + source + '\n```')
            for name in FIGURES:
                if name in source:
                    pieces.append(f"![Actual learner result: {name.removesuffix('.png').replace('_', ' ')}](../assets/{name})")
                    displayed.append(name)
    require(sorted(displayed) == sorted(FIGURES), 'Every figure must appear exactly once in the instruction page.')
    page = HERE.parents[1] / 'instructions/pages/04_[11.4]_Concept_Erasure.md'
    page.write_text('\n\n'.join(pieces) + '\n')
    expected = HERE / 'expected_outputs'
    expected.mkdir(exist_ok=True)
    for origin, target in [(learner / 'figures/native_summary.json', 'native_summary.json'),
                           (fresh / 'reproduction.json', 'fresh_reproduction.json')]:
        (expected / target).write_bytes(origin.read_bytes())
    record = dict(execution, learner_directory=str(learner), fresh_directory=str(fresh),
        full_release_validation=True, published_code_matches_executed_own_answers=True,
        published_exercises_sha256=file_digest(HERE / f'{STEM}_exercises.ipynb'),
        published_solutions_sha256=file_digest(HERE / f'{STEM}_solutions.ipynb'),
        published_page_sha256=file_digest(page), published_figures=actual_figures, visual_review=review)
    (expected / 'learner_execution.json').write_text(json.dumps(record, indent=2) + '\n')
    print(f"Published {len(student['cells'])} cells, {execution['code_cells']} executed code cells, {len(FIGURES)} controlled figures.")
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learner-directory', required=True, type=Path)
    parser.add_argument('--fresh-directory', required=True, type=Path)
    args = parser.parse_args()
    publish(args.learner_directory, args.fresh_directory)

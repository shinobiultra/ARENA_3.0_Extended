"""Publish a completed own-answer study, then leave CUDA acceptance to the report.

This command performs file/provenance work only. It never trains a hidden model
or substitutes an independent candidate for the learner's own experiment.
"""
from __future__ import annotations

import argparse
import ast
import gzip
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from chapter5_modern_architectures.exercises.part5_diffusion_language_models.build_notebooks import HERE,STEM,MARKER,lesson_cells,complete_learner
from chapter5_modern_architectures.exercises.part5_diffusion_language_models.study_evidence import validate_text_study,sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--learner-directory',type=Path,required=True)
    parser.add_argument('--data-directory',type=Path,required=True)
    args=parser.parse_args();run=args.learner_directory
    audit=validate_text_study(run,args.data_directory)
    old=json.loads((HERE/f'{STEM}_exercises.ipynb').read_text())
    learner={**old,'cells':lesson_cells()}
    solved=complete_learner(learner)
    actual=json.loads((run/'completed_learner.ipynb').read_text())
    if len(actual['cells'])!=len(solved['cells']):raise ValueError('Published and completed learner cell inventories differ')
    for index,(executed,reference) in enumerate(zip(actual['cells'],solved['cells'])):
        if executed['cell_type']!=reference['cell_type']:raise ValueError('Published cell kinds differ')
        if reference['cell_type']!='code':continue
        actual_code=''.join(executed['source']);expected_code=''.join(reference['source'])
        if index==1:
            suffix=f'\nassets_dir = Path({str(run/"figures")!r})\n'
            if not actual_code.endswith(suffix):raise ValueError('Unexpected output-path execution override')
            actual_code=actual_code[:-len(suffix)]
        if ast.dump(ast.parse(actual_code))!=ast.dump(ast.parse(expected_code)):
            raise ValueError('Actual executed code differs from the learner own answers')
        reference['outputs']=executed['outputs'];reference['execution_count']=executed['execution_count']
    for suffix,notebook in [('exercises',learner),('solutions',solved)]:
        (HERE/f'{STEM}_{suffix}.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
    for figure in (run/'figures').glob('*.png'):
        (HERE.parents[1]/'instructions/assets'/figure.name).write_bytes(figure.read_bytes())
    page=MARKER.sub('',(HERE/'lesson.md').read_text()).lstrip()
    anchors={
        '## Try It Yourself\n':['diffusion_copy_controls.png','diffusion_copy_trajectory.png'],
        '## Signature Result — answers, controls and all sampling budgets':['diffusion_text_training.png'],
        '### Interpret the whole task matrix':['diffusion_text_task_controls.png','diffusion_text_budget_quality.png',
            'diffusion_text_temperature_blocks.png','diffusion_text_arithmetic_gallery.png'],
        '### Measure actual cost on the same batch':['diffusion_text_trajectory.png','diffusion_text_entropy_commitment.png','diffusion_text_story_gallery.png'],
        '## Try It Yourself — change the fact, not just the answer label':['diffusion_text_cost_quality.png'],
        '## Try It Yourself — continue the same real prefix':['diffusion_text_schedule_play.png'],
    }
    for anchor,names in anchors.items():
        if page.count(anchor)!=1:raise ValueError('Missing or ambiguous figure placement anchor')
        embeds='\n\n'.join(f'![Actual learner result: {name.removesuffix(".png").replace("_"," ")}](../assets/{name})' for name in names)
        page=page.replace(anchor,embeds+'\n\n'+anchor,1)
    (HERE.parents[1]/'instructions/pages/05_[5.5]_Diffusion_Language_Models.md').write_text(page.strip()+'\n')
    expected=HERE/'expected_outputs';expected.mkdir(exist_ok=True)
    (expected/'completed_text_study.json.gz').write_bytes((run/'experiments/completed_text_study.json.gz').read_bytes())
    for name in ('protocol.json','checkpoint_selection.json'):
        (expected/name).write_bytes((run/'experiments'/name).read_bytes())
    data_target=expected/'text_data';data_target.mkdir(exist_ok=True)
    archive={}
    for path in sorted(args.data_directory.glob('*.json')):
        contents=path.read_bytes();compressed=gzip.compress(contents,mtime=0)
        (data_target/(path.name+'.gz')).write_bytes(compressed)
        archive[path.name]={'sha256':hashlib.sha256(contents).hexdigest(),'gzip_sha256':hashlib.sha256(compressed).hexdigest()}
    (data_target/'manifest.json').write_text(json.dumps(archive,indent=2)+'\n')
    manifest=json.loads((run/'execution.json').read_text())
    manifest.update(data_directory=str(args.data_directory.resolve()),measurement_audit=audit,
        published_evidence_sha256=sha256(expected/'completed_text_study.json.gz'),
        published_exercises_sha256=sha256(HERE/f'{STEM}_exercises.ipynb'),
        published_solutions_sha256=sha256(HERE/f'{STEM}_solutions.ipynb'),
        published_page_sha256=sha256(HERE.parents[1]/'instructions/pages/05_[5.5]_Diffusion_Language_Models.md'))
    (expected/'text_learner_execution.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Published actual 90-cell learner, 23 exercise groups, 12 figures and all measured rows; independent CUDA report remains required.')


if __name__=='__main__':main()

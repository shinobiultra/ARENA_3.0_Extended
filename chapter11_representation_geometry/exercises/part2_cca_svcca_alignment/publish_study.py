"""Publish the exact completed own-answer learner and its actual figures."""
from __future__ import annotations
import argparse,copy,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment.build_notebooks import HERE,STEM,lesson_cells,complete_learner
from chapter11_representation_geometry.exercises.part2_cca_svcca_alignment.alignment_evidence import validate_learner,FIGURES,sha256


def publish(learner_directory,fresh_directory):
    run=Path(learner_directory).resolve();fresh=Path(fresh_directory).resolve();audit=validate_learner(run)
    reproduction=json.loads((fresh/'fresh_reproduction.json').read_text())
    if reproduction['status']!='PASS' or reproduction['changed_sources'] or reproduction['learner_directory']!=str(run):
        raise ValueError('Publish only with the completed fresh reproduction of this learner.')
    for name,digest in reproduction['source_inputs'].items():
        if sha256(HERE/name)!=digest:raise ValueError('Fresh reproduction source changed: '+name)
    learner={'cells':lesson_cells(),'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                             'language_info':{'name':'python'}},'nbformat':4,'nbformat_minor':5}
    solved=complete_learner(learner);executed=json.loads((run/'completed_learner.ipynb').read_text())['cells']
    for student,answer,actual in zip(learner['cells'],solved['cells'],executed,strict=True):
        if student['cell_type']!='code':continue
        answer.update(outputs=copy.deepcopy(actual['outputs']),execution_count=actual['execution_count'])
        if 'raise NotImplementedError' not in ''.join(student['source']):
            student.update(outputs=copy.deepcopy(actual['outputs']),execution_count=actual['execution_count'])
    for suffix,notebook in [('exercises',learner),('solutions',solved)]:
        (HERE/f'{STEM}_{suffix}.ipynb').write_text(json.dumps(notebook,indent=1)+'\n')
    assets=HERE.parents[1]/'instructions/assets';assets.mkdir(exist_ok=True)
    for name in FIGURES:(assets/name).write_bytes((run/'figures'/name).read_bytes())
    pieces=[];shown=[]
    for cell in learner['cells']:
        source=''.join(cell['source']).strip()
        if cell['cell_type']=='markdown':pieces.append(source);continue
        pieces.append('```python\n'+source+'\n```')
        for name in FIGURES:
            if name in source:
                pieces.append(f'![Actual learner result: {name.removesuffix(".png").replace("_"," ")}](../assets/{name})');shown.append(name)
    if shown!=[name for cell in learner['cells'] if cell['cell_type']=='code' for name in FIGURES if name in ''.join(cell['source'])] or set(shown)!=set(FIGURES) or len(shown)!=len(FIGURES):
        raise ValueError('Each measured figure must occur once in the instruction page.')
    page=HERE.parents[1]/'instructions/pages/02_[11.2]_CCA_SVCCA_and_Representation_Alignment.md'
    page.write_text('\n\n'.join(pieces)+'\n')
    expected=HERE/'expected_outputs';expected.mkdir(exist_ok=True)
    (expected/'alignment_study.json.gz').write_bytes((run/'actual_alignment_study/alignment_study.json.gz').read_bytes())
    (expected/'fresh_reproduction.json').write_bytes((fresh/'fresh_reproduction.json').read_bytes())
    record=json.loads((run/'execution.json').read_text())
    record.update(learner_directory=str(run),fresh_directory=str(fresh),full_release_validation=True,
                  measurement_audit=audit,published_code_matches_executed_own_answers=True,
                  published_exercises_sha256=sha256(HERE/f'{STEM}_exercises.ipynb'),published_solutions_sha256=sha256(HERE/f'{STEM}_solutions.ipynb'),
                  published_page_sha256=sha256(page),published_figures={name:sha256(assets/name) for name in FIGURES})
    (expected/'learner_execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'Published {len(learner["cells"])} cells, {audit["own_answer_code_cells"]} own-answer code cells, {len(FIGURES)} actual figures.')
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--learner-directory',type=Path,required=True);parser.add_argument('--fresh-directory',type=Path,required=True)
    args=parser.parse_args();publish(args.learner_directory,args.fresh_directory)


if __name__=='__main__':main()

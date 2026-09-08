"""Publish the real, completed learner with a checked heading-only render update."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.build_notebooks import HERE,STEM,lesson_cells,complete_learner
from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.geometry_evidence import FIGURES,sha256,validate_learner,validate_plot_heading_update


def publish(run):
    run=Path(run).resolve();audit=validate_learner(run)
    learner={'cells':lesson_cells(),'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
                                            'language_info':{'name':'python','version':'3.14'}},'nbformat':4,'nbformat_minor':5}
    solved=complete_learner(learner);actual=json.loads((run/'completed_learner.ipynb').read_text())
    executed=iter(c for c in actual['cells'] if c['cell_type']=='code')
    heading_record=None
    if sha256(HERE/'geometry_plotting.py')!=json.loads((run/'execution.json').read_text())['source_inputs']['geometry_plotting.py']:
        heading_record=validate_plot_heading_update(run)
        rendered=json.loads((run/'layout_update/render.ipynb').read_text())
        updated_image=next(o for c in rendered['cells'] for o in c.get('outputs',[]) if 'image/png' in o.get('data',{}))
    for student,answer in zip(learner['cells'],solved['cells'],strict=True):
        if answer['cell_type']!='code':continue
        measured=next(executed);outputs=copy.deepcopy(measured['outputs'])
        if heading_record and 'plots.plot_angle_play(angle_records, assets_dir)' in ''.join(answer['source']):
            outputs=[copy.deepcopy(updated_image) if 'image/png' in o.get('data',{}) else o for o in outputs]
        answer.update(outputs=outputs,execution_count=measured['execution_count'])
        # Worked examples retain their actual visible outputs; exercises keep their stubs.
        if 'raise NotImplementedError' not in ''.join(student['source']):
            student.update(outputs=copy.deepcopy(outputs),execution_count=measured['execution_count'])
    for suffix,nb in [('exercises',learner),('solutions',solved)]:
        (HERE/f'{STEM}_{suffix}.ipynb').write_text(json.dumps(nb,indent=1)+'\n')
    for name in FIGURES:
        source=run/'layout_update'/name if heading_record and name=='geometry_mistral_angle_radius.png' else run/'figures'/name
        (HERE.parents[1]/'instructions/assets'/name).write_bytes(source.read_bytes())
    call_figures={'plots.plot_calendar_projections(':['geometry_mistral_projections.png'],
                  'plots.plot_calendar_layers(':['geometry_mistral_layer_sweep.png'],
                  'plots.plot_calendar_final_controls(':['geometry_mistral_final_controls.png'],
                  'plots.plot_calendar_all_donors(':['geometry_mistral_all_donors.png'],
                  'plots.plot_calendar_examples(':['geometry_mistral_actual_answers.png'],
                  'plots.plot_projection_sweep(':['geometry_projection_stability.png'],
                  'plots.plot_projection_gallery(':['geometry_projection_gallery.png'],
                  'plots.plot_angle_play(angle_records, assets_dir)':['geometry_mistral_angle_radius.png']}
    pieces=[];shown=[]
    for cell in learner['cells']:
        source=''.join(cell['source']).strip()
        if cell['cell_type']=='markdown':pieces.append(source);continue
        pieces.append('```python\n'+source+'\n```')
        figures=[name for name in FIGURES if name in source]
        for call,names in call_figures.items():
            if call in source:figures+=names
        for name in figures:
            if name in shown:raise ValueError('Ambiguous figure placement: '+name)
            pieces.append(f'![Actual learner result: {name.removesuffix(".png").replace("_"," ")}](../assets/{name})');shown.append(name)
    if set(shown)!=set(FIGURES):raise ValueError('Not every actual figure is placed in the instruction page.')
    page=HERE.parents[1]/'instructions/pages/01_[11.1]_PCA_SVD_and_Geometry_Controls.md'
    page.write_text('\n\n'.join(pieces)+'\n')
    expected=HERE/'expected_outputs';measured=expected/'measured_geometry';measured.mkdir(exist_ok=True)
    for source in (run/'experiments').iterdir():
        if source.suffix=='.pt':continue
        (measured/source.name).write_bytes(source.read_bytes())
    record=json.loads((run/'execution.json').read_text())
    record.update(learner_directory=str(run),full_release_validation=True,measurement_audit=audit,heading_only_render_update=heading_record,
                  published_code_matches_executed_own_answers=True,
                  published_exercises_sha256=sha256(HERE/f'{STEM}_exercises.ipynb'),published_solutions_sha256=sha256(HERE/f'{STEM}_solutions.ipynb'),
                  published_page_sha256=sha256(page),published_figures={name:sha256(HERE.parents[1]/'instructions/assets'/name) for name in FIGURES})
    (expected/'learner_execution.json').write_text(json.dumps(record,indent=2)+'\n')
    print(f'Published {len(learner["cells"])} cells, {audit["own_answer_code_cells"]} executed code cells and {len(shown)} actual figures.')
    return record


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--learner-directory',type=Path,required=True)
    args=parser.parse_args();publish(args.learner_directory)


if __name__=='__main__':main()

"""Execute the complete own-answer learner and bind its actual sources and outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.build_notebooks import build_notebooks,complete_from_own_answers


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_paths():
    # Bind actual learner inputs. Reports/README/publication metadata are separate
    # consumers of this record and cannot retroactively become kernel inputs.
    section=[HERE/name for name in (
        'estimator_methods.py','estimator_methods_tests.py','estimator_study.py',
        'estimator_plots.py','lesson.md','lesson_cells.py','build_notebooks.py',
        'execute_learner.py','protocol.md','data_manifest.json','protocol_freeze.json',
    )]
    published=HERE.parent/'part15_gradientshap'
    dependencies=sorted(published.glob('*.py'))+[published/'data_manifest.json']
    return section+dependencies+[ROOT/'uv.lock']


def validate_design():
    record=json.loads((HERE/'protocol_freeze.json').read_text())
    for name in ('protocol.md','data_manifest.json'):
        if digest(HERE/name)!=record['scientific_design'][name]:
            raise ValueError('Frozen scientific design changed: '+name)
    if record['native_fits_at_freeze']!=0 or record['native_estimator_runs_at_freeze']!=0:
        raise ValueError('Expected an actual pre-native scientific record.')
    return record


def execute(output:Path,analytic_only=False):
    import nbformat
    from nbclient import NotebookClient
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):raise ValueError('Use a new empty output directory.')
    output.mkdir(parents=True,exist_ok=True)
    validate_design()
    student,_=build_notebooks();completed=complete_from_own_answers(student)
    if analytic_only:
        stop=next(i for i,c in enumerate(completed['cells']) if c.get('metadata',{}).get('arena',{}).get('calling_cell')=='native_study')
        completed['cells']=completed['cells'][:stop]
    # nbformat's JSON reader normalizes all source fields before kernel execution.
    nb=nbformat.reads(json.dumps(completed),as_version=4)
    before={str(p.relative_to(ROOT)):digest(p) for p in source_paths()}
    for name in before:
        destination=output/'source_snapshot'/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/name,destination)
    os.environ['ARENA_ESTIMATOR_OUTPUT_DIRECTORY']=str(output/'figures')
    start=time.perf_counter()
    record={'status':'RUNNING','scope':'analytic prefix only' if analytic_only else 'complete own-answer 16.17 learner',
        'completed_from_own_answers':True,'source_inputs':before,'execution_device':'cpu','analytic_only':analytic_only}
    failure=None
    try:
        client=NotebookClient(nb,timeout=1200,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
        client.execute()
        record['status']='PASS'
    except BaseException as error:
        failure=error;record['status']='FAIL';record['failure']=repr(error)
    finally:
        path=output/'completed_learner.ipynb';nbformat.write(nb,path)
        after={str(p.relative_to(ROOT)):digest(p) for p in source_paths()}
        record.update({'changed_sources':[name for name in set(before)|set(after) if before.get(name)!=after.get(name)],
            'elapsed_seconds':time.perf_counter()-start,'cells':len(nb.cells),
            'code_cells':sum(c.cell_type=='code' for c in nb.cells),
            'executed_code_cells':sum(c.cell_type=='code' and c.execution_count is not None for c in nb.cells),
            'embedded_figures':sum('image/png' in out.get('data',{}) for c in nb.cells if c.cell_type=='code' for out in c.outputs),
            'completed_learner_sha256':digest(path),'native_summary_exists':(output/'figures/native/summary.json').exists()})
        if record['changed_sources']:record['status']='FAIL'
        record['figures']={p.name:{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted((output/'figures').glob('*.png'))}
        if not analytic_only and record['status']=='PASS':
            if record['embedded_figures']!=10 or not record['native_summary_exists']:
                record['status']='FAIL';record['failure']='Complete native study and all ten figures required.'
        (output/'execution.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k not in {'source_inputs','figures'}},indent=2))
    if failure:raise failure
    if record['status']!='PASS':raise AssertionError('Execution did not pass.')
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--analytic-only',action='store_true')
    args=parser.parse_args();execute(args.output,args.analytic_only)

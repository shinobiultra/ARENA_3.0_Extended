"""Execute the current geometry draft using only its own solution dropdowns.

Foundation validation is deliberately separate from an accepted real-model
lesson; no command in this development executor publishes a release.
"""
from __future__ import annotations

import argparse
import hashlib
import gzip
import json
import os
from pathlib import Path
import sys
import time

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from chapter11_representation_geometry.exercises.part1_pca_svd_geometry_controls.build_notebooks import HERE,lesson_cells,complete_learner


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scope',choices=['foundations','full'],default='foundations')
    parser.add_argument('--output-directory',type=Path,required=True)
    args=parser.parse_args();run=args.output_directory.resolve();run.mkdir(parents=True,exist_ok=False)
    names=['lesson.md','build_notebooks.py','geometry_methods.py','geometry_methods_tests.py','execute_learner.py',
           'geometry_protocol.json','geometry_data.py','geometry_study.py','geometry_plotting.py']
    inputs={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in names}
    snapshot=run/'source_snapshot';snapshot.mkdir()
    for name in names:(snapshot/name).write_bytes((HERE/name).read_bytes())
    notebook=complete_learner({'cells':lesson_cells(),'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'nbformat':4,'nbformat_minor':5})
    notebook['cells'][1]['source'].append(f'\nassets_dir = Path({str(run / "figures")!r})\nassets_dir.mkdir(exist_ok=True)\n')
    notebook=nbformat.reads(json.dumps(notebook),as_version=4)
    started=time.time();state={'status':'RUNNING','scope':args.scope,'full_release_validation':False,
                              'source_inputs':inputs,'completed_from_own_answers':True}
    def save(): (run/'execution.json').write_text(json.dumps(state,indent=2)+'\n')
    save()
    def cell_finished(cell, cell_index, **kwargs):
        state['last_completed_cell'] = cell_index
        state['elapsed_seconds'] = time.time() - started
        nbformat.write(notebook, run/'completed_learner.ipynb')
        save()
    try:
        NotebookClient(notebook,timeout=3600,kernel_name='python3',on_cell_executed=cell_finished,resources={'metadata':{'path':str(ROOT)}}).execute(
            env={**os.environ,'ARENA_GEOMETRY_SCOPE':args.scope})
        state.update(status='PASS',code_cells=sum(c.cell_type=='code' for c in notebook.cells),figures=sorted(f.name for f in (run/'figures').glob('*.png')))
    except BaseException as error:
        state.update(status='FAIL',error=str(error));raise
    finally:
        state.update(elapsed_seconds=time.time()-started,changed_sources=[name for name,digest in inputs.items() if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=digest])
        if state['changed_sources']:state['status']='FAIL'
        nbformat.write(notebook,run/'completed_learner.ipynb');save()
    if state['status'] != 'PASS':
        raise RuntimeError('Learner validation did not pass; inspect execution.json.')
    print(json.dumps(state,indent=2))


if __name__=='__main__':main()

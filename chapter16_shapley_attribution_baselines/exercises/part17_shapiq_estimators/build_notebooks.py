"""Build an ARENA exercise/solution pair from prose and visible function sources."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import re
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
from chapter16_shapley_attribution_baselines.exercises.part17_shapiq_estimators.lesson_cells import CELLS

STEM='16.17_One_Query_Many_Interactions'
TOKEN=re.compile(r'<!-- (code|exercise|solution): ([\w,]+) -->')


def function_sources():
    source=(HERE/'estimator_methods.py').read_text()
    return {node.name:ast.get_source_segment(source,node) for node in ast.parse(source).body if isinstance(node,ast.FunctionDef)}


def _stub(source):
    node=ast.parse(source).body[0]
    doc=node.body[0] if isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str) else None
    node.body=([doc] if doc else [])+[ast.Raise(exc=ast.Call(func=ast.Name(id='NotImplementedError',ctx=ast.Load()),args=[],keywords=[]),cause=None)]
    return ast.unparse(ast.fix_missing_locations(node))


def _cell(kind,source,**arena):
    cell={'cell_type':kind,'id':hashlib.sha256((kind+'\n'+source).encode()).hexdigest()[:12],
          'metadata':{'arena':arena} if arena else {},'source':source}
    if kind=='code':cell.update({'execution_count':None,'outputs':[]})
    return cell


def build_notebooks():
    text=(HERE/'lesson.md').read_text();sources=function_sources()
    student=[];solved=[];position=0
    for token in TOKEN.finditer(text):
        prose=text[position:token.start()].strip()
        if prose:student.append(_cell('markdown',prose));solved.append(_cell('markdown',prose))
        kind,key=token.groups()
        if kind=='code':
            code=CELLS[key].strip();cell=_cell('code',code,calling_cell=key)
            student.append(copy.deepcopy(cell));solved.append(cell)
        elif kind=='exercise':
            names=key.split(',')
            tests='\n'.join(f'tests.test_{name}(fn={name})' for name in names)
            answer='\n\n'.join(sources[name] for name in names)+'\n\n'+tests
            stub='\n\n'.join(_stub(sources[name]) for name in names)+'\n\n'+tests
            # Shared id binds the answer-filled learner to its exact solved cell.
            identifier=hashlib.sha256(('exercise:'+key).encode()).hexdigest()[:12]
            sc=_cell('code',stub,exercise_functions=names);sc['id']=identifier
            ac=_cell('code',answer,exercise_functions=names);ac['id']=identifier
            student.append(sc);solved.append(ac)
        else:
            answer='\n\n'.join(sources[name] for name in key.split(','))
            student.append(_cell('markdown','<details>\n<summary>Solution</summary>\n\n```python\n'+answer+'\n```\n\n</details>',solution_for=key.split(',')))
        position=token.end()
    tail=text[position:].strip()
    if tail:student.append(_cell('markdown',tail));solved.append(_cell('markdown',tail))
    metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
              'language_info':{'name':'python','version':'3.14'},
              'arena':{'section':'16.17','status':'unpublished_development','original_chapters_modified':False}}
    make=lambda cells:{'nbformat':4,'nbformat_minor':5,'metadata':copy.deepcopy(metadata),'cells':cells}
    return make(student),make(solved)


def complete_from_own_answers(student):
    notebook=copy.deepcopy(student);answers={}
    for cell in notebook['cells']:
        names=cell.get('metadata',{}).get('arena',{}).get('solution_for')
        if names:
            code=re.search(r'```python\n(.*?)\n```',cell['source'],re.S).group(1)
            answers[tuple(names)]=code
    cells=[]
    for cell in notebook['cells']:
        info=cell.get('metadata',{}).get('arena',{})
        if info.get('solution_for'):continue
        if info.get('exercise_functions'):
            names=info['exercise_functions']
            cell['source']=answers[tuple(names)]+'\n\n'+'\n'.join(f'tests.test_{name}(fn={name})' for name in names)
        cells.append(cell)
    notebook['cells']=cells
    _,solution=build_notebooks()
    assert [(c['cell_type'],c['source']) for c in cells]==[(c['cell_type'],c['source']) for c in solution['cells']]
    return notebook


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--draft-output',type=Path,required=True)
    args=parser.parse_args();args.draft_output.mkdir(parents=True,exist_ok=True)
    student,solved=build_notebooks();complete_from_own_answers(student)
    for variant,notebook in [('exercises',student),('solutions',solved)]:
        path=args.draft_output/f'DRAFT_{STEM}_{variant}.ipynb'
        path.write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n')
        print(path)
    print('Eight exercise groups, nine visible methods; own-answer/source parity checked.')


if __name__=='__main__':main()

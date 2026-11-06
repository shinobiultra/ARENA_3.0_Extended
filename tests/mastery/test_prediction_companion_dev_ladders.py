"""DEV teaching ladders: fresh notebook-cwd startup, with protected targets trapped.

These checks execute only distinct exposed arithmetic. They do not grant target
execution authority or establish human learning, calibration, or unseen status.
"""
from __future__ import annotations

import ast
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
UNITS = ('circuit', 'shapley', 'sage', 'ssd', 'graph', 'sae')

# Run in an isolated interpreter whose initial path omits the repository. The
# import hook installs traps as each module first loads, before setup can call it.
DEV_HARNESS = r'''
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import sys
payload=json.loads(sys.stdin.read())
root=Path(payload['root'])
assert str(root) not in sys.path
calls=[]
def forbidden(*args, **kwargs):
    calls.append('protected target/API')
    raise AssertionError('protected target/API reached by setup or DEV cell')
protected={'run_experiment','freeze_inputs','commit_prediction','reveal_methods',
           'prepare','_prepare','_case','run','_tensors','_evaluate','freeze','_freeze',
           'commit','bind_session','binding_path','snapshot_sources','solutions'}

def guarded_namespace(factory,unit):
    def build(*args,**kwargs):
        ns=factory(*args,**kwargs)
        if unit=='circuit':
            ns['compile_induction']=forbidden
        guard_name={'shapley':'train_coalitions','sage':'restriction_coefficients',
                    'ssd':'ssd_recurrent','graph':'make_graph','sae':'affine'}.get(unit)
        if guard_name in ns:
            original=ns[guard_name]
            def checked(first,*args,**kwargs):
                if unit=='graph':
                    assert len(first['node_order'])==4, 'only distinct four-node DEV allowed'
                elif unit=='shapley':
                    assert tuple(first.shape)==(3,2), 'only three-row DEV allowed'
                elif unit=='sage':
                    assert tuple(first.shape)==(2,2), 'only two-feature DEV allowed'
                elif unit=='ssd':
                    assert tuple(first.shape)==(3,2), 'only T3 DEV allowed'
                elif unit=='sae':
                    assert tuple(first.shape)==(2,2), 'only two-row, two-dimensional DEV allowed'
                return original(first,*args,**kwargs)
            ns[guard_name]=checked
        return ns
    return build

class TrapLoader(importlib.abc.Loader):
    def __init__(self,original):self.original=original
    def create_module(self,spec):return self.original.create_module(spec)
    def exec_module(self,module):
        self.original.exec_module(module)
        for name in protected:
            if hasattr(module,name):setattr(module,name,forbidden)
        for name in ('method_namespace','practice_namespace','namespace'):
            if hasattr(module,name):setattr(module,name,guarded_namespace(getattr(module,name),payload['unit']))
class TrapFinder(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.startswith('mastery_hub.units.') and fullname.rsplit('.',1)[-1] in ('lesson','records','runner'):
            spec=importlib.machinery.PathFinder.find_spec(fullname,path)
            if spec and spec.loader:spec.loader=TrapLoader(spec.loader)
            return spec
sys.meta_path.insert(0,TrapFinder())
namespace={}
exec(compile(payload['setup'],'<normal notebook cwd setup>','exec'),namespace)
assert str(root) in sys.path, 'setup must explicitly bootstrap the checkout root'
namespace.update(payload['sources'])
if payload['unit']=='sae':namespace['method_source']=payload['reference']
import matplotlib.pyplot as plt
figures=[]
for cell in payload['dev_cells']:
    exec(compile(cell['source'],cell['id'],'exec'),namespace)
    for number in plt.get_fignums():
        figure=plt.figure(number)
        output=Path(payload['artifacts'])/(cell['id']+'-'+str(number)+'.png')
        figure.savefig(output,dpi=100)
        figures.append(str(output))
    plt.close('all')
assert len(figures)>=2, 'paired controls and observed transfer must be visible'
assert not calls
print('DEV-ONLY OK '+payload['unit']+' '+str(len(figures))+' figures; protected target calls=0')
'''


def dev_payload(unit, artifacts):
    lesson = importlib.import_module(f'mastery_hub.units.{unit}_prediction.lesson')
    cells = lesson.build_cells()
    reference = getattr(lesson, 'REFERENCE_METHODS', getattr(lesson, 'SOLUTION_SOURCE', ''))
    tree = ast.parse(reference)
    functions = {node.name: ast.get_source_segment(reference, node) for node in tree.body}
    groups = {
        'circuit': {'COMPILER': lesson.METHOD_NAMES[:3], 'SCORES': lesson.METHOD_NAMES[3:6],
                    'AUC': ('roc_auc',), 'ASSEMBLY': ('assemble_result',)},
        'graph': {'GRAPH': ('make_graph','activations','scalar_metric'),
                  'SCORES': ('exact_edge_effects','integrated_scores'),
                  'SELECTION': ('threshold_graph','top_path'),
                  'INTERVENTIONS': ('path_report','graph_controls'), 'ASSEMBLY': ('assemble_result',)},
    }
    if unit in groups:
        grouping = groups[unit]
    elif unit == 'sae':
        grouping = {name.upper(): (name,) for name in lesson.METHOD_NAMES}
    else:
        stages = ('training','utility','shapley','strata','assembly') if unit == 'shapley' else lesson.STAGES
        grouping = {stage.upper(): (name,) for stage,name in zip(stages,lesson.METHOD_NAMES,strict=True)}
    return {
        'root': str(ROOT), 'unit':unit, 'artifacts':str(artifacts),
        'setup':next(cell['source'] for cell in cells if cell['id'] in ('setup',unit+'-setup')),
        'sources':{name+'_SOURCE':'\n\n'.join(functions[method] for method in names) for name,names in grouping.items()},
        'reference':reference,
        'dev_cells':[cell for cell in cells if cell['cell_type']=='code' and
                     (cell['id'].startswith('dev-') or cell['id'].startswith(unit+'-dev-'))],
    }


@pytest.mark.parametrize('unit',UNITS)
@pytest.mark.parametrize('wrong_forecast',[False,True])
def test_setup_from_normal_notebook_directory_and_complete_dev_cycle(unit,wrong_forecast,tmp_path):
    payload=dev_payload(unit,tmp_path)
    if wrong_forecast:
        # Being wrong must produce evidence for reflection, never block observation.
        replacements={
            'dev_forecast':'[999.,999.]','dev_forecast_ratio':'999.',
            'dev_forecast_b':'999.','dev_forecast_first':'[999.,999.]',
            'dev_forecast_faithfulness':'999.','dev_forecast_rows':'[999.,999.]',
            'dev_transfer_forecast':{
                'circuit':'[999.,999.]','shapley':'999.',
                'sage':"{'coefficient':999.,'credit':999.}",
                'ssd':'t.tensor([999.,999.],dtype=t.float64)',
                'graph':'[999.,999.,999.]',
                'sae':"{'row_mse':[999.,999.],'contrast':999.}",
            }[unit],
        }
        class WrongGuesses(ast.NodeTransformer):
            def visit_Assign(self,node):
                if (len(node.targets)==1 and isinstance(node.targets[0],ast.Name)
                        and node.targets[0].id in replacements):
                    node.value=ast.parse(replacements[node.targets[0].id],mode='eval').body
                return node
        for cell in payload['dev_cells']:
            cell['source']=ast.unparse(ast.fix_missing_locations(WrongGuesses().visit(ast.parse(cell['source']))))
    env={key:value for key,value in os.environ.items() if key!='PYTHONPATH'}
    env.update(MPLBACKEND='Agg',MPLCONFIGDIR=str(tmp_path/'matplotlib'))
    result=subprocess.run([sys.executable,'-I','-c',DEV_HARNESS],input=json.dumps(payload),text=True,
                          cwd=ROOT/f'mastery_hub/units/{unit}_prediction',env=env,
                          capture_output=True,timeout=45)
    assert result.returncode==0, result.stdout+'\n'+result.stderr
    assert 'protected target calls=0' in result.stdout


@pytest.mark.parametrize('unit',UNITS)
def test_dev_experiment_reflection_and_observed_transfer_precede_exact_freeze(unit):
    lesson=importlib.import_module(f'mastery_hub.units.{unit}_prediction.lesson')
    cells=lesson.build_cells()
    freeze=next(i for i,c in enumerate(cells) if c['cell_type']=='code' and
                ('lesson.freeze_inputs(' in c['source'] or 'records.freeze(' in c['source']))
    dev=[(i,c) for i,c in enumerate(cells) if c['id'].startswith('dev-') or c['id'].startswith(unit+'-dev-')]
    assert dev and all(i<freeze for i,c in dev)
    text='\n'.join(c['source'] for i,c in dev)
    assert 'forecast' in text.lower() and 'observation' in text.lower()
    assert 'rules A out' in text or 'rule A out' in text or 'rules one' in text or 'rules A' in text or 'ruling A out' in text
    assert 'Worked DEV reflection' in text and 'dev-transfer' in ' '.join(c['id'] for i,c in dev)
    assert not any('run_experiment(' in c['source'] or 'records.freeze(' in c['source'] for i,c in dev)
    if unit in ('graph','sae'):
        assert 'target remains dormant' in text.lower() or 'target remains dormant' in '\n'.join(c['source'] for c in cells).lower()
    else:
        assert 'unscored self-rating' in text


def test_compiler_substep_checks_work_without_later_implementations():
    from mastery_hub.units.circuit_prediction import lesson
    reference=lesson.REFERENCE_METHODS
    helper=ast.get_source_segment(reference,ast.parse(reference).body[0])
    ns=lesson.practice_namespace(helper)
    assert lesson.practice_test(ns,'select').startswith('select:')
    compiler=next(c['source'] for c in lesson.build_cells() if c['id']=='circuit-compiler')
    assert compiler.index("'select'") < compiler.index("'induction'") < compiler.index("'compiler'")
    # A wrong selection predicate scaling must fail before composition/noise.
    broken=lesson.practice_namespace(helper.replace('model.cfg.d_head**.5','1.'))
    with pytest.raises(AssertionError):lesson.practice_test(broken,'select')

"""Real compact-model and learner-surface checks for the open capstone sprint."""
from pathlib import Path
import copy
import importlib.util
import sys
import numpy as np
import pytest
import torch as t

SECTION=Path(__file__).resolve().parents[1]/'chapter10_capstone_research_sprint/exercises/part1_capstone_research_sprint'
def load(name):
    spec=importlib.util.spec_from_file_location(name,SECTION/f'{name}.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module
sprint=load('sprint');checks=load('sprint_tests')
t.set_num_threads(1)

@pytest.mark.parametrize('name',['split_examples','budget_counts','mean_direction','rank_coordinates','patch_coordinates','patch_direction','select_on_development','paired_group_interval','assess_methodology'])
def test_student_contract(name):
    getattr(checks,f'test_{name}')(getattr(sprint,name))

def test_training_contract():
    checks.test_train_digit_model(sprint.train_digit_model,sprint.DEFAULT_PLAN)

def test_registration_is_bound_and_disjoint():
    x,y,ids,digest=sprint.load_digit_task([3,8]);parts=sprint.split_examples(y)
    registration=sprint.register_plan(sprint.DEFAULT_PLAN,parts,digest)
    sprint.verify_registration(registration)
    registration['payload']['plan']['primary_threshold']=.1
    with pytest.raises(ValueError,match='changed'):sprint.verify_registration(registration)
    parts['confirmation'][0]=parts['train'][0]
    with pytest.raises(ValueError):sprint.register_plan(sprint.DEFAULT_PLAN,parts,digest)

@pytest.mark.parametrize('model,digits',[('mlp',[3,8]),('linear',[1,7])])
def test_real_choice_and_negative_result(model,digits):
    plan=copy.deepcopy(sprint.DEFAULT_PLAN);plan.update(model=model,digits=digits)
    result=sprint.run_registered_sprint(plan)
    assert result['methodological_acceptance']
    assert result['primary_ruled_out_at_this_precision']
    assert result['selection']['method'] in plan['candidate_methods']
    assert len(np.unique(result['confirmation']['pair_ids']))>=40
    assert result['readout_rows']<=result['registration']['payload']['budget']['readout_rows']
    assert len(result['confirmation']['scores'])==88
    assert result['confirmation']['scores']['full_patch'].mean()>=.95
    assert result['confirmation']['scores']['no_patch'].mean()<=.05
    # Retain the unexpected control success; it is a scientific limitation, not grounds for pruning.
    if model=='mlp':
        assert result['confirmation']['scores']['shuffled_direction'].mean()==1.
        assert result['confirmation']['scores']['units_1'].mean()==0.

def test_no_events_does_not_make_an_exact_zero_population_interval():
    interval=sprint.bounded_pair_interval(np.zeros(88),np.repeat(np.arange(44),2))
    assert interval[0]==interval[1]==0.
    assert .2<interval[2]<.21

def test_source_notebook_answers_are_visible():
    import nbformat
    exercise=nbformat.read(SECTION/'10.1_Capstone_Research_Sprint_exercises.ipynb',as_version=4)
    solution=nbformat.read(SECTION/'10.1_Capstone_Research_Sprint_solutions.ipynb',as_version=4)
    text='\n'.join(c.source for c in exercise.cells)
    for marker in ['Choose your task','Register before fitting','A negative result that passes','shuffled-label','Hoeffding','Compact Write-up']:
        assert marker in text
    for name in ['split_examples','budget_counts','train_digit_model','mean_direction','rank_coordinates','patch_coordinates','patch_direction','select_on_development','paired_group_interval','assess_methodology']:
        assert f'def {name}(' in text
        assert f'test_{name}(' in text
    assert not any('import sprint\n' in c.source or 'from sprint import' in c.source for c in solution.cells if c.cell_type=='code')

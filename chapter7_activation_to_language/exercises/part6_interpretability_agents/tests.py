"""Independent CPU fixtures; student functions are passed explicitly."""
import numpy as np
import json
import itertools

def test_render_scene(fn):
    s={"color":"red","shape":"circle","background":"plain"};im=fn(s)
    assert im.shape==(7,7,3) and np.isfinite(im).all()
    assert im[...,0].sum()==5 and im[...,2].sum()==0
    s=dict(s,shape="square",background="striped");im=fn(s)
    assert np.count_nonzero(im[2:5,2:5,0])==9 and np.count_nonzero(im[:,0,0])==4
    try:fn(dict(s,color="green"))
    except ValueError:pass
    else:raise AssertionError("Unknown scene accepted")
    print("All tests in test_render_scene passed!")

def test_image_features(fn):
    im=np.zeros((7,7,3));im[3,3,0]=1
    np.testing.assert_array_equal(fn(im),[1,1,0])
    im[2,2,0]=1;im[0,0]=.5
    np.testing.assert_array_equal(fn(im),[1,0,1])
    im[3,3]=[0,0,1]
    assert fn(im)[0]==0
    print("All tests in test_image_features passed!")

def test_hypothesis_predictions(fn):
    F=np.array([[1.,0.,1.],[0.,1.,0.],[1.,1.,0.],[0.,0.,1.]])
    np.testing.assert_array_equal(fn("red-and-circle",F),[0,0,1,0])
    np.testing.assert_array_equal(fn("stripes",F),[1,0,0,1])
    try:fn("red OR circle",F)
    except ValueError:pass
    else:raise AssertionError("Unknown explanation silently mapped")
    print("All tests in test_hypothesis_predictions passed!")

def test_simulation_error(fn):
    assert fn(np.array([1.,0.,1.,0.]),np.array([1.,1.,0.,0.]))==.5
    assert fn(np.array([.5]),np.array([1.]))==.25
    try:fn(np.array([]),np.array([]))
    except ValueError:pass
    else:raise AssertionError("Empty evaluation must fail")
    print("All tests in test_simulation_error passed!")

def test_validate_action(fn):
    s={"color":"red","shape":"circle","background":"plain"};a={"tool":"activation","scene":s};r=fn(a);s["color"]="blue";assert r["scene"]["color"]=="red"
    bad=[dict(a,code="print(1)"),{"tool":"shell","scene":s},{"tool":"ablate","scene":s,"feature":True},{"tool":"steer","scene":s,"feature":0,"amount":float("nan")},{"tool":"steer","scene":s,"feature":0,"amount":2.}]
    for action in bad:
        try:fn(action)
        except ValueError:pass
        else:raise AssertionError("Invalid proposed action accepted")
    print("All tests in test_validate_action passed!")

def test_execute_tool(fn):
    s={"color":"red","shape":"circle","background":"striped"};target=lambda f:float(f@[2.,3.,5.])
    assert fn({"tool":"activation","scene":s},target)["activation"]==10
    assert fn({"tool":"ablate","scene":s,"feature":2},target)["activation"]==5
    assert fn({"tool":"steer","scene":s,"feature":0,"amount":-.5},target)["activation"]==9
    assert s["color"]=="red"
    print("All tests in test_execute_tool passed!")

def test_parse_backend_action(fn):
    text='{"tool":"activation","scene":{"color":"red","shape":"circle","background":"plain"}}'
    assert fn(text)["tool"]=="activation"
    for bad in ['print(1)','{"tool":"activation","tool":"shell"}',text[:-1]+',"code":"unsafe"}', '{"tool":"steer","scene":{},"amount":NaN,"feature":0}']:
        try:fn(bad)
        except ValueError:pass
        else:raise AssertionError("Invalid backend text accepted")
    print("All tests in test_parse_backend_action passed!")

def test_consistent_hypotheses(fn):
    s={"color":"red","shape":"circle","background":"plain"};c=["red","circle","red-and-circle","stripes"]
    assert fn(c,[])==c
    assert fn(c,[{"scene":s,"activation":1.}])==c[:3]
    assert fn(c,[{"scene":s,"activation":.5}])==[]
    print("All tests in test_consistent_hypotheses passed!")

def test_probe_disagreement(fn):
    scenes=[{"color":"red","shape":"square","background":"plain"},{"color":"red","shape":"circle","background":"plain"}]
    np.testing.assert_allclose(fn(["red","circle"],scenes),[.25,0])
    np.testing.assert_allclose(fn(["red"],scenes),[0,0])
    print("All tests in test_probe_disagreement passed!")

def test_choose_probe(fn):
    a={"color":"red","shape":"circle","background":"plain"};b=dict(a,shape="square");pool=[a,b]
    assert fn(["red","circle"],pool,[])==b
    assert fn(["red","circle"],pool,[{"scene":b,"activation":0}]) is None
    assert fn(["red"],pool,[]) is None
    chosen=fn(["red","circle"],pool,[]);chosen["color"]="blue";assert b["color"]=="red"
    print("All tests in test_choose_probe passed!")

def test_run_agent(fn):
    pool=[{"color":c,"shape":s,"background":"plain"} for c in ["red","blue"] for s in ["circle","square"]]
    calls=[]
    def target(f):calls.append(tuple(f));return float(f[0]*f[1])
    r=fn(target,["red","circle","red-and-circle"],pool,[],2)
    assert r["calls"]==len(calls)<=2 and len(r["rounds"])==r["calls"]
    assert r["remaining"]==["red-and-circle"] and r["reason"]=="identified"
    r=fn(target,["red","circle"],pool,[],0)
    assert r["calls"]==0 and r["explanation"] is None and r["reason"]=="budget_exhausted"
    r=fn(lambda f:.5,["red","circle"],pool,[],1)
    assert r["reason"]=="contradicted" and r["explanation"] is None
    print("All tests in test_run_agent passed!")

def test_one_shot_explanation(fn):
    s={"color":"red","shape":"circle","background":"plain"};history=[{"scene":s,"activation":1.}]
    assert fn(["red","circle","red-and-circle"],history)=="red"
    assert fn(["stripes"],history) is None
    print("All tests in test_one_shot_explanation passed!")

def test_intervention_effect(fn):
    s={"color":"blue","shape":"circle","background":"plain"};target=lambda f:float(f[0]+2*f[1]+3*f[2])
    assert fn(s,1,target)==2
    assert fn(s,0,target,amount=1)==-1
    assert fn(s,2,target)==0
    print("All tests in test_intervention_effect passed!")

def test_score_explanations(fn):
    scenes=[{"color":"red","shape":"square","background":"striped"},{"color":"blue","shape":"circle","background":"striped"}]
    r=fn({"wrong":"red","right":"red-and-circle","none":None},scenes,lambda f:float(f[0]*f[1]))
    assert r["observed"]==[0,0]
    assert r["methods"]["wrong"]["mse"]==.5 and r["methods"]["right"]["mse"]==0
    assert r["methods"]["none"]==dict(predicted=None,mse=None,abstained=True)
    print("All tests in test_score_explanations passed!")

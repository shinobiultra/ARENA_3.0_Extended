"""Independent CPU fixtures; student functions are passed explicitly."""
import numpy as np
import json
import itertools

def test_word_properties(fn):
    np.testing.assert_array_equal(fn(["Apple","ant","boat","cat"]),[[1,0,1,0],[1,0,0,0],[0,1,0,0],[0,0,0,0]])
    assert fn([]).shape==(0,4)
    try:fn([""])
    except ValueError:pass
    else:raise AssertionError("Empty words have no first letter")
    print("All tests in test_word_properties passed!")

def test_encode_relu(fn):
    x=np.array([[1.,0.,1.],[1.,0.,0.],[0.,1.,0.]])
    E=np.array([[1.,0.],[0.,1.],[-1.,0.]])
    np.testing.assert_allclose(fn(x,E,np.zeros(2)),[[0,0],[1,0],[0,1]])
    np.testing.assert_allclose(fn(x,E,np.array([-2.,-2.])),0)
    print("All tests in test_encode_relu passed!")

def test_decode_features(fn):
    D=np.array([[1.,0.,1.],[0.,2.,0.]])
    np.testing.assert_allclose(fn(np.eye(2),D,np.array([.1,.2,.3])),D+[.1,.2,.3])
    np.testing.assert_allclose(fn(np.zeros((2,2)),D,np.ones(3)),np.ones((2,3)))
    print("All tests in test_decode_features passed!")

def test_reconstruction_mse(fn):
    x=np.zeros((2,3));r=x.copy();r[0,2]=2
    assert fn(x,r)==4/6
    assert fn(x,x)==0
    try:fn(x,np.zeros((2,1)))
    except ValueError:pass
    else:raise AssertionError("No broadcasted error metrics")
    print("All tests in test_reconstruction_mse passed!")

def test_top_examples(fn):
    np.testing.assert_array_equal(fn(np.array([0.,2.,2.,1.]),3),[1,2,3])
    np.testing.assert_array_equal(fn(np.ones(2),10),[0,1])
    assert len(fn(np.ones(2),0))==0
    print("All tests in test_top_examples passed!")

def test_precision_recall(fn):
    assert fn(np.array([0.,1.,1.,1.,0.]),np.array([1,1,1,1,0]))==(1.,.75)
    assert fn(np.zeros(3),np.array([1,0,0]))==(0.,0.)
    assert fn(np.ones(2),np.zeros(2))==(0.,0.)
    assert fn(np.array([.5,1.]),np.array([1,1]),.5)==(1.,.5)
    print("All tests in test_precision_recall passed!")

def test_activation_overlap(fn):
    a=np.array([[1.,0.],[1.,0.],[0.,0.]])
    b=np.array([[1.,0.],[0.,0.],[1.,0.]])
    np.testing.assert_allclose(fn(a,b),[[1/3,0],[0,0]])
    np.testing.assert_allclose(fn(a,a),[[1,0],[0,0]])
    print("All tests in test_activation_overlap passed!")

def test_absorption_candidates(fn):
    concept=np.array([1,1,0,1],bool);primary=np.array([0.,1.,0.,0.]);alt=np.array([[1.],[0.],[1.],[0.]])
    np.testing.assert_array_equal(fn(concept,primary,alt),[True,False,False,False])
    assert not fn(concept,primary,np.empty((4,0))).any()
    print("All tests in test_absorption_candidates passed!")

def test_head_preactivation_contributions(fn):
    z=np.array([[[1.,2.],[3.,4.]],[[0.,1.],[2.,0.]]]);E=np.arange(12.).reshape(4,3)-4
    actual=fn(z,E)
    assert actual.shape==(2,2,3)
    np.testing.assert_allclose(actual.sum(1),z.reshape(2,4)@E)
    for n,h,f in itertools.product(range(2),range(2),range(3)):
        assert actual[n,h,f]==sum(z[n,h,d]*E[h*2+d,f] for d in range(2))
    print("All tests in test_head_preactivation_contributions passed!")

def test_head_decoder_logit_effects(fn):
    D=np.array([[1.,0.,2.,0.],[0.,1.,0.,3.]])
    O=np.arange(16.).reshape(2,2,4)/10;U=np.arange(12.).reshape(4,3)/7
    actual=fn(D,O,U)
    for f,h in itertools.product(range(2),range(2)):
        np.testing.assert_allclose(actual[f,h],D[f,h*2:h*2+2]@O[h]@U)
    np.testing.assert_allclose(actual.sum(1),D@O.reshape(4,4)@U)
    print("All tests in test_head_decoder_logit_effects passed!")

def test_ablate_features(fn):
    x=np.array([[3.,7.]]);codes=np.array([[1.,2.]]);D=np.eye(2)
    np.testing.assert_allclose(fn(x,codes,D,[0]),[[2,7]])
    np.testing.assert_allclose(fn(x,codes,D,[]),x)
    np.testing.assert_allclose(fn(x,codes,D,[0,1]),x-codes@D)
    try:fn(x,codes,D,[0,0])
    except ValueError:pass
    else:raise AssertionError("No double ablations")
    print("All tests in test_ablate_features passed!")

def test_correct_class_margin(fn):
    logits=np.array([[3.,1.],[-1.,4.],[0.,0.]])
    np.testing.assert_allclose(fn(logits,np.array([0,1,1])),[2,5,0])
    np.testing.assert_allclose(fn(logits+100,np.array([0,1,1])),[2,5,0])
    print("All tests in test_correct_class_margin passed!")

def test_dashboard_record(fn):
    r=fn(2,"A",["apple","ant","boat"],np.array([0.,2.,1.]),2)
    assert r["feature"]==2 and r["n_examples"]==3 and r["n_active"]==2
    assert r["examples"]==[{"text":"ant","activation":2.},{"text":"boat","activation":1.}]
    assert json.loads(json.dumps(r,allow_nan=False))==r
    print("All tests in test_dashboard_record passed!")

def test_validate_dashboard(fn):
    r={"schema":"arena.feature-dashboard.study.v1","feature":0,"label":"A","n_examples":1,"n_active":1,"examples":[{"text":"ant","activation":1.}]}
    assert fn(r) is True
    for value in [float("nan"),float("inf"),-1.,True]:
        bad=json.loads(json.dumps(r));bad["examples"][0]["activation"]=value
        try:fn(bad)
        except ValueError:pass
        else:raise AssertionError("Invalid activation accepted")
    bad=dict(r,n_active=2)
    try:fn(bad)
    except ValueError:pass
    else:raise AssertionError("Invalid denominator accepted")
    print("All tests in test_validate_dashboard passed!")

"""Independent analytic oracles for the CPU study edition."""
import numpy as np

def test_causal_attention(fn):
    q=np.zeros((3,2));v=np.array([[1.],[3.],[8.]])
    np.testing.assert_allclose(fn(q,q,v).ravel(),[1,2,4])
    w=v.copy();w[-1]=100
    np.testing.assert_allclose(fn(q,q,w)[:2],[[1],[2]])
    # Nonuniform dot products: independent scalar-loop oracle checks Q/K use,
    # the sqrt(d_k) scale, causal source limits and vector-valued outputs.
    import math
    queries=np.array([[1.5,-.4],[.2,1.3],[-.7,.8]])
    keys=np.array([[.5,-1.],[1.2,.3],[-.4,.9]])
    values=np.array([[1.,2.],[5.,-3.],[-2.,4.]])
    expected=np.empty_like(values)
    for destination in range(len(queries)):
        scores=[sum(float(queries[destination,d])*float(keys[source,d])
                    for d in range(queries.shape[1])) / math.sqrt(queries.shape[1])
                for source in range(destination+1)]
        unnormalised=[math.exp(score-max(scores)) for score in scores]
        total=sum(unnormalised)
        for feature in range(values.shape[1]):
            expected[destination,feature]=sum(
                mass*float(values[source,feature])
                for source,mass in enumerate(unnormalised))/total
    np.testing.assert_allclose(fn(queries,keys,values),expected,rtol=1e-12,atol=1e-12)
    print("All tests in test_causal_attention passed!")

def test_selective_scan(fn):
    u=np.array([[1.],[2.],[3.]])
    np.testing.assert_allclose(fn(u,np.ones_like(u),np.array([0.]),np.ones_like(u),np.ones_like(u)).ravel(),[1,3,6])
    y=fn(np.ones((2,1)),np.ones((2,1)),np.array([-np.log(2)]),np.ones((2,1)),np.ones((2,1)))
    np.testing.assert_allclose(y.ravel(),[.5/np.log(2),.75/np.log(2)])
    print("All tests in test_selective_scan passed!")

def test_token_nll(fn):
    x=np.array([[[0.,np.log(3)],[1000.,1000.]]])
    np.testing.assert_allclose(fn(x,np.array([[1,0]])),[[-np.log(.75),np.log(2)]])
    print("All tests in test_token_nll passed!")

def test_masked_token_loss(fn):
    assert fn(np.array([[1.,100.,3.]]),np.array([[True,False,True]]))==2
    try: fn(np.ones((1,2)),np.zeros((1,2),dtype=bool))
    except ValueError: pass
    else: raise AssertionError("Empty mask has no denominator")
    print("All tests in test_masked_token_loss passed!")

def test_latent_prediction_loss(fn):
    assert fn(np.array([[1.,3.]]),np.array([[0.,1.]]))==2.5
    assert fn(np.zeros((2,3)),np.zeros((2,3)))==0
    print("All tests in test_latent_prediction_loss passed!")

def test_center_features(fn):
    x=np.array([[1.,2.],[3.,6.]])
    np.testing.assert_allclose(fn(x),[[-1,-2],[1,2]]);np.testing.assert_array_equal(x,[[1,2],[3,6]])
    print("All tests in test_center_features passed!")

def test_linear_cka(fn):
    x=np.array([[1.,0.],[-1,0],[0,2],[0,-2]])
    q=np.array([[0.,-1],[1,0]])
    assert np.isclose(fn(x,x@q+7),1)
    h=np.eye(4)-np.ones((4,4))/4;k=h@(x@x.T)@h;l=h@((x@np.diag([3,1]))@(x@np.diag([3,1])).T)@h
    oracle=np.sum(k*l)/np.sqrt(np.sum(k*k)*np.sum(l*l))
    assert np.isclose(fn(x,x@np.diag([3,1])),oracle)
    assert np.isnan(fn(x,np.zeros((4,2))))
    print("All tests in test_linear_cka passed!")

def test_effective_rank(fn):
    x=np.array([[1.,0],[-1,0],[0,1],[0,-1]])
    assert np.isclose(fn(x),2)
    assert np.isclose(fn(x[:,:1]),1)
    assert fn(np.ones((3,2)))==0
    print("All tests in test_effective_rank passed!")

def test_fit_ridge_probe(fn):
    x=np.array([[-1.],[1.]])+5;y=np.array([-1.,1.])
    w,b=fn(x,y,ridge=2)
    np.testing.assert_allclose(w,[.5]);assert np.isclose(b,-2.5)
    # Augmented least-squares oracle penalizes weights but not bias.
    rng=np.random.default_rng(3);x=rng.normal(size=(9,3));y=rng.normal(size=9);r=.2
    aug=np.column_stack([x,np.ones(9)]);pen=np.column_stack([np.sqrt(r)*np.eye(3),np.zeros(3)])
    ref=np.linalg.lstsq(np.vstack([aug,pen]),np.r_[y,np.zeros(3)],rcond=None)[0]
    w,b=fn(x,y,r);np.testing.assert_allclose(np.r_[w,b],ref)
    print("All tests in test_fit_ridge_probe passed!")

def test_probe_accuracy(fn):
    assert fn(np.array([[-1.],[0.],[1.]]),np.array([-1,1,1]),np.ones(1),0)==2/3
    print("All tests in test_probe_accuracy passed!")

def test_cumulative_tokens(fn):
    np.testing.assert_array_equal(fn([2,3],4,8),[64,160])
    np.testing.assert_array_equal(fn([2,3],4,8,2),[128,320])
    print("All tests in test_cumulative_tokens passed!")

def test_align_examples(fn):
    np.testing.assert_array_equal(fn(["a","b"],["b","a"],np.array([[2],[1]])),[[1],[2]])
    try: fn(["a","b"],["a","a"],np.ones((2,1)))
    except ValueError: pass
    else: raise AssertionError("Duplicates must fail")
    print("All tests in test_align_examples passed!")

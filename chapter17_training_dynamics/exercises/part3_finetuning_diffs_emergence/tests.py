"""Independent analytic oracles for the CPU study edition."""
import numpy as np

def test_parameter_diff(fn):
    b={"w":np.array([1.,2.])}; a={"w":np.array([3.,1.])}
    np.testing.assert_array_equal(fn(b,a)["w"],[2,-1]);np.testing.assert_array_equal(b["w"],[1,2])
    try: fn(b,{"q":np.ones(2)})
    except ValueError: pass
    else: raise AssertionError("Missing parameter must fail")
    print("All tests in test_parameter_diff passed!")

def test_relative_change(fn):
    assert np.isclose(fn(np.array([3.,4.]),np.array([6.,8.])),1)
    assert np.isnan(fn(np.zeros(2),np.ones(2)))
    print("All tests in test_relative_change passed!")

def test_activation_difference(fn):
    x=np.array([[1.,2.],[3.,4.]])
    np.testing.assert_array_equal(fn(x,x+2,["a","b"],["a","b"]),np.full((2,2),2))
    try: fn(x,x,["a","b"],["b","a"])
    except ValueError: pass
    else: raise AssertionError("Order matters")
    print("All tests in test_activation_difference passed!")

def test_paired_stack(fn):
    np.testing.assert_array_equal(fn(np.array([[1,2]]),np.array([[3,4]])),[[1,2,3,4]])
    print("All tests in test_paired_stack passed!")

def test_soft_threshold(fn):
    np.testing.assert_allclose(fn(np.array([-2.,-.2,0,.2,2]),.5),[-1.5,0,0,0,1.5])
    print("All tests in test_soft_threshold passed!")

def test_infer_codes(fn):
    x=np.array([[2.,-1.,.01]])
    np.testing.assert_allclose(fn(x,np.eye(3),.1),[[1.9,-.9,0]],atol=1e-10)
    np.testing.assert_allclose(fn(x,2*np.eye(3),.1),[[.975,-.475,0]],atol=1e-10)
    np.testing.assert_array_equal(fn(x,np.zeros((2,3))),np.zeros((1,2)))
    # A correlated dictionary couples coordinates; diagonal one-shot shrinkage is wrong.
    rows=np.array([[2.,-1.],[-.4,.7]])
    dictionary=np.array([[1.,0.],[.5,1.]])
    penalty=.1
    # Largest eigenvalue of [[1,.5],[.5,1.25]], derived by its quadratic formula.
    eta=2/(2.25+np.sqrt(1.0625))
    def scalar_iteration(steps):
        result=[]
        for target in rows:
            codes=[0.,0.]
            for _ in range(steps):
                residual=[sum(codes[j]*dictionary[j,k] for j in range(2))-target[k] for k in range(2)]
                next_codes=[]
                for j in range(2):
                    proposal=codes[j]-eta*sum(residual[k]*dictionary[j,k] for k in range(2))
                    next_codes.append(max(proposal-eta*penalty,0)-max(-proposal-eta*penalty,0))
                codes=next_codes
            result.append(codes)
        return np.array(result)
    for steps in (0,1,2,9):
        np.testing.assert_allclose(fn(rows,dictionary,penalty,steps=steps),scalar_iteration(steps),rtol=1e-12,atol=1e-12)
    converged=fn(rows,dictionary,penalty,steps=1000)
    # Independent optimality conditions of the convex objective, including inactive entries.
    gradient=(converged@dictionary-rows)@dictionary.T
    active=np.abs(converged)>1e-8
    np.testing.assert_allclose(gradient[active]+penalty*np.sign(converged[active]),0,atol=1e-8)
    assert np.all(np.abs(gradient[~active])<=penalty+1e-8)
    np.testing.assert_allclose(converged[0],[2.325,-.85],atol=1e-8)
    np.testing.assert_array_equal(rows,[[2.,-1.],[-.4,.7]])
    np.testing.assert_array_equal(dictionary,[[1.,0.],[.5,1.]])
    print("All tests in test_infer_codes passed!")

def test_reconstruct_pair(fn):
    a,b=fn(np.array([[2.,3.]]),np.array([[1,0,4,0],[0,1,0,5]]))
    np.testing.assert_array_equal(a,[[2,3]]);np.testing.assert_array_equal(b,[[8,15]])
    print("All tests in test_reconstruct_pair passed!")

def test_signed_margins(fn):
    h=np.array([[1.,2.],[-1.,3.]])
    np.testing.assert_allclose(fn(h,np.array([2.,1.]),np.array([1,-1])),[4,-1])
    print("All tests in test_signed_margins passed!")

def test_capability_accuracy(fn):
    assert fn(np.array([-1,0,1,2]))==.5
    print("All tests in test_capability_accuracy passed!")

def test_threshold_crossing(fn):
    assert fn([0,.2,.5],[.5,.75,1])==.5
    assert fn([0,.2],[.5,.75]) is None
    print("All tests in test_threshold_crossing passed!")

def test_component_patch(fn):
    r=np.array([[1.,2.],[3.,4.]]);d=np.array([[9.,8.],[7.,6.]])
    np.testing.assert_allclose(fn(r,d,0),[11,11]);np.testing.assert_array_equal(r,[[1,2],[3,4]])
    print("All tests in test_component_patch passed!")

def test_patch_effects(fn):
    early=np.zeros((2,2));late=np.array([[2.,0],[-2.,0]]);y=np.array([1,-1])
    np.testing.assert_allclose(fn(early,late,y),[1,0])
    print("All tests in test_patch_effects passed!")

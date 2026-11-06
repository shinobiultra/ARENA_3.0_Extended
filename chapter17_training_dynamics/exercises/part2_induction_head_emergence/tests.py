"""Independent analytic oracles for the CPU study edition."""
import numpy as np

def test_repeat_tokens(fn):
    np.testing.assert_array_equal(fn(3,2),[0,1,2,0,1,2])
    try: fn(1,2)
    except ValueError: pass
    else: raise AssertionError("Reject degenerate block")
    print("All tests in test_repeat_tokens passed!")

def test_previous_attention(fn):
    a=fn(4)
    np.testing.assert_array_equal(a,[[0,0,0,0],[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    print("All tests in test_previous_attention passed!")

def test_prefix_keys(fn):
    np.testing.assert_array_equal(fn(np.array([2,0,1]),3),[[0,0,0],[0,0,1],[1,0,0]])
    np.testing.assert_array_equal(fn(np.array([2,0,1]),3,0),np.zeros((3,3)))
    print("All tests in test_prefix_keys passed!")

def test_causal_softmax(fn):
    a=fn(np.full((3,3),1000.))
    np.testing.assert_allclose(a,[[1,0,0],[.5,.5,0],[1/3,1/3,1/3]])
    assert np.isfinite(a).all()
    print("All tests in test_causal_softmax passed!")

def test_induction_pattern(fn):
    tok=np.array([0,1,2,0,1,2])
    a=fn(tok,3)
    assert a[3,1]>.99, "Second A should attend to first B"
    assert a[4,2]>.99, "Second B should attend to first C"
    np.testing.assert_allclose(fn(tok,3,0)[4,:5],np.full(5,.2))
    print("All tests in test_induction_pattern passed!")

def test_prefix_score(fn):
    a=np.zeros((6,6)); a[3,1]=.2;a[4,2]=.8
    assert np.isclose(fn(a,3),.5), "Use offset +1, not diagonal -period"
    print("All tests in test_prefix_score passed!")

def test_copy_logits(fn):
    a=np.array([[1,0],[.25,.75]])
    np.testing.assert_allclose(fn(a,np.array([1,0]),2,4),[[0,4],[3,1]])
    print("All tests in test_copy_logits passed!")

def test_next_token_loss(fn):
    x=np.array([[0.,np.log(3)],[np.log(4),0],[0,0]])
    np.testing.assert_allclose(fn(x,np.array([0,1,0])),[-np.log(.75),-np.log(.8)])
    print("All tests in test_next_token_loss passed!")

def test_icl_gain(fn):
    assert np.isclose(fn(np.array([3,5,999,1,2,999]),3),2.5)
    print("All tests in test_icl_gain passed!")

def test_ablate_previous(fn):
    x=np.ones((3,2)); y=fn(x)
    np.testing.assert_array_equal(y,np.zeros_like(x));np.testing.assert_array_equal(x,np.ones((3,2)))
    print("All tests in test_ablate_previous passed!")

def test_composition_score(fn):
    assert np.isclose(fn(np.eye(2),np.eye(2)),1/np.sqrt(2))
    assert fn(np.zeros((2,2)),np.eye(2))==0
    assert fn(np.diag([1.,0]),np.diag([0.,1]))==0
    print("All tests in test_composition_score passed!")

def test_first_crossing(fn):
    assert fn([10,20,30],[.1,.9,.8],.85)==20
    assert fn([10,20],[.1,.2],.9) is None
    print("All tests in test_first_crossing passed!")

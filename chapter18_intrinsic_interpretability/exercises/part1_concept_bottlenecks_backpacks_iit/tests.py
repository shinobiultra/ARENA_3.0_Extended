"""Independent analytic fixtures for the CPU intrinsic-intervention study edition."""
import numpy as np


def passed(name):
    print(f"All tests in {name} passed!")


def test_concept_probabilities(fn):
    x=np.array([[-1000.,0.,1000.],[-2.,1.,3.]])
    actual=fn(x)
    assert actual.shape==x.shape
    np.testing.assert_allclose(actual[0],[0,.5,1],atol=1e-14)
    np.testing.assert_allclose(actual[1],1/(1+np.exp(-x[1])))
    assert np.isfinite(actual).all()
    passed('test_concept_probabilities')


def test_concept_readout(fn):
    c=np.array([[[1.,0.],[0.,1.]],[[2.,-1.],[0.,0.]]])
    w=np.array([[2.,1.],[3.,-1.]])
    expected=np.empty((2,2,2))
    for b in range(2):
        for i in range(2):
            for o in range(2):
                expected[b,i,o]=1+sum(c[b,i,j]*w[j,o] for j in range(2))
    np.testing.assert_allclose(fn(c,w,np.ones(2)),expected)
    passed('test_concept_readout')


def test_correct_concept(fn):
    c=np.array([[.9,.1],[.2,.8]]);before=c.copy()
    np.testing.assert_allclose(fn(c,1,[1,0]),[[.9,1],[.2,0]])
    np.testing.assert_array_equal(c,before)
    np.testing.assert_allclose(fn(c,0,.5),[[.5,.1],[.5,.8]])
    passed('test_correct_concept')


def test_binary_accuracy(fn):
    assert np.isclose(fn([0,1,1,0],[0,0,1,0]),.75)
    assert fn(np.zeros((2,3)),np.ones((2,3)))==0
    passed('test_binary_accuracy')


def test_leakage_readout(fn):
    c=np.array([[0.,.9],[.2,.1],[.8,.9],[1.,.1]])
    np.testing.assert_array_equal(fn(c),[0,1,0,1])
    c[:,1]=1-c[:,1]
    np.testing.assert_array_equal(fn(c),[0,1,0,1])
    passed('test_leakage_readout')


def test_causal_weights(fn):
    score=np.zeros((2,3,3));score[0,1,0]=np.log(3);score[1,2]=[0,np.log(2),np.log(3)]
    w=fn(score)
    np.testing.assert_allclose(w[0,1],[.75,.25,0])
    np.testing.assert_allclose(w[1,2],[1/6,2/6,3/6])
    np.testing.assert_allclose(w.sum(-1),np.ones((2,3)))
    assert np.all(w[:,np.triu_indices(3,1)[0],np.triu_indices(3,1)[1]]==0)
    np.testing.assert_allclose(fn(score+1000),w)
    passed('test_causal_weights')


def test_backpack_hidden(fn):
    ids=np.array([0,1]);s=np.arange(12.).reshape(2,2,3)
    a=np.array([[[1,0],[.25,.75]],[[1,0],[.8,.2]]])
    expected=np.zeros((2,3))
    for i in range(2):
        for k in range(2):
            for j in range(2):
                expected[i]+=a[k,i,j]*s[ids[j],k]
    np.testing.assert_allclose(fn(ids,s,a),expected)
    passed('test_backpack_hidden')


def test_sense_readout(fn):
    s=np.arange(24.).reshape(3,2,4);u=np.arange(12.).reshape(4,3)
    expected=np.stack([np.stack([np.dot(v,u) for v in word]) for word in s])
    np.testing.assert_allclose(fn(s,u),expected)
    passed('test_sense_readout')


def test_edit_sense(fn):
    s=np.arange(12.).reshape(3,2,2);before=s.copy()
    out=fn(s,1,0,[-2,3]);expected=before.copy();expected[1,0]=[2,8]
    np.testing.assert_allclose(out,expected)
    np.testing.assert_array_equal(s,before)
    passed('test_edit_sense')


def test_predicted_edit_delta(fn):
    ids=np.array([0,1,0]);a=np.arange(18.).reshape(2,3,3)/18
    delta=np.array([2.,-1.]);u=np.array([[1.,2.],[3.,-1.]])
    expected=np.zeros((3,2))
    for i in range(3):
        for j in range(3):
            if ids[j]==0:
                expected[i]+=a[1,i,j]*np.dot(delta,u)
    np.testing.assert_allclose(fn(ids,a,0,1,delta,u),expected)
    np.testing.assert_allclose(fn(ids,a,8,1,delta,u),np.zeros((3,2)))
    passed('test_predicted_edit_delta')


def test_swap_subspace(fn):
    np.testing.assert_allclose(fn([1,4],[3,9],[1,0]),[3,4])
    d=np.array([1.,1.])/np.sqrt(2);b=np.array([[1.,2.],[4.,0.]]);s=np.array([[3.,5.],[-1.,2.]])
    out=fn(b,s,d);perp=np.array([-1.,1.])/np.sqrt(2)
    np.testing.assert_allclose(out@d,s@d,atol=1e-14)
    np.testing.assert_allclose(out@perp,b@perp,atol=1e-14)
    np.testing.assert_allclose(fn(b,b,d),b)
    try:
        fn(b,s,[2,0])
    except ValueError:
        pass
    else:
        raise AssertionError('nonunit direction accepted')
    passed('test_swap_subspace')


def test_causal_target(fn):
    b=np.array([[0,1],[1,0]]);s=np.array([[1,0],[0,1]])
    np.testing.assert_allclose(fn(b,s),[3,0])
    np.testing.assert_allclose(fn(b,s,4),[5,0])
    passed('test_causal_target')


def test_interchange_rows(fn):
    c=np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    q=np.array([[1.,-1.],[1.,1.]])/np.sqrt(2)
    h,y=fn(c,q,q[:,0])
    expected_c=np.array([[donor[0],base[1]] for base in c for donor in c])
    np.testing.assert_allclose(h,expected_c@q.T,atol=1e-14)
    np.testing.assert_allclose(y,[2*d[0]+b[1] for b in c for d in c])
    passed('test_interchange_rows')


def test_fit_readout(fn):
    np.testing.assert_allclose(fn([[1,0],[0,1]],[2,3]),[2,3])
    # Rank-deficient observational data has a minimum-norm, non-causal solution.
    np.testing.assert_allclose(fn([[0,0],[1,1]],[0,3]),[1.5,1.5])
    passed('test_fit_readout')


def test_interchange_agreement(fn):
    assert np.isclose(fn([0,1,2.1],[0,1,2]),2/3)
    assert fn([1e6+1],[1e6])==0, 'relative tolerance silently accepted a wrong value'
    passed('test_interchange_agreement')

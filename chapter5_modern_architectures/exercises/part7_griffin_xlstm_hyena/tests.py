"""Independent CPU fixtures; student functions are passed explicitly."""
import numpy as np
import json
import itertools

def test_rg_coefficients(fn):
    np.testing.assert_allclose(fn(np.array([0.]),np.array([0.])),[1/16])
    a=fn(np.array([-1000.,0.,1000.]),np.array([1000.]))
    assert np.isfinite(a).all() and np.all((a>=0)&(a<=1))
    assert a[0]==1 and a[-1]==0
    print("All tests in test_rg_coefficients passed!")

def test_rg_lru_step(fn):
    x=np.array([2.,-3.]);p=np.array([5.,7.]);g=np.array([1.,.5])
    np.testing.assert_allclose(fn(x,p,0,g),[2,-1.5])
    np.testing.assert_allclose(fn(x,p,1,g),p)
    np.testing.assert_allclose(fn(np.array([1.]),np.array([0.]),.8,1),[.6])
    try:fn(x,p,1.1,g)
    except ValueError:pass
    else:raise AssertionError("Invalid coefficient accepted")
    print("All tests in test_rg_lru_step passed!")

def test_rg_lru_scan(fn):
    x=np.array([[1.],[0.],[2.],[0.]])
    np.testing.assert_allclose(fn(x,.8,1),[[.6],[.48],[1.584],[1.2672]])
    np.testing.assert_allclose(fn(x,.8,1,[0,0,1,0]),[[.6],[.48],[1.2],[.96]])
    np.testing.assert_allclose(fn(x,.8,0),0)
    print("All tests in test_rg_lru_scan passed!")

def test_local_attention(fn):
    q=np.zeros((4,2));v=np.arange(4.)[:,None]
    np.testing.assert_allclose(fn(q,q,v,2),[[0],[.5],[1.5],[2.5]])
    np.testing.assert_allclose(fn(q,q,v,1),v)
    v[-1]=100;np.testing.assert_allclose(fn(q,q,v,2)[:3],[[0],[.5],[1.5]])
    print("All tests in test_local_attention passed!")

def test_griffin_recurrent_block(fn):
    x=np.array([[1.,-1.],[0.,2.]])
    np.testing.assert_allclose(fn(x,np.eye(2),np.eye(2),np.zeros((2,2)),np.ones((1,2)),.8,1),x)
    actual=fn(x,np.eye(2),np.zeros((2,2)),np.eye(2),np.ones((1,2)),.8,1)
    np.testing.assert_allclose(actual,x)
    # Identity memory and a one-tap filter isolate the gate/residual composition.
    z=x;gelu=.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+.044715*z**3)))
    np.testing.assert_allclose(fn(x,np.eye(2),np.eye(2),np.eye(2),np.ones((1,2)),0,1),x+gelu*x)
    print("All tests in test_griffin_recurrent_block passed!")

def test_slstm_step(fn):
    h,state=fn(np.array([2.]),np.array([1000.]),np.array([0.]),1.,(np.zeros(1),np.zeros(1),np.full(1,-np.inf)))
    np.testing.assert_allclose(h,[2]);assert np.isfinite(h).all()
    h,state=fn(np.array([-2.]),np.array([1000.+np.log(3)]),np.array([0.]),1.,state)
    np.testing.assert_allclose(h,[-1.])
    h2,_=fn(np.array([7.]),np.array([0.]),np.array([0.]),0.,(np.zeros(1),np.zeros(1),np.full(1,-np.inf)))
    np.testing.assert_allclose(h2,0)
    print("All tests in test_slstm_step passed!")

def test_slstm_scan(fn):
    v=np.array([[1.],[3.],[-2.]]);li=np.log(np.array([[1.],[2.],[4.]]));lf=np.log(.7)
    want=[]
    for t in range(3):
        w=np.array([np.exp(li[s,0])*.7**(t-s) for s in range(t+1)])
        want.append([sum(w*v[:t+1,0])/sum(w)])
    np.testing.assert_allclose(fn(v,li,lf),want)
    np.testing.assert_allclose(fn(v,li+900,lf),want,rtol=1e-12)
    print("All tests in test_slstm_scan passed!")

def test_mlstm_step(fn):
    C=np.zeros((1,1));n=np.zeros(1)
    h,(C,n)=fn(np.array([1.]),np.array([1.]),np.array([2.]),1.,1.,1.,(C,n));np.testing.assert_allclose(h,[2])
    h,(C,n)=fn(np.array([1.]),np.array([-1.]),np.array([5.]),1.,1.,1.,(C,n))
    # Signed normalizer cancels to zero; the floor is one, not a divide-by-zero.
    np.testing.assert_allclose(h,[-3]);np.testing.assert_allclose(n,[0])
    h,_=fn(np.array([1.,0.]),np.array([1.,0.]),np.array([2.]),1.,0.,1.,(np.zeros((1,2)),np.zeros(2)))
    np.testing.assert_allclose(h,[np.sqrt(2)])
    print("All tests in test_mlstm_step passed!")

def test_mlstm_scan(fn):
    q=np.array([[1.,0.],[0.,1.],[1.,1.]]);k=np.array([[1.,0.],[0.,1.],[1.,0.]])
    v=np.array([[1.,2.,3.],[4.,5.,6.],[-1.,1.,2.]])
    ig=np.array([1.,2.,1.]);fg=np.array([.5,.7,.8]);out,C,n=fn(q,k,v,ig,fg)
    for t in range(3):
        weights=np.array([ig[s]*np.prod(fg[s+1:t+1]) for s in range(t+1)])
        scalar=(k[:t+1]@q[t])/np.sqrt(2)*weights
        want=(scalar[:,None]*v[:t+1]).sum(0)/max(abs(scalar.sum()),1)
        np.testing.assert_allclose(out[t],want)
    assert C.shape==(3,2) and n.shape==(2,)
    print("All tests in test_mlstm_scan passed!")

def test_mlstm_parallel(fn):
    q=np.ones((3,1));k=np.array([[1.],[-1.],[1.]]);v=np.array([[2.],[5.],[3.]])
    np.testing.assert_allclose(fn(q,k,v,1.,1.),[[2],[-3],[0]])
    np.testing.assert_allclose(fn(q,k,v,1.,0.),[[2],[-5],[3]])
    assert np.isfinite(fn(q,k,v,0.,1.)).all()
    print("All tests in test_mlstm_parallel passed!")

def test_fft_causal_convolution(fn):
    x=np.arange(10.).reshape(5,2);h=np.array([[1.,2.],[-1.,.5],[.25,0.]])
    want=np.stack([np.convolve(x[:,c],h[:,c])[:len(x)] for c in range(2)],axis=1)
    np.testing.assert_allclose(fn(x,h),want,atol=1e-12)
    late=np.zeros((5,1));late[-1]=1
    np.testing.assert_allclose(fn(late,np.array([[1.],[2.],[3.]])),[[0],[0],[0],[0],[1]],atol=1e-12)
    print("All tests in test_fft_causal_convolution passed!")

def test_implicit_filter(fn):
    W1=np.array([[0.],[1.],[0.]]);W2=np.array([[2.,-1.]])
    actual=fn(3,W1,W2,np.array([0.,np.log(2)]))
    np.testing.assert_allclose(actual[0],[2*np.sin(1),-np.sin(1)])
    np.testing.assert_allclose(actual[1],[0,0],atol=1e-15)
    np.testing.assert_allclose(actual[2],[-2*np.sin(1),np.sin(1)/4])
    np.testing.assert_allclose(fn(5,np.zeros((3,2)),np.ones((2,1)),.1),0)
    print("All tests in test_implicit_filter passed!")

def test_hyena_order_two(fn):
    v=np.array([[1.],[2.],[0.],[3.]]);g1=np.array([[1.],[0.],[2.],[1.]]);g2=np.array([[0.],[1.],[1.],[2.]])
    h1=np.array([[1.],[.5]]);h2=np.array([[1.],[-1.]])
    inner=np.convolve(v[:,0],h1[:,0])[:4]*g1[:,0]
    want=(np.convolve(inner,h2[:,0])[:4]*g2[:,0])[:,None]
    np.testing.assert_allclose(fn(v,g1,g2,h1,h2),want,atol=1e-12)
    np.testing.assert_allclose(fn(v,np.zeros_like(g1),g2,h1,h2),0,atol=1e-12)
    print("All tests in test_hyena_order_two passed!")

def test_hybrid_stack(fn):
    x=np.array([[1.,2.],[3.,4.],[0.,1.]])
    Z=np.zeros((2,2));I=np.eye(2)
    np.testing.assert_allclose(fn(x,0.,0.,Z,Z,Z,2),x)
    # With no recurrent write and zero queries/keys, attention is a window mean.
    want=np.array([[1.,2.],[2.,3.],[1.5,2.5]])
    np.testing.assert_allclose(fn(x,0.,0.,Z,Z,I,2),x+want)
    # With a=.8, sqrt(1-a^2)=.6. These are literal recurrent states,
    # not results from rg_lru_scan or the supplied hybrid implementation.
    recurrent=np.array([[.6,1.2],[2.28,3.36],[1.824,3.288]])
    h=x+recurrent
    np.testing.assert_allclose(fn(x,.8,1.,Z,Z,Z,2),h)
    # Zero queries/keys produce a window mean of the UPDATED residual.
    # Nonidentity V additionally rejects using the original residual values.
    V=np.array([[1.,2.],[-.5,.25]])
    projected=h@V
    mean=np.stack([projected[0],(projected[0]+projected[1])/2,
                   (projected[1]+projected[2])/2])
    np.testing.assert_allclose(fn(x,.8,1.,Z,Z,V,2),h+mean)
    # Nonzero q/k ensure attention weights also see the updated residual.
    Q=np.array([[.2,-.1],[.3,.4]]);K=np.array([[.5,.1],[-.2,.6]])
    want=[]
    for t in range(len(x)):
        ids=range(max(0,t-1),t+1)
        logits=np.array([sum((h[t]@Q)[c]*(h[j]@K)[c] for c in range(2))/np.sqrt(2) for j in ids])
        weights=np.exp(logits-logits.max());weights/=weights.sum()
        want.append(h[t]+sum((weights[n]*projected[j] for n,j in enumerate(ids)),start=np.zeros(2)))
    np.testing.assert_allclose(fn(x,.8,1.,Q,K,V,2),want)
    print("All tests in test_hybrid_stack passed!")

def test_retained_fraction(fn):
    np.testing.assert_allclose(fn(np.array([.6,.48,-.3])),[1,.8,-.5])
    for x in [np.array([0.,1.]),np.array([]),np.array([1.,np.nan])]:
        try:fn(x)
        except ValueError:pass
        else:raise AssertionError("Invalid normalization accepted")
    print("All tests in test_retained_fraction passed!")

def test_trace_error(fn):
    assert fn(np.array([1.,0.,-1.]),np.ones(3))==5/3
    assert fn(np.ones(3),np.ones(3))==0
    try:fn(np.ones(2),np.ones(1))
    except ValueError:pass
    else:raise AssertionError("Metric broadcasting changes the cohort")
    print("All tests in test_trace_error passed!")

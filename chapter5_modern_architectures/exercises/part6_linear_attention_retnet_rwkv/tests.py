"""Independent CPU fixtures; student functions are passed explicitly."""
import numpy as np
import json
import itertools

def test_positive_features(fn):
    x=np.array([[-2.,0.,3.]])
    y=fn(x)
    np.testing.assert_allclose(y,[[np.exp(-2),1,4]])
    assert np.all(y>0) and np.array_equal(x,[[-2,0,3]])
    assert np.isfinite(fn(np.array([1000.]))).all()
    print("All tests in test_positive_features passed!")

def test_kernel_attention_parallel(fn):
    q=np.eye(3);v=np.array([[2.,1.],[3.,2.],[5.,4.]])
    np.testing.assert_allclose(fn(q,q,v),v)
    q=np.ones((3,1))
    np.testing.assert_allclose(fn(q,q,v),np.cumsum(v,axis=0)/np.arange(1,4)[:,None])
    np.testing.assert_allclose(fn(np.zeros((2,1)),np.ones((2,1)),v[:2]),0)
    for sign in [-1]:
        try:fn(q*sign,q,v)
        except ValueError:pass
        else:raise AssertionError("Negative features must be rejected")
    print("All tests in test_kernel_attention_parallel passed!")

def test_kernel_attention_recurrent(fn):
    q=np.array([[1.,0.],[1.,1.],[0.,1.]])
    k=np.array([[1.,0.],[0.,1.],[1.,1.]])
    v=np.array([[2.],[5.],[8.]])
    out,S,z=fn(q,k,v)
    want=[]
    for t in range(3):
        w=np.array([sum(q[t,f]*k[s,f] for f in range(2)) for s in range(t+1)])
        want.append([sum(w[s]*v[s,0] for s in range(t+1))/sum(w)])
    np.testing.assert_allclose(out,want)
    np.testing.assert_allclose(S,k.T@v);np.testing.assert_allclose(z,k.sum(0))
    v2=v.copy();v2[-1]=999
    np.testing.assert_allclose(fn(q,k,v2)[0][:-1],out[:-1])
    print("All tests in test_kernel_attention_recurrent passed!")

def test_causal_softmax_attention(fn):
    q=np.zeros((3,2));k=np.ones((3,2));v=np.array([[1.],[3.],[8.]])
    np.testing.assert_allclose(fn(q,k,v),[[1],[2],[4]])
    np.testing.assert_allclose(fn(np.ones((2,1))*1000,np.ones((2,1)),np.array([[2.],[4.]])),[[2],[3]])
    try:fn(q,k,v,0)
    except ValueError:pass
    else:raise AssertionError("Reject zero temperature")
    print("All tests in test_causal_softmax_attention passed!")

def test_retention_parallel(fn):
    q=np.ones((3,1));v=np.array([[1.],[2.],[4.]])
    np.testing.assert_allclose(fn(q,q,v,.5),[[1],[2.5],[5.25]])
    np.testing.assert_allclose(fn(q,q,v,0),v)
    np.testing.assert_allclose(fn(q,q,v,1),[[1],[3],[7]])
    try:fn(q,q,v,1.1)
    except ValueError:pass
    else:raise AssertionError("Invalid decay")
    print("All tests in test_retention_parallel passed!")

def test_retention_recurrent(fn):
    q=np.array([[1.,-1.],[2.,1.]]);k=np.array([[1.,2.],[-1.,1.]]);v=np.array([[3.],[5.]])
    init=np.array([[2.],[1.]]);old=init.copy();out,S=fn(q,k,v,.4,init)
    want=[]
    for t in range(2):
        state=.4**(t+1)*init+sum((.4**(t-s)*np.outer(k[s],v[s]) for s in range(t+1)),start=np.zeros_like(init))
        want.append(q[t]@state)
    np.testing.assert_allclose(out,want);np.testing.assert_allclose(init,old);np.testing.assert_allclose(S,state)
    print("All tests in test_retention_recurrent passed!")

def test_retention_chunkwise(fn):
    q=np.array([[1.,0.],[0.,1.],[1.,1.],[-1.,2.],[2.,1.]])
    k=q[::-1].copy();v=np.arange(10.).reshape(5,2)
    want=np.array([sum((.7**(t-s)*(q[t]@k[s])*v[s] for s in range(t+1)),start=np.zeros(2)) for t in range(5)])
    state=sum((.7**(4-s)*np.outer(k[s],v[s]) for s in range(5)),start=np.zeros((2,2)))
    for chunk in [1,2,3,7]:
        actual,last=fn(q,k,v,.7,chunk)
        np.testing.assert_allclose(actual,want);np.testing.assert_allclose(last,state)
    print("All tests in test_retention_chunkwise passed!")

def test_decay_horizon(fn):
    out=fn(np.array([0.,.5,1.]),.125)
    np.testing.assert_allclose(out[:2],[1,3]);assert np.isinf(out[2])
    for g,h in zip([.2,.7,.95],fn(np.array([.2,.7,.95]))):
        assert g**h<=.01+1e-14 and g**(h-1)>.01-1e-14
    print("All tests in test_decay_horizon passed!")

def test_time_mix(fn):
    x=np.array([[1.,2.],[3.,4.],[5.,6.]])
    np.testing.assert_allclose(fn(x,1),x)
    np.testing.assert_allclose(fn(x,0),[[0,0],[1,2],[3,4]])
    np.testing.assert_allclose(fn(x,np.array([.25,.75]))[0],[.25,1.5])
    print("All tests in test_time_mix passed!")

def test_rwkv_wkv(fn):
    k=np.array([[.2,-.3],[.5,.1],[-.2,.8]]);v=np.array([[1.,2.],[3.,4.],[7.,8.]])
    d=np.array([.4,.8]);u=np.array([.1,-.2]);want=[]
    for t in range(3):
        weights=np.array([np.exp(k[s])*d**(t-1-s) for s in range(t)]+[np.exp(u+k[t])])
        want.append((weights*v[:t+1]).sum(0)/weights.sum(0))
    np.testing.assert_allclose(fn(k,v,d,u),want)
    np.testing.assert_allclose(fn(k+1000,v,d,u),want,rtol=1e-12)
    assert np.isfinite(fn(k+1000,v,d,u)).all()
    print("All tests in test_rwkv_wkv passed!")

def test_rwkv_channel_mix(fn):
    x=np.array([[-2.,3.],[1.,-1.]])
    np.testing.assert_allclose(fn(x,x,np.eye(2),np.eye(2),np.zeros((2,2))),[[0,4.5],[.5,0]])
    np.testing.assert_allclose(fn(x,x,np.eye(2),np.zeros((2,2)),np.eye(2)),0)
    print("All tests in test_rwkv_channel_mix passed!")

def test_rwkv_block(fn):
    x=np.array([[1.,-1.],[2.,0.],[0.,2.]])
    w={key:.5 for key in ["mix_k","mix_v","mix_r","mix_ck","mix_cr","decay"]};w["bonus"]=0.
    for key in ["K","V","R","O","CK","CV","CR"]:w[key]=np.eye(2)
    w["O"]=np.zeros((2,2));w["CV"]=np.zeros((2,2))
    np.testing.assert_allclose(fn(x,w),x)
    w["CV"]=np.eye(2);w["CR"]=np.zeros((2,2))
    z=(x-x.mean(-1,keepdims=True))/np.sqrt(x.var(-1,keepdims=True)+1e-5)
    np.testing.assert_allclose(fn(x,w),x+.5*np.maximum((z+np.vstack([np.zeros(2),z[:-1]]))*.5,0)**2)
    # Isolate a nonzero time branch. Enumerate its causal weights directly,
    # independently of time_mix/rwkv_wkv or any reference implementation.
    w["CV"]=np.zeros((2,2))
    w.update(mix_k=np.array([.2,.8]),mix_v=np.array([.7,.3]),
             mix_r=np.array([.4,.6]),decay=np.array([.4,.8]),bonus=np.array([.1,-.2]))
    w["K"]=np.array([[.5,-.3],[.2,.7]])
    w["V"]=np.array([[1.,.4],[-.2,.8]])
    w["R"]=np.array([[.6,-.5],[.3,.2]])
    w["O"]=np.array([[.7,.2],[-.1,.9]])
    shifted=np.vstack([np.zeros(2),z[:-1]])
    keys=(w["mix_k"]*z+(1-w["mix_k"])*shifted)@w["K"]
    values=(w["mix_v"]*z+(1-w["mix_v"])*shifted)@w["V"]
    gates=1/(1+np.exp(-((w["mix_r"]*z+(1-w["mix_r"])*shifted)@w["R"])))
    memory=np.empty_like(x)
    for t in range(len(x)):
        for c in range(x.shape[1]):
            history=[np.exp(keys[j,c])*w["decay"][c]**(t-1-j) for j in range(t)]
            weights=history+[np.exp(w["bonus"][c]+keys[t,c])]
            memory[t,c]=sum(weights[j]*values[j,c] for j in range(t+1))/sum(weights)
    np.testing.assert_allclose(fn(x,w),x+(gates*memory)@w["O"],rtol=1e-12,atol=1e-12)
    print("All tests in test_rwkv_block passed!")

def test_last_write_episodes(fn):
    episodes=fn(3)
    assert len(episodes)==6
    assert {(int(e["values"][0].argmax()),e["answer"]) for e in episodes}==set(itertools.permutations(range(3),2))
    for e in episodes:
        assert e["values"][-1,e["answer"]]==1 and e["query"].shape==(2,)
        np.testing.assert_array_equal(e["keys"][[0,2]],[[1,0],[1,0]])
    print("All tests in test_last_write_episodes passed!")

def test_shuffle_values(fn):
    v=np.arange(12.).reshape(4,3);before=v.copy();a=fn(v,4);b=fn(v,4)
    np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(v,before)
    assert sorted(map(tuple,a))==sorted(map(tuple,v))
    assert not np.shares_memory(a,v)
    print("All tests in test_shuffle_values passed!")

def test_recall_accuracy(fn):
    assert fn(np.array([[1.,1.],[0.,2.]]),np.array([0,1]))==1
    assert fn(np.array([[1.,1.],[0.,2.]]),np.array([1,1]))==.5
    try:fn(np.empty((0,2)),np.array([]))
    except ValueError:pass
    else:raise AssertionError("Empty cohort has no accuracy")
    print("All tests in test_recall_accuracy passed!")

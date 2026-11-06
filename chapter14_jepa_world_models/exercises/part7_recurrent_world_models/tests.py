import numpy as np
import torch as t
import torch.nn.functional as F

def passed(name):
    print(f"All tests in {name} passed!")


def test_chain_step(fn):
    n,r=fn(np.array([0,0,3,4]),np.array([0,2,2,2]));np.testing.assert_array_equal(n,[0,1,4,4]);np.testing.assert_array_equal(r,[0,0,1,1])
    passed("test_chain_step")


def test_straight_through_state(fn):
    logits=t.tensor([[0.,2.,1.]],requires_grad=True);z=fn(logits,sample=False)
    t.testing.assert_close(z,t.tensor([[0.,1.,0.]]));(z*t.tensor([[1.,2.,4.]])).sum().backward()
    p=t.softmax(logits.detach(),-1);expected=p*(t.tensor([[1.,2.,4.]])-(p*t.tensor([[1.,2.,4.]])).sum())
    t.testing.assert_close(logits.grad,expected)
    passed("test_straight_through_state")


def test_categorical_kl(fn):
    q=t.tensor([[np.log(.75),np.log(.25)]],dtype=t.float64);p=t.zeros_like(q)
    expected=.75*np.log(1.5)+.25*np.log(.5)
    t.testing.assert_close(fn(q,p),t.tensor([expected],dtype=t.float64));t.testing.assert_close(fn(q,q),t.zeros(1,dtype=t.float64))
    passed("test_categorical_kl")


def test_rssm_transition(fn):
    class Memory:
     def __call__(self,x,h):return h+t.stack([x[:,0]+x[:,-1],x[:,1]+x[:,-2]],-1)
    model={"memory":Memory(),"prior":lambda h:t.cat([h,t.zeros(len(h),1)],-1),"posterior":lambda x:x[:,-3:]}
    h=t.zeros(1,2);z=t.tensor([[0.,1.,0.]])
    h1,z1,p,q=fn(model,h,z,t.tensor([2]),t.tensor([[0.,0.,1.]]),sample=False)
    t.testing.assert_close(h1,t.tensor([[1.,1.]]));t.testing.assert_close(z1,t.tensor([[0.,0.,1.]]));assert q is not None
    _,zi,_,qi=fn(model,h,z,t.tensor([2]),None,sample=False);assert qi is None;t.testing.assert_close(zi,t.tensor([[1.,0.,0.]]))
    passed("test_rssm_transition")


def test_balanced_kl(fn):
    q=t.tensor([[2.,0.]],requires_grad=True);p=t.tensor([[0.,1.]],requires_grad=True)
    loss=fn(q,p,free_nats=0.);loss.backward()
    qp=q.detach().softmax(-1);pp=p.detach().softmax(-1)
    logratio=qp.log()-pp.log();kl=(qp*logratio).sum(-1,keepdim=True)
    t.testing.assert_close(p.grad,.5*(pp-qp))
    t.testing.assert_close(q.grad,.1*qp*(logratio-kl))
    q2=t.zeros(2,3,requires_grad=True);p2=t.zeros(2,3,requires_grad=True);l=fn(q2,p2,free_nats=.2);assert abs(l.item()-.12)<1e-6
    l.backward();assert q2.grad.abs().sum()==0 and p2.grad.abs().sum()==0
    passed("test_balanced_kl")


def test_lambda_returns(fn):
    np.testing.assert_allclose(fn(np.array([1.,2.]),np.array([7.,3.,4.]),gamma=.5,lam=0),[2.5,4.])
    np.testing.assert_allclose(fn(np.array([1.,2.]),np.array([7.,3.,4.]),gamma=.5,lam=1),[3.,4.])
    passed("test_lambda_returns")


def test_simnorm(fn):
    x=t.tensor([[0.,0.,1.,1.]])
    y=fn(x,group_size=2);t.testing.assert_close(y,t.full_like(x,.5))
    t.testing.assert_close(fn(x+100,group_size=2),y)
    try:fn(x,group_size=3)
    except ValueError:pass
    else:raise AssertionError("unequal groups must fail")
    passed("test_simnorm")


def test_td_world_loss(fn):
    z=t.tensor([[1.,0.]],requires_grad=True);target=t.tensor([[0.,1.]],requires_grad=True)
    r=t.tensor([0.],requires_grad=True);q=t.tensor([1.],requires_grad=True);n=t.tensor([2.],requires_grad=True)
    l=fn(z,target,r,t.tensor([1.]),q,n,gamma=.5);assert abs(l.item()-3)<1e-7
    l.backward();assert target.grad is None and n.grad is None;assert z.grad is not None and q.grad is not None
    passed("test_td_world_loss")


def test_plan_latent_actions(fn):
    # A controlled model gives state-independent reward for action2.
    class Reward:
     def __call__(self,x):return x[:,-1:]
    class Dynamics:
     def __call__(self,x):return x[:,:5]
    class Q:
     def __call__(self,x):return t.zeros(len(x),1)
    m={"dynamics":Dynamics(),"reward":Reward(),"q":Q()}
    a,v,tr=fn(m,0,horizon=2,gamma=1.);np.testing.assert_array_equal(a,[2,2]);assert abs(v-2)<1e-6;assert tr.shape==(3,)
    passed("test_plan_latent_actions")


def test_quantize_observation(fn):
    c=t.tensor([[0.,0.],[2.,0.],[0.,2.]])
    z,idx=fn(t.tensor([[1.8,.1],[0.,1.7],[1.,0.]]),c)
    t.testing.assert_close(idx,t.tensor([1,2,0]));t.testing.assert_close(z,c[idx])
    passed("test_quantize_observation")


def test_vq_loss(fn):
    e=t.tensor([[1.,0.]],requires_grad=True);q=t.tensor([[0.,0.]],requires_grad=True)
    l=fn(e,q,beta=.25);assert abs(l.item()-.625)<1e-7;l.backward()
    t.testing.assert_close(e.grad,t.tensor([[.25,0.]]));t.testing.assert_close(q.grad,t.tensor([[-1.,0.]]))
    passed("test_vq_loss")


def test_causal_attention(fn):
    q=t.zeros(1,3,2);v=t.tensor([[[1.,0.],[3.,2.],[100.,100.]]])
    o=fn(q,q,v);t.testing.assert_close(o[0,0],v[0,0]);t.testing.assert_close(o[0,1],t.tensor([2.,1.]))
    v2=v.clone();v2[:,2]=999;t.testing.assert_close(fn(q,q,v2)[:,:2],o[:,:2])
    passed("test_causal_attention")


def test_interleave_trace(fn):
    s=fn(t.tensor([[0,1,2],[4,3,3]]),t.tensor([[2,2],[0,1]]))
    t.testing.assert_close(s,t.tensor([[0,7,1,7,2],[4,5,3,6,3]]))
    passed("test_interleave_trace")


def test_iris_logits(fn):
    class Zero:
     def __call__(self,x):return t.zeros_like(x)
    model={"embedding":lambda tok:F.one_hot(tok,num_classes=5).float(),"position":lambda pos:t.zeros(len(pos),5),"query":Zero(),"key":Zero(),"value":Zero(),"norm":lambda x:x,"mlp":Zero(),"readout":lambda x:x}
    z=fn(model,t.tensor([[0,1,2]]));t.testing.assert_close(z,F.one_hot(t.tensor([[0,1,2]]),5).float())

    # A second fixture exercises both residual branches. Hidden width 3, Q/K
    # width 2 and output width 5 make accidental dimension assumptions visible.
    dtype = t.float64
    embedding = t.tensor([[.2, -.4, .1], [.7, .3, -.2], [-.3, .8, .4],
                          [.6, -.5, .9], [-.8, .2, -.6]], dtype=dtype)
    position = t.tensor([[0., .1, -.2], [.2, -.1, .3], [-.1, .3, .2],
                         [.4, -.2, -.1]], dtype=dtype)
    query = t.tensor([[.3, -.2], [.5, .4], [-.1, .6]], dtype=dtype)
    key = t.tensor([[.2, .7], [-.4, .1], [.6, -.3]], dtype=dtype)
    value = t.tensor([[.5, -.2, .3], [.1, .6, -.4], [-.3, .2, .7]], dtype=dtype)
    up = t.tensor([[.2, -.3, .5, .1], [.4, .1, -.2, .6], [-.5, .7, .3, -.4]], dtype=dtype)
    down = t.tensor([[.3, -.2, .1], [.5, .4, -.3], [-.2, .6, .7], [.4, -.1, .2]], dtype=dtype)
    readout = t.tensor([[.2, -.4, .6, .1, -.3], [.5, .3, -.2, .7, .4],
                        [-.1, .8, .4, -.5, .6]], dtype=dtype)

    def normalize(x):
        return F.layer_norm(x, (3,), eps=1e-5)

    model = {"embedding": lambda ids: embedding[ids], "position": lambda ids: position[ids],
             "query": lambda x: x @ query, "key": lambda x: x @ key,
             "value": lambda x: x @ value, "norm": normalize,
             "mlp": lambda x: t.tanh(x @ up) @ down, "readout": lambda x: x @ readout}
    tokens = t.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
    x = embedding[tokens] + position[None]
    # Scalar prefix loops are an independent oracle for causal attention;
    # neither the student's attention helper nor the reference is imported.
    attention = t.empty_like(x)
    for batch in range(len(tokens)):
        for target in range(tokens.shape[1]):
            scores = t.stack([(x[batch, target] @ query) @ (x[batch, source] @ key)
                              / (2.0 ** .5) for source in range(target + 1)])
            weights = scores.softmax(0)
            attention[batch, target] = sum(weights[source] * (x[batch, source] @ value)
                                           for source in range(target + 1))
    after_attention = x + attention
    # Explicit population mean/variance also checks the model's normalization call.
    centered = after_attention - after_attention.mean(-1, keepdim=True)
    normalized = centered / (centered.square().mean(-1, keepdim=True) + 1e-5).sqrt()
    expected = (after_attention + t.tanh(normalized @ up) @ down) @ readout
    actual = fn(model, tokens)
    t.testing.assert_close(actual, expected)
    assert not t.allclose(expected, x @ readout), "Fixture must expose omitted residual branches."
    assert not t.allclose(expected, after_attention @ readout), "Fixture must expose an omitted MLP."
    changed_future = tokens.clone()
    changed_future[:, 2:] = t.tensor([[4, 0], [4, 3]])
    changed = fn(model, changed_future)
    t.testing.assert_close(changed[:, :2], actual[:, :2])
    assert not t.allclose(changed[:, 2:], actual[:, 2:]), "Changed tokens must affect their own logits."
    passed("test_iris_logits")


def test_reward_imagination_gap(fn):
    assert abs(fn(np.array([2.,3.]),np.array([1.,1.]),gamma=.5)-2.)<1e-10
    assert fn(np.array([0.]),np.array([1.]))==-1.
    passed("test_reward_imagination_gap")


def test_linear_room_probe(fn):
    features=np.tile(np.eye(5),(2,1));labels=np.tile(np.arange(5),2);fit=np.arange(10)<5
    p,a=fn(features,labels,fit);np.testing.assert_array_equal(p,labels);assert a==1
    badlabels=labels.copy();badlabels[:5]=np.roll(badlabels[:5],1)
    _,a=fn(features,badlabels,fit);assert a==0
    passed("test_linear_room_probe")

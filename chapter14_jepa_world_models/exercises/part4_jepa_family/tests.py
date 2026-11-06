import numpy as np
import torch as t

def passed(name):
    print(f"All tests in {name} passed!")


def test_masked_mean(fn):
    x=t.tensor([[[1.,3.],[99.,99.],[5.,7.]]],requires_grad=True)
    y=fn(x,t.tensor([[True,False,True]]));t.testing.assert_close(y,t.tensor([[3.,5.]]))
    y.sum().backward();t.testing.assert_close(x.grad[0,1],t.zeros(2))
    try: fn(x,t.zeros(1,3,dtype=t.bool))
    except ValueError: pass
    else: raise AssertionError("all-hidden context must fail")
    passed("test_masked_mean")


def test_jepa_mse(fn):
    p=t.tensor([[1.,3.]],requires_grad=True);q=t.tensor([[2.,1.]],requires_grad=True)
    loss=fn(p,q);assert abs(loss.item()-2.5)<1e-7
    loss.backward();t.testing.assert_close(p.grad,t.tensor([[-1.,2.]]));assert q.grad is None
    passed("test_jepa_mse")


def test_ema_update(fn):
    teacher=[t.tensor([2.,4.],requires_grad=True)];student=[t.tensor([6.,0.],requires_grad=True)]
    fn(teacher,student,.75);t.testing.assert_close(teacher[0],t.tensor([3.,3.]));assert teacher[0].grad_fn is None
    fn(teacher,student,0);t.testing.assert_close(teacher[0],student[0])
    passed("test_ema_update")


def test_collapse_statistics(fn):
    v,r=fn(np.ones((8,3))*7);np.testing.assert_allclose(v,0);assert r==0
    x=np.array([[1,0],[-1,0],[0,1],[0,-1]],dtype=float);v,r=fn(x)
    np.testing.assert_allclose(v,[.5,.5]);assert abs(r-2)<1e-10
    passed("test_collapse_statistics")


def test_time_views(fn):
    c,y=fn(np.array([2.,-1.]),np.array([.5,0.]))
    np.testing.assert_allclose(c,[[2,2.5,3,3.5],[-1,-1,-1,-1]])
    np.testing.assert_allclose(y,[[4,4.5,5,5.5],[-1,-1,-1,-1]])
    passed("test_time_views")


def test_audio_views(fn):
    c,y=fn(np.array([1.,.5]),np.array([.25,1.]))
    np.testing.assert_allclose(c,[[1,.25,0,0],[.5,1,0,0]],atol=1e-12)
    np.testing.assert_allclose(y,c,atol=1e-12)
    passed("test_audio_views")


def test_motion_content_views(fn):
    c,y=fn(np.array([2.]),np.array([-.5]))
    np.testing.assert_allclose(c,[[2,1.5,2,1]]);np.testing.assert_allclose(y,[[1,.5,2,1]])
    passed("test_motion_content_views")


def test_text_views(fn):
    c,y=fn(np.array([2,1]),np.array([3,4]))
    np.testing.assert_allclose(c,[[2,3,5,1],[1,4,5,1]])
    np.testing.assert_allclose(y,[[3,2,5,1],[4,1,5,1]])
    passed("test_text_views")


def test_visual_language_views(fn):
    c,y=fn(np.array([.8]),np.array([.2]));np.testing.assert_allclose(c,[[.4,.1,1,0]]);np.testing.assert_allclose(y,[[.8,.2,.6,1]])
    passed("test_visual_language_views")


def test_fit_affine(fn):
    x=np.array([[0.,0.],[1.,0.],[0.,1.],[1.,1.]])
    y=np.c_[2*x[:,0]-x[:,1]+3,x[:,1]-4]
    w=fn(x,y);np.testing.assert_allclose(np.c_[x,np.ones(4)]@w,y,atol=1e-10)
    np.testing.assert_allclose(w,[[2,0],[-1,1],[3,-4]],atol=1e-10)
    passed("test_fit_affine")


def test_normalized_probe_error(fn):
    truth=np.array([[-1.,-2.],[1.,2.]])
    np.testing.assert_allclose(fn(np.zeros_like(truth),truth),[1,1]);np.testing.assert_allclose(fn(truth,truth),[0,0])
    try:fn(np.ones((3,1)),np.ones((3,1)))
    except ValueError:pass
    else:raise AssertionError("constant factor must fail")
    passed("test_normalized_probe_error")


def test_ablate_motion(fn):
    x=np.array([[1.,3.,7.,1.],[2.,1.,4.,1.]])
    y=fn(x);np.testing.assert_allclose(y,[[1,1,7,1],[2,2,4,1]]);np.testing.assert_allclose(x[:,1],[3,1])
    passed("test_ablate_motion")


def test_retrieval_accuracy(fn):
    p=np.array([[1.,0.],[0.,1.]])
    a,d=fn(p,np.array([[0.,1.],[1.,0.]]),np.array([1,0]));assert a==1
    np.testing.assert_allclose(d,[[2,0],[0,2]])
    a,_=fn(np.zeros((1,2)),np.zeros((2,2)),np.array([0]));assert a==1
    passed("test_retrieval_accuracy")

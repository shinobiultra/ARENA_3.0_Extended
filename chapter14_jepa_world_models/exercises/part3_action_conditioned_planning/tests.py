import numpy as np
import torch as t

def passed(name):
    print(f"All tests in {name} passed!")


def test_world_step(fn):
    s = np.array([[0., 2., 8.], [3., -1., 4.]])
    saved = s.copy()
    np.testing.assert_allclose(fn(s, np.array([1., -1.])), [[2.5, 3., 8.], [1.5, -2., 4.]])
    np.testing.assert_allclose(fn(np.array([0., 0., 1.]), 1., gain=2.), [1., 2., 1.])
    np.testing.assert_array_equal(s, saved)
    passed("test_world_step")


def test_transition_features(fn):
    np.testing.assert_allclose(fn(np.array([[2., 3., 4.], [5., 6., 7.]]), np.array([-1., 1.])), [[2,3,4,-1,1],[5,6,7,1,1]])
    np.testing.assert_allclose(fn(np.zeros(3), np.array(0.)), [0,0,0,0,1])
    passed("test_transition_features")


def test_fit_dynamics(fn):
    x = np.array([[1.,0.],[0.,1.],[1.,1.],[-1.,2.]])
    w = np.array([[2.,-1.],[3.,4.]])
    np.testing.assert_allclose(fn(x,x@w),w,atol=1e-10)
    z = np.zeros((3,2)); np.testing.assert_allclose(fn(z,np.ones((3,1))),np.zeros((2,1)))
    passed("test_fit_dynamics")


def test_predict_step(fn):
    w=np.zeros((5,3)); w[0,0]=1;w[1,0]=1;w[1,1]=1;w[2,2]=1;w[3,:2]=[.5,1]
    np.testing.assert_allclose(fn(np.array([[0.,0.,9.],[2.,1.,8.]]),np.array([1.,-1.]),w),[[.5,1,9],[2.5,0,8]])
    passed("test_predict_step")


def test_imagine(fn):
    w=np.zeros((5,3));w[:3]=np.eye(3);w[3,0]=1
    r=fn(np.zeros(3),np.array([[1.,2.],[-1.,0.]]),w)
    np.testing.assert_allclose(r[:,:,0],[[0,1,3],[0,-1,-1]])
    assert r.shape==(2,3,3)
    assert fn(np.zeros(3),np.empty((1,0)),w).shape==(1,1,3)
    passed("test_imagine")


def test_goal_cost(fn):
    tr=np.array([[[0,0,0],[1,2,7]],[[0,0,0],[2,0,9]]],dtype=float)
    a=np.array([[1.,0.],[0.,0.]])
    np.testing.assert_allclose(fn(tr,a,[2,0,7]),[9.02,0])
    np.testing.assert_allclose(fn(tr,a,[2,0,7],nuisance_weight=3),[9.02,12])
    passed("test_goal_cost")


def test_select_elites(fn):
    a=np.arange(8).reshape(4,2)
    np.testing.assert_array_equal(fn(a,np.array([2.,1.,1.,3.]),2),a[[1,2]])
    try: fn(a,np.zeros(4),0)
    except ValueError: pass
    else: raise AssertionError("zero elites must fail")
    passed("test_select_elites")


def test_cem_plan(fn):
    w=np.zeros((5,3));w[:3]=np.eye(3);w[3,0]=1
    plan,h=fn(np.zeros(3),np.array([2.,0.,0.]),w,horizon=3,seed=5)
    assert plan.shape==(3,) and len(h)==7
    assert np.max(np.abs(plan))<=1 and np.all(np.diff(h)<=1e-12)
    assert (plan.sum()-2)**2 < .08
    p2,_=fn(np.zeros(3),np.array([2.,0.,0.]),w,horizon=3,seed=5)
    np.testing.assert_array_equal(plan,p2)
    passed("test_cem_plan")


def test_run_mpc(fn):
    w=np.zeros((5,3));w[:3]=np.eye(3);w[3,0]=.5;w[3,1]=1;w[1,0]=1
    s,a=fn(np.zeros(3),np.array([1.,0.,0.]),w,decisions=2,gain=.7)
    assert s.shape==(3,3) and a.shape==(2,)
    np.testing.assert_allclose(s[1:,1]-s[:-1,1],.7*a,atol=1e-10)
    np.testing.assert_allclose(s[1:,0]-s[:-1,0],s[:-1,1]+.35*a,atol=1e-10)
    passed("test_run_mpc")


def test_physical_error(fn):
    np.testing.assert_allclose(fn(np.array([[3.,4.,99.],[0.,0.,-9.]]),[0,0,0]),[5,0])
    passed("test_physical_error")


def test_controllability(fn):
    w=np.zeros((5,3));w[:3]=[[1,0,0],[1,1,0],[0,0,1]];w[3]=[.5,1,0]
    np.testing.assert_allclose(fn(w),[[.5,1.5,2.5],[1,1,1],[0,0,0]])
    assert np.linalg.matrix_rank(fn(w))==2
    passed("test_controllability")


def test_patch_latent(fn):
    r=np.array([0.,1.,2.]);d=np.array([7.,8.,9.]);o=fn(r,d,[2])
    np.testing.assert_array_equal(o,[0,1,9]);np.testing.assert_array_equal(r,[0,1,2])
    np.testing.assert_array_equal(fn(r,d,[0,1]),[7,8,2])
    passed("test_patch_latent")

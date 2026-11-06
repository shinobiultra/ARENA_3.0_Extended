"""Independent analytic fixtures; no pretrained weights or model calls."""
import numpy as np

def passed(name):print(f"All tests in {name} passed!")
def test_render_scene(fn):
    x=np.array([[1.,0.,0.],[-1.,0.,1.]])
    y=fn(x);assert y.shape==(2,8,8,3)
    np.testing.assert_allclose(y[0,2,1],[.75,.3,.25]);np.testing.assert_allclose(y[1,0,0],[.35]*3)
    np.testing.assert_allclose(x,[[1,0,0],[-1,0,1]])
    passed('test_render_scene')
def test_object_color_score(fn):
    x=np.zeros((2,2,2,3));x[0,0,0]=[.8,.1,.2];x[1,0,0]=[.1,.9,.7]
    mask=np.array([[True,False],[False,False]])
    np.testing.assert_allclose(fn(x,mask),[.6,-.6]);passed('test_object_color_score')
def test_paired_direction(fn):
    a=np.array([[2,7,3],[4,-2,8]]);b=np.array([[0,7,3],[0,-2,8]])
    np.testing.assert_allclose(fn(a,b),[3,0,0]);passed('test_paired_direction')
def test_steer_state(fn):
    a=np.array([[1.,2.,3.],[4.,5.,6.]]);copy=a.copy()
    np.testing.assert_allclose(fn(a,[2,0,-1],-.5),[[0,2,3.5],[3,5,6.5]])
    np.testing.assert_array_equal(a,copy);passed('test_steer_state')
def test_remove_direction(fn):
    np.testing.assert_allclose(fn([[1.,2.,3.]],np.array([2.,0.,0.])),[[0,2,3]])
    try:fn([[1,2,3]],[0,0,0])
    except ValueError:pass
    else:raise AssertionError('zero direction must fail')
    passed('test_remove_direction')
def test_lora_update(fn):
    np.testing.assert_allclose(fn(np.array([[1],[2]]),np.array([[3,4,5]]),.5),[[1.5,2,2.5],[3,4,5]])
    passed('test_lora_update')
def test_adapter_forward(fn):
    x=np.array([[1.,2.]])
    w=np.array([[1,0],[0,2],[1,1.]])
    b=np.array([[1.],[2.],[0.]]);a=np.array([[3.,-1.]])
    np.testing.assert_allclose(fn(x,w,b,a,.5),[[1.5,5,3]])
    passed('test_adapter_forward')
def test_fit_right_factor(fn):
    x=np.eye(3);b=np.array([[1.],[0.],[0.]])
    target=np.zeros((3,3));target[0,0]=.6
    np.testing.assert_allclose(fn(x,target,b),[[.6,0,0]],atol=1e-12);passed('test_fit_right_factor')
def test_rank_reconstruction(fn):
    m=np.diag([3.,2.,1.]);np.testing.assert_allclose(fn(m,1),np.diag([3,0,0]))
    np.testing.assert_allclose(fn(m,0),np.zeros((3,3)));np.testing.assert_allclose(fn(m,3),m)
    passed('test_rank_reconstruction')
def test_background_damage(fn):
    a=np.zeros((1,2,2,3));b=a.copy();b[0,0,0]=1
    mask=np.array([[True,False],[False,False]])
    np.testing.assert_allclose(fn(a,b,mask),[0]);b[0,1,1]=.6
    np.testing.assert_allclose(fn(a,b,mask),[.2]);passed('test_background_damage')
def test_matched_nuisance_update(fn):
    u=np.diag([.6,0,0]);actual=fn(u)
    expected=np.zeros((3,3));expected[2,0]=.6
    np.testing.assert_allclose(actual,expected);assert np.linalg.matrix_rank(actual)==1
    passed('test_matched_nuisance_update')

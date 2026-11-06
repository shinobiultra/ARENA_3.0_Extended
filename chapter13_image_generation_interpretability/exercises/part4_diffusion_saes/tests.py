"""Analytic and finite-difference CPU checks; no learned model imports."""
import numpy as np

def passed(n):print(f"All tests in {n} passed!")
def test_sparse_encode(fn):
    np.testing.assert_allclose(fn([[2.,-1.]],np.eye(2),.5),[[1.5,0.]])
    np.testing.assert_allclose(fn([[-1.,-2.]],np.eye(2),0),[[0,0]])
    passed('test_sparse_encode')
def test_sparse_decode(fn):
    np.testing.assert_allclose(fn([[2,3]],[[1,0,2],[0,1,1]]),[[2,3,7]])
    passed('test_sparse_decode')
def test_render_sparse_scene(fn):
    x=np.array([[1,0,1,0]],float)/np.sqrt(2);y=fn(x)
    assert y.shape==(1,8,10,3);np.testing.assert_allclose(y[0,2,1],[.9,.2,.2]);np.testing.assert_allclose(y[0,2,7],[.2,.2,.2])
    passed('test_render_sparse_scene')
def test_reconstruction_mse(fn):
    assert abs(fn([[1,3]],[[0,1]])-2.5)<1e-12;passed('test_reconstruction_mse')
def test_sae_objective(fn):
    assert abs(fn([[1.,3.]],[[0.,1.]],[[2.,0.]],.3)-2.8)<1e-12
    passed('test_sae_objective')
def test_active_features(fn):
    np.testing.assert_array_equal(fn([[0,2,0],[1,2,3]]),[1,3]);passed('test_active_features')
def test_decoder_gradient(fn):
    x=np.array([[.2,-.3,.4],[.8,.2,-.2]]);z=np.array([[1.,.5],[.3,1.]]);d=np.array([[.1,.2,.3],[.4,.5,.6]])
    numerical=np.zeros_like(d);eps=1e-6
    for i in range(2):
      for j in range(3):
        plus=d.copy();minus=d.copy();plus[i,j]+=eps;minus[i,j]-=eps
        numerical[i,j]=(np.mean((z@plus-x)**2)-np.mean((z@minus-x)**2))/(2*eps)
    np.testing.assert_allclose(fn(x,z,d),numerical,atol=1e-9);passed('test_decoder_gradient')
def test_edit_feature(fn):
    z=np.array([[1,2],[3,4.]]);before=z.copy()
    np.testing.assert_allclose(fn(z,1,0),[[1,0],[3,0]]);np.testing.assert_array_equal(z,before)
    passed('test_edit_feature')
def test_residual_preserving_patch(fn):
    x=np.array([[3.,4.,9.]]);z=np.array([[1.,2.]]);d=np.array([[1,0,0],[0,1,0.]])
    np.testing.assert_allclose(fn(x,z,[[0,2]],d),[[2,4,9]])
    np.testing.assert_array_equal(fn(x,z,z,d),x);passed('test_residual_preserving_patch')
def test_masked_contrast(fn):
    x=np.zeros((1,2,2,3));x[0,0,0]=[.8,.2,.5];m=np.array([[True,False],[False,False]])
    np.testing.assert_allclose(fn(x,m,0),[.6]);np.testing.assert_allclose(fn(x,m,2),[.3]);passed('test_masked_contrast')
def test_outside_mask_mae(fn):
    x=np.zeros((1,2,2,3));y=x.copy();y[0,0,0]=1;m=np.array([[True,False],[False,False]])
    np.testing.assert_allclose(fn(x,y,m),[0]);y[0,1,1]=.3
    np.testing.assert_allclose(fn(x,y,m),[.1]);passed('test_outside_mask_mae')
def test_feature_contrast(fn):
    np.testing.assert_allclose(fn([[2,1],[4,3],[1,0]],[1,1,0]),[2,2])
    try:fn([[1,2]],[1])
    except ValueError:pass
    else:raise AssertionError('missing group accepted')
    passed('test_feature_contrast')

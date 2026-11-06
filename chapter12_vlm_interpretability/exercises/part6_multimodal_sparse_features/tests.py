"""Small independent analytic fixtures for multimodal sparse methods."""
import numpy as np

def _ok(name):print(f'All tests in test_{name} passed!')

def test_pixel_statistics(fn):
 x=np.zeros((2,2,2,3));x[0,:,:,0]=.6;x[1,:,:,2]=.3
 np.testing.assert_allclose(fn(x),[[1,0,0,.2],[0,0,1,.1]])
 _ok('pixel_statistics')

def test_word_features(fn):
 np.testing.assert_allclose(fn(['blue triangle','red square'],['red square','green circle','blue triangle']),[[0,0,1,1.3],[1,0,0,1.]])
 try:fn(['missing'],['x'])
 except ValueError:pass
 else:raise AssertionError('Unknown words require explicit handling')
 _ok('word_features')

def test_sae_encode(fn):
 np.testing.assert_allclose(fn(np.array([[2.,-1.],[0.,3.]]),np.eye(2),[-1.,.5]),[[1,0],[0,3.5]])
 _ok('sae_encode')

def test_reconstruct(fn):
 z=np.array([[1.,2.],[0,0]]);d=np.array([[2,0,1],[0,3,-1]])
 np.testing.assert_allclose(fn(z,d,np.array([1,2,3])),[[3,8,2],[1,2,3]])
 _ok('reconstruct')

def test_reconstruction_error(fn):
 assert fn(np.array([[1.,2.],[3,4]]),np.zeros((2,2)))==15
 _ok('reconstruction_error')

def test_transcoder_targets(fn):
 x=np.array([[1.,2.],[-1,1.]])
 w=np.array([[1,0,-1],[0,1,1]])
 np.testing.assert_allclose(fn(x,w,np.array([0,-2,1.])),[[1,0,2],[0,0,3]])
 _ok('transcoder_targets')

def test_fit_decoder(fn):
 z=np.array([[1.,0],[0,1],[1,1],[0,0]])
 y=z@np.array([[2.,-1],[3.,4]])+np.array([.5,-.25])
 d,b=fn(z,y,ridge=0)
 np.testing.assert_allclose(d,[[2,-1],[3,4]],atol=1e-10);np.testing.assert_allclose(b,[.5,-.25],atol=1e-10)
 # Intercept is not penalised: a constant target stays exact even under ridge.
 d,b=fn(z,np.full((4,1),7.),ridge=100)
 np.testing.assert_allclose(d,0,atol=1e-10);np.testing.assert_allclose(b,7,atol=1e-10)
 _ok('fit_decoder')

def test_crosscoder_encode(fn):
 x=np.array([[1.,0],[0,1]]);y=np.array([[0.,2.],[3.,0.]])
 np.testing.assert_allclose(fn(x,y,np.eye(2),np.eye(2),[-.5,0]),[[.5,2],[2.5,1]])
 np.testing.assert_allclose(fn(x,np.zeros_like(y),np.eye(2),np.eye(2),[0,0]),x)
 np.testing.assert_allclose(fn(np.array([[1.,0.]]),np.array([[-2.,0.]]),np.eye(2),np.eye(2),[0,0]),[[0,0]])
 _ok('crosscoder_encode')

def test_crosscoder_loss(fn):
 x=np.array([[1.,2.],[3,4]]);y=np.array([[2.],[0.]]);z=np.array([[1.,0.],[2.,1.]])
 assert abs(fn(x,y,np.zeros_like(x),np.zeros_like(y),z,.1)-17.2)<1e-10
 _ok('crosscoder_loss')

def test_feature_overlap(fn):
 x=np.array([[1,1,0],[0,1,0]],float);y=np.array([[1,0,0],[0,1,0]],float)
 np.testing.assert_allclose(fn(x,y,.5),[1,.5,0]);_ok('feature_overlap')

def test_derange_pairs(fn):
 x=np.arange(12).reshape(4,3);out=fn(x)
 assert sorted(out[:,0])==sorted(x[:,0]) and np.all(out[:,0]!=x[:,0]);np.testing.assert_array_equal(x,np.arange(12).reshape(4,3))
 _ok('derange_pairs')

def test_feature_edges(fn):
 d=np.array([[1.,2.,0.],[-1,0,3]])
 e=np.array([[1,0],[0,2],[4,-1]])
 np.testing.assert_allclose(fn(d,e),[[1,4],[11,-3]])
 _ok('feature_edges')

def test_path_contribution(fn):
 z=np.array([[2.,3.],[1.,0.]])
 edges=np.array([[1,2],[3,4.]])
 readout=np.array([2.,-1.])
 expected=np.array([[[4,-4],[18,-12]],[[2,-2],[0,0]]])
 np.testing.assert_allclose(fn(z,edges,readout),expected)
 _ok('path_contribution')

def test_latent_ablation(fn):
 actual=np.array([[5.,7.,11.]])
 z=np.array([[2.,3.]])
 d=np.array([[1.,0,2],[0,1,0.]])
 np.testing.assert_allclose(fn(actual,z,d,0),[[3,7,7]])
 np.testing.assert_allclose(actual,[[5,7,11]])
 _ok('latent_ablation')

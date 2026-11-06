"""Semantic callback tests; NumPy only, with independent small exact oracles."""
import numpy as np

def _ok(name): print(f'All tests in test_{name} passed!')

def test_grid_features(fn):
 image=np.zeros((1,4,4,3));image[:,:2,:2,0]=1;image[:,2:,2:,2]=1
 np.testing.assert_allclose(fn(image,2),[[[1,0],[0,0],[0,0],[0,1]]]);_ok('grid_features')

def test_assemble_tokens(fn):
 feats=np.array([[[1,0],[0,1],[0,0],[.5,.5]]],float);h=fn(feats,np.array([1]))
 assert h.shape==(1,6,6)
 np.testing.assert_allclose(h[0,:4,:2],feats[0]);np.testing.assert_allclose(h[0,:4,2:],np.eye(4))
 np.testing.assert_allclose(h[0,4,2:],[0,1,0,0]);np.testing.assert_allclose(h[0,5],0);_ok('assemble_tokens')

def test_scaled_attention(fn):
 q=np.array([[2.,0.]]);k=np.array([[1.,0.],[0.,1.]]);v=np.array([[1.,2.],[3.,4.]])
 weights=np.exp(np.array([2.,0.])/np.sqrt(2));weights/=weights.sum()
 np.testing.assert_allclose(fn(q,k,v),[weights@v]);_ok('scaled_attention')

def test_knockout_keys(fn):
 p=np.array([[.2,.3,.5],[1.,0.,0.]])
 out=fn(p,np.array([False,True,False]));np.testing.assert_allclose(out,[[2/7,0,5/7],[1,0,0]])
 try:fn(p,np.ones(3,dtype=bool))
 except ValueError:pass
 else:raise AssertionError('All keys blocked must be rejected')
 _ok('knockout_keys')

def test_flow_step(fn):
 h=np.zeros((1,6,6));h[0,:4,2:]=np.eye(4);h[0,0,0]=1;h[0,3,1]=1;h[0,4,2]=1
 out=fn(h,0)
 expected=np.exp([8.,0,0,0]);expected/=sum(expected)
 np.testing.assert_allclose(out[0,4,:2],[expected[0],expected[3]])
 np.testing.assert_allclose(out[0,5,:2],0)
 nxt=fn(out,1);np.testing.assert_allclose(nxt[0,5,:2],out[0,4,:2]);np.testing.assert_allclose(h[0,4,:2],0);_ok('flow_step')

def test_run_flow(fn):
 h=np.zeros((1,6,6));h[0,:4,2:]=np.eye(4);h[0,0,0]=1;h[0,4,2]=1
 cache=fn(h)
 assert list(cache)==[0,1,2,3]
 assert cache[1][0,4,0]>.99 and cache[1][0,5,0]==0
 assert cache[2][0,5,0]>.99
 cache[0][0,0,0]=99;assert h[0,0,0]==1;_ok('run_flow')

def test_candidate_margin(fn):
 scores=np.array([[5.,2.,9.],[1.,6.,0.]])
 np.testing.assert_allclose(fn(scores,np.array([2,1]),np.array([0,0])),[4.,5.]);_ok('candidate_margin')

def test_replace_positions(fn):
 corrupt=np.zeros((2,4,3));clean=np.arange(24).reshape(2,4,3);out=fn(corrupt,clean,[1,3])
 np.testing.assert_array_equal(out[:,[1,3]],clean[:,[1,3]]);np.testing.assert_array_equal(out[:,[0,2]],0)
 np.testing.assert_array_equal(corrupt,0);_ok('replace_positions')

def test_normalised_recovery(fn):
 np.testing.assert_allclose(fn(np.array([3.,1.]),np.array([5.,3.]),np.array([1.,-1.])),[.5,.5])
 np.testing.assert_allclose(fn(np.array([7.,-1.]),np.array([5.,5.]),np.array([1.,1.])),[1.5,-.5])
 for clean in [np.array([1.]),np.array([float('nan')])]:
  try:fn(np.array([2.]),clean,np.array([1.]))
  except ValueError:pass
  else:raise AssertionError('Undefined or reversed denominator accepted')
 _ok('normalised_recovery')

def test_patch_sweep(fn):
 clean=np.zeros((1,6,6));clean[0,:4,2:]=np.eye(4);clean[0,0,0]=1;clean[0,4,2]=1
 corrupt=clean.copy();corrupt[0,0,:2]=[0,1]
 out=fn(clean,corrupt,{'object':[0],'background':[1],'last':[5],'full':list(range(6))},np.array([0]),np.array([1]))
 np.testing.assert_allclose(out['object'],[1,0,0,0],atol=1e-10)
 np.testing.assert_allclose(out['background'],0,atol=1e-10)
 np.testing.assert_allclose(out['last'],[0,0,1,1],atol=1e-10)
 np.testing.assert_allclose(out['full'],1,atol=1e-10);_ok('patch_sweep')

def test_decode_visual_tokens(fn):
 h=np.zeros((1,3,4));h[0,0,:2]=[2,-1];h[0,1,:2]=[-3,4]
 out=fn(h,np.array([[1.,0.],[0.,1.],[0,0],[0,0]]))
 assert out.shape==(1,3) and list(out[0])==[0,1,0];_ok('decode_visual_tokens')

def test_patchscope_readout(fn):
 source=np.zeros((1,6,6));source[0,0,:2]=[0,1]
 target=np.zeros((1,6,6));target[0,:4,2:]=np.eye(4);target[0,4,3]=1
 logits=fn(source,target,0,1)
 assert logits[0,1]>.99 and logits[0,0]==0
 np.testing.assert_allclose(target[0,1,:2],0);_ok('patchscope_readout')

def test_validate_layout(fn):
 assert fn(6,[0,1,2,3],5)==(0,1,2,3)
 for image,answer in [([0,0],5),([0,6],5),([0,1],1)]:
  try:fn(6,image,answer)
  except ValueError:pass
  else:raise AssertionError('Ambiguous/invalid image-answer layout accepted')
 _ok('validate_layout')

"""Independent callback fixtures for the CPU typography/sparse-feature organism."""
import numpy as np

def _ok(name): print(f'All tests in test_{name} passed!')

def test_normalise_rows(fn):
 x=np.array([[3.,4.],[0.,0.]])
 np.testing.assert_allclose(fn(x),[[.6,.8],[0.,0.]])
 np.testing.assert_allclose(fn(x*7),fn(x));_ok('normalise_rows')

def test_image_channels(fn):
 images=np.zeros((2,8,8,3));images[0,:4,:,0]=1;images[1,:4,:,1]=1
 templates=np.zeros((3,4,8,3));templates[1]=1;templates[2]=2
 images[1,4:]=1
 out=fn(images,templates)
 np.testing.assert_allclose(out,[[1,0,0,1,0,0],[0,1,0,0,1,0]])
 # Mixed RGB distinguishes sum normalization from the preceding L2 exercise.
 # Literal oracle: [.6, .8, 0] / 1.4 = [3/7, 4/7, 0].
 mixed=np.zeros((2,8,8,3));mixed[0,:4]=[.6,.8,0.]
 mixed[0,4:]=.25;mixed[1,4:]=1.
 templates_with_blank=np.empty((4,4,8,3))
 templates_with_blank[0]=.25;templates_with_blank[1]=.5
 templates_with_blank[2]=.75;templates_with_blank[3]=1.
 mixed_before=mixed.copy();templates_before=templates_with_blank.copy()
 np.testing.assert_allclose(fn(mixed,templates_with_blank),
                            [[3/7,4/7,0,1,0,0],[0,0,0,0,0,0]],
                            err_msg='RGB uses sum normalization; zero RGB and a blank word match stay zero.')
 np.testing.assert_array_equal(mixed,mixed_before)
 np.testing.assert_array_equal(templates_with_blank,templates_before)
 _ok('image_channels')

def test_contrastive_logits(fn):
 x=np.array([[3.,0.],[0.,2.]])
 np.testing.assert_allclose(fn(x,np.eye(2),2),[[2,0],[0,2]])
 np.testing.assert_allclose(fn(x*5,np.eye(2),2),fn(x,np.eye(2),2));_ok('contrastive_logits')

def test_encode_sae(fn):
 np.testing.assert_allclose(fn(np.array([[1.,-2.],[3.,4.]]),np.eye(2),np.array([-.5,1.])),[[.5,0],[2.5,5]])
 _ok('encode_sae')

def test_decode_sae(fn):
 z=np.array([[1.,2.],[0.,0.]]);w=np.array([[2.,0.,1.],[0.,3.,-1.]])
 np.testing.assert_allclose(fn(z,w,np.array([.1,.2,.3])),[[2.1,6.2,-.7],[.1,.2,.3]]);_ok('decode_sae')

def test_sparse_loss(fn):
 x=np.array([[1.,2.],[3.,4.]]);r=np.zeros_like(x);z=np.array([[1.,0.],[2.,3.]])
 assert abs(fn(x,r,z,.2)-15.6)<1e-10;_ok('sparse_loss')

def test_sae_step(fn):
 rng=np.random.default_rng(4);x=rng.normal(size=(3,2));we=rng.normal(size=(2,3));be=np.array([2.,2.,2.]);wd=rng.normal(size=(3,2));bd=np.zeros(2)
 params=[we,be,wd,bd];alpha=.03;lr=1e-4
 def objective(ps):
  z=np.maximum(0,x@ps[0]+ps[1]);r=z@ps[2]+ps[3]
  return ((r-x)**2).sum()/len(x)+alpha*z.sum()/len(x)
 expected=[]
 for p in params:
  grad=np.zeros_like(p)
  for index in np.ndindex(p.shape):
   old=p[index];p[index]=old+1e-6;plus=objective(params);p[index]=old-1e-6;minus=objective(params);p[index]=old;grad[index]=(plus-minus)/2e-6
  expected.append(p-lr*grad)
 actual=fn(x,*[p.copy() for p in params],alpha,lr)
 for a,b in zip(actual,expected): np.testing.assert_allclose(a,b,atol=1e-9)
 _ok('sae_step')

def test_fit_sae(fn):
 x=np.array([[1.,0.],[0.,1.],[1.,1.],[0.,0.]])
 result=fn(x,width=4,steps=160,lr=.05,alpha=.001,seed=2)
 we,be,wd,bd,losses=result
 assert we.shape==(2,4) and wd.shape==(4,2) and len(losses)==160
 assert losses[-1]<losses[0]*.25
 same=fn(x,width=4,steps=160,lr=.05,alpha=.001,seed=2)
 for a,b in zip(result,same):np.testing.assert_allclose(a,b)
 _ok('fit_sae')

def test_top_examples(fn):
 a=np.array([[.2,5.],[.9,1.],[.9,2.]])
 assert list(fn(a,0,2))==[1,2];_ok('top_examples')

def test_feature_direction(fn):
 np.testing.assert_allclose(fn(np.array([[3.,4.],[0.,2.]]),0),[.6,.8]);_ok('feature_direction')

def test_ablate_direction(fn):
 x=np.array([[2.,3.],[-1.,4.]])
 np.testing.assert_allclose(fn(x,np.array([1.,0.]),1),[[0,3],[0,4]])
 np.testing.assert_allclose(fn(x,np.array([1.,0.]),0),x)
 np.testing.assert_allclose(fn(x,np.array([1.,0.]),.5),[[1,3],[-.5,4]]);_ok('ablate_direction')

def test_forced_choice_accuracy(fn):
 assert fn(np.array([[2,1],[3,4],[1,1]]),np.array([0,1,1]))==2/3
 _ok('forced_choice_accuracy')

def test_tradeoff_curve(fn):
 # Decoder e1 is removed at alpha1, flipping only the conflicted second row.
 x=np.array([[2.,0.],[.5,2.]])
 out=fn(x,np.eye(2),np.array([0,0]),np.array([0.,1.]),[0.,1.])
 np.testing.assert_allclose(out,[.5,1.]);_ok('tradeoff_curve')

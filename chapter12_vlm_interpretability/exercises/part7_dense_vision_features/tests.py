"""Independent NumPy fixtures for dense patch analysis; no pretrained model loading."""
import numpy as np

def _ok(name): print(f'All tests in test_{name} passed!')

def test_patch_means(fn):
 x=np.arange(4*6*3).reshape(4,6,3)
 expected=np.array([[x[r:r+2,c:c+2].mean((0,1)) for c in range(0,6,2)] for r in range(0,4,2)])
 np.testing.assert_allclose(fn(x,2),expected)
 _ok('patch_means')

def test_fit_pca(fn):
 x=np.array([[3.,1.,0.],[-3,1,0],[0,-2,0],[0,0,0]])
 mean,components=fn(x,2)
 np.testing.assert_allclose(mean,x.mean(0))
 np.testing.assert_allclose(components@components.T,np.eye(2),atol=1e-12)
 # Distinct variances identify the two axes up to sign.
 np.testing.assert_allclose(np.abs(components),[[1,0,0],[0,1,0]],atol=1e-12)
 _ok('fit_pca')

def test_project_pca(fn):
 x=np.array([[2.,4.,8.],[1,3,5.]])
 np.testing.assert_allclose(fn(x,np.array([1,2,3]),np.array([[1,0,0],[0,0,1]])),[[1,5],[0,2]])
 _ok('project_pca')

def test_cosine_matches(fn):
 source=np.array([[4.,0.],[0,3.],[-1,0.]])
 target=np.array([[0.,2.],[-4,0.],[2,0.]])
 indices,scores=fn(source,target)
 np.testing.assert_array_equal(indices,[2,0,1]);np.testing.assert_allclose(scores,1)
 _ok('cosine_matches')

def test_mutual_matches(fn):
 # A->B [0,0,2], B->A [1,0,2]: source1 andsource2 are mutual.
 np.testing.assert_array_equal(fn(np.array([0,0,2]),np.array([1,0,2])),[False,True,True])
 _ok('mutual_matches')

def test_correspondence_accuracy(fn):
 assert fn(np.array([0,1,2,1]),np.array([2,0,1,1]),np.array([1,2,0,1]),np.array([False,True,True,True]))==2/3
 _ok('correspondence_accuracy')

def test_norm_outliers(fn):
 x=np.array([[1.,0],[0,1],[20,0],[1,0]])
 np.testing.assert_array_equal(fn(x,5),[False,False,True,False]);_ok('norm_outliers')

def test_attention_weights(fn):
 q=np.array([1.,0.]);keys=np.array([[2.,0.],[0.,2.]])
 expected=np.exp([2/np.sqrt(2),0]);expected/=sum(expected)
 np.testing.assert_allclose(fn(q,keys),expected);_ok('attention_weights')

def test_teacher_student_loss(fn):
 teacher=np.array([[2.,0.],[0,2.]])
 student=np.array([[1.,0.],[0.,1.]])
 # teacher temperature1, studenttemperature2, centre0: independent scalarCE.
 p=np.exp([2.,0.]);p/=sum(p);q=np.exp([.5,0]);q/=sum(q)
 expected=-sum(p*np.log(q))
 assert abs(fn(teacher,student,np.zeros(2),1,2)-expected)<1e-12
 assert fn(teacher,student[::-1],np.zeros(2),1,2)>expected
 _ok('teacher_student_loss')

def test_masked_pool(fn):
 x=np.array([[1.,2.],[3,4],[9,8.]])
 np.testing.assert_allclose(fn(x,np.array([True,True,False])),[2,3])
 try:fn(x,np.zeros(3,dtype=bool))
 except ValueError:pass
 else:raise AssertionError('Empty prompt-region pool must fail')
 _ok('masked_pool')

def test_prompt_mask(fn):
 x=np.array([[1.,0.],[0.,1.],[1,1.]])
 mask,scores=fn(x,np.array([1.,0.]),np.array([0.,1.]),.1)
 np.testing.assert_array_equal(mask,[True,False,False]);np.testing.assert_allclose(scores,[1,-1,0],atol=1e-12)
 _ok('prompt_mask')

def test_mask_iou(fn):
 assert fn(np.array([True,True,False]),np.array([False,True,True]))==1/3
 assert fn(np.zeros(3,dtype=bool),np.zeros(3,dtype=bool))==1
 _ok('mask_iou')

def test_replace_region(fn):
 x=np.zeros((4,2));donor=np.arange(8).reshape(4,2);mask=np.array([False,True,False,True])
 out=fn(x,donor,mask);np.testing.assert_array_equal(out[mask],donor[mask]);np.testing.assert_array_equal(out[~mask],0);np.testing.assert_array_equal(x,0)
 _ok('replace_region')

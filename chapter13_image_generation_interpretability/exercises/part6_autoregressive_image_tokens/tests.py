"""Exact image geometry, gradient and causal-prefix fixtures."""
import numpy as np

def passed(n):print(f"All tests in {n} passed!")
def test_patch_means(fn):
    image=np.arange(4*6*3).reshape(4,6,3)
    actual=fn(image,2);expected=np.array([[image[y:y+2,x:x+2].mean((0,1)) for x in range(0,6,2)] for y in range(0,4,2)])
    np.testing.assert_allclose(actual,expected);passed('test_patch_means')
def test_nearest_codes(fn):
    cb=np.array([[0.,0.],[2.,0.],[0.,2.]])
    np.testing.assert_array_equal(fn([[1,0],[0,1.8],[1.9,0]],cb),[0,2,1]);passed('test_nearest_codes')
def test_decode_tokens(fn):
    cb=np.array([[.2,.2,.2],[.8,.2,.2]])
    y=fn([[0,1],[1,0]],cb,2);assert y.shape==(4,4,3)
    np.testing.assert_allclose(y[0,3],cb[1]);np.testing.assert_allclose(y[3,3],cb[0]);passed('test_decode_tokens')
def test_vq_objective(fn):
    import torch as t
    e=t.tensor([[1.,3.]],dtype=t.float64,requires_grad=True);q=t.tensor([[0.,1.]],dtype=t.float64,requires_grad=True)
    loss=fn(e,q,.25);assert abs(loss.item()-3.125)<1e-12
    loss.backward();t.testing.assert_close(e.grad,t.tensor([[.25,.5]],dtype=t.float64));t.testing.assert_close(q.grad,t.tensor([[-1.,-2.]],dtype=t.float64))
    passed('test_vq_objective')
def test_straight_through_quantized(fn):
    import torch as t
    e=t.tensor([1.,3.],requires_grad=True);q=t.tensor([0.,2.],requires_grad=True)
    out=fn(e,q);t.testing.assert_close(out,q);out.sum().backward()
    t.testing.assert_close(e.grad,t.ones(2));assert q.grad is None or t.equal(q.grad,t.zeros(2))
    passed('test_straight_through_quantized')
def test_code_usage(fn):
    counts,ppl=fn([[0,0],[1,1]],3);np.testing.assert_array_equal(counts,[2,2,0]);assert abs(ppl-2)<1e-12
    assert fn([2,2,2],3)[1]==1;passed('test_code_usage')
def test_quantization_mse(fn):
    assert abs(fn([[1.,0.],[2.,3.]],[0,1],[[0.,0.],[2.,1.]])-1.25)<1e-12
    passed('test_quantization_mse')
def test_causal_attention(fn):
    q=np.zeros((3,2));k=np.zeros((3,2));v=np.array([[1.,0.],[3.,2.],[8.,9.]])
    out,w=fn(q,k,v);np.testing.assert_allclose(out,[[1,0],[2,1],[4,11/3]])
    assert np.all(w[np.triu_indices(3,1)]==0)
    v[-1]=[100,100];out2,_=fn(q,k,v);np.testing.assert_array_equal(out[:2],out2[:2])
    passed('test_causal_attention')
def test_next_token_logits(fn):
    layout=np.array([[1,1],[0,0]],bool)
    assert fn([2],0,layout).argmax()==2
    assert fn([1,2],1,layout).argmax()==2
    assert fn([1,2,2],2,layout).argmax()==0
    passed('test_next_token_logits')
def test_greedy_image_tokens(fn):
    layout=np.array([[0,1],[1,0]],bool)
    np.testing.assert_array_equal(fn(2,layout),[[0,2],[2,0]]);passed('test_greedy_image_tokens')
def test_regenerate_with_patch(fn):
    layout=np.array([[1,1],[0,1]],bool)
    np.testing.assert_array_equal(fn(1,layout,0,2),[[2,2],[0,2]])
    np.testing.assert_array_equal(fn(1,layout,3,2),[[1,1],[0,2]])
    passed('test_regenerate_with_patch')
def test_blue_mask_iou(fn):
    image=np.array([[[.2,.2,.8],[.2,.2,.2]],[[.2,.2,.2],[.2,.2,.2]]])
    assert fn(image,np.ones((2,2),bool))==.25
    assert fn(image,np.array([[0,1],[1,0]],bool))==0;passed('test_blue_mask_iou')
def test_outside_patch_changes(fn):
    x=np.zeros((2,2,3));y=x.copy();y[0,0]=1;mask=np.array([[1,0],[0,0]],bool)
    assert fn(x,y,mask)==0;y[1,1]=1;assert fn(x,y,mask)==1
    passed('test_outside_patch_changes')

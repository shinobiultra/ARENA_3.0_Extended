"""Independent geometry, direct-softmax and mask oracles."""
import numpy as np

def passed(n):print(f"All tests in {n} passed!")
def test_patchify(fn):
    x=np.arange(6*4*3).reshape(6,4,3);y=fn(x,2);assert y.shape==(6,12)
    np.testing.assert_array_equal(y[1],x[:2,2:4].ravel());np.testing.assert_array_equal(y[2],x[2:4,:2].ravel())
    passed('test_patchify')
def test_unpatchify(fn):
    image=np.arange(6*4*3).reshape(6,4,3)
    patches=np.stack([image[y:y+2,x:x+2].ravel() for y in range(0,6,2) for x in range(0,4,2)])
    np.testing.assert_array_equal(fn(patches,6,4,2),image);passed('test_unpatchify')
def test_refinement_step(fn):
    np.testing.assert_allclose(fn([0,2],[2,0],.25),[.5,1.5]);np.testing.assert_array_equal(fn([1,2],[8,9],0),[1,2])
    passed('test_refinement_step')
def test_adaptive_layer_norm(fn):
    x=np.array([[1.,3.],[7.,7.]])
    expected=np.array([[-1,1],[0,0]])/np.sqrt(1+1e-6)*2+.3
    np.testing.assert_allclose(fn(x,1,.3),expected);passed('test_adaptive_layer_norm')
def test_scaled_attention(fn):
    out,w=fn(np.zeros((2,3)),np.array([[1,2,3],[4,5,6.]]),np.array([[1,2],[3,8.]]))
    np.testing.assert_allclose(w,np.full((2,2),.5));np.testing.assert_allclose(out,[[2,5],[2,5]])
    q=np.array([[3.,-2.]]);k=np.array([[1,0],[0,1.]])
    probs=np.exp(np.array([3.,-2.])/np.sqrt(2));probs/=probs.sum()
    np.testing.assert_allclose(fn(q,k,np.eye(2))[0],[probs]);passed('test_scaled_attention')
def test_dit_attention_block(fn):
    x=np.array([[1.,2.],[3.,-1.]]);i=np.eye(2)
    np.testing.assert_array_equal(fn(x,.2,.3,0.,i,i,i,i),x)
    # Independent row-wise oracle: no calls to student normalization/attention.
    tokens=np.array([[.3,-1.2,.8],[1.1,.2,-.4],[-.7,.9,.4]])
    scale=np.array([.2,-.35,.1]);shift=np.array([-.1,.25,.3]);gate=.4
    wq=np.array([[.7,-.2],[.3,.8],[-.5,.4]])
    wk=np.array([[-.4,.6],[.9,.1],[.2,-.7]])
    wv=np.array([[.5,.2],[-.3,.8],[.6,-.4]])
    wo=np.array([[.7,-.2,.3],[-.5,.9,.4]])
    conditioned=np.empty_like(tokens)
    for row_index,row in enumerate(tokens):
        mean=sum(row)/len(row)
        variance=sum((value-mean)**2 for value in row)/len(row)
        for feature,value in enumerate(row):
            conditioned[row_index,feature]=(1+scale[feature])*(value-mean)/np.sqrt(variance+1e-6)+shift[feature]
    queries=conditioned@wq;keys=conditioned@wk;values=conditioned@wv
    expected=np.empty_like(tokens)
    for row_index,query in enumerate(queries):
        scores=np.array([sum(query*key)/np.sqrt(len(query)) for key in keys])
        weights=np.exp(scores-max(scores));weights/=sum(weights)
        attention=np.array([sum(weights*values[:,feature]) for feature in range(values.shape[1])])
        expected[row_index]=tokens[row_index]+gate*(attention@wo)
    np.testing.assert_allclose(fn(tokens,scale,shift,gate,wq,wk,wv,wo),expected,rtol=1e-10,atol=1e-12)
    np.testing.assert_array_equal(fn(tokens,scale,shift,0.,wq,wk,wv,wo),tokens)
    passed('test_dit_attention_block')
def test_readonly_concept_attention(fn):
    q=np.array([[2.,0.],[0.,2.]]);k=np.zeros((2,2));v=np.zeros((2,2));p=np.eye(2)
    a=fn(q,k,v,p,p,q,p,p);b=fn(q,k,v,p,p,q[::-1]*3,p*2,p[::-1])
    np.testing.assert_array_equal(a[0],b[0]);np.testing.assert_array_equal(a[2],b[2])
    assert a[3].shape==(2,4);assert a[2].shape==(2,4)
    assert not np.allclose(a[1],b[1]);passed('test_readonly_concept_attention')
def test_output_space_map(fn):
    y=fn([[2.,0.],[0.,1.]],np.eye(2));expected=np.array([[np.e**2,1],[1,np.e]])
    expected/=expected.sum(-1,keepdims=True);np.testing.assert_allclose(y,expected)
    np.testing.assert_allclose(fn([[3.,7.]],[[1.,2.],[1.,2.]]),[[.5,.5]])
    passed('test_output_space_map')
def test_average_layer_maps(fn):
    np.testing.assert_allclose(fn([[[.2,.8]],[[.6,.4]]]),[[.4,.6]]);passed('test_average_layer_maps')
def test_raw_key_map(fn):
    np.testing.assert_allclose(fn(np.eye(2),np.zeros((3,2))),np.full((3,2),.5))
    queries=np.array([[1.3,-.4],[-.2,.9]])
    keys=np.array([[.2,1.1],[-.7,.4],[1.2,-.3]])
    # Direct query/key scalar products, then a separate concept-axis softmax.
    expected=[]
    for key in keys:
        scores=np.array([sum(key*query)/np.sqrt(len(key)) for query in queries])
        probabilities=np.exp(scores-max(scores));probabilities/=sum(probabilities)
        expected.append(probabilities)
    np.testing.assert_allclose(fn(queries,keys),expected,rtol=1e-10,atol=1e-12)
    passed('test_raw_key_map')
def test_segmentation_iou(fn):
    assert fn([1,1,0,0],[1,0,1,0])==1/3
    assert fn([0,0],[0,0])==1;assert fn([1,0],[0,1])==0
    passed('test_segmentation_iou')

"""Explicit attention/gradient and independent causal-generation checks."""
import copy
import pytest
import torch as t

from . import text_model as reference


def test_greedy_ar_records_model_entropy(fn=reference.sample_autoregressive):
    class FixedLogits(t.nn.Module):
        def forward(self,tokens,valid,*,causal,cache=None):
            return t.tensor([0.,.1,9.],dtype=t.float64).expand(*tokens.shape,3),None
    _,_,records=fn(FixedLogits(),t.tensor([[0]]),t.tensor([[True]]),output_length=1,
        mask_id=2,eos_id=1,pad_id=0,temperature=0.,use_cache=False)
    probs=t.tensor([0.,.1],dtype=t.float64).softmax(-1)
    expected=-(probs*probs.log()).sum()
    t.testing.assert_close(records[0]['entropy'],expected[None],rtol=1e-12,atol=1e-12)
    print('Greedy AR reports predictive uncertainty rather than point-mass sampling entropy!')


def test_attention_value_and_gradient(cls=reference.TextAttention):
    with t.random.fork_rng():
        t.manual_seed(12)
        module=cls(6,2).double()
    x=t.randn(2,4,6,generator=t.Generator().manual_seed(5),dtype=t.float64,requires_grad=True)
    valid=t.tensor([[True,True,False,True],[True,False,True,True]])
    for causal in (True,False):
        actual,_=module(x,valid,causal=causal)
        q,k,v=module.qkv(x).chunk(3,-1)
        rows=[]
        for b in range(2):
            positions=[]
            for i in range(4):
                head_outputs=[]
                indices=[j for j in range(4) if valid[b,j] and (not causal or j<=i)]
                for h in range(2):
                    sl=slice(3*h,3*h+3)
                    scores=t.stack([(q[b,i,sl]*k[b,j,sl]).sum()/3**.5 for j in indices])
                    values=t.stack([v[b,j,sl] for j in indices])
                    head_outputs.append((scores.softmax(0)[:,None]*values).sum(0))
                positions.append(t.cat(head_outputs))
            rows.append(t.stack(positions))
        expected=module.output(t.stack(rows))
        t.testing.assert_close(actual,expected,rtol=1e-11,atol=1e-11)
        targets=(x,*module.parameters())
        g1=t.autograd.grad(actual.square().sum(),targets,retain_graph=True)
        g2=t.autograd.grad(expected.square().sum(),targets,retain_graph=True)
        for a,e in zip(g1,g2):t.testing.assert_close(a,e,rtol=1e-10,atol=1e-10)
    print('Explicit head-by-head attention values and gradients agree!')


def test_transformer_cache_and_causal_mask(cls=reference.TextTransformer):
    with t.random.fork_rng():
        t.manual_seed(17)
        model=cls(11,width=12,heads=3,layers=2,max_length=12).double().eval()
    tokens=t.tensor([[1,0,2,3,4,5],[1,2,0,0,3,4]])
    valid=t.tensor([[True,False,True,True,True,True],[True,True,False,False,True,True]])
    full,_=model(tokens,valid,causal=True)
    outputs=[];cache=None
    for start,end in ((0,3),(3,4),(4,6)):
        piece,cache=model(tokens[:,start:end],valid[:,:end],causal=True,cache=cache)
        outputs.append(piece)
    t.testing.assert_close(t.cat(outputs,1),full,rtol=1e-11,atol=1e-11)
    changed=tokens.clone(); changed[:,-1]=8
    after,_=model(changed,valid,causal=True)
    t.testing.assert_close(after[:,:-1],full[:,:-1],rtol=0,atol=0)
    bidir,_=model(tokens,valid,causal=False)
    bidir_changed,_=model(changed,valid,causal=False)
    assert not t.allclose(bidir[:,0],bidir_changed[:,0]), 'Causal caching must preserve full-forward values while bidirectional attention can use future tokens. Required condition: not t.allclose(bidir[:, 0], bidir_changed[:, 0])'
    with pytest.raises(ValueError,match='causal'):model(tokens[:,:1],valid,causal=False,cache=cache)
    print('Ragged context masking, multi-token cache parity and future isolation passed!')


def test_ar_loss_shift_and_gradient(fn=reference.autoregressive_target_loss):
    logits=t.randn(2,5,6,generator=t.Generator().manual_seed(10),dtype=t.float64,requires_grad=True)
    tokens=t.tensor([[0,1,2,3,4],[0,1,3,4,2]])
    eligible=t.tensor([[False,False,True,True,True],[False,False,False,True,True]])
    actual=fn(logits,tokens,eligible,5)
    terms=[]
    for row in range(2):
        values=[]
        for j in range(1,5):
            if eligible[row,j]:values.append(t.logsumexp(logits[row,j-1,:5],0)-logits[row,j-1,tokens[row,j]])
        terms.append(t.stack(values).mean())
    expected=t.stack(terms).mean()
    t.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    a=t.autograd.grad(actual,logits,retain_graph=True)[0];e=t.autograd.grad(expected,logits)[0]
    t.testing.assert_close(a,e,rtol=1e-12,atol=1e-12)
    assert a[:,-1].eq(0).all() and a[...,5].eq(0).all(), 'Only logits immediately before eligible targets contribute; future and MASK-logit gradients must be zero. Required condition: a[:, -1].eq(0).all() and a[..., 5].eq(0).all()'
    print('Shifted target-only loss and exact gradient passed!')


def test_ar_cached_uncached_generations(fn=reference.sample_autoregressive):
    with t.random.fork_rng():
        t.manual_seed(11)
        model=reference.TextTransformer(13,width=12,heads=3,layers=2,max_length=12).double().train()
    context=t.tensor([[1,0,0,2],[1,3,0,2]])
    valid=context.ne(0)
    args=dict(output_length=6,mask_id=12,eos_id=11,pad_id=0,temperature=0.)
    cached=fn(model,context,valid,use_cache=True,**args)
    uncached=fn(model,context,valid,use_cache=False,**args)
    assert model.training, 'Cached and uncached generation must agree and restore model mode; caching should reduce token work. Required condition: model.training'
    assert t.equal(cached[0],uncached[0]) and t.equal(cached[1],uncached[1]), 'Cached and uncached generation must agree and restore model mode; caching should reduce token work. Required condition: t.equal(cached[0], uncached[0]) and t.equal(cached[1], uncached[1])'
    assert sum(x['input_tokens'] for x in cached[2])<sum(x['input_tokens'] for x in uncached[2]), "Cached and uncached generation must agree and restore model mode; caching should reduce token work. Required condition: sum((x['input_tokens'] for x in cached[2])) < sum((x['input_tokens'] for x in uncached[2]))"
    print('Actual cached/uncached generated tokens agree, with input work counted!')


def test_ar_individual_eos(fn=reference.sample_autoregressive):
    class DifferentEOS(t.nn.Module):
        def forward(self,tokens,valid,*,causal,cache=None):
            step=valid.shape[1]-2
            logits=t.full((*tokens.shape,6),-100.,dtype=t.float64)
            logits[0,:,(4 if step==0 else 2)]=100.
            logits[1,:,(4 if step==2 else 3)]=100.
            return logits,[True]
    result,lengths,records=fn(DifferentEOS(),t.tensor([[1,2],[1,2]]),t.ones(2,2,dtype=t.bool),
        output_length=5,mask_id=5,eos_id=4,pad_id=0,temperature=0.,use_cache=False)
    assert t.equal(result,t.tensor([[4,0,0,0,0],[3,3,4,0,0]])), 'Each example must stop at its own EOS and retain correct lengths and active-row counts. Required condition: t.equal(result, t.tensor([[4, 0, 0, 0, 0], [3, 3, 4, 0, 0]]))'
    assert t.equal(lengths,t.tensor([1,3])) and len(records)==3, 'Each example must stop at its own EOS and retain correct lengths and active-row counts. Required condition: t.equal(lengths, t.tensor([1, 3])) and len(records) == 3'
    assert t.equal(records[1]['active'],t.tensor([False,True])), "Each example must stop at its own EOS and retain correct lengths and active-row counts. Required condition: t.equal(records[1]['active'], t.tensor([False, True]))"
    print('Per-example EOS and actual early-stop call counts passed!')


def test_timing_path_has_identical_tokens_without_trace_storage():
    from .text_diffusion_methods import sample_text_diffusion
    with t.random.fork_rng():
        t.manual_seed(5)
        model=reference.TextTransformer(13,width=12,heads=3,layers=2,max_length=12).double().eval()
    context=t.tensor([[1,2],[1,3]]);valid=t.ones_like(context,dtype=t.bool)
    args=dict(output_length=6,mask_id=12,eos_id=11,pad_id=0,temperature=0.)
    a=reference.sample_autoregressive(model,context,valid,record_trace=True,**args)
    b=reference.sample_autoregressive(model,context,valid,record_trace=False,**args)
    assert t.equal(a[0],b[0]) and t.equal(a[1],b[1]), 'Removing trace storage must leave sampled tokens, lengths and forward-call counts unchanged. Required condition: t.equal(a[0], b[0]) and t.equal(a[1], b[1])'
    assert all('entropy' not in row and 'active' not in row for row in b[2]), "Removing trace storage must leave sampled tokens, lengths and forward-call counts unchanged. Required condition: all(('entropy' not in row and 'active' not in row for row in b[2]))"
    inputs=t.cat((context,t.zeros(2,4,dtype=t.long)),1);eligible=t.zeros_like(inputs,dtype=t.bool);eligible[:,2:]=True
    def fn(tokens,p):return model(tokens,t.ones_like(tokens,dtype=t.bool),causal=False)[0]
    for sampler in ('ancestral','uniform','confidence'):
        with t.no_grad():
            a=sample_text_diffusion(fn,inputs,eligible,12,steps=4,sampler=sampler,record_trace=True,generator=t.Generator().manual_seed(9))
            b=sample_text_diffusion(fn,inputs,eligible,12,steps=4,sampler=sampler,record_trace=False,generator=t.Generator().manual_seed(9))
        assert t.equal(a[0],b[0]) and b[1] is None, 'Removing trace storage must leave sampled tokens, lengths and forward-call counts unchanged. Required condition: t.equal(a[0], b[0]) and b[1] is None'
        assert len(a[2])==len(b[2]) and all('entropy' not in row for row in b[2]), "Removing trace storage must leave sampled tokens, lengths and forward-call counts unchanged. Required condition: len(a[2]) == len(b[2]) and all(('entropy' not in row for row in b[2]))"
    print('Instrumented and timing-only paths generate identical tokens for AR and all diffusion samplers!')

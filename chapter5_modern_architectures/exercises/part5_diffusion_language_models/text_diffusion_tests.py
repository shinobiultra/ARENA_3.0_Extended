"""Independent small-state oracles for the forthcoming text progression."""
from itertools import product

import pytest
import torch as t

from . import text_diffusion_methods as reference


def test_confidence_and_entropy_survive_greedy_sampling(fn=reference.sample_text_diffusion):
    # Position 1 is uncertain; position 2 is confident. Greedy draws alone
    # are point masses at BOTH positions, so cannot establish this ordering.
    logits=t.tensor([[[0.,0.,0.],[0.,.1,9.],[0.,4.,9.]]],dtype=t.float64)
    def model(tokens,p):return logits.expand(len(tokens),-1,-1)
    inputs=t.tensor([[0,2,2]]);eligible=t.tensor([[False,True,True]])
    _,states,records=fn(model,inputs,eligible,2,steps=2,sampler='confidence',temperatures=(0.,0.))
    assert t.equal(states[1,0],t.tensor([0,2,1])), 'Reveal the more confident right-hand token first, even with greedy token draws.'
    expected_probs=t.tensor([[0.,.1],[0.,4.]],dtype=t.float64).softmax(-1)
    expected=-(expected_probs*expected_probs.log()).sum(-1)
    t.testing.assert_close(records[0]['entropy'][0],expected,rtol=1e-12,atol=1e-12)
    assert expected[0]>expected[1]>0, 'Model uncertainty remains nonzero and larger at the uncertain token.'
    print('Greedy sampling preserves model confidence ordering and predictive entropy!')


def test_target_masking(fn=reference.mask_targets):
    clean=t.tensor([[0,1,2,0]]).expand(40000,-1).clone()
    eligible=t.tensor([[False,True,True,False]]).expand_as(clean)
    probability=t.full((len(clean),),.35)
    out,mask=fn(clean,eligible,probability,3,generator=t.Generator().manual_seed(10))
    assert t.equal(out[~eligible],clean[~eligible]), 'Forward corruption must preserve conditioning, follow the chosen probability and reproduce seeded masks. Required condition: t.equal(out[~eligible], clean[~eligible])'
    assert not mask[~eligible].any(), 'Forward corruption must preserve conditioning, follow the chosen probability and reproduce seeded masks. Required condition: not mask[~eligible].any()'
    assert abs(mask[eligible].double().mean()-.35)<.01, 'Forward corruption must preserve conditioning, follow the chosen probability and reproduce seeded masks. Required condition: abs(mask[eligible].double().mean() - 0.35) < 0.01'
    again=fn(clean,eligible,probability,3,generator=t.Generator().manual_seed(10))
    assert t.equal(out,again[0]) and t.equal(mask,again[1]), 'Forward corruption must preserve conditioning, follow the chosen probability and reproduce seeded masks. Required condition: t.equal(out, again[0]) and t.equal(mask, again[1])'
    for p in (0.,1.):
        x,m=fn(clean,eligible,t.full_like(probability,p),3)
        assert t.equal(m,eligible if p else t.zeros_like(eligible)), 'Forward corruption must preserve conditioning, follow the chosen probability and reproduce seeded masks. Required condition: t.equal(m, eligible if p else t.zeros_like(eligible))'
    print('Conditioning identity, empirical noising and seed checks passed!')


def test_finite_loss_exact_bound_and_gradient(fn=reference.finite_diffusion_loss):
    # Two clean tokens, three vocabulary symbols including MASK=2, T=3.
    # Explicitly enumerate the joint forward chain x_s -> x_t and compute
    # conditional KL/reconstruction; do not use the proposed T/k weighting.
    clean=t.tensor([[0,1]],dtype=t.long); eligible=t.ones_like(clean,dtype=t.bool)
    masks=list(product((False,True),repeat=2))
    logits=t.randn(4,2,3,generator=t.Generator().manual_seed(22),dtype=t.float64,requires_grad=True)
    total=3; estimator=logits.sum()*0; oracle=logits.sum()*0
    for k in range(1,total+1):
        pt=k/total; ps=(k-1)/total
        for state_id,mask in enumerate(masks):
            current=clean.clone()
            current[0,t.tensor(mask)]=2
            state_probability=pt**sum(mask)*(1-pt)**(2-sum(mask))
            estimator=estimator+state_probability/total*fn(logits[state_id:state_id+1],clean,current,eligible,t.tensor([k]),total,2)
            if not state_probability:
                continue
            predicted=logits[state_id,:,:2].exp()
            predicted=predicted/predicted.sum(-1,keepdim=True)
            for position in range(2):
                x0=int(clean[0,position]); xt=int(current[0,position])
                if xt!=2:
                    continue
                # Forward joint masses P(x_s=x0,x_t=MASK) and P(MASK,MASK).
                joints=[(1-ps)*(pt-ps)/(1-ps), ps]
                conditional=[x/sum(joints) for x in joints]
                model_masses=[predicted[position,x0]*(pt-ps)/pt,t.tensor(ps/pt,dtype=t.float64)]
                kl=logits.sum()*0
                for q,p in zip(conditional,model_masses):
                    if q:kl=kl+q*(t.tensor(q,dtype=t.float64).log()-p.log())
                oracle=oracle+state_probability*kl/2
    t.testing.assert_close(estimator,oracle,rtol=1e-12,atol=1e-12)
    actual=t.autograd.grad(estimator,logits,retain_graph=True)[0]
    expected=t.autograd.grad(oracle,logits)[0]
    t.testing.assert_close(actual,expected,rtol=1e-12,atol=1e-12)
    assert t.equal(actual[...,2],t.zeros_like(actual[...,2])), 'The weighted objective must match the enumerated forward-chain bound and must not train a MASK prediction. Required condition: t.equal(actual[..., 2], t.zeros_like(actual[..., 2]))'
    print('Enumerated forward-chain bound and exact gradient agree!')


def test_finite_loss_zero_masks_and_denominator(fn=reference.finite_diffusion_loss):
    logits=t.zeros(1,4,4,dtype=t.float64,requires_grad=True)
    clean=t.tensor([[0,1,1,1]]); eligible=t.tensor([[False,True,True,True]])
    unmasked=fn(logits,clean,clean,eligible,t.tensor([2]),4,3)
    assert unmasked==0, 'Normalize by every eligible target; a zero-mask sample must have zero loss and zero gradients. Required condition: unmasked == 0'
    assert t.equal(t.autograd.grad(unmasked,logits,retain_graph=True)[0],t.zeros_like(logits)), 'Normalize by every eligible target; a zero-mask sample must have zero loss and zero gradients. Required condition: t.equal(t.autograd.grad(unmasked, logits, retain_graph=True)[0], t.zeros_like(logits))'
    corrupted=clean.clone();corrupted[0,1]=3
    loss=fn(logits,clean,corrupted,eligible,t.tensor([2]),4,3)
    t.testing.assert_close(loss,2*t.tensor(3.,dtype=t.float64).log()/3)
    with pytest.raises(ValueError,match='conditioning'):
        fn(logits,clean,corrupted.masked_fill(~eligible,3),eligible,t.tensor([2]),4,3)
    print('Zero-mask gradients and all-target normalization passed!')


def test_clean_distribution(fn=reference.clean_prediction_probs):
    logits=t.tensor([[[1.,3.,1e6]]],dtype=t.float64,requires_grad=True)
    original=logits.detach().clone()
    p=fn(logits,2,2.)
    expected=t.tensor([[[1/(1+t.exp(t.tensor(1.))),1/(1+t.exp(t.tensor(-1.))),0]]],dtype=t.float64)
    t.testing.assert_close(p,expected,rtol=1e-7,atol=1e-7)
    assert t.equal(fn(logits,2,0.),t.tensor([[[0.,1.,0.]]])), 'Clean prediction excludes MASK, preserves logits and applies the declared temperature. Required condition: t.equal(fn(logits, 2, 0.0), t.tensor([[[0.0, 1.0, 0.0]]]))'
    assert t.equal(logits.detach(),original), 'Clean prediction excludes MASK, preserves logits and applies the declared temperature. Required condition: t.equal(logits.detach(), original)'
    with pytest.raises(ValueError):fn(logits,2,-1.)
    print('MASK exclusion, temperature and greedy prediction passed!')


def test_reverse_kernel_matches_bayes(fn=reference.absorbing_reverse_probs):
    current=t.tensor([[2,1]])
    clean=t.tensor([[[.3,.7,0],[.8,.2,0]]],dtype=t.float64)
    pt,ps=.75,.25
    probs=fn(clean,current,pt,ps,2)
    # Prior mixture over x0 combined with the exact absorbing forward joint.
    joint=t.tensor([.3*(pt-ps),.7*(pt-ps),ps],dtype=t.float64)
    t.testing.assert_close(probs[0,0],joint/joint.sum(),rtol=1e-12,atol=1e-12)
    t.testing.assert_close(probs[0,1],t.tensor([0.,1.,0.],dtype=t.float64))
    t.testing.assert_close(probs.sum(-1),t.ones_like(current,dtype=t.float64))
    assert fn(clean,current,pt,0.,2)[0,0,2]==0, 'Reverse probabilities must obey Bayes normalization, preserve visible tokens and reveal all masks at zero noise. Required condition: fn(clean, current, pt, 0.0, 2)[0, 0, 2] == 0'
    assert fn(clean,current,pt,pt,2)[0,0,2]==1, 'Reverse probabilities must obey Bayes normalization, preserve visible tokens and reveal all masks at zero noise. Required condition: fn(clean, current, pt, pt, 2)[0, 0, 2] == 1'
    with pytest.raises(ValueError):fn(clean,current,0.,0.,2)
    print('Bayes reverse kernel, normalization and carry-over passed!')


def test_diffusion_sampler_context_and_block_semantics(fn=reference.sample_text_diffusion):
    conditioning=t.tensor([[1,0,0,0,0,0],[2,0,0,0,0,0]])
    eligible=t.tensor([[False,False,True,True,True,True]]).expand_as(conditioning)
    seen=[]
    def oracle(tokens,p):
        seen.append(tokens.clone())
        result=t.full((*tokens.shape,4),-30.,dtype=t.float64)
        answer=tokens[:,0,None].expand_as(tokens)
        return result.scatter(-1,answer[...,None],30.)
    for sampler in ('ancestral','uniform','confidence'):
        seen.clear()
        result,states,records=fn(oracle,conditioning,eligible,3,steps=4,sampler=sampler,
            temperatures=(0.,0.),block_size=2,generator=t.Generator().manual_seed(4))
        assert t.equal(result[:,2:],conditioning[:,:1].expand(-1,4)), 'The sampler must preserve conditioning and completed blocks, finish every target, and never revise ancestral reveals. Required condition: t.equal(result[:, 2:], conditioning[:, :1].expand(-1, 4))'
        assert t.equal(states[:,:,:2],conditioning[None,:,:2].expand(len(states),-1,-1)), 'The sampler must preserve conditioning and completed blocks, finish every target, and never revise ancestral reveals. Required condition: t.equal(states[:, :, :2], conditioning[None, :, :2].expand(len(states), -1, -1))'
        assert len(records)==8 and states.shape==(9,2,6), 'The sampler must preserve conditioning and completed blocks, finish every target, and never revise ancestral reveals. Required condition: len(records) == 8 and states.shape == (9, 2, 6)'
        assert all(t.equal(state[:,2:4],result[:,2:4]) for state in seen[4:]), 'The sampler must preserve conditioning and completed blocks, finish every target, and never revise ancestral reveals. Required condition: all((t.equal(state[:, 2:4], result[:, 2:4]) for state in seen[4:]))'
        assert all(state[:,4:].eq(3).all() for state in seen[:4]), 'The sampler must preserve conditioning and completed blocks, finish every target, and never revise ancestral reveals. Required condition: all((state[:, 4:].eq(3).all() for state in seen[:4]))'
        if sampler=='ancestral':
            assert not ((states[:-1]!=3)&(states[1:]!=states[:-1])).any(), 'The sampler must preserve conditioning and completed blocks, finish every target, and never revise ancestral reveals. Required condition: not ((states[:-1] != 3) & (states[1:] != states[:-1])).any()'
    print('Live sampler context, completed-block identity and ancestral monotonicity passed!')


def test_sampler_temperature_schedule_and_seed(fn=reference.sample_text_diffusion):
    context=t.tensor([[0,0,0,0]]).expand(12,-1)
    eligible=t.tensor([[False,True,True,True]]).expand_as(context)
    def model(tokens,p):return t.tensor([0.,1.,2.,3.,100.]).expand(*tokens.shape,5)
    for sampler in ('ancestral','uniform','confidence'):
        args=dict(steps=5,sampler=sampler,temperatures=(1.,.2))
        a=fn(model,context,eligible,4,generator=t.Generator().manual_seed(11),**args)
        b=fn(model,context,eligible,4,generator=t.Generator().manual_seed(11),**args)
        assert t.equal(a[1],b[1]) and not a[0][:,1:].eq(4).any(), 'A fixed seed must reproduce the entire trajectory while applying both schedule endpoints. Required condition: t.equal(a[1], b[1]) and (not a[0][:, 1:].eq(4).any())'
        assert a[2][0]['temperature']==1 and abs(a[2][-1]['temperature']-.2)<1e-12, "A fixed seed must reproduce the entire trajectory while applying both schedule endpoints. Required condition: a[2][0]['temperature'] == 1 and abs(a[2][-1]['temperature'] - 0.2) < 1e-12"
    print('Actual temperature schedule and same-seed state parity passed!')


def test_stable_commitment_revisions(fn=reference.stable_token_commitment):
    states=t.tensor([[[3,0,3]],[[1,1,3]],[[2,0,3]],[[2,0,3]]])
    assert t.equal(fn(states,3),t.tensor([[2,2,-1]])), 'Commitment is the first permanent match to the final token; an unrevealed final MASK has no commitment. Required condition: t.equal(fn(states, 3), t.tensor([[2, 2, -1]]))'
    assert t.equal(fn(t.tensor([[[1]]]),3),t.tensor([[0]])), 'Commitment is the first permanent match to the final token; an unrevealed final MASK has no commitment. Required condition: t.equal(fn(t.tensor([[[1]]]), 3), t.tensor([[0]]))'
    print('Stable commitment distinguishes initial reveals from later revisions!')


def test_entropy_and_text_metrics(entropy_fn=reference.entropy_from_probs, edit_fn=reference.levenshtein_distance, grammar_fn=reference.balanced_parentheses):
    p=t.tensor([[0.,.5,.5],[0.,1.,0.]],dtype=t.float64)
    t.testing.assert_close(entropy_fn(p),t.tensor([2.,1.],dtype=t.float64).log())
    for left,right,expected in [('', 'ab',2),('kitten','sitting',3),('abc','abc',0),([1,2],[2,1],2)]:
        assert edit_fn(left,right)==expected, 'Unit insert, delete and substitute costs must match the exact examples.'
    for text in ('()', '(())', '()(()())'):assert grammar_fn(text), 'Accept these nonempty balanced strings.'
    for text in ('',')(', '(()', '(a)'):assert not grammar_fn(text), 'Reject empty, early-close, unfinished and non-parenthesis strings.'
    print('Zero-probability entropy, edit distance and strict grammar checks passed!')


def test_remasking_never_uses_unsupervised_visible_token_logits(fn=reference.sample_text_diffusion):
    conditioning=t.zeros(12,7,dtype=t.long)
    eligible=t.ones_like(conditioning,dtype=t.bool);eligible[:,0]=False
    def adversarial(tokens,p):
        # MASK predictions are perfect. Visible-position predictions are wrong:
        # the masked-only objective never supervises those output logits.
        prediction=t.where(tokens.eq(3),1,2)
        logits=t.full((*tokens.shape,4),-100.)
        return logits.scatter(-1,prediction[...,None],100.)
    for sampler in ('ancestral','confidence','uniform'):
        output,states,_=fn(adversarial,conditioning,eligible,3,steps=6,sampler=sampler,
            temperatures=(0.,0.),generator=t.Generator().manual_seed(12))
        assert output[:,1:].eq(1).all(), 'Use predictions only at masked positions; carry previously revealed values unchanged.'
        assert not ((states[:-1]!=3)&(states[1:]!=states[:-1])).any(), 'A wrong unconstrained logit at a visible token must not rewrite the generated state.'
    print('Adversarial visible-position predictions cannot rewrite any committed token!')

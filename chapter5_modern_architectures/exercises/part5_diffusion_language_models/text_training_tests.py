"""Independent optimizer updates and matched-control training oracles."""
from copy import deepcopy

import pytest
import torch as t

from . import text_training as training
from .text_model import TextTransformer


def small_data():
    sequence=t.tensor([[1,2,3,4,5,6],[1,3,2,6,5,4],[1,4,2,3,6,5],[1,5,2,4,3,6]])
    eligible=t.tensor([[False,False,False,True,True,True]]).expand_as(sequence).clone()
    def bundle(split):return {'sequence':sequence.clone(),'eligible':eligible.clone(),'valid':t.ones_like(eligible),
        'task_ids':t.tensor([0,0,1,1]),'row_ids':[f'{split}-{i}' for i in range(4)],'tasks':['a','b'],'split':split}
    return bundle('train'),bundle('development')


def factory():return TextTransformer(9,width=8,heads=2,layers=1,max_length=8)


def test_shuffle_preserves_context_and_task_marginals(fn=training.permute_targets_within_task):
    train,_=small_data()
    actual,permutation=fn(train['sequence'],train['eligible'],train['task_ids'],generator=t.Generator().manual_seed(3))
    assert sorted(permutation.tolist())==list(range(4)), 'Keep each original target exactly once, including permutation fixed points.'
    assert t.equal(train['task_ids'][permutation],train['task_ids']), 'Targets must remain inside their original task.'
    assert t.equal(actual[:,:3],train['sequence'][:,:3]), 'The shuffled control must preserve each input context.'
    assert t.equal(actual[:,3:],train['sequence'][permutation,3:]), 'Move complete target strings, not individual token marginals.'
    print('Within-task permutations preserve contexts and the complete target multiset!')


def test_fixed_noise_is_independent_of_training_rng(fn=training.fixed_development_noise):
    _,dev=small_data()
    before=t.random.get_rng_state().clone()
    actual=fn(dev,total_steps=4,mask_id=8,repeats=3)
    assert t.equal(t.random.get_rng_state(),before), 'Fixed development noise must not consume the global training RNG.'
    rng=t.Generator().manual_seed(1729)
    expected_steps=t.randint(1,5,(12,),generator=rng)
    draws=t.rand((12,6),generator=rng)
    clean=dev['sequence'].repeat(3,1)
    expected=clean.masked_fill(dev['eligible'].repeat(3,1)&(draws<expected_steps[:,None]/4),8)
    assert t.equal(actual['steps'],expected_steps) and t.equal(actual['noisy'],expected), 'Use the declared independent seed, all three repeats, uniform discrete times and target-only masking.'
    print('Fixed development masks match an independent RNG oracle without touching training randomness!')


def test_training_matches_two_explicit_adamw_updates(fn=training.fit_text_model):
    train,dev=small_data()
    for kind in ('autoregressive','diffusion'):
        # Constant key-bias shifts are mathematically invisible to attention.
        # Float32 roundoff leaves tiny gradients that Adam's epsilon can amplify;
        # use float64 to compare algebraically equivalent loss expressions.
        actual,record=fn(lambda:factory().double(),train,dev,kind=kind,shuffled=False,seed=2,mask_id=8,total_steps=4,
            updates=2,batch_size=4,development_every=2,learning_rate=.003,weight_decay=.01)
        with t.random.fork_rng():
            t.manual_seed(2); oracle=factory().double()
        first=[t.zeros_like(p) for p in oracle.parameters()]
        second=[t.zeros_like(p) for p in oracle.parameters()]
        for step,batch in enumerate(record['preview_batches'],1):
            indices=t.tensor(batch['indices']);clean=train['sequence'][indices]
            noisy=t.tensor(batch['noisy']);valid=train['valid'][indices]
            logits,_=oracle(noisy,valid,causal=kind=='autoregressive')
            terms=[]
            for b in range(4):
                losses=[]
                for pos in (3,4,5):
                    source=logits[b,pos-1 if kind=='autoregressive' else pos,:8]
                    nll=t.logsumexp(source,0)-source[clean[b,pos]]
                    if kind=='diffusion':nll=nll*(4/batch['steps'][b] if noisy[b,pos]==8 else 0.)
                    losses.append(nll)
                terms.append(t.stack(losses).mean())
            loss=t.stack(terms).mean()
            gradients=t.autograd.grad(loss,tuple(oracle.parameters()))
            norm=t.stack([g.norm(2) for g in gradients]).norm(2)
            factor=(1/(norm+1e-6)).clamp(max=1.)
            with t.no_grad():
                for p,g,m,v in zip(oracle.parameters(),gradients,first,second):
                    g=g*factor
                    m.mul_(.9).add_(g,alpha=.1)
                    v.mul_(.999).addcmul_(g,g,value=.001)
                    p.mul_(1-.003*.01)
                    p.addcdiv_(m,v.sqrt()/(1-.999**step)**.5+1e-8,value=-.003/(1-.9**step))
        assert record['selected_step']==2, "The chosen two-step checkpoint must match the independently derived clipped AdamW update. Required condition: record['selected_step'] == 2"
        for p,q in zip(actual.parameters(),oracle.parameters()):
            t.testing.assert_close(p,q,rtol=1e-9,atol=1e-10)
    print('Both objectives match two independently derived clipped AdamW updates!')


def test_matched_initialization_batches_and_shuffle(fn=training.fit_text_model):
    train,dev=small_data();records=[]
    for kind in ('autoregressive','diffusion'):
        for shuffled in (False,True):
            _,record=fn(factory,train,dev,kind=kind,shuffled=shuffled,seed=2,mask_id=8,
                updates=2,batch_size=4,development_every=2)
            records.append(record)
            permutation=t.tensor(record['target_permutation'])
            assert t.equal(permutation.sort().values,t.arange(4)), 'All four conditions must share initial weights and batches, with the same valid within-task control permutation. Required condition: t.equal(permutation.sort().values, t.arange(4))'
            assert t.equal(train['task_ids'][permutation],train['task_ids']), "All four conditions must share initial weights and batches, with the same valid within-task control permutation. Required condition: t.equal(train['task_ids'][permutation], train['task_ids'])"
    assert len({r['initial_parameters_sha256'] for r in records})==1, "All four conditions must share initial weights and batches, with the same valid within-task control permutation. Required condition: len({r['initial_parameters_sha256'] for r in records}) == 1"
    assert len({r['batch_indices_sha256'] for r in records})==1, "All four conditions must share initial weights and batches, with the same valid within-task control permutation. Required condition: len({r['batch_indices_sha256'] for r in records}) == 1"
    assert len({r['development_noisy_sha256'] for r in records})==1, "All four conditions must share initial weights and batches, with the same valid within-task control permutation. Required condition: len({r['development_noisy_sha256'] for r in records}) == 1"
    assert records[1]['target_permutation']==records[3]['target_permutation'], "All four conditions must share initial weights and batches, with the same valid within-task control permutation. Required condition: records[1]['target_permutation'] == records[3]['target_permutation']"
    assert records[1]['target_permutation']!=[0,1,2,3], "All four conditions must share initial weights and batches, with the same valid within-task control permutation. Required condition: records[1]['target_permutation'] != [0, 1, 2, 3]"
    print('Four matched conditions share weights, batches, fixed dev noise and within-task permutations!')


def test_final_data_cannot_select_checkpoints(fn=training.fit_text_model):
    train,dev=small_data();dev['split']='final'
    with pytest.raises(ValueError,match='splits only'):
        fn(factory,train,dev,kind='diffusion',shuffled=False,seed=2,mask_id=8,updates=2)
    print('The checkpoint-selection API rejects final-evaluation rows!')

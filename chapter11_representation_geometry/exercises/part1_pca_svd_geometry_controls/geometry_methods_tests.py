"""Numerical oracles for the rewritten geometry learner, independent of plots."""
from types import SimpleNamespace

import pytest
import torch as t
from sklearn.manifold import trustworthiness

from . import geometry_methods as m


def test_pca_fits_only_training_and_preserves_total_variance(methods=m):
    x=t.tensor([[3.,0.,0.],[-3.,0.,0.],[0.,1.,0.],[0.,-1.,0.]],dtype=t.float64)
    mean,basis,ratio=methods.fit_pca(x,2)
    t.testing.assert_close(mean,t.zeros(3,dtype=x.dtype))
    t.testing.assert_close(basis@basis.T,t.diag(t.tensor([1.,1.,0.],dtype=x.dtype)))
    t.testing.assert_close(ratio,t.tensor([.9,.1],dtype=x.dtype))
    # A test-only direction must not rotate a train-fitted basis.
    projected=(t.tensor([[0.,0.,1e9]],dtype=x.dtype)-mean)@basis
    t.testing.assert_close(projected,t.zeros(1,2,dtype=x.dtype))
    with pytest.raises(ValueError,match='Constant'):methods.fit_pca(t.ones(4,3),2)
    print('PCA checks passed: training-only fitting and total explained variance agree.')



def test_ridge_has_unpenalized_intercept_and_known_shrinkage(methods=m):
    x=t.tensor([[-1.],[1.]],dtype=t.float64);y=3*x+7
    mean,w,bias=methods.fit_ridge_coordinates(x,y,ridge=2.)
    t.testing.assert_close(w,t.tensor([[1.5]],dtype=x.dtype))
    t.testing.assert_close(bias,t.tensor([7.],dtype=x.dtype))
    shifted=methods.fit_ridge_coordinates(x+100,y+42,ridge=2.)
    t.testing.assert_close(shifted[1],w)
    t.testing.assert_close(shifted[0],mean+100)
    t.testing.assert_close(shifted[2],bias+42)
    assert float(methods.coordinate_r2(y,t.ones_like(y)*7))==0., 'The mean-only baseline defines zero R².'
    assert methods.coordinate_r2(y,t.zeros_like(y))<0, 'A bad predictor must retain negative R².'
    print('Ridge checks passed: the intercept stays unpenalized and slopes shrink correctly.')



def test_template_centering_is_explicitly_transductive(methods=m):
    ring=methods.circular_coordinates(t.arange(7,dtype=t.float64),7)
    offsets=t.tensor([[10.,30.],[-70.,4.]],dtype=t.float64)
    grid=ring[None]+offsets[:,None]
    t.testing.assert_close(methods.center_template_grid(grid),ring.expand(2,-1,-1))
    incomplete=methods.center_template_grid(grid[:,:3])
    assert not t.allclose(incomplete[0],ring[:3]), 'Batch composition changes transductive template centering.'
    print('Template centering checks passed: balanced batch offsets are removed.')



def test_angles_wrap_around_and_zero_abstains(methods=m):
    labels=t.arange(12,dtype=t.float64)
    coords=methods.circular_coordinates(labels,12)
    assert t.equal(methods.angular_predictions(coords*9,12),labels.long()), 'Positive radius must not change phase decoding.'
    assert methods.angular_predictions(t.tensor([[1.,-1e-6]]),12).item()==0, 'December-to-January wraparound is continuous.'
    assert methods.angular_predictions(t.zeros(1,2),12).item()==-1, 'Undefined angles must not silently count as class zero.'
    print('Angle checks passed: wraparound, radius invariance and zero-vector abstention agree.')



def test_calendar_order_null_is_distinct_from_class_identification(methods=m):
    ring=methods.circular_coordinates(t.arange(7,dtype=t.float64),7)
    permutation=t.tensor([0,3,1,5,2,6,4])
    assert methods.cyclic_distance_agreement(ring)>.999999, 'Exact chronological circle distances should agree perfectly with calendar chord distances.'
    assert methods.cyclic_distance_agreement(ring[permutation])<.5, 'A wrong calendar order must destroy a perfect ring match.'
    labels=t.arange(7)
    assert t.equal(methods.knn_predictions(ring[permutation],labels,ring[permutation],1),labels), 'All lexical classes remain perfectly distinguishable after relabeling.'
    orders=t.stack([t.arange(7),permutation])
    observed,null,p=methods.cyclic_permutation_test(ring,orders)
    assert null.shape==(2,) and abs(float(p)-2/3)<1e-6, 'The +1 permutation p-value includes tied perfect rotations.'
    with pytest.raises(ValueError,match='permute'):methods.cyclic_permutation_test(ring,t.zeros(2,7,dtype=t.long))
    print('Calendar-order checks passed: identity and chronological geometry are distinct.')



def test_neighborhood_ranks_match_independent_library_and_duality(methods=m):
    gen=t.Generator().manual_seed(1141)
    high=t.randn(31,5,generator=gen,dtype=t.float64)
    low=t.randn(31,2,generator=gen,dtype=t.float64)
    trust,continuity=methods.neighborhood_scores(high,low,4)
    expected_trust=trustworthiness(high.numpy(),low.numpy(),n_neighbors=4)
    expected_continuity=trustworthiness(low.numpy(),high.numpy(),n_neighbors=4)
    assert abs(float(trust)-expected_trust)<1e-7, 'Trustworthiness penalizes introduced neighbors by original-space rank.'
    assert abs(float(continuity)-expected_continuity)<1e-7, 'Continuity reverses the missing-neighbor question.'
    identity=methods.neighborhood_scores(high,high,4)
    assert tuple(float(x) for x in identity)==(1.,1.), 'An unchanged embedding must preserve every neighborhood in both directions.'
    with pytest.raises(ValueError,match='n/2'):methods.neighborhood_scores(high,low,16)
    print('Neighborhood checks passed: trustworthiness and continuity match independent rank oracles.')



def test_procrustes_recovers_rotation_reflection_and_translation(methods=m):
    x=t.tensor([[0.,0.],[1.,0.],[0.,2.],[-2.,3.]],dtype=t.float64)
    rotation=t.tensor([[0.,1.],[1.,0.]],dtype=x.dtype)
    y=x@rotation+t.tensor([42.,-17.],dtype=x.dtype)
    xm,ym,r=methods.fit_procrustes(x,y)
    t.testing.assert_close(r,rotation)
    heldout=t.tensor([[3.,-9.]],dtype=x.dtype)
    t.testing.assert_close((heldout-xm)@r+ym,heldout@rotation+t.tensor([42.,-17.],dtype=x.dtype))
    print('Procrustes checks passed: translations, rotations and reflections are recovered.')



def test_minimum_norm_coordinate_edit_matches_constraint_solution(methods=m):
    gen=t.Generator().manual_seed(9)
    basis=t.linalg.qr(t.randn(7,4,generator=gen,dtype=t.float64)).Q
    weights=t.tensor([[2.,0.],[1.,1.],[0.,3.],[1.,-1.]],dtype=t.float64)
    x=t.randn(6,7,generator=gen,dtype=t.float64);mean=t.randn(7,generator=gen,dtype=x.dtype)
    bias=t.tensor([.7,-1.],dtype=x.dtype);target=t.randn(6,2,generator=gen,dtype=x.dtype)
    edited=methods.replace_coordinates(x,mean,basis,weights,bias,target)
    t.testing.assert_close((edited-mean)@basis@weights+bias,target)
    # Independent full-space KKT solution of min ||delta||² under delta @ A = residual.
    a=basis@weights;residual=target-((x-mean)@a+bias)
    oracle=residual@t.linalg.solve(a.T@a,a.T)
    t.testing.assert_close(edited-x,oracle)
    complement=t.eye(7,dtype=x.dtype)-a@t.linalg.solve(a.T@a,a.T)
    t.testing.assert_close((edited-x)@complement,t.zeros_like(x),atol=1e-12,rtol=0)
    same=methods.replace_coordinates(x,mean,basis,weights,bias,(x-mean)@a+bias)
    t.testing.assert_close(same,x)
    with pytest.raises(ValueError,match='controllable'):methods.replace_coordinates(x,mean,basis,t.ones_like(weights),bias,target)
    with pytest.raises(ValueError,match='orthonormal'):methods.replace_coordinates(x,mean,basis*2,weights,bias,target)
    print('Coordinate-edit checks passed: constraints hold with the minimum possible edit norm.')



def test_donor_subspace_does_not_copy_complement(methods=m):
    basis=t.eye(4,dtype=t.float64)[:,:2]
    x=t.tensor([[1.,2.,30.,40.]],dtype=t.float64);donor=t.tensor([[8.,9.,-3.,-4.]],dtype=x.dtype)
    t.testing.assert_close(methods.replace_subspace(x,donor,basis),t.tensor([[8.,9.,30.,40.]],dtype=x.dtype))
    print('Donor-subspace checks passed: the complement is preserved.')



def test_exact_ring_plane_changes_answer_and_preserves_nuisance(methods=m):
    gen=t.Generator().manual_seed(82);rotation=t.linalg.qr(t.randn(6,6,generator=gen,dtype=t.float64)).Q
    labels=t.arange(7,dtype=t.float64);coordinates=methods.circular_coordinates(labels,7)
    # Exact residual-stream model: phase determines answer; an independent coordinate is a second task.
    h=t.cat([coordinates,t.zeros(7,4,dtype=t.float64)],-1)@rotation.T
    mean,basis,_=methods.fit_pca(h,2);z=(h-mean)@basis
    zm,w,bias=methods.fit_ridge_coordinates(z,coordinates)
    probe_bias=bias-zm@w
    receiver=h+19*rotation[:,2]
    desired=methods.circular_coordinates((labels+3)%7,7)
    edited=methods.replace_coordinates(receiver,mean,basis,w,probe_bias,desired)
    logits=(edited@rotation[:,:2])@coordinates.T
    assert t.equal(logits.argmax(-1),(labels.long()+3)%7), 'Replacing the learned plane must transfer every circular class.'
    t.testing.assert_close(edited@rotation[:,2],t.full((7,),19.,dtype=t.float64))
    random=methods.replace_subspace(receiver,h.roll(3,0),rotation[:,2:4])
    assert t.equal((random@rotation[:,:2]@coordinates.T).argmax(-1),labels.long()), 'An orthogonal rank-two control must not transfer calendar identity.'
    print('Exact-model checks passed: the calendar answer changes and the unrelated variable stays fixed.')



class TinyNativeModel(t.nn.Module):
    def __init__(self,fail=False):
        super().__init__();self.block=t.nn.Identity();self.fail=fail
    def forward(self,input_ids,attention_mask,use_cache=False):
        hidden=t.nn.functional.one_hot(input_ids,4).float();hidden=self.block(hidden)
        if self.fail:raise RuntimeError('deliberate forward failure')
        return SimpleNamespace(logits=hidden)


def test_semantic_positions_respect_both_padding_sides_and_cleanup_hooks(methods=m):
    model=TinyNativeModel();encoded={'input_ids':t.tensor([[0,1,2],[3,1,0]]),'attention_mask':t.tensor([[0,1,1],[1,1,0]])}
    positions=t.tensor([2,1]);replacement=t.tensor([[9.,8.,7.,6.],[5.,4.,3.,2.]])
    result=methods.forward_with_residual_replacement(model,model.block,encoded,positions,replacement)
    t.testing.assert_close(result[t.arange(2),positions],replacement)
    t.testing.assert_close(result[:,0],t.nn.functional.one_hot(encoded['input_ids'][:,0],4).float())
    assert not model.block._forward_pre_hooks, 'Successful calls must remove their hooks.'
    with pytest.raises(ValueError,match='padding'):
        methods.forward_with_residual_replacement(model,model.block,encoded,t.tensor([0,2]),replacement)
    assert not model.block._forward_pre_hooks, 'Validation failures must remove their hooks too.'
    model.fail=True
    with pytest.raises(RuntimeError,match='deliberate'):
        methods.forward_with_residual_replacement(model,model.block,encoded,positions,replacement)
    assert not model.block._forward_pre_hooks, 'A model exception must not leave a persistent intervention behind.'
    print('Native-intervention checks passed: semantic positions, padding and hook cleanup agree.')



def test_mean_ablation_and_literal_author_coordinates_answer_different_questions(methods=m):
    mean=t.tensor([10.,20.,30.,40.],dtype=t.float64)
    basis=t.eye(4,dtype=t.float64)[:,:3]
    weights=t.tensor([[2.,0.],[0.,3.],[0.,0.]],dtype=t.float64)
    bias=t.tensor([1.,-2.],dtype=t.float64);target=t.tensor([[5.,7.]],dtype=t.float64)
    centered=methods.mean_ablate_coordinates(mean,basis,weights,bias,target)
    t.testing.assert_close(centered,t.tensor([[12.,23.,30.,40.]],dtype=t.float64))
    raw=methods.author_raw_circle_replacement(mean,basis,weights,target)
    t.testing.assert_close(raw,t.tensor([[2.5,7/3,30.,40.]],dtype=t.float64))
    # The literal author QR formula removes the mean's raw-plane component.
    t.testing.assert_close(raw@basis@weights,target)
    t.testing.assert_close((centered-mean)@basis@weights+bias,target)
    assert not t.allclose(raw,centered), 'Do not silently equate raw and affine-centered coordinate clamps.'
    receiver=mean+t.tensor([[0.,0.,99.,-25.]],dtype=t.float64)
    pure=methods.replace_coordinates(receiver,mean,basis,weights,bias,target)
    t.testing.assert_close(pure[:,2:],receiver[:,2:])
    t.testing.assert_close(centered[:,2:],mean[None,2:])
    print('Mean-ablation checks passed: raw and affine coordinate conventions are distinguished.')


def test_native_capture_preserves_baseline_and_cleans_up_on_failure(methods=m):
    model=TinyNativeModel();encoded={'input_ids':t.tensor([[0,1,2],[3,1,0]]),'attention_mask':t.tensor([[0,1,1],[1,1,0]])};positions=t.tensor([2,1])
    baseline=model(**encoded).logits
    logits,cached=methods.capture_residual_positions(model,[model.block],encoded,positions)
    t.testing.assert_close(logits,baseline)
    t.testing.assert_close(cached[:,0],baseline[t.arange(2),positions])
    assert cached.shape==(2,1,4) and not model.block._forward_pre_hooks, 'Caching must preserve layer/row alignment and remove hooks.'
    with pytest.raises(ValueError,match='padding'):
        methods.capture_residual_positions(model,[model.block],encoded,t.tensor([0,2]))
    assert not model.block._forward_pre_hooks, 'Failed capture must remove its own hook before the next model call.'
    print('Native-capture checks passed: baseline outputs and hook cleanup are preserved.')


def test_token_offsets_do_not_assume_identical_prompt_lengths(methods=m):
    offsets=t.tensor([[[0,0],[0,3],[4,10],[11,13]],[[0,4],[5,12],[13,15],[0,0]]])
    mask=t.tensor([[0,1,1,1],[1,1,1,0]])
    assert t.equal(methods.positions_from_spans(offsets,mask,t.tensor([[4,10],[5,12]])),t.tensor([2,1])), 'Different padding and prompt lengths must still select the semantic word in each row.'
    with pytest.raises(ValueError,match='exactly one'):methods.positions_from_spans(offsets,mask,t.tensor([[0,10],[5,12]]))
    with pytest.raises(ValueError,match='whole semantic'):methods.positions_from_spans(offsets,mask,t.tensor([[3,10],[5,12]]))
    print('Token-alignment checks passed: unequal prompt lengths do not change the selected word.')


def test_restricted_calendar_success_does_not_hide_full_vocabulary_failure(methods=m):
    logits=t.tensor([[5.,4.,100.,-1.],[2.,3.,1.,0.]])
    result=methods.score_calendar_logits(logits,t.tensor([1,0]),t.tensor([0,1]),t.tensor([0,1]))
    assert result['desired_restricted_correct'].tolist()==[True,True], 'The desired answer wins among calendar words.'
    assert result['desired_full_correct'].tolist()==[False,True], 'A non-calendar top word is still a full-vocabulary failure.'
    t.testing.assert_close(result['desired_minus_original_logit'],t.ones(2))
    with pytest.raises(ValueError,match='Every model row'):
        methods.score_calendar_logits(logits,t.tensor([1]),t.tensor([0]),t.tensor([0,1]))
    print('Answer-scoring checks passed: restricted-choice success cannot hide a different vocabulary winner.')


def test_least_squares_repeated_fits_and_rank_deficiency(methods=m):
    generator=t.Generator().manual_seed(712)
    x=t.randn(72,5,generator=generator)
    x[:,4]=x[:,1]+x[:,2]
    target=t.randn(72,2,generator=generator)
    first=methods.fit_ridge_coordinates(x,target)
    for _ in range(20):
        repeated=methods.fit_ridge_coordinates(x,target)
        for actual,expected in zip(repeated,first,strict=True):
            assert t.equal(actual,expected), 'An unchanged CPU fit must be repeatable before BF16 intervention rounding.'
    mean,weights,bias=first
    centered=x-mean;residual=centered@weights-(target-bias)
    t.testing.assert_close(centered.T@residual,t.zeros(5,2),atol=2e-5,rtol=0)
    null=t.tensor([0.,-1.,-1.,0.,1.])
    t.testing.assert_close(null@weights,t.zeros(2),atol=2e-6,rtol=0)
    print('Repeated CPU fits agree exactly and rank-deficient least squares has the minimum-norm solution.')

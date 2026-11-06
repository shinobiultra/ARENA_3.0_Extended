# %%


from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

_here = Path.cwd().resolve()
_candidates = [_here, *_here.parents]
_root = next((p for p in _candidates if (p / "chapter18_intrinsic_interpretability" / "exercises").exists()), None)
if _root is not None:
    exercises_dir = _root / "chapter18_intrinsic_interpretability" / "exercises"
else:
    exercises_dir = next(p for p in _candidates if (p / "part1_concept_bottlenecks_backpacks_iit").exists())
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
from part1_concept_bottlenecks_backpacks_iit import tests as tests
MAIN = __name__ == "__main__"
np.set_printoptions(precision=4, suppress=True)

# %%

def concept_probabilities(scores):
    scores = np.asarray(scores, dtype=float)
    return np.exp(-np.logaddexp(0.0, -scores))


if MAIN:
    tests.test_concept_probabilities(concept_probabilities)

# %%

def concept_readout(concepts, weights, bias):
    return np.asarray(concepts) @ np.asarray(weights) + np.asarray(bias)


if MAIN:
    tests.test_concept_readout(concept_readout)

# %%

def correct_concept(concepts, index, replacement):
    result = np.array(concepts, dtype=float, copy=True)
    result[..., index] = replacement
    return result


if MAIN:
    tests.test_correct_concept(correct_concept)

# %%

def binary_accuracy(predictions, targets):
    return float(np.mean(np.asarray(predictions) == np.asarray(targets)))


if MAIN:
    tests.test_binary_accuracy(binary_accuracy)

# %%

def leakage_readout(concepts):
    p = np.asarray(concepts)[..., 0]
    return (p - np.where(p >= 0.5, 0.9, 0.1) > 0.0).astype(int)


if MAIN:
    tests.test_leakage_readout(leakage_readout)

# %%

if MAIN:
    concepts = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
    y = concepts[:,1].astype(int)
    soft = np.column_stack([0.1 + 0.8*concepts[:,0] + 0.1*(2*concepts[:,1]-1), 0.1+0.8*concepts[:,1]])
    corrected = correct_concept(soft, 1, 1-y)
    leaky_before = leakage_readout(soft)
    leaky_after = leakage_readout(corrected)
    honest_after = (concept_readout(corrected, np.array([[0.],[1.]]), [0.])[:,0] >= .5).astype(int)
    print("Thresholded concept accuracy:", binary_accuracy((soft >= .5).astype(int), concepts))
    print("Respect opposite-shape correction: honest / leaky =", binary_accuracy(honest_after, 1-y), binary_accuracy(leaky_after, 1-y))
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(['honest readout','leaky readout','thresholded control'], [binary_accuracy(honest_after,1-y), binary_accuracy(leaky_after,1-y), binary_accuracy(leakage_readout((corrected>=.5).astype(float)),1-y)])
    ax.set(ylim=(0,1.1), ylabel='Agreement with corrected shape',title='Correct concepts do not guarantee correct interventions')
    plt.show()

# %%

def causal_weights(scores):
    scores = np.asarray(scores, dtype=float)
    n = scores.shape[-1]
    masked = np.where(np.tril(np.ones((n,n), dtype=bool)), scores, -np.inf)
    shift = masked - np.max(masked, axis=-1, keepdims=True)
    weights = np.exp(shift)
    return weights / weights.sum(axis=-1, keepdims=True)


if MAIN:
    tests.test_causal_weights(causal_weights)

# %%

def backpack_hidden(token_ids, senses, alpha):
    return np.einsum("kij,jkd->id", alpha, np.asarray(senses)[np.asarray(token_ids)])


if MAIN:
    tests.test_backpack_hidden(backpack_hidden)

# %%

def sense_readout(senses, unembedding):
    return np.asarray(senses) @ np.asarray(unembedding)


if MAIN:
    tests.test_sense_readout(sense_readout)

# %%

def edit_sense(senses, token, sense, delta):
    edited = np.array(senses, dtype=float, copy=True)
    edited[token, sense] += np.asarray(delta)
    return edited


if MAIN:
    tests.test_edit_sense(edit_sense)

# %%

def predicted_edit_delta(token_ids, alpha, token, sense, delta, unembedding):
    mass = np.asarray(alpha)[sense][:, np.asarray(token_ids) == token].sum(axis=-1)
    return mass[:,None] * (np.asarray(delta) @ np.asarray(unembedding))[None,:]


if MAIN:
    tests.test_predicted_edit_delta(predicted_edit_delta)

# %%

if MAIN:
    senses = np.zeros((3,2,2))
    senses[0] = [[3.,0.],[0.,3.]]  # bank; loan and river carry no fixture signal
    U = np.eye(2)  # output labels: money, water
    contexts = {'bank loan': np.array([0,1]), 'bank river':np.array([0,2])}
    context_scores = {}
    for name in contexts:
        score = np.zeros((2,2,2))
        score[:,1,0] = [np.log(9), -np.log(9)] if name=='bank loan' else [-np.log(9), np.log(9)]
        context_scores[name] = causal_weights(score)
    edit_delta = np.array([-2.,0.])
    edited = edit_sense(senses, 0, 0, edit_delta)
    for name, ids in contexts.items():
        alpha = context_scores[name]
        before = backpack_hidden(ids,senses,alpha) @ U
        after = backpack_hidden(ids,edited,alpha) @ U
        oracle = predicted_edit_delta(ids,alpha,0,0,edit_delta,U)
        assert np.allclose(after-before,oracle)
        print(name, 'before / after final logits:', before[-1], after[-1])
    print('Non-contextual bank sense logits:', sense_readout(senses,U)[0])

# %%

def swap_subspace(base, donor, direction):
    direction = np.asarray(direction, dtype=float)
    if not np.isclose(direction @ direction, 1.0):
        raise ValueError("direction must have unit norm")
    base = np.asarray(base)
    return base + ((np.asarray(donor)-base) @ direction)[...,None]*direction


if MAIN:
    tests.test_swap_subspace(swap_subspace)

# %%

def causal_target(base_concepts, donor_concepts, a_weight=2.0):
    return a_weight*np.asarray(donor_concepts)[...,0] + np.asarray(base_concepts)[...,1]


if MAIN:
    tests.test_causal_target(causal_target)

# %%

def interchange_rows(concepts, rotation, direction, a_weight=2.0):
    concepts = np.asarray(concepts, dtype=float)
    hidden = concepts @ np.asarray(rotation).T
    base = np.repeat(hidden, len(hidden), axis=0)
    donor = np.tile(hidden, (len(hidden),1))
    base_c = np.repeat(concepts,len(concepts),axis=0)
    donor_c = np.tile(concepts,(len(concepts),1))
    return swap_subspace(base,donor,direction), causal_target(base_c,donor_c,a_weight)


if MAIN:
    tests.test_interchange_rows(interchange_rows)

# %%

def fit_readout(features, targets):
    return np.linalg.lstsq(np.asarray(features), np.asarray(targets), rcond=None)[0]


if MAIN:
    tests.test_fit_readout(fit_readout)

# %%

def interchange_agreement(predictions, targets, atol=1e-8):
    return float(np.mean(np.isclose(predictions, targets, atol=atol, rtol=0)))


if MAIN:
    tests.test_interchange_agreement(interchange_agreement)

# %%

if MAIN:
    Q = np.array([[1.,-1.],[1.,1.]])/np.sqrt(2)
    direction = Q[:,0]
    observed = np.array([[0.,0.],[1.,1.]])
    all_concepts = np.array([[0.,0.],[0.,1.],[1.,0.],[1.,1.]])
    obs_h = observed @ Q.T
    obs_y = 2*observed[:,0]+observed[:,1]
    w_observed = fit_readout(obs_h,obs_y)
    train_h, train_y = interchange_rows(observed,Q,direction)
    w_iit = fit_readout(np.concatenate([obs_h,train_h]),np.concatenate([obs_y,train_y]))
    audit_h,audit_y = interchange_rows(all_concepts,Q,direction)
    wrong_h,_ = interchange_rows(all_concepts,Q,Q[:,1])
    pred_observed = audit_h @ w_observed
    pred_iit = audit_h @ w_iit
    print('Readout in concept coordinates: observational / IIT', Q.T @ w_observed, Q.T @ w_iit)
    print('All-pair interchange agreement: observational / IIT / wrong subspace', interchange_agreement(pred_observed,audit_y),interchange_agreement(pred_iit,audit_y),interchange_agreement(wrong_h @ w_iit,audit_y))
    print('Observation-only fitting residual:', np.max(np.abs(obs_h @ w_observed-obs_y)))

# %%

if MAIN:
    fig, axes = plt.subplots(1,3,figsize=(14,3.8))
    labels = ['00','01','10','11']
    for ax, pred, title in zip(axes[:2],[pred_observed,pred_iit],['Observational fit: error','IIT readout: error']):
        im=ax.imshow((pred-audit_y).reshape(4,4),vmin=-.5,vmax=.5,cmap='coolwarm')
        ax.set(xticks=range(4),xticklabels=labels,yticks=range(4),yticklabels=labels,xlabel='Donor (a,b)',ylabel='Base (a,b)',title=title)
    fig.colorbar(im,ax=list(axes[:2]),label='Prediction minus causal target',shrink=.8)
    edit_effect=[]; wrong_effect=[]
    wrong_edit=edit_sense(senses,0,1,edit_delta)
    for name,ids in contexts.items():
        alpha=context_scores[name]
        original=backpack_hidden(ids,senses,alpha)@U
        edit_effect.append((backpack_hidden(ids,edited,alpha)@U-original)[-1,0])
        wrong_effect.append((backpack_hidden(ids,wrong_edit,alpha)@U-original)[-1,0])
    x=np.arange(2)
    axes[2].bar(x-.18,edit_effect,width=.35,label='finance sense edit')
    axes[2].bar(x+.18,wrong_effect,width=.35,label='wrong sense control')
    axes[2].set(xticks=x,xticklabels=list(contexts),ylabel='Change in money logit',title='A global edit has context-dependent effects')
    axes[2].legend(fontsize=8)
    plt.show()
    print('All-pair agreement:', interchange_agreement(pred_observed,audit_y), interchange_agreement(pred_iit,audit_y), interchange_agreement(wrong_h@w_iit,audit_y))

# %%

if MAIN:
    a_weight = 4.0  # editable causal rule, not a new random seed
    obs_new = a_weight*observed[:,0]+observed[:,1]
    w_obs_new = fit_readout(obs_h,obs_new)
    train_new_h,train_new_y = interchange_rows(observed,Q,direction,a_weight)
    w_iit_new = fit_readout(np.concatenate([obs_h,train_new_h]),np.concatenate([obs_new,train_new_y]))
    audit_new_h,audit_new_y = interchange_rows(all_concepts,Q,direction,a_weight)
    print('New concept-coordinate coefficients:',Q.T@w_obs_new,Q.T@w_iit_new)
    print('New agreement:',interchange_agreement(audit_new_h@w_obs_new,audit_new_y),interchange_agreement(audit_new_h@w_iit_new,audit_new_y))
    fig,ax=plt.subplots(figsize=(6,3))
    ax.plot(audit_new_y,audit_new_h@w_iit_new,'o',label='IIT readout')
    ax.plot(audit_new_y,audit_new_h@w_obs_new,'x',label='observation-only control')
    ax.plot([0,a_weight+1],[0,a_weight+1],'k--',label='causal oracle')
    ax.set(xlabel='Abstract interchange target',ylabel='Predicted value',title='Changed causal rule')
    ax.legend();plt.show()

# %%

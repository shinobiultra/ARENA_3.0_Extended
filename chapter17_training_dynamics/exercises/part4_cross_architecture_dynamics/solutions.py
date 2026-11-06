# %%


import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
chapter = "chapter17_training_dynamics"
section = "part4_cross_architecture_dynamics"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
import part4_cross_architecture_dynamics.tests as tests
import part4_cross_architecture_dynamics.utils as utils
MAIN = __name__ == "__main__"
np.set_printoptions(precision=4, suppress=True)

# %%

def causal_attention(q, k, v):
    scores=q@k.T/np.sqrt(q.shape[-1])
    scores=np.where(np.tril(np.ones(scores.shape,dtype=bool)),scores,-np.inf)
    weights=np.exp(scores-scores.max(axis=-1,keepdims=True))
    weights/=weights.sum(axis=-1,keepdims=True)
    return weights@v


if MAIN:
    tests.test_causal_attention(causal_attention)

# %%

def selective_scan(u, delta, a, b, c):
    state=np.zeros(u.shape[1]);outputs=[]
    for t in range(len(u)):
        decay=np.exp(delta[t]*a)
        integral=np.empty_like(decay)
        nonzero=a!=0
        integral[nonzero]=np.expm1(delta[t,nonzero]*a[nonzero])/a[nonzero]
        integral[~nonzero]=delta[t,~nonzero]
        state=decay*state+integral*b[t]*u[t]
        outputs.append(c[t]*state)
    return np.array(outputs)


if MAIN:
    tests.test_selective_scan(selective_scan)

# %%

if MAIN:
    sequence=np.array([[1.],[-1.],[1.],[1.],[-1.],[-1.]])
    q=np.zeros((len(sequence),1))
    attention_output=causal_attention(q,q,sequence).ravel()
    delta=np.ones_like(sequence); b=np.ones_like(sequence); c=np.ones_like(sequence)
    scan_output=selective_scan(sequence,delta,np.array([-1.]),b,c).ravel()
    selected_b=np.zeros_like(sequence);selected_b[0]=1
    memory_output=selective_scan(sequence,delta,np.array([0.]),selected_b,c).ravel()
    print("position token causal-mean leaky-scan selected-first-memory")
    for i,row in enumerate(zip(sequence.ravel(),attention_output,scan_output,memory_output)):
        print(i,*[f"{v:+.3f}" for v in row])
    fig,ax=plt.subplots(figsize=(8,3))
    for vals,label in [(attention_output,"Uniform causal attention"),(scan_output,"Leaky SSM"),(memory_output,"Select first input, no decay")]:
        ax.plot(np.arange(6),vals,"o-",label=label)
    ax.set(xlabel="Token position",ylabel="Scalar output",title="A chosen memory mechanism vs two baselines")
    ax.legend();utils.finish_figure(fig)

# %%

def token_nll(logits, targets):
    m=logits.max(axis=-1,keepdims=True)
    log_z=m[...,0]+np.log(np.exp(logits-m).sum(axis=-1))
    chosen=np.take_along_axis(logits,targets[...,None],axis=-1)[...,0]
    return log_z-chosen


if MAIN:
    tests.test_token_nll(token_nll)

# %%

def masked_token_loss(losses, mask):
    if not np.any(mask):
        raise ValueError("No masked targets")
    return float(np.mean(losses[mask]))


if MAIN:
    tests.test_masked_token_loss(masked_token_loss)

# %%

def latent_prediction_loss(prediction, target):
    if prediction.shape != target.shape:
        raise ValueError("Prediction/target shape mismatch")
    return float(np.mean((prediction-target)**2))


if MAIN:
    tests.test_latent_prediction_loss(latent_prediction_loss)

# %%

if MAIN:
    site_losses=np.array([[1.,100.,3.]])
    mask=np.array([[True,False,True]])
    print("masked slice loss",masked_token_loss(site_losses,mask))
    print("latent loss",latent_prediction_loss(np.array([[1.,3.]]),np.array([[0.,1.]])))
    print("scaled latent loss",latent_prediction_loss(np.array([[2.,6.]]),np.array([[0.,2.]])))
    print("collapsed latent loss",latent_prediction_loss(np.zeros((2,3)),np.zeros((2,3))))

# %%

def center_features(x):
    return x-x.mean(axis=0,keepdims=True)


if MAIN:
    tests.test_center_features(center_features)

# %%

def linear_cka(x, y):
    if len(x)!=len(y):
        raise ValueError("Need same examples")
    x=center_features(x);y=center_features(y)
    den=np.linalg.norm(x.T@x)*np.linalg.norm(y.T@y)
    return float(np.linalg.norm(x.T@y)**2/den) if den else float("nan")


if MAIN:
    tests.test_linear_cka(linear_cka)

# %%

def effective_rank(x):
    s=np.linalg.svd(center_features(x),compute_uv=False)
    if s.sum()==0:
        return 0.0
    p=s[s>0]/s.sum()
    return float(np.exp(-np.sum(p*np.log(p))))


if MAIN:
    tests.test_effective_rank(effective_rank)

# %%

def fit_ridge_probe(x, targets, ridge=.1):
    if ridge<=0:
        raise ValueError("Use positive ridge")
    mean_x=x.mean(axis=0);mean_y=targets.mean()
    xc=x-mean_x;yc=targets-mean_y
    w=np.linalg.solve(xc.T@xc+ridge*np.eye(x.shape[1]),xc.T@yc)
    return w,float(mean_y-mean_x@w)


if MAIN:
    tests.test_fit_ridge_probe(fit_ridge_probe)

# %%

def probe_accuracy(x, targets, weights, bias):
    return float(np.mean(targets*(x@weights+bias)>0))


if MAIN:
    tests.test_probe_accuracy(probe_accuracy)

# %%

if MAIN:
    sequences=np.array(list(product([-1.,1.],repeat=6)))
    labels=sequences[:,0]
    noise=sequences[:,1:4]
    ids=["".join("1" if x>0 else "0" for x in row) for row in sequences]
    rng=np.random.default_rng(17)
    order=rng.permutation(len(sequences));train=order[:32];dev=order[32:]
    # These are distinct finite sequences with overlapping abstract feature combinations.
    # TRAIN fits probes; DEV is monitored throughout; there is no untouched TEST.
    q,_=np.linalg.qr(rng.normal(size=(4,4)))
    row_shuffle=rng.permutation(len(sequences))
    shuffled_train_labels=rng.permutation(labels[train])
    strengths=np.array([0.,.01,.05,.1,.3,1.])
    representations=[np.column_stack([g*labels,noise]) for g in strengths]
    rank=[];cka=[];cka_wrong=[];acc=[];acc_control=[]
    for rep in representations:
        rank.append(effective_rank(rep));cka.append(linear_cka(rep,rep@q))
        cka_wrong.append(linear_cka(rep,(rep@q)[row_shuffle]))
        w,bias=fit_ridge_probe(rep[train],labels[train],ridge=.1)
        wc,bc=fit_ridge_probe(rep[train],shuffled_train_labels,ridge=.1)
        acc.append(probe_accuracy(rep[dev],labels[dev],w,bias))
        acc_control.append(probe_accuracy(rep[dev],labels[dev],wc,bc))
    print("g rank CKA_rotation CKA_wrong_rows DEV_probe shuffled_TRAIN_probe")
    for row in zip(strengths,rank,cka,cka_wrong,acc,acc_control):
        print(" ".join(f"{x:.4f}" for x in row))
    fig,axes=plt.subplots(1,3,figsize=(14,3.5))
    axes[0].plot(strengths,rank,"o-",label="Constructed signal + nuisance")
    axes[0].axhline(effective_rank(noise),ls="--",color="gray",label="Nuisance-only rank")
    axes[0].set(xlabel="Chosen signal amplitude",ylabel="Singular-value effective rank",title="Rank is not task information");axes[0].legend()
    axes[1].plot(strengths,cka,"o-",label="Orthogonal rotation")
    axes[1].plot(strengths,cka_wrong,"o-",label="Wrong example pairing")
    axes[1].set(xlabel="Chosen signal amplitude",ylabel="Centered linear CKA",title="Coordinate vs identity control");axes[1].legend()
    axes[2].plot(strengths,acc,"o-",label="Frozen-ridge DEV probe")
    axes[2].plot(strengths,acc_control,"o-",label="Shuffled TRAIN-label probe")
    axes[2].axhline(.5,ls="--",color="gray",label="Balanced chance")
    axes[2].set(xlabel="Chosen signal amplitude",ylabel="Monitored DEV accuracy",title="Same task, same probe protocol");axes[2].legend()
    utils.finish_figure(fig)

# %%

def cumulative_tokens(updates, batch_size, sequence_length, passes=1):
    if batch_size<=0 or sequence_length<=0 or passes<=0 or np.any(np.asarray(updates)<0):
        raise ValueError("Invalid budget count")
    return np.cumsum(np.asarray(updates))*batch_size*sequence_length*passes


if MAIN:
    tests.test_cumulative_tokens(cumulative_tokens)

# %%

def align_examples(reference_ids, candidate_ids, candidate):
    if len(set(reference_ids))!=len(reference_ids) or len(set(candidate_ids))!=len(candidate_ids):
        raise ValueError("Example IDs must be unique")
    if set(reference_ids)!=set(candidate_ids) or len(candidate_ids)!=len(candidate):
        raise ValueError("Example identity mismatch")
    positions={eid:i for i,eid in enumerate(candidate_ids)}
    return candidate[[positions[eid] for eid in reference_ids]]


if MAIN:
    tests.test_align_examples(align_examples)

# %%

if MAIN:
    aligned=align_examples(ids,[ids[i] for i in row_shuffle],(representations[-1]@q)[row_shuffle])
    print("aligned rotation CKA",linear_cka(representations[-1],aligned))
    print("one-pass token budget",cumulative_tokens([10,10,20],8,6))
    print("two-pass token budget",cumulative_tokens([10,10,20],8,6,passes=2))

# %%

if MAIN:
    play_scale=10.
    base=representations[-1]
    changed=base@np.diag([play_scale,1,1,1])
    w,bias=fit_ridge_probe(changed[train],labels[train],ridge=.1)
    wc,bc=fit_ridge_probe(changed[train],shuffled_train_labels,ridge=.1)
    print("anisotropic CKA",linear_cka(base,changed))
    print("anisotropic effective rank",effective_rank(changed))
    print("refit DEV accuracy",probe_accuracy(changed[dev],labels[dev],w,bias))
    print("shuffled-label refit DEV accuracy",probe_accuracy(changed[dev],labels[dev],wc,bc))
    fig,ax=plt.subplots(figsize=(6,3))
    for rep,label in [(base,"Equal feature scales"),(changed,"Answer channel ×10")]:
        singular=np.linalg.svd(center_features(rep),compute_uv=False)
        ax.plot(np.arange(1,5),singular/singular.sum(),"o-",label=label)
    ax.set(xlabel="Singular value index",ylabel="Normalized singular-value mass",title="Same answer, different spectral geometry")
    ax.legend();utils.finish_figure(fig)

# %%

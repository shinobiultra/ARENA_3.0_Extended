# %%


from pathlib import Path
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

chapter = "chapter5_modern_architectures"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part6_linear_attention_retnet_rwkv.tests as tests

MAIN = __name__ == "__main__"
# NumPy arrays stay on CPU; float64 makes small analytic comparisons transparent.

# %%

def positive_features(x):
    """ELU(x)+1, with no in-place mutation."""
    out = np.empty_like(np.asarray(x, dtype=float))
    mask = x >= 0
    out[mask] = x[mask] + 1
    out[~mask] = np.exp(x[~mask])
    return out


if MAIN:
    tests.test_positive_features(positive_features)

# %%

def kernel_attention_parallel(q, k, v):
    """Causal normalized attention for supplied nonnegative features."""
    q,k,v = map(np.asarray,(q,k,v))
    if q.shape != k.shape or q.ndim != 2 or v.ndim != 2 or len(v) != len(q):
        raise ValueError("Expected matching time axes and q/k feature widths")
    if np.any(q<0) or np.any(k<0):
        raise ValueError("Normalized teaching kernel requires nonnegative features")
    weights = np.tril(q @ k.T)
    mass = weights.sum(axis=1,keepdims=True)
    weights = np.divide(weights,mass,out=np.zeros_like(weights,dtype=float),where=mass>0)
    return weights @ v


if MAIN:
    tests.test_kernel_attention_parallel(kernel_attention_parallel)

# %%

def kernel_attention_recurrent(q, k, v):
    """Online normalized kernel attention with a fixed-size state."""
    q,k,v = map(np.asarray,(q,k,v))
    if q.shape != k.shape or q.ndim != 2 or v.ndim != 2 or len(v)!=len(q) or np.any(q<0) or np.any(k<0):
        raise ValueError("Invalid nonnegative time/feature inputs")
    state=np.zeros((k.shape[1],v.shape[1])); normalizer=np.zeros(k.shape[1]); outputs=[]
    for qt,kt,vt in zip(q,k,v):
        state += np.outer(kt,vt)
        normalizer += kt
        denom=qt @ normalizer
        outputs.append(qt @ state / denom if denom>0 else np.zeros(v.shape[1]))
    return np.asarray(outputs).reshape(len(q),v.shape[1]),state,normalizer


if MAIN:
    tests.test_kernel_attention_recurrent(kernel_attention_recurrent)

# %%

def causal_softmax_attention(q, k, v, temperature=1.0):
    """Stable causal softmax baseline, with no future leakage."""
    if temperature<=0:
        raise ValueError("temperature must be positive")
    out=[]
    for t,qt in enumerate(q):
        scores=qt @ k[:t+1].T / temperature
        weights=np.exp(scores-scores.max())
        out.append(weights @ v[:t+1] / weights.sum())
    return np.asarray(out)


if MAIN:
    tests.test_causal_softmax_attention(causal_softmax_attention)

# %%

if MAIN:
    q=np.array([[1.,0.],[0.,1.],[1.,0.]])
    v=np.eye(3)
    reads,state,z=kernel_attention_recurrent(q,q,v)
    display(pd.DataFrame(reads,index=["red→A","blue→B","red→C"],columns=["A","B","C"]))
    fig,ax=plt.subplots(figsize=(7,3))
    ax.plot(reads[:,0],label="old red value A");ax.plot(reads[:,2],label="new red value C")
    ax.plot(kernel_attention_recurrent(q,q,np.zeros_like(v))[0][:,0],label="zero-value control",linestyle="--")
    ax.set(xlabel="write index",ylabel="readout coefficient",title="A repeated key accumulates both values")
    ax.legend();fig.tight_layout();plt.show()

# %%

def retention_parallel(q, k, v, gamma):
    """Unnormalized real retention with causal exponential weights."""
    if not 0<=gamma<=1:
        raise ValueError("gamma must lie in [0,1]")
    T=len(q);lag=np.arange(T)[:,None]-np.arange(T)[None,:]
    decay=np.where(lag>=0,gamma**np.maximum(lag,0),0.0)
    return ((q @ k.T)*decay) @ v


if MAIN:
    tests.test_retention_parallel(retention_parallel)

# %%

def retention_recurrent(q, k, v, gamma, initial=None):
    """Read and return a retained state, including continuation support."""
    if not 0<=gamma<=1:
        raise ValueError("gamma must lie in [0,1]")
    S=np.zeros((k.shape[1],v.shape[1])) if initial is None else np.array(initial,dtype=float,copy=True)
    out=[]
    for qt,kt,vt in zip(q,k,v):
        S=gamma*S+np.outer(kt,vt)
        out.append(qt @ S)
    return np.asarray(out).reshape(len(q),v.shape[1]),S


if MAIN:
    tests.test_retention_recurrent(retention_recurrent)

# %%

def retention_chunkwise(q, k, v, gamma, chunk_size):
    """Parallel within chunks, recurrent between chunks; return final state."""
    if chunk_size<1 or not 0<=gamma<=1:
        raise ValueError("Positive chunk and valid decay required")
    S=np.zeros((k.shape[1],v.shape[1]));outs=[]
    for start in range(0,len(q),chunk_size):
        qc,kc,vc=q[start:start+chunk_size],k[start:start+chunk_size],v[start:start+chunk_size]
        B=len(qc);lag=np.arange(B)[:,None]-np.arange(B)[None,:]
        D=np.where(lag>=0,gamma**np.maximum(lag,0),0.)
        outs.append(((qc@kc.T)*D)@vc+(qc@S)*gamma**np.arange(1,B+1)[:,None])
        S=gamma**B*S+kc.T @ (vc*gamma**np.arange(B-1,-1,-1)[:,None])
    return np.concatenate(outs),S


if MAIN:
    tests.test_retention_chunkwise(retention_chunkwise)

# %%

def decay_horizon(gammas, remaining=0.01):
    """Smallest integer lag whose retained coefficient is at most remaining."""
    g=np.asarray(gammas,dtype=float)
    if np.any((g<0)|(g>1)) or not 0<remaining<1:
        raise ValueError("Invalid decay or threshold")
    out=np.full(g.shape,np.inf);out[g==0]=1
    mask=(g>0)&(g<1)
    out[mask]=np.ceil(np.log(remaining)/np.log(g[mask]))
    return out


if MAIN:
    tests.test_decay_horizon(decay_horizon)

# %%

if MAIN:
    q=np.array([[1.,0.],[0.,1.],[1.,0.],[1.,1.],[0.,1.]])
    k=q.copy();v=np.arange(10.).reshape(5,2)/10
    parallel=retention_parallel(q,k,v,.7)
    recurrent,state=retention_recurrent(q,k,v,.7)
    chunked,last=retention_chunkwise(q,k,v,.7,2)
    display(pd.DataFrame({"parallel":parallel[:,0],"recurrent":recurrent[:,0],"chunked":chunked[:,0],"reset-at-every-token control":retention_parallel(q,k,v,0)[:,0]}))
    fig,ax=plt.subplots(figsize=(7,3))
    for gamma in [.3,.7,.95]:ax.plot(gamma**np.arange(16),label=f"decay {gamma}")
    ax.plot(np.r_[1.,np.zeros(15)],label="no-memory control",linestyle="--")
    ax.set(xlabel="lag",ylabel="write coefficient",title="Multiple retention heads remember different lags");ax.legend();fig.tight_layout();plt.show()
    print("parallel/recurrent max error",np.max(np.abs(parallel-recurrent)))
    print("parallel/chunked max error",np.max(np.abs(parallel-chunked)))

# %%

def time_mix(x, mix):
    """Mix current and previous token, with zero initial previous input."""
    previous=np.concatenate([np.zeros_like(x[:1]),x[:-1]],axis=0)
    return mix*x+(1-mix)*previous


if MAIN:
    tests.test_time_mix(time_mix)

# %%

def rwkv_wkv(keys, values, decay, bonus):
    """Stable channelwise RWKV-4 WKV read and update."""
    decay=np.asarray(decay)
    if np.any((decay<=0)|(decay>1)):
        raise ValueError("decay must be in (0,1]")
    a=np.zeros(values.shape[1]);b=np.zeros_like(a);p=np.full_like(a,-np.inf);out=[]
    for key,value in zip(keys,values):
        r=bonus+key;scale=np.maximum(p,r)
        old=np.exp(p-scale);new=np.exp(r-scale)
        out.append((old*a+new*value)/(old*b+new))
        next_scale=np.maximum(p+np.log(decay),key)
        old=np.exp(p+np.log(decay)-next_scale);new=np.exp(key-next_scale)
        a=old*a+new*value;b=old*b+new;p=next_scale
    return np.asarray(out)


if MAIN:
    tests.test_rwkv_wkv(rwkv_wkv)

# %%

def rwkv_channel_mix(xk, xr, Wk, Wv, Wr):
    """Squared-ReLU channel expansion with a receptance gate."""
    receptance=1/(1+np.exp(-(xr@Wr)))
    hidden=np.maximum(xk@Wk,0)**2
    return receptance*(hidden@Wv)


if MAIN:
    tests.test_rwkv_channel_mix(rwkv_channel_mix)

# %%

def rwkv_block(x, weights):
    """One pre-normalized RWKV-4-style time/channel residual block."""
    def norm(z):
        return (z-z.mean(-1,keepdims=True))/np.sqrt(z.var(-1,keepdims=True)+1e-5)
    z=norm(x);w=weights
    k=time_mix(z,w["mix_k"])@w["K"]
    v=time_mix(z,w["mix_v"])@w["V"]
    r=1/(1+np.exp(-(time_mix(z,w["mix_r"])@w["R"])))
    y=x+(r*rwkv_wkv(k,v,w["decay"],w["bonus"]))@w["O"]
    z=norm(y)
    return y+rwkv_channel_mix(time_mix(z,w["mix_ck"]),time_mix(z,w["mix_cr"]),w["CK"],w["CV"],w["CR"])


if MAIN:
    tests.test_rwkv_block(rwkv_block)

# %%

if MAIN:
    keys=np.zeros((5,1));values=np.array([[1.],[0.],[0.],[0.],[2.]])
    fig,ax=plt.subplots(figsize=(7,3))
    for bonus in [0.,2.]:ax.plot(rwkv_wkv(keys,values,.8,bonus)[:,0],marker="o",label=f"bonus {bonus}")
    ax.plot(rwkv_wkv(keys,np.zeros_like(values),.8,0)[:,0],label="zero-value control",linestyle="--")
    ax.set(xlabel="token index",ylabel="weighted value",title="The current bonus changes a read, not the stored past");ax.legend();fig.tight_layout();plt.show()

# %%

def last_write_episodes(n_values=4):
    """Enumerate all unequal old/new value pairs with a distractor."""
    if n_values<2:
        raise ValueError("Need at least two values")
    episodes=[]
    for old in range(n_values):
        for new in range(n_values):
            if old==new:continue
            episodes.append(dict(keys=np.array([[1.,0.],[0.,1.],[1.,0.]]),values=np.eye(n_values)[[old,(old+1)%n_values,new]],query=np.array([1.,0.]),answer=new))
    return episodes


if MAIN:
    tests.test_last_write_episodes(last_write_episodes)

# %%

def shuffle_values(values, seed=0):
    """Break key/value binding by permuting complete value rows."""
    order=np.random.default_rng(seed).permutation(len(values))
    return np.array(values[order],copy=True)


if MAIN:
    tests.test_shuffle_values(shuffle_values)

# %%

def recall_accuracy(scores, answers):
    """Top-1 accuracy with first-index tie breaking and explicit denominator."""
    scores=np.asarray(scores);answers=np.asarray(answers)
    if scores.ndim!=2 or len(scores)==0 or answers.shape!=(len(scores),) or not np.isfinite(scores).all() or np.any((answers<0)|(answers>=scores.shape[1])):
        raise ValueError("Invalid recall cohort")
    return float(np.mean(scores.argmax(-1)==answers))


if MAIN:
    tests.test_recall_accuracy(recall_accuracy)

# %%

if MAIN:
    episodes=last_write_episodes();names=["kernel average","retention γ=.5","softmax content","shuffled bindings"]
    all_scores={name:[] for name in names};rows=[]
    for index,e in enumerate(episodes):
        qs=np.repeat(e["query"][None,:],3,axis=0);k,v=e["keys"],e["values"]
        scores=[kernel_attention_parallel(qs,k,v)[-1],retention_parallel(qs,k,v,.5)[-1],causal_softmax_attention(qs,k,v,.1)[-1],retention_parallel(qs,k,shuffle_values(v,index),.5)[-1]]
        rows.append({"old":chr(65+v[0].argmax()),"new/target":chr(65+e["answer"]),**{name:chr(65+score.argmax()) for name,score in zip(names,scores)}})
        for name,score in zip(names,scores):all_scores[name].append(score)
    answers=np.array([e["answer"] for e in episodes]);accuracies=[recall_accuracy(np.array(all_scores[name]),answers) for name in names]
    display(pd.DataFrame(rows))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].bar(names,accuracies);axes[0].set(ylim=(0,1.05),ylabel="correct / 12 episodes",title="Latest-write recall on the exact frozen cohort");axes[0].tick_params(axis="x",rotation=25)
    matrix=np.array([all_scores[name][0] for name in names]);im=axes[1].imshow(matrix,aspect="auto",cmap="viridis");axes[1].set(xticks=range(4),xticklabels=list("ABCD"),yticks=range(4),yticklabels=names,xlabel="answer value",title="Episode 0: coefficients, including the broken-binding control");fig.colorbar(im,ax=axes[1],label="readout coefficient")
    fig.tight_layout();plt.show()
    print(dict(zip(names,accuracies)))

# %%

if MAIN:
    play_gamma=1.0
    e=last_write_episodes()[0]
    play_query=np.array([0.,1.])  # change only one condition at a time
    q=np.repeat(play_query[None,:],3,axis=0)
    display(pd.DataFrame({"retention":retention_parallel(q,e["keys"],e["values"],play_gamma)[-1],"kernel baseline":kernel_attention_parallel(q,e["keys"],e["values"])[-1],"zero-write control":retention_parallel(q,e["keys"],np.zeros_like(e["values"]),play_gamma)[-1]},index=list("ABCD")))

# %%

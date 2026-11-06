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
import part7_griffin_xlstm_hyena.tests as tests

MAIN = __name__ == "__main__"
# NumPy arrays stay on CPU; float64 makes small analytic comparisons transparent.

# %%

def rg_coefficients(gate_logits, decay_parameter, c=8.0):
    """Stable RG-LRU recurrence coefficients in [0,1]."""
    if c<=0:raise ValueError("Positive recurrence scale required")
    gate=np.exp(-np.logaddexp(0.,-np.asarray(gate_logits)))
    return np.exp(-c*np.logaddexp(0.,np.asarray(decay_parameter))*gate)


if MAIN:
    tests.test_rg_coefficients(rg_coefficients)

# %%

def rg_lru_step(x, previous, a, input_gate):
    """One variance-scaled RG-LRU update with supplied gates."""
    a=np.asarray(a)
    if np.any((a<0)|(a>1)):raise ValueError("Invalid recurrence coefficient")
    return a*previous+np.sqrt(np.maximum(1-a*a,0))*input_gate*x


if MAIN:
    tests.test_rg_lru_step(rg_lru_step)

# %%

def rg_lru_scan(x, coefficients, input_gates, resets=None):
    """Sequence update with explicit incoming-state resets."""
    x=np.asarray(x);a=np.broadcast_to(coefficients,x.shape);g=np.broadcast_to(input_gates,x.shape)
    resets=np.zeros(len(x),bool) if resets is None else np.asarray(resets,bool)
    h=np.zeros(x.shape[1]);out=[]
    for t in range(len(x)):
        if resets[t]:h=np.zeros_like(h)
        h=rg_lru_step(x[t],h,a[t],g[t]);out.append(h.copy())
    return np.asarray(out)


if MAIN:
    tests.test_rg_lru_scan(rg_lru_scan)

# %%

def local_attention(q, k, v, window):
    """Stable causal sliding-window softmax attention."""
    if type(window) is not int or window<1:raise ValueError("Positive integer window required")
    out=[]
    for t in range(len(q)):
        start=max(0,t-window+1)
        scores=q[t]@k[start:t+1].T/np.sqrt(q.shape[1])
        weights=np.exp(scores-scores.max());weights/=weights.sum()
        out.append(weights@v[start:t+1])
    return np.asarray(out)


if MAIN:
    tests.test_local_attention(local_attention)

# %%

def griffin_recurrent_block(x, W_in, W_gate, W_out, conv_kernel, coefficients, input_gates):
    """A supplied-weight Griffin recurrent branch with residual addition."""
    projected=x@W_in
    filtered=np.stack([np.convolve(projected[:,c],conv_kernel[:,c],mode="full")[:len(x)] for c in range(projected.shape[1])],axis=1)
    h=rg_lru_scan(filtered,coefficients,input_gates)
    z=x@W_gate
    gate=.5*z*(1+np.tanh(np.sqrt(2/np.pi)*(z+.044715*z**3)))
    return x+(gate*h)@W_out


if MAIN:
    tests.test_griffin_recurrent_block(griffin_recurrent_block)

# %%

if MAIN:
    pulse=np.zeros((9,2));pulse[0,0]=1
    rg=rg_lru_scan(pulse,.8,1)
    reset=rg_lru_scan(pulse,.8,1,np.arange(9)>0)
    att=local_attention(np.zeros_like(pulse),np.zeros_like(pulse),pulse,3)
    fig,ax=plt.subplots(figsize=(8,3));ax.plot(rg[:,0],marker="o",label="RG-LRU raw state");ax.plot(att[:,0],marker="s",label="uniform local attention, window 3");ax.plot(reset[:,0],linestyle="--",label="reset-every-gap control");ax.set(xlabel="lag after the write",ylabel="readout of written channel",title="The same pulse reaches different memory mechanisms");ax.legend();fig.tight_layout();plt.show()
    print("RG first/final",rg[0,0],rg[-1,0])

# %%

def slstm_step(z, log_i, log_f, output_gate, state):
    """Stabilized scalar/vector sLSTM memory with a shared log scale."""
    c,n,m=state
    next_m=np.maximum(log_f+m,log_i)
    f=np.exp(log_f+m-next_m);i=np.exp(log_i-next_m)
    next_c=f*c+i*z;next_n=f*n+i
    hidden=output_gate*np.divide(next_c,next_n,out=np.zeros_like(np.asarray(next_c,dtype=float)),where=next_n>0)
    return hidden,(next_c,next_n,next_m)


if MAIN:
    tests.test_slstm_step(slstm_step)

# %%

def slstm_scan(values, log_input, log_forget, output_gates=1.0):
    """Return normalized sLSTM memory at every time with supplied log gates."""
    shape=values.shape;li=np.broadcast_to(log_input,shape);lf=np.broadcast_to(log_forget,shape);og=np.broadcast_to(output_gates,shape)
    state=(np.zeros(shape[1]),np.zeros(shape[1]),np.full(shape[1],-np.inf));out=[]
    for t in range(len(values)):
        h,state=slstm_step(values[t],li[t],lf[t],og[t],state);out.append(h.copy())
    return np.asarray(out)


if MAIN:
    tests.test_slstm_scan(slstm_scan)

# %%

def mlstm_step(q, k, v, input_gate, forget_gate, output_gate, state):
    """mLSTM value-by-key memory with signed denominator floor."""
    if input_gate<0 or forget_gate<0:raise ValueError("Activated memory gates must be nonnegative")
    C,n=state;scaled_k=k/np.sqrt(len(k))
    C=forget_gate*C+input_gate*np.outer(v,scaled_k)
    n=forget_gate*n+input_gate*scaled_k
    return output_gate*(C@q)/max(abs(float(n@q)),1.),(C,n)


if MAIN:
    tests.test_mlstm_step(mlstm_step)

# %%

def mlstm_scan(q, k, v, input_gates, forget_gates):
    """Sequence mLSTM with supplied activated gates and unit output gate."""
    T=len(q);ig=np.broadcast_to(input_gates,(T,));fg=np.broadcast_to(forget_gates,(T,))
    state=(np.zeros((v.shape[1],k.shape[1])),np.zeros(k.shape[1]));out=[]
    for t in range(T):
        h,state=mlstm_step(q[t],k[t],v[t],ig[t],fg[t],1.,state);out.append(h)
    return np.asarray(out),state[0],state[1]


if MAIN:
    tests.test_mlstm_scan(mlstm_scan)

# %%

def mlstm_parallel(q, k, v, input_gates, forget_gates):
    """Parallel mLSTM core by explicitly enumerating causal gate products."""
    T=len(q);ig=np.broadcast_to(input_gates,(T,));fg=np.broadcast_to(forget_gates,(T,));weights=np.zeros((T,T))
    if np.any(ig<0) or np.any(fg<0):raise ValueError("Nonnegative activated gates required")
    for t in range(T):
        for s in range(t+1):weights[t,s]=ig[s]*np.prod(fg[s+1:t+1])
    scores=(q@k.T)/np.sqrt(k.shape[1])*weights
    return (scores@v)/np.maximum(np.abs(scores.sum(1,keepdims=True)),1)


if MAIN:
    tests.test_mlstm_parallel(mlstm_parallel)

# %%

if MAIN:
    values=np.array([[1.],[0.],[0.],[-1.]])
    base_i=np.array([[0.],[-20.],[-20.],[0.]])
    revision_i=base_i.copy();revision_i[-1]=np.log(9.)
    base=slstm_scan(values,base_i,0.)[:,0];revised=slstm_scan(values,revision_i,0.)[:,0]
    fig,ax=plt.subplots(figsize=(8,3));ax.plot(base,marker="o",label="equal old/new write weights");ax.plot(revised,marker="s",label="new write weight ×9");ax.plot(slstm_scan(np.zeros_like(values),revision_i,0.)[:,0],linestyle="--",label="zero-value control");ax.set(xlabel="token index",ylabel="normalized scalar memory",title="A stronger new write revises the remembered sign");ax.legend();fig.tight_layout();plt.show();print("final equal/reweighted",base[-1],revised[-1])

# %%

def fft_causal_convolution(x, kernel):
    """Linear causal channelwise convolution using zero-padded FFTs."""
    if x.ndim!=2 or kernel.ndim!=2 or x.shape[1]!=kernel.shape[1] or len(x)==0 or len(kernel)==0:
        raise ValueError("Nonempty channel-aligned signals required")
    length=len(x)+len(kernel)-1
    size=1<<(length-1).bit_length()
    spectrum=np.fft.rfft(x,n=size,axis=0)*np.fft.rfft(kernel,n=size,axis=0)
    return np.fft.irfft(spectrum,n=size,axis=0)[:len(x)]


if MAIN:
    tests.test_fft_causal_convolution(fft_causal_convolution)

# %%

def implicit_filter(length, W1, W2, decay):
    """Sine-MLP positional filter with a channelwise exponential window."""
    if type(length) is not int or length<1 or np.any(np.asarray(decay)<0):raise ValueError("Positive length and nonnegative decay required")
    t=np.arange(length,dtype=float)/max(length-1,1)
    positions=np.stack([t,np.cos(np.pi*t),np.sin(np.pi*t)],axis=1)
    return (np.sin(positions@W1)@W2)*np.exp(-np.arange(length)[:,None]*decay)


if MAIN:
    tests.test_implicit_filter(implicit_filter)

# %%

def hyena_order_two(values, gate1, gate2, filter1, filter2):
    """Two implicit-filter convolutions interleaved with diagonal gates."""
    return gate2*fft_causal_convolution(gate1*fft_causal_convolution(values,filter1),filter2)


if MAIN:
    tests.test_hyena_order_two(hyena_order_two)

# %%

if MAIN:
    W1=np.array([[0.],[1.],[0.]]);W2=np.array([[1.,1.]])
    filters=implicit_filter(9,W1,W2,np.array([.1,.3]))
    signal=np.zeros((9,2));signal[0]=1
    filtered=fft_causal_convolution(signal,filters)
    fig,ax=plt.subplots(figsize=(8,3))
    for c in range(2):ax.plot(filtered[:,c],marker="o",label=f"implicit channel {c}")
    ax.plot(fft_causal_convolution(np.zeros_like(signal),filters)[:,0],linestyle="--",label="zero-input control")
    ax.axhline(0,color="black",linewidth=.7);ax.set(xlabel="lag",ylabel="response to the original write",title="Sine-MLP filter shape and decay are both visible");ax.legend();fig.tight_layout();plt.show()

# %%

def hybrid_stack(x, coefficients, input_gates, Wq, Wk, Wv, window):
    """A recurrent residual followed by a local-attention residual."""
    h=x+rg_lru_scan(x,coefficients,input_gates)
    return h+local_attention(h@Wq,h@Wk,h@Wv,window)


if MAIN:
    tests.test_hybrid_stack(hybrid_stack)

# %%

def retained_fraction(trace):
    """Divide a scalar response trajectory by its first nonzero declared write."""
    trace=np.asarray(trace,dtype=float)
    if trace.ndim!=1 or len(trace)==0 or not np.isfinite(trace).all() or trace[0]==0:raise ValueError("Finite nonzero declared initial response required")
    return trace/trace[0]


if MAIN:
    tests.test_retained_fraction(retained_fraction)

# %%

def trace_error(trace, target):
    """Mean squared query-readout error with all query positions retained."""
    trace=np.asarray(trace);target=np.asarray(target)
    if trace.shape!=target.shape or trace.ndim!=1 or len(trace)==0 or not np.isfinite(trace).all() or not np.isfinite(target).all():raise ValueError("Finite aligned query traces required")
    return float(np.mean((trace-target)**2))


if MAIN:
    tests.test_trace_error(trace_error)

# %%

if MAIN:
    x=np.zeros((9,2));x[0,0]=1.;wrong=x[:,::-1].copy()
    rg=rg_lru_scan(x,.8,1)[:,0]
    scalar=slstm_scan(x,0.,0.)[:,0]
    keys=np.array([[1.,0.]]+[[0.,1.]]*8);queries=np.repeat([[1.,0.]],9,axis=0)
    matrix=mlstm_scan(queries,keys,x,1.,1.)[0][:,0]
    local=local_attention(np.zeros_like(x),np.zeros_like(x),x,3)[:,0]
    filter_=implicit_filter(9,np.array([[0.],[1.],[0.]]),np.ones((1,2)),.1)
    hyena=hyena_order_two(x,np.ones_like(x),np.ones_like(x),np.ones((1,2)),filter_)[:,0]
    traces={"RG-LRU":rg,"sLSTM scalar":scalar,"mLSTM addressed":matrix,"Hyena filter":hyena,"window-3 attention":local}
    wrong_control=rg_lru_scan(wrong,.8,1)[:,0]
    reset_control=rg_lru_scan(x,.8,1,np.arange(9)>0)[:,0]
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,trace in traces.items():axes[0].plot(retained_fraction(trace),marker=".",label=name)
    axes[0].plot(wrong_control,linestyle="--",label="wrong-channel control");axes[0].plot(reset_control/rg[0],linestyle=":",label="reset-carry control");axes[0].set(xlabel="lag after fixed write",ylabel="fraction of initial readout",title="Different mechanisms retain different information");axes[0].legend(fontsize=8)
    hybrid=hybrid_stack(x,.8,1,np.eye(2),np.eye(2),np.eye(2),3)[:,0]
    display(pd.DataFrame({**traces,"wrong-channel control":wrong_control,"hybrid raw":hybrid}))
    matrix_plot=np.stack([retained_fraction(trace) for trace in traces.values()]+[wrong_control])
    im=axes[1].imshow(matrix_plot,aspect="auto",cmap="coolwarm",vmin=-1,vmax=1);axes[1].set(yticks=range(6),yticklabels=[*traces,"wrong-channel control"],xlabel="lag",title="Cue response including a matched-norm wrong write");fig.colorbar(im,ax=axes[1],label="fraction of initial response")
    fig.tight_layout();plt.show()
    print("raw final RG/local",rg[-1],local[-1])
    print("descriptive normalized trace MSE",{name:trace_error(retained_fraction(trace),np.ones(9)) for name,trace in traces.items()})

# %%

if MAIN:
    play_a=.95
    play=rg_lru_scan(x,play_a,1)[:,0]
    display(pd.DataFrame({"a=.8 raw":rg,"a=.95 raw":play,"a=.95 fraction":retained_fraction(play),"wrong-channel control":rg_lru_scan(wrong,play_a,1)[:,0]}))
    print("new raw final",play[-1])

# %%

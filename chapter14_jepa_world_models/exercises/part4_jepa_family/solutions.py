# %%


from pathlib import Path
import sys
import itertools
import numpy as np
import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
chapter = "chapter14_jepa_world_models"
section = "part4_jepa_family"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part4_jepa_family.tests as tests
MAIN = __name__ == "__main__"
device = t.device("cpu")
np.set_printoptions(precision=4, suppress=True)

# %%

def masked_mean(tokens, visible):
    weights = visible.to(tokens.dtype)[..., None]
    count = weights.sum(dim=1)
    if t.any(count == 0):
        raise ValueError("each context needs a visible token")
    return (tokens * weights).sum(dim=1) / count


if MAIN:
    tests.test_masked_mean(masked_mean)

# %%

def jepa_mse(prediction, target):
    return ((prediction - target.detach())**2).mean()


if MAIN:
    tests.test_jepa_mse(jepa_mse)

# %%

def ema_update(teacher, student, momentum):
    if not 0 <= momentum <= 1:
        raise ValueError("momentum must be in [0,1]")
    with t.no_grad():
        for target, source in zip(teacher, student, strict=True):
            target.mul_(momentum).add_(source, alpha=1-momentum)


if MAIN:
    tests.test_ema_update(ema_update)

# %%

def collapse_statistics(embeddings):
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    variance = np.mean(centered**2, axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    energy = singular**2
    if energy.sum() < 1e-20:
        return variance, 0.0
    p = energy[energy > 1e-20] / energy.sum()
    return variance, float(np.exp(-np.sum(p * np.log(p))))


if MAIN:
    tests.test_collapse_statistics(collapse_statistics)

# %%

if MAIN:
    t.manual_seed(4)
    encoder = t.nn.Linear(4,2,bias=False,device="cpu")
    teacher = t.nn.Linear(4,2,bias=False,device="cpu")
    teacher.load_state_dict(encoder.state_dict())
    predictor = t.nn.Linear(2,2,bias=False,device="cpu")
    x=t.tensor([[1.,0.,1.,0.],[0.,1.,0.,1.],[1.,1.,1.,1.]])
    optimizer=t.optim.SGD(list(encoder.parameters())+list(predictor.parameters()),lr=.05)
    loss=jepa_mse(predictor(encoder(x)),teacher(x));optimizer.zero_grad();loss.backward();optimizer.step()
    ema_update(list(teacher.parameters()),list(encoder.parameters()),.9)
    print("finite one-update loss:",round(float(loss.detach()),6))
    print("collapsed target loss:",float(jepa_mse(t.zeros(3,2),t.zeros(3,2))))

# %%

def time_views(intercept, slope):
    times = np.arange(8, dtype=float)
    series = np.asarray(intercept)[:, None] + np.asarray(slope)[:, None] * times
    return series[:, :4], series[:, 4:]


if MAIN:
    tests.test_time_views(time_views)

# %%

def audio_views(amplitude1, amplitude2):
    time = np.arange(32, dtype=float) / 32
    views = []
    for phase in [0., np.pi/3]:
        signal = np.asarray(amplitude1)[:,None]*np.sin(2*np.pi*3*time+phase) + np.asarray(amplitude2)[:,None]*np.sin(2*np.pi*7*time+phase)
        magnitude = np.abs(np.fft.rfft(signal, axis=1)) / 16
        views.append(magnitude[:, [3,7,10,12]])
    return tuple(views)


if MAIN:
    tests.test_audio_views(audio_views)

# %%

def motion_content_views(content, velocity):
    content, velocity = np.asarray(content), np.asarray(velocity)
    context = np.stack([content, content+velocity, content, np.ones_like(content)],axis=1)
    target = np.stack([content+2*velocity, content+3*velocity, content, np.ones_like(content)],axis=1)
    return context, target


if MAIN:
    tests.test_motion_content_views(motion_content_views)

# %%

def text_views(count_red, count_blue):
    count_red, count_blue = np.asarray(count_red), np.asarray(count_blue)
    context = np.stack([count_red, count_blue, count_red+count_blue, np.ones_like(count_red)],axis=1)
    target = np.stack([count_blue, count_red, count_red+count_blue, np.ones_like(count_red)],axis=1)
    return context, target


if MAIN:
    tests.test_text_views(text_views)

# %%

def visual_language_views(red, blue):
    red, blue = np.asarray(red), np.asarray(blue)
    images = np.zeros((len(red),2,4,4))
    images[:,0,:,:2] = red[:,None,None]
    images[:,1,:,2:] = blue[:,None,None]
    context = np.stack([images[:,0].mean((1,2)), images[:,1].mean((1,2)), np.ones_like(red), np.zeros_like(red)],axis=1)
    answer_embeddings = np.stack([red, blue, red-blue, np.ones_like(red)],axis=1)
    return context, answer_embeddings


if MAIN:
    tests.test_visual_language_views(visual_language_views)

# %%

if MAIN:
    ctime,ytime=time_views(np.array([.5]),np.array([.2]))
    caudio,yaudio=audio_views(np.array([.8]),np.array([.3]))
    cmotion,ymotion=motion_content_views(np.array([.5]),np.array([.2]))
    fig,axs=plt.subplots(1,3,figsize=(12,3.2))
    axs[0].plot(range(4),ctime[0],"o-",label="visible context");axs[0].plot(range(4,8),ytime[0],"o-",label="future target")
    axs[0].plot(range(8),np.repeat(ctime[0,-1],8),"--",label="persistence baseline")
    axs[0].set(xlabel="time",ylabel="position",title="Time-series causal split")
    axs[1].bar(np.arange(4)-.15,caudio[0],.3,label="phase 0");axs[1].bar(np.arange(4)+.15,yaudio[0],.3,label="phase-shift control")
    axs[1].set(xticks=range(4),xticklabels=[3,7,10,12],xlabel="Fourier bin",ylabel="amplitude",title="Audio target invariance")
    axs[2].plot(range(4),np.r_[cmotion[0,:2],ymotion[0,:2]],"o-",label="moving marker")
    axs[2].plot(range(4),np.repeat(cmotion[0,0],4),"--",label="static control")
    axs[2].set(xlabel="frame",ylabel="position",title="Content fixed, motion changes")
    for ax in axs:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.tight_layout();plt.show()

# %%

def fit_affine(inputs, targets):
    design = np.c_[inputs, np.ones(len(inputs))]
    return np.linalg.lstsq(design, targets, rcond=None)[0]


if MAIN:
    tests.test_fit_affine(fit_affine)

# %%

def normalized_probe_error(prediction, truth):
    mse = np.mean((prediction-truth)**2, axis=0)
    variance = np.var(truth, axis=0)
    if np.any(variance < 1e-12):
        raise ValueError("probe factors must vary")
    return mse / variance


if MAIN:
    tests.test_normalized_probe_error(normalized_probe_error)

# %%

if MAIN:
    factors=np.array(list(itertools.product(np.linspace(.2,1.,8),repeat=2)))
    fit=np.arange(len(factors))%2==0
    check=~fit
    E=np.array([[1.,.2],[-.3,1.],[.5,-.4],[.1,.2]])
    adapters={"time":time_views,"audio":audio_views,"motion/content":motion_content_views,"paired text":text_views,"vision→answer":visual_language_views}
    rows=[];ranks=[];latent_errors=[];retrieval_pairs={}
    for name,adapter in adapters.items():
        adapter_factors = np.rint((factors-.2)*7/.8)+1 if name == "paired text" else factors
        c,y=adapter(*adapter_factors.T);teacher_y=y@E
        affine=fit_affine(c[fit],teacher_y[fit])
        predicted=np.c_[c,np.ones(len(c))]@affine
        shuffle=np.random.default_rng(23).permutation(fit.sum())
        bad=fit_affine(c[fit],teacher_y[fit][shuffle])
        bad_pred=np.c_[c,np.ones(len(c))]@bad
        outputs=[]
        for representation in [predicted,bad_pred,np.zeros_like(predicted)]:
            probe=fit_affine(representation[fit],factors[fit])
            decoded=np.c_[representation[check],np.ones(check.sum())]@probe
            outputs.append(normalized_probe_error(decoded,factors[check]).mean())
        # A shuffled teacher can remain an invertible linear map: audit direct target prediction too.
        latent_errors.append([np.mean((predicted[check]-teacher_y[check])**2),np.mean((bad_pred[check]-teacher_y[check])**2),0.])
        rows.append(outputs);ranks.append(collapse_statistics(predicted)[1]);retrieval_pairs[name]=(predicted[check],teacher_y[check])
    rows=np.asarray(rows);latent_errors=np.asarray(latent_errors)
    print("probe normalized errors [aligned, shuffled-fit, collapsed]:")
    for name,row in zip(adapters,rows):print(name,np.round(row,4))
    print("target latent MSE [aligned, shuffled-fit, collapsed target]:")
    print(np.round(latent_errors,4))

# %%

if MAIN:
    fig,axs=plt.subplots(1,2,figsize=(11,4))
    for ax,values,title in [(axs[0],rows,"Can a supervised probe recover the factors?"),(axs[1],latent_errors,"Latent fit (collapsed uses zero targets)")]:
        im=ax.imshow(values,aspect="auto",cmap="magma",vmin=0)
        ax.set(xticks=range(3),xticklabels=["aligned","shuffled fit","collapsed"],yticks=range(5),yticklabels=list(adapters),title=title,xlabel="predictor/control")
        for i in range(5):
            for j in range(3):ax.text(j,i,f"{values[i,j]:.3f}",ha="center",va="center",color="white" if values[i,j]<values.max()*.5 else "black")
        fig.colorbar(im,ax=ax,label="normalized factor MSE" if ax is axs[0] else "latent MSE")
    fig.tight_layout();plt.show()

# %%

def ablate_motion(context):
    out=np.array(context,copy=True)
    out[:,1]=out[:,0]
    return out


if MAIN:
    tests.test_ablate_motion(ablate_motion)

# %%

def retrieval_accuracy(predicted, candidates, correct_indices):
    distance=((predicted[:,None,:]-candidates[None,:,:])**2).sum(axis=-1)
    retrieved=np.argmin(distance,axis=1)
    return float(np.mean(retrieved==correct_indices)), distance


if MAIN:
    tests.test_retrieval_accuracy(retrieval_accuracy)

# %%

if MAIN:
    p,y=retrieval_pairs["vision→answer"]
    accuracy,dist=retrieval_accuracy(p,y,np.arange(len(p)))
    wrong=np.roll(np.arange(len(p)),1)
    bad_accuracy,bad_dist=retrieval_accuracy(p,y[wrong],np.arange(len(p)))
    c,y_motion=motion_content_views(*factors.T)
    wm=fit_affine(c[fit],(y_motion@E)[fit])
    true_predictions=np.c_[c,np.ones(len(c))]@wm
    ablated_predictions=np.c_[ablate_motion(c),np.ones(len(c))]@wm
    target=y_motion@E
    print("vision-answer top-one:",accuracy,"permuted-answer control:",bad_accuracy)
    print("motion target MSE aligned/ablated:",round(float(np.mean((true_predictions-target)**2)),6),round(float(np.mean((ablated_predictions-target)**2)),6))
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for ax,d,title in [(axs[0],dist,"Aligned answer candidates"),(axs[1],bad_dist,"Permuted answer candidates")]:
        im=ax.imshow(d,cmap="viridis",aspect="auto");fig.colorbar(im,ax=ax,label="squared embedding distance")
        ax.set(xlabel="candidate index",ylabel="query index",title=title)
    fig.tight_layout();plt.show()

# %%

if MAIN:
    play_context, play_target=audio_views(np.array([.8]),np.array([.3]))
    play_target[:,2]=.6  # genuinely new target-only tone
    baseline_context,baseline_target=audio_views(*factors.T)
    audio_predictor=fit_affine(baseline_context[fit],(baseline_target@E)[fit])
    play_prediction=np.c_[play_context,np.ones(1)]@audio_predictor
    print("new-tone latent squared error:",float(np.mean((play_prediction-play_target@E)**2)))

# %%

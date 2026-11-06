# %%


import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
chapter = "chapter17_training_dynamics"
section = "part3_finetuning_diffs_emergence"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
import part3_finetuning_diffs_emergence.tests as tests
import part3_finetuning_diffs_emergence.utils as utils
MAIN = __name__ == "__main__"
np.set_printoptions(precision=4, suppress=True)

# %%

def parameter_diff(before, after):
    if before.keys() != after.keys():
        raise ValueError("Checkpoint keys differ")
    if any(before[k].shape != after[k].shape for k in before):
        raise ValueError("Checkpoint shapes differ")
    return {k: after[k] - before[k] for k in before}


if MAIN:
    tests.test_parameter_diff(parameter_diff)

# %%

def relative_change(before, after):
    denom = np.linalg.norm(before)
    return float(np.linalg.norm(after-before)/denom) if denom else float("nan")


if MAIN:
    tests.test_relative_change(relative_change)

# %%

def activation_difference(before, after, before_ids, after_ids):
    if list(before_ids) != list(after_ids) or len(set(before_ids)) != len(before_ids):
        raise ValueError("Require matching unique ordered examples")
    if before.shape != after.shape or len(before_ids) != len(before):
        raise ValueError("Paired activation shape mismatch")
    return after-before


if MAIN:
    tests.test_activation_difference(activation_difference)

# %%

if MAIN:
    inputs = np.array(list(product([-1.,1.],repeat=3)))
    a,b,c = inputs.T
    targets = a*b
    base_hidden = np.column_stack([a,b,.25*a*b,c])
    late_hidden = np.column_stack([a,b,a*b,c])
    example_ids = [f"{int(x):+d}/{int(y):+d}/{int(z):+d}" for x,y,z in inputs]
    delta_hidden = activation_difference(base_hidden,late_hidden,example_ids,example_ids)
    print("input         target   delta(product feature)")
    for eid,y,d in zip(example_ids,targets,delta_hidden[:,2]):
        print(eid, f"{y:+.0f}", f"{d:+.2f}")

# %%

def paired_stack(before, after):
    if before.shape != after.shape:
        raise ValueError("Paired activations must have equal shapes")
    return np.concatenate([before,after],axis=1)


if MAIN:
    tests.test_paired_stack(paired_stack)

# %%

def soft_threshold(x, penalty):
    if penalty < 0:
        raise ValueError("Penalty must be nonnegative")
    return np.sign(x)*np.maximum(np.abs(x)-penalty,0)


if MAIN:
    tests.test_soft_threshold(soft_threshold)

# %%

def infer_codes(x, dictionary, penalty=.05, steps=200):
    lipschitz = np.linalg.norm(dictionary,ord=2)**2
    z = np.zeros((len(x),len(dictionary)))
    if lipschitz == 0:
        return z
    eta = 1/lipschitz
    for _ in range(steps):
        grad = (z@dictionary-x)@dictionary.T
        z = soft_threshold(z-eta*grad, eta*penalty)
    return z


if MAIN:
    tests.test_infer_codes(infer_codes)

# %%

def reconstruct_pair(codes, dictionary):
    joint = codes@dictionary
    if joint.shape[1] % 2:
        raise ValueError("Paired feature blocks must have equal width")
    return tuple(np.split(joint,2,axis=1))


if MAIN:
    tests.test_reconstruct_pair(reconstruct_pair)

# %%

if MAIN:
    joint = paired_stack(base_hidden,late_hidden)
    dictionary = np.concatenate([np.diag([1,1,.25,1]), np.eye(4)],axis=1)
    codes = infer_codes(joint,dictionary,penalty=.05)
    rec0,rec1 = reconstruct_pair(codes,dictionary)
    wrong_joint = paired_stack(base_hidden,late_hidden[::-1])
    wrong_codes = infer_codes(wrong_joint,dictionary,penalty=.05)
    print("paired reconstruction MSE",round(float(np.mean((codes@dictionary-joint)**2)),6))
    print("wrong-pair reconstruction MSE",round(float(np.mean((wrong_codes@dictionary-wrong_joint)**2)),6))
    fig,axes=plt.subplots(1,2,figsize=(10,3))
    for ax,arr,title in zip(axes,[codes@dictionary-joint,wrong_codes@dictionary-wrong_joint],["Aligned pairs residual", "Reversed rows control residual"]):
        im=ax.imshow(arr,vmin=-1,vmax=1,cmap="coolwarm",aspect="auto")
        ax.set(title=title,xlabel="Before features | after features",ylabel="Ordered finite input")
    fig.colorbar(im,ax=list(axes),label="Reconstruction residual")
    plt.show()

# %%

def signed_margins(hidden, readout, targets):
    if not np.isin(targets,[-1,1]).all():
        raise ValueError("Use signed labels")
    return targets*(hidden@readout)


if MAIN:
    tests.test_signed_margins(signed_margins)

# %%

def capability_accuracy(margins):
    return float(np.mean(np.asarray(margins)>0))


if MAIN:
    tests.test_capability_accuracy(capability_accuracy)

# %%

def threshold_crossing(coordinates, accuracies, threshold=.9):
    hits=np.flatnonzero(np.asarray(accuracies)>=threshold)
    return coordinates[int(hits[0])] if len(hits) else None


if MAIN:
    tests.test_threshold_crossing(threshold_crossing)

# %%

if MAIN:
    coordinates = np.linspace(0,1,21)
    history, margin_history = [], []
    for g in coordinates:
        hidden=np.column_stack([a,b,(.25+.75*g)*a*b,c])
        readout=np.array([.4,.4,2*g,.15])
        margins=signed_margins(hidden,readout,targets)
        history.append(capability_accuracy(margins));margin_history.append(margins)
    margin_history=np.array(margin_history)
    print("accuracy at g=0,.25,.5,.75,1:",np.array(history)[[0,5,10,15,20]])
    print("first coordinate at accuracy >= .9:",threshold_crossing(coordinates,history))
    fig,axes=plt.subplots(1,2,figsize=(10,3))
    axes[0].plot(coordinates,margin_history,alpha=.6)
    axes[0].axhline(0,color="black",ls="--",label="Decision boundary")
    axes[0].set(xlabel="Chosen model coordinate g",ylabel="Signed logit margin",title="Smooth score changes");axes[0].legend()
    axes[1].step(coordinates,history,where="post",label="Product-task accuracy")
    axes[1].axhline(.5,color="gray",ls="--",label="Balanced constant baseline")
    axes[1].set(xlabel="Chosen model coordinate g",ylabel="Finite-domain accuracy",title="Discrete capability score");axes[1].legend()
    utils.finish_figure(fig)

# %%

def component_patch(recipient, donor, component):
    patched=recipient.copy()
    patched[:,component]=donor[:,component]
    return patched.sum(axis=1)


if MAIN:
    tests.test_component_patch(component_patch)

# %%

def patch_effects(early, late, targets):
    clean=capability_accuracy(targets*late.sum(axis=1))
    return np.array([clean-capability_accuracy(targets*component_patch(late,early,j)) for j in range(late.shape[1])])


if MAIN:
    tests.test_patch_effects(patch_effects)

# %%

if MAIN:
    early_readout=np.array([.4,.4,0.,.15]); late_readout=np.array([.4,.4,2.,.15])
    early_contrib=base_hidden*early_readout;late_contrib=late_hidden*late_readout
    effects=patch_effects(early_contrib,late_contrib,targets)
    print("necessity accuracy losses [a,b,product,nuisance]:",effects)
    print("late accuracy:",capability_accuracy(targets*late_contrib.sum(axis=1)))
    fig,axes=plt.subplots(1,2,figsize=(11,3))
    axes[0].bar(["a","b","ab","c control"],effects)
    axes[0].set(xlabel="Replaced late component",ylabel="Accuracy drop",title="One-component replacement")
    axes[1].bar(["activation ab","weight ab","both ab","nuisance"],[
     capability_accuracy(signed_margins(late_hidden,late_readout,targets))-capability_accuracy(signed_margins(base_hidden,late_readout,targets)),
     capability_accuracy(signed_margins(late_hidden,late_readout,targets))-capability_accuracy(signed_margins(late_hidden,early_readout,targets)),effects[2],effects[3]])
    axes[1].set(xlabel="Early values restored in late context",ylabel="Accuracy drop",title="Activation and readout are distinct interventions")
    axes[1].tick_params(axis="x",rotation=15)
    utils.finish_figure(fig)

# %%

if MAIN:
    nuisance_weight=1.5
    play_early=early_contrib.copy();play_late=late_contrib.copy()
    play_early[:,3]=c*nuisance_weight;play_late[:,3]=c*nuisance_weight
    print("changed-condition late accuracy",capability_accuracy(targets*play_late.sum(axis=1)))
    print("changed-condition patch losses",patch_effects(play_early,play_late,targets))

# %%

# %%


from pathlib import Path
import sys
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

chapter = "chapter6_sparse_feature_methods"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part3_sae_pathologies_dashboards.tests as tests

MAIN = __name__ == "__main__"
# NumPy arrays stay on CPU; float64 makes small analytic comparisons transparent.

# %%

def word_properties(words):
    """Return explicit first-letter/identity coordinates for benign words."""
    if any(not word for word in words):
        raise ValueError("Every example needs a word")
    return np.array([[w.lower().startswith("a"),w.lower().startswith("b"),w.lower()=="apple",0.] for w in words],dtype=float).reshape(len(words),4)


if MAIN:
    tests.test_word_properties(word_properties)

# %%

def encode_relu(x, encoder, bias):
    """Encode `(examples,input)` into nonnegative `(examples,features)` codes."""
    return np.maximum(x@encoder+bias,0)


if MAIN:
    tests.test_encode_relu(encode_relu)

# %%

def decode_features(codes, decoder, bias):
    """Reconstruct activations from codes and row-wise decoder directions."""
    return codes@decoder+bias


if MAIN:
    tests.test_decode_features(decode_features)

# %%

def reconstruction_mse(x, reconstructed):
    """Mean squared error over all examples and all coordinates."""
    if x.shape!=reconstructed.shape or x.size==0:
        raise ValueError("Need nonempty matching activation arrays")
    return float(np.mean((x-reconstructed)**2))


if MAIN:
    tests.test_reconstruction_mse(reconstruction_mse)

# %%

if MAIN:
    WORDS=["apple","apricot","ant","avocado","banana","berry","boat","bean"]
    X=word_properties(WORDS)
    E_narrow=np.array([[1.,0.],[0.,1.],[0.,0.],[0.,0.]])
    D_narrow=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])
    E_wide=np.array([[1.,0.,0.],[0.,1.,0.],[-1.,0.,1.],[0.,0.,0.]])
    D_wide=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.],[1.,0.,1.,0.]])
    Z_narrow=encode_relu(X,E_narrow,np.zeros(2));Z_wide=encode_relu(X,E_wide,np.zeros(3))
    W_U=np.array([[1.,0.],[0.,1.],[0.,0.],[0.,0.]])
    labels=X[:,0].astype(bool)

# %%

if MAIN:
    errors=[reconstruction_mse(X,decode_features(Z_narrow,D_narrow,np.zeros(4))),reconstruction_mse(X,decode_features(Z_wide,D_wide,np.zeros(4))),reconstruction_mse(X,np.zeros_like(X))]
    display(pd.DataFrame({"word":WORDS,"A concept":labels,"narrow A":Z_narrow[:,0],"wide general A":Z_wide[:,0],"wide apple":Z_wide[:,2]}))
    fig,axes=plt.subplots(1,2,figsize=(10,3))
    axes[0].bar(["width 2","width 3","zero reconstruction"],errors);axes[0].set(ylabel="MSE over 32 coordinates",title="Reconstruction alone favors width 3")
    axes[1].imshow(np.c_[Z_narrow[:,0],Z_wide[:,0],Z_wide[:,2]],aspect="auto",vmin=0,vmax=1);axes[1].set(xticks=range(3),xticklabels=["narrow A","wide A","wide apple"],yticks=range(8),yticklabels=WORDS,title="Where did apple's A coordinate go?")
    fig.tight_layout();plt.show();print("MSE narrow/wide/zero",errors)

# %%

def top_examples(activations, k=5):
    """Stable descending example indices, retaining deterministic ties."""
    if k<0 or np.asarray(activations).ndim!=1:
        raise ValueError("Expected a vector and nonnegative k")
    return np.argsort(-np.asarray(activations),kind="stable")[:k]


if MAIN:
    tests.test_top_examples(top_examples)

# %%

def precision_recall(activations, concept, threshold=0.0):
    """Concept precision and recall with explicit inactive-set behavior."""
    active=np.asarray(activations)>threshold;positive=np.asarray(concept,dtype=bool)
    if active.shape!=positive.shape:
        raise ValueError("Activation and concept axes must match")
    tp=np.count_nonzero(active&positive)
    return (float(tp/active.sum()) if active.any() else 0.,float(tp/positive.sum()) if positive.any() else 0.)


if MAIN:
    tests.test_precision_recall(precision_recall)

# %%

def activation_overlap(left, right, threshold=0.0):
    """Pairwise Jaccard overlap of active example sets."""
    a=left>threshold;b=right>threshold
    intersection=a.astype(int).T@b.astype(int)
    union=a.sum(0)[:,None]+b.sum(0)[None,:]-intersection
    return np.divide(intersection,union,out=np.zeros_like(intersection,dtype=float),where=union>0)


if MAIN:
    tests.test_activation_overlap(activation_overlap)

# %%

def absorption_candidates(concept, primary, alternatives, threshold=0.0):
    """Flag positives missed by primary but active in any alternate feature."""
    return np.asarray(concept,dtype=bool)&(primary<=threshold)&np.any(alternatives>threshold,axis=1)


if MAIN:
    tests.test_absorption_candidates(absorption_candidates)

# %%

if MAIN:
    precision,recall=precision_recall(Z_wide[:,0],labels)
    shuffled=np.random.default_rng(3).permutation(labels)
    control=precision_recall(Z_wide[:,0],shuffled)
    candidates=absorption_candidates(labels,Z_wide[:,0],Z_wide[:,[1,2]])
    display(pd.DataFrame({"word":WORDS,"candidate":candidates,"general A":Z_wide[:,0],"apple alternative":Z_wide[:,2]}))
    fig,axes=plt.subplots(1,2,figsize=(10,3))
    axes[0].bar(np.arange(2)-.18,[precision,recall],.36,label="true first letter");axes[0].bar(np.arange(2)+.18,control,.36,label="shuffled-label control");axes[0].set(xticks=[0,1],xticklabels=["precision","recall"],ylim=(0,1.1),title="A perfect top-example story can miss a positive");axes[0].legend()
    im=axes[1].imshow(activation_overlap(Z_narrow,Z_wide),vmin=0,vmax=1);axes[1].set(xticks=range(3),xticklabels=["general A","B","apple"],yticks=range(2),yticklabels=["narrow A","narrow B"],title="Activity splits across dictionaries");fig.colorbar(im,ax=axes[1],label="Jaccard overlap")
    fig.tight_layout();plt.show();print("precision/recall",(precision,recall),"shuffled",control)

# %%

def head_preactivation_contributions(head_outputs, encoder):
    """Per-head contributions to encoder preactivations before bias/ReLU."""
    n,h,d=head_outputs.shape
    return np.einsum("nhd,hdf->nhf",head_outputs,encoder.reshape(h,d,-1))


if MAIN:
    tests.test_head_preactivation_contributions(head_preactivation_contributions)

# %%

def head_decoder_logit_effects(decoder, W_O, W_U):
    """Map each feature/head decoder slice through output and unembedding."""
    h,d,r=W_O.shape
    return np.einsum("fhd,hdr,rv->fhv",decoder.reshape(len(decoder),h,d),W_O,W_U)


if MAIN:
    tests.test_head_decoder_logit_effects(head_decoder_logit_effects)

# %%

def ablate_features(x, codes, decoder, indices):
    """Subtract selected decoded features while retaining reconstruction error."""
    indices=list(indices)
    if len(set(indices))!=len(indices) or any(i<0 or i>=codes.shape[1] for i in indices):
        raise ValueError("Unique valid feature indices required")
    return np.array(x,copy=True)-codes[:,indices]@decoder[indices]


if MAIN:
    tests.test_ablate_features(ablate_features)

# %%

def correct_class_margin(logits, class_ids):
    """Per-example correct-minus-other margin for a two-class task."""
    ids=np.asarray(class_ids,dtype=int)
    if logits.shape!=(len(ids),2) or np.any((ids<0)|(ids>1)):
        raise ValueError("Expected valid two-class logits")
    return logits[np.arange(len(ids)),ids]-logits[np.arange(len(ids)),1-ids]


if MAIN:
    tests.test_correct_class_margin(correct_class_margin)

# %%

if MAIN:
    conditions={"no edit":[],"general A only":[0],"apple only":[2],"A plus apple":[0,2],"unrelated B control":[1]}
    class_ids=np.where(labels,0,1)
    effects={}
    for name,ids in conditions.items():
        edited=ablate_features(X,Z_wide,D_wide,ids)
        effects[name]=correct_class_margin(X@W_U,class_ids)-correct_class_margin(edited@W_U,class_ids)
    display(pd.DataFrame(effects,index=WORDS))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    matrix=np.stack([effects[name][labels] for name in conditions],axis=1)
    im=axes[0].imshow(matrix,aspect="auto",vmin=0,vmax=1,cmap="magma");axes[0].set(xticks=range(5),xticklabels=list(conditions),yticks=range(4),yticklabels=np.array(WORDS)[labels],title="First-letter margin drop per A word");axes[0].tick_params(axis="x",rotation=35);fig.colorbar(im,ax=axes[0],label="margin drop")
    contributions=head_preactivation_contributions(X.reshape(8,2,2),E_wide)
    axes[1].bar(np.arange(8)-.18,contributions[:,0,0],.36,label="head 0");axes[1].bar(np.arange(8)+.18,contributions[:,1,0],.36,label="head 1");axes[1].set(xticks=range(8),xticklabels=WORDS,ylabel="signed encoder preactivation contribution",title="Apple cancels in the general-A encoder");axes[1].tick_params(axis="x",rotation=35);axes[1].legend();fig.tight_layout();plt.show()
    print("mean A-word margin drops",{name:float(value[labels].mean()) for name,value in effects.items()})

# %%

def dashboard_record(feature, label, words, activations, k=5):
    """Build a JSON-compatible record with ordered examples and activity counts."""
    if len(words)!=len(activations) or not np.isfinite(activations).all():
        raise ValueError("Finite aligned examples required")
    indices=top_examples(activations,k)
    return dict(schema="arena.feature-dashboard.study.v1",feature=int(feature),label=str(label),n_examples=len(words),n_active=int(np.count_nonzero(activations>0)),examples=[dict(text=words[i],activation=float(activations[i])) for i in indices])


if MAIN:
    tests.test_dashboard_record(dashboard_record)

# %%

def validate_dashboard(record):
    """Fail closed on malformed local dashboard interchange records."""
    keys={"schema","feature","label","n_examples","n_active","examples"}
    if set(record)!=keys or record["schema"]!="arena.feature-dashboard.study.v1" or not isinstance(record["label"],str):
        raise ValueError("Unsupported dashboard schema")
    for key in ["feature","n_examples","n_active"]:
        if type(record[key]) is not int or record[key]<0:raise ValueError("Invalid dashboard count")
    if record["n_active"]>record["n_examples"] or not isinstance(record["examples"],list) or len(record["examples"])>record["n_examples"]:
        raise ValueError("Inconsistent dashboard cohort")
    for example in record["examples"]:
        if set(example)!={"text","activation"} or not isinstance(example["text"],str) or type(example["activation"]) not in (int,float) or not np.isfinite(example["activation"]) or example["activation"]<0:
            raise ValueError("Invalid dashboard example")
    return True


if MAIN:
    tests.test_validate_dashboard(validate_dashboard)

# %%

if MAIN:
    record=dashboard_record(0,"starts with A",WORDS,Z_wide[:,0],k=8)
    assert validate_dashboard(record)
    display(pd.DataFrame(record["examples"]))
    fig,ax=plt.subplots(figsize=(7,3))
    ax.hist(Z_wide[labels,0],bins=[-.1,.1,1.1],alpha=.65,label="true A words")
    ax.hist(Z_wide[~labels,0],bins=[-.1,.1,1.1],alpha=.65,label="non-A control")
    ax.set(xlabel="general-A feature activation",ylabel="number of words",title="The positive cohort contains an inactive word");ax.legend();fig.tight_layout();plt.show()
    serialized=json.dumps(record,indent=2,allow_nan=False)
    print(serialized)
    print("precision/recall",precision_recall(Z_wide[:,0],labels))

# %%

if MAIN:
    play_words=["apple","azure","applejack","berry"]
    play_X=word_properties(play_words)
    play_E=E_wide.copy();play_E[2,0]=-.5
    play_codes=encode_relu(play_X,play_E,np.zeros(3))
    display(pd.DataFrame({"word":play_words,"original general A":encode_relu(play_X,E_wide,np.zeros(3))[:,0],"edited general A":play_codes[:,0],"unrelated B control":play_codes[:,1]}))

# %%

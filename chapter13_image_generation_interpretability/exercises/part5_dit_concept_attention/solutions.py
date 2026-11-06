# %%


import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

chapter = "chapter13_image_generation_interpretability"
section = "part5_dit_concept_attention"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).is_dir())
exercises_dir = root_dir / chapter / "exercises"
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part5_dit_concept_attention.tests as tests
import part5_dit_concept_attention.utils as utils
MAIN = __name__ == "__main__"

# %%

def patchify(image, patch_size):
    h,w,c = np.asarray(image).shape
    if h%patch_size or w%patch_size:
        raise ValueError("image dimensions must divide into patches")
    return np.asarray(image).reshape(h//patch_size,patch_size,w//patch_size,patch_size,c).transpose(0,2,1,3,4).reshape(-1,patch_size*patch_size*c)


if MAIN:
    tests.test_patchify(patchify)

# %%

def unpatchify(tokens, height, width, patch_size):
    channels = np.asarray(tokens).shape[-1]//(patch_size**2)
    return np.asarray(tokens).reshape(height//patch_size,width//patch_size,patch_size,patch_size,channels).transpose(0,2,1,3,4).reshape(height,width,channels)


if MAIN:
    tests.test_unpatchify(unpatchify)

# %%

def refinement_step(state, target, rate):
    if not 0 <= rate <= 1:
        raise ValueError("rate must lie in [0,1]")
    return (1-rate)*np.asarray(state)+rate*np.asarray(target)


if MAIN:
    tests.test_refinement_step(refinement_step)

# %%

if MAIN:
    target=np.full((8,8,3),.25);target[:4,:4]=[.85,.15,.15]
    initial=np.full_like(target,.45)
    state=initial.copy();trajectory=[state.copy()]
    for step in range(4):
        state=refinement_step(state,target,.5);trajectory.append(state.copy())
    null_state=initial.copy()
    for step in range(4):null_state=refinement_step(null_state,np.full_like(target,.25),.5)
    print("Patch roundtrip max error:",np.max(np.abs(unpatchify(patchify(target,2),8,8,2)-target)))
    utils.image_grid([trajectory[0],trajectory[1],trajectory[4],null_state],["Initial state","Step1","Step4","Null-target control"],"Explicit state refinement: the target supplies the image")
    plt.show()

# %%

def adaptive_layer_norm(tokens, scale, shift, eps=1e-6):
    x=np.asarray(tokens,dtype=np.float64)
    normalized=(x-x.mean(axis=-1,keepdims=True))/np.sqrt(x.var(axis=-1,keepdims=True)+eps)
    return (1+np.asarray(scale))*normalized+np.asarray(shift)


if MAIN:
    tests.test_adaptive_layer_norm(adaptive_layer_norm)

# %%

def scaled_attention(queries, keys, values):
    scores=np.asarray(queries) @ np.asarray(keys).T / np.sqrt(np.asarray(queries).shape[-1])
    weights=np.exp(scores-scores.max(axis=-1,keepdims=True))
    weights/=weights.sum(axis=-1,keepdims=True)
    return weights @ np.asarray(values),weights


if MAIN:
    tests.test_scaled_attention(scaled_attention)

# %%

def dit_attention_block(tokens, scale, shift, gate, wq, wk, wv, wo):
    conditioned=adaptive_layer_norm(tokens,scale,shift)
    output,_=scaled_attention(conditioned@wq,conditioned@wk,conditioned@wv)
    return np.asarray(tokens)+np.asarray(gate)*(output@wo)


if MAIN:
    tests.test_dit_attention_block(dit_attention_block)

# %%

if MAIN:
    toy_tokens=np.array([[1.,2.,0.,-1.],[0.,1.,3.,-2.],[2.,0.,1.,-1.]])
    identity=np.eye(4)
    closed=dit_attention_block(toy_tokens,.2,.3,0.,identity,identity,identity,identity)
    opened=dit_attention_block(toy_tokens,.2,.3,.5,identity,identity,identity,identity)
    print("Zero gate max change:",np.max(np.abs(closed-toy_tokens)))
    fig,axes=plt.subplots(1,2,figsize=(8,3))
    for ax,array,name in zip(axes,[closed-toy_tokens,opened-toy_tokens],["Zero-gate control","Open attention branch"]):
        im=ax.imshow(array,cmap='coolwarm',vmin=-1,vmax=1);ax.set(title=name,xlabel="Feature",ylabel="Patch token")
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Residual change");plt.show()

# %%

def readonly_concept_attention(image_queries, image_keys, image_values, prompt_keys, prompt_values, concept_queries, concept_keys, concept_values):
    image_output,image_weights=scaled_attention(image_queries,np.concatenate([image_keys,prompt_keys]),np.concatenate([image_values,prompt_values]))
    concept_output,concept_weights=scaled_attention(concept_queries,np.concatenate([image_keys,concept_keys]),np.concatenate([image_values,concept_values]))
    return image_output,concept_output,image_weights,concept_weights


if MAIN:
    tests.test_readonly_concept_attention(readonly_concept_attention)

# %%

def output_space_map(image_output, concept_output):
    scores=np.asarray(image_output) @ np.asarray(concept_output).T
    prob=np.exp(scores-scores.max(axis=-1,keepdims=True))
    return prob/prob.sum(axis=-1,keepdims=True)


if MAIN:
    tests.test_output_space_map(output_space_map)

# %%

def average_layer_maps(layer_maps):
    return np.asarray(layer_maps).mean(axis=0)


if MAIN:
    tests.test_average_layer_maps(average_layer_maps)

# %%

def raw_key_map(concept_queries, image_keys):
    scores=np.asarray(image_keys) @ np.asarray(concept_queries).T / np.sqrt(np.asarray(image_keys).shape[-1])
    weights=np.exp(scores-scores.max(axis=-1,keepdims=True))
    return weights/weights.sum(axis=-1,keepdims=True)


if MAIN:
    tests.test_raw_key_map(raw_key_map)

# %%

def segmentation_iou(predicted, truth):
    p=np.asarray(predicted,dtype=bool);g=np.asarray(truth,dtype=bool)
    union=np.count_nonzero(p|g)
    return float(np.count_nonzero(p&g)/union) if union else 1.0


if MAIN:
    tests.test_segmentation_iou(segmentation_iou)

# %%

if MAIN:
    truth=np.zeros((4,4),bool);truth[:2,:2]=True
    classes=np.where(truth.ravel(),0,1)
    image_queries=4*np.eye(2)[classes]
    image_keys=np.zeros((16,2));image_values=np.zeros((16,2))
    prompt_keys=np.eye(2);prompt_values=np.eye(2)
    concept_queries=4*np.eye(2);concept_keys=np.eye(2);concept_values=np.eye(2)
    image_out,concept_out,_,_=readonly_concept_attention(image_queries,image_keys,image_values,prompt_keys,prompt_values,concept_queries,concept_keys,concept_values)
    map_output=average_layer_maps(np.stack([output_space_map(image_out,concept_out)]*2))
    map_raw=raw_key_map(concept_queries,image_keys)
    map_swapped=output_space_map(image_out,concept_out[::-1])
    conditions=[map_output,map_raw,map_swapped]
    ious=[segmentation_iou(m.argmax(-1).reshape(4,4)==0,truth) for m in conditions]
    print("IoU output / raw-key / swapped:",np.round(ious,3))
    fig,axes=plt.subplots(1,4,figsize=(14,3.5),layout="constrained")
    axes[0].imshow(target);axes[0].set(title="Known square",xlabel="x pixel",ylabel="y pixel")
    for ax,m,name,score in zip(axes[1:],conditions,["Output space","Raw-key baseline","Swapped concepts"],ious):
        im=ax.imshow(m[:,0].reshape(4,4),vmin=0,vmax=1,cmap='magma');ax.set(title=f"{name}\nIoU={score:.2f}",xlabel="Patch x",ylabel="Patch y")
    fig.colorbar(im,ax=axes[1:].ravel().tolist(),label="Class0 probability");plt.show()
    changed_concepts=concept_queries*0-5
    image_out_again,*_=readonly_concept_attention(image_queries,image_keys,image_values,prompt_keys,prompt_values,changed_concepts,concept_keys,concept_values)
    print("Image output change after concept-query edit:",np.max(np.abs(image_out_again-image_out)))

# %%

if MAIN:
    play_truth=np.zeros((4,4),bool);play_truth[2:,2:]=True
    play_queries=4*np.eye(2)[np.where(play_truth.ravel(),0,1)]
    play_image,play_concept,*_=readonly_concept_attention(play_queries,image_keys,image_values,prompt_keys,prompt_values,concept_queries,concept_keys,concept_values)
    play_map=output_space_map(play_image,play_concept)
    print("Moved-square IoU:",segmentation_iou(play_map.argmax(-1).reshape(4,4)==0,play_truth))
    fig,axes=plt.subplots(1,2,figsize=(6,3))
    axes[0].imshow(play_truth,cmap='gray');axes[0].set(title="Moved ground truth",xlabel="Patch x",ylabel="Patch y")
    axes[1].imshow(play_map[:,0].reshape(4,4),vmin=0,vmax=1,cmap='magma');axes[1].set(title="Output-space map",xlabel="Patch x",ylabel="Patch y")
    fig.tight_layout();plt.show()

# %%

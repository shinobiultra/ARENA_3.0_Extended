# %%


import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

chapter = "chapter13_image_generation_interpretability"
section = "part6_autoregressive_image_tokens"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).is_dir())
exercises_dir = root_dir / chapter / "exercises"
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part6_autoregressive_image_tokens.tests as tests
import part6_autoregressive_image_tokens.utils as utils
MAIN = __name__ == "__main__"

# %%

def patch_means(image, patch_size):
    h,w,c=np.asarray(image).shape
    if h%patch_size or w%patch_size:
        raise ValueError("nondivisible image geometry")
    return np.asarray(image).reshape(h//patch_size,patch_size,w//patch_size,patch_size,c).mean(axis=(1,3))


if MAIN:
    tests.test_patch_means(patch_means)

# %%

def nearest_codes(vectors, codebook):
    distances=((np.asarray(vectors)[...,None,:]-np.asarray(codebook))**2).sum(axis=-1)
    return distances.argmin(axis=-1)


if MAIN:
    tests.test_nearest_codes(nearest_codes)

# %%

def decode_tokens(code_grid, codebook, patch_size):
    return np.asarray(codebook)[np.asarray(code_grid,dtype=int)].repeat(patch_size,axis=0).repeat(patch_size,axis=1)


if MAIN:
    tests.test_decode_tokens(decode_tokens)

# %%

if MAIN:
    codebook=np.array([[.2,.2,.2],[.8,.2,.2],[.2,.2,.8]])
    layout=np.zeros((4,4),bool);layout[:2,:2]=True
    red_grid=np.where(layout,1,0)
    red_image=decode_tokens(red_grid,codebook,2)
    encoded=nearest_codes(patch_means(red_image,2),codebook)
    print("Re-encoded token grid:");print(encoded)
    utils.image_grid([red_image,decode_tokens(np.zeros((4,4),int),codebook,2)], ["Red square (tokens)","All-gray baseline"],"Known vocabulary: semantic content is visible in decoded pixels")
    plt.show()

# %%

import torch as t

# %%

def vq_objective(encoder_values, quantized_values, commitment=0.25):
    codebook_loss=(encoder_values.detach()-quantized_values).square().mean()
    commitment_loss=(encoder_values-quantized_values.detach()).square().mean()
    return codebook_loss+commitment*commitment_loss


if MAIN:
    tests.test_vq_objective(vq_objective)

# %%

def straight_through_quantized(encoder_values, quantized_values):
    return encoder_values+(quantized_values-encoder_values).detach()


if MAIN:
    tests.test_straight_through_quantized(straight_through_quantized)

# %%

def code_usage(codes, vocabulary_size):
    counts=np.bincount(np.asarray(codes,dtype=int).ravel(),minlength=vocabulary_size)
    probabilities=counts/counts.sum()
    used=probabilities>0
    perplexity=float(np.exp(-(probabilities[used]*np.log(probabilities[used])).sum()))
    return counts,perplexity


if MAIN:
    tests.test_code_usage(code_usage)

# %%

def quantization_mse(vectors, codes, codebook):
    return float(np.mean((np.asarray(vectors)-np.asarray(codebook)[np.asarray(codes,dtype=int)])**2))


if MAIN:
    tests.test_quantization_mse(quantization_mse)

# %%

if MAIN:
    counts,perplexity=code_usage(encoded,3)
    collapsed_counts,collapsed_perplexity=code_usage(np.zeros_like(encoded),3)
    print("Code counts:",counts,"perplexity:",round(perplexity,6))
    print("Collapsed counts:",collapsed_counts,"perplexity:",collapsed_perplexity)
    print("Quantization MSE:",quantization_mse(patch_means(red_image,2),encoded,codebook))
    fig,ax=plt.subplots(figsize=(7,3));positions=np.arange(3)
    ax.bar(positions-.18,counts,width=.36,label=f"Scene vocabulary PPL={perplexity:.2f}")
    ax.bar(positions+.18,collapsed_counts,width=.36,label="Collapsed control PPL=1")
    ax.set(xticks=positions,xticklabels=["Gray","Red","Blue"],ylabel="Token count",title="Usage is not image quality");ax.legend();fig.tight_layout();plt.show()

# %%

def causal_attention(queries, keys, values):
    scores=np.asarray(queries) @ np.asarray(keys).T / np.sqrt(np.asarray(queries).shape[-1])
    allowed=np.tril(np.ones(scores.shape,dtype=bool))
    scores=np.where(allowed,scores,-np.inf)
    weights=np.exp(scores-scores.max(axis=-1,keepdims=True))
    weights/=weights.sum(axis=-1,keepdims=True)
    return weights @ np.asarray(values),weights


if MAIN:
    tests.test_causal_attention(causal_attention)

# %%

def next_token_logits(prefix, position, object_layout, vocabulary_size=3):
    layout=np.asarray(object_layout,dtype=bool).ravel()
    if not layout[position]:
        logits=np.zeros(vocabulary_size);logits[0]=20.
        return logits
    prior_objects=np.flatnonzero(layout[:position])
    source=int(prior_objects[-1]+1) if len(prior_objects) else 0
    length=len(prefix)
    queries=np.zeros((length,length));queries[-1,source]=100*np.sqrt(length)
    keys=np.eye(length)
    values=np.eye(vocabulary_size)[np.asarray(prefix,dtype=int)]
    output,_=causal_attention(queries,keys,values)
    return 20*output[-1]


if MAIN:
    tests.test_next_token_logits(next_token_logits)

# %%

def greedy_image_tokens(condition, object_layout, vocabulary_size=3):
    prefix=[int(condition)]
    for position in range(np.asarray(object_layout).size):
        logits=next_token_logits(prefix,position,object_layout,vocabulary_size)
        prefix.append(int(np.argmax(logits)))
    return np.array(prefix[1:]).reshape(np.asarray(object_layout).shape)


if MAIN:
    tests.test_greedy_image_tokens(greedy_image_tokens)

# %%

if MAIN:
    generated_red=greedy_image_tokens(1,layout)
    generated_blue=greedy_image_tokens(2,layout)
    print("Red and blue grids:");print(generated_red);print(generated_blue)
    # Inspect the copy head just before the third object position (raster position4).
    prefix=[1]+generated_red.ravel()[:4].tolist()
    queries=np.zeros((5,5));queries[-1,2]=100*np.sqrt(5)
    _,head_weights=causal_attention(queries,np.eye(5),np.eye(3)[prefix])
    fig,axes=plt.subplots(1,3,figsize=(10,3.5))
    axes[0].imshow(decode_tokens(generated_red,codebook,2));axes[0].set(title="Red condition",xlabel="x pixel",ylabel="y pixel")
    axes[1].imshow(decode_tokens(generated_blue,codebook,2));axes[1].set(title="Blue condition",xlabel="x pixel",ylabel="y pixel")
    axes[2].imshow(head_weights,vmin=0,vmax=1,cmap='magma');axes[2].set(title="Causal copy head",xlabel="Key: condition + prefix",ylabel="Query position")
    fig.tight_layout();plt.show()

# %%

def regenerate_with_patch(condition, object_layout, patch_position, donor_token, vocabulary_size=3):
    prefix=[int(condition)]
    for position in range(np.asarray(object_layout).size):
        token=int(donor_token) if position==patch_position else int(np.argmax(next_token_logits(prefix,position,object_layout,vocabulary_size)))
        prefix.append(token)
    return np.array(prefix[1:]).reshape(np.asarray(object_layout).shape)


if MAIN:
    tests.test_regenerate_with_patch(regenerate_with_patch)

# %%

def blue_mask_iou(image, truth_mask):
    predicted=np.asarray(image)[...,2]-np.asarray(image)[...,0] > .3
    truth=np.asarray(truth_mask,dtype=bool)
    union=np.count_nonzero(predicted|truth)
    return float(np.count_nonzero(predicted&truth)/union) if union else 1.0


if MAIN:
    tests.test_blue_mask_iou(blue_mask_iou)

# %%

def outside_patch_changes(reference_image, edited_image, patch_mask):
    changed=np.any(np.abs(np.asarray(edited_image)-np.asarray(reference_image))>1e-9,axis=-1)
    return int(np.count_nonzero(changed & ~np.asarray(patch_mask,dtype=bool)))


if MAIN:
    tests.test_outside_patch_changes(outside_patch_changes)

# %%

if MAIN:
    local_grid=generated_red.copy();local_grid[0,0]=2
    regenerated=regenerate_with_patch(1,layout,0,2)
    wrong_region=generated_red.copy();wrong_region[-1,-1]=2
    conditions=[generated_red,local_grid,regenerated,wrong_region]
    images=[decode_tokens(c,codebook,2) for c in conditions]
    truth_pixels=layout.repeat(2,axis=0).repeat(2,axis=1)
    forced_patch=np.zeros((8,8),bool);forced_patch[:2,:2]=True
    scores=[blue_mask_iou(image,truth_pixels) for image in images]
    changed=[outside_patch_changes(images[0],image,forced_patch) for image in images]
    print("Blue-object IoU base/local/regenerated/wrong-region:",scores)
    print("Changed pixels outside first token:",changed)
    utils.image_grid(images,[f"{name}: IoU={score:.2f}" for name,score in zip(["Baseline","Local token edit","Regenerated future","Wrong-region edit"],scores)],"One token can control later tokens through causal continuation")
    plt.show()

# %%

if MAIN:
    play_position=5
    play_grid=regenerate_with_patch(1,layout,play_position,2)
    play_image=decode_tokens(play_grid,codebook,2)
    print("Late-patch blue IoU:",blue_mask_iou(play_image,truth_pixels))
    utils.image_grid([images[0],play_image],["Baseline","Last object token forced blue"],"Forecast causal reach before patching")
    plt.show()

# %%

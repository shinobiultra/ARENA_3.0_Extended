# %%


from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "chapter12_vlm_interpretability").exists())
exercises_dir = root_dir / "chapter12_vlm_interpretability" / "exercises"
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
from part4_visual_token_flow import tests as tests
from part4_visual_token_flow import utils as utils
MAIN = __name__ == "__main__"
np.set_printoptions(precision=3, suppress=True)

# %%

def grid_features(images, grid=2):
    batch,height,width,_ = images.shape
    if height % grid or width % grid:
        raise ValueError("Image dimensions must divide the grid")
    patches = images.reshape(batch,grid,height//grid,grid,width//grid,3).transpose(0,1,3,2,4,5)
    return patches.mean(axis=(3,4))[..., [0,2]].reshape(batch,grid*grid,2)


if MAIN:
    tests.test_grid_features(grid_features)

# %%

def assemble_tokens(features, query_patch):
    h = np.zeros((len(features),6,6))
    h[:,:4,:2] = features
    h[:,:4,2:] = np.eye(4)[None]
    h[:,4,2:] = np.eye(4)[query_patch]
    return h


if MAIN:
    tests.test_assemble_tokens(assemble_tokens)

# %%

def validate_layout(sequence_length, image_positions, answer_position):
    positions = tuple(int(i) for i in image_positions)
    if len(set(positions)) != len(positions) or any(i < 0 or i >= sequence_length for i in positions):
        raise ValueError("Image positions must be unique and in sequence bounds")
    if answer_position < 0 or answer_position >= sequence_length or answer_position in positions:
        raise ValueError("Answer position must be separate and in bounds")
    return positions


if MAIN:
    tests.test_validate_layout(validate_layout)

# %%

if MAIN:
    images = np.stack([utils.render_grid(0,c) for c in [0,1]])
    features = grid_features(images)
    initial = assemble_tokens(features,np.array([0,0]))
    validate_layout(initial.shape[1],range(4),5)
    fig,axes=plt.subplots(1,3,figsize=(10,3))
    for ax,image,title in zip(axes[:2],images,["clean: red target","corrupt: blue target"]):
        ax.imshow(image);ax.set_title(title);ax.axis("off")
    axes[2].imshow(initial[0],aspect="auto",cmap="viridis");axes[2].set(xlabel="red, blue, four spatial coordinates",ylabel="four image / question / answer",title="An inspectable token layout")
    plt.tight_layout();plt.show()

# %%

def scaled_attention(q, k, v):
    scores = q @ k.T / np.sqrt(q.shape[-1])
    weights = np.exp(scores - scores.max(axis=-1,keepdims=True))
    weights /= weights.sum(axis=-1,keepdims=True)
    return weights @ v


if MAIN:
    tests.test_scaled_attention(scaled_attention)

# %%

def knockout_keys(probabilities, blocked):
    out = probabilities.copy()
    out[...,blocked] = 0
    mass = out.sum(axis=-1,keepdims=True)
    if np.any(mass <= 0):
        raise ValueError("No remaining attention mass")
    return out / mass


if MAIN:
    tests.test_knockout_keys(knockout_keys)

# %%

def flow_step(h, layer):
    out = h.copy()
    if layer == 0:
        for b in range(len(h)):
            out[b,4,:2] += scaled_attention(h[b,4:5,2:] * 16,h[b,:4,2:],h[b,:4,:2])[0]
    elif layer == 1:
        out[:,5,:2] += h[:,4,:2]
    elif layer != 2:
        raise ValueError("The organism has transitions 0,1,2")
    return out


if MAIN:
    tests.test_flow_step(flow_step)

# %%

def run_flow(initial):
    cache = {0: initial.copy()}
    for layer in range(3):
        cache[layer+1] = flow_step(cache[layer],layer)
    return cache


if MAIN:
    tests.test_run_flow(run_flow)

# %%

if MAIN:
    clean_cache = run_flow(initial[:1])
    corrupt_cache = run_flow(initial[1:])
    flow = np.stack([clean_cache[layer][0,:,:2].sum(axis=1) for layer in range(4)])
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    axes[0].imshow(flow,aspect="auto",cmap="magma");axes[0].set(xlabel="image0 image1 image2 image3 question answer",ylabel="boundary",title="Where colour content is present")
    attention = np.exp(np.array([[8.,0,0,0]]));attention/=attention.sum(axis=1,keepdims=True)
    blocked = knockout_keys(attention,np.array([True,False,False,False]))
    axes[1].bar(np.arange(4)-.15,attention[0],width=.3,label="normal");axes[1].bar(np.arange(4)+.15,blocked[0],width=.3,label="target-key knockout");axes[1].set(xlabel="visual key index",ylabel="question attention",title="Knockout redistributes mass");axes[1].legend()
    plt.tight_layout();plt.show()

# %%

def candidate_margin(scores, clean_ids, corrupt_ids):
    rows = np.arange(len(scores))
    return scores[rows,clean_ids] - scores[rows,corrupt_ids]


if MAIN:
    tests.test_candidate_margin(candidate_margin)

# %%

def replace_positions(corrupt, clean, positions):
    out = corrupt.copy()
    out[:,positions] = clean[:,positions]
    return out


if MAIN:
    tests.test_replace_positions(replace_positions)

# %%

def normalised_recovery(patched, clean, corrupt):
    denominator = clean - corrupt
    if not np.isfinite(denominator).all() or np.any(denominator <= 0):
        raise ValueError("Recovery needs a finite positive contrast")
    return (patched - corrupt) / denominator


if MAIN:
    tests.test_normalised_recovery(normalised_recovery)

# %%

def patch_sweep(clean_initial, corrupt_initial, position_sets, clean_ids, corrupt_ids):
    clean = run_flow(clean_initial)
    corrupt = run_flow(corrupt_initial)
    clean_margin = candidate_margin(clean[3][:,5,:2],clean_ids,corrupt_ids)
    corrupt_margin = candidate_margin(corrupt[3][:,5,:2],clean_ids,corrupt_ids)
    curves = {}
    for name,positions in position_sets.items():
        curves[name] = []
        for boundary in range(4):
            state = replace_positions(corrupt[boundary],clean[boundary],positions)
            for layer in range(boundary,3):
                state = flow_step(state,layer)
            patched_margin = candidate_margin(state[:,5,:2],clean_ids,corrupt_ids)
            curves[name].append(float(normalised_recovery(patched_margin,clean_margin,corrupt_margin).mean()))
    return curves


if MAIN:
    tests.test_patch_sweep(patch_sweep)

# %%

if MAIN:
    position_sets = {"queried object":[0],"matched background":[1],"question":[4],"last position":[5],"full sequence":list(range(6))}
    curves = patch_sweep(initial[:1],initial[1:],position_sets,np.array([0]),np.array([1]))
    fig,ax=plt.subplots(figsize=(9,4))
    values=np.array(list(curves.values()));im=ax.imshow(values,vmin=0,vmax=1,cmap="viridis",aspect="auto")
    ax.set(xticks=range(4),yticks=range(len(curves)),yticklabels=list(curves),xlabel="cached residual boundary",title="Patching recovery: information moves before the answer does")
    for row in range(len(curves)):
        for col in range(4):ax.text(col,row,f"{values[row,col]:.2f}",ha="center",va="center",color="white" if values[row,col]<.5 else "black")
    fig.colorbar(im,ax=ax,label="normalised red-vs-blue recovery");plt.tight_layout();plt.show()

# %%

def decode_visual_tokens(hidden, readout):
    return (hidden @ readout).argmax(axis=-1)


if MAIN:
    tests.test_decode_visual_tokens(decode_visual_tokens)

# %%

def patchscope_readout(source_hidden, target_initial, source_position, target_position):
    target = target_initial.copy()
    target[:,target_position,:2] = source_hidden[:,source_position,:2]
    return run_flow(target)[3][:,5,:2]


if MAIN:
    tests.test_patchscope_readout(patchscope_readout)

# %%

if MAIN:
    readout = np.zeros((6,2));readout[:2] = np.eye(2)
    lens = np.stack([decode_visual_tokens(clean_cache[layer],readout)[0] for layer in range(4)])
    inspection = assemble_tokens(np.zeros((1,4,2)),np.array([1]))
    conditional = patchscope_readout(clean_cache[0],inspection,0,1)[0]
    wrong_source = patchscope_readout(corrupt_cache[0],inspection,0,1)[0]
    text_only = run_flow(inspection)[3][0,5,:2]
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    axes[0].imshow(lens,aspect="auto",cmap="coolwarm_r",vmin=0,vmax=1);axes[0].set(xlabel="sequence position (zero-state ties are red)",ylabel="boundary",title="A lens can decode unused visual tokens")
    labels=["injected clean object","counterfactual donor","text-only target"]
    axes[1].bar(np.arange(3)-.15,[conditional[0],wrong_source[0],text_only[0]],.3,label="red score");axes[1].bar(np.arange(3)+.15,[conditional[1],wrong_source[1],text_only[1]],.3,label="blue score");axes[1].set(xticks=range(3),xticklabels=labels,ylabel="declared readout score",title="Activation-conditioned decoding controls");axes[1].tick_params(axis="x",labelrotation=15);axes[1].legend()
    plt.tight_layout();plt.show()

# %%

if MAIN:
    play_initial = assemble_tokens(features,np.array([3,3]))
    play_curves = patch_sweep(play_initial[:1],play_initial[1:],{"old object":[0],"new target":[3],"question":[4],"answer":[5]},np.array([1]),np.array([0]))
    fig,ax=plt.subplots(figsize=(8,3));ax.imshow(np.array(list(play_curves.values())),aspect="auto",vmin=0,vmax=1);ax.set(yticks=range(4),yticklabels=list(play_curves),xticks=range(4),xlabel="boundary",title="Changed question: follow the new target");plt.tight_layout();plt.show()

# %%

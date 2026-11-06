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
from part6_multimodal_sparse_features import tests as tests
from part6_multimodal_sparse_features import utils as utils
MAIN = __name__ == "__main__"
np.set_printoptions(precision=3, suppress=True)

# %%

def pixel_statistics(images):
    means = images.mean(axis=(1,2))
    colour = np.eye(3)[means.argmax(axis=1)]
    brightness = images.mean(axis=(1,2,3))[:,None]
    return np.concatenate([colour,brightness],axis=1)


if MAIN:
    tests.test_pixel_statistics(pixel_statistics)

# %%

def word_features(words, vocabulary):
    indices = [vocabulary.index(word) for word in words]
    return np.concatenate([np.eye(len(vocabulary))[indices],np.array([len(word)/10 for word in words])[:,None]],axis=1)


if MAIN:
    tests.test_word_features(word_features)

# %%

if MAIN:
    images, labels, words = utils.render_examples()
    visual = pixel_statistics(images)
    text = word_features(words,list(utils.NAMES))
    fig,axes=plt.subplots(2,3,figsize=(9,5))
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[i]);ax.set_title(f"{words[i]}\nbrightness {visual[i,-1]:.3f}; length {text[i,-1]:.1f}");ax.axis("off")
    fig.suptitle("Paired examples: semantic identity plus modality-specific nuisance")
    plt.tight_layout();plt.show()

# %%

def sae_encode(x, encoder, bias):
    return np.maximum(x @ encoder + bias,0)


if MAIN:
    tests.test_sae_encode(sae_encode)

# %%

def reconstruct(z, decoder, bias):
    return z @ decoder + bias


if MAIN:
    tests.test_reconstruct(reconstruct)

# %%

def reconstruction_error(target, prediction):
    return float(np.mean(np.sum((target-prediction)**2,axis=1)))


if MAIN:
    tests.test_reconstruction_error(reconstruction_error)

# %%

def transcoder_targets(x, weights, bias):
    return np.maximum(x @ weights + bias,0)


if MAIN:
    tests.test_transcoder_targets(transcoder_targets)

# %%

def fit_decoder(z, targets, ridge=1e-6):
    zc = z-z.mean(axis=0)
    yc = targets-targets.mean(axis=0)
    if ridge < 0:
        raise ValueError("Ridge must be nonnegative")
    if ridge == 0:
        decoder = np.linalg.lstsq(zc,yc,rcond=None)[0]
    else:
        decoder = np.linalg.solve(zc.T @ zc + ridge*np.eye(z.shape[1]),zc.T @ yc)
    intercept = targets.mean(axis=0)-z.mean(axis=0) @ decoder
    return decoder,intercept


if MAIN:
    tests.test_fit_decoder(fit_decoder)

# %%

if MAIN:
    # A semantic-only sparse code deliberately omits brightness.
    semantic_code = visual[:,:3]
    mlp_weights = np.array([[1.,0,0],[0,1,0],[0,0,1],[0,0,0]])
    mlp_bias = np.zeros(3)
    mlp_target = transcoder_targets(visual,mlp_weights,mlp_bias)
    sae_decoder,sae_bias = fit_decoder(semantic_code,visual)
    trans_decoder,trans_bias = fit_decoder(semantic_code,mlp_target)
    sae_errors = (reconstruct(semantic_code,sae_decoder,sae_bias)-visual)**2
    trans_errors = (reconstruct(semantic_code,trans_decoder,trans_bias)-mlp_target)**2
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    axes[0].imshow(sae_errors,aspect="auto",cmap="magma");axes[0].set(xlabel="RGB identity then brightness",ylabel="paired example",title="SAE target penalises omitted brightness")
    axes[1].imshow(trans_errors,aspect="auto",cmap="magma",vmin=0,vmax=max(float(sae_errors.max()),1e-6));axes[1].set(xlabel="three answer-MLP coordinates",ylabel="paired example",title="Transcoder target ignores brightness")
    plt.tight_layout();plt.show()
    print("SAE target error",round(reconstruction_error(visual,reconstruct(semantic_code,sae_decoder,sae_bias)),6))
    print("transcoder target error",round(reconstruction_error(mlp_target,reconstruct(semantic_code,trans_decoder,trans_bias)),6))

# %%

def crosscoder_encode(x, y, w_x, w_y, bias):
    return np.maximum(x @ w_x + y @ w_y + bias,0)


if MAIN:
    tests.test_crosscoder_encode(crosscoder_encode)

# %%

def crosscoder_loss(x, y, x_hat, y_hat, z, alpha):
    return reconstruction_error(x,x_hat)+reconstruction_error(y,y_hat)+alpha*float(z.sum(axis=1).mean())


if MAIN:
    tests.test_crosscoder_loss(crosscoder_loss)

# %%

def feature_overlap(image_z, text_z, threshold=0.1):
    left = image_z > threshold
    right = text_z > threshold
    intersection = (left & right).sum(axis=0)
    union = (left | right).sum(axis=0)
    return intersection / np.maximum(union,1)


if MAIN:
    tests.test_feature_overlap(feature_overlap)

# %%

def derange_pairs(x):
    if len(x) < 2:
        raise ValueError("Need at least two paired examples")
    return np.roll(x,1,axis=0).copy()


if MAIN:
    tests.test_derange_pairs(derange_pairs)

# %%

if MAIN:
    w_x = np.zeros((4,5));w_y = np.zeros((4,5))
    w_x[:3,:3] = .5*np.eye(3);w_y[:3,:3] = .5*np.eye(3)
    w_x[3,3] = 1;w_y[3,4] = 1
    z = crosscoder_encode(visual,text,w_x,w_y,np.zeros(5))
    image_z = crosscoder_encode(visual,np.zeros_like(text),w_x,w_y,np.zeros(5))
    text_z = crosscoder_encode(np.zeros_like(visual),text,w_x,w_y,np.zeros(5))
    d_x = np.zeros((5,4));d_y = np.zeros((5,4))
    d_x[:3,:3] = np.eye(3);d_y[:3,:3] = np.eye(3);d_x[3,3]=1;d_y[4,3]=1
    objective = crosscoder_loss(visual,text,reconstruct(z,d_x,np.zeros(4)),reconstruct(z,d_y,np.zeros(4)),z,.01)
    print("known crosscoder objective",round(objective,6))
    feature_names=["red object","green object","blue object","brightness","text length"]
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].imshow(image_z,aspect="auto",cmap="magma");axes[0].set(xlabel="shared semantic / image / text features",ylabel="image row",title="Image-only codes")
    axes[1].imshow(text_z,aspect="auto",cmap="magma");axes[1].set(xlabel="same five latent indices",ylabel="paired word row",title="Text-only codes")
    idx=np.arange(5)
    axes[2].bar(idx-.17,feature_overlap(image_z,text_z),.34,label="matched");axes[2].bar(idx+.17,feature_overlap(image_z,derange_pairs(text_z)),.34,label="shuffled pair");axes[2].set(xticks=idx,xticklabels=feature_names,ylabel="support Jaccard",title="Shared support needs correct pairing");axes[2].tick_params(axis="x",labelrotation=60);axes[2].legend()
    plt.tight_layout();plt.show()

# %%

def feature_edges(source_decoder, downstream_encoder):
    return source_decoder @ downstream_encoder


if MAIN:
    tests.test_feature_edges(feature_edges)

# %%

def path_contribution(z, edges, contrast_readout):
    return z[:,:,None] * edges[None,:,:] * contrast_readout[None,None,:]


if MAIN:
    tests.test_path_contribution(path_contribution)

# %%

def latent_ablation(actual_hidden, z, decoder, feature):
    return actual_hidden - z[:,feature:feature+1] * decoder[feature:feature+1]


if MAIN:
    tests.test_latent_ablation(latent_ablation)

# %%

if MAIN:
    edges = feature_edges(d_x,mlp_weights)
    contrast = np.array([1.,0.,-1.])
    contributions = path_contribution(z,edges,contrast)
    baseline = transcoder_targets(visual,mlp_weights,mlp_bias) @ contrast
    red_ablated = transcoder_targets(latent_ablation(visual,z,d_x,0),mlp_weights,mlp_bias) @ contrast
    nuisance_ablated = transcoder_targets(latent_ablation(visual,z,d_x,3),mlp_weights,mlp_bias) @ contrast
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].imshow(edges,aspect="auto",cmap="Blues",vmin=0,vmax=1);axes[0].set(yticks=range(5),yticklabels=feature_names,xticks=range(3),xticklabels=["red answer","green answer","blue answer"],title="Exact sparse-feature edge graph")
    axes[1].imshow(contributions[0],aspect="auto",cmap="coolwarm",vmin=-1,vmax=1);axes[1].set(xlabel="answer feature",ylabel="source feature",title="Red-card contributions to red−blue")
    axes[2].plot(baseline,"o-",label="actual hidden state");axes[2].plot(red_ablated,"s-",label="red-feature ablation");axes[2].plot(nuisance_ablated,"--",label="brightness-feature control");axes[2].set(xlabel="same ordered paired example",ylabel="red−blue answer contrast",title="Feature effect agrees with the known route");axes[2].legend()
    plt.tight_layout();plt.show()

# %%

if MAIN:
    play_weights=mlp_weights.copy();play_weights[3,0]=2.0
    play_edges=feature_edges(d_x,play_weights)
    play_baseline=transcoder_targets(visual,play_weights,mlp_bias) @ contrast
    play_removed=transcoder_targets(latent_ablation(visual,z,d_x,3),play_weights,mlp_bias) @ contrast
    fig,axes=plt.subplots(1,2,figsize=(10,3.5));axes[0].imshow(play_edges,aspect="auto",cmap="Blues");axes[0].set(xlabel="answer feature",ylabel="source feature",title="Changed task gives brightness a route")
    axes[1].plot(play_baseline,"o-",label="brightness-dependent task");axes[1].plot(play_removed,"s--",label="brightness ablated");axes[1].set(xlabel="same example",ylabel="red−blue contrast",title="Predict a formerly absent causal effect");axes[1].legend();plt.tight_layout();plt.show()

# %%

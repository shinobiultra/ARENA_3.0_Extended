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
from part2_clip_saes_typographic import tests as tests
from part2_clip_saes_typographic import utils as utils
MAIN = __name__ == "__main__"
np.set_printoptions(precision=3, suppress=True)

# %%

def normalise_rows(x, eps=1e-12):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


if MAIN:
    tests.test_normalise_rows(normalise_rows)

# %%

def image_channels(images, templates):
    half = images.shape[1] // 2
    rgb = images[:, :half].mean(axis=(1, 2))
    rgb = rgb / np.maximum(rgb.sum(axis=-1, keepdims=True), 1e-12)
    errors = ((images[:, None, half:] - templates[None]) ** 2).mean(axis=(2, 3, 4))
    word_ids = errors.argmin(axis=1)
    words = np.zeros((len(images), 3))
    valid = word_ids < 3
    words[np.arange(len(images))[valid], word_ids[valid]] = 1
    return np.concatenate([rgb, words], axis=1)


if MAIN:
    tests.test_image_channels(image_channels)

# %%

def contrastive_logits(images, captions, scale=1.0):
    return scale * normalise_rows(images) @ normalise_rows(captions).T


if MAIN:
    tests.test_contrastive_logits(contrastive_logits)

# %%

if MAIN:
    object_ids = np.repeat(np.arange(3), 4)
    word_ids = np.tile(np.array([-1, 0, 1, 2]), 3)
    images = np.stack([utils.render_card(int(o), None if w < 0 else int(w)) for o,w in zip(object_ids,word_ids)])
    channels = image_channels(images, utils.word_templates())
    mixing = np.linalg.qr(np.random.default_rng(7).normal(size=(6,6)))[0]
    text_gain = 1.6
    weighted_channels = channels * np.array([1,1,1,text_gain,text_gain,text_gain])
    x = weighted_channels @ mixing
    captions = np.concatenate([np.eye(3), np.eye(3)], axis=1) @ mixing
    scores = contrastive_logits(x, captions)
    fig, axes = plt.subplots(3,4,figsize=(10,7))
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[i]); ax.set_title(f"object {utils.NAMES[object_ids[i]]}\nretrieved {utils.NAMES[scores[i].argmax()]}",fontsize=8);ax.axis("off")
    fig.suptitle("Rendered object/word conflicts: every row has the same object")
    plt.tight_layout();plt.show()

# %%

def encode_sae(x, w_e, b_e):
    return np.maximum(x @ w_e + b_e, 0)


if MAIN:
    tests.test_encode_sae(encode_sae)

# %%

def decode_sae(z, w_d, b_d):
    return z @ w_d + b_d


if MAIN:
    tests.test_decode_sae(decode_sae)

# %%

def sparse_loss(x, reconstruction, z, alpha):
    return float(np.mean(np.sum((x - reconstruction)**2, axis=1)) + alpha * np.mean(z.sum(axis=1)))


if MAIN:
    tests.test_sparse_loss(sparse_loss)

# %%

def sae_step(x, w_e, b_e, w_d, b_d, alpha, lr):
    pre = x @ w_e + b_e
    z = np.maximum(pre, 0)
    residual = z @ w_d + b_d - x
    d_reconstruction = 2 * residual / len(x)
    d_pre = (d_reconstruction @ w_d.T + alpha / len(x)) * (pre > 0)
    return (w_e - lr * x.T @ d_pre,
            b_e - lr * d_pre.sum(axis=0),
            w_d - lr * z.T @ d_reconstruction,
            b_d - lr * d_reconstruction.sum(axis=0))


if MAIN:
    tests.test_sae_step(sae_step)

# %%

def fit_sae(x, width=12, steps=400, lr=0.04, alpha=0.01, seed=0):
    rng = np.random.default_rng(seed)
    w_e = rng.normal(scale=0.2, size=(x.shape[1], width))
    b_e = np.full(width, 0.1)
    w_d = w_e.T.copy()
    b_d = np.zeros(x.shape[1])
    losses = []
    for _ in range(steps):
        z = encode_sae(x, w_e, b_e)
        losses.append(sparse_loss(x, decode_sae(z,w_d,b_d), z, alpha))
        w_e,b_e,w_d,b_d = sae_step(x,w_e,b_e,w_d,b_d,alpha,lr)
    return w_e,b_e,w_d,b_d,losses


if MAIN:
    tests.test_fit_sae(fit_sae)

# %%

if MAIN:
    w_e,b_e,w_d,b_d,losses = fit_sae(x)
    z = encode_sae(x,w_e,b_e)
    reconstruction = decode_sae(z,w_d,b_d)
    fig, axes = plt.subplots(1,2,figsize=(11,3.5))
    axes[0].plot(losses,label="SAE reconstruction + sparsity");axes[0].axhline(losses[0],ls=":",color="grey",label="initial objective");axes[0].set(xlabel="full-batch CPU step",ylabel="objective",title="A small bounded fit");axes[0].legend()
    axes[1].imshow(z,aspect="auto",cmap="magma");axes[1].set(xlabel="latent",ylabel="rendered card",title="Activations: no concept labels assumed")
    plt.tight_layout();plt.show()
    print("reconstruction squared error per card",np.mean(np.sum((reconstruction-x)**2,axis=1)).round(4))

# %%

def top_examples(activations, feature, k=4):
    return np.argsort(-activations[:, feature], kind="stable")[:k]


if MAIN:
    tests.test_top_examples(top_examples)

# %%

def feature_direction(w_d, feature):
    row = w_d[feature]
    return row / max(np.linalg.norm(row), 1e-12)


if MAIN:
    tests.test_feature_direction(feature_direction)

# %%

def ablate_direction(x, unit_direction, strength=1.0):
    return x - strength * (x @ unit_direction)[:,None] * unit_direction[None]


if MAIN:
    tests.test_ablate_direction(ablate_direction)

# %%

if MAIN:
    # Choose on the already-seen gallery; this is a labelled diagnostic, not discovery on TEST.
    word_triangle = (word_ids == 2).astype(float)
    centered = z - z.mean(axis=0)
    correlation = centered.T @ (word_triangle-word_triangle.mean())
    correlation /= np.maximum(np.linalg.norm(centered,axis=0)*np.linalg.norm(word_triangle-word_triangle.mean()),1e-12)
    feature = int(correlation.argmax())
    direction = feature_direction(w_d, feature)
    chosen = top_examples(z,feature,4)
    fig, axes = plt.subplots(1,4,figsize=(10,3))
    for ax,i in zip(axes,chosen):
        ax.imshow(images[i]);ax.set_title(f"card {i}: latent {z[i,feature]:.2f}");ax.axis("off")
    fig.suptitle(f"Top cards for monitored triangle-word latent {feature}")
    plt.tight_layout();plt.show()

# %%

def forced_choice_accuracy(scores, targets):
    return float(np.mean(scores.argmax(axis=1) == targets))


if MAIN:
    tests.test_forced_choice_accuracy(forced_choice_accuracy)

# %%

def tradeoff_curve(x, captions, targets, direction, strengths):
    return np.array([forced_choice_accuracy(contrastive_logits(ablate_direction(x,direction,a),captions),targets) for a in strengths])


if MAIN:
    tests.test_tradeoff_curve(tradeoff_curve)

# %%

if MAIN:
    strengths = np.linspace(0,1,11)
    clean = word_ids < 0
    attack = (word_ids == 2) & (object_ids != 2)
    text_images = np.stack([utils.render_card(0,j,False) for j in range(3)])
    text_x = (image_channels(text_images,utils.word_templates()) * np.array([1,1,1,text_gain,text_gain,text_gain])) @ mixing
    wrong_direction = feature_direction(w_d,int(np.argmin(np.abs(correlation))))
    fig, ax = plt.subplots(figsize=(9,4))
    for label,values,targets,style in [
        ("clean objects",x[clean],object_ids[clean],"-o"),
        ("triangle-label attacks",x[attack],object_ids[attack],"-s"),
        ("legitimate text-only reading",text_x,np.arange(3),"-^")]:
        ax.plot(strengths,tradeoff_curve(values,captions,targets,direction,strengths),style,label=label)
    ax.plot(strengths,tradeoff_curve(x[attack],captions,object_ids[attack],wrong_direction,strengths),"--",color="grey",label="wrong-feature attack control")
    ax.set(xlabel="removed projection fraction",ylabel="forced-choice accuracy",ylim=(-.05,1.05),title="Typography mitigation has a task-dependent cost")
    ax.legend();plt.tight_layout();plt.show()
    print("counts: clean",clean.sum(),"attacks",attack.sum(),"text-only",len(text_x))

# %%

if MAIN:
    play_text_gain = 0.7
    play_x = (channels * np.array([1,1,1,play_text_gain,play_text_gain,play_text_gain])) @ mixing
    play_scores = contrastive_logits(play_x,captions)
    fig,ax=plt.subplots(figsize=(5,4));ax.imshow(play_scores,aspect="auto",cmap="coolwarm");ax.set(xlabel="caption: circle / square / triangle",ylabel="same ordered card",title=f"Changed text gain {play_text_gain}; compare original gallery")
    plt.tight_layout();plt.show()

# %%

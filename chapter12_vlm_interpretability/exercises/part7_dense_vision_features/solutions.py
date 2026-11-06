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
from part7_dense_vision_features import tests as tests
from part7_dense_vision_features import utils as utils
MAIN = __name__ == "__main__"
np.set_printoptions(precision=3, suppress=True)

# %%

def patch_means(image, patch=8):
    height,width,channels = image.shape
    if height % patch or width % patch:
        raise ValueError("Image shape must divide patch size")
    return image.reshape(height//patch,patch,width//patch,patch,channels).mean(axis=(1,3))


if MAIN:
    tests.test_patch_means(patch_means)

# %%

def teacher_student_loss(teacher_logits, student_logits, centre, teacher_temperature=0.04, student_temperature=0.1):
    if teacher_temperature <= 0 or student_temperature <= 0:
        raise ValueError("Temperatures must be positive")
    t = (teacher_logits-centre)/teacher_temperature
    t -= t.max(axis=1,keepdims=True)
    p = np.exp(t);p /= p.sum(axis=1,keepdims=True)
    s = student_logits/student_temperature
    s -= s.max(axis=1,keepdims=True)
    log_q = s-np.log(np.exp(s).sum(axis=1,keepdims=True))
    return float(-(p*log_q).sum(axis=1).mean())


if MAIN:
    tests.test_teacher_student_loss(teacher_student_loss)

# %%

if MAIN:
    source_image,source_labels = utils.part_image(0)
    target_image,target_labels = utils.part_image(2)
    source_grid = patch_means(source_image)
    target_grid = patch_means(target_image)
    source = source_grid.reshape(-1,3)
    target = target_grid.reshape(-1,3)
    prototype_logits = np.eye(3)*3
    paired_loss = teacher_student_loss(prototype_logits,prototype_logits,np.zeros(3),.5,.5)
    shuffled_loss = teacher_student_loss(prototype_logits,np.roll(prototype_logits,1,axis=0),np.zeros(3),.5,.5)
    fig,axes=plt.subplots(1,3,figsize=(12,3))
    axes[0].imshow(source_image);axes[0].set_title("source: body / cap / handle");axes[0].axis("off")
    axes[1].imshow(target_image);axes[1].set_title("same parts, translated");axes[1].axis("off")
    axes[2].bar(["paired view prototypes","shuffled pairing"],[paired_loss,shuffled_loss]);axes[2].set(ylabel="teacher→student cross-entropy",title="Objective component sanity check");axes[2].tick_params(axis="x",labelrotation=15)
    plt.tight_layout();plt.show()
    print("prototype CE: paired",round(paired_loss,6),"shuffled",round(shuffled_loss,6))

# %%

def fit_pca(features, k=3):
    if k < 1 or k > min(features.shape):
        raise ValueError("Invalid component count")
    mean = features.mean(axis=0)
    _,_,vt = np.linalg.svd(features-mean,full_matrices=False)
    components = vt[:k].copy()
    for row in components:
        if row[np.abs(row).argmax()] < 0:
            row *= -1
    return mean,components


if MAIN:
    tests.test_fit_pca(fit_pca)

# %%

def project_pca(features, mean, components):
    return (features-mean) @ components.T


if MAIN:
    tests.test_project_pca(project_pca)

# %%

def cosine_matches(source, target):
    s = source / np.maximum(np.linalg.norm(source,axis=1,keepdims=True),1e-12)
    t = target / np.maximum(np.linalg.norm(target,axis=1,keepdims=True),1e-12)
    scores = s @ t.T
    indices = scores.argmax(axis=1)
    return indices,scores[np.arange(len(source)),indices]


if MAIN:
    tests.test_cosine_matches(cosine_matches)

# %%

def mutual_matches(forward, backward):
    return backward[forward] == np.arange(len(forward))


if MAIN:
    tests.test_mutual_matches(mutual_matches)

# %%

def correspondence_accuracy(source_labels, target_labels, matches, source_mask):
    if not np.any(source_mask):
        raise ValueError("No selected source patches")
    return float(np.mean(source_labels[source_mask] == target_labels[matches[source_mask]]))


if MAIN:
    tests.test_correspondence_accuracy(correspondence_accuracy)

# %%

if MAIN:
    mean,components = fit_pca(source)
    pca_source = project_pca(source,mean,components)
    pca_target = project_pca(target,mean,components)
    low,high = pca_source.min(axis=0),pca_source.max(axis=0)
    forward,similarities = cosine_matches(source,target)
    backward,_ = cosine_matches(target,source)
    mutual = mutual_matches(forward,backward)
    foreground = source_labels.reshape(-1)>0
    shuffle = np.random.default_rng(5).permutation(len(target))
    shuffled_forward,_ = cosine_matches(source,target[shuffle])
    matched_accuracy = correspondence_accuracy(source_labels.ravel(),target_labels.ravel(),forward,foreground)
    shuffled_accuracy = correspondence_accuracy(source_labels.ravel(),target_labels.ravel(),shuffled_forward,foreground)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    axes[0].imshow(utils.grid_rgb(pca_source,source_labels.shape,low,high));axes[0].set_title("source PCA in one fitted basis");axes[0].axis("off")
    axes[1].imshow(utils.grid_rgb(pca_target,target_labels.shape,low,high));axes[1].set_title("target in the SAME PCA basis");axes[1].axis("off")
    axes[2].bar(["cosine part match","spatially shuffled control"],[matched_accuracy,shuffled_accuracy]);axes[2].set(ylabel="source-foreground part accuracy",ylim=(0,1.05),title="Similarity requires spatial correspondence");axes[2].tick_params(axis="x",labelrotation=15)
    plt.tight_layout();plt.show()
    print("foreground patches",foreground.sum(),"mutual foreground matches",(mutual & foreground).sum(),"part accuracy",matched_accuracy,"shuffled",shuffled_accuracy)

# %%

def norm_outliers(features, multiplier=5.0):
    norms = np.linalg.norm(features,axis=1)
    return norms > multiplier * np.median(norms)


if MAIN:
    tests.test_norm_outliers(norm_outliers)

# %%

def attention_weights(query, keys):
    scores = keys @ query / np.sqrt(keys.shape[1])
    scores -= scores.max()
    weights = np.exp(scores)
    return weights / weights.sum()


if MAIN:
    tests.test_attention_weights(attention_weights)

# %%

if MAIN:
    artifact = target.copy();artifact[0] *= 100
    query = np.array([10.,0,0])
    raw_attention = attention_weights(query,artifact)
    unit_artifact = artifact/np.maximum(np.linalg.norm(artifact,axis=1,keepdims=True),1e-12)
    unit_attention = attention_weights(query,unit_artifact)
    outliers = norm_outliers(artifact)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5))
    for ax,values,title in zip(axes,[np.linalg.norm(artifact,axis=1),raw_attention,unit_attention],["injected background norm","raw-key red-query attention","unit-key diagnostic control"]):
        im=ax.imshow(values.reshape(target_labels.shape),cmap="magma");ax.set(title=title,xlabel="patch column",ylabel="patch row");fig.colorbar(im,ax=ax,shrink=.7)
    plt.tight_layout();plt.show()
    print("outlier indices",np.flatnonzero(outliers).tolist(),"raw artifact attention",round(raw_attention[0],6),"unit-key artifact attention",round(unit_attention[0],6))

# %%

def masked_pool(features, mask):
    if not np.any(mask):
        raise ValueError("Prompt region is empty")
    return features[mask].mean(axis=0)


if MAIN:
    tests.test_masked_pool(masked_pool)

# %%

def prompt_mask(features, positive, negative, margin=0.0):
    f = features/np.maximum(np.linalg.norm(features,axis=1,keepdims=True),1e-12)
    p = positive/max(np.linalg.norm(positive),1e-12)
    n = negative/max(np.linalg.norm(negative),1e-12)
    scores = f @ p - f @ n
    return scores > margin,scores


if MAIN:
    tests.test_prompt_mask(prompt_mask)

# %%

def mask_iou(predicted, target):
    intersection = np.logical_and(predicted,target).sum()
    union = np.logical_or(predicted,target).sum()
    return float(intersection/union) if union else 1.0


if MAIN:
    tests.test_mask_iou(mask_iou)

# %%

def replace_region(target, donor, mask):
    out = target.copy()
    out[mask] = donor[mask]
    return out


if MAIN:
    tests.test_replace_region(replace_region)

# %%

if MAIN:
    truth = target_labels.ravel()==1
    positive = masked_pool(target,truth)
    negative = masked_pool(target,target_labels.ravel()==0)
    wrong_positive = masked_pool(target,target_labels.ravel()==3)
    corrupt = target.copy();corrupt[truth] = utils.PALETTE[3]
    background = np.zeros(len(target),dtype=bool)
    background[np.flatnonzero(target_labels.ravel()==0)[:int(truth.sum())]] = True
    object_patched = replace_region(corrupt,target,truth)
    bg_patched = replace_region(corrupt,target,background)
    conditions={"clean target":target,"corrupt red features":corrupt,"red-part patch":object_patched,"equal-size background patch":bg_patched}
    masks={name:prompt_mask(values,positive,negative)[0] for name,values in conditions.items()}
    masks["wrong blue prompt"] = prompt_mask(target,wrong_positive,negative)[0]
    fig,axes=plt.subplots(2,3,figsize=(11,6))
    axes.flat[0].imshow(target_image);axes.flat[0].set_title("Pixel image; target is red PART");axes.flat[0].axis("off")
    for ax,(name,mask) in zip(list(axes.flat)[1:],masks.items()):
        ax.imshow(mask.reshape(target_labels.shape),vmin=0,vmax=1,cmap="gray");ax.set_title(f"{name}\nIoU {mask_iou(mask,truth):.2f}");ax.axis("off")
    fig.suptitle("Prompt mask reads local features: object vs matched-background replacement")
    plt.tight_layout(rect=(0,0,1,.94),h_pad=3.0);plt.show()
    print("mask IoUs",[(name,mask_iou(mask,truth)) for name,mask in masks.items()])

# %%

if MAIN:
    whole_object = target_labels.ravel()>0
    part_mask = masks["clean target"]
    whole_iou = mask_iou(part_mask,whole_object)
    fig,axes=plt.subplots(1,2,figsize=(8,3));axes[0].imshow(part_mask.reshape(target_labels.shape),cmap="gray");axes[0].set_title("Unchanged predicted red-part mask");axes[1].imshow(whole_object.reshape(target_labels.shape),cmap="gray");axes[1].set_title(f"Changed whole-object target; IoU {whole_iou:.2f}")
    for ax in axes:ax.axis("off")
    plt.tight_layout();plt.show()

# %%

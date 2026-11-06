# %%


import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

chapter = "chapter13_image_generation_interpretability"
section = "part4_diffusion_saes"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).is_dir())
exercises_dir = root_dir / chapter / "exercises"
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part4_diffusion_saes.tests as tests
import part4_diffusion_saes.utils as utils
MAIN = __name__ == "__main__"

# %%

def sparse_encode(activations, encoder, threshold):
    return np.maximum(np.asarray(activations) @ np.asarray(encoder).T - threshold, 0.)


if MAIN:
    tests.test_sparse_encode(sparse_encode)

# %%

def sparse_decode(codes, decoder):
    return np.asarray(codes) @ np.asarray(decoder)


if MAIN:
    tests.test_sparse_decode(sparse_decode)

# %%

def render_sparse_scene(activations):
    x = np.atleast_2d(np.asarray(activations,dtype=np.float64))
    left = (x[:,0]+x[:,2])/np.sqrt(2)
    right = (x[:,1]+x[:,3])/np.sqrt(2)
    background = (x[:,0]+x[:,1]-x[:,2]-x[:,3])/2
    stripe = (x[:,0]-x[:,1]-x[:,2]+x[:,3])/2
    images = np.ones((len(x),8,10,3))*(.2+.1*background[:,None,None,None])
    for i in range(len(x)):
        images[i,2:6,1:4] = [.2+.7*left[i],.2,.2]
        images[i,2:6,6:9] = [.2,.2,.2+.7*right[i]]
        images[i,3:4] += .1*stripe[i]
    return np.clip(images,0,1)


if MAIN:
    tests.test_render_sparse_scene(render_sparse_scene)

# %%

def reconstruction_mse(activations, reconstruction):
    return float(np.mean((np.asarray(activations)-np.asarray(reconstruction))**2))


if MAIN:
    tests.test_reconstruction_mse(reconstruction_mse)

# %%

def sae_objective(activations, reconstruction, codes, penalty):
    return float(np.mean((np.asarray(activations)-np.asarray(reconstruction))**2) + penalty*np.mean(np.abs(codes)))


if MAIN:
    tests.test_sae_objective(sae_objective)

# %%

def active_features(codes):
    return np.count_nonzero(np.asarray(codes)>0,axis=-1)


if MAIN:
    tests.test_active_features(active_features)

# %%

def decoder_gradient(activations, codes, decoder):
    reconstruction = np.asarray(codes) @ np.asarray(decoder)
    return 2*np.asarray(codes).T @ (reconstruction-np.asarray(activations)) / np.asarray(activations).size


if MAIN:
    tests.test_decoder_gradient(decoder_gradient)

# %%

if MAIN:
    decoder = np.array([[1,0,1,0],[0,1,0,1],[1,1,-1,-1]],dtype=float)
    decoder[:2] /= np.sqrt(2);decoder[2] /= 2
    residual_axis = np.array([1.,-1.,-1.,1.])/2
    latent_codes = np.array([[.8,.6,.2],[.2,.8,.6],[.9,0.,.1]])
    activations = latent_codes @ decoder + .12*residual_axis
    thresholds = [0.,.1,.4]
    errors=[];activity=[]
    for threshold in thresholds:
        z = sparse_encode(activations,decoder,threshold)
        errors.append(reconstruction_mse(activations,sparse_decode(z,decoder)))
        activity.append(active_features(z).mean())
    print("Thresholds:",thresholds,"MSE:",np.round(errors,6),"mean L0:",np.round(activity,3))
    fig,axes=plt.subplots(1,2,figsize=(9,3))
    axes[0].plot(thresholds,errors,'o-',label="Thresholded SAE");axes[0].axhline(np.mean(activations**2),ls='--',label="Zero reconstruction")
    axes[0].set(xlabel="Threshold",ylabel="Activation MSE",title="Fidelity is separate from sparsity");axes[0].legend()
    axes[1].plot(thresholds,activity,'o-',label="Positive feature count");axes[1].axhline(3,ls='--',label="Dense 3-feature baseline")
    axes[1].set(xlabel="Threshold",ylabel="Mean active features",title="Threshold trade-off");axes[1].legend();fig.tight_layout();plt.show()

# %%

def edit_feature(codes, feature, value):
    edited = np.array(codes,copy=True)
    edited[:,feature] = value
    return edited


if MAIN:
    tests.test_edit_feature(edit_feature)

# %%

def residual_preserving_patch(activations, original_codes, edited_codes, decoder):
    return np.asarray(activations)+(np.asarray(edited_codes)-np.asarray(original_codes)) @ np.asarray(decoder)


if MAIN:
    tests.test_residual_preserving_patch(residual_preserving_patch)

# %%

def masked_contrast(images, mask, channel):
    pixels = np.asarray(images)[:,np.asarray(mask,dtype=bool)]
    return (pixels[...,channel]-pixels[...,1]).mean(axis=1)


if MAIN:
    tests.test_masked_contrast(masked_contrast)

# %%

def outside_mask_mae(reference, edited, mask):
    return np.abs(np.asarray(reference)[:,~np.asarray(mask,dtype=bool)]-np.asarray(edited)[:,~np.asarray(mask,dtype=bool)]).mean(axis=(1,2))


if MAIN:
    tests.test_outside_mask_mae(outside_mask_mae)

# %%

if MAIN:
    left_mask = np.zeros((8,10),bool);left_mask[2:6,1:4]=True
    right_mask = np.zeros((8,10),bool);right_mask[2:6,6:9]=True
    x = activations[:1].copy()
    z = sparse_encode(x,decoder,.1)
    ablated = edit_feature(z,0,0.)
    nuisance = edit_feature(z,2,0.)
    patched = residual_preserving_patch(x,z,ablated,decoder)
    wrong_feature = residual_preserving_patch(x,z,nuisance,decoder)
    reconstruction_only = sparse_decode(ablated,decoder)
    images = render_sparse_scene(np.concatenate([x,patched,wrong_feature,reconstruction_only]))
    print("Target red contrast:",np.round(masked_contrast(images,left_mask,0),3))
    print("Right blue contrast:",np.round(masked_contrast(images,right_mask,2),3))
    print("Outside-target MAE:",np.round(outside_mask_mae(np.repeat(images[:1],4,axis=0),images,left_mask),5))
    utils.image_grid(images,["Original","Feature0 ablated","Nuisance feature ablated","Reconstruction replacement"],"Residual-preserving ablation keeps unrelated structure")
    plt.show()

# %%

def feature_contrast(codes, labels):
    labels = np.asarray(labels,dtype=bool)
    if not labels.any() or labels.all():
        raise ValueError("both label groups required")
    return np.asarray(codes)[labels].mean(axis=0)-np.asarray(codes)[~labels].mean(axis=0)


if MAIN:
    tests.test_feature_contrast(feature_contrast)

# %%

if MAIN:
    cohort_codes=np.array([[.7,.2,.9],[.6,.4,.8],[.1,.3,0.],[.2,.5,.1]])
    labels=np.array([1,1,0,0],bool)
    association=feature_contrast(cohort_codes,labels)
    # Previously measured intervention scores, not values inferred from association.
    causal_red_change=[]
    for feature in range(3):
        changed=residual_preserving_patch(x,z,edit_feature(z,feature,0.),decoder)
        causal_red_change.append(float(masked_contrast(render_sparse_scene(changed),left_mask,0)[0]-masked_contrast(render_sparse_scene(x),left_mask,0)[0]))
    print("Association:",association.round(3),"ablated target change:",np.round(causal_red_change,3))
    fig,ax=plt.subplots(figsize=(7,3.5));positions=np.arange(3)
    ax.bar(positions-.18,association,width=.36,label="Label association")
    ax.bar(positions+.18,causal_red_change,width=.36,label="Actual red-pixel intervention")
    ax.axhline(0,color='black',lw=.7);ax.set(xticks=positions,xticklabels=["Left color","Right color","Background"],ylabel="Signed change (different units)",title="Association is not the edit effect")
    ax.legend();fig.tight_layout();plt.show()

# %%

if MAIN:
    play_threshold=.4
    play_codes=sparse_encode(x,decoder,play_threshold)
    play_patch=residual_preserving_patch(x,play_codes,edit_feature(play_codes,0,0.),decoder)
    print("Remaining red contrast:",masked_contrast(render_sparse_scene(play_patch),left_mask,0).round(3))
    print("Reversed-label associations:",feature_contrast(cohort_codes,~labels).round(3))
    utils.image_grid(render_sparse_scene(np.concatenate([x,play_patch])),["Original","Higher-threshold ablation"],"Predict the residual's remaining color")
    plt.show()

# %%

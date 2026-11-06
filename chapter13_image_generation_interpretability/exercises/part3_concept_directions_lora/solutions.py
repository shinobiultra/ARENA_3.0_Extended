# %%


import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

chapter = "chapter13_image_generation_interpretability"
section = "part3_concept_directions_lora"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).is_dir())
exercises_dir = root_dir / chapter / "exercises"
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part3_concept_directions_lora.tests as tests
import part3_concept_directions_lora.utils as utils
MAIN = __name__ == "__main__"

# %%

def render_scene(states):
    states = np.atleast_2d(np.asarray(states, dtype=np.float64))
    images = np.ones((len(states), 8, 8, 3)) * (0.25 + 0.1*states[:, 2, None, None, None])
    for i, (color, stripe, background) in enumerate(states):
        images[i, 2:6, 1:5] = [0.5+0.25*color, 0.3, 0.5-0.25*color]
        images[i, 3:5, 1:5] += 0.05*stripe
    return np.clip(images, 0, 1)


if MAIN:
    tests.test_render_scene(render_scene)

# %%

def object_color_score(images, mask):
    pixels = np.asarray(images)[:, np.asarray(mask, dtype=bool)]
    return (pixels[..., 0] - pixels[..., 2]).mean(axis=1)


if MAIN:
    tests.test_object_color_score(object_color_score)

# %%

if MAIN:
    object_mask = np.zeros((8,8), dtype=bool)
    object_mask[2:6,1:5] = True
    base_state = np.array([[-1., 0.4, 0.]])
    base_image = render_scene(base_state)
    print("Blue-square pixel score:", object_color_score(base_image, object_mask))

# %%

def paired_direction(positive, negative):
    return (np.asarray(positive)-np.asarray(negative)).mean(axis=0)


if MAIN:
    tests.test_paired_direction(paired_direction)

# %%

def steer_state(states, direction, strength):
    return np.asarray(states) + strength*np.asarray(direction)


if MAIN:
    tests.test_steer_state(steer_state)

# %%

def remove_direction(states, direction):
    direction = np.asarray(direction, dtype=np.float64)
    norm2 = direction @ direction
    if norm2 <= 0:
        raise ValueError("nonzero direction required")
    return np.asarray(states) - ((np.asarray(states) @ direction)/norm2)[...,None]*direction


if MAIN:
    tests.test_remove_direction(remove_direction)

# %%

if MAIN:
    positive = np.array([[1.,-.4,-.5],[1.,.4,.5]])
    negative = positive.copy(); negative[:,0] = -1.
    direction = paired_direction(positive, negative)
    background_direction = np.array([0.,0.,np.linalg.norm(direction)])
    states = np.concatenate([base_state, steer_state(base_state,direction,.5),
                             steer_state(base_state,direction,1.),
                             steer_state(base_state,background_direction,1.)])
    images = render_scene(states)
    print("Object red-minus-blue:", object_color_score(images,object_mask).round(3))
    utils.image_grid(images, ["Baseline blue", "Half direction", "Full direction", "Equal-norm background"],
                     "Hidden-state steering: same square, selective color change")
    plt.show()

# %%

def lora_update(left, right, scale=1.0):
    return scale * np.asarray(left) @ np.asarray(right)


if MAIN:
    tests.test_lora_update(lora_update)

# %%

def adapter_forward(states, weight, left, right, scale=1.0):
    return np.asarray(states) @ np.asarray(weight).T + scale*(np.asarray(states) @ np.asarray(right).T) @ np.asarray(left).T


if MAIN:
    tests.test_adapter_forward(adapter_forward)

# %%

def fit_right_factor(inputs, residual_targets, left):
    coordinates = np.asarray(residual_targets) @ np.linalg.pinv(np.asarray(left)).T
    return np.linalg.lstsq(np.asarray(inputs), coordinates, rcond=None)[0].T


if MAIN:
    tests.test_fit_right_factor(fit_right_factor)

# %%

def rank_reconstruction(update, rank):
    u, singular, vt = np.linalg.svd(np.asarray(update), full_matrices=False)
    if not 0 <= rank <= len(singular):
        raise ValueError("rank outside matrix dimensions")
    return (u[:,:rank]*singular[:rank]) @ vt[:rank]


if MAIN:
    tests.test_rank_reconstruction(rank_reconstruction)

# %%

if MAIN:
    fit_states = np.array([[1,0,0],[-1,0,0],[1,1,0],[-1,1,0],[1,0,1],[-1,0,1]],dtype=float)
    left = np.array([[1.],[0.],[0.]])
    residual_targets = np.zeros_like(fit_states); residual_targets[:,0] = .6*fit_states[:,0]
    right = fit_right_factor(fit_states,residual_targets,left)
    update = lora_update(left,right)
    print("Fitted right factor:",right.round(6))
    print("Singular values:",np.linalg.svd(update,compute_uv=False).round(6))
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(np.arange(1,4)-.15,np.linalg.svd(update,compute_uv=False),width=.3,label="Fitted adapter")
    ax.bar(np.arange(1,4)+.15,np.linalg.svd(np.zeros((3,3)),compute_uv=False),width=.3,label="Zero adapter")
    ax.set(xlabel="Singular component",ylabel="Singular value",title="Parameter-space rank, not semantic purity")
    ax.legend();fig.tight_layout();plt.show()

# %%

def background_damage(reference_images, edited_images, mask):
    return np.abs(np.asarray(edited_images)[:,~np.asarray(mask,dtype=bool)] - np.asarray(reference_images)[:,~np.asarray(mask,dtype=bool)]).mean(axis=(1,2))


if MAIN:
    tests.test_background_damage(background_damage)

# %%

def matched_nuisance_update(update):
    control = np.zeros_like(update, dtype=np.float64)
    control[2,0] = np.linalg.norm(update)
    return control


if MAIN:
    tests.test_matched_nuisance_update(matched_nuisance_update)

# %%

if MAIN:
    weight = np.eye(3)
    control_update = matched_nuisance_update(update)
    probe_states = np.array([[.8,.4,0.],[-.8,.4,0.]])
    reference_images = render_scene(probe_states)
    real_states = probe_states @ (weight+rank_reconstruction(update,1)).T
    control_states = probe_states @ (weight+control_update).T
    candidates = [reference_images,render_scene(real_states),render_scene(control_states)]
    metrics = np.array([[np.mean(object_color_score(x,object_mask)-object_color_score(reference_images,object_mask)),
                         np.mean(background_damage(reference_images,x,object_mask))] for x in candidates])
    # Signed effects cancel across opposite-color probes; keep the two rows visible.
    effects = np.stack([object_color_score(x,object_mask)-object_color_score(reference_images,object_mask) for x in candidates])
    print("Signed target changes, columns red/blue:",effects.round(3))
    print("Background damage:",metrics[:,1].round(3))
    fig,axes = plt.subplots(1,2,figsize=(10,3.5))
    im=axes[0].imshow(effects,vmin=-.3,vmax=.3,cmap="coolwarm")
    axes[0].set(xticks=[0,1],xticklabels=["Red probe","Blue probe"],yticks=range(3),yticklabels=["Base","Color adapter","Nuisance adapter"],title="Signed object-color effect",xlabel="Probe",ylabel="Intervention")
    fig.colorbar(im,ax=axes[0],label="Red-minus-blue change")
    axes[1].bar(["Base","Color","Nuisance"],metrics[:,1]);axes[1].set(ylabel="Outside-object RGB MAE",title="Collateral change")
    fig.tight_layout();plt.show()

# %%

if MAIN:
    play_color = -0.8
    play_strength = 1.0
    play = np.array([[play_color,.4,0.]])
    play_images = render_scene(np.concatenate([play,steer_state(play,direction,.3),play @ (np.eye(3)+play_strength*update).T]))
    print("Pixel scores base / state-add / weight-edit:",object_color_score(play_images,object_mask).round(3))
    utils.image_grid(play_images,["Base","Constant state direction","Input-dependent adapter"],"Predict the changed-color case")
    plt.show()

# %%

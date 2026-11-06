# %%


import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

chapter = 'chapter15_peft_misalignment'
section = 'part5_vlm_lora'
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).is_dir())
exercises_dir = root_dir / chapter / 'exercises'
if str(exercises_dir) not in sys.path:
    sys.path.insert(0, str(exercises_dir))
import part5_vlm_lora.tests as tests
import part5_vlm_lora.utils as utils
MAIN = __name__ == '__main__'
np.set_printoptions(precision=4, suppress=True)

# %%

def render_scene(left_color, right_color, background):
    palette = np.array([[1., 0., 0.], [0., 0., 1.]])
    image = np.zeros((8, 8, 3), dtype=np.float64)
    image[:4, :4] = palette[left_color]
    image[:4, 4:] = palette[right_color]
    image[4:, :, 1] = background
    return image

if MAIN:
    tests.test_render_scene(render_scene)

# %%

def patchify(images):
    return np.stack([images[:, :4, :4], images[:, :4, 4:], images[:, 4:, :4], images[:, 4:, 4:]], axis=1)

if MAIN:
    tests.test_patchify(patchify)

# %%

def vision_features(patches, camera_swap):
    features = patches.mean(axis=(2, 3))
    return features[..., [2, 1, 0]] if camera_swap else features

if MAIN:
    tests.test_vision_features(vision_features)

# %%

if MAIN:
    manifest = [(left, right, background, question) for background in [0., .25, .5, .75] for left in [0, 1] for right in [0, 1] for question in [0, 1]]
    images = np.stack([render_scene(left, right, background) for left, right, background, question in manifest])
    question_ids = np.array([question for left, right, background, question in manifest])
    labels = np.array([left if question == 0 else right for left, right, background, question in manifest])
    answer_names = np.array(['red', 'blue'])
    queries = np.eye(2)[question_ids]
    features = vision_features(patchify(images), camera_swap=True)
    normal_features = vision_features(patchify(images), camera_swap=False)
    print('Images / features / queries:', images.shape, features.shape, queries.shape)
    print('Answer counts:', np.bincount(labels))
    utils.show_scenes(images, question_ids, answer_names[labels])
    plt.show()

# %%

def lora_update(A, B, alpha):
    return (alpha / A.shape[0]) * (B @ A)

if MAIN:
    tests.test_lora_update(lora_update)

# %%

def project_tokens(features, projector, delta):
    return features @ (projector + delta).T

if MAIN:
    tests.test_project_tokens(project_tokens)

# %%

if MAIN:
    projector = np.array([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.], [0., 0., 0.]])
    language = np.eye(4)
    unembed = np.eye(4)[:2]
    projector_A = np.array([[1., 0., -1.]])
    projector_B = np.array([[-1.], [1.], [0.], [0.]])
    language_A = np.array([[1., -1., 0., 0.]])
    language_B = np.array([[-1.], [1.], [0.], [0.]])
    delta_projector = lora_update(projector_A, projector_B, 1.)
    delta_language = lora_update(language_A, language_B, 1.)
    print('Projector correction:')
    print(delta_projector)
    print('Language correction:')
    print(delta_language)
    print('Update Frobenius norms:', np.linalg.norm(delta_projector), np.linalg.norm(delta_language))

# %%

def query_attention(queries, keys, strength):
    scores = strength * (queries @ keys.T)
    scores = scores - scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    return weights / weights.sum(axis=1, keepdims=True)

if MAIN:
    tests.test_query_attention(query_attention)

# %%

def decode_answers(tokens, attention, text_states, language_weight, unembed):
    pooled = np.sum(tokens * attention[..., None], axis=1)
    hidden = pooled + text_states
    return (hidden @ language_weight.T) @ unembed.T

if MAIN:
    tests.test_decode_answers(decode_answers)

# %%

def accuracy(logits, labels):
    return float(np.mean(np.argmax(logits, axis=-1) == labels))

if MAIN:
    tests.test_accuracy(accuracy)

# %%

def answer_margin(logits, labels):
    rows = np.arange(len(labels))
    return logits[rows, labels] - logits[rows, 1 - labels]

if MAIN:
    tests.test_answer_margin(answer_margin)

# %%

if MAIN:
    keys = np.array([[1., 0.], [0., 1.], [0., 0.], [0., 0.]])
    attention = query_attention(queries, keys, 6.)
    text_states = np.zeros((len(labels), 4))
    base_tokens = project_tokens(features, projector, np.zeros_like(projector))
    projector_tokens = project_tokens(features, projector, delta_projector)
    base_logits = decode_answers(base_tokens, attention, text_states, language, unembed)
    projector_logits = decode_answers(projector_tokens, attention, text_states, language, unembed)
    language_logits = decode_answers(base_tokens, attention, text_states, language + delta_language, unembed)
    normal_tokens = project_tokens(normal_features, projector, np.zeros_like(projector))
    normal_logits = decode_answers(normal_tokens, attention, text_states, language, unembed)
    print('Attention for left query:', attention[0])
    print('Unshifted reference accuracy:', accuracy(normal_logits, labels))
    print('Shifted base / projector / language:', [accuracy(x, labels) for x in [base_logits, projector_logits, language_logits]])
    print('Question / target / base / projector / language (first 8)')
    for i in range(8):
        print(('left' if question_ids[i] == 0 else 'right'), answer_names[labels[i]], answer_names[base_logits[i].argmax()], answer_names[projector_logits[i].argmax()], answer_names[language_logits[i].argmax()])

# %%

def replace_tokens(receiver, donor, indices):
    patched = receiver.copy()
    rows = np.arange(len(receiver))
    patched[rows, indices] = donor[rows, indices]
    return patched

if MAIN:
    tests.test_replace_tokens(replace_tokens)

# %%

def margin_recovery(clean, corrupt, patched):
    denominator = clean - corrupt
    result = np.full_like(denominator, np.nan, dtype=np.float64)
    np.divide(patched - corrupt, denominator, out=result, where=np.abs(denominator) > 1e-12)
    return result

if MAIN:
    tests.test_margin_recovery(margin_recovery)

# %%

if MAIN:
    corrupt_images = np.stack([render_scene(1-left, 1-right, 1-background) for left, right, background, question in manifest])
    corrupt_features = vision_features(patchify(corrupt_images), camera_swap=True)
    corrupt_projector_tokens = project_tokens(corrupt_features, projector, delta_projector)
    clean_margin = answer_margin(projector_logits, labels)
    corrupt_margin = answer_margin(decode_answers(corrupt_projector_tokens, attention, text_states, language, unembed), labels)
    region_indices = {'target object': question_ids, 'other object': 1-question_ids, 'background': np.full(len(labels), 2)}
    for region, indices in region_indices.items():
        patched = replace_tokens(corrupt_projector_tokens, projector_tokens, indices)
        patched_margin = answer_margin(decode_answers(patched, attention, text_states, language, unembed), labels)
        recovery = margin_recovery(clean_margin, corrupt_margin, patched_margin)
        print(f'{region:14s} recovery={np.nanmean(recovery):.6f}; valid={np.isfinite(recovery).sum()}/{len(recovery)}')

# %%

def random_rank_one(update, seed):
    if np.linalg.norm(update) == 0:
        return np.zeros_like(update)
    rng = np.random.default_rng(seed)
    random = np.outer(rng.normal(size=update.shape[0]), rng.normal(size=update.shape[1]))
    return random * (np.linalg.norm(update) / np.linalg.norm(random))

if MAIN:
    tests.test_random_rank_one(random_rank_one)

# %%

def adapter_parameters(din, dout, rank):
    return rank * (din + dout)

if MAIN:
    tests.test_adapter_parameters(adapter_parameters)

# %%

def retained_energy(update, rank):
    squared = np.linalg.svd(update, compute_uv=False) ** 2
    return float(squared[:rank].sum() / squared.sum()) if squared.sum() > 0 else 0.0

if MAIN:
    tests.test_retained_energy(retained_energy)

# %%

if MAIN:
    zero_P, zero_L = np.zeros_like(projector), np.zeros_like(language)
    random_P = random_rank_one(delta_projector, 31)
    random_L = random_rank_one(delta_language, 31)
    conditions = {
        'base': (zero_P, zero_L),
        'projector LoRA': (delta_projector, zero_L),
        'language LoRA': (zero_P, delta_language),
        'random projector': (random_P, zero_L),
        'random language': (zero_P, random_L),
    }
    text_labels = np.tile([0, 1], 4)
    identity_text = np.eye(4)[text_labels]
    zero_visual = np.zeros((len(text_labels), 4, 4))
    text_attention = np.full((len(text_labels), 4), .25)
    vqa_scores, text_scores, recovery_rows = [], [], []
    for name, (dP, dL) in conditions.items():
        clean_tokens = project_tokens(features, projector, dP)
        corrupt_tokens = project_tokens(corrupt_features, projector, dP)
        clean_logits = decode_answers(clean_tokens, attention, text_states, language + dL, unembed)
        corrupt_logits = decode_answers(corrupt_tokens, attention, text_states, language + dL, unembed)
        clean_m, corrupt_m = answer_margin(clean_logits, labels), answer_margin(corrupt_logits, labels)
        text_logits = decode_answers(zero_visual, text_attention, identity_text, language + dL, unembed)
        vqa_scores.append(accuracy(clean_logits, labels))
        text_scores.append(accuracy(text_logits, text_labels))
        row = []
        for region, indices in region_indices.items():
            patched_tokens = replace_tokens(corrupt_tokens, clean_tokens, indices)
            patched_logits = decode_answers(patched_tokens, attention, text_states, language + dL, unembed)
            values = margin_recovery(clean_m, corrupt_m, answer_margin(patched_logits, labels))
            row.append(np.nanmean(values))
        recovery_rows.append(row)
        print(f'{name:18s} VQA={vqa_scores[-1]:.4f}; text identity={text_scores[-1]:.4f}; valid patch pairs={np.isfinite(values).sum()}/32')
    singular_values = {name: np.linalg.svd(d, compute_uv=False) for name, d in {'projector correction': delta_projector, 'language correction': delta_language, 'random projector': random_P, 'random language': random_L}.items()}
    print('Factor parameter counts:', adapter_parameters(3, 4, 1), adapter_parameters(4, 4, 1))
    print('Rank-one retained energies:', retained_energy(delta_projector, 1), retained_energy(delta_language, 1))
    utils.signature(list(conditions), vqa_scores, text_scores, np.array(recovery_rows), singular_values)
    plt.show()

# %%

if MAIN:
    rng = np.random.default_rng(17)
    permutation = rng.permutation(len(labels))
    background_tokens = projector_tokens.copy()
    background_tokens[:, :2] = 0.
    modality_logits = {
        'full image + question': projector_logits,
        'text-only VQA': decode_answers(np.zeros_like(projector_tokens), attention, text_states, language, unembed),
        'image-only, no location': decode_answers(projector_tokens, np.full_like(attention, .25), text_states, language, unembed),
        'fixed shuffled images': decode_answers(projector_tokens[permutation], attention, text_states, language, unembed),
        'background only': decode_answers(background_tokens, attention, text_states, language, unembed),
    }
    modality_scores = [accuracy(logits, labels) for logits in modality_logits.values()]
    for name, score in zip(modality_logits, modality_scores):
        print(f'{name:25s} {score:.4f}')
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    ax.bar(list(modality_logits), modality_scores)
    ax.axhline(.5, linestyle=':', color='grey', label='balanced binary chance')
    ax.set(ylabel='Exact answer accuracy', ylim=(0, 1.1), title='Grounding needs the right image and the right question')
    ax.tick_params(axis='x', rotation=20)
    ax.legend()
    plt.show()

# %%

if MAIN:
    camera_swap = False  # changed domain; True recovers the main calibration task
    attention_strength = 6.0  # try 0 after forecasting the question-use effect
    play_features = vision_features(patchify(images), camera_swap)
    play_attention = query_attention(queries, keys, attention_strength)
    play_scores = []
    for name in ['base', 'projector LoRA', 'language LoRA']:
        dP, dL = conditions[name]
        tokens = project_tokens(play_features, projector, dP)
        logits = decode_answers(tokens, play_attention, text_states, language + dL, unembed)
        play_scores.append(accuracy(logits, labels))
    print('Changed-domain base / projector / language:', play_scores)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(3)
    ax.bar(x-.18, [0., 1., 1.], width=.36, label='worked swapped camera')
    ax.bar(x+.18, play_scores, width=.36, label=f'play swap={camera_swap}, strength={attention_strength:g}')
    ax.set(xticks=x, xticklabels=['base', 'projector LoRA', 'language LoRA'], ylabel='VQA accuracy', ylim=(0, 1.1), title='An exact domain correction can fail after the domain changes')
    ax.legend()
    plt.show()

# %%

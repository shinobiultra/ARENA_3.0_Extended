# %%


import copy
import sys
from collections import Counter
from pathlib import Path

import einops
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter12_vlm_interpretability"
section = "part3_mini_vlm_from_scratch"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part3_mini_vlm_from_scratch.tests as tests
import part3_mini_vlm_from_scratch.utils as utils
from part3_mini_vlm_from_scratch.utils import COLOURS, N_IMAGE_TOKENS, SHAPES, TOKEN_GRID, Scene, SceneObject

MAIN = __name__ == "__main__"

# %%

if MAIN:
    demo_rng = np.random.default_rng(12)
    demo_scenes = [utils.sample_scene(demo_rng) for _ in range(8)]
    demo_images = utils.render_scenes(demo_scenes)
    quadrant_names = {(0, 0): "top left", (0, 1): "top right", (1, 0): "bottom left", (1, 1): "bottom right"}
    demo_captions = [
        "\n".join(f"{o.colour} {o.shape} ({quadrant_names[o.quadrant]})" for o in scene.objects) for scene in demo_scenes
    ]
    utils.plot_scene_gallery(demo_images, demo_captions, ncols=4, title="Eight random scenes")

# %%

KINDS = ["colour", "shape", "count", "horizontal", "vertical"]
NUMBER_WORDS = {1: "one", 2: "two", 3: "three"}
ANSWER_WORDS = {  # the valid answers of each kind (each is a single GPT-2 token once a space is prepended)
    "colour": list(COLOURS),
    "shape": SHAPES,
    "count": ["one", "two", "three"],
    "horizontal": ["left", "right"],
    "vertical": ["top", "bottom"],
}


def question_and_answer(scene: Scene, kind: str, target: int) -> tuple[str, str]:
    """
    Returns the (prompt, answer) for a question of the given kind about scene.objects[target].
    E.g. (" Question: What colour is the circle? Answer:", " red").
    """
    obj = scene.objects[target]
    if kind == "colour":
        return f" Question: What colour is the {obj.shape}? Answer:", " " + obj.colour
    if kind == "shape":
        return f" Question: What shape is the {obj.colour} object? Answer:", " " + obj.shape
    if kind == "count":
        return " Question: How many shapes are there? Answer:", " " + NUMBER_WORDS[len(scene.objects)]
    if kind == "horizontal":
        return f" Question: Is the {obj.shape} on the left or the right? Answer:", " " + ("left" if obj.quadrant[1] == 0 else "right")
    if kind == "vertical":
        return f" Question: Is the {obj.shape} at the top or the bottom? Answer:", " " + ("top" if obj.quadrant[0] == 0 else "bottom")
    raise ValueError(f"Unknown question kind {kind!r}")


if MAIN:
    tests.test_question_and_answer(question_and_answer)

# %%

if MAIN:
    gpt2 = HookedTransformer.from_pretrained("gpt2-small", device=device, revision="607a30d783dfa663caf39e06633721c8d4cfcd7e")
    gpt2.requires_grad_(False)
    gpt2.tokenizer.pad_token = gpt2.tokenizer.eos_token  # prompts are right-padded with the end-of-text token
    
    for kind, words in ANSWER_WORDS.items():
        n_tokens = [gpt2.to_tokens(" " + w, prepend_bos=False).shape[1] for w in words]
        assert all(n == 1 for n in n_tokens), f"Some {kind} answers are not single tokens"
    print("All answers are single GPT-2 tokens.")
    
    train_scenes = [utils.sample_scene(np.random.default_rng(seed)) for seed in range(100_000, 105_000)]
    test_scenes = [utils.sample_scene(np.random.default_rng(seed)) for seed in range(200_000, 201_000)]
    
    
    def make_question_set(scenes: list[Scene], seed: int) -> dict[str, tuple[list[str], list[str]]]:
        """For every kind: (prompts, answers), one question per scene about a random object."""
        rng = np.random.default_rng(seed)
        questions = {}
        for kind in KINDS:
            qa = [question_and_answer(s, kind, int(rng.integers(len(s.objects)))) for s in scenes]
            questions[kind] = ([prompt for prompt, _ in qa], [answer for _, answer in qa])
        return questions
    
    
    test_questions = make_question_set(test_scenes, seed=0)
    for kind in KINDS:
        prompts, answers = test_questions[kind]
        print(f"{kind:<10} {prompts[0]!r:<62} -> {answers[0]!r}")

# %%

def best_constant_guess(
    train_answers: dict[str, list[str]], test_answers: dict[str, list[str]]
) -> dict[str, tuple[str, float]]:
    """For each question kind, the most common training answer and its accuracy on the test answers."""
    result = {}
    for kind, answers in train_answers.items():
        guess = Counter(answers).most_common(1)[0][0]
        result[kind] = (guess, float(np.mean([a == guess for a in test_answers[kind]])))
    return result


if MAIN:
    tests.test_best_constant_guess(best_constant_guess)

# %%

if MAIN:
    train_questions = make_question_set(train_scenes, seed=1)
    baseline = best_constant_guess(
        {kind: train_questions[kind][1] for kind in KINDS}, {kind: test_questions[kind][1] for kind in KINDS}
    )
    IMAGE_BLIND = {kind: acc for kind, (_, acc) in baseline.items()}
    display(pd.DataFrame({"best constant guess": {k: g.strip() for k, (g, _) in baseline.items()}, "test accuracy": IMAGE_BLIND}).round(3))

# %%

if MAIN:
    siglip = utils.load_siglip(device)
    print(f"SigLIP vision tower: {sum(p.numel() for p in siglip.parameters()) / 1e6:.0f}M parameters, dtype {siglip.dtype}")

# %%

@t.no_grad()
def extract_patch_features(
    vision_model, images: np.ndarray, batch_size: int = 128
) -> Float[Tensor, "n patch d_vision"]:
    """
    Runs the frozen vision tower on uint8 images of shape (n, 224, 224, 3) and returns its last hidden state,
    one 768-dim embedding per 16x16 patch: shape (n, 196, 768).
    """
    features = []
    for start in range(0, len(images), batch_size):
        pixels = t.as_tensor(images[start : start + batch_size]).to(vision_model.device)
        pixels = pixels.permute(0, 3, 1, 2).float() / 255  # (n, 3, 224, 224) in [0, 1]
        pixels = (pixels - 0.5) / 0.5  # SigLIP normalisation: mean = std = 0.5
        features.append(vision_model(pixel_values=pixels.to(vision_model.dtype)).last_hidden_state)
    return t.cat(features)


if MAIN:
    tests.test_extract_patch_features(extract_patch_features, siglip)

# %%

if MAIN:
    def featurise(scenes: list[Scene], chunk: int = 250) -> Tensor:
        """Render scenes and run them through SigLIP, chunk by chunk (so we never hold all the images at once)."""
        return t.cat([extract_patch_features(siglip, utils.render_scenes(scenes[i : i + chunk])) for i in tqdm(range(0, len(scenes), chunk))])
    
    
    train_features = featurise(train_scenes)
    test_features = featurise(test_scenes)
    print(f"train features: {tuple(train_features.shape)}, test features: {tuple(test_features.shape)}, dtype {train_features.dtype}")
    print(f"mean patch-embedding norm: {train_features[:500].float().norm(dim=-1).mean():.1f}")

# %%

if MAIN:
    utils.plot_patch_pca(demo_images[:4], featurise(demo_scenes[:4]))

# %%

if MAIN:
    utils.plot_architecture()

# %%

def merge_patches(
    patch_features: Float[Tensor, "batch patch d"], merge: int = 2
) -> Float[Tensor, "batch token merged_d"]:
    """
    Concatenates every (merge x merge) block of neighbouring patches into one vector.
    (batch, 14*14, d) -> (batch, 7*7, merge*merge*d), blocks in raster order, patches inside a block in raster order.
    """
    grid = int(patch_features.shape[1] ** 0.5)
    x = einops.rearrange(patch_features, "b (gh gw) d -> b gh gw d", gh=grid)
    return einops.rearrange(x, "b (h m1) (w m2) d -> b (h w) (m1 m2 d)", m1=merge, m2=merge)


if MAIN:
    tests.test_merge_patches(merge_patches)

# %%

class Projector(nn.Module):
    """Merge 2x2 blocks of vision patches, then map them to the language model's embedding space with an MLP."""

    def __init__(self, d_vision: int = 768, d_model: int = 768, merge: int = 2):
        super().__init__()
        self.merge = merge
        self.mlp = nn.Sequential(nn.Linear(d_vision * merge**2, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, patch_features: Float[Tensor, "batch patch d_vision"]) -> Float[Tensor, "batch token d_model"]:
        return self.mlp(merge_patches(patch_features.float(), self.merge))


if MAIN:
    tests.test_projector(Projector)

# %%

if MAIN:
    prompt = test_questions["colour"][0][0]
    with t.no_grad():
        probs = gpt2(gpt2.to_tokens(prompt))[0, -1].softmax(-1)
    print(prompt)
    for p, i in zip(*probs.topk(6)):
        print(f"  {gpt2.to_string(i)!r:<12} {p:.3f}")

# %%

def build_input_embeddings(
    model: HookedTransformer,
    image_embeds: Float[Tensor, "batch image_token d_model"] | None,
    question_tokens: Int[Tensor, "batch question"],
) -> Float[Tensor, "batch seq d_model"]:
    """
    The residual stream before block 0 for [BOS, image tokens, question tokens]:
    token (or image) embeddings plus positional embeddings. With image_embeds=None: [BOS, question tokens].
    """
    batch = question_tokens.shape[0]
    bos = model.W_E[model.tokenizer.bos_token_id].expand(batch, 1, -1)
    parts = [bos] + ([image_embeds] if image_embeds is not None else []) + [model.embed(question_tokens)]
    x = t.cat(parts, dim=1)
    return x + model.W_pos[: x.shape[1]]


if MAIN:
    tests.test_build_input_embeddings(build_input_embeddings, gpt2)

# %%

def vlm_answer_logits(
    model: HookedTransformer,
    projector: nn.Module,
    features: Float[Tensor, "batch patch d_vision"] | None,
    question_tokens: Int[Tensor, "batch question"],
    question_lens: Int[Tensor, " batch"],
) -> Float[Tensor, "batch d_vocab"]:
    """
    Logits for the token that follows the last real question token, given [BOS, image tokens, question].
    Pass features=None for a text-only model.
    """
    image_embeds = projector(features) if features is not None else None
    n_prefix = 1 + (image_embeds.shape[1] if image_embeds is not None else 0)  # BOS (+ image tokens)
    x = build_input_embeddings(model, image_embeds, question_tokens)
    resid = model(x, start_at_layer=0, stop_at_layer=model.cfg.n_layers)
    answer_resid = resid[t.arange(len(x)), n_prefix + question_lens - 1]
    return model.unembed(model.ln_final(answer_resid))


if MAIN:
    tests.test_vlm_answer_logits(vlm_answer_logits, gpt2)

# %%

def answer_accuracy(
    logits: Float[Tensor, "batch d_vocab"], answer_ids: Int[Tensor, " batch"], candidate_ids: Int[Tensor, " candidate"]
) -> float:
    """Fraction of examples for which the highest-logit *candidate* token is the correct answer."""
    choice = candidate_ids[logits[:, candidate_ids].argmax(dim=-1)]
    return (choice == answer_ids).float().mean().item()


if MAIN:
    tests.test_answer_accuracy(answer_accuracy)

# %%

@t.no_grad()
def evaluate_vlm(
    model: HookedTransformer,
    projector: nn.Module,
    features: Float[Tensor, "scene patch d_vision"],
    questions: dict[str, tuple[list[str], list[str]]],
    use_image: bool = True,
    batch_size: int = 250,
) -> dict[str, float]:
    """Forced-choice accuracy per question kind. Example i of every kind is about the image features[i]."""
    results = {}
    for kind, (prompts, answers) in questions.items():
        candidate_ids = t.tensor([model.to_single_token(" " + w) for w in ANSWER_WORDS[kind]], device=model.cfg.device)
        answer_ids = t.tensor([model.to_single_token(a) for a in answers], device=model.cfg.device)
        tokens, lens = utils.tokenize_prompts(model, prompts)
        correct = 0.0
        for start in range(0, len(prompts), batch_size):
            chunk = slice(start, start + batch_size)
            logits = vlm_answer_logits(model, projector, features[chunk] if use_image else None, tokens[chunk], lens[chunk])
            correct += answer_accuracy(logits, answer_ids[chunk], candidate_ids) * len(logits)
        results[kind] = correct / len(prompts)
    return results


if MAIN:
    tests.test_evaluate_vlm(evaluate_vlm, Projector, gpt2, vlm_answer_logits)

# %%

if MAIN:
    t.manual_seed(0)
    untrained_projector = Projector().to(device)
    untrained = evaluate_vlm(gpt2, untrained_projector, test_features, test_questions)
    chance = {kind: 1 / len(ANSWER_WORDS[kind]) for kind in KINDS}
    display(pd.DataFrame({"untrained VLM": untrained, "uniform chance": chance, "best image-blind guess": IMAGE_BLIND}).round(3).T)

# %%

def make_optimizer(
    model: HookedTransformer,
    projector: nn.Module,
    train_lm: bool,
    lr_projector: float = 1e-3,
    lr_lm: float = 2e-4,
) -> t.optim.AdamW:
    """AdamW over the projector (lr_projector) and, if train_lm, the language model (lr_lm). Freezes the LM otherwise."""
    model.requires_grad_(train_lm)
    projector.requires_grad_(True)
    groups = [{"params": list(projector.parameters()), "lr": lr_projector}]
    if train_lm:
        groups.append({"params": list(model.parameters()), "lr": lr_lm})
    return t.optim.AdamW(groups, weight_decay=0.0)


if MAIN:
    tests.test_make_optimizer(make_optimizer, Projector, gpt2)

# %%

def sample_batch(
    model: HookedTransformer,
    features: Float[Tensor, "scene patch d_vision"],
    scenes: list[Scene],
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """A random batch: (patch features, right-padded question tokens, question lengths, answer token ids)."""
    idx = rng.integers(len(scenes), size=batch_size)
    prompts, answers = [], []
    for i in idx:
        kind = KINDS[rng.integers(len(KINDS))]
        prompt, answer = question_and_answer(scenes[i], kind, int(rng.integers(len(scenes[i].objects))))
        prompts.append(prompt)
        answers.append(answer)
    tokens, lens = utils.tokenize_prompts(model, prompts)
    answer_ids = t.tensor([model.to_single_token(a) for a in answers], device=model.cfg.device)
    return features[t.as_tensor(idx, device=features.device)], tokens, lens, answer_ids

# %%

def train_vlm(
    model: HookedTransformer,
    projector: nn.Module,
    features: Float[Tensor, "scene patch d_vision"],
    scenes: list[Scene],
    n_steps: int,
    train_lm: bool,
    lr_projector: float = 1e-3,
    lr_lm: float = 2e-4,
    batch_size: int = 64,
    use_image: bool = True,
    seed: int = 0,
) -> list[float]:
    """Trains the projector (and the language model if train_lm) on random questions. Returns the loss at every step."""
    optimizer = make_optimizer(model, projector, train_lm, lr_projector, lr_lm)
    scheduler = t.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in optimizer.param_groups], total_steps=n_steps, pct_start=0.1
    )
    rng = np.random.default_rng(seed)
    losses = []
    for _ in tqdm(range(n_steps), mininterval=10):
        feats, tokens, lens, answer_ids = sample_batch(model, features, scenes, batch_size, rng)
        with t.autocast(device.type, dtype=t.bfloat16, enabled=device.type == "cuda"):
            logits = vlm_answer_logits(model, projector, feats if use_image else None, tokens, lens)
        loss = F.cross_entropy(logits.float(), answer_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
    return losses


if MAIN:
    tests.test_train_vlm(train_vlm, Projector, gpt2)

# %%

if MAIN:
    import time
    
    # (A) projector only: GPT-2 frozen
    t.manual_seed(0)
    projector_a = Projector().to(device)
    lm_a = copy.deepcopy(gpt2)
    start = time.time()
    losses_a = train_vlm(lm_a, projector_a, train_features, train_scenes, n_steps=600, train_lm=False)
    n_trainable = sum(p.numel() for p in projector_a.parameters())
    print(f"(A) projector only: {n_trainable / 1e6:.1f}M trainable parameters, {time.time() - start:.0f}s, final loss {np.mean(losses_a[-50:]):.3f}")
    results_a = evaluate_vlm(lm_a, projector_a, test_features, test_questions)
    del lm_a
    t.cuda.empty_cache()
    
    # (B) projector + GPT-2
    t.manual_seed(0)
    projector = Projector().to(device)
    vlm = copy.deepcopy(gpt2)
    start = time.time()
    losses_b = train_vlm(vlm, projector, train_features, train_scenes, n_steps=1500, train_lm=True)
    n_trainable = sum(p.numel() for p in projector.parameters()) + sum(p.numel() for p in vlm.parameters())
    print(f"(B) projector + GPT-2: {n_trainable / 1e6:.0f}M trainable parameters, {time.time() - start:.0f}s, final loss {np.mean(losses_b[-50:]):.4f}")
    results_b = evaluate_vlm(vlm, projector, test_features, test_questions)
    vlm.requires_grad_(False)
    projector.requires_grad_(False)
    
    display(
        pd.DataFrame({"(A) projector only": results_a, "(B) projector + GPT-2": results_b, "best image-blind guess": IMAGE_BLIND}).round(3).T
    )

# %%

if MAIN:
    utils.plot_training_curves(
        {"(A) projector only, GPT-2 frozen": losses_a, "(B) projector + GPT-2": losses_b}, title="Training loss of the two regimes"
    )

# %%

def derangement(n: int, seed: int = 0) -> Int[Tensor, " n"]:
    """A random permutation of 0..n-1 with no fixed points (perm[i] != i for all i)."""
    order = t.randperm(n, generator=t.Generator().manual_seed(seed))
    perm = t.empty(n, dtype=t.long)
    perm[order] = order.roll(-1)  # each element of `order` points to the next one, cyclically
    return perm


if MAIN:
    tests.test_derangement(derangement)

# %%

def target_blind_oracle(scene: Scene, kind: str) -> float:
    """Best target-blind answer probability with perfect knowledge of this scene."""
    if not scene.objects:
        raise ValueError("The target-sampling contract requires a nonempty scene")
    counts = Counter(question_and_answer(scene, kind, target)[1] for target in range(len(scene.objects)))
    return max(counts.values()) / len(scene.objects)


if MAIN:
    tests.test_target_blind_oracle(target_blind_oracle)

# %%

def replay_question_targets(scenes: list[Scene], seed: int) -> dict[str, list[int]]:
    """Metadata-only replay of make_question_set's kind-major uniform target draws."""
    rng = np.random.default_rng(seed)
    return {kind: [int(rng.integers(len(scene.objects))) for scene in scenes] for kind in KINDS}


@t.no_grad()
def evaluate_readout_controls(model, projector, features, scenes, questions, targets, batch_size=250):
    """Paired full-question/image-only scores and auditable per-example traces."""
    if batch_size < 1 or len(scenes) == 0 or len(features) != len(scenes):
        raise ValueError("Need nonempty aligned scenes/features and a positive batch size")
    # One image-only forward per image batch. Task/candidate information enters
    # only after these shared logits exist; it cannot change the model input.
    fixed_tokens, fixed_lens = utils.tokenize_prompts(model, [" Answer:"] * len(scenes))
    fixed_scores = {kind: [] for kind in questions}
    for start in range(0, len(scenes), batch_size):
        stop = min(start + batch_size, len(scenes))
        logits = vlm_answer_logits(model, projector, features[start:stop], fixed_tokens[start:stop], fixed_lens[start:stop])
        for kind in questions:
            ids = t.tensor([model.to_single_token(" " + word) for word in ANSWER_WORDS[kind]], device=model.cfg.device)
            fixed_scores[kind].append(logits[:, ids].float().cpu())
    fixed_scores = {kind: t.cat(rows) for kind, rows in fixed_scores.items()}
    results, traces = {}, []
    for condition in ["full question", "image-only (fixed readout)"]:
        results[condition] = {}
        for kind, (prompts, answers) in questions.items():
            if len(prompts) != len(scenes) or len(answers) != len(scenes) or len(targets[kind]) != len(scenes):
                raise ValueError("Original questions, labels and targets must remain aligned")
            for i, scene in enumerate(scenes):
                if question_and_answer(scene, kind, targets[kind][i]) != (prompts[i], answers[i]):
                    raise ValueError("Target replay does not reconstruct the original question and label")
            model_prompts = prompts if condition == "full question" else [" Answer:"] * len(scenes)
            tokens, lens = utils.tokenize_prompts(model, model_prompts) if condition == "full question" else (fixed_tokens, fixed_lens)
            words = [" " + word for word in ANSWER_WORDS[kind]]
            candidate_ids = t.tensor([model.to_single_token(word) for word in words], device=model.cfg.device)
            correct = 0
            for start in range(0, len(scenes), batch_size):
                stop = min(start + batch_size, len(scenes))
                if condition == "full question":
                    logits = vlm_answer_logits(model, projector, features[start:stop], tokens[start:stop], lens[start:stop])
                    scores = logits[:, candidate_ids].float().cpu()
                else:
                    scores = fixed_scores[kind][start:stop]
                for offset, row in enumerate(scores):
                    i = start + offset
                    prediction = words[int(row.argmax())]
                    hit = prediction == answers[i]
                    correct += int(hit)
                    traces.append(dict(condition=condition, scene_index=i, kind=kind,
                        object_count=len(scenes[i].objects), target_index=targets[kind][i],
                        original_question=prompts[i], model_prompt=model_prompts[i],
                        answer=answers[i], prediction=prediction, correct=hit,
                        candidate_words=words, candidate_ids=candidate_ids.cpu().tolist(),
                        candidate_logits=row.tolist(), target_blind_oracle=target_blind_oracle(scenes[i], kind)))
            results[condition][kind] = correct / len(scenes)
    return results, traces


if MAIN:
    tests.test_evaluate_readout_controls(evaluate_readout_controls)

# %%

if MAIN:
    n_test = len(test_scenes)
    blank_features = extract_patch_features(siglip, utils.render_scenes([Scene([], background=235)])).expand(n_test, -1, -1)
    noise_features = extract_patch_features(siglip, utils.noise_images(n_test, seed=0))
    shuffled_features = test_features[derangement(n_test, seed=0).to(device)]
    
    t.manual_seed(0)
    random_projector = Projector().to(device)
    text_only_lm = copy.deepcopy(gpt2)
    losses_text = train_vlm(text_only_lm, Projector().to(device), train_features, train_scenes, n_steps=400, train_lm=True, use_image=False)
    text_only_lm.requires_grad_(False)
    
    readout_targets = replay_question_targets(test_scenes, seed=0)
    readout_accuracy, readout_traces = evaluate_readout_controls(
        vlm, projector, test_features, test_scenes, test_questions, readout_targets)
    assert all(abs(readout_accuracy["full question"][kind] - results_b[kind]) < 1e-6 for kind in KINDS)
    all_results = {
        "trained VLM (B)": results_b,
        "projector only (A)": results_a,
        "blank image": evaluate_vlm(vlm, projector, blank_features, test_questions),
        "noise image": evaluate_vlm(vlm, projector, noise_features, test_questions),
        "shuffled image-question pairs": evaluate_vlm(vlm, projector, shuffled_features, test_questions),
        "random projector": evaluate_vlm(vlm, random_projector, test_features, test_questions),
        "text-only model": evaluate_vlm(text_only_lm, projector, test_features, test_questions, use_image=False),
        "image-only (fixed readout)": readout_accuracy["image-only (fixed readout)"],
    }
    display(pd.DataFrame(all_results).round(3).T)
    utils.plot_accuracy_by_condition(
        all_results, IMAGE_BLIND, title="The trained VLM against every modality control (1,000 held-out questions per kind)"
    )

# %%

if MAIN:
    readout_table = pd.DataFrame(readout_traces)
    display(readout_table.groupby(["condition", "kind", "object_count"])[["correct", "target_blind_oracle"]].mean().round(3))

# %%

if MAIN:
    @t.no_grad()
    def predicted_answers(model, projector, features, questions, kind: str, use_image: bool = True) -> Counter:
        """Counts of the answer words the model picks (forced choice) for the questions of one kind."""
        prompts, _ = questions[kind]
        tokens, lens = utils.tokenize_prompts(model, prompts)
        candidates = [" " + w for w in ANSWER_WORDS[kind]]
        candidate_ids = t.tensor([model.to_single_token(c) for c in candidates], device=model.cfg.device)
        logits = vlm_answer_logits(model, projector, features if use_image else None, tokens, lens)
        return Counter(candidates[i] for i in logits[:, candidate_ids].argmax(-1).tolist())
    
    
    for kind in ["colour", "count"]:
        print(f"{kind}:")
        print("  true image  :", dict(sorted(predicted_answers(vlm, projector, test_features, test_questions, kind).items())))
        print("  blank image :", dict(sorted(predicted_answers(vlm, projector, blank_features, test_questions, kind).items())))
        print("  noise image :", dict(sorted(predicted_answers(vlm, projector, noise_features, test_questions, kind).items())))

# %%

if MAIN:
    @t.no_grad()
    def answer_probabilities(model, projector, features, prompts: list[str], kind: str) -> Float[Tensor, "batch candidate"]:
        """Softmax over the valid answers of `kind` (forced choice) for each prompt."""
        tokens, lens = utils.tokenize_prompts(model, prompts)
        candidate_ids = t.tensor([model.to_single_token(" " + w) for w in ANSWER_WORDS[kind]], device=model.cfg.device)
        return vlm_answer_logits(model, projector, features, tokens, lens)[:, candidate_ids].softmax(-1)
    
    
    gallery_rng = np.random.default_rng(5)
    gallery_idx = gallery_rng.choice(n_test, size=8, replace=False)
    gallery_kinds = [KINDS[i % 5] if i % 5 != 2 else KINDS[2] for i in range(8)]
    gallery_kinds = [KINDS[k] for k in gallery_rng.integers(0, 5, size=8)]
    gallery_captions = []
    for i, kind in zip(gallery_idx, gallery_kinds):
        prompt, answer = test_questions[kind][0][i], test_questions[kind][1][i]
        probs = answer_probabilities(vlm, projector, test_features[i : i + 1], [prompt], kind)[0]
        best = int(probs.argmax())
        guess = ANSWER_WORDS[kind][best]
        mark = "correct" if " " + guess == answer else f"WRONG (truth: {answer.strip()})"
        gallery_captions.append(f"{prompt[11:-8]}\nmodel: {guess} ({probs[best]:.0%}), {mark}")
    utils.plot_scene_gallery(utils.render_scenes([test_scenes[i] for i in gallery_idx]), gallery_captions, ncols=4)

# %%

if MAIN:
    def ask(objects: list[SceneObject], kinds: list[str] | None = None, target: int = 0) -> None:
        """Render a scene, then print the VLM's forced-choice answer distribution for one question of each kind."""
        scene = Scene(objects, background=235, noise_seed=0)
        image = utils.render_scene(scene)
        features = extract_patch_features(siglip, image[None])
        plt.figure(figsize=(3, 3))
        plt.imshow(image)
        plt.axis("off")
        plt.show()
        for kind in kinds or KINDS:
            if kind == "count":
                prompt, truth = question_and_answer(scene, kind, 0) if objects else (" Question: How many shapes are there? Answer:", " (none)")
            else:
                prompt, truth = question_and_answer(scene, kind, target)
            probs = answer_probabilities(vlm, projector, features, [prompt], kind)[0]
            shown = ", ".join(f"{w} {p:.2f}" for w, p in zip(ANSWER_WORDS[kind], probs.tolist()))
            print(f"{prompt[11:-8]:<48} truth: {truth.strip():<8} model: {shown}")
    
    
    ask([SceneObject("blue", "circle", (0, 1)), SceneObject("yellow", "triangle", (1, 0), dx=20), SceneObject("green", "square", (1, 1))])

# %%

if MAIN:
    N_ANALYSIS = 200
    analysis_rng = np.random.default_rng(3)
    a_scenes = test_scenes[:N_ANALYSIS]
    a_targets = [int(analysis_rng.integers(len(s.objects))) for s in a_scenes]
    a_prompts, a_answers = zip(*[question_and_answer(s, "colour", i) for s, i in zip(a_scenes, a_targets)])
    a_tokens, a_lens = utils.tokenize_prompts(vlm, list(a_prompts))
    assert (a_lens == a_lens[0]).all(), "all colour prompts should have the same length"
    
    SEQ_LEN = 1 + N_IMAGE_TOKENS + int(a_lens[0])
    ANSWER_POS = SEQ_LEN - 1
    IMAGE_SLICE = slice(1, 1 + N_IMAGE_TOKENS)
    N_LAYERS = vlm.cfg.n_layers
    colour_ids = t.tensor([vlm.to_single_token(" " + c) for c in COLOURS], device=device)
    a_target_colour = t.tensor([list(COLOURS).index(s.objects[i].colour) for s, i in zip(a_scenes, a_targets)], device=device)
    
    with t.no_grad():
        a_embeds = build_input_embeddings(vlm, projector(test_features[:N_ANALYSIS]), a_tokens)
        _, a_cache = vlm.run_with_cache(
            a_embeds,
            start_at_layer=0,
            stop_at_layer=N_LAYERS,
            names_filter=lambda name: name.endswith("resid_pre") or name.endswith("hook_pattern") or name == f"blocks.{N_LAYERS - 1}.hook_resid_post",
        )
        a_logits = vlm.unembed(vlm.ln_final(vlm(a_embeds, start_at_layer=0, stop_at_layer=N_LAYERS)[:, ANSWER_POS]))
    
    print(f"sequence length {SEQ_LEN}, answer position {ANSWER_POS}")
    print("question tokens:", [vlm.to_string(tok) for tok in a_tokens[0]])
    print(f"forced-choice accuracy on these {N_ANALYSIS} questions: {answer_accuracy(a_logits, colour_ids[a_target_colour], colour_ids):.1%}")
    print(f"free-generation accuracy (top-1 over the whole vocabulary): {(a_logits.argmax(-1) == colour_ids[a_target_colour]).float().mean():.1%}")

# %%

def random_background_mask(
    object_masks: Bool[Tensor, "batch token"], occupied_masks: Bool[Tensor, "batch token"], seed: int = 0
) -> Bool[Tensor, "batch token"]:
    """For each row, a random set of background tokens (not in occupied_masks) as large as the object's token set."""
    generator = t.Generator().manual_seed(seed)
    out = t.zeros_like(object_masks)
    for b in range(len(object_masks)):
        k = int(object_masks[b].sum())
        free = (~occupied_masks[b]).nonzero().flatten().cpu()
        chosen = free[t.randperm(len(free), generator=generator)[:k]]
        out[b, chosen.to(out.device)] = True
    return out


if MAIN:
    tests.test_random_background_mask(random_background_mask)

# %%

if MAIN:
    def token_set_masks(scenes: list[Scene], targets: list[int], seed: int = 0) -> dict[str, Tensor]:
        """Boolean (n, 49) masks: the target object's tokens, the other objects' tokens, and the same-size random control."""
        object_masks = t.tensor(np.stack([utils.object_token_mask(s.objects[i]).flatten() for s, i in zip(scenes, targets)]), device=device)
        occupied = t.tensor(
            np.stack([np.any([utils.object_token_mask(o).flatten() for o in s.objects], axis=0) for s in scenes]), device=device
        )
        return {
            "object": object_masks,
            "other objects": occupied & ~object_masks,
            "occupied": occupied,
            "random background": random_background_mask(object_masks, occupied, seed),
        }
    
    
    a_masks = token_set_masks(a_scenes, a_targets)
    print(f"tokens per object: {a_masks['object'].sum(1).float().mean():.1f} on average (min {a_masks['object'].sum(1).min()}, max {a_masks['object'].sum(1).max()})")
    print(f"background tokens: {(~a_masks['occupied']).sum(1).float().mean():.1f} on average")
    show = list(range(5))
    utils.plot_token_maps(
        [a_scenes[i] for i in show],
        [a_targets[i] for i in show],
        [a_masks["object"][show].float().cpu().numpy(), a_masks["random background"][show].float().cpu().numpy(), a_masks["other objects"][show].float().cpu().numpy()],
        ["the object's tokens", "same-size random\nbackground (control)", "other objects'\ntokens"],
        [f"question about the {a_scenes[i].objects[a_targets[i]].colour} {a_scenes[i].objects[a_targets[i]].shape}" for i in show],
        suptitle="The token sets we will compare (yellow = in the set; white outline = the queried object)",
    )

# %%

if MAIN:
    LAYER_LABELS = ["embed"] + [f"pre {layer}" for layer in range(1, N_LAYERS)] + ["final"]
    # (13, batch, 49, d_model): the residual stream at the image tokens, before every block and after the last one
    image_resid = t.stack(
        [a_cache["resid_pre", layer][:, IMAGE_SLICE] for layer in range(N_LAYERS)] + [a_cache["resid_post", N_LAYERS - 1][:, IMAGE_SLICE]]
    )
    
    
    def top_tokens(resid: Float[Tensor, "d_model"], k: int = 3) -> str:
        probs = vlm.unembed(vlm.ln_final(resid)).softmax(-1)
        return ", ".join(f"{vlm.to_string(i)!r}" for i in probs.topk(k).indices)
    
    
    b = 0
    obj_token = int(a_masks["object"][b].nonzero()[0])
    bg_token = int((~a_masks["occupied"][b]).nonzero()[0])
    print(f"scene 0: the question is about the {a_scenes[b].objects[a_targets[b]].colour} {a_scenes[b].objects[a_targets[b]].shape}")
    rows = {
        LAYER_LABELS[layer]: {"object token": top_tokens(image_resid[layer, b, obj_token]), "background token": top_tokens(image_resid[layer, b, bg_token])}
        for layer in [0, 1, 4, 8, 10, 11, 12]
    }
    with pd.option_context("display.max_colwidth", 60):
        display(pd.DataFrame(rows).T)

# %%

def lens_ranks(
    model: HookedTransformer, resid: Float[Tensor, "layer token d_model"], word_id: int
) -> Int[Tensor, "layer token"]:
    """Rank (1 = top) of `word_id` in the logit-lens decoding of every residual stream vector."""
    logits = model.unembed(model.ln_final(resid))
    return (logits > logits[..., word_id, None]).sum(dim=-1) + 1


if MAIN:
    tests.test_lens_ranks(lens_ranks, vlm)

# %%

def restricted_lens_correct(
    model: HookedTransformer,
    resid: Float[Tensor, "layer token d_model"],
    candidate_ids: Int[Tensor, " candidate"],
    target_index: int,
) -> Bool[Tensor, "layer token"]:
    """True where the logit lens, restricted to the candidate words, puts the target candidate first."""
    logits = model.ln_final(resid) @ model.W_U[:, candidate_ids] + model.b_U[candidate_ids]
    return logits.argmax(dim=-1) == target_index


if MAIN:
    tests.test_restricted_lens_correct(restricted_lens_correct, vlm)

# %%

if MAIN:
    lens_stats = {key: [] for key in ["rank_obj", "rank_bg", "rank_rand", "acc_obj", "acc_bg", "acc_rand"]}
    for b in range(N_ANALYSIS):
        word_id, target = colour_ids[a_target_colour[b]], int(a_target_colour[b])
        obj_mask, bg_mask = a_masks["object"][b], ~a_masks["occupied"][b]
        resid = image_resid[:, b]  # (13, 49, d_model)
        noise = t.randn(resid[:, obj_mask].shape, generator=t.Generator().manual_seed(b)).to(device)
        noise = noise / noise.norm(dim=-1, keepdim=True) * resid[:, obj_mask].norm(dim=-1, keepdim=True)  # same norm as the real vectors
        ranks = lens_ranks(vlm, resid, word_id)
        correct = restricted_lens_correct(vlm, resid, colour_ids, target)
        lens_stats["rank_obj"].append(ranks[:, obj_mask].float().median(dim=1).values)
        lens_stats["rank_bg"].append(ranks[:, bg_mask].float().median(dim=1).values)
        lens_stats["rank_rand"].append(lens_ranks(vlm, noise, word_id).float().median(dim=1).values)
        lens_stats["acc_obj"].append(correct[:, obj_mask].float().mean(dim=1))
        lens_stats["acc_bg"].append(correct[:, bg_mask].float().mean(dim=1))
        lens_stats["acc_rand"].append(restricted_lens_correct(vlm, noise, colour_ids, target).float().mean(dim=1))
    lens_stats = {k: t.stack(v) for k, v in lens_stats.items()}  # each (N_ANALYSIS, 13)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    utils.plot_layer_curves(
        {
            "object tokens": lens_stats["rank_obj"].median(0).values.cpu().numpy(),
            "background tokens": lens_stats["rank_bg"].median(0).values.cpu().numpy(),
            "random vectors (norm-matched)": lens_stats["rank_rand"].median(0).values.cpu().numpy(),
        },
        "Plain logit lens: rank of the true colour word",
        "median rank among 50,257 tokens (log scale)",
        controls=("background tokens", "random vectors (norm-matched)"),
        logy=True,
        ax=axes[0],
    )
    utils.plot_layer_curves(
        {
            "object tokens": lens_stats["acc_obj"].mean(0).cpu().numpy(),
            "background tokens": lens_stats["acc_bg"].mean(0).cpu().numpy(),
            "random vectors (norm-matched)": lens_stats["acc_rand"].mean(0).cpu().numpy(),
        },
        "Lens restricted to the six colour words",
        "fraction decoded as the true colour",
        controls=("background tokens", "random vectors (norm-matched)"),
        ylim=(0, 1.02),
        hlines={"chance (1/6)": 1 / 6},
        ax=axes[1],
    )
    for ax in axes:
        ax.set_xticks(range(13))
        ax.set_xticklabels(LAYER_LABELS, rotation=45, ha="right", fontsize=8)
    plt.tight_layout()
    plt.show()

# %%

def attention_mass(
    patterns: Float[Tensor, "layer head query key"], query_pos: int, key_mask: Bool[Tensor, " key"]
) -> Float[Tensor, "layer head"]:
    """Total attention from position `query_pos` onto the key positions where `key_mask` is True."""
    return (patterns[:, :, query_pos] * key_mask).sum(dim=-1)


if MAIN:
    tests.test_attention_mass(attention_mass)

# %%

if MAIN:
    def key_mask_for(token_mask: Tensor | None = None, positions: slice | None = None) -> Tensor:
        """A (SEQ_LEN,) boolean key mask: either image tokens given by a (49,) mask, or a slice of sequence positions."""
        mask = t.zeros(SEQ_LEN, dtype=t.bool, device=device)
        if token_mask is not None:
            mask[IMAGE_SLICE] = token_mask
        if positions is not None:
            mask[positions] = True
        return mask
    
    
    mass = {name: [] for name in ["object", "same-size random background", "other objects", "BOS", "question"]}
    for b in range(N_ANALYSIS):
        patterns = t.stack([a_cache["pattern", layer][b] for layer in range(N_LAYERS)])  # (layer, head, query, key)
        mass["object"].append(attention_mass(patterns, ANSWER_POS, key_mask_for(a_masks["object"][b])))
        mass["same-size random background"].append(attention_mass(patterns, ANSWER_POS, key_mask_for(a_masks["random background"][b])))
        mass["other objects"].append(attention_mass(patterns, ANSWER_POS, key_mask_for(a_masks["other objects"][b])))
        mass["BOS"].append(attention_mass(patterns, ANSWER_POS, key_mask_for(positions=slice(0, 1))))
        mass["question"].append(attention_mass(patterns, ANSWER_POS, key_mask_for(positions=slice(1 + N_IMAGE_TOKENS, ANSWER_POS))))
    mass = {name: t.stack(v) for name, v in mass.items()}  # each (N_ANALYSIS, layer, head)
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
    utils.plot_layer_curves(
        {name: mass[name].mean((0, 2)).cpu().numpy() for name in ["object", "same-size random background", "other objects", "question"]},
        "Attention from the answer position (mean over heads and 200 questions)",
        "fraction of the answer position's attention",
        xlabel="layer",
        controls=("same-size random background", "other objects"),
        ax=axes[0],
    )
    utils.plot_head_heatmap(mass["object"].mean(0), "Attention on the object's tokens, per head", ax=axes[1], vmax=1)
    plt.tight_layout()
    plt.show()
    
    per_head = mass["object"].mean(0)
    top = per_head.flatten().topk(8)
    print("heads with the most attention on the object's tokens (layer.head: object / same-size random control):")
    for value, index in zip(top.values.tolist(), top.indices.tolist()):
        layer, head = divmod(index, vlm.cfg.n_heads)
        control = mass["same-size random background"].mean(0)[layer, head].item()
        print(f"  L{layer}.H{head}: {value:.2f} / {control:.2f}")

# %%

if MAIN:
    best_layer, best_head = divmod(int(per_head.flatten().argmax()), vlm.cfg.n_heads)
    show = [0, 1, 2, 3, 4]
    head_maps = np.stack([a_cache["pattern", best_layer][b, best_head, ANSWER_POS, IMAGE_SLICE].cpu().numpy() for b in show])
    utils.plot_token_maps(
        [a_scenes[i] for i in show],
        [a_targets[i] for i in show],
        [head_maps],
        [f"attention of L{best_layer}.H{best_head}\nfrom the answer position"],
        [f"question about the {a_scenes[i].objects[a_targets[i]].colour} {a_scenes[i].objects[a_targets[i]].shape}" for i in show],
        suptitle=f"Where does the answer position look? (head L{best_layer}.H{best_head}; white outline = the queried object)",
    )

# %%

if MAIN:
    pairs = utils.make_counterfactual_pairs(100, seed=7)
    clean_scenes, corrupt_scenes = [p[0] for p in pairs], [p[1] for p in pairs]
    pair_targets = [p[2] for p in pairs]
    clean_features, corrupt_features = featurise(clean_scenes), featurise(corrupt_scenes)
    
    pair_prompts = [question_and_answer(scene, "colour", target)[0] for scene, target in zip(clean_scenes, pair_targets)]
    pair_tokens, pair_lens = utils.tokenize_prompts(vlm, pair_prompts)
    assert (pair_lens == a_lens[0]).all()
    clean_ids = t.tensor([vlm.to_single_token(" " + s.objects[i].colour) for s, i in zip(clean_scenes, pair_targets)], device=device)
    corrupt_ids = t.tensor([vlm.to_single_token(" " + s.objects[i].colour) for s, i in zip(corrupt_scenes, pair_targets)], device=device)
    
    with t.no_grad():
        clean_embeds = build_input_embeddings(vlm, projector(clean_features), pair_tokens)
        corrupt_embeds = build_input_embeddings(vlm, projector(corrupt_features), pair_tokens)
        _, clean_cache = vlm.run_with_cache(
            clean_embeds, start_at_layer=0, stop_at_layer=N_LAYERS, names_filter=lambda name: name.endswith("resid_pre")
        )
        clean_resids = {layer: clean_cache["resid_pre", layer] for layer in range(N_LAYERS)}  # layer -> (100, 60, 768)
        clean_logits = vlm.unembed(vlm.ln_final(vlm(clean_embeds, start_at_layer=0, stop_at_layer=N_LAYERS)[:, ANSWER_POS]))
        corrupt_logits = vlm.unembed(vlm.ln_final(vlm(corrupt_embeds, start_at_layer=0, stop_at_layer=N_LAYERS)[:, ANSWER_POS]))
    
    print(f"the model gets the clean scenes right: {(clean_logits.argmax(-1) == clean_ids).float().mean():.0%}, the corrupted scenes right: {(corrupt_logits.argmax(-1) == corrupt_ids).float().mean():.0%}")
    rows = t.arange(len(pairs))
    ld_clean = (clean_logits[rows, clean_ids] - clean_logits[rows, corrupt_ids]).mean().item()
    ld_corrupt = (corrupt_logits[rows, clean_ids] - corrupt_logits[rows, corrupt_ids]).mean().item()
    print(f"mean logit difference (clean colour - corrupt colour): clean run {ld_clean:+.1f}, corrupted run {ld_corrupt:+.1f}")

# %%

def patched_answer_logits(
    model: HookedTransformer,
    corrupt_embeds: Float[Tensor, "batch seq d_model"],
    clean_resid: Float[Tensor, "batch seq d_model"],
    layer: int,
    patch_mask: Bool[Tensor, "batch seq"],
    answer_pos: int,
) -> Float[Tensor, "batch d_vocab"]:
    """Logits at the answer position of the corrupted run, with resid_pre of block `layer` overwritten from the clean run where patch_mask."""
    def patch_hook(resid: Float[Tensor, "batch seq d_model"], hook) -> Float[Tensor, "batch seq d_model"]:
        resid = resid.clone()
        resid[patch_mask] = clean_resid[patch_mask]
        return resid

    resid = model.run_with_hooks(
        corrupt_embeds,
        start_at_layer=0,
        stop_at_layer=model.cfg.n_layers,
        fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", patch_hook)],
    )
    return model.unembed(model.ln_final(resid[:, answer_pos]))


if MAIN:
    tests.test_patched_answer_logits(patched_answer_logits, vlm)

# %%

def logit_diff_recovery(
    patched_logits: Float[Tensor, "batch d_vocab"],
    clean_logits: Float[Tensor, "batch d_vocab"],
    corrupt_logits: Float[Tensor, "batch d_vocab"],
    clean_ids: Int[Tensor, " batch"],
    corrupt_ids: Int[Tensor, " batch"],
) -> Float[Tensor, " batch"]:
    """0 = patched run looks like the corrupted run, 1 = it looks like the clean run (in logit difference between the two answers)."""
    def logit_diff(logits: Tensor) -> Tensor:
        rows = t.arange(len(logits), device=logits.device)
        return logits[rows, clean_ids] - logits[rows, corrupt_ids]

    return (logit_diff(patched_logits) - logit_diff(corrupt_logits)) / (logit_diff(clean_logits) - logit_diff(corrupt_logits))


if MAIN:
    tests.test_logit_diff_recovery(logit_diff_recovery)

# %%

def patching_sweep(
    model: HookedTransformer,
    corrupt_embeds: Float[Tensor, "batch seq d_model"],
    clean_resids: dict[int, Float[Tensor, "batch seq d_model"]],
    masks: dict[str, Bool[Tensor, "batch seq"]],
    clean_logits: Float[Tensor, "batch d_vocab"],
    corrupt_logits: Float[Tensor, "batch d_vocab"],
    clean_ids: Int[Tensor, " batch"],
    corrupt_ids: Int[Tensor, " batch"],
    answer_pos: int,
    layers: list[int],
) -> dict[str, list[float]]:
    """For each named position mask: the mean (over the batch) recovery from patching it at each layer in `layers`."""
    curves = {name: [] for name in masks}
    with t.no_grad():
        for layer in layers:
            for name, mask in masks.items():
                patched = patched_answer_logits(model, corrupt_embeds, clean_resids[layer], layer, mask, answer_pos)
                recovery = logit_diff_recovery(patched, clean_logits, corrupt_logits, clean_ids, corrupt_ids)
                curves[name].append(recovery.mean().item())
    return curves


if MAIN:
    tests.test_patching_sweep(patching_sweep, vlm)

# %%

if MAIN:
    p_masks = token_set_masks(clean_scenes, pair_targets, seed=1)
    
    
    def seq_mask(token_mask: Tensor | None = None, positions: slice | None = None) -> Tensor:
        """(100, SEQ_LEN) boolean mask: image tokens given by a (100, 49) mask, and/or a slice of sequence positions."""
        mask = t.zeros(len(pairs), SEQ_LEN, dtype=t.bool, device=device)
        if token_mask is not None:
            mask[:, IMAGE_SLICE] = token_mask
        if positions is not None:
            mask[:, positions] = True
        return mask
    
    
    patch_sets = {
        "object tokens": seq_mask(p_masks["object"]),
        "same-size random background": seq_mask(p_masks["random background"]),
        "other objects": seq_mask(p_masks["other objects"]),
        "all image tokens": seq_mask(t.ones_like(p_masks["object"])),
        "question tokens (50-58)": seq_mask(positions=slice(1 + N_IMAGE_TOKENS, ANSWER_POS)),
        "answer position (59)": seq_mask(positions=slice(ANSWER_POS, ANSWER_POS + 1)),
    }
    patch_curves = patching_sweep(
        vlm, corrupt_embeds, clean_resids, patch_sets, clean_logits, corrupt_logits, clean_ids, corrupt_ids, ANSWER_POS, list(range(N_LAYERS))
    )
    utils.plot_layer_curves(
        patch_curves,
        "Patching clean -> corrupted: fraction of the logit difference recovered",
        "recovery (0 = corrupted answer, 1 = clean answer)",
        controls=("same-size random background", "other objects", "all image tokens"),
        ylim=(-0.05, 1.08),
    )
    print(pd.DataFrame(patch_curves, index=[f"{layer}" for layer in range(N_LAYERS)]).T.round(2).to_string())

# %%

if MAIN:
    @t.no_grad()
    def single_token_recovery(pair: int, layer: int) -> Float[Tensor, "token"]:
        """Recovery obtained by patching each of the 49 image tokens (separately) of one pair at the given layer."""
        mask = t.zeros(N_IMAGE_TOKENS, SEQ_LEN, dtype=t.bool, device=device)
        mask[t.arange(N_IMAGE_TOKENS), 1 + t.arange(N_IMAGE_TOKENS)] = True
        expand = lambda x: x[pair : pair + 1].expand(N_IMAGE_TOKENS, *x.shape[1:])
        patched = patched_answer_logits(vlm, expand(corrupt_embeds), expand(clean_resids[layer]), layer, mask, ANSWER_POS)
        return logit_diff_recovery(patched, expand(clean_logits), expand(corrupt_logits), expand(clean_ids), expand(corrupt_ids))
    
    
    show = [0, 1, 2, 3, 4]
    map_layers = [0, 6, 10, 11]
    token_maps = [np.stack([single_token_recovery(i, layer).cpu().numpy() for i in show]) for layer in map_layers]
    for layer, m in zip(map_layers, token_maps):
        in_obj = np.concatenate([m[k][p_masks["object"][i].cpu().numpy()] for k, i in enumerate(show)])
        in_bg = np.concatenate([m[k][~p_masks["occupied"][i].cpu().numpy()] for k, i in enumerate(show)])
        print(f"layer {layer:>2}: mean recovery of a single object token {in_obj.mean():.2f} (max {in_obj.max():.2f}), of a single background token {in_bg.mean():.3f} (max {in_bg.max():.3f})")
    utils.plot_token_maps(
        corrupt_scenes[:5],
        pair_targets[:5],
        token_maps,
        [f"patch one token\nat layer {layer}" for layer in map_layers],
        [f"corrupted scene: {corrupt_scenes[i].objects[pair_targets[i]].colour} {corrupt_scenes[i].objects[pair_targets[i]].shape}\n(clean: {clean_scenes[i].objects[pair_targets[i]].colour})" for i in show],
        vmax=0.5,
        suptitle="Fraction of the logit difference recovered by patching ONE image token (white outline = the recoloured object)",
    )

# %%

def block_attention_logits(
    model: HookedTransformer,
    embeds: Float[Tensor, "batch seq d_model"],
    layers: list[int],
    block: Bool[Tensor, "batch seq seq"],
    answer_pos: int,
) -> Float[Tensor, "batch d_vocab"]:
    """Answer-position logits when, in every block in `layers`, query q may not attend to key k wherever block[b, q, k]."""
    def block_hook(scores: Float[Tensor, "batch head query key"], hook) -> Float[Tensor, "batch head query key"]:
        return scores.masked_fill(block[:, None], float("-inf"))

    resid = model.run_with_hooks(
        embeds,
        start_at_layer=0,
        stop_at_layer=model.cfg.n_layers,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_attn_scores", block_hook) for layer in layers],
    )
    return model.unembed(model.ln_final(resid[:, answer_pos]))


if MAIN:
    tests.test_block_attention_logits(block_attention_logits, vlm)

# %%

if MAIN:
    def route_block(query: str, keys: str) -> Bool[Tensor, "batch seq seq"]:
        """Boolean (N_ANALYSIS, seq, seq) 'do not attend' mask for a route: queries in {answer, question} x keys in {object, ...}."""
        query_pos = {"answer": slice(ANSWER_POS, ANSWER_POS + 1), "question": slice(1 + N_IMAGE_TOKENS, ANSWER_POS)}[query]
        key_mask = {
            "object": lambda: t.stack([key_mask_for(a_masks["object"][b]) for b in range(N_ANALYSIS)]),
            "random background": lambda: t.stack([key_mask_for(a_masks["random background"][b]) for b in range(N_ANALYSIS)]),
            "image": lambda: key_mask_for(positions=IMAGE_SLICE).expand(N_ANALYSIS, -1),
            "question": lambda: key_mask_for(positions=slice(1 + N_IMAGE_TOKENS, ANSWER_POS)).expand(N_ANALYSIS, -1),
        }[keys]()
        block = t.zeros(N_ANALYSIS, SEQ_LEN, SEQ_LEN, dtype=t.bool, device=device)
        block[:, query_pos, :] = key_mask[:, None, :]
        return block
    
    
    @t.no_grad()
    def blocked_accuracy(layers: list[int], block: Tensor) -> float:
        logits = block_attention_logits(vlm, a_embeds, layers, block, ANSWER_POS)
        return answer_accuracy(logits, colour_ids[a_target_colour], colour_ids)
    
    
    routes = {
        "answer -> object tokens": ("answer", "object"),
        "answer -> same-size random background": ("answer", "random background"),
        "answer -> all image tokens": ("answer", "image"),
        "question tokens -> all image tokens": ("question", "image"),
        "answer -> question tokens": ("answer", "question"),
    }
    windows = [(0, 3), (3, 6), (6, 9), (9, 12)]
    knockout = {}
    for name, (query, keys) in routes.items():
        block = route_block(query, keys)
        knockout[name] = [blocked_accuracy(list(range(lo, hi)), block) for lo, hi in windows] + [blocked_accuracy(list(range(N_LAYERS)), block)]
    window_labels = [f"layers {lo}-{hi - 1}" for lo, hi in windows] + ["all layers"]
    display(pd.DataFrame(knockout, index=window_labels).T.round(2))
    
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, (name, accs) in enumerate(knockout.items()):
        style = "--s" if "random" in name or name.startswith("question") else "-o"
        ax.plot(range(len(accs)), accs, style, label=name, color=[utils.BLUE, utils.GREY, utils.GREEN, utils.ORANGE, utils.PURPLE][i], markersize=5)
    ax.axhline(IMAGE_BLIND["colour"], color="k", ls=":", lw=1.2, label="best image-blind guess")
    ax.set_xticks(range(len(window_labels)))
    ax.set_xticklabels(window_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("colour accuracy (forced choice)")
    ax.set_xlabel("blocked in which layers")
    ax.set_title("Attention knockout: block a route, measure the answer")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    plt.tight_layout()
    plt.show()

# %%

if MAIN:
    late = slice(9, 12)  # the layers where the answer position reads the image
    early = slice(0, 9)
    summary = pd.DataFrame(
        {
            "object's tokens": {
                "decodable: restricted lens accuracy (mean over layers)": lens_stats["acc_obj"][:, :12].mean().item(),
                "attended: attention mass from answer position, layers 9-11": mass["object"][:, late].mean().item(),
                "sufficient: patching recovery, layers 0-8": float(np.mean([patch_curves["object tokens"][layer] for layer in range(9)])),
                "necessary: accuracy when the answer position can't see them, layers 9-11": knockout["answer -> object tokens"][3],
            },
            "same-size control": {
                "decodable: restricted lens accuracy (mean over layers)": lens_stats["acc_bg"][:, :12].mean().item(),
                "attended: attention mass from answer position, layers 9-11": mass["same-size random background"][:, late].mean().item(),
                "sufficient: patching recovery, layers 0-8": float(np.mean([patch_curves["same-size random background"][layer] for layer in range(9)])),
                "necessary: accuracy when the answer position can't see them, layers 9-11": knockout["answer -> same-size random background"][3],
            },
        }
    )
    display(summary.round(2))

# %%

if MAIN:
    @t.no_grad()
    def patch_demo(clean: list[SceneObject], corrupt: list[SceneObject], layer: int, where="object", target: int = 0) -> None:
        clean_scene, corrupt_scene = Scene(clean, noise_seed=0), Scene(corrupt, noise_seed=0)
        prompt, _ = question_and_answer(corrupt_scene, "colour", target)
        tokens, _ = utils.tokenize_prompts(vlm, [prompt])
        feats = featurise([clean_scene, corrupt_scene])
        embeds = build_input_embeddings(vlm, projector(feats), tokens.expand(2, -1))
        _, cache = vlm.run_with_cache(embeds, start_at_layer=0, stop_at_layer=N_LAYERS, names_filter=lambda name: name.endswith("resid_pre"))
        logits = vlm.unembed(vlm.ln_final(vlm(embeds, start_at_layer=0, stop_at_layer=N_LAYERS)[:, ANSWER_POS]))
        obj = np.asarray(utils.object_token_mask(corrupt_scene.objects[target])).flatten()
        mask = t.zeros(1, SEQ_LEN, dtype=t.bool, device=device)
        if where == "object":
            mask[0, IMAGE_SLICE] = t.tensor(obj, device=device)
        elif where == "background":
            occupied = t.tensor(np.any([utils.object_token_mask(o).flatten() for o in corrupt_scene.objects], axis=0), device=device)[None]
            mask[0, IMAGE_SLICE] = random_background_mask(t.tensor(obj, device=device)[None], occupied)[0]
        elif where == "all":
            mask[0, IMAGE_SLICE] = True
        elif where == "answer":
            mask[0, ANSWER_POS] = True
        else:
            for row, col in where:
                mask[0, 1 + TOKEN_GRID * row + col] = True
        patched = patched_answer_logits(vlm, embeds[1:], cache["resid_pre", layer][:1], layer, mask, ANSWER_POS)
        candidates = [" " + c for c in COLOURS]
    
        def show(name: str, lg: Tensor) -> str:
            probs = lg[colour_ids].softmax(-1)
            best = int(probs.argmax())
            return f"{name:<12} -> {candidates[best].strip():<7} ({probs[best]:.2f})"
    
        print(prompt[11:-8])
        print(show("clean scene", logits[0]), "|", show("corrupted", logits[1]), "|", show(f"patched L{layer}", patched[0]))
        print(f"patched {int(mask.sum())} positions ({where!r} at layer {layer})")
    
    
    clean_objects = [SceneObject("red", "circle", (0, 0)), SceneObject("blue", "square", (1, 1))]
    corrupt_objects = [SceneObject("green", "circle", (0, 0)), SceneObject("blue", "square", (1, 1))]
    for where, layer in [("object", 4), ("background", 4), ("object", 11), ("answer", 11)]:
        patch_demo(clean_objects, corrupt_objects, layer, where)

# %%

if MAIN:
    # Free what we no longer need (rerun the earlier cells if you want these back), then load Qwen
    for name in ["text_only_lm", "train_features", "a_cache", "image_resid", "a_embeds", "clean_cache", "clean_resids", "clean_embeds", "corrupt_embeds"]:
        globals().pop(name, None)
    t.cuda.empty_cache()
    qwen, qwen_processor = utils.load_qwen(device)
    qwen_tokenizer = qwen_processor.tokenizer
    N_QWEN_LAYERS = qwen.config.text_config.num_hidden_layers
    print(f"Qwen3-VL-2B-Instruct: {sum(p.numel() for p in qwen.parameters()) / 1e9:.2f}B parameters, {N_QWEN_LAYERS} language-model layers, d_model {qwen.config.text_config.hidden_size}")
    
    example_scene = test_scenes[0]
    example_obj = example_scene.objects[0]
    example_question = f"What colour is the {example_obj.shape}? Answer with one word."
    example_inputs = utils.qwen_inputs(qwen_processor, utils.render_scene(example_scene), example_question, device)
    ids = example_inputs["input_ids"][0]
    n_image = int((ids == qwen.config.image_token_id).sum())
    print(f"prompt: {len(ids)} tokens, of which {n_image} image tokens (an 8 x 8 grid)")
    print("first tokens:", [qwen_tokenizer.decode(i) for i in ids[:5]])
    print("last tokens: ", [qwen_tokenizer.decode(i) for i in ids[-8:]])
    with t.no_grad():
        generated = qwen.generate(**example_inputs, max_new_tokens=4, do_sample=False)
    print(f"the question about the {example_obj.colour} {example_obj.shape} -> {qwen_tokenizer.decode(generated[0, len(ids):], skip_special_tokens=True)!r}")

# %%

@t.no_grad()
def qwen_forced_choice(model, inputs, candidate_ids: Int[Tensor, " candidate"]) -> Float[Tensor, " candidate"]:
    """Logits of the candidate tokens at the last position of the prompt."""
    logits = model(**inputs).logits[0, -1]
    return logits[candidate_ids]


if MAIN:
    tests.test_qwen_forced_choice(qwen_forced_choice, qwen, qwen_processor)

# %%

if MAIN:
    qwen_colour_ids = t.tensor([qwen_tokenizer.encode(c)[0] for c in COLOURS], device=device)
    assert all(len(qwen_tokenizer.encode(c)) == 1 for c in COLOURS), "every colour should be a single Qwen token"
    
    N_QWEN = 100
    qwen_rng = np.random.default_rng(11)
    q_scenes = test_scenes[:N_QWEN]
    q_targets = [int(qwen_rng.integers(len(s.objects))) for s in q_scenes]
    q_images = utils.render_scenes(q_scenes)
    q_conditions = {
        "true image": q_images,
        "blank image": np.stack([utils.render_scene(Scene([], background=235))] * N_QWEN),
        "noise image": utils.noise_images(N_QWEN, seed=1),
        "shuffled image-question pairs": q_images[derangement(N_QWEN, seed=0).numpy()],
    }
    qwen_accuracy = {}
    qwen_control_traces = []
    for name, images in tqdm(q_conditions.items()):
        correct = 0
        for i in range(N_QWEN):
            obj = q_scenes[i].objects[q_targets[i]]
            inputs = utils.qwen_inputs(qwen_processor, images[i], f"What colour is the {obj.shape}? Answer with one word.", device)
            scores = qwen_forced_choice(qwen, inputs, qwen_colour_ids).float().cpu()
            prediction = list(COLOURS)[int(scores.argmax())]
            correct += prediction == obj.colour
            qwen_control_traces.append(dict(condition=name, scene_index=i, kind="colour",
                object_count=len(q_scenes[i].objects), target_index=q_targets[i],
                model_prompt=f"What colour is the {obj.shape}? Answer with one word.",
                answer=obj.colour, prediction=prediction, correct=prediction == obj.colour,
                candidate_words=list(COLOURS), candidate_ids=qwen_colour_ids.cpu().tolist(),
                candidate_logits=scores.tolist(), target_blind_oracle=target_blind_oracle(q_scenes[i], "colour")))
        qwen_accuracy[name] = correct / N_QWEN
    # Literal image-only input: true scene, empty question; same targets and six scoring candidates.
    correct = 0
    for i in range(N_QWEN):
        obj = q_scenes[i].objects[q_targets[i]]
        inputs = utils.qwen_inputs(qwen_processor, q_images[i], "", device)
        scores = qwen_forced_choice(qwen, inputs, qwen_colour_ids).float().cpu()
        prediction = list(COLOURS)[int(scores.argmax())]
        correct += prediction == obj.colour
        qwen_control_traces.append(dict(condition="image-only (no question)", scene_index=i, kind="colour",
            object_count=len(q_scenes[i].objects), target_index=q_targets[i], model_prompt="",
            answer=obj.colour, prediction=prediction, correct=prediction == obj.colour,
            candidate_words=list(COLOURS), candidate_ids=qwen_colour_ids.cpu().tolist(),
            candidate_logits=scores.tolist(), target_blind_oracle=target_blind_oracle(q_scenes[i], "colour")))
    qwen_accuracy["image-only (no question)"] = correct / N_QWEN
    print({k: f"{v:.0%}" for k, v in qwen_accuracy.items()})
    utils.plot_bars(
        qwen_accuracy,
        "Qwen3-VL-2B: colour accuracy (forced choice) with the true image and with modality controls",
        colours=[utils.BLUE] + [utils.GREY] * 3 + [utils.ORANGE],
        hlines={"chance (1/6)": 1 / 6},
    )
    display(pd.DataFrame(qwen_control_traces).groupby(["condition", "object_count"])[["correct", "target_blind_oracle"]].mean().round(3))

# %%

@t.no_grad()
def qwen_patched_logits(
    model, inputs, layer: int, positions: Tensor, source_hidden: Float[Tensor, "1 seq d_model"]
) -> Float[Tensor, " vocab"]:
    """Last-position logits of the run on `inputs`, with the residual stream before `layer` overwritten from `source_hidden` at `positions`."""
    def patch_hook(module, args, kwargs):
        hidden = (kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]).clone()
        hidden[:, positions] = source_hidden[:, positions]
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hidden
            return args, kwargs
        return (hidden,) + tuple(args[1:]), kwargs

    handle = model.model.language_model.layers[layer].register_forward_pre_hook(patch_hook, with_kwargs=True)
    try:
        return model(**inputs).logits[0, -1]
    finally:
        handle.remove()


if MAIN:
    tests.test_qwen_patched_logits(qwen_patched_logits, qwen, qwen_processor)

# %%

def qwen_layer_schedule(num_layers: int) -> list[int]:
    """Every third layer plus the final layer, each intervention exactly once."""
    if num_layers < 1:
        raise ValueError("A model must contain at least one layer")
    return list(dict.fromkeys([*range(0, num_layers, 3), num_layers - 1]))


if MAIN:
    tests.test_qwen_layer_schedule(qwen_layer_schedule)

# %%

if MAIN:
    qwen_layers = qwen_layer_schedule(N_QWEN_LAYERS)
    qwen_set_names = ["object tokens", "same-size random background", "other objects", "all image tokens", "last position"]
    qwen_recovery = {name: np.zeros(len(qwen_layers)) for name in qwen_set_names}
    qwen_pair_traces = []
    qwen_pair_selection = []
    n_valid = 0
    for k in tqdm(range(20)):
        clean_scene, corrupt_scene, idx = pairs[k]
        obj = clean_scene.objects[idx]
        question = f"What colour is the {obj.shape}? Answer with one word."
        inputs_clean = utils.qwen_inputs(qwen_processor, utils.render_scene(clean_scene), question, device)
        inputs_corrupt = utils.qwen_inputs(qwen_processor, utils.render_scene(corrupt_scene), question, device)
        clean_ci, corrupt_ci = list(COLOURS).index(obj.colour), list(COLOURS).index(corrupt_scene.objects[idx].colour)
        lc, lk = qwen_forced_choice(qwen, inputs_clean, qwen_colour_ids), qwen_forced_choice(qwen, inputs_corrupt, qwen_colour_ids)
        qwen_pair_selection.append(dict(pair_index=k, target_index=idx,
            clean_colour=obj.colour, corrupt_colour=corrupt_scene.objects[idx].colour,
            clean_candidate_logits=lc.float().cpu().tolist(), corrupt_candidate_logits=lk.float().cpu().tolist(),
            kept=int(lc.argmax()) == clean_ci and int(lk.argmax()) == corrupt_ci))
        if int(lc.argmax()) != clean_ci or int(lk.argmax()) != corrupt_ci:
            continue  # keep only pairs the model gets right on both scenes
        n_valid += 1
        cache = utils.qwen_cache_layer_inputs(qwen, inputs_clean)
        seq = inputs_clean["input_ids"].shape[1]
        image_positions = (inputs_clean["input_ids"][0] == qwen.config.image_token_id).nonzero().flatten()
        obj_mask = t.tensor(utils.qwen_object_token_mask(obj).flatten(), device=device)[None]
        occupied = t.tensor(np.any([utils.qwen_object_token_mask(o).flatten() for o in clean_scene.objects], axis=0), device=device)[None]
        token_sets = {
            "object tokens": obj_mask[0],
            "same-size random background": random_background_mask(obj_mask, occupied, seed=k)[0],
            "other objects": (occupied & ~obj_mask)[0],
            "all image tokens": t.ones_like(obj_mask[0]),
        }
        for name in qwen_set_names:
            positions = t.zeros(seq, dtype=t.bool, device=device)
            if name == "last position":
                positions[-1] = True
            else:
                positions[image_positions[token_sets[name]]] = True
            for li, layer in enumerate(qwen_layers):
                logits = qwen_patched_logits(qwen, inputs_corrupt, layer, positions, cache[layer])[qwen_colour_ids].float()
                ld = lambda lg: (lg[clean_ci] - lg[corrupt_ci]).item()
                denominator = ld(lc.float()) - ld(lk.float())
                if not np.isfinite(denominator) or denominator <= 0:
                    raise ValueError("Selected pair has an invalid recovery denominator")
                recovery = (ld(logits) - ld(lk.float())) / denominator
                if not np.isfinite(recovery):
                    raise ValueError("Nonfinite intervention recovery")
                qwen_recovery[name][li] += recovery
                qwen_pair_traces.append(dict(pair_index=k, layer=layer, token_set=name,
                    target_index=idx, clean_colour=obj.colour, corrupt_colour=corrupt_scene.objects[idx].colour,
                    clean_logit_difference=ld(lc.float()), corrupt_logit_difference=ld(lk.float()),
                    patched_logit_difference=ld(logits), denominator=denominator, recovery=recovery,
                    positions=positions.nonzero().flatten().cpu().tolist()))
    if n_valid == 0:
        raise ValueError("No correctly answered counterfactual pairs; do not publish undefined recovery")
    qwen_recovery = {name: (values / n_valid).tolist() for name, values in qwen_recovery.items()}
    print(f"{n_valid} of 20 pairs kept (both scenes answered correctly)")
    print(pd.DataFrame(qwen_recovery, index=qwen_layers).T.round(2).to_string())

# %%

if MAIN:
    mini_curves = {
        "object tokens": patch_curves["object tokens"],
        "same-size random background": patch_curves["same-size random background"],
        "other objects": patch_curves["other objects"],
        "all image tokens": patch_curves["all image tokens"],
        "last position": patch_curves["answer position (59)"],
    }
    utils.plot_relative_depth_curves(
        {
            f"mini VLM (GPT-2 Small, {N_LAYERS} layers)": (mini_curves, list(range(N_LAYERS)), N_LAYERS),
            f"Qwen3-VL-2B ({N_QWEN_LAYERS} layers)": (qwen_recovery, qwen_layers, N_QWEN_LAYERS),
        },
        controls=("same-size random background", "other objects", "all image tokens"),
    )

# %%

if MAIN:
    del qwen
    t.cuda.empty_cache()

# %%

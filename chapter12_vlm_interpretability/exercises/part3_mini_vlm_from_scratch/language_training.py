"""Development-only training and genuine word-generation metrics for the pilot.

The training loop and generation metrics are intended as notebook exercises.
Data collation is supplied plumbing. Final split evaluation is a separate step.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import torch as t

from .language_data import VOCAB, answer_words, tokenize_language_examples
from .language_vlm import WordMiniVLM, greedy_word_generation, teacher_forced_answer_loss

TASKS = ("color", "shape", "caption")


def make_word_task_batch(dataset, *, omit_question=False):
    """Every image appears with both VQA questions and a caption request."""
    scenes = tuple(scene for scene in dataset.scenes for _ in TASKS)
    tasks = TASKS * len(dataset.scenes)
    tokens = tokenize_language_examples(scenes, tasks, omit_question=omit_question)
    expected = t.full((len(scenes), 8), VOCAB["<pad>"], dtype=t.long)
    for row, (scene, task) in enumerate(zip(scenes, tasks, strict=True)):
        ids = [VOCAB[word] for word in answer_words(scene, task)] + [VOCAB["<eos>"]]
        expected[row, :len(ids)] = t.tensor(ids)
    return {**tokens, "images": dataset.images.repeat_interleave(len(TASKS), dim=0),
            "task_ids": t.arange(len(TASKS)).repeat(len(dataset.scenes)), "expected_tokens": expected}


def generated_word_exact_match(generation, expected_tokens):
    """Count complete word sequences including EOS, with no hidden bad-token cleanup."""
    if generation.tokens.shape != expected_tokens.shape:
        raise ValueError("Generated and reference token matrices must have the same padded width")
    expected_lengths = (expected_tokens != VOCAB["<pad>"]).sum(1)
    return (generation.finished & (generation.lengths == expected_lengths)
            & (generation.tokens == expected_tokens).all(dim=1))


@t.inference_mode()
def evaluate_word_model(model, batch, *, batch_size=64, zero_images=False):
    """Generate from answer-free word prompts and report every objective separately."""
    device = next(model.parameters()).device
    matches, generated = [], []
    for start in range(0, len(batch["images"]), batch_size):
        indices = slice(start, start + batch_size)
        images = batch["images"][indices].to(device)
        if zero_images:
            images = t.zeros_like(images)
        prompt_lengths = batch["prompt_lengths"][indices].to(device)
        width = int(prompt_lengths.max())
        valid = t.arange(width, device=device)[None, :] < prompt_lengths[:, None]
        prompt_ids = batch["token_ids"][indices, :width].to(device).masked_fill(~valid, VOCAB["<pad>"])
        result = greedy_word_generation(model, images, prompt_ids, valid, max_new_tokens=8)
        matches.append(generated_word_exact_match(result, batch["expected_tokens"][indices].to(device)).cpu())
        generated.append(result.tokens.cpu())
    exact = t.cat(matches)
    return {"exact_match": float(exact.float().mean()),
            **{task + "_exact_match": float(exact[batch["task_ids"] == index].float().mean())
               for index, task in enumerate(TASKS)},
            "per_example_exact_match": exact.tolist(), "generated_tokens": t.cat(generated).tolist()}


@dataclass
class WordTrainingRun:
    model: WordMiniVLM
    mode: str
    seed: int
    selected_step: int
    history: list[dict]
    development: dict
    changed_parameter_groups: dict[str, bool]
    elapsed_seconds: float


def train_word_model(train_dataset, development_dataset, *, steps=600, batch_size=48,
                     learning_rate=.003, seed=0, mode="joint", device="cpu",
                     evaluation_every=100, model_kwargs=None):
    """Fit on training instances and choose the best tested development checkpoint.

    No final familiar/compositional split is generated or scored here. Modes
    share initialization, examples and optimizer-step count; sequence lengths
    differ for image-only training, so FLOP costs are not claimed identical.
    """
    if mode not in ("joint", "text-only", "image-only", "shuffled-pairs"):
        raise ValueError("Choose a declared training/control mode")
    if train_dataset.split != "train" or development_dataset.split != "development":
        raise ValueError("Checkpoint selection must use training and development data only")
    if min(steps, batch_size, evaluation_every) < 1:
        raise ValueError("Training and evaluation counts must be positive")
    started = time.perf_counter()
    t.manual_seed(seed)
    model = WordMiniVLM(**(model_kwargs or {})).to(device)
    initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
    train = make_word_task_batch(train_dataset, omit_question=mode == "image-only")
    development = make_word_task_batch(development_dataset, omit_question=mode == "image-only")
    training_tensors = {key: value.to(device) for key, value in train.items()}
    if mode == "text-only":
        training_tensors["images"] = t.zeros_like(training_tensors["images"])
    if mode == "shuffled-pairs":
        generator = t.Generator().manual_seed(seed + 2000)
        permutation = t.randperm(len(train_dataset.scenes), generator=generator)
        image_indices = (permutation[:, None] * len(TASKS) + t.arange(len(TASKS))[None, :]).flatten().to(device)
        training_tensors["images"] = training_tensors["images"][image_indices]
    optimizer = t.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = t.Generator().manual_seed(seed + 1000)
    history, best_state, best_metric, best_step = [], None, -1., 0
    for step in range(steps):
        model.train()
        selected = t.randint(len(train["images"]), (batch_size,), generator=generator).to(device)
        logits = model(training_tensors["images"][selected], training_tensors["token_ids"][selected],
                       training_tensors["valid_mask"][selected])
        loss = teacher_forced_answer_loss(logits, training_tensors["token_ids"][selected],
                                          training_tensors["target_mask"][selected])
        optimizer.zero_grad()
        loss.backward()
        t.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        if (step + 1) % evaluation_every == 0 or step == steps - 1:
            measured = evaluate_word_model(model, development, zero_images=mode == "text-only")
            row = {"step": step + 1, "loss": float(loss.detach()),
                   **{key: value for key, value in measured.items() if key.endswith("exact_match") and not key.startswith("per_")}}
            history.append(row)
            if measured["exact_match"] > best_metric:
                best_metric, best_step = measured["exact_match"], step + 1
                best_state = copy.deepcopy(model.state_dict())
            print(f"{mode} seed {seed} step {step + 1}: loss {float(loss.detach()):.3f}; "
                  f"dev generation {measured['exact_match']:.3f} "
                  f"(color {measured['color_exact_match']:.3f}, shape {measured['shape_exact_match']:.3f}, caption {measured['caption_exact_match']:.3f})", flush=True)
    model.load_state_dict(best_state)
    model.eval()
    changed = {prefix: any(not t.equal(value, initial[name]) for name, value in model.state_dict().items()
                           if name.startswith(prefix))
               for prefix in ("vision.", "connector.", "word_embedding.", "decoder.", "unembed.")}
    measured = evaluate_word_model(model, development, zero_images=mode == "text-only")
    return WordTrainingRun(model, mode, seed, best_step, history, measured, changed, time.perf_counter() - started)

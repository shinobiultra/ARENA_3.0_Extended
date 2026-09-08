"""Controlled word overlays for the existing 12.1 pretrained CLIP exercise.

Rendering/loading are provided. The notebook implements the causal comparison
metric and visibly calls its own embedding extraction function.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch as t
from PIL import Image, ImageDraw, ImageFont

from .pixel_clip import PIXEL_CAPTIONS, PIXEL_RGB, PIXEL_SHAPES


def make_typographic_dataset(*, examples_per_class: int = 4, seed: int = 400) -> dict:
    generator = t.Generator().manual_seed(seed)
    font = ImageFont.truetype("DejaVuSans.ttf", size=20)
    labels = t.arange(12).repeat_interleave(examples_per_class)
    attack_labels = labels // 3 * 3 + (labels % 3 + 1) % 3
    images = {"clean": []}
    for placement in ("background", "object"):
        for word in (
            "blank_card",
            "correct_word",
            "conflicting_word",
            "irrelevant_word",
            "scrambled_word",
        ):
            images[f"{placement}/{word}"] = []
    boxes = []
    for label, attack_label in zip(labels.tolist(), attack_labels.tolist(), strict=True):
        color, shape = divmod(label, 3)
        cx = int(t.randint(85, 140, (), generator=generator))
        cy = int(t.randint(105, 150, (), generator=generator))
        radius = int(t.randint(28, 43, (), generator=generator))
        box = (cx - radius, cy - radius, cx + radius, cy + radius)
        clean = Image.new("RGB", (224, 224), "white")
        draw = ImageDraw.Draw(clean)
        if shape == 0:
            draw.rectangle(box, fill=PIXEL_RGB[color])
        elif shape == 1:
            draw.ellipse(box, fill=PIXEL_RGB[color])
        else:
            draw.polygon(
                ((cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)),
                fill=PIXEL_RGB[color],
            )
        images["clean"].append(clean)
        boxes.append(box)
        conflict = PIXEL_CAPTIONS[attack_label][2:]
        order = t.randperm(len(conflict.replace(" ", "")), generator=generator).tolist()
        letters = conflict.replace(" ", "")
        scrambled = "".join(letters[index] for index in order)
        for placement, y in (("background", 8), ("object", cy - 15)):
            card = (8, y, 216, y + 30)
            for condition, word in (
                ("blank_card", ""),
                ("correct_word", PIXEL_CAPTIONS[label][2:]),
                ("conflicting_word", conflict),
                ("irrelevant_word", "daily news"),
                ("scrambled_word", scrambled),
            ):
                image = clean.copy()
                draw = ImageDraw.Draw(image)
                draw.rectangle(card, fill="white")
                draw.text((112, y + 15), word, font=font, fill="black", anchor="mm")
                images[f"{placement}/{condition}"].append(image)
    return {
        "images": images,
        "labels": labels,
        "attack_labels": attack_labels,
        "object_boxes": boxes,
        "seed": seed,
        "font_name": "DejaVuSans.ttf",
        "font_size": 20,
        "font_sha256": hashlib.sha256(Path(font.path).read_bytes()).hexdigest(),
    }


def typographic_attack_metrics(
    baseline_logits: t.Tensor,
    condition_logits: t.Tensor,
    true_labels: t.Tensor,
    attack_labels: t.Tensor,
) -> dict:
    """Separate ordinary errors from targeted flips among baseline-correct images."""
    if baseline_logits.shape != condition_logits.shape or baseline_logits.ndim != 2:
        raise ValueError("baseline and condition need equal rectangular logit matrices")
    if true_labels.shape != (len(baseline_logits),) or attack_labels.shape != true_labels.shape:
        raise ValueError("provide one true and attack label per image")
    if bool((true_labels == attack_labels).any()):
        raise ValueError("a targeted attack must name a different class")
    index = t.arange(len(baseline_logits), device=baseline_logits.device)
    true_labels, attack_labels = true_labels.to(index.device), attack_labels.to(index.device)
    baseline_correct = baseline_logits.argmax(1) == true_labels
    prediction = condition_logits.argmax(1)
    baseline_margin = baseline_logits[index, true_labels] - baseline_logits[index, attack_labels]
    condition_margin = condition_logits[index, true_labels] - condition_logits[index, attack_labels]
    denominator = int(baseline_correct.sum())
    return {
        "image_count": len(index),
        "baseline_correct_count": denominator,
        "baseline_accuracy": float(baseline_correct.float().mean()),
        "accuracy": float((prediction == true_labels).float().mean()),
        "attack_choice_rate": float((prediction == attack_labels).float().mean()),
        "targeted_flip_rate": (
            float((prediction[baseline_correct] == attack_labels[baseline_correct]).float().mean())
            if denominator
            else None
        ),
        "mean_margin_change": float((condition_margin - baseline_margin).mean()),
    }


def run_typographic_study(
    model, processor, dataset: dict, *, extract_fn, metric_fn, batch_size: int = 24
) -> dict:
    """Provided batching; call the supplied learner extractor and metric explicitly."""
    device = next(model.parameters()).device
    captions = [f"a simple {caption[2:]} on a white background" for caption in PIXEL_CAPTIONS]
    matrices = {}
    for condition, images in dataset["images"].items():
        rows = []
        for start in range(0, len(images), batch_size):
            batch = processor(
                images=images[start : start + batch_size],
                text=captions,
                return_tensors="pt",
                padding=True,
            ).to(device)
            with t.inference_mode():
                image_embeddings, text_embeddings = extract_fn(model, batch)
                rows.append((model.logit_scale.exp() * image_embeddings @ text_embeddings.T).cpu())
        matrices[condition] = t.cat(rows)
    metrics = []
    for condition, logits in matrices.items():
        placement = condition.split("/")[0]
        card_key = f"{placement}/blank_card"
        baseline_key = card_key if card_key in matrices else "clean"
        clean = metric_fn(matrices["clean"], logits, dataset["labels"], dataset["attack_labels"])
        matched = metric_fn(
            matrices[baseline_key], logits, dataset["labels"], dataset["attack_labels"]
        )
        metrics.append(
            {
                "condition": condition,
                "matched_baseline": baseline_key,
                **clean,
                "matched_card_targeted_flip_rate": matched["targeted_flip_rate"],
                "matched_card_margin_change": matched["mean_margin_change"],
                "matched_card_correct_count": matched["baseline_correct_count"],
            }
        )
    return {
        "rows": metrics,
        "matrices": matrices,
        "seed": dataset["seed"],
        "font_sha256": dataset["font_sha256"],
        "font_size": dataset["font_size"],
        "image_count": len(dataset["labels"]),
        "captions": captions,
    }


def plot_typographic_study(result: dict, dataset: dict):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    for row, placement in enumerate(("background", "object")):
        selected = [
            item for item in result["rows"] if item["condition"].startswith(placement + "/")
        ]
        names = [item["condition"].split("/")[1].replace("_", " ") for item in selected]
        axes[row, 0].bar(names, [item["accuracy"] for item in selected])
        axes[row, 0].axhline(
            result["rows"][0]["accuracy"], linestyle="--", color="black", label="Clean image"
        )
        axes[row, 0].set(
            title=f"{placement}: zero-shot classification",
            ylim=(0, 1.05),
            ylabel="True-class accuracy",
        )
        axes[row, 0].tick_params(axis="x", rotation=40)
        axes[row, 0].legend(fontsize=8)
        axes[row, 1].bar(names, [item["matched_card_margin_change"] for item in selected])
        axes[row, 1].axhline(0, color="black", linewidth=0.8)
        axes[row, 1].set(
            title="Word effect beyond the blank card", ylabel="Change in true-minus-attack logit"
        )
        axes[row, 1].tick_params(axis="x", rotation=40)
        condition = f"{placement}/conflicting_word"
        # Fixed example zero, never selected by attack success.
        axes[row, 2].imshow(dataset["images"][condition][0])
        prediction = int(result["matrices"][condition][0].argmax())
        axes[row, 2].set_title(
            f"Fixed example 0: {PIXEL_CAPTIONS[int(dataset['labels'][0])]}\n"
            f"CLIP predicts: {PIXEL_CAPTIONS[prediction]}",
            fontsize=10,
        )
        axes[row, 2].axis("off")
    return figure

"""Provided controlled RGB scenes and word-token batches for the Mini VLM.

Images, questions and answers stay explicit. Model inputs contain pixels and
word IDs; renderer labels and geometry are reserved for targets/evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch as t
from PIL import Image, ImageDraw

COLORS = ("red", "blue", "green", "yellow")
SHAPES = ("square", "circle", "triangle")
RGB = ((230, 35, 30), (25, 65, 225), (25, 160, 55), (235, 200, 30))
HELDOUT_COMBINATIONS = frozenset({("red", "triangle"), ("blue", "circle"), ("green", "square")})
WORDS = ("<pad>", "<bos>", "<eos>", "<image>", "</image>", "<answer>",
         "what", "color", "shape", "is", "the", "object", "?", "describe", "image", ".", "a",
         *COLORS, *SHAPES)
VOCAB = {word: index for index, word in enumerate(WORDS)}
PROMPTS = {"color": "what color is the object ?", "shape": "what shape is the object ?",
           "caption": "describe the image ."}
SPLIT_SEEDS = {"train": 100, "development": 200, "familiar": 300, "compositional": 400}


@dataclass(frozen=True)
class LanguageScene:
    scene_id: str
    color: str
    shape: str
    cx: int
    cy: int
    radius: int
    background: int
    rgb_jitter: tuple[int, int, int]
    noise_seed: int
    style: str

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """Exclusive upper pixel bounds, including the renderer's boundary pixels."""
        return (self.cx - self.radius, self.cy - self.radius,
                self.cx + self.radius + 1, self.cy + self.radius + 1)


@dataclass(frozen=True)
class LanguageDataset:
    scenes: tuple[LanguageScene, ...]
    images: t.Tensor
    object_masks: t.Tensor
    split: str
    seed: int


def render_language_scene(scene: LanguageScene, *, image_size: int = 32) -> tuple[t.Tensor, t.Tensor]:
    """Render a scene and exact occupied pixels; matched donors share sensor noise."""
    if scene.color not in COLORS or scene.shape not in SHAPES or scene.style not in ("solid", "muted"):
        raise ValueError("Unknown controlled scene attribute")
    x1, y1, x2, y2 = scene.bbox
    if not (0 <= x1 < x2 <= image_size and 0 <= y1 < y2 <= image_size):
        raise ValueError("Scene bounding box must fit inside the image")
    channels = RGB[COLORS.index(scene.color)]
    rgb = tuple(int(max(0, min(255, channel + jitter))) for channel, jitter in zip(channels, scene.rgb_jitter, strict=True))
    if scene.style == "muted":
        rgb = tuple(round(.75 * channel + .25 * 170) for channel in rgb)
    picture = Image.new("RGB", (image_size, image_size), (scene.background,) * 3)
    mask_image = Image.new("L", (image_size, image_size), 0)
    for image, fill in ((picture, rgb), (mask_image, 1)):
        draw = ImageDraw.Draw(image)
        box = (x1, y1, x2 - 1, y2 - 1)
        if scene.shape == "square":
            draw.rectangle(box, fill=fill)
        elif scene.shape == "circle":
            draw.ellipse(box, fill=fill)
        else:
            draw.polygon(((scene.cx, y1), (x1, y2 - 1), (x2 - 1, y2 - 1)), fill=fill)
    pixels = t.from_numpy(np.asarray(picture).copy()).permute(2, 0, 1).float() / 255
    generator = t.Generator().manual_seed(scene.noise_seed)
    noise = .004 * t.randn(pixels.shape, generator=generator)
    mask = t.from_numpy(np.asarray(mask_image).copy()).bool()
    return (pixels + noise).clamp(0, 1), mask


def make_language_dataset(split: str, *, examples_per_combination: int = 32) -> LanguageDataset:
    """Separate renderer draws; three combinations are never used during fitting.

    Train/development use nine familiar combinations and solid colors. Final
    familiar/compositional splits use new instances and a muted palette.
    """
    if split not in SPLIT_SEEDS or examples_per_combination < 1:
        raise ValueError("Choose a declared split and positive instance count")
    generator = t.Generator().manual_seed(SPLIT_SEEDS[split])
    scenes, images, masks = [], [], []
    for color in COLORS:
        for shape in SHAPES:
            heldout = (color, shape) in HELDOUT_COMBINATIONS
            if heldout != (split == "compositional"):
                continue
            for instance in range(examples_per_combination):
                radius = int(t.randint(5, 8, (), generator=generator))
                scene = LanguageScene(
                    scene_id=f"{split}/{color}/{shape}/{instance}", color=color, shape=shape,
                    cx=int(t.randint(radius + 1, 31 - radius, (), generator=generator)),
                    cy=int(t.randint(radius + 1, 31 - radius, (), generator=generator)),
                    radius=radius, background=int(t.randint(215, 256, (), generator=generator)),
                    rgb_jitter=tuple(t.randint(-12, 13, (3,), generator=generator).tolist()),
                    noise_seed=int(t.randint(0, 2**31 - 1, (), generator=generator)),
                    style="solid" if split in ("train", "development") else "muted",
                )
                image, mask = render_language_scene(scene)
                scenes.append(scene)
                images.append(image)
                masks.append(mask)
    return LanguageDataset(tuple(scenes), t.stack(images), t.stack(masks), split, SPLIT_SEEDS[split])


def counterfactual_scene(scene: LanguageScene, *, color: str | None = None,
                         shape: str | None = None) -> LanguageScene:
    """Change semantic attributes while preserving geometry, style and sensor noise."""
    return replace(scene, color=scene.color if color is None else color,
                   shape=scene.shape if shape is None else shape, scene_id=scene.scene_id + "/counterfactual")


def answer_words(scene: LanguageScene, task: str) -> list[str]:
    if task == "color":
        return [scene.color]
    if task == "shape":
        return [scene.shape]
    if task == "caption":
        return ["a", scene.color, scene.shape, "."]
    raise ValueError("Choose color, shape or caption")


def tokenize_language_examples(scenes, tasks, *, include_answers: bool = True,
                               omit_question: bool = False) -> dict[str, t.Tensor]:
    """Right-pad explicit BOS/question/answer-marker/answer/EOS word sequences.

    target_mask marks answer words and EOS in the unshifted sequence. The
    student's teacher-forced loss must shift predictions and this mask itself.
    """
    if len(scenes) != len(tasks) or not scenes:
        raise ValueError("One declared task is required for each scene")
    sequences, masks, prompt_lengths = [], [], []
    for scene, task in zip(scenes, tasks, strict=True):
        prompt = ["<bos>", *([] if omit_question else PROMPTS[task].split()), "<answer>"]
        answer = [*answer_words(scene, task), "<eos>"] if include_answers else []
        sequences.append([VOCAB[word] for word in prompt + answer])
        masks.append([False] * len(prompt) + [True] * len(answer))
        prompt_lengths.append(len(prompt))
    length = max(map(len, sequences))
    ids = t.full((len(sequences), length), VOCAB["<pad>"], dtype=t.long)
    valid, targets = t.zeros_like(ids, dtype=t.bool), t.zeros_like(ids, dtype=t.bool)
    for index, (sequence, target) in enumerate(zip(sequences, masks, strict=True)):
        ids[index, :len(sequence)] = t.tensor(sequence)
        valid[index, :len(sequence)] = True
        targets[index, :len(sequence)] = t.tensor(target)
    return {"token_ids": ids, "valid_mask": valid, "target_mask": targets,
            "prompt_lengths": t.tensor(prompt_lengths)}


def decode_answer(ids) -> str:
    words = []
    for token in ids:
        word = WORDS[int(token)]
        if word == "<eos>":
            break
        words.append(word)
    return " ".join(words)

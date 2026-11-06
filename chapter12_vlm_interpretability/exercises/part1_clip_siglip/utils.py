"""Plotting, rendering and model-loading helpers for [12.1] CLIP and SigLIP from Scratch.

Nothing in here is part of the method you're learning: it's the code that draws coloured shapes,
loads the pretrained checkpoints and makes pictures. All of the losses, encoders, training loops
and analyses live in the notebook.
"""

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from PIL import Image, ImageDraw

SECTION_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------------------------------------------------
# Drawing coloured shapes
# ---------------------------------------------------------------------------------------------------------------------

COLOUR_RGB = {
    "red": (220, 40, 40),
    "green": (40, 160, 60),
    "blue": (50, 80, 220),
    "yellow": (240, 210, 40),
    "purple": (140, 60, 190),
    "orange": (245, 140, 30),
}

QUADRANT_CENTRES = {"top left": (8, 8), "top right": (24, 8), "bottom left": (8, 24), "bottom right": (24, 24)}


def _polygon(shape: str, cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    if shape == "triangle":
        return [(cx, cy - r), (cx - 0.95 * r, cy + 0.8 * r), (cx + 0.95 * r, cy + 0.8 * r)]
    if shape == "star":
        points = []
        for i in range(10):
            radius = r if i % 2 == 0 else 0.42 * r
            angle = -math.pi / 2 + i * math.pi / 5
            points.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        return points
    if shape == "heart":
        points = []
        for k in range(60):
            a = 2 * math.pi * k / 60
            x = 16 * math.sin(a) ** 3
            y = -(13 * math.cos(a) - 5 * math.cos(2 * a) - 2 * math.cos(3 * a) - math.cos(4 * a))
            points.append((cx + x * r / 17, cy + y * r / 17))
        return points
    raise ValueError(f"Unknown polygon shape {shape!r}")


def shape_mask(shape: str, size: int, cx: float, cy: float, r: float, supersample: int = 4) -> np.ndarray:
    """Anti-aliased [size, size] mask in [0, 1] for a shape of 'radius' r centred at (cx, cy)."""
    big = size * supersample
    mask = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(mask)
    cxb, cyb, rb = cx * supersample, cy * supersample, r * supersample
    if shape == "circle":
        draw.ellipse([cxb - rb, cyb - rb, cxb + rb, cyb + rb], fill=255)
    elif shape == "square":
        draw.rectangle([cxb - 0.88 * rb, cyb - 0.88 * rb, cxb + 0.88 * rb, cyb + 0.88 * rb], fill=255)
    else:
        draw.polygon(_polygon(shape, cxb, cyb, rb), fill=255)
    return np.asarray(mask.resize((size, size), Image.LANCZOS), dtype=np.float32) / 255.0


def _paint(canvas: np.ndarray, mask: np.ndarray, colour: np.ndarray) -> np.ndarray:
    return canvas * (1 - mask[..., None]) + colour * mask[..., None]


def render_tiny(shape: str, colour: str, position: str, rng: np.random.Generator) -> np.ndarray:
    """A 32x32 uint8 RGB image of one coloured shape in one quadrant, with random jitter, size, shade and noise."""
    cx, cy = QUADRANT_CENTRES[position]
    mask = shape_mask(shape, 32, cx + rng.uniform(-2, 2), cy + rng.uniform(-2, 2), rng.uniform(5.0, 7.0))
    canvas = np.full((32, 32, 3), rng.uniform(215, 255), dtype=np.float32)
    rgb = np.array(COLOUR_RGB[colour], dtype=np.float32) + rng.uniform(-20, 20, 3)
    img = _paint(canvas, mask, rgb) + rng.normal(0, 2.0, (32, 32, 3))
    return np.clip(img, 0, 255).astype(np.uint8)


def render_tiny_pair(
    left: tuple[str, str], right: tuple[str, str], rng: np.random.Generator
) -> np.ndarray:
    """A 32x32 uint8 image with two shapes side by side. `left` and `right` are (shape, colour) pairs."""
    canvas = np.full((32, 32, 3), rng.uniform(215, 255), dtype=np.float32)
    for (shape, colour), cx in [(left, 9.0), (right, 23.0)]:
        mask = shape_mask(shape, 32, cx + rng.uniform(-1.5, 1.5), 16 + rng.uniform(-2, 2), rng.uniform(4.5, 6.0))
        canvas = _paint(canvas, mask, np.array(COLOUR_RGB[colour], dtype=np.float32) + rng.uniform(-20, 20, 3))
    return np.clip(canvas + rng.normal(0, 2.0, canvas.shape), 0, 255).astype(np.uint8)


def render_large(
    shape: str, colour: str, rng: np.random.Generator, pattern: str = "plain", size: int = 224
) -> np.ndarray:
    """A 224x224 uint8 RGB image of one coloured shape (pattern: 'plain', 'striped' or 'dotted') for the real models."""
    r = rng.uniform(0.22, 0.34) * size
    cx = size / 2 + rng.uniform(-0.12, 0.12) * size
    cy = size / 2 + rng.uniform(-0.12, 0.12) * size
    mask = shape_mask(shape, size, cx, cy, r)
    fill = np.broadcast_to(np.array(COLOUR_RGB[colour], dtype=np.float32) + rng.uniform(-15, 15, 3), (size, size, 3)).copy()
    yy, xx = np.mgrid[0:size, 0:size]
    if pattern == "striped":
        off = (((xx + yy) / (size / 14)) % 2) >= 1
        fill[off] = 250
    elif pattern == "dotted":
        period = size / 9
        fx, fy = (xx % period) - period / 2, (yy % period) - period / 2
        fill[(fx**2 + fy**2) >= (0.28 * period) ** 2] = 250
    elif pattern != "plain":
        raise ValueError(f"Unknown pattern {pattern!r}")
    canvas = np.full((size, size, 3), rng.uniform(120, 190), dtype=np.float32)  # a mid-grey background keeps the outline of striped shapes visible
    return np.clip(_paint(canvas, mask, fill), 0, 255).astype(np.uint8)


def render_large_pair(
    left: tuple[str, str], right: tuple[str, str], rng: np.random.Generator, size: int = 224
) -> np.ndarray:
    """A 224x224 uint8 image with two plain shapes side by side. `left` and `right` are (shape, colour) pairs."""
    canvas = np.full((size, size, 3), rng.uniform(235, 255), dtype=np.float32)
    for (shape, colour), cx in [(left, 0.27 * size), (right, 0.73 * size)]:
        mask = shape_mask(shape, size, cx, size / 2 + rng.uniform(-0.05, 0.05) * size, rng.uniform(0.15, 0.2) * size)
        canvas = _paint(canvas, mask, np.array(COLOUR_RGB[colour], dtype=np.float32) + rng.uniform(-10, 10, 3))
    return np.clip(canvas, 0, 255).astype(np.uint8)


def to_pil(images: np.ndarray | list[np.ndarray]) -> list[Image.Image]:
    """Convert a stack of uint8 HxWx3 arrays into a list of PIL images."""
    return [Image.fromarray(np.asarray(x)) for x in images]


# ---------------------------------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------------------------------

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_cifar10_sample() -> tuple[np.ndarray, np.ndarray]:
    """Load the byte-preserved 200-image CIFAR-10 fixture (uint8 RGB, int64 labels).

    Krizhevsky (2009). The historical helper described 20 test images per class;
    original dataset indices and selection seed were not recorded. Manifest row
    indices preserve the original NPZ ordering, not claimed CIFAR dataset IDs.
    """
    import hashlib
    import json

    fixture = SECTION_DIR / "fixtures" / "cifar10_sample"
    manifest = json.loads((fixture / "manifest.json").read_text())

    def require(condition: bool, message: str) -> None:
        if not condition:
            raise ValueError(f"Invalid CIFAR-10 fixture: {message}")

    def digest(data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    require(manifest["schema"] == "arena.cifar10-png-fixture.v1", "schema")
    require(manifest["images"]["shape"] == [200, 32, 32, 3]
            and manifest["images"]["dtype"] == "|u1", "image shape/dtype")
    require(manifest["labels"]["shape"] == [200]
            and manifest["labels"]["dtype"] == "<i8", "label shape/dtype")
    require(len(manifest["records"]) == 200, "record count")
    images = np.empty((200, 32, 32, 3), dtype=np.uint8)
    labels = np.empty(200, dtype="<i8")
    for index, record in enumerate(manifest["records"]):
        require(record["fixture_index"] == index
                and record["file"] == f"{index:03d}.png", "row order/path")
        path = fixture / record["file"]
        require(not path.is_symlink(), "image symlink")
        require(digest(path.read_bytes()) == record["file_sha256"], "PNG hash")
        with Image.open(path) as picture:
            require(picture.format == "PNG" and picture.mode == "RGB"
                    and picture.size == (32, 32), "PNG format/mode/shape")
            pixels = np.asarray(picture)
            require(pixels.dtype == np.uint8
                    and digest(pixels.tobytes()) == record["pixels_sha256"], "pixel hash")
            images[index] = pixels
        label = record["label"]
        require(type(label) is int and 0 <= label <= 9, "label")
        labels[index] = label
        require(digest(labels[index:index + 1].tobytes()) == record["label_sha256"], "label hash")
    require(digest(images.tobytes()) == manifest["images"]["bytes_sha256"], "image array hash")
    require(digest(labels.tobytes()) == manifest["labels"]["bytes_sha256"], "label array hash")
    return images, labels


# ---------------------------------------------------------------------------------------------------------------------
# Pretrained models
# ---------------------------------------------------------------------------------------------------------------------

PRETRAINED_NAMES = {"CLIP": "openai/clip-vit-base-patch32", "SigLIP": "google/siglip-base-patch16-224"}


@dataclass
class PretrainedPair:
    """
    A pretrained image/text encoder pair behind a minimal interface.

    `embed_images` and `embed_texts` return the *raw* (not yet L2-normalised) joint-space embeddings, on the CPU.
    `logit_scale` is the learned multiplier (CLIP: 100.0; SigLIP: about 117) and `logit_bias` the learned
    bias (None for CLIP, which has none).
    """

    name: str
    model: object
    processor: object
    device: t.device
    logit_scale: float
    logit_bias: float | None
    text_max_length: int | None

    @t.no_grad()
    def embed_images(self, images: list[Image.Image], batch_size: int = 16) -> t.Tensor:
        out = []
        for i in range(0, len(images), batch_size):
            pixels = self.processor(images=images[i : i + batch_size], return_tensors="pt")["pixel_values"]
            out.append(self.model.get_image_features(pixel_values=pixels.to(self.device)).pooler_output.float().cpu())
        return t.cat(out)

    @t.no_grad()
    def embed_texts(self, texts: list[str], batch_size: int = 128) -> t.Tensor:
        out = []
        for i in range(0, len(texts), batch_size):
            kwargs = {"padding": "max_length", "max_length": self.text_max_length} if self.text_max_length else {"padding": True}
            tokens = self.processor(text=texts[i : i + batch_size], return_tensors="pt", **kwargs).to(self.device)
            out.append(self.model.get_text_features(**tokens).pooler_output.float().cpu())
        return t.cat(out)

    @t.no_grad()
    def official_logits(self, images: list[Image.Image], texts: list[str]) -> t.Tensor:
        """The checkpoint's own `logits_per_image` (scale and bias already applied), for checking our code against."""
        kwargs = {"padding": "max_length", "max_length": self.text_max_length} if self.text_max_length else {"padding": True}
        inputs = self.processor(images=images, text=texts, return_tensors="pt", **kwargs).to(self.device)
        return self.model(**inputs).logits_per_image.float().cpu()


def load_pretrained(kind: str, device: t.device | str = "cuda") -> PretrainedPair:
    """Load 'CLIP' (openai/clip-vit-base-patch32) or 'SigLIP' (google/siglip-base-patch16-224) in eval mode."""
    import transformers
    from transformers import CLIPModel, CLIPProcessor, SiglipModel, SiglipProcessor

    transformers.logging.set_verbosity_error()
    name = PRETRAINED_NAMES[kind]
    model_cls, processor_cls = (CLIPModel, CLIPProcessor) if kind == "CLIP" else (SiglipModel, SiglipProcessor)
    device = t.device(device)
    model = model_cls.from_pretrained(name).to(device).eval().requires_grad_(False)
    processor = processor_cls.from_pretrained(name)
    bias = model.logit_bias.item() if hasattr(model, "logit_bias") else None
    return PretrainedPair(
        name=name,
        model=model,
        processor=processor,
        device=device,
        logit_scale=model.logit_scale.exp().item(),
        logit_bias=bias,
        text_max_length=64 if kind == "SigLIP" else None,
    )


def free_memory(*objects) -> None:
    """Drop references to big models and empty the CUDA cache (call as `free_memory(pair)` then `del pair`)."""
    import gc

    gc.collect()
    if t.cuda.is_available():
        t.cuda.empty_cache()


# ---------------------------------------------------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------------------------------------------------


def show_image_grid(images: np.ndarray, titles: list[str], ncols: int = 8, scale: float = 1.6, title: str | None = None):
    """Show a grid of uint8 images with a title above each."""
    nrows = math.ceil(len(images) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(scale * ncols, scale * nrows * 1.1), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, img, ttl in zip(axes.flat, images, titles):
        ax.imshow(img, interpolation="nearest")
        ax.set_title(ttl, fontsize=7)
    if title:
        fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_loss_vs_scale(scales: np.ndarray, curves: dict[str, np.ndarray], n: int, title: str):
    """InfoNCE loss as a function of the logit scale for several embedding conditions, with the log(N) line."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for (name, values), style in zip(curves.items(), ["-o", "--s", ":^"]):
        ax.plot(scales, values, style, label=name, markersize=4)
    ax.axhline(math.log(n), color="gray", ls="-.", lw=1, label=f"log N = {math.log(n):.2f}")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1.0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("logit scale (1 / temperature)")
    ax.set_ylabel("symmetric InfoNCE loss (linear below 1, log above)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_siglip_init_terms(batch_sizes: list[int], terms: dict[str, tuple[list[float], list[float]]], title: str):
    """Positive and negative terms of the sigmoid loss at initialisation, for several (scale, bias) initialisations."""
    fig, axes = plt.subplots(1, len(terms), figsize=(5.2 * len(terms), 4), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (name, (pos, neg)) in zip(axes, terms.items()):
        ax.plot(batch_sizes, pos, "-o", label="positive pairs (pull together)", markersize=4)
        ax.plot(batch_sizes, neg, "--s", label="negative pairs (push apart)", markersize=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("batch size N")
        ax.set_title(name, fontsize=10)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("loss contribution at init")
    axes[0].legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_retrieval_heatmaps(
    sims: dict[str, np.ndarray],
    heldout_mask: np.ndarray,
    metrics: dict[str, str],
    block: int = 4,
    title: str = "",
):
    """
    Side-by-side image x caption similarity heatmaps on a shared colour scale.

    sims[name] is an [n, n] array (row i = test image of class i, column j = caption of class j). Rows of classes
    that were held out of training get a red tick mark on the left axis. `metrics[name]` is shown in each subtitle.
    """
    fig, axes = plt.subplots(1, len(sims), figsize=(5.6 * len(sims) + 0.8, 5.6), squeeze=False)
    vmin = min(float(s.min()) for s in sims.values())
    vmax = max(float(s.max()) for s in sims.values())
    held = np.nonzero(heldout_mask)[0]
    for ax, (name, sim) in zip(axes[0], sims.items()):
        im = ax.imshow(sim, cmap="viridis", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"{name}\n{metrics[name]}", fontsize=8)
        ax.set_xlabel("caption (class index)")
        ax.set_xticks(range(0, sim.shape[1], 20))
        ax.set_ylabel("test image (class index)")
        ax.set_yticks(range(0, sim.shape[0], 20))
        if len(held):
            ax.set_yticks(held, minor=True)
            ax.tick_params(axis="y", which="minor", colors="red", length=5, width=1.2)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="cosine similarity")
    suffix = "  (red ticks on the left axis mark classes held out of training)" if len(held) else ""
    fig.suptitle(title + suffix, fontsize=10)
    plt.show()


def plot_training_curves(curves: dict[str, tuple[list[int], list[float]]], title: str, ylabel: str, chance: float | None = None):
    """Line plot of (steps, values) per run, with an optional chance line."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    styles = ["-", "-", "--", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]
    for (name, (steps, values)), style in zip(curves.items(), styles):
        ax.plot(steps, values, linestyle=style, label=name, lw=1.8)
    if chance is not None:
        ax.axhline(chance, color="gray", ls="-.", lw=1, label=f"chance ({chance:.3f})")
    ax.set_xlabel("training step")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_class_heatmaps(
    mats: dict[str, np.ndarray], row_labels: list[str], col_labels: list[str], title: str, ncols: int = 3, cbar_label: str = "mean cosine similarity"
):
    """Grid of small class x caption heatmaps (e.g. CIFAR-10 class means) on one shared colour scale."""
    nrows = math.ceil(len(mats) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols + 1.5, 4.4 * nrows), squeeze=False, constrained_layout=True)
    vmin = min(float(m.min()) for m in mats.values())
    vmax = max(float(m.max()) for m in mats.values())
    for ax in axes.flat:
        ax.axis("off")
    for k, (ax, (name, mat)) in enumerate(zip(axes.flat, mats.items())):
        ax.axis("on")
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(name, fontsize=9)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels if k % ncols == 0 else [], fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label=cbar_label)
    fig.suptitle(title, fontsize=11)
    plt.show()


def plot_bars(groups: dict[str, dict[str, float]], title: str, ylabel: str, chance: float | None = None, log: bool = False):
    """Grouped bars: groups[condition][series] = value. One cluster per condition."""
    conditions = list(groups)
    series = list(next(iter(groups.values())))
    width = 0.8 / len(series)
    fig, ax = plt.subplots(figsize=(max(6.5, 1.8 * len(conditions) + 2), 4.2))
    for k, s in enumerate(series):
        xs = np.arange(len(conditions)) + (k - (len(series) - 1) / 2) * width
        vals = [groups[c][s] for c in conditions]
        bars = ax.bar(xs, vals, width, label=s)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.3g}", ha="center", va="bottom", fontsize=7)
    if chance is not None:
        ax.axhline(chance, color="gray", ls="-.", lw=1, label=f"chance ({chance:.2f})")
    if log:
        ax.set_yscale("log")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_modality_gap(
    panels: dict[str, tuple[np.ndarray, np.ndarray]], title: str
):
    """
    Scatter plots of image and text embeddings. The horizontal axis is the position along the modality-gap direction
    (the unit vector from the centre of the captions to the centre of the images); the vertical axis is the first
    principal component of what is left once that direction is removed.
    panels[name] = (image_embeddings, text_embeddings), both [n, d] arrays (already L2-normalised).
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.6), squeeze=False)
    for ax, (name, (img, txt)) in zip(axes[0], panels.items()):
        both = np.concatenate([img, txt])
        mean = both.mean(0)
        gap = img.mean(0) - txt.mean(0)
        gap_dir = gap / max(np.linalg.norm(gap), 1e-12)
        centred = both - mean
        residual = centred - np.outer(centred @ gap_dir, gap_dir)
        _, _, vt = np.linalg.svd(residual, full_matrices=False)
        project = lambda x: np.stack([(x - mean) @ gap_dir, (x - mean) @ vt[0]], axis=1)
        pi, pt = project(img), project(txt)
        ax.scatter(*pi.T, s=8, alpha=0.5, label="images")
        ax.scatter(*pt.T, s=8, alpha=0.5, label="captions")
        ax.scatter(*pi.mean(0), marker="X", s=120, color="C0", edgecolor="k", zorder=5)
        ax.scatter(*pt.mean(0), marker="X", s=120, color="C1", edgecolor="k", zorder=5)
        ax.set_title(f"{name}\n(gap norm {np.linalg.norm(gap):.2f})", fontsize=9)
        ax.set_xlabel("position along the modality-gap direction")
        ax.set_ylabel("first principal component orthogonal to it")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_steering_curves(
    panels: dict[str, dict[str, tuple[list[float], list[float]]]], title: str, ylabel: str
):
    """One panel per model; each panel has one (alphas, success) line per direction."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.6 * len(panels), 4.2), sharey=True, squeeze=False)
    styles = {"image direction": "-o", "caption direction": "-s", "held-out shapes": "--^", "random direction (same norm)": ":x", "shape direction (wrong attribute)": "-.d"}
    for ax, (name, curves) in zip(axes[0], panels.items()):
        for label, (alphas, vals) in curves.items():
            ax.plot(alphas, vals, styles.get(label, "-o"), label=label, markersize=4)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("steering strength alpha")
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.03, 1.03)
    axes[0][0].set_ylabel(ylabel)
    axes[0][0].legend(fontsize=7)
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_direction_agreement(agreement: dict[str, dict[str, dict[str, list[float]]]], title: str):
    """
    Strip plots of cosine similarities. agreement[model][attribute][series] is a list of cosines, one per attribute pair;
    one panel per model, one cluster per attribute, one colour per series.
    """
    models = list(agreement)
    attributes = list(next(iter(agreement.values())))
    series = list(next(iter(next(iter(agreement.values())).values())))
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, len(models), figsize=(5.8 * len(models), 4.4), sharey=True, squeeze=False)
    for ax, model in zip(axes[0], models):
        for k, name in enumerate(series):
            for j, attr in enumerate(attributes):
                vals = np.array(agreement[model][attr][name])
                xs = j + (k - (len(series) - 1) / 2) * 0.27 + rng.uniform(-0.05, 0.05, len(vals))
                ax.scatter(xs, vals, s=22, color=f"C{k}", alpha=0.8, label=name if j == 0 else None)
                ax.hlines(vals.mean(), j + (k - 1) * 0.27 - 0.1, j + (k - 1) * 0.27 + 0.1, color=f"C{k}", lw=2)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_xticks(range(len(attributes)))
        ax.set_xticklabels([f"{a} pairs ({len(agreement[model][a][series[0]])})" for a in attributes])
        ax.set_title(model, fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0][0].set_ylabel("cosine(image-derived direction, caption-derived direction)")
    axes[0][0].legend(fontsize=7, loc="upper right")
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()

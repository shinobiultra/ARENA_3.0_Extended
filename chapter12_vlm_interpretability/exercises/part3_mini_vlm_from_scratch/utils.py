"""Data, loading and plotting helpers for [12.3] Mini VLM from Scratch.

Nothing in here is part of the method you're learning: it's the scene renderer (the "world" we ask
questions about), model-loading helpers, and plotting code. All of the VLM logic (projector, input
assembly, training, lenses, attention analysis, patching) lives in the notebook.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Rectangle
from PIL import Image, ImageDraw

# ---------------------------------------------------------------------------------------------
# The world: scenes of coloured shapes
# ---------------------------------------------------------------------------------------------

IMAGE_SIZE = 224  # SigLIP-base-patch16-224 takes 224x224 images
PATCH_SIZE = 16  # ... and cuts them into 16x16 pixel patches
PATCH_GRID = IMAGE_SIZE // PATCH_SIZE  # 14 x 14 = 196 patches
MERGE = 2  # we will merge 2x2 blocks of patches into one image token
TOKEN_GRID = PATCH_GRID // MERGE  # 7 x 7 = 49 image tokens
N_IMAGE_TOKENS = TOKEN_GRID**2
TOKEN_PIXELS = IMAGE_SIZE // TOKEN_GRID  # each image token covers 32 x 32 pixels

COLOURS = {
    "red": (215, 40, 40),
    "green": (40, 160, 70),
    "blue": (50, 80, 220),
    "yellow": (240, 210, 40),
    "purple": (150, 60, 190),
    "orange": (245, 135, 30),
}
SHAPES = ["circle", "square", "triangle"]
QUADRANTS = [(0, 0), (0, 1), (1, 0), (1, 1)]  # (row, col); row 0 is the top, col 0 is the left


@dataclass
class SceneObject:
    """One coloured shape. `quadrant` is (row, col) in a 2x2 layout; dx, dy jitter it; size is the bounding box side."""

    colour: str
    shape: str
    quadrant: tuple[int, int]
    dx: int = 0
    dy: int = 0
    size: int = 60

    @property
    def centre(self) -> tuple[int, int]:
        row, col = self.quadrant
        return (col * IMAGE_SIZE // 2 + IMAGE_SIZE // 4 + self.dx, row * IMAGE_SIZE // 2 + IMAGE_SIZE // 4 + self.dy)


@dataclass
class Scene:
    """A list of objects on a grey background. Objects always have distinct shapes, colours and quadrants."""

    objects: list[SceneObject] = field(default_factory=list)
    background: int = 235
    noise_seed: int = 0


def sample_scene(rng: np.random.Generator, n_objects: int | None = None) -> Scene:
    """A random scene with 1-3 objects (probabilities 0.25 / 0.4 / 0.35), distinct shapes, colours and quadrants."""
    if n_objects is None:
        n_objects = int(rng.choice([1, 2, 3], p=[0.25, 0.4, 0.35]))
    shapes = rng.permutation(SHAPES)[:n_objects]
    colours = rng.permutation(list(COLOURS))[:n_objects]
    quadrants = [QUADRANTS[i] for i in rng.permutation(4)[:n_objects]]
    objects = [
        SceneObject(
            str(colour), str(shape), quadrant, int(rng.integers(-12, 13)), int(rng.integers(-12, 13)), int(rng.integers(52, 69))
        )
        for colour, shape, quadrant in zip(colours, shapes, quadrants)
    ]
    return Scene(objects, background=int(rng.integers(215, 250)), noise_seed=int(rng.integers(1 << 30)))


def _draw_shape(draw: ImageDraw.ImageDraw, obj: SceneObject, fill, scale: int = 1) -> None:
    cx, cy = obj.centre
    half = obj.size // 2
    if obj.shape == "circle":
        draw.ellipse([(cx - half) * scale, (cy - half) * scale, (cx + half) * scale, (cy + half) * scale], fill=fill)
    elif obj.shape == "square":
        draw.rectangle([(cx - half) * scale, (cy - half) * scale, (cx + half) * scale, (cy + half) * scale], fill=fill)
    else:
        draw.polygon(
            [(cx * scale, (cy - half) * scale), ((cx - half) * scale, (cy + half) * scale), ((cx + half) * scale, (cy + half) * scale)],
            fill=fill,
        )


def render_scene(scene: Scene) -> np.ndarray:
    """Render a scene to a (224, 224, 3) uint8 array (anti-aliased, with a little pixel noise)."""
    scale = 2
    image = Image.new("RGB", (IMAGE_SIZE * scale,) * 2, (scene.background,) * 3)
    draw = ImageDraw.Draw(image)
    for obj in scene.objects:
        _draw_shape(draw, obj, COLOURS[obj.colour], scale)
    array = np.asarray(image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)).astype(np.float32)
    array = array + np.random.default_rng(scene.noise_seed).normal(0, 3.0, array.shape)
    return np.clip(array, 0, 255).astype(np.uint8)


def render_scenes(scenes: list[Scene]) -> np.ndarray:
    """Render a list of scenes to a (n, 224, 224, 3) uint8 array."""
    return np.stack([render_scene(scene) for scene in scenes])


def noise_images(n: int, seed: int = 0) -> np.ndarray:
    """(n, 224, 224, 3) uint8 images of independent uniform pixel noise."""
    return np.random.default_rng(seed).integers(0, 256, (n, IMAGE_SIZE, IMAGE_SIZE, 3)).astype(np.uint8)


def object_token_mask(obj: SceneObject, min_fraction: float = 0.2) -> np.ndarray:
    """
    Boolean (7, 7) mask of the image tokens that overlap the object, i.e. tokens whose 32x32 pixel block
    is at least `min_fraction` covered by the object.
    """
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    _draw_shape(ImageDraw.Draw(image), obj, 255)
    covered = (np.asarray(image) > 127).reshape(TOKEN_GRID, TOKEN_PIXELS, TOKEN_GRID, TOKEN_PIXELS).mean(axis=(1, 3))
    return covered >= min_fraction


def recolour_object(scene: Scene, index: int, colour: str) -> Scene:
    """A copy of the scene in which only object `index` has a different colour (same pixels noise, same geometry)."""
    new_scene = copy.deepcopy(scene)
    new_scene.objects[index].colour = colour
    return new_scene


def make_counterfactual_pairs(n: int, seed: int) -> list[tuple[Scene, Scene, int]]:
    """
    `n` triples (clean scene, corrupted scene, index): the corrupted scene is the clean scene with object `index`
    recoloured to a colour that no object in the scene has. Everything else (geometry, noise) is identical.
    """
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        scene = sample_scene(rng)
        index = int(rng.integers(len(scene.objects)))
        unused = [c for c in COLOURS if c not in {o.colour for o in scene.objects}]
        pairs.append((scene, recolour_object(scene, index, str(rng.choice(unused))), index))
    return pairs


# ---------------------------------------------------------------------------------------------
# Model loading and tokenisation plumbing
# ---------------------------------------------------------------------------------------------


def load_siglip(device: t.device | str):
    """The frozen vision tower of SigLIP-base (ViT-B/16, 224px), in fp16 and eval mode."""
    from transformers import SiglipVisionModel

    model = SiglipVisionModel.from_pretrained("google/siglip-base-patch16-224", revision="7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed", dtype=t.float16).to(device).eval()
    model.requires_grad_(False)
    return model


def encode(model, text: str) -> list[int]:
    """GPT-2 token ids of `text`, without a BOS token (note: `model.tokenizer.encode` would prepend one)."""
    return model.to_tokens(text, prepend_bos=False)[0].tolist()


def tokenize_prompts(model, prompts: list[str]) -> tuple[t.Tensor, t.Tensor]:
    """
    Tokenise prompts with GPT-2's tokenizer (no BOS), right-padded with the end-of-text token.
    Returns (tokens, lengths) with shapes (batch, max_len) and (batch,), on the model's device.
    """
    pad = model.tokenizer.eos_token_id
    ids = [encode(model, p) for p in prompts]
    lengths = t.tensor([len(i) for i in ids])
    tokens = t.full((len(ids), int(lengths.max())), pad, dtype=t.long)
    for row, i in enumerate(ids):
        tokens[row, : len(i)] = t.tensor(i)
    return tokens.to(model.cfg.device), lengths.to(model.cfg.device)


def load_qwen(device: t.device | str):
    """Qwen3-VL-2B-Instruct (bf16, eval mode) and its processor."""
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", revision="89644892e4d85e24eaac8bacfd4f463576704203")
    model = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", revision="89644892e4d85e24eaac8bacfd4f463576704203", dtype=t.bfloat16).to(device).eval()
    model.requires_grad_(False)
    return model, processor


def qwen_inputs(processor, image: np.ndarray, question: str, device: t.device | str):
    """
    Chat-template a single (image, question) pair for Qwen3-VL. The 224px scene is upsampled to 256x256, which
    Qwen turns into an 8x8 grid of image tokens (each covering a 32x32 pixel block of the 256px image).
    """
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    picture = Image.fromarray(image).resize((256, 256), Image.BICUBIC)
    return processor(text=[text], images=[picture], return_tensors="pt").to(device)


def qwen_object_token_mask(obj: SceneObject, min_fraction: float = 0.2) -> np.ndarray:
    """Boolean (8, 8) mask of the Qwen image tokens (32x32 blocks of the 256px image) that overlap the object."""
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)
    _draw_shape(ImageDraw.Draw(image), obj, 255)
    resized = np.asarray(image.resize((256, 256), Image.BILINEAR)).astype(np.float32) / 255
    return resized.reshape(8, 32, 8, 32).mean(axis=(1, 3)) >= min_fraction


def qwen_cache_layer_inputs(model, inputs) -> dict[int, t.Tensor]:
    """
    Run Qwen once and return {layer: hidden state going INTO that decoder layer}, i.e. the residual stream
    before the layer, each of shape (1, seq, 2048). (This is Qwen's equivalent of `resid_pre`.)
    """
    layers = model.model.language_model.layers
    cache: dict[int, t.Tensor] = {}
    handles = []
    for index, layer in enumerate(layers):

        def hook(module, args, kwargs, index=index):
            hidden = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
            cache[index] = hidden.detach().clone()

        handles.append(layer.register_forward_pre_hook(hook, with_kwargs=True))
    try:
        with t.no_grad():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return cache


# ---------------------------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------------------------

BLUE, ORANGE, GREEN, RED, PURPLE, GREY = "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f"


def plot_scene_gallery(images: np.ndarray, captions: list[str], ncols: int = 4, title: str | None = None) -> None:
    """A grid of scenes with a caption under each."""
    nrows = (len(images) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.1 * ncols, 3.7 * nrows), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, image, caption in zip(axes.flat, images, captions):
        ax.imshow(image)
        ax.set_title(caption, fontsize=8)
    if title:
        fig.suptitle(title)
    plt.tight_layout(h_pad=1.5)
    plt.show()


def plot_patch_pca(images: np.ndarray, features: t.Tensor, reference_features: t.Tensor | None = None) -> None:
    """
    With two arguments, fit raw PCA jointly on the displayed scenes and show two rows.
    An explicit reference_features argument retains the three-row reference-centred display:
    Colour every 16x16 patch of each scene by the top-3 principal components of its SigLIP embedding, in two ways.
    Row 2: PCA of the raw embeddings. Row 3: PCA after subtracting, from every patch, the average embedding that
    patch *position* has over the reference scenes (which removes what depends on where the patch is, leaving what
    depends on what is in it). Each PCA is fit on `reference_features` (n, 196, 768), then applied to the scenes shown.
    """
    if reference_features is None:
        # The published two-argument lesson fits RAW PCA jointly on the four
        # displayed scenes. No extra training/reference cohort is introduced.
        if not isinstance(features, t.Tensor):
            raise TypeError("Patch features must be a torch Tensor")
        if features.ndim != 3 or len(images) != len(features) or len(features) == 0:
            raise ValueError("Need aligned nonempty images and (scene, patch, d) features")
        if features.shape[1] != PATCH_GRID**2 or features.shape[2] < 3:
            raise ValueError("Need 196 patches and at least three feature dimensions")
        shown = features.float().cpu()
        flat = shown.reshape(-1, shown.shape[-1])
        centre = flat.mean(0)
        _, _, v = t.pca_lowrank(flat - centre, q=3, center=False)
        projection = ((flat - centre) @ v[:, :3]).reshape(len(images), PATCH_GRID, PATCH_GRID, 3)
        lo, hi = projection.flatten(0, 2).quantile(0.02, dim=0), projection.flatten(0, 2).quantile(0.98, dim=0)
        projection = ((projection - lo) / (hi - lo).clamp_min(t.finfo(projection.dtype).eps)).clamp(0, 1).numpy()
        fig, axes = plt.subplots(2, len(images), figsize=(2.8 * len(images), 5.6), squeeze=False)
        for i, image in enumerate(images):
            axes[0, i].imshow(image)
            axes[0, i].set_title("scene", fontsize=8)
            axes[1, i].imshow(projection[i], interpolation="nearest")
            axes[1, i].set_title("SigLIP patches: top-3 PCs as RGB", fontsize=8)
        for ax in axes.flat:
            ax.axis("off")
        plt.tight_layout()
        plt.show()
        return
    n = len(images)
    ref = reference_features.float().cpu()
    position_mean = ref.mean(0)  # (196, 768): the average embedding at each patch position
    fig, axes = plt.subplots(3, n, figsize=(2.8 * n, 8.4), squeeze=False)
    for row, (title, subtract) in enumerate([("raw patch embeddings", False), ("minus the mean at each position", True)], start=1):
        offset = position_mean if subtract else 0
        flat_ref = (ref - offset).reshape(-1, ref.shape[-1])
        centre = flat_ref.mean(0)
        _, _, v = t.pca_lowrank(flat_ref - centre, q=3, center=False)
        proj = ((features.float().cpu() - offset).reshape(-1, ref.shape[-1]) - centre) @ v[:, :3]
        proj = proj.reshape(n, PATCH_GRID, PATCH_GRID, 3)
        lo, hi = proj.flatten(0, 2).quantile(0.02, dim=0), proj.flatten(0, 2).quantile(0.98, dim=0)
        proj = ((proj - lo) / (hi - lo)).clamp(0, 1).numpy()
        for i in range(n):
            axes[row, i].imshow(proj[i], interpolation="nearest")
            axes[row, i].set_title(f"top-3 PCs as RGB: {title}", fontsize=7)
    for i in range(n):
        axes[0, i].imshow(images[i])
        axes[0, i].set_title("scene", fontsize=8)
    for ax in axes.flat:
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_architecture() -> None:
    """A schematic of the mini VLM and of the token sequence the language model sees."""
    fig, ax = plt.subplots(figsize=(12, 4.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.4)
    ax.axis("off")

    def box(x, y, w, h, text, colour, alpha=0.3, fontsize=9):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.03", fc=colour, ec=colour, alpha=alpha))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1.4))

    # top row: the image pipeline, ending directly above the image-token segment of the sequence
    y = 3.2
    box(0.1, y, 1.3, 0.9, "image\n224x224", GREY)
    box(1.75, y, 1.5, 0.9, "SigLIP ViT-B/16\n(frozen)", BLUE)
    box(3.6, y, 1.6, 0.9, "merge 2x2 patches\n196x768 -> 49x3072", GREY, fontsize=8)
    box(5.55, y, 1.5, 0.9, "projector MLP\n(trained)", ORANGE)
    box(7.4, y, 1.7, 0.9, "49 image tokens\n(d_model = 768)", GREEN)
    for x0, x1 in [(1.4, 1.75), (3.25, 3.6), (5.2, 5.55), (7.05, 7.4)]:
        arrow(x0, y + 0.45, x1, y + 0.45)
    box(9.5, y, 2.4, 0.9, '"Question: What colour is\nthe circle? Answer:"\n-> GPT-2 token embeddings', BLUE, fontsize=8)
    # bottom row: the sequence
    ax.text(0.1, 1.75, "the sequence the\nlanguage model reads\n(12 transformer blocks,\ncausal attention)", fontsize=9, va="center")
    seq_y = 1.3
    ax.add_patch(Rectangle((6.0, seq_y), 0.8, 0.8, fc=GREY, alpha=0.4, ec="k"))
    ax.text(6.4, seq_y + 0.4, "BOS", ha="center", va="center", fontsize=8)
    ax.add_patch(Rectangle((6.8, seq_y), 2.5, 0.8, fc=GREEN, alpha=0.4, ec="k"))
    ax.text(8.05, seq_y + 0.4, "49 image tokens", ha="center", va="center", fontsize=9)
    ax.add_patch(Rectangle((9.3, seq_y), 1.6, 0.8, fc=BLUE, alpha=0.3, ec="k"))
    ax.text(10.1, seq_y + 0.4, "question tokens", ha="center", va="center", fontsize=9)
    ax.add_patch(Rectangle((10.9, seq_y), 1.0, 0.8, fc=RED, alpha=0.3, ec="k"))
    ax.text(11.4, seq_y + 0.4, "answer", ha="center", va="center", fontsize=9)
    arrow(8.25, y - 0.05, 8.05, seq_y + 0.85)
    arrow(10.7, y - 0.05, 10.1, seq_y + 0.85)
    arrow(10.35, seq_y - 0.05, 11.3, seq_y - 0.05)
    ax.text(8.0, 0.55, 'the model predicts the answer (e.g. " red") at the last question token', fontsize=8.5)
    ax.set_title("A LLaVA-style VLM, small enough to train in minutes", fontsize=11)
    plt.tight_layout()
    plt.show()


def plot_training_curves(curves: dict[str, list[float]], smooth: int = 25, title: str = "Training loss") -> None:
    """Smoothed training-loss curves, one per run (log y-axis)."""
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, losses in curves.items():
        losses = np.asarray(losses)
        kernel = np.ones(smooth) / smooth
        ax.plot(np.convolve(losses, kernel, mode="valid"), label=name)
    ax.set_yscale("log")
    ax.set_xlabel("training step")
    ax.set_ylabel(f"answer-token cross-entropy ({smooth}-step moving avg)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_accuracy_by_condition(
    results: dict[str, dict[str, float]], chance: dict[str, float], title: str, kinds: list[str] | None = None
) -> None:
    """
    Grouped bar chart: one group per question kind, one bar per condition (results[condition][kind]).
    `chance[kind]` is drawn as a black tick over each group (the best an image-blind model could do).
    """
    kinds = kinds or list(chance)
    conditions = list(results)
    width = 0.8 / len(conditions)
    colours = [BLUE, "#8fb7dc", ORANGE, GREEN, PURPLE, RED, GREY, "#bcbd22"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, condition in enumerate(conditions):
        xs = np.arange(len(kinds)) + (i - (len(conditions) - 1) / 2) * width
        ax.bar(xs, [results[condition][k] for k in kinds], width, label=condition, color=colours[i % len(colours)])
    for j, kind in enumerate(kinds):
        ax.plot([j - 0.42, j + 0.42], [chance[kind]] * 2, "k-", lw=2.5, label="best image-blind guess" if j == 0 else None)
    ax.set_xticks(range(len(kinds)))
    ax.set_xticklabels(kinds)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy (forced choice among the valid answers)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=4, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    plt.show()


def plot_layer_curves(
    curves: dict[str, list[float]],
    title: str,
    ylabel: str,
    xlabel: str = "layer (residual stream before block)",
    controls: tuple[str, ...] = (),
    logy: bool = False,
    ylim: tuple[float, float] | None = None,
    hlines: dict[str, float] | None = None,
    ax=None,
) -> None:
    """One line per entry of `curves`; names in `controls` are drawn dashed and grey-ish."""
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 4))
    palette = [BLUE, ORANGE, GREEN, RED, PURPLE]
    control_palette = [GREY, "#8c564b", "#bcbd22", "#17becf"]
    i = j = 0
    for name, values in curves.items():
        if name in controls:
            ax.plot(range(len(values)), values, "--s", color=control_palette[j % 4], label=name, markersize=3.5)
            j += 1
        else:
            ax.plot(range(len(values)), values, "-o", color=palette[i % 5], label=name, markersize=4)
            i += 1
    for name, value in (hlines or {}).items():
        ax.axhline(value, color="k", ls=":", lw=1.2, label=name)
    if logy:
        ax.set_yscale("log")
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    if show:
        plt.tight_layout()
        plt.show()


def plot_head_heatmap(values: t.Tensor, title: str, ax=None, vmax: float | None = None) -> None:
    """Heatmap of a (layer, head) array."""
    show = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 4.6))
    values = values.detach().float().cpu().numpy()
    im = ax.imshow(values, cmap="viridis", origin="lower", vmin=0, vmax=vmax)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_xticks(range(values.shape[1]))
    ax.set_yticks(range(values.shape[0]))
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046)
    if show:
        plt.tight_layout()
        plt.show()


def _outline(ax, obj: SceneObject, colour: str = "white", lw: float = 1.5, ls: str = "-") -> None:
    """Draw the outline of an object (in image pixel coordinates) on an axis."""
    cx, cy = obj.centre
    half = obj.size / 2
    if obj.shape == "circle":
        patch = Circle((cx, cy), half, fill=False, ec=colour, lw=lw, ls=ls)
    elif obj.shape == "square":
        patch = Rectangle((cx - half, cy - half), obj.size, obj.size, fill=False, ec=colour, lw=lw, ls=ls)
    else:
        patch = Polygon([(cx, cy - half), (cx - half, cy + half), (cx + half, cy + half)], closed=True, fill=False, ec=colour, lw=lw, ls=ls)
    ax.add_patch(patch)


def plot_token_maps(
    scenes: list[Scene],
    targets: list[int],
    maps: list[np.ndarray],
    row_titles: list[str],
    captions: list[str],
    vmin: float = 0.0,
    vmax: float = 1.0,
    suptitle: str | None = None,
    cmap: str = "magma",
) -> None:
    """
    One column per scene. Row 0 shows the scene with the target object outlined in white. Each further row shows one
    (n_scenes, 49) map as a 7x7 heatmap (one cell per image token), with the objects outlined: the target object in
    solid white, the others dashed.
    """
    n = len(scenes)
    images = render_scenes(scenes)
    fig, axes = plt.subplots(len(maps) + 1, n, figsize=(2.5 * n + 0.8, 2.55 * (len(maps) + 1)), squeeze=False)
    extent = (0, IMAGE_SIZE, IMAGE_SIZE, 0)
    for i, (scene, target) in enumerate(zip(scenes, targets)):
        axes[0, i].imshow(images[i])
        _outline(axes[0, i], scene.objects[target], "white", 2.0)
        axes[0, i].set_title(captions[i], fontsize=8)
        for r, (m, title) in enumerate(zip(maps, row_titles), start=1):
            ax = axes[r, i]
            im = ax.imshow(np.asarray(m[i]).reshape(TOKEN_GRID, TOKEN_GRID), extent=extent, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
            for j, obj in enumerate(scene.objects):
                _outline(ax, obj, "white" if j == target else "#aaaaaa", 2.0 if j == target else 1.0, "-" if j == target else "--")
            if i == 0:
                ax.set_ylabel(title, fontsize=8)
        for r in range(len(maps) + 1):
            axes[r, i].set_xticks([])
            axes[r, i].set_yticks([])
    for r, title in enumerate(["scene"] + row_titles):
        axes[r, 0].set_ylabel(title, fontsize=8)
    fig.colorbar(im, ax=axes[1:, :].ravel().tolist(), fraction=0.02, pad=0.01)
    if suptitle:
        fig.suptitle(suptitle, fontsize=10)
    plt.show()


def plot_bars(
    values: dict[str, float],
    title: str,
    ylabel: str = "accuracy",
    colours: list[str] | None = None,
    hlines: dict[str, float] | None = None,
) -> None:
    """Simple bar chart (first bar blue, the rest grey unless `colours` is given); optional labelled horizontal lines."""
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    names, vals = list(values), list(values.values())
    bars = ax.bar(names, vals, color=colours or [BLUE] + [GREY] * (len(vals) - 1))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.0%}", ha="center")
    for name, value in (hlines or {}).items():
        ax.axhline(value, color="k", ls=":", lw=1.3, label=name)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    if hlines:
        ax.legend(loc="upper right", fontsize=8)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.show()


def plot_relative_depth_curves(
    panels: dict[str, tuple[dict[str, list[float]], list[int], int]], controls: tuple[str, ...] = ()
) -> None:
    """
    One panel per model. panels[title] = (curves, layers, n_layers), where curves[name] has one value per entry of
    `layers`. The x-axis is relative depth layer / n_layers, so models with different depths can be compared.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(6.4 * len(panels), 4.2), sharey=True, squeeze=False)
    palette = [BLUE, ORANGE, GREEN, RED, PURPLE]
    control_palette = [GREY, "#8c564b", "#bcbd22", "#17becf"]
    for ax, (title, (curves, layers, n_layers)) in zip(axes[0], panels.items()):
        i = j = 0
        for name, values in curves.items():
            xs = [layer / n_layers for layer in layers]
            if name in controls:
                ax.plot(xs, values, "--s", color=control_palette[j % 4], label=name, markersize=3.5)
                j += 1
            else:
                ax.plot(xs, values, "-o", color=palette[i % 5], label=name, markersize=4)
                i += 1
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("relative depth (layer / number of layers)")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("recovery (0 = corrupted answer, 1 = clean answer)")
    axes[0, 0].legend(fontsize=8)
    plt.tight_layout()
    plt.show()


def plot_knockout(knockout: dict[str, list[float]], window_labels: list[str], chance: float) -> None:
    """Grouped bars: one group per window of blocked layers, one bar per route (knockout[route][window])."""
    routes = list(knockout)
    width = 0.8 / len(routes)
    colours = [BLUE, GREY, GREEN, ORANGE, PURPLE, RED]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for i, route in enumerate(routes):
        xs = np.arange(len(window_labels)) + (i - (len(routes) - 1) / 2) * width
        ax.bar(xs, knockout[route], width, label=route, color=colours[i % len(colours)])
    ax.axhline(chance, color="k", ls=":", lw=1.3, label="best image-blind guess")
    ax.set_xticks(range(len(window_labels)))
    ax.set_xticklabels(window_labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("colour accuracy (forced choice)")
    ax.set_xlabel("attention is blocked in these layers")
    ax.set_title("Attention knockout: block a route, measure the answer", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.17))
    plt.tight_layout()
    plt.show()

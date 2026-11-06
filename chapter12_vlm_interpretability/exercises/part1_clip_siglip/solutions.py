# %%


import itertools
import math
import sys
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Float, Int
from PIL import Image
from torch import Tensor
from tqdm.auto import tqdm

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter12_vlm_interpretability"
section = "part1_clip_siglip"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_clip_siglip.tests as tests
import part1_clip_siglip.utils as utils

pd.set_option("display.width", 220)  # don't wrap wide tables
pd.set_option("display.max_columns", 20)

MAIN = __name__ == "__main__"

# %%

def similarity_logits(
    image_emb: Float[Tensor, "n_images d"],
    text_emb: Float[Tensor, "n_texts d"],
    logit_scale: float | Tensor,
    logit_bias: float | Tensor = 0.0,
) -> Float[Tensor, "n_images n_texts"]:
    """Scaled cosine similarities between every image and every text: logit_scale * cos + logit_bias."""
    image_emb = F.normalize(image_emb, dim=-1)
    text_emb = F.normalize(text_emb, dim=-1)
    return logit_scale * image_emb @ text_emb.T + logit_bias


if MAIN:
    tests.test_similarity_logits(similarity_logits)

# %%

def clip_loss(logits: Float[Tensor, "n n"]) -> Float[Tensor, ""]:
    """Symmetric InfoNCE: average of image-to-text and text-to-image cross-entropy, with matches on the diagonal."""
    labels = t.arange(logits.shape[0], device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


if MAIN:
    tests.test_clip_loss(clip_loss)

# %%

if MAIN:
    t.manual_seed(0)
    N, d = 64, 32
    z_img = F.normalize(t.randn(N, d), dim=-1)
    conditions = {
        "aligned": z_img,
        "noisy-aligned": F.normalize(z_img + 0.5 * F.normalize(t.randn(N, d), dim=-1), dim=-1),
        "mismatched (control)": z_img.roll(1, dims=0),
    }
    scales = np.logspace(-1, 2.5, 15)
    curves = {
        name: np.array([clip_loss(similarity_logits(z_img, z_txt, s)).item() for s in scales])
        for name, z_txt in conditions.items()
    }
    utils.plot_loss_vs_scale(scales, curves, N, "InfoNCE loss as a function of the logit scale (N = 64)")
    
    for s in [1, 10, 100]:
        row = {name: f"{clip_loss(similarity_logits(z_img, z_txt, s)).item():8.3f}" for name, z_txt in conditions.items()}
        print(f"scale {s:>3}: " + "   ".join(f"{name}: {v}" for name, v in row.items()))

# %%

def siglip_loss_terms(logits: Float[Tensor, "n n"]) -> tuple[Float[Tensor, ""], Float[Tensor, ""]]:
    """Returns (positive_term, negative_term) of the sigmoid loss, each divided by the batch size N."""
    n = logits.shape[0]
    eye = t.eye(n, dtype=t.bool, device=logits.device)
    positive_term = F.softplus(-logits[eye]).sum() / n
    negative_term = F.softplus(logits[~eye]).sum() / n
    return positive_term, negative_term


def siglip_loss(logits: Float[Tensor, "n n"]) -> Float[Tensor, ""]:
    """The pairwise sigmoid loss: positive term plus negative term."""
    positive_term, negative_term = siglip_loss_terms(logits)
    return positive_term + negative_term


if MAIN:
    tests.test_siglip_loss_terms(siglip_loss_terms)

# %%

if MAIN:
    t.manual_seed(0)
    batch_sizes = [4, 16, 64, 256, 1024, 4096]
    inits = {"paper init: scale 10, bias -10": (10.0, -10.0), "CLIP-style init: scale 14.3, bias 0": (1 / 0.07, 0.0)}
    terms = {}
    for name, (scale, bias) in inits.items():
        pos, neg = [], []
        for n in batch_sizes:
            z = F.normalize(t.randn(n, 64), dim=-1), F.normalize(t.randn(n, 64), dim=-1)
            p, q = siglip_loss_terms(similarity_logits(*z, scale, bias))
            pos.append(p.item())
            neg.append(q.item())
        terms[name] = (pos, neg)
        print(f"{name}\n  positive term: " + ", ".join(f"{v:.3g}" for v in pos) + "\n  negative term: " + ", ".join(f"{v:.3g}" for v in neg))
    utils.plot_siglip_init_terms(batch_sizes, terms, "Sigmoid loss at initialisation (random embeddings)")

# %%

def clip_loss_grad(logits: Float[Tensor, "n n"]) -> Float[Tensor, "n n"]:
    """Gradient of `clip_loss` with respect to the logit matrix, computed by hand."""
    n = logits.shape[0]
    row_softmax = logits.softmax(dim=1)
    column_softmax = logits.softmax(dim=0)
    return (row_softmax + column_softmax - 2 * t.eye(n, device=logits.device)) / (2 * n)


def siglip_loss_grad(logits: Float[Tensor, "n n"]) -> Float[Tensor, "n n"]:
    """Gradient of `siglip_loss` with respect to the logit matrix, computed by hand."""
    n = logits.shape[0]
    return (logits.sigmoid() - t.eye(n, device=logits.device)) / n


if MAIN:
    tests.test_loss_gradients(clip_loss_grad, siglip_loss_grad, clip_loss, siglip_loss)

# %%

if MAIN:
    t.manual_seed(1)
    logits = 3 * F.normalize(t.randn(8, 16), dim=-1) @ F.normalize(t.randn(8, 16), dim=-1).T
    for shift in [0.0, 5.0, -5.0]:
        shifted = logits + shift
        print(
            f"shift {shift:+.0f}:  CLIP loss {clip_loss(shifted).item():.4f}  (sum of gradient entries {clip_loss_grad(shifted).sum().item():+.1e})"
            f"   SigLIP loss {siglip_loss(shifted).item():.4f}  (sum of gradient entries {siglip_loss_grad(shifted).sum().item():+.4f})"
        )

# %%

def false_negative_fraction(caption_ids: Int[Tensor, " n"]) -> float:
    """Fraction of items in the batch whose caption is shared by at least one other item in the batch."""
    same = caption_ids[:, None] == caption_ids[None, :]
    same = same & ~t.eye(len(caption_ids), dtype=t.bool, device=caption_ids.device)
    return same.any(dim=1).float().mean().item()


if MAIN:
    tests.test_false_negative_fraction(false_negative_fraction)

# %%

if MAIN:
    generator = t.Generator().manual_seed(0)
    for n_distinct in [100, 1000, 100_000]:
        rates = [false_negative_fraction(t.randint(0, n_distinct, (64,), generator=generator)) for _ in range(500)]
        print(f"batch of 64 drawn with replacement from {n_distinct:>7,} distinct captions: {np.mean(rates):6.1%} of rows have a false negative")

# %%

COLOURS = list(utils.COLOUR_RGB)
SHAPES = ["circle", "square", "triangle", "star", "heart"]
POSITIONS = list(utils.QUADRANT_CENTRES)
CLASSES = list(itertools.product(COLOURS, SHAPES, POSITIONS))  # (colour, shape, position); the class index is the list index
HELD_OUT_COMBOS = {("red", "heart"), ("green", "star"), ("blue", "triangle"), ("yellow", "circle"), ("purple", "square"), ("orange", "heart")}


def caption_of(cls: tuple[str, str, str]) -> str:
    colour, shape, position = cls
    return f"a {colour} {shape} in the {position}"


CAPTIONS = [caption_of(cls) for cls in CLASSES]
held_out_mask = t.tensor([(colour, shape) in HELD_OUT_COMBOS for colour, shape, _ in CLASSES])
train_classes = (~held_out_mask).nonzero().squeeze(-1)  # 96 classes


def render_world(n_variants: int, seed: int) -> Float[Tensor, "variant class 3 32 32"]:
    """Render `n_variants` random images of every class, as float tensors in [0, 1]."""
    rng = np.random.default_rng(seed)
    images = np.stack(
        [[utils.render_tiny(shape, colour, position, rng) for colour, shape, position in CLASSES] for _ in range(n_variants)]
    )
    return t.tensor(images).permute(0, 1, 4, 2, 3).float() / 255

# %%

if MAIN:
    train_images = render_world(24, seed=0).to(device)  # 24 random images of each of the 120 classes (the 24 held-out classes are never sampled)
    test_images = render_world(1, seed=1)[0].to(device)  # one fresh image per class, for evaluation
    print(f"{len(CLASSES)} classes, {len(train_classes)} for training, {int(held_out_mask.sum())} held out")
    print(f"train_images: {tuple(train_images.shape)}, test_images: {tuple(test_images.shape)}")
    print(f"example caption: {CAPTIONS[0]!r}")
    
    # one image per (colour, shape) combination, all in the top-left quadrant
    top_left = [i for i, (_, _, position) in enumerate(CLASSES) if position == "top left"]
    utils.show_image_grid(
        (test_images[top_left].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8),
        [f"{CLASSES[i][0]} {CLASSES[i][1]}" + ("\n(held out)" if held_out_mask[i] else "") for i in top_left],
        ncols=10,
        title="The tiny world (top-left quadrant only; each class also appears in the other three)",
    )

# %%

VOCAB = ["<pad>", "<bos>", "<eos>", "a", "in", "the", "and"] + COLOURS + SHAPES + ["top", "bottom", "left", "right"]
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
MAX_LEN = 10


def tokenize_captions(captions: list[str], vocab: list[str], max_len: int) -> Int[Tensor, "n max_len"]:
    """<bos> + word ids + <eos>, right-padded with <pad> (id 0) to max_len."""
    word_to_id = {word: i for i, word in enumerate(vocab)}
    ids = t.full((len(captions), max_len), PAD_ID, dtype=t.long)
    for row, caption in enumerate(captions):
        tokens = [BOS_ID] + [word_to_id[word] for word in caption.split()] + [EOS_ID]
        ids[row, : len(tokens)] = t.tensor(tokens)
    return ids


if MAIN:
    tests.test_tokenize_captions(tokenize_captions, VOCAB, MAX_LEN)

# %%

if MAIN:
    CAPTION_IDS = tokenize_captions(CAPTIONS, VOCAB, MAX_LEN).to(device)
    print(CAPTIONS[0], "->", CAPTION_IDS[0].tolist())
    print(" ".join(VOCAB[i] for i in CAPTION_IDS[0].tolist()))

# %%

def sample_batch(
    images: Float[Tensor, "variant class 3 32 32"],
    allowed_classes: Int[Tensor, " n_allowed"],
    batch_size: int,
    generator: t.Generator,
) -> tuple[Float[Tensor, "batch 3 32 32"], Int[Tensor, " batch"]]:
    """A batch of `batch_size` images of distinct, randomly chosen classes (and their class indices)."""
    chosen = allowed_classes[t.randperm(len(allowed_classes), generator=generator)[:batch_size]]
    variants = t.randint(0, images.shape[0], (batch_size,), generator=generator)
    class_ids = chosen.to(images.device)
    return images[variants.to(images.device), class_ids], class_ids


if MAIN:
    tests.test_sample_batch(sample_batch)

# %%

if MAIN:
    generator = t.Generator().manual_seed(0)
    iid = [false_negative_fraction(train_classes[t.randint(0, len(train_classes), (64,), generator=generator)]) for _ in range(200)]
    unique = [false_negative_fraction(sample_batch(train_images, train_classes, 64, generator)[1]) for _ in range(200)]
    print(f"false-negative rows, batch of 64, classes drawn with replacement: {np.mean(iid):.1%}")
    print(f"false-negative rows, batch of 64, `sample_batch`:                 {np.mean(unique):.1%}")

# %%

def patchify(images: Float[Tensor, "batch channel height width"], patch_size: int) -> Float[Tensor, "batch n_patches patch_dim"]:
    """Cut images into non-overlapping patches and flatten each one: (B, C, H, W) -> (B, (H/p)*(W/p), C*p*p)."""
    b, c, h, w = images.shape
    p = patch_size
    patches = images.reshape(b, c, h // p, p, w // p, p)  # split each spatial axis into (patch index, offset within patch)
    patches = patches.permute(0, 2, 4, 1, 3, 5)  # (batch, patch row, patch col, channel, row in patch, col in patch)
    return patches.reshape(b, (h // p) * (w // p), c * p * p)


if MAIN:
    tests.test_patchify(patchify)

# %%

class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.causal = n_heads, causal
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
        batch, seq, d_model = x.shape
        head_dim = d_model // self.n_heads
        qkv = self.qkv(x).reshape(batch, seq, 3, self.n_heads, head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # each: (batch, n_heads, seq, head_dim)
        scores = q @ k.transpose(-1, -2) / math.sqrt(head_dim)
        if self.causal:
            future = t.ones(seq, seq, dtype=t.bool, device=x.device).triu(diagonal=1)
            scores = scores.masked_fill(future, float("-inf"))
        mixed = scores.softmax(dim=-1) @ v  # (batch, n_heads, seq, head_dim)
        return self.out(mixed.transpose(1, 2).reshape(batch, seq, d_model))


if MAIN:
    tests.test_self_attention(SelfAttention)

# %%

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, causal)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, x: Float[Tensor, "batch seq d_model"]) -> Float[Tensor, "batch seq d_model"]:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))

# %%

class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, n_layers: int, n_heads: int, max_len: int):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(0.02 * t.randn(max_len, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, causal=True) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, ids: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch d_model"]:
        x = self.token_embedding(ids) + self.pos_embedding[: ids.shape[1]]
        for block in self.blocks:
            x = block(x)
        x = self.ln_final(x)
        eos_position = (ids == EOS_ID).int().argmax(dim=-1)  # position of the (only) <eos> in each caption
        return x[t.arange(len(ids), device=ids.device), eos_position]


if MAIN:
    tests.test_text_encoder(TextEncoder, EOS_ID)

# %%

class ImageEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int, image_size: int = 32, patch_size: int = 8):
        super().__init__()
        self.patch_size = patch_size
        n_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Linear(3 * patch_size**2, d_model)
        self.cls_token = nn.Parameter(t.zeros(1, 1, d_model))
        self.pos_embedding = nn.Parameter(0.02 * t.randn(1, n_patches + 1, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, causal=False) for _ in range(n_layers)])
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, images: Float[Tensor, "batch 3 height width"]) -> Float[Tensor, "batch d_model"]:
        tokens = self.patch_embed(patchify(images, self.patch_size))
        x = t.cat([self.cls_token.expand(len(images), -1, -1), tokens], dim=1) + self.pos_embedding
        for block in self.blocks:
            x = block(x)
        return self.ln_final(x)[:, 0]


if MAIN:
    tests.test_image_encoder(ImageEncoder, patchify)

# %%

class TinyCLIP(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        d_embed: int = 32,
        n_layers: int = 2,
        n_heads: int = 4,
        max_len: int = MAX_LEN,
        siglip: bool = False,
        scale_init: float | None = None,
        bias_init: float = -10.0,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(d_model, n_layers, n_heads)
        self.text_encoder = TextEncoder(vocab_size, d_model, n_layers, n_heads, max_len)
        self.image_proj = nn.Linear(d_model, d_embed, bias=False)
        self.text_proj = nn.Linear(d_model, d_embed, bias=False)
        if scale_init is None:
            scale_init = 10.0 if siglip else 1 / 0.07
        self.logit_scale = nn.Parameter(t.tensor(math.log(scale_init)))
        if siglip:
            self.logit_bias = nn.Parameter(t.tensor(float(bias_init)))
        else:
            self.register_parameter("logit_bias", None)

    def encode_image(self, images: Float[Tensor, "batch 3 32 32"]) -> Float[Tensor, "batch d_embed"]:
        return self.image_proj(self.image_encoder(images))

    def encode_text(self, ids: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch d_embed"]:
        return self.text_proj(self.text_encoder(ids))

    def forward(self, images: Float[Tensor, "batch 3 32 32"], ids: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch batch"]:
        bias = 0.0 if self.logit_bias is None else self.logit_bias
        return similarity_logits(self.encode_image(images), self.encode_text(ids), self.logit_scale.exp(), bias)


if MAIN:
    tests.test_tiny_clip(TinyCLIP, VOCAB, similarity_logits)

# %%

if MAIN:
    t.manual_seed(0)
    model = TinyCLIP(len(VOCAB)).to(device)
    n_image = sum(p.numel() for p in model.image_encoder.parameters())
    n_text = sum(p.numel() for p in model.text_encoder.parameters())
    print(f"image tower: {n_image:,} parameters, text tower: {n_text:,} parameters, projections + scale: {sum(p.numel() for p in model.parameters()) - n_image - n_text:,}")
    logits = model(train_images[0, :4], CAPTION_IDS[:4])
    print("logits for 4 images x 4 captions:", tuple(logits.shape), " initial scale:", round(model.logit_scale.exp().item(), 2))
    del model

# %%

def retrieval_metrics(
    sim: Float[Tensor, "n_images n_captions"],
    image_classes: Int[Tensor, " n_images"],
    caption_classes: Int[Tensor, " n_captions"],
) -> dict[str, float]:
    """Image-to-text and text-to-image recall@1, where 'correct' means the retrieved item has the right class."""
    best_caption = sim.argmax(dim=1)  # for each image
    best_image = sim.argmax(dim=0)  # for each caption
    image_to_text = (caption_classes[best_caption] == image_classes).float().mean().item()
    text_to_image = (image_classes[best_image] == caption_classes).float().mean().item()
    return {"image_to_text": image_to_text, "text_to_image": text_to_image}


if MAIN:
    tests.test_retrieval_metrics(retrieval_metrics)

# %%

def make_control_pairs(
    class_ids: Int[Tensor, " batch"], mode: str, generator: t.Generator, fixed_perm: Int[Tensor, " n_classes"] | None = None
) -> Int[Tensor, " batch"]:
    """Class ids whose captions are paired with the batch's images, under the chosen pairing rule."""
    if mode == "paired":
        return class_ids
    if mode == "shuffled":
        return class_ids[t.randperm(len(class_ids), generator=generator).to(class_ids.device)]
    if mode == "permuted":
        assert fixed_perm is not None, "mode='permuted' needs fixed_perm"
        return fixed_perm.to(class_ids.device)[class_ids]
    raise ValueError(f"Unknown pairing mode {mode!r}")


if MAIN:
    tests.test_make_control_pairs(make_control_pairs)

# %%

def train_tiny_clip(
    model: TinyCLIP,
    images: Float[Tensor, "variant class 3 32 32"],
    caption_ids: Int[Tensor, "class seq"],
    train_classes: Int[Tensor, " n_train"],
    loss_name: str = "clip",
    control: str = "paired",
    fixed_perm: Int[Tensor, " n_classes"] | None = None,
    steps: int = 2000,
    batch_size: int = 64,
    lr: float = 5e-4,
    seed: int = 0,
    eval_fn: Callable[[TinyCLIP], float] | None = None,
    eval_every: int = 100,
) -> dict[str, list]:
    """Train `model` in place with the CLIP or SigLIP loss. Returns the per-step losses (and periodic evaluations)."""
    loss_fn = {"clip": clip_loss, "siglip": siglip_loss}[loss_name]
    generator = t.Generator().manual_seed(seed)
    optimizer = t.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = t.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, total_steps=steps, pct_start=0.1)
    history: dict[str, list] = {"loss": [], "eval_steps": [], "eval": []}
    model.train()
    for step in tqdm(range(steps), leave=False):
        batch_images, class_ids = sample_batch(images, train_classes, batch_size, generator)
        caption_classes = make_control_pairs(class_ids, control, generator, fixed_perm)
        loss = loss_fn(model(batch_images, caption_ids[caption_classes]))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        with t.no_grad():
            model.logit_scale.clamp_(0, math.log(100))
        history["loss"].append(loss.item())
        if eval_fn is not None and ((step + 1) % eval_every == 0 or step == steps - 1):
            model.eval()
            history["eval_steps"].append(step + 1)
            history["eval"].append(eval_fn(model))
            model.train()
    model.eval()
    return history


if MAIN:
    tests.test_train_tiny_clip(train_tiny_clip, TinyCLIP, tokenize_captions, VOCAB, MAX_LEN)

# %%

@t.no_grad()
def embed_all(model: TinyCLIP, images: Tensor, caption_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Raw image embeddings of all images and raw text embeddings of all captions (model in eval mode)."""
    was_training = model.training
    model.eval()
    out = model.encode_image(images), model.encode_text(caption_ids)
    model.train(was_training)
    return out


def evaluate_tiny(model: TinyCLIP, images: Tensor, caption_ids: Tensor, held_out: Tensor) -> dict[str, float]:
    """Retrieval of the 120 test images against all 120 captions, split into seen and held-out classes."""
    image_emb, text_emb = embed_all(model, images, caption_ids)
    sim = similarity_logits(image_emb, text_emb, 1.0)  # plain cosine similarity
    classes = t.arange(len(images), device=sim.device)
    held_out = held_out.to(sim.device)
    overall = retrieval_metrics(sim, classes, classes)
    seen = retrieval_metrics(sim[~held_out], classes[~held_out], classes)
    heldout = retrieval_metrics(sim[held_out], classes[held_out], classes)
    return {
        "i2t": overall["image_to_text"],
        "t2i": overall["text_to_image"],
        "i2t_seen": seen["image_to_text"],
        "i2t_held_out": heldout["image_to_text"],
    }

# %%

if MAIN:
    def fresh_model(seed: int = 0, **kwargs) -> TinyCLIP:
        t.manual_seed(seed)
        return TinyCLIP(len(VOCAB), **kwargs).to(device)
    
    
    eval_i2t = lambda m: evaluate_tiny(m, test_images, CAPTION_IDS, held_out_mask)["i2t"]
    runs: dict[str, tuple[TinyCLIP, dict]] = {}
    
    for name, control in [("CLIP loss, paired captions", "paired"), ("CLIP loss, shuffled captions (control)", "shuffled")]:
        m = fresh_model()
        runs[name] = (m, train_tiny_clip(m, train_images, CAPTION_IDS, train_classes, "clip", control, eval_fn=eval_i2t))
        r = evaluate_tiny(m, test_images, CAPTION_IDS, held_out_mask)
        print(f"{name:<42} final loss {np.mean(runs[name][1]['loss'][-50:]):.3f}   i2t R@1 {r['i2t']:.3f}   t2i R@1 {r['t2i']:.3f}")
    untrained = fresh_model()
    r = evaluate_tiny(untrained, test_images, CAPTION_IDS, held_out_mask)
    print(f"{'untrained (control)':<42} {'':>18}   i2t R@1 {r['i2t']:.3f}   t2i R@1 {r['t2i']:.3f}")
    print(f"chance: {1 / len(CLASSES):.3f}")

# %%

if MAIN:
    sims, metrics = {}, {}
    for name, m in [
        ("CLIP loss, paired captions", runs["CLIP loss, paired captions"][0]),
        ("shuffled captions (control)", runs["CLIP loss, shuffled captions (control)"][0]),
        ("untrained (control)", untrained),
    ]:
        image_emb, text_emb = embed_all(m, test_images, CAPTION_IDS)
        sims[name] = similarity_logits(image_emb, text_emb, 1.0).cpu().numpy()
        r = evaluate_tiny(m, test_images, CAPTION_IDS, held_out_mask)
        metrics[name] = f"image-to-text R@1 = {r['i2t']:.2f}  (seen {r['i2t_seen']:.2f}, held-out {r['i2t_held_out']:.2f})"
    utils.plot_retrieval_heatmaps(sims, held_out_mask.numpy(), metrics, title="Tiny CLIP: image-caption cosine similarity on fresh test images")

# %%

if MAIN:
    fixed_perm = t.randperm(len(CLASSES), generator=t.Generator().manual_seed(123))
    m_perm = fresh_model()
    hist_perm = train_tiny_clip(m_perm, train_images, CAPTION_IDS, train_classes, "clip", "permuted", fixed_perm=fixed_perm, eval_fn=eval_i2t)
    image_emb, text_emb = embed_all(m_perm, test_images, CAPTION_IDS)
    sim = similarity_logits(image_emb, text_emb, 1.0)
    classes = t.arange(len(CLASSES), device=device)
    true_pairing = retrieval_metrics(sim, classes, classes)["image_to_text"]
    # the caption this model was taught to associate with class c is caption fixed_perm[c]
    learned_pairing = retrieval_metrics(sim, fixed_perm.to(device), classes)["image_to_text"]
    print(f"final training loss: {np.mean(hist_perm['loss'][-50:]):.3f}")
    print(f"image-to-text R@1 against the TRUE captions:               {true_pairing:.3f}")
    print(f"image-to-text R@1 against the captions it was TRAINED on:  {learned_pairing:.3f}")
    utils.plot_retrieval_heatmaps(
        {
            "columns in true class order": sim.cpu().numpy(),
            "columns re-ordered by the training labels": sim[:, fixed_perm.to(device)].cpu().numpy(),
        },
        held_out_mask.numpy(),
        {
            "columns in true class order": f"R@1 against the true captions = {true_pairing:.2f}",
            "columns re-ordered by the training labels": f"R@1 against its training labels = {learned_pairing:.2f}",
        },
        title="Tiny CLIP trained on a consistent but wrong labelling (the same similarity matrix, two column orders)",
    )

# %%

if MAIN:
    for name, control, steps, kwargs in [
        ("SigLIP loss, paired captions", "paired", 2000, dict(siglip=True)),
        ("SigLIP loss, shuffled captions (control)", "shuffled", 2000, dict(siglip=True)),
        ("SigLIP loss, paired, bias init 0 / scale 14.3", "paired", 2000, dict(siglip=True, scale_init=1 / 0.07, bias_init=0.0)),
        ("same, trained 3x longer (6,000 steps)", "paired", 6000, dict(siglip=True, scale_init=1 / 0.07, bias_init=0.0)),
    ]:
        m = fresh_model(**kwargs)
        runs[name] = (m, train_tiny_clip(m, train_images, CAPTION_IDS, train_classes, "siglip", control, steps=steps, eval_fn=eval_i2t, eval_every=steps // 20))
        r = evaluate_tiny(m, test_images, CAPTION_IDS, held_out_mask)
        bias = f"{m.logit_bias.item():+.2f}"
        print(f"{name:<48} final loss {np.mean(runs[name][1]['loss'][-50:]):.3f}   i2t R@1 {r['i2t']:.3f} (held-out {r['i2t_held_out']:.3f})   scale {m.logit_scale.exp().item():.1f}  bias {bias}")
    for name in ["CLIP loss, paired captions", "CLIP loss, shuffled captions (control)"]:
        m = runs[name][0]
        r = evaluate_tiny(m, test_images, CAPTION_IDS, held_out_mask)
        print(f"{name:<48} final loss {np.mean(runs[name][1]['loss'][-50:]):.3f}   i2t R@1 {r['i2t']:.3f} (held-out {r['i2t_held_out']:.3f})   scale {m.logit_scale.exp().item():.1f}")
    
    utils.plot_training_curves(
        {name: (h["eval_steps"], h["eval"]) for name, (m, h) in runs.items()},
        "Image-to-text R@1 on fresh test images during training",
        "image-to-text R@1 (120 classes)",
        chance=1 / len(CLASSES),
    )

# %%

if MAIN:
    @t.no_grad()
    def ask_tiny_clip(model: TinyCLIP, colour: str, shape: str, position: str, seed: int = 123, k: int = 5) -> None:
        """Render a fresh image of the given class, show it, and print the model's top-k captions."""
        rng = np.random.default_rng(seed)
        image = utils.render_tiny(shape, colour, position, rng)
        pixels = t.tensor(image).permute(2, 0, 1).float().div(255)[None].to(device)
        image_emb, text_emb = embed_all(model, pixels, CAPTION_IDS)
        cosines = similarity_logits(image_emb, text_emb, 1.0)[0]
        plt.figure(figsize=(1.5, 1.5))
        plt.imshow(image, interpolation="nearest")
        plt.axis("off")
        plt.title(f"{colour} {shape}, {position}", fontsize=7)
        plt.show()
        top = cosines.topk(k)
        for value, idx in zip(top.values.tolist(), top.indices.tolist()):
            held = "  (held-out class)" if held_out_mask[idx] else ""
            print(f"  {value:+.3f}  {CAPTIONS[idx]!r}{held}")
    
    
    ask_tiny_clip(runs["CLIP loss, paired captions"][0], "green", "star", "top left")

# %%

if MAIN:
    pairs = {kind: utils.load_pretrained(kind, device) for kind in ["CLIP", "SigLIP"]}
    for kind, pair in pairs.items():
        vision, text = pair.model.vision_model, pair.model.text_model
        n_vision = sum(p.numel() for p in vision.parameters()) / 1e6
        n_text = sum(p.numel() for p in text.parameters()) / 1e6
        cfg = pair.model.config.vision_config
        dim = pair.embed_texts(["a photo"]).shape[-1]
        bias = "none" if pair.logit_bias is None else f"{pair.logit_bias:.2f}"
        print(f"{kind:<7} {pair.name:<34} vision tower {n_vision:5.1f}M, text tower {n_text:5.1f}M, patch {cfg.patch_size}, image {cfg.image_size}px, embedding dim {dim}, logit scale {pair.logit_scale:6.2f}, logit bias {bias}")

# %%

if MAIN:
    cifar_images, cifar_labels = utils.load_cifar10_sample()  # uint8 [200, 32, 32, 3], int [200]
    pil_images = utils.to_pil(cifar_images)
    labels = t.tensor(cifar_labels)
    class_names = utils.CIFAR10_CLASSES
    class_ids = t.arange(len(class_names))
    
    utils.show_image_grid(
        np.concatenate([cifar_images[labels == c][:2] for c in range(10)]),
        [class_names[c] for c in range(10) for _ in range(2)],
        ncols=10,
        scale=1.3,
        title="CIFAR-10 sample (32x32 pixels, shown at 2 per class)",
    )

# %%

if MAIN:
    def captions_for(names: list[str], template: str = "a photo of a {}") -> list[str]:
        return [template.format(name) for name in names]
    
    
    check_images, check_texts = pil_images[::40], captions_for(class_names[:5])
    for kind, pair in pairs.items():
        ours = similarity_logits(
            pair.embed_images(check_images), pair.embed_texts(check_texts), pair.logit_scale, 0.0 if pair.logit_bias is None else pair.logit_bias
        )
        official = pair.official_logits(check_images, check_texts)
        print(f"{kind:<7} max |ours - official| over {ours.numel()} logits: {(ours - official).abs().max().item():.2e}   (logits range from {official.min():.1f} to {official.max():.1f})")

# %%

def zero_shot_probabilities(logits: Float[Tensor, "n_images n_texts"], kind: str) -> Float[Tensor, "n_images n_texts"]:
    """'softmax': each row sums to 1 over the candidate texts (CLIP). 'sigmoid': each pair on its own (SigLIP)."""
    if kind == "softmax":
        return logits.softmax(dim=-1)
    if kind == "sigmoid":
        return logits.sigmoid()
    raise ValueError(f"Unknown kind {kind!r}")


if MAIN:
    tests.test_zero_shot_probabilities(zero_shot_probabilities)

# %%

if MAIN:
    def class_means(sim: Tensor, image_labels: Tensor, n_classes: int = 10) -> Tensor:
        """Mean of each row-block of `sim`, grouped by image label: [n_images, n_texts] -> [n_classes, n_texts]."""
        return t.stack([sim[image_labels == c].mean(0) for c in range(n_classes)])
    
    
    rng = np.random.default_rng(0)
    noise_images = utils.to_pil(rng.integers(0, 256, size=cifar_images.shape, dtype=np.uint8))
    shuffled_labels = labels[t.randperm(len(labels), generator=t.Generator().manual_seed(0))]
    
    text_emb, image_emb, noise_emb = {}, {}, {}
    for kind, pair in pairs.items():
        text_emb[kind] = pair.embed_texts(captions_for(class_names))
        image_emb[kind] = pair.embed_images(pil_images)
        noise_emb[kind] = pair.embed_images(noise_images)
    
    mats, rows = {}, []
    for kind in pairs:
        for condition, emb, lab in [("real images", image_emb[kind], labels), ("shuffled labels (control)", image_emb[kind], shuffled_labels), ("noise images (control)", noise_emb[kind], labels)]:
            cos = similarity_logits(emb, text_emb[kind], 1.0)
            mats[f"{kind}: {condition}"] = class_means(cos, lab).numpy()
            r = retrieval_metrics(cos, lab, class_ids)
            rows.append({"model": kind, "condition": condition, "image-to-text R@1": r["image_to_text"], "text-to-image R@1": r["text_to_image"]})
    utils.plot_class_heatmaps(mats, class_names, captions_for(class_names, "{}"), "Mean cosine similarity between image classes (rows) and class captions (columns)")
    display(pd.DataFrame(rows).set_index(["model", "condition"]).round(3))

# %%

if MAIN:
    first_of_class = t.stack([(labels == c).nonzero()[0, 0] for c in range(10)])
    for kind, pair in pairs.items():
        bias = 0.0 if pair.logit_bias is None else pair.logit_bias
        logits = similarity_logits(image_emb[kind][first_of_class], text_emb[kind], pair.logit_scale, bias)
        wrong = logits.roll(1, dims=1)  # image i is now scored against caption i-1, the pairing on the diagonal is wrong
        if kind == "CLIP":
            print(f"CLIP   (N=10, log N = {math.log(10):.2f}):  InfoNCE loss with correct pairing {clip_loss(logits):7.3f},  with shifted pairing {clip_loss(wrong):7.3f}")
        else:
            pos, neg = siglip_loss_terms(logits)
            wpos, wneg = siglip_loss_terms(wrong)
            print(f"SigLIP (N=10):             sigmoid loss with correct pairing {pos + neg:7.3f} (positive {pos:.3f} + negative {neg:.3f}),  with shifted pairing {wpos + wneg:7.3f} (positive {wpos:.3f} + negative {wneg:.3f})")
        print(f"       the model's own logits as probabilities for image 0 (a {class_names[0]}):", ", ".join(f"{p:.2f}" for p in zero_shot_probabilities(logits, "softmax" if kind == "CLIP" else "sigmoid")[0, :4].tolist()), "...")

# %%

if MAIN:
    instruments = ["piano", "guitar", "violin", "trumpet", "drum", "flute", "harp", "saxophone", "cello", "banjo"]
    instrument_emb = {kind: pair.embed_texts(captions_for(instruments)) for kind, pair in pairs.items()}
    
    
    def max_probability(kind: str, images: Tensor, texts: Tensor, readout: str) -> float:
        pair = pairs[kind]
        bias = 0.0 if pair.logit_bias is None else pair.logit_bias
        probs = zero_shot_probabilities(similarity_logits(images, texts, pair.logit_scale, bias), readout)
        return probs.max(dim=-1).values.mean().item()
    
    
    readouts = {"CLIP, softmax": ("CLIP", "softmax"), "SigLIP, softmax": ("SigLIP", "softmax"), "SigLIP, sigmoid": ("SigLIP", "sigmoid")}
    conditions = {
        "real images,\nmatching captions": lambda k: (image_emb[k], text_emb[k]),
        "real images,\nunrelated captions": lambda k: (image_emb[k], instrument_emb[k]),
        "noise images,\nmatching captions": lambda k: (noise_emb[k], text_emb[k]),
    }
    confidence = {
        cond: {name: max_probability(kind, *get(kind), readout) for name, (kind, readout) in readouts.items()}
        for cond, get in conditions.items()
    }
    for cond, values in confidence.items():
        print(f"{cond.replace(chr(10), ' '):<36}" + "   ".join(f"{name}: {v:.3g}" for name, v in values.items()))
    utils.plot_bars(confidence, "Mean over images of the largest probability over the candidate captions", "mean max probability", log=True)

# %%

if MAIN:
    def classify_image(image: Image.Image, labels: list[str], template: str = "a photo of a {}") -> None:
        """Print each model's probability for every label."""
        captions = captions_for(labels, template)
        rows = []
        for kind, pair in pairs.items():
            bias = 0.0 if pair.logit_bias is None else pair.logit_bias
            logits = similarity_logits(pair.embed_images([image]), pair.embed_texts(captions), pair.logit_scale, bias)
            softmax = zero_shot_probabilities(logits, "softmax")[0].tolist()
            sigmoid = zero_shot_probabilities(logits, "sigmoid")[0].tolist() if kind == "SigLIP" else [float("nan")] * len(labels)
            rows.append(pd.DataFrame({f"{kind} softmax": softmax, **({"SigLIP sigmoid": sigmoid} if kind == "SigLIP" else {})}, index=labels))
        display(pd.concat(rows, axis=1).round(4))
    
    
    from sklearn.datasets import load_sample_images
    
    temple, flower = [Image.fromarray(x) for x in load_sample_images().images]
    utils.show_image_grid(np.stack([np.asarray(temple.resize((320, 214))), np.asarray(flower.resize((320, 214)))]), ["sample image 1", "sample image 2"], ncols=2, scale=3)
    classify_image(temple, ["temple", "castle", "car", "cat", "dog", "pagoda", "piano"])
    classify_image(flower, ["flower", "bee", "car", "cat", "dog", "garden", "piano"])

# %%

if MAIN:
    PATTERNS = ["plain", "striped", "dotted"]
    COMBOS = list(itertools.product(SHAPES, COLOURS, PATTERNS))  # (shape, colour, pattern); the combo index is the list index
    
    
    def combo_caption(shape: str, colour: str, pattern: str) -> str:
        return f"a {colour} {shape}" if pattern == "plain" else f"a {colour} {pattern} {shape}"
    
    
    rng = np.random.default_rng(0)
    N_RENDERS = 3
    factorial_images = [utils.render_large(shape, colour, rng, pattern) for _ in range(N_RENDERS) for shape, colour, pattern in COMBOS]
    image_combo = t.arange(len(COMBOS)).repeat(N_RENDERS)  # combo index of each of the 270 images
    attrs = {
        "shape": [c[0] for c in COMBOS],
        "colour": [c[1] for c in COMBOS],
        "pattern": [c[2] for c in COMBOS],
    }
    image_attrs = {name: [values[i] for i in image_combo.tolist()] for name, values in attrs.items()}
    
    show = [0, 17, 38, 49, 61, 85]
    utils.show_image_grid(
        np.stack([factorial_images[i] for i in show]),
        [combo_caption(*COMBOS[i]) for i in show],
        ncols=6,
        scale=2.3,
        title="Six of the 270 test images and their captions",
    )

# %%

if MAIN:
    fact = {}
    for kind, pair in pairs.items():
        fact[kind] = {
            "image": pair.embed_images(utils.to_pil(factorial_images)),
            "text": pair.embed_texts([combo_caption(*c) for c in COMBOS]),
        }
    
    rows = []
    for kind, pair in pairs.items():
        row = {"model": kind}
        sim = similarity_logits(fact[kind]["image"], fact[kind]["text"], 1.0)
        row["90-way retrieval (colour+shape+pattern)"] = retrieval_metrics(sim, image_combo, t.arange(90))["image_to_text"]
        for attr, template in [("colour", "a {} shape"), ("shape", "a {}"), ("pattern", "a {} shape")]:
            values = {"colour": COLOURS, "shape": SHAPES, "pattern": PATTERNS}[attr]
            texts = pair.embed_texts([template.format(v) if v != "plain" else "a shape" for v in values])
            sim = similarity_logits(fact[kind]["image"], texts, 1.0)
            truth = t.tensor([values.index(v) for v in image_attrs[attr]])
            row[f"{attr} only ({len(values)}-way)"] = retrieval_metrics(sim, truth, t.arange(len(values)))["image_to_text"]
        rows.append(row)
    display(pd.DataFrame(rows).set_index("model").round(3))
    print(f"chance: 90-way {1 / 90:.3f}, colour {1 / 6:.3f}, shape {1 / 5:.3f}, pattern {1 / 3:.3f}")

# %%

def modality_gap(image_emb: Float[Tensor, "n d"], text_emb: Float[Tensor, "m d"]) -> Float[Tensor, " d"]:
    """Mean of the normalised image embeddings minus mean of the normalised text embeddings."""
    return F.normalize(image_emb, dim=-1).mean(dim=0) - F.normalize(text_emb, dim=-1).mean(dim=0)


if MAIN:
    tests.test_modality_gap(modality_gap)

# %%

if MAIN:
    def gap_stats(image_emb: Tensor, text_emb: Tensor, image_to_text: Tensor) -> dict[str, float]:
        """image_to_text[i] = index of image i's own caption."""
        ui, vt = F.normalize(image_emb, dim=-1), F.normalize(text_emb, dim=-1)
        cos = ui @ vt.T
        own = t.zeros_like(cos, dtype=t.bool)
        own[t.arange(len(ui)), image_to_text] = True
        n_i, n_t = len(ui), len(vt)
        return {
            "gap norm": modality_gap(image_emb, text_emb).norm().item(),
            "matched cos": cos[own].mean().item(),
            "unmatched cos": cos[~own].mean().item(),
            "image-image cos": ((ui @ ui.T).sum() - n_i).item() / (n_i * (n_i - 1)),
            "text-text cos": ((vt @ vt.T).sum() - n_t).item() / (n_t * (n_t - 1)),
        }
    
    
    tiny_img_emb, tiny_txt_emb, gap_panels, gap_rows = {}, {}, {}, {}
    for name, m in [("tiny CLIP, untrained", untrained), ("tiny CLIP, trained", runs["CLIP loss, paired captions"][0])]:
        tiny_img_emb[name], tiny_txt_emb[name] = [x.cpu() for x in embed_all(m, test_images, CAPTION_IDS)]
        gap_rows[name] = gap_stats(tiny_img_emb[name], tiny_txt_emb[name], t.arange(120))
        gap_panels[name] = (F.normalize(tiny_img_emb[name], dim=-1).numpy(), F.normalize(tiny_txt_emb[name], dim=-1).numpy())
    for kind in pairs:
        gap_rows[kind] = gap_stats(fact[kind]["image"], fact[kind]["text"], image_combo)
        gap_panels[kind] = (F.normalize(fact[kind]["image"], dim=-1).numpy(), F.normalize(fact[kind]["text"], dim=-1).numpy())
    display(pd.DataFrame(gap_rows).T.round(3))
    utils.plot_modality_gap({k: gap_panels[k] for k in ["CLIP", "SigLIP", "tiny CLIP, untrained", "tiny CLIP, trained"]}, "Image and caption embeddings, seen along the modality-gap direction (X = centre of each cloud)")

# %%

if MAIN:
    temperature_models, temperature_panels = {}, {}
    temperature_rows = {"learned scale (about 19)": gap_rows["tiny CLIP, trained"] | {"i2t R@1": evaluate_tiny(runs["CLIP loss, paired captions"][0], test_images, CAPTION_IDS, held_out_mask)["i2t"]}}
    for scale in [1 / 0.07, 100.0]:
        m = fresh_model(scale_init=scale)
        m.logit_scale.requires_grad_(False)  # keep the temperature fixed during training
        train_tiny_clip(m, train_images, CAPTION_IDS, train_classes, "clip", "paired")
        img_e, txt_e = [x.cpu() for x in embed_all(m, test_images, CAPTION_IDS)]
        label = f"frozen scale {scale:.1f} (temperature {1 / scale:.2f})"
        temperature_rows[label] = gap_stats(img_e, txt_e, t.arange(120)) | {"i2t R@1": evaluate_tiny(m, test_images, CAPTION_IDS, held_out_mask)["i2t"]}
        temperature_panels[label] = (F.normalize(img_e, dim=-1).numpy(), F.normalize(txt_e, dim=-1).numpy())
    display(pd.DataFrame(temperature_rows).T[["gap norm", "matched cos", "unmatched cos", "image-image cos", "text-text cos", "i2t R@1"]].round(3))
    utils.plot_modality_gap(temperature_panels, "Tiny CLIPs trained with a frozen logit scale: the gap opens at CLIP's temperature")

# %%

def accuracy_after_shift(
    image_emb: Float[Tensor, "n d"], text_emb: Float[Tensor, "m d"], correct_text: Int[Tensor, " n"], shift: Float[Tensor, " d"]
) -> float:
    """Zero-shot top-1 accuracy after adding `shift` to the (normalised) image embeddings and re-normalising."""
    shifted = F.normalize(F.normalize(image_emb, dim=-1) + shift, dim=-1)
    predictions = (shifted @ F.normalize(text_emb, dim=-1).T).argmax(dim=-1)
    return (predictions == correct_text).float().mean().item()


if MAIN:
    tests.test_accuracy_after_shift(accuracy_after_shift)

# %%

if MAIN:
    shift_rows = []
    for kind in pairs:
        img, txt = fact[kind]["image"], fact[kind]["text"]
        gap = modality_gap(img, txt)
        row = {"model": kind, "no shift": accuracy_after_shift(img, txt, image_combo, t.zeros_like(gap))}
        for alpha in [0.5, 1.0]:
            row[f"close {alpha:.1f} of the gap"] = accuracy_after_shift(img, txt, image_combo, -alpha * gap)
        generator = t.Generator().manual_seed(0)
        random_accs = []
        for _ in range(20):
            direction = F.normalize(t.randn(gap.shape, generator=generator), dim=0)
            random_accs.append(accuracy_after_shift(img, txt, image_combo, direction * gap.norm()))
        row["random shift, same norm (mean of 20)"] = float(np.mean(random_accs))
        row["random shift, same norm (min..max)"] = f"{min(random_accs):.2f}..{max(random_accs):.2f}"
        shift_rows.append(row)
    display(pd.DataFrame(shift_rows).set_index("model").round(3))

# %%

def attribute_direction(emb: Float[Tensor, "n d"], values: list[str], a: str, b: str) -> Float[Tensor, " d"]:
    """Difference of means of the normalised embeddings: (rows with value b) minus (rows with value a)."""
    emb = F.normalize(emb, dim=-1)
    is_a = t.tensor([v == a for v in values])
    is_b = t.tensor([v == b for v in values])
    return emb[is_b].mean(dim=0) - emb[is_a].mean(dim=0)


if MAIN:
    tests.test_attribute_direction(attribute_direction)

# %%

if MAIN:
    def cosine(u: Tensor, v: Tensor) -> float:
        return F.cosine_similarity(u, v, dim=0).item()
    
    
    values_of = {"colour": COLOURS, "shape": SHAPES, "pattern": PATTERNS}
    generator = np.random.default_rng(0)
    agreement: dict[str, dict[str, dict[str, list[float]]]] = {}
    for kind in pairs:
        img, txt = fact[kind]["image"], fact[kind]["text"]
        agreement[kind] = {}
        for attr, values in values_of.items():
            other = "shape" if attr != "shape" else "colour"  # the 'different attribute' control uses the other attribute's pairs
            other_pairs = list(itertools.combinations(values_of[other], 2))
            shuffled = [image_attrs[attr][i] for i in generator.permutation(len(img))]
            real, shuffled_control, wrong_attr = [], [], []
            for k, (a, b) in enumerate(itertools.combinations(values, 2)):
                d_txt = attribute_direction(txt, attrs[attr], a, b)
                real.append(cosine(attribute_direction(img, image_attrs[attr], a, b), d_txt))
                shuffled_control.append(cosine(attribute_direction(img, shuffled, a, b), d_txt))
                c, d = other_pairs[k % len(other_pairs)]
                wrong_attr.append(cosine(attribute_direction(img, image_attrs[other], c, d), d_txt))
            agreement[kind][attr] = {"image vs caption direction": real, "shuffled image labels (control)": shuffled_control, "different attribute (control)": wrong_attr}
    rows = [
        {"model": kind, "attribute": attr, **{k: f"{np.mean(v):+.2f} [{min(v):+.2f}, {max(v):+.2f}]" for k, v in series.items()}}
        for kind, per_attr in agreement.items()
        for attr, series in per_attr.items()
    ]
    display(pd.DataFrame(rows).set_index(["model", "attribute"]))
    utils.plot_direction_agreement(agreement, "Do the image tower and the text tower agree on the direction of an attribute change?")

# %%

def steer_and_retrieve(
    image_emb: Float[Tensor, "n d"], direction: Float[Tensor, " d"], alpha: float, text_emb: Float[Tensor, "m d"]
) -> Int[Tensor, " n"]:
    """Index of the nearest text for each image after moving the image embedding by alpha * direction."""
    steered = F.normalize(F.normalize(image_emb, dim=-1) + alpha * direction, dim=-1)
    return (steered @ F.normalize(text_emb, dim=-1).T).argmax(dim=-1)


if MAIN:
    tests.test_steer_and_retrieve(steer_and_retrieve)

# %%

if MAIN:
    alphas = [round(0.25 * k, 2) for k in range(11)]
    red_images = [i for i, c in enumerate(image_attrs["colour"]) if c == "red"]
    target = {i: COMBOS.index((image_attrs["shape"][i], "blue", image_attrs["pattern"][i])) for i in red_images}
    
    
    def success_rate(kind: str, rows: list[int], direction: Tensor, alpha: float) -> float:
        pred = steer_and_retrieve(fact[kind]["image"][rows], direction, alpha, fact[kind]["text"])
        return (pred == t.tensor([target[i] for i in rows])).float().mean().item()
    
    
    steering = {}
    for kind in pairs:
        img, txt = fact[kind]["image"], fact[kind]["text"]
        d_img = attribute_direction(img, image_attrs["colour"], "red", "blue")
        d_txt = attribute_direction(txt, attrs["colour"], "red", "blue")
        train_rows = [i for i, s in enumerate(image_attrs["shape"]) if s in ("circle", "square", "triangle")]
        d_partial = attribute_direction(img[train_rows], [image_attrs["colour"][i] for i in train_rows], "red", "blue")
        d_shape = attribute_direction(img, image_attrs["shape"], "circle", "square")
        held_rows = [i for i in red_images if image_attrs["shape"][i] in ("star", "heart")]
        generator = t.Generator().manual_seed(0)
        random_dirs = [d_img.norm() * F.normalize(t.randn(d_img.shape, generator=generator), dim=0) for _ in range(20)]
        steering[kind] = {
            "image direction": (alphas, [success_rate(kind, red_images, d_img, a) for a in alphas]),
            "caption direction": (alphas, [success_rate(kind, red_images, d_txt, a) for a in alphas]),
            "held-out shapes": (alphas, [success_rate(kind, held_rows, d_partial, a) for a in alphas]),
            "random direction (same norm)": (alphas, [float(np.mean([success_rate(kind, red_images, r, a) for r in random_dirs])) for a in alphas]),
            "shape direction (wrong attribute)": (alphas, [success_rate(kind, red_images, d_shape, a) for a in alphas]),
        }
        best = {name: max(v) for name, (_, v) in steering[kind].items()}
        print(f"{kind:<7} |d_img| = {d_img.norm():.3f}, |d_txt| = {d_txt.norm():.3f}.  Best success over alpha: " + ", ".join(f"{n}: {v:.2f}" for n, v in best.items()))
    utils.plot_steering_curves(steering, "Steering red images towards 'blue': fraction whose nearest caption is the same shape and pattern in blue", "success rate")

# %%

if MAIN:
    @t.no_grad()
    def steer_image(kind: str, shape: str, from_colour: str, to_colour: str, pattern: str = "plain", strengths=(0.0, 0.5, 1.0, 1.5, 3.0), seed: int = 7) -> None:
        pair = pairs[kind]
        image = utils.render_large(shape, from_colour, np.random.default_rng(seed), pattern)
        emb = pair.embed_images([Image.fromarray(image)])
        direction = attribute_direction(fact[kind]["image"], image_attrs["colour"], from_colour, to_colour)
        plt.figure(figsize=(1.8, 1.8))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"{kind}: {from_colour} {shape} ({pattern})", fontsize=7)
        plt.show()
        captions = [combo_caption(*c) for c in COMBOS]
        for alpha in strengths:
            idx = steer_and_retrieve(emb, direction, alpha, fact[kind]["text"])[0].item()
            print(f"  alpha = {alpha:3.1f}  ->  {captions[idx]!r}")
    
    
    steer_image("SigLIP", "star", "yellow", "purple", "striped")

# %%

def variance_partition(emb: Float[Tensor, "n d"], factors: dict[str, list]) -> dict[str, float]:
    """eta-squared of each factor on the normalised embeddings, plus the leftover 'interaction' share."""
    emb = F.normalize(emb, dim=-1)
    centred = emb - emb.mean(dim=0)
    total = (centred**2).sum()
    shares = {}
    for name, labels in factors.items():
        between = 0.0
        for value in sorted(set(labels)):
            group = centred[t.tensor([label == value for label in labels])]
            between += len(group) * (group.mean(dim=0) ** 2).sum()
        shares[name] = (between / total).item()
    shares["interaction"] = 1 - sum(shares.values())
    return shares


if MAIN:
    tests.test_variance_partition(variance_partition)

# %%

if MAIN:
    OBJECTS = ["shirt", "jacket", "dress", "skirt", "coat", "hat"]
    CLOTH_COLOURS = ["red", "blue", "green", "black", "white", "yellow"]
    MATERIALS = ["denim", "leather", "wool", "silk", "cotton"]
    STYLES = ["vintage", "formal", "casual", "sporty"]
    clothes = list(itertools.product(OBJECTS, CLOTH_COLOURS, MATERIALS, STYLES))
    cloth_captions = [f"a {style} {colour} {material} {obj}" for obj, colour, material, style in clothes]
    cloth_factors = {
        "object": [c[0] for c in clothes],
        "colour": [c[1] for c in clothes],
        "material": [c[2] for c in clothes],
        "style": [c[3] for c in clothes],
    }
    print(len(cloth_captions), "captions, e.g.", cloth_captions[:2])
    
    cloth_emb = {kind: pair.embed_texts(cloth_captions) for kind, pair in pairs.items()}
    generator = np.random.default_rng(0)
    partition = {}
    for kind in pairs:
        real = variance_partition(cloth_emb[kind], cloth_factors)
        null = [
            variance_partition(cloth_emb[kind], {name: [labels[i] for i in generator.permutation(len(labels))] for name, labels in cloth_factors.items()})
            for _ in range(20)
        ]
        partition[kind] = real
        partition[f"{kind} (shuffled labels)"] = {k: float(np.mean([n[k] for n in null])) for k in real}
    groups = {
        ("interaction\n(what is left over)" if factor == "interaction" else factor): {name: shares[factor] for name, shares in partition.items()}
        for factor in ["object", "colour", "material", "style", "interaction"]
    }
    display(pd.DataFrame(partition).round(3))
    utils.plot_bars(groups, "Share of the variance of 720 clothing-caption embeddings explained by each factor", "share of variance")

# %%

def winoground_scores(
    s_aa: Float[Tensor, " n"], s_ab: Float[Tensor, " n"], s_ba: Float[Tensor, " n"], s_bb: Float[Tensor, " n"]
) -> dict[str, float]:
    """Winoground-style text, image and group scores, plus pooled pairwise accuracy. s_xy = score(image x, caption y)."""
    a_prefers_own, b_prefers_own = s_aa > s_ab, s_bb > s_ba  # each image prefers its own caption
    caption_a_prefers_own, caption_b_prefers_own = s_aa > s_ba, s_bb > s_ab  # each caption prefers its own image
    text = a_prefers_own & b_prefers_own
    image = caption_a_prefers_own & caption_b_prefers_own
    return {
        "text": text.float().mean().item(),
        "image": image.float().mean().item(),
        "group": (text & image).float().mean().item(),
        "pair_accuracy": 0.5 * (a_prefers_own.float().mean().item() + b_prefers_own.float().mean().item()),
    }


if MAIN:
    tests.test_winoground_scores(winoground_scores)

# %%

if MAIN:
    rng = np.random.default_rng(0)
    items = []
    for _ in range(150):
        s1, s2 = rng.choice(SHAPES, 2, replace=False)
        c1, c2 = rng.choice(COLOURS, 2, replace=False)
        c3 = rng.choice([c for c in COLOURS if c not in (c1, c2)])
        items.append((str(s1), str(c1), str(s2), str(c2), str(c3)))
    scene_a = [utils.render_large_pair((s1, c1), (s2, c2), rng) for s1, c1, s2, c2, _ in items]
    scene_b = [utils.render_large_pair((s1, c2), (s2, c1), rng) for s1, c1, s2, c2, _ in items]
    caption_a = [f"a {c1} {s1} and a {c2} {s2}" for s1, c1, s2, c2, _ in items]
    caption_b = [f"a {c2} {s1} and a {c1} {s2}" for s1, c1, s2, c2, _ in items]
    caption_c = [f"a {c3} {s1} and a {c2} {s2}" for s1, c1, s2, c2, c3 in items]  # control: a colour that isn't in the picture
    utils.show_image_grid(
        np.stack([scene_a[0], scene_b[0], scene_a[1], scene_b[1]]),
        [f"A: {caption_a[0]}", f"B: {caption_b[0]}", f"A: {caption_a[1]}", f"B: {caption_b[1]}"],
        ncols=4,
        scale=2.6,
        title="Two items of the synthetic Winoground (images A and B differ only by swapping the two colours)",
    )

# %%

if MAIN:
    def binding_report(img_a: Tensor, img_b: Tensor, cap_a: Tensor, cap_b: Tensor, cap_c: Tensor) -> dict[str, float]:
        """Winoground scores for swapped colours, plus pairwise accuracy on the 'wrong colour word' control."""
        cos = lambda x, y: (F.normalize(x, dim=-1) * F.normalize(y, dim=-1)).sum(-1)
        scores = winoground_scores(cos(img_a, cap_a), cos(img_a, cap_b), cos(img_b, cap_a), cos(img_b, cap_b))
        # how often do the two images choose the SAME caption (i.e. the choice ignores the image)?
        scores["same_caption_for_both"] = ((cos(img_a, cap_a) > cos(img_a, cap_b)) == (cos(img_b, cap_a) > cos(img_b, cap_b))).float().mean().item()
        scores["control_accuracy"] = (cos(img_a, cap_a) > cos(img_a, cap_c)).float().mean().item()
        scores["caption_cos_swap"] = cos(cap_a, cap_b).mean().item()
        scores["caption_cos_wrong_colour"] = cos(cap_a, cap_c).mean().item()
        return scores
    
    
    binding = {}
    for kind, pair in pairs.items():
        binding[kind] = binding_report(
            pair.embed_images(utils.to_pil(scene_a)), pair.embed_images(utils.to_pil(scene_b)),
            pair.embed_texts(caption_a), pair.embed_texts(caption_b), pair.embed_texts(caption_c),
        )
    display(pd.DataFrame(binding).T.rename(columns={
        "text": "text score", "image": "image score", "group": "group score", "pair_accuracy": "swap: pair accuracy", "same_caption_for_both": "both images pick the same caption",
        "control_accuracy": "wrong colour word: accuracy", "caption_cos_swap": "cos(caption A, swapped B)", "caption_cos_wrong_colour": "cos(caption A, wrong-colour C)",
    }).round(3))
    print("chance: text 0.25, image 0.25, group 0.167, pair accuracy 0.5")

# %%

if MAIN:
    TWO_CLASSES = [(c1, s1, c2, s2) for c1 in COLOURS for s1 in SHAPES for c2 in COLOURS for s2 in SHAPES if c1 != c2 and s1 != s2]
    TWO_INDEX = {cls: i for i, cls in enumerate(TWO_CLASSES)}
    swap_of = t.tensor([TWO_INDEX[(c2, s1, c1, s2)] for c1, s1, c2, s2 in TWO_CLASSES])  # colour-swapped twin of every class
    wrong_colour_of = t.tensor([TWO_INDEX[([c for c in COLOURS if c not in (c1, c2)][0], s1, c2, s2)] for c1, s1, c2, s2 in TWO_CLASSES])
    two_captions = [f"a {c1} {s1} and a {c2} {s2}" for c1, s1, c2, s2 in TWO_CLASSES]
    two_caption_ids = tokenize_captions(two_captions, VOCAB, MAX_LEN).to(device)
    
    twin_pairs = sorted({tuple(sorted((i, int(swap_of[i])))) for i in range(len(TWO_CLASSES))})  # 300 swap-closed pairs
    order = np.random.default_rng(0).permutation(len(twin_pairs))
    test_pairs = [twin_pairs[i] for i in order[: len(twin_pairs) // 5]]
    train_pairs = [twin_pairs[i] for i in order[len(twin_pairs) // 5 :]]
    two_train_classes = t.tensor(sorted(i for pair in train_pairs for i in pair))
    
    
    def render_two_object_world(n_variants: int, seed: int) -> Tensor:
        rng = np.random.default_rng(seed)
        images = np.stack([[utils.render_tiny_pair((s1, c1), (s2, c2), rng) for c1, s1, c2, s2 in TWO_CLASSES] for _ in range(n_variants)])
        return t.tensor(images).permute(0, 1, 4, 2, 3).float() / 255
    
    
    two_train_images = render_two_object_world(8, seed=0).to(device)
    two_test_images = render_two_object_world(1, seed=1)[0].to(device)
    print(f"{len(TWO_CLASSES)} classes in {len(twin_pairs)} swap pairs: {len(train_pairs)} pairs for training, {len(test_pairs)} held-out pairs for testing")
    print("caption example:", two_captions[0], "<->", two_captions[int(swap_of[0])], "(swapped colours)")

# %%

if MAIN:
    def tiny_binding(model: TinyCLIP, pairs_to_test: list[tuple[int, int]]) -> dict[str, float]:
        """Winoground scores of a tiny two-object model on the given (class A, swapped class B) pairs, with the wrong-colour control."""
        a = t.tensor([p[0] for p in pairs_to_test])
        b = swap_of[a]
        c = wrong_colour_of[a]
        img_a, _ = embed_all(model, two_test_images[a.to(device)], two_caption_ids[:1])
        img_b, _ = embed_all(model, two_test_images[b.to(device)], two_caption_ids[:1])
        _, cap_all = embed_all(model, two_test_images[:1], two_caption_ids)
        return binding_report(img_a, img_b, cap_all[a.to(device)], cap_all[b.to(device)], cap_all[c.to(device)])
    
    
    two_runs = {}
    for name, trained in [("tiny CLIP, trained on two-object scenes", True), ("tiny CLIP, untrained", False)]:
        m = fresh_model()
        if trained:
            hist = train_tiny_clip(m, two_train_images, two_caption_ids, two_train_classes, "clip", "paired", steps=2000)
            print(f"trained for 2000 steps, final loss {np.mean(hist['loss'][-50:]):.3f}")
        two_runs[name] = tiny_binding(m, test_pairs)
    display(pd.DataFrame(two_runs).T.round(3))

# %%

if MAIN:
    summary = {
        "CLIP ViT-B/32": binding["CLIP"],
        "SigLIP base": binding["SigLIP"],
        "tiny CLIP, trained\non two-object scenes": two_runs["tiny CLIP, trained on two-object scenes"],
        "tiny CLIP,\nuntrained": two_runs["tiny CLIP, untrained"],
    }
    utils.plot_bars(
        {name: {"swapped colours (binding)": v["pair_accuracy"], "wrong colour word (control)": v["control_accuracy"]} for name, v in summary.items()},
        "Does the image prefer its own caption? Binding test versus easy control (held-out items)",
        "fraction of images preferring the correct caption",
        chance=0.5,
    )

# %%

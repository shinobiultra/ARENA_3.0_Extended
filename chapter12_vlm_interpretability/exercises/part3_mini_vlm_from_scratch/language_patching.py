"""Stage-aware visual interventions during actual autoregressive generation."""
from __future__ import annotations

import torch as t
from torch import nn

from .language_vlm import ActivationPatch, greedy_word_generation


def visual_rows_from_pixel_mask(pixel_mask: t.Tensor, *, patch_size: int) -> t.Tensor:
    """Mark row-major patch tokens that overlap any selected image pixel."""
    if pixel_mask.ndim != 3 or pixel_mask.dtype != t.bool:
        raise ValueError("Pixel masks must be boolean [batch, height, width]")
    batch, height, width = pixel_mask.shape
    if patch_size < 1 or height % patch_size or width % patch_size:
        raise ValueError("Patch size must divide both image dimensions")
    patches = pixel_mask.reshape(batch, height // patch_size, patch_size, width // patch_size, patch_size)
    return patches.permute(0, 1, 3, 2, 4).flatten(3).any(-1).flatten(1)


def sample_visual_row_controls(object_rows: t.Tensor, *, samples: int = 8, seed: int = 701):
    """Sample equal-size background and unrestricted row sets without filtering outcomes."""
    if object_rows.ndim != 2 or object_rows.dtype != t.bool or samples < 1:
        raise ValueError("Use boolean [batch, patch] masks and positive sample count")
    generator = t.Generator().manual_seed(seed)
    result = {"background": [], "unrestricted_random": []}
    for _ in range(samples):
        background = t.zeros_like(object_rows)
        random = t.zeros_like(object_rows)
        for row, selected in enumerate(object_rows.cpu()):
            count = int(selected.sum())
            available = (~selected).nonzero().flatten()
            if count < 1 or len(available) < count:
                raise ValueError("Each example needs enough background rows for the same-size control")
            choices = available[t.randperm(len(available), generator=generator)[:count]]
            background[row, choices.to(background.device)] = True
            choices = t.randperm(len(selected), generator=generator)[:count]
            random[row, choices.to(random.device)] = True
        result["background"].append(background)
        result["unrestricted_random"].append(random)
    return result


def generate_with_visual_patch(model, clean_pixels, donor_pixels, prompt_ids, prompt_valid, *,
                               stage: str, visual_rows: t.Tensor, full_sequence: bool = False,
                               max_new_tokens: int = 8):
    """Recompute donor activations on the same current word prefix at every step.

    Visual masks index patch rows only. Later decoder stages translate these
    to their positions between image markers. full_sequence also replaces
    marker and text states, which may already contain image information.
    """
    if clean_pixels.shape != donor_pixels.shape:
        raise ValueError("Clean and donor image batches must have matching shapes")
    if visual_rows.shape != (len(clean_pixels), model.num_visual_tokens) or visual_rows.dtype != t.bool:
        raise ValueError("Select visual patch rows with a boolean batch mask")
    if visual_rows.device != clean_pixels.device or donor_pixels.device != clean_pixels.device:
        raise ValueError("Images and visual row masks must share a device")

    class PatchedGenerationModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.base = model
            self.train(model.training)

        def forward(self, pixels, ids, valid):
            _, donor_cache = self.base(donor_pixels, ids, valid, return_cache=True)
            if stage not in donor_cache:
                raise ValueError("Unknown cached intervention stage")
            donor = donor_cache[stage]
            if full_sequence:
                rows = t.ones(donor.shape[:2], dtype=t.bool, device=donor.device)
            elif stage.startswith("decoder_"):
                rows = t.zeros(donor.shape[:2], dtype=t.bool, device=donor.device)
                rows[:, 1:1 + self.base.num_visual_tokens] = visual_rows
            else:
                rows = visual_rows
            return self.base(pixels, ids, valid, patch=ActivationPatch(stage, rows, donor))

    return greedy_word_generation(PatchedGenerationModel(), clean_pixels, prompt_ids, prompt_valid,
                                  max_new_tokens=max_new_tokens)


def conditional_targeted_flips(clean_correct, donor_correct, patched_is_donor):
    """Report the declared joint-correct denominator and keep unconditional counts."""
    tensors = (clean_correct, donor_correct, patched_is_donor)
    if any(value.dtype != t.bool or value.ndim != 1 for value in tensors):
        raise ValueError("Use one boolean outcome per example")
    if len({tuple(value.shape) for value in tensors}) != 1:
        raise ValueError("Outcome vectors must describe the same examples")
    eligible = clean_correct & donor_correct
    count = int(eligible.sum())
    flips = int((eligible & patched_is_donor).sum())
    return {"examples": len(clean_correct), "eligible": count, "targeted_flips": flips,
            "conditional_flip_rate": flips / count if count else None,
            "unconditional_donor_answer_rate": float(patched_is_donor.float().mean())}

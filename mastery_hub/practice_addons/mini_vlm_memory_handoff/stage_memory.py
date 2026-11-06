"""Optional visible memory plumbing for the boundary before MiniVLM section 5.

Imports no numerical/model packages and never selects an accelerator. Call only
from an existing lesson namespace, which already supplies torch (`t`) and `nn`.
The helper changes device placement, not the taught methods or saved evidence.
"""
from __future__ import annotations


def move_minivlm_stage(namespace, target_device="cpu"):
    """Move only the earlier MiniVLM stage; retain handles for replay and play cells.

    The explicit owner list deliberately excludes Qwen, even if this cell is rerun.
    Tensor aliases shared by a cache and a play input remain aliases after moving.
    """
    t, nn = namespace["t"], namespace["nn"]
    selected_qwen = namespace.get("qwen")
    owners = (
        "gpt2", "lm_a", "vlm", "siglip", "projector_a", "projector",
        "untrained_projector", "random_projector", "text_only_lm",
        "train_features", "test_features", "blank_features", "noise_features", "shuffled_features",
        "clean_features", "corrupt_features", "a_cache", "clean_cache", "clean_resids",
        "a_embeds", "image_resid", "clean_embeds", "corrupt_embeds",
        "a_tokens", "a_lens", "a_logits", "a_target_colour", "colour_ids", "a_masks",
        "pair_tokens", "pair_lens", "clean_ids", "corrupt_ids", "clean_logits", "corrupt_logits",
        "p_masks", "patch_sets", "lens_stats", "mass", "per_head", "patterns",
        "probs", "word_id", "obj_mask", "bg_mask", "resid", "noise", "ranks", "correct",
        "rows", "tok", "p", "i", "v", "m", "in_obj", "in_bg", "block", "_",
    )
    moved = {}

    def move(value):
        identity = id(value)
        if identity in moved:
            return moved[identity]
        if isinstance(value, nn.Module):
            # Protect the selected new model even if a learner reused an old name.
            if value is selected_qwen:
                return value
            result = value.to(target_device)
        elif isinstance(value, t.Tensor):
            result = value.detach().to(target_device)
        elif isinstance(value, dict):
            moved[identity] = value
            for key, item in list(value.items()):
                value[key] = move(item)
            return value
        elif isinstance(value, list):
            moved[identity] = value
            value[:] = [move(item) for item in value]
            return value
        elif isinstance(value, tuple):
            result = tuple(move(item) for item in value)
        else:
            return value
        moved[identity] = result
        return result

    for name in owners:
        if name not in namespace:
            continue
        value = namespace[name]
        if name in ("a_cache", "clean_cache"):
            # TransformerLens ActivationCache owns a dict as well as its model.
            # The model is moved by its own named handle above, without replacing it.
            move(value.cache_dict)
        else:
            namespace[name] = move(value)

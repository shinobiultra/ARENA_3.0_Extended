"""Read-only aliases for exact existing metadata required by eager Python defaults.

The unchanged legacy module defines data-only standard dataclasses and literal
prompt records at import. It performs no model, tensor, random or runtime calls.
No metadata is copied, normalized or mutated here: class, list and record identity
remain identical before and after explicit notebook plumbing initialization.
"""
from chapter9_alignment_interpretability.exercises.part5_rome_model_editing.solutions import (
    PromptRecord,
    HELDOUT_PARAPHRASE_RECORDS,
)

__all__ = ["PromptRecord", "HELDOUT_PARAPHRASE_RECORDS"]

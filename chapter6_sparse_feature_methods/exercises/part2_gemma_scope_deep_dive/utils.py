"""Deterministic data plumbing for the Gemma Scope deep-dive exercises."""

from __future__ import annotations

import math
from pathlib import Path

import torch as t


FEATURE_NAMES = (
    "technical concept",
    "train-only formatting confound",
    "shared prose",
    "rare phrase",
    "matched random control",
    "dead feature",
)

TOKEN_NAMES = (" code", " story", " function", " river", " neutral", " syntax")

POSITIVE_TEXTS = (
    "def parse_config(data):",
    "multiply the matrix by W",
    "HTTP status 404",
    "validate the JSON schema",
    "mask padding tokens",
    "compute cosine similarity",
    "assert tensor.shape == expected",
    "cache the residual stream",
    "decode sparse activations",
    "normalize each decoder row",
    "rank documents by embedding",
    "apply a forward hook",
    "measure reconstruction error",
    "load a safetensors artifact",
    "broadcast the attention mask",
    "run the CPU unit test",
    "threshold the preactivation",
    "project through the unembedding",
    "ablate one learned feature",
    "sweep the steering coefficient",
)

NEGATIVE_TEXTS = (
    "the lantern glowed softly",
    "rain crossed the garden",
    "a child watched the clouds",
    "sunset warmed the lake",
    "fresh bread cooled nearby",
    "friends listened to music",
    "waves reached the beach",
    "snow covered the path",
    "the old clock chimed",
    "leaves moved in the wind",
    "a boat drifted downstream",
    "morning light filled the room",
    "the market opened early",
    "tea steamed by the window",
    "stars appeared above the hill",
    "a painter cleaned her brushes",
    "the train crossed the valley",
    "footsteps faded in the hall",
    "a violin played next door",
    "the moon rose over the roofs",
)


def _dct_decoder(d_model: int = 6) -> t.Tensor:
    """Return a deterministic orthonormal DCT-II basis as decoder rows."""

    columns = t.arange(d_model, dtype=t.float64) + 0.5
    rows = []
    for feature_id in range(d_model):
        scale = math.sqrt(1 / d_model) if feature_id == 0 else math.sqrt(2 / d_model)
        rows.append(scale * t.cos(math.pi * feature_id * columns / d_model))
    return t.stack(rows).float()


def make_ground_truth_organism() -> dict[str, object]:
    """Build a sparse-feature system whose latents and causal effects are known.

    The raw artifact uses unequal decoder norms. Its encoder columns, encoder
    biases, and thresholds are inversely scaled, so normalizing the dictionary
    is an exactly function-preserving coordinate change.
    """

    canonical_decoder = _dct_decoder()
    raw_decoder_norms = t.tensor([0.5, 2.0, 1.5, 0.75, 1.25, 3.0])
    threshold = t.full((6,), 0.2)
    b_dec = t.tensor([0.05, -0.03, 0.02, 0.00, 0.04, -0.01])

    raw_artifact = {
        "w_enc": canonical_decoder.T / raw_decoder_norms,
        "w_dec": canonical_decoder * raw_decoder_norms[:, None],
        "b_enc": t.zeros(6),
        "b_dec": b_dec,
        "threshold": threshold / raw_decoder_norms,
    }

    texts: list[str] = []
    labels: list[bool] = []
    train_mask: list[bool] = []
    latent_rows: list[list[float]] = []
    rare_indices = {5, 22, 37}

    for pair_id, (positive, negative) in enumerate(zip(POSITIVE_TEXTS, NEGATIVE_TEXTS)):
        for is_positive, text in ((True, positive), (False, negative)):
            row_id = len(texts)
            is_train = pair_id < 8
            concept = 1.0 + 0.1 * (pair_id % 4) if is_positive else 0.0
            if is_train:
                confound = 0.9 if is_positive else 0.0
            else:
                confound = 0.8 if pair_id % 2 == 0 else 0.0
            shared = 0.65 if pair_id % 3 != 0 else 0.0
            rare = 1.1 if row_id in rare_indices else 0.0
            random_control = 0.75 if pair_id % 2 == 1 else 0.0
            latent_rows.append([concept, confound, shared, rare, random_control, 0.0])
            texts.append(text)
            labels.append(is_positive)
            train_mask.append(is_train)

    latent_codes = t.tensor(latent_rows)
    residuals = latent_codes @ canonical_decoder + b_dec
    labels_tensor = t.tensor(labels, dtype=t.bool)
    train_mask_tensor = t.tensor(train_mask, dtype=t.bool)
    shuffled_labels = labels_tensor.clone()
    shuffled_labels[~train_mask_tensor] = False
    # The held-out shuffle assigns each class six zero scores and balances the
    # rank sum of the four non-zero score levels exactly.
    shuffled_positive_pairs = {8, 9, 10, 11, 12, 15}
    shuffled_zero_pairs = {8, 9, 10, 11, 12, 13}
    for pair_id in shuffled_positive_pairs:
        shuffled_labels[2 * pair_id] = True
    for pair_id in shuffled_zero_pairs:
        shuffled_labels[2 * pair_id + 1] = True

    target = canonical_decoder[0]
    syntax = canonical_decoder[1]
    control = canonical_decoder[4]
    unembedding = t.stack(
        (
            1.2 * target,
            -1.0 * target,
            0.9 * target + 0.1 * syntax,
            -0.7 * target,
            control,
            syntax,
        ),
        dim=1,
    )

    return {
        "raw_artifact": raw_artifact,
        "canonical_decoder": canonical_decoder,
        "raw_decoder_norms": raw_decoder_norms,
        "feature_names": FEATURE_NAMES,
        "token_names": TOKEN_NAMES,
        "texts": texts,
        "labels": labels_tensor,
        "train_mask": train_mask_tensor,
        "shuffled_labels": shuffled_labels,
        "latent_codes": latent_codes,
        "residuals": residuals,
        "unembedding": unembedding,
        "target_feature_id": 0,
        "control_feature_id": 4,
    }


def load_pinned_real_evidence(section_dir: Path) -> dict[str, object]:
    """Load and strictly validate the committed real-model evidence."""

    import json

    report = json.loads((section_dir / "verification_report.json").read_text())
    gpu = report["metrics"]["gpu_test"]
    assert gpu["gemma_scope_repo_id"] == "google/gemma-scope-2-1b-it"
    assert gpu["gemma_scope_revision"] == "b0fa29457c3601df0a70c48a15534c738d7c10e0"
    assert gpu["gemma_scope_artifact_path"] == "resid_post/layer_13_width_16k_l0_small"
    assert gpu["gemma_scope_real_activation_model_id"] == "google/gemma-3-1b-it"
    assert gpu["gemma_scope_real_activation_preflight_passed"]
    return gpu

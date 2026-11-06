# %%


MAIN = __name__ == "__main__"

# %%

import json
import os
import sys
from pathlib import Path

if MAIN:
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import matplotlib.pyplot as plt
import numpy as np
import torch as t
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

chapter = "chapter5_modern_architectures"
section = "part2_gemma_scope_feature_steering"
root_dir = Path(__file__).resolve().parents[3]
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
assets_dir = root_dir / "chapter6_sparse_feature_methods" / "exercises" / "part2_gemma_scope" / "media"
for path in (root_dir, exercises_dir):
    if str(path) not in sys.path:
        sys.path.append(str(path))

import chapter5_modern_architectures.exercises.part2_gemma_scope_feature_steering.tests as tests

if MAIN:
    t.set_grad_enabled(False)
if MAIN:
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "font.size": 10,
    })

TARGET = "#B43A32"
FEATURE_CONTROL = "#167D8D"
DIRECTION_CONTROL = "#5F6875"
SHUFFLE_CONTROL = "#D39A2C"

# %%

FEATURE_NAMES = [
    "technical", "narrative", "positive", "question",
    "number", "first_person", "past_tense", "instruction",
]

TECHNICAL_BASES = [
    "A parser validates a JSON object", "A tensor stores a matrix product",
    "A compiler lowers a function", "An API returns a typed response",
    "A unit test checks a decoder", "A kernel multiplies two arrays",
    "A tokenizer masks padding", "A gradient updates the weights",
    "A cache stores attention keys", "A schema constrains tool arguments",
    "A sparse code reconstructs a residual", "A hook captures layer output",
    "A benchmark measures latency", "A debugger traces the call stack",
    "A batch contains token sequences", "A loss compares model logits",
    "A vector index retrieves documents", "A dtype controls precision",
    "A checkpoint stores parameters", "A test isolates a regression",
]
NARRATIVE_BASES = [
    "A lantern glows beside the river", "A child watches clouds cross the hill",
    "The garden smells of rain", "A boat drifts across the lake",
    "Fresh bread cools by the window", "Friends listen to music in the park",
    "Snow settles on the empty road", "Waves reach the beach at dusk",
    "A traveler rests beneath an oak", "Moonlight falls across the porch",
    "A painter opens a blue umbrella", "The village wakes before sunrise",
    "A kettle whistles in the kitchen", "Leaves turn slowly in the wind",
    "A violin echoes through the hall", "Candles flicker near the doorway",
    "A dog sleeps beside the fire", "Footsteps fade along the path",
    "A letter waits on the table", "Lanterns shine during the festival",
]

def build_exact_organism() -> dict[str, object]:
    rows = []
    sparse_rows = []
    for label, bases in ((1, TECHNICAL_BASES), (0, NARRATIVE_BASES)):
        for index, base in enumerate(bases):
            text = base
            if index % 4 >= 2:
                text = "I remember that " + text[0].lower() + text[1:]
            if index % 4 < 2:
                text += "?"
            else:
                text += "."
            if index % 3 == 0:
                text += f" It mentions {index + 1} details."
            if index % 5 < 2:
                text += " It worked well."
            if index % 5 in (2, 3):
                text += " Please explain."

            features = t.zeros(len(FEATURE_NAMES))
            features[0] = (1.0 + 0.1 * (index % 4)) * label
            features[1] = (1.0 + 0.1 * (index % 4)) * (1 - label)
            nuisance_active = [
                index % 5 < 2,
                index % 4 < 2,
                index % 3 == 0,
                index % 4 >= 2,
                index % 5 < 2,
                index % 5 in (2, 3),
            ]
            for feature_id, active in enumerate(nuisance_active, start=2):
                if active:
                    features[feature_id] = 0.65 + 0.05 * (index % 3)
            rows.append({
                "text": text,
                "label": label,
                "split": "discovery" if index < 8 else "heldout",
            })
            sparse_rows.append(features)

    generator = t.Generator().manual_seed(5)
    basis, _ = t.linalg.qr(t.randn(12, len(FEATURE_NAMES), generator=generator))
    w_dec = basis.T.contiguous()
    w_enc = w_dec.T.contiguous()
    b_dec = t.linspace(-0.15, 0.15, 12)
    b_enc = t.zeros(len(FEATURE_NAMES))
    threshold = t.full((len(FEATURE_NAMES),), 0.25)
    ground_truth = t.stack(sparse_rows)
    residuals = ground_truth @ w_dec + b_dec
    return {
        "rows": rows,
        "ground_truth": ground_truth,
        "residuals": residuals,
        "w_enc": w_enc,
        "w_dec": w_dec,
        "b_enc": b_enc,
        "b_dec": b_dec,
        "threshold": threshold,
    }

if MAIN:
    organism = build_exact_organism()
if MAIN:
    print("Cold-open examples from the exact organism:")
if MAIN:
    for row in organism["rows"][:2] + organism["rows"][20:22]:
        print(f"  label={row['label']} split={row['split']:<9} | {row['text']}")
if MAIN:
    print("Ground-truth sparse-code shape:", tuple(organism["ground_truth"].shape))

# %%

def jumprelu_encode(
    x: t.Tensor,
    *,
    w_enc: t.Tensor,
    b_enc: t.Tensor,
    b_dec: t.Tensor,
    threshold: t.Tensor,
) -> t.Tensor:
    """Encode residual vectors with the JumpReLU SAE equations."""
    pre = (x.float() - b_dec.float()) @ w_enc.float() + b_enc.float()
    return pre * (pre > threshold.float())


def sae_decode(
    feature_acts: t.Tensor,
    *,
    w_dec: t.Tensor,
    b_dec: t.Tensor,
) -> t.Tensor:
    """Decode sparse feature activations into the residual stream."""
    return feature_acts.float() @ w_dec.float() + b_dec.float()
if MAIN:
    tests.test_jumprelu_encode_decode(jumprelu_encode, sae_decode)

if MAIN:
    exact_feature_acts = jumprelu_encode(
        organism["residuals"],
        w_enc=organism["w_enc"],
        b_enc=organism["b_enc"],
        b_dec=organism["b_dec"],
        threshold=organism["threshold"],
    )
if MAIN:
    exact_reconstruction = sae_decode(
        exact_feature_acts,
        w_dec=organism["w_dec"],
        b_dec=organism["b_dec"],
    )
if MAIN:
    code_error = (exact_feature_acts - organism["ground_truth"]).abs().max().item()
if MAIN:
    reconstruction_error = (exact_reconstruction - organism["residuals"]).abs().max().item()
if MAIN:
    print(f"Maximum code error: {code_error:.2e}")
if MAIN:
    print(f"Maximum reconstruction error: {reconstruction_error:.2e}")

# %%

def select_feature_by_mean_difference(
    feature_scores: t.Tensor,
    labels: t.Tensor,
    discovery_mask: t.Tensor,
) -> int:
    """Choose a feature using discovery rows only."""
    feature_scores = feature_scores.float()
    labels = labels.bool()
    discovery_mask = discovery_mask.bool()
    discovery_scores = feature_scores[discovery_mask]
    discovery_labels = labels[discovery_mask]
    if not bool(discovery_labels.any()) or not bool((~discovery_labels).any()):
        raise ValueError("Discovery data needs positive and negative examples.")
    margin = (
        discovery_scores[discovery_labels].mean(dim=0)
        - discovery_scores[~discovery_labels].mean(dim=0)
    )
    return int(margin.argmax().item())


def binary_roc_auc(scores: t.Tensor, labels: t.Tensor) -> float:
    """Return binary ROC AUC with average ranks for ties."""
    scores = scores.detach().flatten().float()
    labels = labels.detach().flatten().bool()
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must have equal length.")
    n_pos = int(labels.sum().item())
    n_neg = int((~labels).sum().item())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs both classes.")
    order = scores.argsort()
    sorted_scores = scores[order]
    ranks_sorted = t.empty_like(sorted_scores)
    _, counts = t.unique_consecutive(sorted_scores, return_counts=True)
    start = 0
    for count in counts.tolist():
        end = start + count
        ranks_sorted[start:end] = (start + 1 + end) / 2
        start = end
    ranks = t.empty_like(scores)
    ranks[order] = ranks_sorted
    numerator = ranks[labels].sum() - n_pos * (n_pos + 1) / 2
    return float((numerator / (n_pos * n_neg)).item())


def topk_example_indices(scores: t.Tensor, k: int = 5) -> t.Tensor:
    """Return indices of the k highest-scoring examples."""
    flat_scores = scores.detach().flatten()
    return flat_scores.topk(min(k, flat_scores.numel())).indices
if MAIN:
    tests.test_feature_discovery_and_heldout_scoring(
        select_feature_by_mean_difference,
        binary_roc_auc,
        topk_example_indices,
    )

# %%

if MAIN:
    exact_labels = t.tensor([row["label"] for row in organism["rows"]], dtype=t.bool)
if MAIN:
    exact_discovery = t.tensor(
        [row["split"] == "discovery" for row in organism["rows"]], dtype=t.bool
    )
# Preserve the organism's declared ties after float32 encode/decode noise.
if MAIN:
    exact_scoring_acts = exact_feature_acts.round(decimals=5)
if MAIN:
    selected_feature_id = select_feature_by_mean_difference(
        exact_scoring_acts, exact_labels, exact_discovery
    )
if MAIN:
    exact_heldout = ~exact_discovery
if MAIN:
    heldout_scores = exact_scoring_acts[exact_heldout, selected_feature_id]
if MAIN:
    heldout_labels = exact_labels[exact_heldout]
if MAIN:
    heldout_rows = [row for row in organism["rows"] if row["split"] == "heldout"]
if MAIN:
    heldout_auc = binary_roc_auc(heldout_scores, heldout_labels)
if MAIN:
    random_feature_ids = list(range(2, len(FEATURE_NAMES)))
if MAIN:
    random_feature_aucs = [
        binary_roc_auc(exact_scoring_acts[exact_heldout, feature_id], heldout_labels)
        for feature_id in random_feature_ids
    ]

if MAIN:
    print(f"Selected feature: {selected_feature_id} ({FEATURE_NAMES[selected_feature_id]})")
if MAIN:
    print(f"Held-out AUC: {heldout_auc:.3f} across {int(exact_heldout.sum())} examples")
if MAIN:
    print("Random-feature AUCs:", [round(value, 3) for value in random_feature_aucs])
if MAIN:
    print("\nTop held-out examples:")
if MAIN:
    for index in topk_example_indices(heldout_scores, k=5).tolist():
        row = heldout_rows[index]
        print(f"  score={heldout_scores[index]:.2f} label={row['label']} | {row['text']}")
if MAIN:
    print("\nLowest held-out examples:")
if MAIN:
    for index in heldout_scores.argsort()[:3].tolist():
        row = heldout_rows[index]
        print(f"  score={heldout_scores[index]:.2f} label={row['label']} | {row['text']}")

# %%

def steer_residual(
    residual: t.Tensor,
    direction: t.Tensor,
    coefficient: float,
) -> t.Tensor:
    """Add a decoder direction to every residual vector."""
    return residual + float(coefficient) * direction.to(residual)


def ablate_feature(
    residual: t.Tensor,
    feature_acts: t.Tensor,
    decoder_vectors: t.Tensor,
    feature_id: int,
    fraction: float = 1.0,
) -> t.Tensor:
    """Subtract a fraction of one feature's decoded contribution."""
    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("fraction must lie in [0, 1].")
    contribution = (
        feature_acts[..., feature_id, None].to(residual)
        * decoder_vectors[feature_id].to(residual)
    )
    return residual - float(fraction) * contribution


def direction_score(residual: t.Tensor, direction: t.Tensor) -> t.Tensor:
    """Project residual vectors onto a direction."""
    return (residual.float() * direction.float()).sum(dim=-1)
if MAIN:
    tests.test_feature_steering_and_ablation(
        steer_residual,
        ablate_feature,
        direction_score,
    )

# %%

if MAIN:
    shuffle_generator = t.Generator().manual_seed(17)
shuffled_aucs = []
if MAIN:
    for _ in range(200):
        permutation = t.randperm(heldout_labels.numel(), generator=shuffle_generator)
        shuffled_aucs.append(binary_roc_auc(heldout_scores, heldout_labels[permutation]))

control_feature_id = 5
if MAIN:
    target_direction = organism["w_dec"][selected_feature_id]
if MAIN:
    control_feature_direction = organism["w_dec"][control_feature_id]
if MAIN:
    random_generator = t.Generator().manual_seed(23)
if MAIN:
    random_direction = t.randn(target_direction.shape, generator=random_generator)
if MAIN:
    random_direction = random_direction / random_direction.norm() * target_direction.norm()

def exact_target_score(residual: t.Tensor) -> t.Tensor:
    return direction_score(residual - organism["b_dec"], target_direction)

if MAIN:
    negative_mask = exact_heldout & ~exact_labels
if MAIN:
    positive_mask = exact_heldout & exact_labels
if MAIN:
    coefficients = t.linspace(-2.0, 2.0, 9)
steering_target = []
steering_feature_control = []
steering_direction_control = []
if MAIN:
    for coefficient in coefficients.tolist():
        base = organism["residuals"][negative_mask]
        steering_target.append(exact_target_score(
            steer_residual(base, target_direction, coefficient)
        ).mean().item())
        steering_feature_control.append(exact_target_score(
            steer_residual(base, control_feature_direction, coefficient)
        ).mean().item())
        steering_direction_control.append(exact_target_score(
            steer_residual(base, random_direction, coefficient)
        ).mean().item())

if MAIN:
    fractions = t.linspace(0.0, 1.0, 6)
ablation_target = []
ablation_control = []
if MAIN:
    positive_residuals = organism["residuals"][positive_mask]
if MAIN:
    positive_acts = exact_scoring_acts[positive_mask]
if MAIN:
    for fraction in fractions.tolist():
        ablation_target.append(exact_target_score(
            ablate_feature(
                positive_residuals, positive_acts, organism["w_dec"],
                selected_feature_id, fraction,
            )
        ).mean().item())
        ablation_control.append(exact_target_score(
            ablate_feature(
                positive_residuals, positive_acts, organism["w_dec"],
                control_feature_id, fraction,
            )
        ).mean().item())

if MAIN:
    assert heldout_auc == 1.0
if MAIN:
    assert max(abs(value - 0.5) for value in random_feature_aucs) < 1e-6
if MAIN:
    assert steering_target[-1] - steering_target[4] > abs(
        steering_direction_control[-1] - steering_direction_control[4]
    )
if MAIN:
    assert ablation_target[-1] < 1e-5
if MAIN:
    assert abs(ablation_control[-1] - ablation_control[0]) < 1e-5

if MAIN:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
if MAIN:
    auc_values = [heldout_auc, float(np.mean(random_feature_aucs)), float(np.mean(shuffled_aucs))]
if MAIN:
    axes[0].bar(
        ["selected\nfeature", "6 nuisance\nfeatures", "200 label\nshuffles"],
        auc_values,
        color=[TARGET, FEATURE_CONTROL, SHUFFLE_CONTROL],
    )
if MAIN:
    axes[0].axhline(0.5, color=DIRECTION_CONTROL, linestyle="--", linewidth=1)
if MAIN:
    axes[0].set_ylim(0, 1.05)
if MAIN:
    axes[0].set_ylabel("held-out ROC AUC")
if MAIN:
    axes[0].set_title("Representation")
if MAIN:
    for index, value in enumerate(auc_values):
        axes[0].text(index, value + 0.03, f"{value:.3f}", ha="center")

if MAIN:
    axes[1].plot(coefficients, steering_target, marker="o", color=TARGET, label="selected feature")
if MAIN:
    axes[1].plot(coefficients, steering_feature_control, marker="o", color=FEATURE_CONTROL, label="random feature")
if MAIN:
    axes[1].plot(coefficients, steering_direction_control, marker="o", color=DIRECTION_CONTROL, label="norm-matched random")
if MAIN:
    axes[1].axvline(0, color="#B8BDC5", linewidth=1)
if MAIN:
    axes[1].set_xlabel("steering coefficient")
if MAIN:
    axes[1].set_ylabel("technical direction score")
if MAIN:
    axes[1].set_title("Steering on held-out negatives")
if MAIN:
    axes[1].legend(frameon=False, fontsize=8)

if MAIN:
    axes[2].plot(fractions, ablation_target, marker="o", color=TARGET, label="ablate selected")
if MAIN:
    axes[2].plot(fractions, ablation_control, marker="o", color=FEATURE_CONTROL, label="ablate random feature")
if MAIN:
    axes[2].set_xlabel("fraction of contribution removed")
if MAIN:
    axes[2].set_ylabel("technical direction score")
if MAIN:
    axes[2].set_title("Ablation on held-out positives")
if MAIN:
    axes[2].legend(frameon=False, fontsize=8)

if MAIN:
    fig.suptitle("Exact sparse-feature organism: held-out detection and causal control", fontsize=13, fontweight="bold")
if MAIN:
    fig.tight_layout()
if MAIN:
    exact_figure_path = assets_dir / "gemma_scope_exact_organism_signature.png"
if MAIN:
    fig.savefig(exact_figure_path, dpi=170, bbox_inches="tight")
if MAIN:
    plt.show()

if MAIN:
    print(
        f"Exact metrics: AUC={heldout_auc:.3f}, nuisance AUC={np.mean(random_feature_aucs):.3f}, "
        f"shuffle AUC={np.mean(shuffled_aucs):.3f}, "
        f"steering delta={steering_target[-1] - steering_target[4]:.3f}, "
        f"full-ablation score={ablation_target[-1]:.3f}."
    )

# %%

PLAY_ROW = 33
if MAIN:
    PLAY_FEATURE_ID = selected_feature_id
PLAY_COEFFICIENT = 0.75

if MAIN:
    play_residual = organism["residuals"][PLAY_ROW]
if MAIN:
    play_direction = organism["w_dec"][PLAY_FEATURE_ID]
if MAIN:
    play_steered = steer_residual(play_residual, play_direction, PLAY_COEFFICIENT)
if MAIN:
    print("Prompt:", organism["rows"][PLAY_ROW]["text"])
if MAIN:
    print(f"Feature: {PLAY_FEATURE_ID} ({FEATURE_NAMES[PLAY_FEATURE_ID]})")
if MAIN:
    print(f"Target score before: {exact_target_score(play_residual):.3f}")
if MAIN:
    print(f"Target score after:  {exact_target_score(play_steered):.3f}")

# %%

GEMMA_SCOPE_REPO = "google/gemma-scope-2-1b-it"
GEMMA_SCOPE_REVISION = "b0fa29457c3601df0a70c48a15534c738d7c10e0"
GEMMA_SCOPE_ARTIFACT = "resid_post/layer_13_width_16k_l0_small"
GEMMA_MODEL_REVISION = "dcc83ea841ab6100d6b47a070329e1ba4cf78752"
def load_released_gemma_scope_sae(
    repo_id: str,
    revision: str,
    artifact_path: str,
) -> tuple[dict, dict[str, t.Tensor]]:
    """Load the pinned Gemma Scope config and tensors on CPU."""
    config_path = hf_hub_download(
        repo_id=repo_id,
        revision=revision,
        filename=f"{artifact_path}/config.json",
    )
    params_path = hf_hub_download(
        repo_id=repo_id,
        revision=revision,
        filename=f"{artifact_path}/params.safetensors",
    )
    config = json.loads(Path(config_path).read_text())
    state = load_file(params_path, device="cpu")
    required = {"w_enc", "w_dec", "b_enc", "b_dec", "threshold"}
    if not required.issubset(state):
        raise ValueError(f"Missing SAE tensors: {sorted(required - set(state))}")
    if config.get("architecture") != "jump_relu":
        raise ValueError("Expected a JumpReLU SAE.")
    if config.get("model_name") != "google/gemma-3-1b-it":
        raise ValueError("Artifact points to an unexpected base model.")
    if config.get("hf_hook_point_in") != "model.layers.13.output":
        raise ValueError("Artifact points to an unexpected hook location.")
    if tuple(state["w_enc"].shape) != (1152, 16384):
        raise ValueError("Unexpected encoder shape.")
    if tuple(state["w_dec"].shape) != (16384, 1152):
        raise ValueError("Unexpected decoder shape.")
    return config, state
if MAIN:
    gemma_scope_config, gemma_scope_state = load_released_gemma_scope_sae(
        GEMMA_SCOPE_REPO,
        GEMMA_SCOPE_REVISION,
        GEMMA_SCOPE_ARTIFACT,
    )
if MAIN:
    tests.test_released_gemma_scope_artifact(
        gemma_scope_config,
        gemma_scope_state,
        jumprelu_encode,
        sae_decode,
    )
if MAIN:
    print(
        "Released artifact:",
        gemma_scope_config["model_name"], "|",
        gemma_scope_config["hf_hook_point_in"], "|",
        f"width={gemma_scope_config['width']}", "|",
        f"d_model={gemma_scope_state['w_dec'].shape[1]}",
    )

# %%

if MAIN:
    from transformers import AutoModelForCausalLM, AutoTokenizer

REAL_TECHNICAL_PROMPTS = (
    "Explain how a Python function validates a JSON schema before calling a tool.",
    "Describe a tensor shape check for a batched matrix multiplication in PyTorch.",
    "Write a concise note about masking invalid API actions before decoding.",
    "Explain how an embedding retrieval system ranks documents with cosine similarity.",
    "Describe a unit test for parsing function-call arguments from model output.",
    "Explain why attention masks should ignore padding tokens during pooling.",
    "Write a short note about checking CUDA tensor dtype and device placement.",
    "Describe how to compute a sparse autoencoder reconstruction error.",
)
REAL_NARRATIVE_PROMPTS = (
    "Describe a quiet garden with rain on green leaves in the early morning.",
    "Tell a calm story about a child watching clouds move over a hill.",
    "Describe the color of sunset light across a lake and a wooden dock.",
    "Write a peaceful paragraph about walking through fresh snow at night.",
    "Describe a small kitchen where bread cools beside a window.",
    "Tell a gentle scene about friends listening to music in a park.",
    "Describe the sound of waves reaching a beach under a clear sky.",
    "Write a short scene about lanterns glowing during an evening festival.",
)

if MAIN:
    tokenizer = AutoTokenizer.from_pretrained(
        gemma_scope_config["model_name"],
        revision=GEMMA_MODEL_REVISION,
    )
if MAIN:
    tokenizer.padding_side = "left"
if MAIN:
    model = AutoModelForCausalLM.from_pretrained(
        gemma_scope_config["model_name"],
        revision=GEMMA_MODEL_REVISION,
        dtype=t.bfloat16,
        device_map="cpu",
    )
if MAIN:
    model.eval()
if MAIN:
    assert next(model.parameters()).device.type == "cpu"

def hidden_from_layer_output(output):
    return output[0] if isinstance(output, tuple) else output


def replace_hidden_in_layer_output(output, hidden):
    return (hidden, *output[1:]) if isinstance(output, tuple) else hidden


def capture_layer_residuals(prompts: list[str] | tuple[str, ...]):
    inputs = tokenizer(
        list(prompts), return_tensors="pt", padding=True,
        truncation=True, max_length=32,
    )
    captured = {}

    def capture_hook(_module, _inputs, output):
        captured["residual"] = hidden_from_layer_output(output).detach().float()

    handle = model.model.layers[13].register_forward_hook(capture_hook)
    try:
        with t.inference_mode():
            model.model(**inputs)
    finally:
        handle.remove()
    return inputs, captured["residual"]


if MAIN:
    real_prompts = REAL_TECHNICAL_PROMPTS + REAL_NARRATIVE_PROMPTS
if MAIN:
    real_labels = t.tensor([1] * 8 + [0] * 8, dtype=t.bool)
if MAIN:
    real_discovery = t.tensor([1] * 4 + [0] * 4 + [1] * 4 + [0] * 4, dtype=t.bool)
if MAIN:
    real_inputs, real_residuals = capture_layer_residuals(real_prompts)
if MAIN:
    real_feature_acts = jumprelu_encode(
        real_residuals,
        w_enc=gemma_scope_state["w_enc"],
        b_enc=gemma_scope_state["b_enc"],
        b_dec=gemma_scope_state["b_dec"],
        threshold=gemma_scope_state["threshold"],
    )
if MAIN:
    real_feature_acts = real_feature_acts.masked_fill(
        ~real_inputs["attention_mask"].bool()[..., None], 0
    )
if MAIN:
    real_feature_scores = real_feature_acts.max(dim=1).values
if MAIN:
    real_feature_id = select_feature_by_mean_difference(
        real_feature_scores, real_labels, real_discovery
    )
if MAIN:
    real_random_feature_id = (real_feature_id + 7919) % gemma_scope_config["width"]
if MAIN:
    real_heldout = ~real_discovery
if MAIN:
    real_auc = binary_roc_auc(
        real_feature_scores[real_heldout, real_feature_id], real_labels[real_heldout]
    )
if MAIN:
    real_random_auc = binary_roc_auc(
        real_feature_scores[real_heldout, real_random_feature_id], real_labels[real_heldout]
    )
if MAIN:
    assert real_auc >= 0.95 and real_auc - real_random_auc >= 0.25

if MAIN:
    print(
        f"Discovery selected feature {real_feature_id}; held-out AUC={real_auc:.3f}; "
        f"random feature {real_random_feature_id} AUC={real_random_auc:.3f}."
    )
if MAIN:
    print("\nAll held-out prompt scores (visible examples and anomalies):")
if MAIN:
    heldout_global_indices = t.nonzero(real_heldout, as_tuple=False).flatten()
if MAIN:
    ordering = real_feature_scores[real_heldout, real_feature_id].argsort(descending=True)
if MAIN:
    for local_index in ordering.tolist():
        global_index = int(heldout_global_indices[local_index].item())
        label = "technical" if real_labels[global_index] else "narrative"
        score = real_feature_scores[global_index, real_feature_id].item()
        print(f"  {score:8.2f} | {label:<9} | {real_prompts[global_index]}")

# %%

def make_steering_hook(direction: t.Tensor, coefficient: float):
    """Return a Hugging Face forward hook that adds one direction."""
    def hook(_module, _inputs, output):
        hidden = hidden_from_layer_output(output)
        steered = steer_residual(hidden, direction, coefficient)
        return replace_hidden_in_layer_output(output, steered)
    return hook


def make_ablation_hook(
    clean_feature_scores: t.Tensor,
    direction: t.Tensor,
    fraction: float,
):
    """Return a hook that removes a clean feature contribution."""
    def hook(_module, _inputs, output):
        hidden = hidden_from_layer_output(output)
        contribution = (
            clean_feature_scores[..., None].to(hidden) * direction.to(hidden)
        )
        ablated = hidden - float(fraction) * contribution
        return replace_hidden_in_layer_output(output, ablated)
    return hook
if MAIN:
    hook_test_hidden = t.zeros(2, 3, 4)
if MAIN:
    hook_test_direction = t.tensor([1.0, 0.0, 0.0, 0.0])
if MAIN:
    tensor_result = make_steering_hook(hook_test_direction, 2.0)(
        None, (), hook_test_hidden
    )
if MAIN:
    tuple_result = make_steering_hook(hook_test_direction, 2.0)(
        None, (), (hook_test_hidden, "cache")
    )
if MAIN:
    t.testing.assert_close(tensor_result[..., 0], t.full((2, 3), 2.0))
if MAIN:
    t.testing.assert_close(tuple_result[0][..., 0], t.full((2, 3), 2.0))
if MAIN:
    assert tuple_result[1] == "cache"
if MAIN:
    print("Hook tests passed for tensor and tuple layer outputs.")

# %%

def readable_extreme_token_ids(effects: t.Tensor, *, largest: bool, k: int = 6) -> list[int]:
    candidates = effects.topk(5000, largest=largest).indices.tolist()
    selected = []
    for token_id in candidates:
        token = tokenizer.decode([token_id])
        stripped = token.strip()
        if (
            stripped
            and token.isascii()
            and token.isprintable()
            and any(character.isalpha() for character in token)
            and "<" not in token
            and len(stripped) >= 2
        ):
            selected.append(token_id)
        if len(selected) == k:
            break
    if len(selected) != k:
        raise RuntimeError("Could not find enough readable DLA tokens.")
    return selected


if MAIN:
    selected_decoder_direction = gemma_scope_state["w_dec"][real_feature_id]
if MAIN:
    random_feature_direction = gemma_scope_state["w_dec"][real_random_feature_id]
if MAIN:
    real_dla = selected_decoder_direction @ model.lm_head.weight.detach().float().T
if MAIN:
    positive_token_ids = readable_extreme_token_ids(real_dla, largest=True)
if MAIN:
    negative_token_ids = readable_extreme_token_ids(real_dla, largest=False)
if MAIN:
    print("Positive DLA tokens:", [tokenizer.decode([token_id]) for token_id in positive_token_ids])
if MAIN:
    print("Negative DLA tokens:", [tokenizer.decode([token_id]) for token_id in negative_token_ids])

def next_token_contrast(logits: t.Tensor) -> t.Tensor:
    positive = logits[:, positive_token_ids].float().mean(dim=-1)
    negative = logits[:, negative_token_ids].float().mean(dim=-1)
    return positive - negative


def score_prompts_with_hook(prompts, hook) -> float:
    inputs = tokenizer(
        list(prompts), return_tensors="pt", padding=True,
        truncation=True, max_length=32,
    )
    handle = model.model.layers[13].register_forward_hook(hook)
    try:
        with t.inference_mode():
            final_logits = model(**inputs).logits[:, -1]
    finally:
        handle.remove()
    return next_token_contrast(final_logits).mean().item()


if MAIN:
    heldout_narrative = REAL_NARRATIVE_PROMPTS[4:]
if MAIN:
    heldout_technical = REAL_TECHNICAL_PROMPTS[4:]
if MAIN:
    real_coefficients = t.tensor([-200.0, -100.0, 0.0, 100.0, 200.0])
if MAIN:
    random_generator = t.Generator().manual_seed(23)
if MAIN:
    real_random_direction = t.randn(selected_decoder_direction.shape, generator=random_generator)
if MAIN:
    real_random_direction = (
        real_random_direction / real_random_direction.norm() * selected_decoder_direction.norm()
    )

if MAIN:
    real_steering_target = [
        score_prompts_with_hook(
            heldout_narrative,
            make_steering_hook(selected_decoder_direction, coefficient),
        )
        for coefficient in real_coefficients.tolist()
    ]
if MAIN:
    real_steering_feature_control = [
        score_prompts_with_hook(
            heldout_narrative,
            make_steering_hook(random_feature_direction, coefficient),
        )
        for coefficient in real_coefficients.tolist()
    ]
if MAIN:
    real_steering_direction_control = [
        score_prompts_with_hook(
            heldout_narrative,
            make_steering_hook(real_random_direction, coefficient),
        )
        for coefficient in real_coefficients.tolist()
    ]

if MAIN:
    technical_inputs, technical_residuals = capture_layer_residuals(heldout_technical)
if MAIN:
    technical_feature_acts = jumprelu_encode(
        technical_residuals,
        w_enc=gemma_scope_state["w_enc"],
        b_enc=gemma_scope_state["b_enc"],
        b_dec=gemma_scope_state["b_dec"],
        threshold=gemma_scope_state["threshold"],
    )
if MAIN:
    real_fractions = t.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
if MAIN:
    real_ablation_target = [
        score_prompts_with_hook(
            heldout_technical,
            make_ablation_hook(
                technical_feature_acts[..., real_feature_id],
                selected_decoder_direction,
                fraction,
            ),
        )
        for fraction in real_fractions.tolist()
    ]
if MAIN:
    real_ablation_control = [
        score_prompts_with_hook(
            heldout_technical,
            make_ablation_hook(
                technical_feature_acts[..., real_random_feature_id],
                random_feature_direction,
                fraction,
            ),
        )
        for fraction in real_fractions.tolist()
    ]

center = 2
if MAIN:
    target_steering_delta = real_steering_target[-1] - real_steering_target[center]
if MAIN:
    feature_control_delta = real_steering_feature_control[-1] - real_steering_feature_control[center]
if MAIN:
    direction_control_delta = real_steering_direction_control[-1] - real_steering_direction_control[center]
if MAIN:
    target_ablation_drop = real_ablation_target[0] - real_ablation_target[-1]
if MAIN:
    control_ablation_drop = real_ablation_control[0] - real_ablation_control[-1]
if MAIN:
    assert target_steering_delta > max(feature_control_delta, direction_control_delta)
if MAIN:
    assert target_ablation_drop > control_ablation_drop + 0.1

if MAIN:
    real_shuffle_generator = t.Generator().manual_seed(31)
real_shuffle_aucs = []
if MAIN:
    real_heldout_scores = real_feature_scores[real_heldout, real_feature_id]
if MAIN:
    real_heldout_labels = real_labels[real_heldout]
if MAIN:
    for _ in range(200):
        permutation = t.randperm(real_heldout_labels.numel(), generator=real_shuffle_generator)
        real_shuffle_aucs.append(
            binary_roc_auc(real_heldout_scores, real_heldout_labels[permutation])
        )

if MAIN:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
if MAIN:
    heldout_technical_scores = real_feature_scores[real_heldout & real_labels, real_feature_id]
if MAIN:
    heldout_narrative_scores = real_feature_scores[real_heldout & ~real_labels, real_feature_id]
if MAIN:
    axes[0].scatter(
        np.zeros(len(heldout_technical_scores)), heldout_technical_scores,
        color=TARGET, s=55, label="technical",
    )
if MAIN:
    axes[0].scatter(
        np.ones(len(heldout_narrative_scores)), heldout_narrative_scores,
        color=FEATURE_CONTROL, s=55, label="narrative",
    )
if MAIN:
    axes[0].set_xticks([0, 1], ["technical", "narrative"])
if MAIN:
    axes[0].set_ylabel(f"max activation of feature {real_feature_id}")
if MAIN:
    axes[0].set_title(f"Held-out AUC {real_auc:.3f}\nrandom feature {real_random_auc:.3f}")

if MAIN:
    axes[1].plot(real_coefficients, real_steering_target, marker="o", color=TARGET, label="selected feature")
if MAIN:
    axes[1].plot(real_coefficients, real_steering_feature_control, marker="o", color=FEATURE_CONTROL, label="random feature")
if MAIN:
    axes[1].plot(real_coefficients, real_steering_direction_control, marker="o", color=DIRECTION_CONTROL, label="norm-matched random")
if MAIN:
    axes[1].axvline(0, color="#B8BDC5", linewidth=1)
if MAIN:
    axes[1].set_xlabel("layer-13 steering coefficient")
if MAIN:
    axes[1].set_ylabel("DLA-aligned next-token contrast")
if MAIN:
    axes[1].set_title("Steering: held-out narratives")
if MAIN:
    axes[1].legend(frameon=False, fontsize=8)

if MAIN:
    axes[2].plot(real_fractions, real_ablation_target, marker="o", color=TARGET, label="ablate selected")
if MAIN:
    axes[2].plot(real_fractions, real_ablation_control, marker="o", color=FEATURE_CONTROL, label="ablate random feature")
if MAIN:
    axes[2].set_xlabel("fraction of clean contribution removed")
if MAIN:
    axes[2].set_ylabel("DLA-aligned next-token contrast")
if MAIN:
    axes[2].set_title("Ablation: held-out technical prompts")
if MAIN:
    axes[2].legend(frameon=False, fontsize=8)

if MAIN:
    fig.suptitle("Pinned Gemma 3 1B + released Gemma Scope SAE: small CPU probe", fontsize=13, fontweight="bold")
if MAIN:
    fig.tight_layout()
if MAIN:
    real_figure_path = assets_dir / "gemma_scope_real_model_probe.png"
if MAIN:
    fig.savefig(real_figure_path, dpi=170, bbox_inches="tight")
if MAIN:
    plt.show()

if MAIN:
    print(
        f"Real metrics: held-out AUC={real_auc:.3f}, random-feature AUC={real_random_auc:.3f}, "
        f"shuffle AUC={np.mean(real_shuffle_aucs):.3f}, "
        f"steering delta={target_steering_delta:.3f} "
        f"(feature control={feature_control_delta:.3f}, direction control={direction_control_delta:.3f}), "
        f"ablation drop={target_ablation_drop:.3f} (control={control_ablation_drop:.3f})."
    )

# %%

YOUR_PROMPT = "Explain why the lantern parser rejected a malformed tensor."
if MAIN:
    _, your_residual = capture_layer_residuals([YOUR_PROMPT])
if MAIN:
    your_acts = jumprelu_encode(
        your_residual,
        w_enc=gemma_scope_state["w_enc"],
        b_enc=gemma_scope_state["b_enc"],
        b_dec=gemma_scope_state["b_dec"],
        threshold=gemma_scope_state["threshold"],
    )
if MAIN:
    print("Prompt:", YOUR_PROMPT)
if MAIN:
    print(f"Selected feature {real_feature_id}: {your_acts[..., real_feature_id].max().item():.2f}")
if MAIN:
    print(f"Random feature {real_random_feature_id}: {your_acts[..., real_random_feature_id].max().item():.2f}")

# %%

import importlib


def _release_solution_module():
    return importlib.import_module("chapter5_modern_architectures.exercises.part2_gemma_scope_feature_steering.solutions")


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    """Run the section's pinned CUDA verification contract."""
    return _release_solution_module().run_gpu_test(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the full experiment used for the checked-in release evidence."""
    return _release_solution_module().run_full_experiment(
        max_vram_gb=max_vram_gb
    )

# %%

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import torch as t
from IPython.display import Markdown, display

deep_dive_chapter = "chapter6_sparse_feature_methods"
deep_dive_section = "part2_gemma_scope_deep_dive"
deep_dive_root_dir = Path(__file__).resolve().parents[3]
deep_dive_exercises_dir = deep_dive_root_dir / deep_dive_chapter / "exercises"
deep_dive_section_dir = deep_dive_exercises_dir / deep_dive_section
if str(deep_dive_root_dir) not in sys.path:
    sys.path.append(str(deep_dive_root_dir))
if str(deep_dive_exercises_dir) not in sys.path:
    sys.path.append(str(deep_dive_exercises_dir))

import chapter6_sparse_feature_methods.exercises.part2_gemma_scope_deep_dive.tests as deep_dive_tests
import chapter6_sparse_feature_methods.exercises.part2_gemma_scope_deep_dive.utils as utils

GT_TIER = "GT-1"
EXERCISE_ID = "6_2_gemma_scope_deep_dive"


def show_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
    display(Markdown("\n".join(lines)))


@dataclass(frozen=True)
class SparseAutoencoder:
    w_enc: t.Tensor
    w_dec: t.Tensor
    b_enc: t.Tensor
    b_dec: t.Tensor
    threshold: t.Tensor

@dataclass(frozen=True)
class ReconstructionReport:
    mse: float
    relative_mse: float
    explained_variance: float

@dataclass(frozen=True)
class HeldOutValidationReport:
    feature_auc: float
    random_feature_auc: float
    shuffled_label_auc: float
    threshold_accuracy: float
    positive_mean: float
    negative_mean: float

# %%

if MAIN:
    deep_dive_organism = utils.make_ground_truth_organism()
if MAIN:
    heldout = ~deep_dive_organism["train_mask"]
if MAIN:
    print(f"examples: {len(deep_dive_organism['texts'])} total, {int(heldout.sum())} held out")
if MAIN:
    print(f"residuals: {tuple(deep_dive_organism['residuals'].shape)}; latents: {tuple(deep_dive_organism['latent_codes'].shape)}")
if MAIN:
    show_table(
        ["held-out text", "label", "true feature 0", "control feature 4"],
        [
            (
                deep_dive_organism["texts"][i],
                "technical" if deep_dive_organism["labels"][i] else "narrative",
                f"{deep_dive_organism['latent_codes'][i, 0]:.1f}",
                f"{deep_dive_organism['latent_codes'][i, 4]:.2f}",
            )
            for i in heldout.nonzero().flatten()[:6].tolist()
        ],
    )

# %%

def load_normalized_sae(
    artifact: Mapping[str, t.Tensor],
) -> tuple[SparseAutoencoder, t.Tensor]:
    """Validate SAE tensors and make every decoder row unit norm.

    Scaling an encoder column, its bias, and its JumpReLU threshold by the
    original decoder norm preserves the represented function exactly.
    """

    required = {"w_enc", "w_dec", "b_enc", "b_dec", "threshold"}
    missing = required.difference(artifact)
    if missing:
        raise KeyError(f"Missing SAE tensors: {sorted(missing)}")

    w_enc = t.as_tensor(artifact["w_enc"]).detach().clone().float()
    w_dec = t.as_tensor(artifact["w_dec"]).detach().clone().float()
    b_enc = t.as_tensor(artifact["b_enc"]).detach().clone().float()
    b_dec = t.as_tensor(artifact["b_dec"]).detach().clone().float()
    threshold = t.as_tensor(artifact["threshold"]).detach().clone().float()

    if w_enc.ndim != 2 or w_dec.ndim != 2:
        raise ValueError("w_enc and w_dec must be rank-2 tensors.")
    d_model, n_features = w_enc.shape
    if w_dec.shape != (n_features, d_model):
        raise ValueError("w_dec must have shape (n_features, d_model).")
    if b_enc.shape != (n_features,) or b_dec.shape != (d_model,):
        raise ValueError("Encoder and decoder biases have incompatible shapes.")
    if threshold.ndim == 0:
        threshold = threshold.expand(n_features).clone()
    if threshold.shape != (n_features,):
        raise ValueError("threshold must be scalar or have shape (n_features,).")
    if not all(t.isfinite(x).all() for x in (w_enc, w_dec, b_enc, b_dec, threshold)):
        raise ValueError("SAE tensors must be finite.")

    decoder_norms = w_dec.norm(dim=-1)
    if (decoder_norms <= 0).any():
        raise ValueError("Every decoder row must have non-zero norm.")
    sae = SparseAutoencoder(
        w_enc=w_enc * decoder_norms,
        w_dec=w_dec / decoder_norms[:, None],
        b_enc=b_enc * decoder_norms,
        b_dec=b_dec,
        threshold=threshold * decoder_norms,
    )
    return sae, decoder_norms

if MAIN:
    deep_dive_tests.test_load_normalized_sae_preserves_function(SparseAutoencoder, load_normalized_sae)

# %%

def encode_jump_relu(residuals: t.Tensor, sae: SparseAutoencoder) -> t.Tensor:
    """Encode residual vectors with Gemma Scope's JumpReLU convention."""

    if residuals.shape[-1] != sae.w_enc.shape[0]:
        raise ValueError("Residual width must match SAE d_model.")
    pre_acts = (residuals.float() - sae.b_dec) @ sae.w_enc + sae.b_enc
    return t.relu(pre_acts) * (pre_acts > sae.threshold)

def reconstruct_from_features(feature_acts: t.Tensor, sae: SparseAutoencoder) -> t.Tensor:
    """Decode sparse feature activations into the residual stream."""

    if feature_acts.shape[-1] != sae.w_dec.shape[0]:
        raise ValueError("Feature width must match the SAE dictionary width.")
    return feature_acts.float() @ sae.w_dec + sae.b_dec

def reconstruction_report(
    residuals: t.Tensor,
    reconstructed: t.Tensor,
) -> ReconstructionReport:
    """Report absolute and scale-aware reconstruction quality."""

    if residuals.shape != reconstructed.shape:
        raise ValueError("Residuals and reconstructions must have identical shapes.")
    residuals = residuals.float()
    reconstructed = reconstructed.float()
    mse = (residuals - reconstructed).square().mean()
    centered_energy = (residuals - residuals.mean(dim=0, keepdim=True)).square().mean()
    if centered_energy <= 0:
        raise ValueError("Explained variance is undefined for constant residuals.")
    relative_mse = mse / centered_energy
    return ReconstructionReport(
        mse=float(mse.item()),
        relative_mse=float(relative_mse.item()),
        explained_variance=float((1 - relative_mse).item()),
    )

if MAIN:
    deep_dive_tests.test_jump_relu_recovers_exact_latents(load_normalized_sae, encode_jump_relu)
if MAIN:
    deep_dive_tests.test_reconstruction_is_exact(
        load_normalized_sae,
        encode_jump_relu,
        reconstruct_from_features,
        reconstruction_report,
    )

# %%

if MAIN:
    sae, raw_decoder_norms = load_normalized_sae(deep_dive_organism["raw_artifact"])
if MAIN:
    feature_acts = encode_jump_relu(deep_dive_organism["residuals"], sae)
if MAIN:
    reconstructed = reconstruct_from_features(feature_acts, sae)
if MAIN:
    recon = reconstruction_report(deep_dive_organism["residuals"], reconstructed)
if MAIN:
    print(f"maximum latent error: {(feature_acts - deep_dive_organism['latent_codes']).abs().max():.2e}")
if MAIN:
    print(f"reconstruction MSE: {recon.mse:.3e}")
if MAIN:
    print(f"explained variance: {recon.explained_variance:.6f}")

# %%

def feature_score_vector(
    feature_acts: t.Tensor,
    feature_id: int,
    *,
    reduction: Literal["max", "mean", "last"] = "max",
) -> t.Tensor:
    """Reduce token-level activations to one score per example."""

    if feature_id < 0 or feature_id >= feature_acts.shape[-1]:
        raise IndexError("feature_id is out of range.")
    if feature_acts.ndim == 2:
        return feature_acts[:, feature_id].float()
    if feature_acts.ndim != 3:
        raise ValueError("feature_acts must have shape (batch, features) or (batch, seq, features).")
    per_token = feature_acts[..., feature_id].float()
    if reduction == "max":
        return per_token.max(dim=-1).values
    if reduction == "mean":
        return per_token.mean(dim=-1)
    if reduction == "last":
        return per_token[:, -1]
    raise ValueError("reduction must be 'max', 'mean', or 'last'.")

def roc_auc_binary(scores: t.Tensor, labels: t.Tensor) -> float:
    """Compute binary ROC AUC from average ranks, including ties."""

    scores = scores.detach().flatten().float()
    labels = labels.detach().flatten().bool()
    if scores.numel() != labels.numel():
        raise ValueError("scores and labels must have the same length.")
    n_pos = int(labels.sum().item())
    n_neg = int((~labels).sum().item())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC needs at least one positive and one negative.")

    order = scores.argsort()
    sorted_scores = scores[order]
    ranks_sorted = t.empty_like(sorted_scores)
    start = 0
    while start < sorted_scores.numel():
        end = start + 1
        while end < sorted_scores.numel() and t.isclose(
            sorted_scores[end], sorted_scores[start], rtol=1e-6, atol=1e-7
        ):
            end += 1
        ranks_sorted[start:end] = (start + 1 + end) / 2
        start = end
    ranks = t.empty_like(scores)
    ranks[order] = ranks_sorted
    pos_rank_sum = ranks[labels].sum()
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc.item())

def validate_heldout_feature(
    feature_scores: t.Tensor,
    labels: t.Tensor,
    random_feature_scores: t.Tensor,
    shuffled_labels: t.Tensor,
) -> HeldOutValidationReport:
    """Compare a feature with random-feature and shuffled-label controls."""

    feature_scores = feature_scores.flatten().float()
    labels = labels.flatten().bool()
    random_feature_scores = random_feature_scores.flatten().float()
    shuffled_labels = shuffled_labels.flatten().bool()
    if not (
        feature_scores.shape
        == labels.shape
        == random_feature_scores.shape
        == shuffled_labels.shape
    ):
        raise ValueError("All held-out vectors must have identical shapes.")
    positives = feature_scores[labels]
    negatives = feature_scores[~labels]
    if positives.numel() == 0 or negatives.numel() == 0:
        raise ValueError("Held-out labels need both classes.")
    threshold = 0.5 * (positives.mean() + negatives.mean())
    predictions = feature_scores >= threshold
    return HeldOutValidationReport(
        feature_auc=roc_auc_binary(feature_scores, labels),
        random_feature_auc=roc_auc_binary(random_feature_scores, labels),
        shuffled_label_auc=roc_auc_binary(feature_scores, shuffled_labels),
        threshold_accuracy=float(predictions.eq(labels).float().mean().item()),
        positive_mean=float(positives.mean().item()),
        negative_mean=float(negatives.mean().item()),
    )

if MAIN:
    deep_dive_tests.test_feature_score_reductions(feature_score_vector)
if MAIN:
    deep_dive_tests.test_auc_and_heldout_controls(
        roc_auc_binary,
        validate_heldout_feature,
        load_normalized_sae,
        encode_jump_relu,
    )

# %%

if MAIN:
    target_scores = feature_score_vector(feature_acts, deep_dive_organism["target_feature_id"])
if MAIN:
    control_scores = feature_score_vector(feature_acts, deep_dive_organism["control_feature_id"])
if MAIN:
    validation = validate_heldout_feature(
        target_scores[heldout],
        deep_dive_organism["labels"][heldout],
        control_scores[heldout],
        deep_dive_organism["shuffled_labels"][heldout],
    )
if MAIN:
    train_confound_auc = roc_auc_binary(
        feature_acts[deep_dive_organism["train_mask"], 1], deep_dive_organism["labels"][deep_dive_organism["train_mask"]]
    )
if MAIN:
    heldout_confound_auc = roc_auc_binary(feature_acts[heldout, 1], deep_dive_organism["labels"][heldout])
if MAIN:
    show_table(
        ["measurement", "value"],
        [
            ("target held-out AUC", f"{validation.feature_auc:.2f}"),
            ("random-feature AUC", f"{validation.random_feature_auc:.2f}"),
            ("shuffled-label AUC", f"{validation.shuffled_label_auc:.2f}"),
            ("threshold accuracy", f"{validation.threshold_accuracy:.2f}"),
            ("confound train -> held-out AUC", f"{train_confound_auc:.2f} -> {heldout_confound_auc:.2f}"),
        ],
    )

# %%

def activation_density(feature_acts: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """Return per-feature firing density and a dead-feature mask."""

    if feature_acts.ndim < 2:
        raise ValueError("feature_acts needs at least one sample axis and one feature axis.")
    flat = feature_acts.reshape(-1, feature_acts.shape[-1])
    density = (flat > 0).float().mean(dim=0)
    return density, density == 0

if MAIN:
    deep_dive_tests.test_density_finds_rare_and_dead_features(activation_density)

# %%

def deep_dive_ablate_feature(feature_acts: t.Tensor, feature_id: int) -> t.Tensor:
    """Set one feature coordinate to zero without mutating the input."""

    if feature_id < 0 or feature_id >= feature_acts.shape[-1]:
        raise IndexError("feature_id is out of range.")
    ablated = feature_acts.clone()
    ablated[..., feature_id] = 0
    return ablated

def steer_residuals(
    residuals: t.Tensor,
    direction: t.Tensor,
    strengths: t.Tensor | list[float],
) -> t.Tensor:
    """Add equal-L2 interventions by normalizing the supplied direction."""

    direction = t.as_tensor(direction, dtype=residuals.dtype, device=residuals.device)
    if direction.ndim != 1 or direction.shape[0] != residuals.shape[-1]:
        raise ValueError("direction must be a vector with residual width.")
    norm = direction.norm()
    if norm <= 0:
        raise ValueError("direction must be non-zero.")
    unit_direction = direction / norm
    strengths = t.as_tensor(strengths, dtype=residuals.dtype, device=residuals.device).flatten()
    view_shape = (strengths.numel(),) + (1,) * residuals.ndim
    return residuals.unsqueeze(0) + strengths.view(view_shape) * unit_direction

if MAIN:
    deep_dive_tests.test_ablation_has_matched_feature_control(
        deep_dive_ablate_feature,
        load_normalized_sae,
        encode_jump_relu,
        reconstruct_from_features,
    )
if MAIN:
    deep_dive_tests.test_steering_normalizes_direction(steer_residuals)

# %%

def direct_logit_attribution(
    decoder_vectors: t.Tensor,
    unembedding: t.Tensor,
    token_ids: t.Tensor | list[int] | None = None,
) -> t.Tensor:
    """Project decoder directions through an unembedding matrix."""

    if decoder_vectors.ndim != 2 or unembedding.ndim != 2:
        raise ValueError("decoder_vectors and unembedding must be rank-2.")
    if decoder_vectors.shape[-1] != unembedding.shape[0]:
        raise ValueError("Decoder width must match the unembedding input width.")
    effects = decoder_vectors.float() @ unembedding.float()
    if token_ids is None:
        return effects
    return effects[:, t.as_tensor(token_ids, device=effects.device)]

if MAIN:
    deep_dive_tests.test_direct_logit_attribution_has_known_tokens(direct_logit_attribution, load_normalized_sae)

# %%

if MAIN:
    density, dead = activation_density(feature_acts)
if MAIN:
    target_id = deep_dive_organism["target_feature_id"]
if MAIN:
    control_id = deep_dive_organism["control_feature_id"]
if MAIN:
    positive_heldout = heldout & deep_dive_organism["labels"]
if MAIN:
    readout = sae.w_dec[target_id]
if MAIN:
    baseline_behavior = (deep_dive_organism["residuals"] - sae.b_dec) @ readout

if MAIN:
    target_recon = reconstruct_from_features(deep_dive_ablate_feature(feature_acts, target_id), sae)
if MAIN:
    control_recon = reconstruct_from_features(deep_dive_ablate_feature(feature_acts, control_id), sae)
if MAIN:
    target_behavior = (target_recon - sae.b_dec) @ readout
if MAIN:
    control_behavior = (control_recon - sae.b_dec) @ readout
if MAIN:
    target_ablation_delta = float(
        (baseline_behavior[positive_heldout] - target_behavior[positive_heldout]).mean()
    )
if MAIN:
    control_ablation_delta = float(
        (baseline_behavior[positive_heldout] - control_behavior[positive_heldout]).mean()
    )
if MAIN:
    if abs(control_ablation_delta) < 1e-6:
        control_ablation_delta = 0.0

if MAIN:
    strengths = t.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
if MAIN:
    target_steered = steer_residuals(deep_dive_organism["residuals"], sae.w_dec[target_id], strengths)
if MAIN:
    control_steered = steer_residuals(deep_dive_organism["residuals"], sae.w_dec[control_id], strengths)
if MAIN:
    baseline_mean = baseline_behavior.mean()
if MAIN:
    target_curve = ((target_steered - sae.b_dec) @ readout).mean(dim=1) - baseline_mean
if MAIN:
    control_curve = ((control_steered - sae.b_dec) @ readout).mean(dim=1) - baseline_mean
if MAIN:
    control_curve = t.where(control_curve.abs() < 1e-6, 0.0, control_curve)
if MAIN:
    dla = direct_logit_attribution(sae.w_dec, deep_dive_organism["unembedding"])

if MAIN:
    heldout_indices = heldout.nonzero().flatten()
if MAIN:
    top_local = target_scores[heldout].topk(5).indices
if MAIN:
    show_table(
        ["top held-out example", "label", "activation"],
        [
            (
                deep_dive_organism["texts"][heldout_indices[i]],
                "technical" if deep_dive_organism["labels"][heldout_indices[i]] else "narrative",
                f"{target_scores[heldout_indices[i]]:.2f}",
            )
            for i in top_local.tolist()
        ],
    )
if MAIN:
    show_table(
        ["DLA token", "effect"],
        [
            (deep_dive_organism["token_names"][i], f"{dla[target_id, i]:+.2f}")
            for i in dla[target_id].argsort(descending=True).tolist()
        ],
    )

if MAIN:
    labels = [f"F{i}" for i in range(6)]
if MAIN:
    deep_dive_fig, deep_dive_axes = plt.subplots(2, 2, figsize=(12, 8))
if MAIN:
    deep_dive_axes[0, 0].bar(
        ["target", "random feature", "shuffled labels"],
        [validation.feature_auc, validation.random_feature_auc, validation.shuffled_label_auc],
        color=["#2563eb", "#94a3b8", "#e76f51"],
    )
if MAIN:
    deep_dive_axes[0, 0].axhline(0.5, color="#222222", linestyle="--", linewidth=1)
if MAIN:
    deep_dive_axes[0, 0].set_ylim(0, 1.05)
if MAIN:
    deep_dive_axes[0, 0].set_ylabel("held-out AUC")
if MAIN:
    deep_dive_axes[0, 0].set_title("Semantic validation")

if MAIN:
    x = t.arange(6).numpy()
if MAIN:
    deep_dive_axes[0, 1].bar(x - 0.18, raw_decoder_norms.numpy(), width=0.36, label="raw", color="#e9c46a")
if MAIN:
    deep_dive_axes[0, 1].bar(
        x + 0.18,
        sae.w_dec.norm(dim=-1).numpy(),
        width=0.36,
        label="normalized",
        color="#2a9d8f",
    )
if MAIN:
    deep_dive_axes[0, 1].set_xticks(x, labels)
if MAIN:
    deep_dive_axes[0, 1].set_ylabel("decoder row norm")
if MAIN:
    deep_dive_axes[0, 1].set_title("Coordinate audit")
if MAIN:
    deep_dive_axes[0, 1].legend(frameon=False)

if MAIN:
    deep_dive_axes[1, 0].bar(labels, density.numpy(), color=["#2563eb", "#f4a261", "#2a9d8f", "#e9c46a", "#94a3b8", "#d62828"])
if MAIN:
    deep_dive_axes[1, 0].set_ylim(0, 0.72)
if MAIN:
    deep_dive_axes[1, 0].set_ylabel("fraction active")
if MAIN:
    deep_dive_axes[1, 0].set_title("Density audit: F5 is dead")

if MAIN:
    deep_dive_axes[1, 1].plot(strengths, target_curve, marker="o", linewidth=2.5, label="target direction", color="#2563eb")
if MAIN:
    deep_dive_axes[1, 1].plot(strengths, control_curve, marker="s", linewidth=2.5, label="matched random", color="#e76f51")
if MAIN:
    deep_dive_axes[1, 1].axhline(0, color="#222222", linewidth=1)
if MAIN:
    deep_dive_axes[1, 1].set_xlabel("steering strength (equal L2)")
if MAIN:
    deep_dive_axes[1, 1].set_ylabel("mean readout change")
if MAIN:
    deep_dive_axes[1, 1].set_title(
        f"Causal control; ablation delta {target_ablation_delta:.2f} vs {control_ablation_delta:.2f}"
    )
if MAIN:
    deep_dive_axes[1, 1].legend(frameon=False)

if MAIN:
    deep_dive_fig.suptitle("Exact sparse-feature validation ladder", fontsize=16)
if MAIN:
    deep_dive_fig.tight_layout()
if MAIN:
    plt.show()

if MAIN:
    analysis = {
        "reconstruction": recon.__dict__,
        "raw_decoder_norms": raw_decoder_norms.tolist(),
        "normalized_decoder_norms": sae.w_dec.norm(dim=-1).tolist(),
        "density": density.tolist(),
        "dead_feature_ids": dead.nonzero().flatten().tolist(),
        "validation": validation.__dict__,
        "train_confound_auc": train_confound_auc,
        "heldout_confound_auc": heldout_confound_auc,
        "target_ablation_delta": target_ablation_delta,
        "control_ablation_delta": control_ablation_delta,
        "steering_strengths": strengths.tolist(),
        "target_steering_delta": target_curve.tolist(),
        "control_steering_delta": control_curve.tolist(),
        "target_dla": dla[target_id].tolist(),
    }
if MAIN:
    deep_dive_tests.test_ground_truth_contract(lambda cpu=True: analysis)


if MAIN:
    asset_path = deep_dive_root_dir / "chapter6_sparse_feature_methods" / "exercises" / "part2_gemma_scope" / "media" / "gemma_scope_ground_truth_signature.png"
if MAIN:
    deep_dive_fig.savefig(asset_path, dpi=180, bbox_inches="tight")
if MAIN:
    print(f"saved {asset_path.relative_to(deep_dive_root_dir)}")

# %%

deep_dive_feature_id = 1  # Try 0 through 5.
if MAIN:
    scores = feature_score_vector(feature_acts, deep_dive_feature_id)
if MAIN:
    train_auc = roc_auc_binary(scores[deep_dive_organism["train_mask"]], deep_dive_organism["labels"][deep_dive_organism["train_mask"]])
if MAIN:
    test_auc = roc_auc_binary(scores[heldout], deep_dive_organism["labels"][heldout])
if MAIN:
    top_ids = scores.topk(4).indices.tolist()
if MAIN:
    print(f"{deep_dive_organism['feature_names'][deep_dive_feature_id]}: train AUC={train_auc:.2f}, held-out AUC={test_auc:.2f}")
if MAIN:
    for i in top_ids:
        print(f"  {scores[i]:.2f}  {deep_dive_organism['texts'][i]}")

# %%

if MAIN:
    play_strengths = t.linspace(-2, 2, 9)
play_feature_id = 0
if MAIN:
    deep_dive_play_direction = sae.w_dec[play_feature_id]
if MAIN:
    play_residuals = steer_residuals(deep_dive_organism["residuals"], deep_dive_play_direction, play_strengths)
if MAIN:
    play_curve = ((play_residuals - sae.b_dec) @ readout).mean(dim=1) - baseline_mean
if MAIN:
    plt.figure(figsize=(6, 3))
if MAIN:
    plt.plot(play_strengths, play_curve, marker="o", color="#2a9d8f")
if MAIN:
    plt.axhline(0, color="#222222", linewidth=1)
if MAIN:
    plt.xlabel("steering strength")
if MAIN:
    plt.ylabel("mean readout change")
if MAIN:
    plt.title(f"Steering with F{play_feature_id}: {deep_dive_organism['feature_names'][play_feature_id]}")
if MAIN:
    plt.tight_layout()
if MAIN:
    plt.show()

# %%

suspicious_feature_id = 3
if MAIN:
    active_ids = (feature_acts[:, suspicious_feature_id] > 0).nonzero().flatten().tolist()
if MAIN:
    print(f"F{suspicious_feature_id} density = {density[suspicious_feature_id]:.3f}")
if MAIN:
    for i in active_ids:
        print(f"  {feature_acts[i, suspicious_feature_id]:.2f}  {deep_dive_organism['texts'][i]}")

# %%

if MAIN:
    gpu = utils.load_pinned_real_evidence(deep_dive_section_dir)
if MAIN:
    real = gpu["gemma_scope_real_activation_preflight"]
if MAIN:
    show_table(
        ["pinned real-model field", "value"],
        [
            ("artifact", gpu["gemma_scope_artifact_path"]),
            ("encoder / decoder", f"{gpu['gemma_scope_w_enc_shape']} / {gpu['gemma_scope_w_dec_shape']}"),
            ("train / held-out prompts", f"{real['train_prompt_count']} / {real['heldout_prompt_count']}"),
            ("selected / random feature", f"{real['selected_feature_id']} / {real['random_control_feature_id']}"),
            ("held-out target / random / shuffle AUC", f"{real['feature_auc']:.3f} / {real['baseline_auc']:.3f} / {real['label_shuffle_auc']:.3f}"),
            ("threshold accuracy", f"{real['threshold_accuracy']:.3f}"),
            ("positive / negative mean", f"{real['positive_mean']:.2f} / {real['negative_mean']:.2f}"),
            ("active features per prompt-token", f"{real['active_features_per_prompt_token']:.1f}"),
            ("prompted reconstruction MSE", f"{real['prompted_reconstruction_mse']:.1f} (not scale-normalized)"),
            ("peak CUDA memory", f"{gpu['peak_vram_gb']:.2f} GB (< 2.3 GB)"),
        ],
    )

# %%

def deep_dive_run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    from chapter6_sparse_feature_methods.exercises.part2_gemma_scope_deep_dive import solutions as reference_solutions

    return reference_solutions.run_gpu_test(max_vram_gb=max_vram_gb)


def deep_dive_run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    return deep_dive_run_gpu_test(max_vram_gb=max_vram_gb)

# %%

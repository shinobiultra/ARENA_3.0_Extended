# %%


from __future__ import annotations
MAIN = __name__ == "__main__"
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch as t

chapter = "chapter5_modern_architectures"
section = "part9_specialist_models"
legacy_section = "part6_multimodal_embedding_function_models"
if MAIN:
    root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if MAIN:
    exercises_dir = root_dir / chapter / "exercises"
if MAIN:
    legacy_section_dir = exercises_dir / legacy_section
if MAIN:
    section_dir = legacy_section_dir
if MAIN:
    canonical_section_dir = exercises_dir / section
if MAIN:
    assets_dir = canonical_section_dir / "media"
if MAIN:
    if str(root_dir) not in sys.path:
        sys.path.append(str(root_dir))
if MAIN:
    if str(exercises_dir) not in sys.path:
        sys.path.append(str(exercises_dir))

if MAIN:
    import part9_specialist_models.tests as tests
else:
    from . import tests


@dataclass(frozen=True)
class EmbeddingRetrievalReport:
    top1_accuracy: float
    mean_reciprocal_rank: float
    mean_positive_similarity: float
    mean_hard_negative_similarity: float
    mean_margin: float


@dataclass(frozen=True)
class RouterReport:
    overall_accuracy: float
    tool_accuracy: float
    abstention_accuracy: float
    hallucination_rate: float

# %%

TOOL_NAMES = ["get_weather", "create_calendar_event", "play_music"]
if MAIN:
    NO_CALL_ID = len(TOOL_NAMES)
if MAIN:
    LABEL_NAMES = TOOL_NAMES + ["NO_CALL"]

# The three coordinates are exact semantic axes, not learned representations.
if MAIN:
    TOOL_EMBEDDINGS = t.eye(3)
TOKEN_VECTORS = {
    "weather": [1.0, 0.0, 0.0],
    "forecast": [1.0, 0.0, 0.0],
    "temperature": [1.0, 0.0, 0.0],
    "rain": [1.0, 0.0, 0.0],
    "calendar": [0.0, 1.0, 0.0],
    "meeting": [0.0, 1.0, 0.0],
    "schedule": [0.0, 1.0, 0.0],
    "appointment": [0.0, 1.0, 0.0],
    "event": [0.0, 1.0, 0.0],
    "music": [0.0, 0.0, 1.0],
    "song": [0.0, 0.0, 1.0],
    "playlist": [0.0, 0.0, 1.0],
    "album": [0.0, 0.0, 1.0],
    "play": [0.0, 0.0, 1.0],
}

REQUESTS_AND_LABELS = [
    ("weather in tokyo", 0),
    ("will it rain in osaka", 0),
    ("temperature for sapporo", 0),
    ("show the kyoto forecast", 0),
    ("schedule lunch tomorrow", 1),
    ("add a meeting friday", 1),
    ("create a calendar event", 1),
    ("book a dentist appointment", 1),
    ("play blue train", 2),
    ("start my focus playlist", 2),
    ("put on some music", 2),
    ("play the album kind of blue", 2),
    ("tell me a joke", 3),
    ("explain photosynthesis", 3),
    ("weather or music", 3),
    ("calendar or playlist", 3),
]
if MAIN:
    REQUESTS = [request for request, _ in REQUESTS_AND_LABELS]
if MAIN:
    LABELS = t.tensor([label for _, label in REQUESTS_AND_LABELS])


def token_vectors_for_requests(requests: list[str]) -> tuple[t.Tensor, t.Tensor]:
    """Turn exact lexicon matches into padded token vectors and an attention mask."""
    tokenized = [re.findall(r"[a-z]+", request.lower()) for request in requests]
    max_length = max(map(len, tokenized))
    embeddings = t.zeros((len(requests), max_length, 3))
    attention_mask = t.zeros((len(requests), max_length), dtype=t.long)
    for row, tokens in enumerate(tokenized):
        attention_mask[row, : len(tokens)] = 1
        for column, token in enumerate(tokens):
            embeddings[row, column] = t.tensor(TOKEN_VECTORS.get(token, [0.0, 0.0, 0.0]))
    return embeddings, attention_mask


if MAIN:
    display(
        pd.DataFrame(
            {
                "request": REQUESTS,
                "ground truth": [LABEL_NAMES[label] for label in LABELS.tolist()],
            }
        )
    )

# %%

def mean_pool_embeddings(token_embeddings: t.Tensor, attention_mask: t.Tensor) -> t.Tensor:
    if token_embeddings.ndim != 3:
        raise ValueError("token_embeddings must have shape (batch, seq, dim).")
    if attention_mask.shape != token_embeddings.shape[:2]:
        raise ValueError("attention_mask must have shape (batch, seq).")
    mask = attention_mask.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
    weighted = token_embeddings * mask.unsqueeze(-1)
    denominator = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    return weighted.sum(dim=1) / denominator


if MAIN:
    tests.test_mean_pool_embeddings_ignores_padding_and_matches_reference(mean_pool_embeddings)

# %%

def l2_normalize(x: t.Tensor, *, eps: float = 1e-12) -> t.Tensor:
    if x.ndim == 0:
        raise ValueError("x must have at least one dimension.")
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def cosine_similarity_matrix(
    query_embeddings: t.Tensor,
    candidate_embeddings: t.Tensor,
) -> t.Tensor:
    if query_embeddings.ndim != 2 or candidate_embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (items, dim).")
    if query_embeddings.shape[1] != candidate_embeddings.shape[1]:
        raise ValueError("query and candidate embedding dimensions must match.")
    return l2_normalize(query_embeddings) @ l2_normalize(candidate_embeddings).T


def retrieval_ranks(similarity: t.Tensor, target_indices: t.Tensor) -> t.Tensor:
    if similarity.ndim != 2:
        raise ValueError("similarity must have shape (queries, candidates).")
    if target_indices.shape != (similarity.shape[0],):
        raise ValueError("target_indices must have shape (queries,).")
    order = similarity.argsort(dim=-1, descending=True)
    matches = order.eq(target_indices[:, None])
    return matches.float().argmax(dim=-1).long() + 1


def embedding_retrieval_report(
    query_embeddings: t.Tensor,
    candidate_embeddings: t.Tensor,
    target_indices: t.Tensor,
) -> EmbeddingRetrievalReport:
    similarity = cosine_similarity_matrix(query_embeddings, candidate_embeddings)
    ranks = retrieval_ranks(similarity, target_indices)
    row = t.arange(similarity.shape[0], device=similarity.device)
    positive = similarity[row, target_indices]
    masked = similarity.clone()
    masked[row, target_indices] = -t.inf
    hard_negative = masked.max(dim=-1).values
    return EmbeddingRetrievalReport(
        top1_accuracy=ranks.eq(1).float().mean().item(),
        mean_reciprocal_rank=(1.0 / ranks.float()).mean().item(),
        mean_positive_similarity=positive.mean().item(),
        mean_hard_negative_similarity=hard_negative.mean().item(),
        mean_margin=(positive - hard_negative).mean().item(),
    )


if MAIN:
    tests.test_retrieval_metrics_rank_pairs_and_hard_negative_margin(
        cosine_similarity_matrix,
        retrieval_ranks,
        embedding_retrieval_report,
    )

# %%

def route_with_abstention(
    similarity: t.Tensor,
    *,
    threshold: float,
    min_margin: float,
    no_call_id: int,
    allowed_tools: t.Tensor | None = None,
) -> t.Tensor:
    if similarity.ndim != 2 or similarity.shape[1] < 2:
        raise ValueError("similarity must have shape (requests, at least_two_tools).")
    if no_call_id < similarity.shape[1]:
        raise ValueError("no_call_id must not overlap a tool index.")

    scores = similarity.clone()
    if allowed_tools is not None:
        if allowed_tools.shape == (scores.shape[1],):
            allowed = allowed_tools.to(device=scores.device, dtype=t.bool).expand_as(scores)
        elif allowed_tools.shape == scores.shape:
            allowed = allowed_tools.to(device=scores.device, dtype=t.bool)
        else:
            raise ValueError("allowed_tools must have shape (tools,) or (requests, tools).")
        scores.masked_fill_(~allowed, -t.inf)

    top_values, top_indices = scores.topk(k=2, dim=-1)
    confident = top_values[:, 0] >= threshold
    unambiguous = (top_values[:, 0] - top_values[:, 1]) >= min_margin
    no_call = t.full_like(top_indices[:, 0], no_call_id)
    return t.where(confident & unambiguous, top_indices[:, 0], no_call)


if MAIN:
    tests.test_route_with_abstention_uses_confidence_margin_and_availability(route_with_abstention)

# %%

def routing_report(
    predictions: t.Tensor,
    labels: t.Tensor,
    *,
    no_call_id: int,
) -> RouterReport:
    if predictions.shape != labels.shape or predictions.ndim != 1:
        raise ValueError("predictions and labels must be one-dimensional and have equal shape.")
    tool_mask = labels.ne(no_call_id)
    no_call_mask = labels.eq(no_call_id)
    if not tool_mask.any() or not no_call_mask.any():
        raise ValueError("labels must contain both tool and no-call examples.")

    correct = predictions.eq(labels)
    return RouterReport(
        overall_accuracy=correct.float().mean().item(),
        tool_accuracy=correct[tool_mask].float().mean().item(),
        abstention_accuracy=correct[no_call_mask].float().mean().item(),
        hallucination_rate=predictions[no_call_mask].ne(no_call_id).float().mean().item(),
    )


if MAIN:
    tests.test_routing_report_separates_selection_and_abstention(routing_report)

# %%

if MAIN:
    token_embeddings, attention_mask = token_vectors_for_requests(REQUESTS)
if MAIN:
    query_embeddings = mean_pool_embeddings(token_embeddings, attention_mask)
if MAIN:
    similarity = cosine_similarity_matrix(query_embeddings, TOOL_EMBEDDINGS)

if MAIN:
    predictions = route_with_abstention(
        similarity,
        threshold=0.60,
        min_margin=0.15,
        no_call_id=NO_CALL_ID,
    )
if MAIN:
    router = routing_report(predictions, LABELS, no_call_id=NO_CALL_ID)

# Negative control: keep the tool names fixed but rotate the schema embeddings.
if MAIN:
    shuffled_similarity = cosine_similarity_matrix(query_embeddings, TOOL_EMBEDDINGS[[1, 2, 0]])
if MAIN:
    shuffled_predictions = route_with_abstention(
        shuffled_similarity,
        threshold=0.60,
        min_margin=0.15,
        no_call_id=NO_CALL_ID,
    )
if MAIN:
    shuffled = routing_report(shuffled_predictions, LABELS, no_call_id=NO_CALL_ID)

# Baseline: always choose the highest-scoring tool, even for ties or zero evidence.
if MAIN:
    always_call_predictions = similarity.argmax(dim=-1)
if MAIN:
    always_call = routing_report(always_call_predictions, LABELS, no_call_id=NO_CALL_ID)

if MAIN:
    results = pd.DataFrame(
        {
            "request": REQUESTS,
            "ground truth": [LABEL_NAMES[index] for index in LABELS.tolist()],
            "router prediction": [LABEL_NAMES[index] for index in predictions.tolist()],
            "top score": similarity.max(dim=-1).values.tolist(),
            "top-two margin": (similarity.topk(2, dim=-1).values[:, 0] - similarity.topk(2, dim=-1).values[:, 1]).tolist(),
        }
    )
if MAIN:
    display(results)

if MAIN:
    fig = plt.figure(figsize=(15, 11), constrained_layout=True)
if MAIN:
    grid = fig.add_gridspec(2, 2, width_ratios=[1.8, 1.0])
if MAIN:
    ax_heatmap = fig.add_subplot(grid[:, 0])
if MAIN:
    image = ax_heatmap.imshow(similarity.numpy(), cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
if MAIN:
    ax_heatmap.set_xticks(range(len(TOOL_NAMES)), TOOL_NAMES, rotation=18, ha="right")
if MAIN:
    row_labels = [
        f"{request}  ->  {LABEL_NAMES[prediction]}"
        for request, prediction in zip(REQUESTS, predictions.tolist())
    ]
if MAIN:
    ax_heatmap.set_yticks(range(len(REQUESTS)), row_labels)
if MAIN:
    ax_heatmap.set_title("Exact request-to-tool cosine similarity", loc="left", fontweight="bold")
if MAIN:
    for row in range(similarity.shape[0]):
        for column in range(similarity.shape[1]):
            value = similarity[row, column].item()
            ax_heatmap.text(column, row, f"{value:.2f}", ha="center", va="center", color="white" if value > 0.55 else "#17202a", fontsize=8)
if MAIN:
    fig.colorbar(image, ax=ax_heatmap, fraction=0.025, pad=0.02, label="cosine similarity")

if MAIN:
    ax_tool = fig.add_subplot(grid[0, 1])
if MAIN:
    tool_values = [router.tool_accuracy, shuffled.tool_accuracy]
if MAIN:
    ax_tool.bar(["router", "shuffled schema"], tool_values, color=["#137c8b", "#c44e52"])
if MAIN:
    ax_tool.set_ylim(0, 1.08)
if MAIN:
    ax_tool.set_ylabel("tool selection accuracy")
if MAIN:
    ax_tool.set_title("Schema-label control", loc="left", fontweight="bold")
if MAIN:
    for index, value in enumerate(tool_values):
        ax_tool.text(index, value + 0.03, f"{value:.2f}", ha="center", fontweight="bold")

if MAIN:
    ax_no_call = fig.add_subplot(grid[1, 1])
if MAIN:
    positions = t.arange(2).numpy()
width = 0.34
if MAIN:
    ax_no_call.bar(positions - width / 2, [router.abstention_accuracy, always_call.abstention_accuracy], width, label="abstention accuracy", color="#137c8b")
if MAIN:
    ax_no_call.bar(positions + width / 2, [router.hallucination_rate, always_call.hallucination_rate], width, label="hallucination rate", color="#d18f00")
if MAIN:
    ax_no_call.set_xticks(positions, ["router", "always call"])
if MAIN:
    ax_no_call.set_ylim(0, 1.08)
if MAIN:
    ax_no_call.set_title("No-call baseline", loc="left", fontweight="bold")
if MAIN:
    ax_no_call.legend(loc="upper center", bbox_to_anchor=(0.5, -0.10), ncol=1)

if MAIN:
    fig.suptitle("An embedding router works only when its schema and abstention rule work", fontsize=16, fontweight="bold")
if MAIN:
    signature_path = assets_dir / "embedding_tool_router_signature.png"
if MAIN:
    fig.savefig(signature_path, dpi=170, bbox_inches="tight")
if MAIN:
    plt.close(fig)

if MAIN:
    assert router.tool_accuracy == 1.0
if MAIN:
    assert router.abstention_accuracy == 1.0
if MAIN:
    assert shuffled.tool_accuracy == 0.0
if MAIN:
    assert always_call.hallucination_rate == 1.0

# %%

PLAY_REQUESTS = [
    "weather in nagoya",
    "calendar or music",
    "translate this paragraph",
]
PLAY_THRESHOLD = 0.60
PLAY_MIN_MARGIN = 0.15
if MAIN:
    PLAY_ALLOWED_TOOLS = t.tensor([True, True, False])

if MAIN:
    play_token_embeddings, play_attention_mask = token_vectors_for_requests(PLAY_REQUESTS)
if MAIN:
    play_embeddings = mean_pool_embeddings(play_token_embeddings, play_attention_mask)
if MAIN:
    play_similarity = cosine_similarity_matrix(play_embeddings, TOOL_EMBEDDINGS)
if MAIN:
    play_predictions = route_with_abstention(
        play_similarity,
        threshold=PLAY_THRESHOLD,
        min_margin=PLAY_MIN_MARGIN,
        no_call_id=NO_CALL_ID,
        allowed_tools=PLAY_ALLOWED_TOOLS,
    )
if MAIN:
    display(
        pd.DataFrame(
            {
                "request": PLAY_REQUESTS,
                "scores": [[round(value, 3) for value in row] for row in play_similarity.tolist()],
                "prediction": [LABEL_NAMES[index] for index in play_predictions.tolist()],
            }
        )
    )

# %%

ANOMALY_REQUESTS = [
    "umbrella conditions in kobe",
    "play music and check weather",
    "reserve time for a concert",
]
if MAIN:
    anomaly_tokens, anomaly_mask = token_vectors_for_requests(ANOMALY_REQUESTS)
if MAIN:
    anomaly_embeddings = mean_pool_embeddings(anomaly_tokens, anomaly_mask)
if MAIN:
    anomaly_similarity = cosine_similarity_matrix(anomaly_embeddings, TOOL_EMBEDDINGS)
if MAIN:
    anomaly_predictions = route_with_abstention(
        anomaly_similarity,
        threshold=0.60,
        min_margin=0.15,
        no_call_id=NO_CALL_ID,
    )
if MAIN:
    display(
        pd.DataFrame(
            {
                "request": ANOMALY_REQUESTS,
                "prediction": [LABEL_NAMES[index] for index in anomaly_predictions.tolist()],
                "reason to inspect": [
                    "A synonym absent from the lexicon causes false abstention.",
                    "Repeated music words can hide a multi-intent request.",
                    "The request implies two concepts without using either toy keyword.",
                ],
            }
        )
    )

# %%

REAL_BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"
REAL_BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
REAL_QUERIES = [
    "Represent this sentence for searching relevant passages: how to mask unavailable function calls in a tool API",
    "Represent this sentence for searching relevant passages: find documents about nearest-neighbor text embedding retrieval",
    "Represent this sentence for searching relevant passages: diagnose object hallucination in a vision language model",
    "Represent this sentence for searching relevant passages: measure abstention when no function should be called",
]
REAL_DOCUMENTS = [
    "Tool schemas should mask unavailable functions before selecting an API call.",
    "Embedding systems are evaluated with paired query document retrieval and hard negatives.",
    "Visual language models can hallucinate objects when text priors beat visual evidence.",
    "No-call examples measure whether a function-calling model abstains instead of inventing a tool.",
]


def run_real_bge_on_cpu() -> tuple[pd.DataFrame, EmbeddingRetrievalReport, EmbeddingRetrievalReport]:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        REAL_BGE_MODEL_ID,
        revision=REAL_BGE_REVISION,
        device="cpu",
    )
    query_embeddings = model.encode(
        REAL_QUERIES,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    document_embeddings = model.encode(
        REAL_DOCUMENTS,
        normalize_embeddings=True,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    similarity = cosine_similarity_matrix(query_embeddings, document_embeddings)
    targets = t.arange(len(REAL_QUERIES))
    report = embedding_retrieval_report(query_embeddings, document_embeddings, targets)
    permuted = embedding_retrieval_report(
        query_embeddings,
        document_embeddings,
        t.tensor([1, 2, 3, 0]),
    )
    rows = pd.DataFrame(
        {
            "query": REAL_QUERIES,
            "retrieved document": [REAL_DOCUMENTS[index] for index in similarity.argmax(dim=-1).tolist()],
            "paired rank": retrieval_ranks(similarity, targets).tolist(),
        }
    )
    return rows, report, permuted


if MAIN:
    RUN_REAL_MODEL = __name__ == "__main__"
if MAIN:
    if RUN_REAL_MODEL:
        real_rows, real_report, real_permuted = run_real_bge_on_cpu()
        display(real_rows)
        display(
            pd.DataFrame(
                [
                    {"condition": "paired", "top-1": real_report.top1_accuracy, "MRR": real_report.mean_reciprocal_rank, "mean margin": real_report.mean_margin},
                    {"condition": "permuted targets", "top-1": real_permuted.top1_accuracy, "MRR": real_permuted.mean_reciprocal_rank, "mean margin": real_permuted.mean_margin},
                ]
            )
        )
    else:
        print("Set RUN_REAL_MODEL = True to run the pinned BGE comparison on CPU.")

# %%

# These wrappers preserve the section's pinned CUDA evidence path. They are not the lesson result.
if MAIN:
    verification_report_path = section_dir / "verification_report.json"


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    from chapter5_modern_architectures.exercises.part6_multimodal_embedding_function_models.solutions import run_gpu_test as run

    return run(max_vram_gb=max_vram_gb)


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    return run_gpu_test(max_vram_gb=max_vram_gb)

# %%

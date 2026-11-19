# %%
"""Reference solutions for [5.6] Embedding Retrieval and Function-Calling Controls."""

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import torch as t

chapter = "chapter5_modern_architectures"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from arena_ext.gated_artifacts import (
    EMBEDDINGGEMMA_300M,
    FUNCTIONGEMMA_270M_IT_BASE,
    hf_model_artifact_access_report,
)

MAIN = __name__ == "__main__"

REAL_BGE_MODEL_ID = "BAAI/bge-small-en-v1.5"
REAL_BGE_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
REAL_EMBEDDINGGEMMA_MODEL_ID = EMBEDDINGGEMMA_300M.repo_id
REAL_EMBEDDINGGEMMA_REVISION = EMBEDDINGGEMMA_300M.revision
REAL_BGE_QUERIES = (
    "Represent this sentence for searching relevant passages: how to mask unavailable function calls in a tool API",
    "Represent this sentence for searching relevant passages: find documents about nearest-neighbor text embedding retrieval",
    "Represent this sentence for searching relevant passages: diagnose object hallucination in a vision language model",
    "Represent this sentence for searching relevant passages: measure abstention when no function should be called",
)
REAL_BGE_DOCUMENTS = (
    "Tool schemas should mask unavailable functions before selecting an API call.",
    "Embedding systems are evaluated with paired query document retrieval and hard negatives.",
    "Visual language models can hallucinate objects when text priors beat visual evidence.",
    "No-call examples measure whether a function-calling model abstains instead of inventing a tool.",
)
REAL_FUNCTIONGEMMA_MODEL_ID = "litert-community/FunctionGemma_270M_Mobile_Actions"
REAL_FUNCTIONGEMMA_REVISION = "e2226c1def35c5443942ebdb90a1da2a9eda836a"
MOBILE_ACTIONS_DATASET_ID = "google/mobile-actions"
MOBILE_ACTIONS_DATASET_REVISION = "e920309bc2acbc2e99a5e3201cf37df2b9fd9151"
FUNCTIONGEMMA_EVAL_EXAMPLE_COUNT = 32


@dataclass(frozen=True)
class EmbeddingRetrievalReport:
    top1_accuracy: float
    mean_reciprocal_rank: float
    mean_positive_similarity: float
    mean_hard_negative_similarity: float
    mean_margin: float


@dataclass(frozen=True)
class CentroidProbe:
    labels: t.Tensor
    centroids: t.Tensor


@dataclass(frozen=True)
class FunctionCallReport:
    accuracy: float
    tool_accuracy: float
    abstention_accuracy: float
    hallucination_rate: float


@dataclass(frozen=True)
class RouterReport:
    overall_accuracy: float
    tool_accuracy: float
    abstention_accuracy: float
    hallucination_rate: float


@dataclass(frozen=True)
class ParsedFunctionCall:
    name: str | None
    arguments: dict[str, str]


_FUNCTION_CALL_RE = re.compile(r"call:([A-Za-z0-9_]+)\{([^}]*)\}")
_FUNCTION_ARG_RE = re.compile(r"([A-Za-z0-9_]+):(?:<escape>(.*?)<escape>|([^,{}]+))")


# %%
def parse_function_call_text(text: str) -> ParsedFunctionCall:
    """Parse the first FunctionGemma-style `call:name{arg:value}` span."""

    match = _FUNCTION_CALL_RE.search(text)
    if match is None:
        return ParsedFunctionCall(name=None, arguments={})

    arguments: dict[str, str] = {}
    for arg_match in _FUNCTION_ARG_RE.finditer(match.group(2)):
        escaped_value, bare_value = arg_match.group(2), arg_match.group(3)
        value = escaped_value if escaped_value is not None else bare_value
        arguments[arg_match.group(1)] = value.strip()
    return ParsedFunctionCall(name=match.group(1), arguments=arguments)


def l2_normalize(x: t.Tensor, *, eps: float = 1e-12) -> t.Tensor:
    """Normalize vectors along the final dimension."""

    if x.ndim == 0:
        raise ValueError("x must have at least one dimension.")
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def mean_pool_embeddings(token_embeddings: t.Tensor, attention_mask: t.Tensor) -> t.Tensor:
    """Mean-pool token embeddings over unmasked positions."""

    if token_embeddings.ndim != 3:
        raise ValueError("token_embeddings must have shape (batch, seq, dim).")
    if attention_mask.shape != token_embeddings.shape[:2]:
        raise ValueError("attention_mask must have shape (batch, seq).")

    mask = attention_mask.to(device=token_embeddings.device, dtype=token_embeddings.dtype)
    weighted = token_embeddings * mask.unsqueeze(-1)
    denom = mask.sum(dim=-1, keepdim=True).clamp_min(1)
    return weighted.sum(dim=1) / denom


def cosine_similarity_matrix(
    query_embeddings: t.Tensor,
    candidate_embeddings: t.Tensor,
) -> t.Tensor:
    """Pairwise cosine similarities between query and candidate embeddings."""

    if query_embeddings.ndim != 2 or candidate_embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (items, dim).")
    if query_embeddings.shape[1] != candidate_embeddings.shape[1]:
        raise ValueError("query and candidate embedding dimensions must match.")

    query = l2_normalize(query_embeddings)
    candidates = l2_normalize(candidate_embeddings)
    return query @ candidates.T


def retrieval_ranks(similarity: t.Tensor, target_indices: t.Tensor) -> t.Tensor:
    """Return the one-indexed rank of each paired target candidate."""

    if similarity.ndim != 2:
        raise ValueError("similarity must have shape (queries, candidates).")
    if target_indices.shape != (similarity.shape[0],):
        raise ValueError("target_indices must have shape (queries,).")
    if target_indices.min() < 0 or target_indices.max() >= similarity.shape[1]:
        raise ValueError("target_indices contains an invalid candidate index.")

    order = similarity.argsort(dim=-1, descending=True)
    matches = order.eq(target_indices[:, None])
    return matches.float().argmax(dim=-1).long() + 1


def route_with_abstention(
    similarity: t.Tensor,
    *,
    threshold: float,
    min_margin: float,
    no_call_id: int,
    allowed_tools: t.Tensor | None = None,
) -> t.Tensor:
    """Choose a tool only when its score and lead over the runner-up are sufficient."""

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


def routing_report(
    predictions: t.Tensor,
    labels: t.Tensor,
    *,
    no_call_id: int,
) -> RouterReport:
    """Separate tool selection from abstention and hallucination behavior."""

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


def embedding_retrieval_report(
    query_embeddings: t.Tensor,
    candidate_embeddings: t.Tensor,
    target_indices: t.Tensor,
) -> EmbeddingRetrievalReport:
    """Summarize paired retrieval quality and hard-negative margins."""

    similarity = cosine_similarity_matrix(query_embeddings, candidate_embeddings)
    ranks = retrieval_ranks(similarity, target_indices)
    batch = similarity.shape[0]
    row = t.arange(batch, device=similarity.device)
    positive = similarity[row, target_indices]

    if similarity.shape[1] == 1:
        hard_negative = positive.new_zeros(positive.shape)
    else:
        masked_similarity = similarity.clone()
        masked_similarity[row, target_indices] = -t.inf
        hard_negative = masked_similarity.max(dim=-1).values

    return EmbeddingRetrievalReport(
        top1_accuracy=ranks.eq(1).float().mean().item(),
        mean_reciprocal_rank=(1.0 / ranks.float()).mean().item(),
        mean_positive_similarity=positive.mean().item(),
        mean_hard_negative_similarity=hard_negative.mean().item(),
        mean_margin=(positive - hard_negative).mean().item(),
    )


def fit_centroid_probe(embeddings: t.Tensor, labels: t.Tensor) -> CentroidProbe:
    """Fit a nearest-centroid probe in normalized embedding space."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings must have shape (items, dim).")
    if labels.shape != (embeddings.shape[0],):
        raise ValueError("labels must have shape (items,).")

    unique_labels = labels.unique(sorted=True)
    centroids = t.stack([embeddings[labels == label].mean(dim=0) for label in unique_labels])
    return CentroidProbe(labels=unique_labels, centroids=l2_normalize(centroids))


def predict_centroid_probe(embeddings: t.Tensor, probe: CentroidProbe) -> t.Tensor:
    """Predict labels by nearest normalized centroid."""

    similarity = l2_normalize(embeddings) @ probe.centroids.to(embeddings.device).T
    centroid_ids = similarity.argmax(dim=-1)
    return probe.labels.to(embeddings.device)[centroid_ids]


def centroid_probe_accuracy(embeddings: t.Tensor, labels: t.Tensor, probe: CentroidProbe) -> float:
    predictions = predict_centroid_probe(embeddings, probe)
    return predictions.eq(labels).float().mean().item()


def mask_disallowed_tools(logits: t.Tensor, allowed_tools: t.Tensor) -> t.Tensor:
    """Set unavailable tool logits to negative infinity before selection."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, tools).")
    if allowed_tools.shape == (logits.shape[1],):
        allowed = allowed_tools.to(device=logits.device, dtype=t.bool).expand_as(logits)
    elif allowed_tools.shape == logits.shape:
        allowed = allowed_tools.to(device=logits.device, dtype=t.bool)
    else:
        raise ValueError("allowed_tools must have shape (tools,) or (batch, tools).")

    return logits.masked_fill(~allowed, -t.inf)


def function_call_report(
    logits: t.Tensor,
    labels: t.Tensor,
    *,
    no_call_id: int,
) -> FunctionCallReport:
    """Measure tool-choice accuracy and no-call hallucination rate separately."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape (batch, tools).")
    if labels.shape != (logits.shape[0],):
        raise ValueError("labels must have shape (batch,).")
    if not 0 <= no_call_id < logits.shape[1]:
        raise ValueError("no_call_id is out of range.")

    predictions = logits.argmax(dim=-1)
    tool_mask = labels.ne(no_call_id)
    abstain_mask = labels.eq(no_call_id)
    accuracy = predictions.eq(labels).float().mean().item()

    if tool_mask.any():
        tool_accuracy = predictions[tool_mask].eq(labels[tool_mask]).float().mean().item()
    else:
        tool_accuracy = float("nan")
    if abstain_mask.any():
        abstention_accuracy = predictions[abstain_mask].eq(no_call_id).float().mean().item()
        hallucination_rate = predictions[abstain_mask].ne(no_call_id).float().mean().item()
    else:
        abstention_accuracy = float("nan")
        hallucination_rate = float("nan")

    return FunctionCallReport(
        accuracy=accuracy,
        tool_accuracy=tool_accuracy,
        abstention_accuracy=abstention_accuracy,
        hallucination_rate=hallucination_rate,
    )


def schema_token_attribution(hidden_states: t.Tensor, schema_vectors: t.Tensor) -> t.Tensor:
    """Project hidden states onto schema-token directions."""

    if hidden_states.shape[-1] != schema_vectors.shape[-1]:
        raise ValueError("hidden state and schema vector dimensions must match.")
    return hidden_states @ schema_vectors.T

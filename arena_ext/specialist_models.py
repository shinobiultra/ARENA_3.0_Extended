"""Utilities for specialist embedding and function-calling model labs."""

from __future__ import annotations

import re
from dataclasses import dataclass

import torch as t


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
class ParsedFunctionCall:
    name: str | None
    arguments: dict[str, str]


_FUNCTION_CALL_RE = re.compile(r"call:([A-Za-z0-9_]+)\{([^}]*)\}")
_FUNCTION_ARG_RE = re.compile(r"([A-Za-z0-9_]+):(?:<escape>(.*?)<escape>|([^,{}]+))")


def parse_function_call_text(text: str) -> ParsedFunctionCall:
    """Parse the first FunctionGemma-style call from generated text."""

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
    """Return one-indexed rank of the target candidate for each query."""

    if similarity.ndim != 2:
        raise ValueError("similarity must have shape (queries, candidates).")
    if target_indices.shape != (similarity.shape[0],):
        raise ValueError("target_indices must have shape (queries,).")
    if target_indices.min() < 0 or target_indices.max() >= similarity.shape[1]:
        raise ValueError("target_indices contains an invalid candidate index.")

    order = similarity.argsort(dim=-1, descending=True)
    matches = order.eq(target_indices[:, None])
    return matches.float().argmax(dim=-1).long() + 1


def embedding_retrieval_report(
    query_embeddings: t.Tensor,
    candidate_embeddings: t.Tensor,
    target_indices: t.Tensor,
) -> EmbeddingRetrievalReport:
    """Summarize retrieval quality for paired embedding examples."""

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
    """Fit a nearest-centroid probe in embedding space."""

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
    """Set disallowed tool logits to negative infinity."""

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
    """Measure tool-choice accuracy and no-call hallucination rate."""

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

"""Student-implemented token gathering and controlled native measurements."""

import torch
from torch import Tensor


def last_token_rows(hidden: Tensor, attention_mask: Tensor) -> Tensor:
    """Gather the last valid token from each padded (batch, tokens, features) row."""
    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2] or min(hidden.shape) == 0:
        raise ValueError("Use nonempty hidden states and their matching token mask.")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("The attention mask must contain only zeros and ones.")
    positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    last = torch.where(attention_mask.bool(), positions, -1).max(dim=1).values
    if (last < 0).any():
        raise ValueError("Every example must contain a valid token.")
    return hidden[torch.arange(len(hidden), device=hidden.device), last]


def paired_opinion_change(scores: Tensor, rows: list[dict]) -> dict:
    """Change only Morgan's opinion, matching by base ID rather than row order."""
    if scores.shape != (len(rows),) or not rows or not torch.isfinite(scores).all():
        raise ValueError("Use one finite score per nonempty row.")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Scores must lie in [0, 1].")
    groups = {}
    for index, row in enumerate(rows):
        opinion = row["distractor"]
        if opinion not in (0, 1):
            raise ValueError("This measurement requires two explicit distracting opinions.")
        group = groups.setdefault(row["base_id"], {})
        if opinion in group:
            raise ValueError("A base review has a duplicate opinion row.")
        group[opinion] = index
    negative, positive, ids = [], [], []
    for base_id in sorted(groups):
        group = groups[base_id]
        if set(group) != {0, 1}:
            raise ValueError("Every base review needs both opinions.")
        a, b = rows[group[0]], rows[group[1]]
        if any(a[key] != b[key] for key in ("review", "label", "wrapper", "variant")):
            raise ValueError("Only the distracting opinion may change within a pair.")
        ids.append(base_id)
        negative.append(scores[group[0]])
        positive.append(scores[group[1]])
    negative, positive = torch.stack(negative), torch.stack(positive)
    difference = positive - negative
    return {"base_ids": ids, "negative_opinion_scores": negative, "positive_opinion_scores": positive,
            "score_changes": difference, "base_reviews": len(ids),
            "prediction_flip_fraction": float(((negative > .5) != (positive > .5)).double().mean()),
            "mean_absolute_score_change": float(difference.abs().mean())}


def model_answer_metrics(logits: Tensor, candidate_token_ids: list[int], labels: Tensor) -> dict:
    """Keep a two-answer classification distinct from unrestricted next-token output."""
    if logits.ndim != 2 or labels.shape != (len(logits),) or not len(logits) or not torch.isfinite(logits).all():
        raise ValueError("Use finite full-vocabulary logits and one label per row.")
    if len(candidate_token_ids) != 2 or len(set(candidate_token_ids)) != 2:
        raise ValueError("Specify distinct negative and positive candidate tokens, in that order.")
    if any(index < 0 or index >= logits.shape[-1] for index in candidate_token_ids):
        raise ValueError("A candidate token is outside the vocabulary.")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("Labels must be binary.")
    log_probability = logits.double().log_softmax(-1)
    candidate_logits = logits[:, candidate_token_ids].double()
    restricted_positive = candidate_logits.softmax(-1)[:, 1]
    candidate_probabilities = log_probability[:, candidate_token_ids].exp()
    winners = logits.argmax(-1)
    targets = torch.tensor(candidate_token_ids, device=logits.device)[labels.long()]
    return {"count": len(logits), "restricted_positive_scores": restricted_positive,
            "candidate_probabilities": candidate_probabilities, "candidate_mass": candidate_probabilities.sum(-1),
            "winning_token_ids": winners,
            "restricted_correct": int(((restricted_positive > .5) == labels).sum()),
            "full_vocabulary_correct": int((winners == targets).sum()),
            "target_full_vocabulary_nll": float(-log_probability[torch.arange(len(logits)), targets].mean())}

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


def embedding_pooling_smoke_test() -> dict:
    token_embeddings = t.tensor(
        [
            [[1.0, 0.0], [3.0, 2.0], [100.0, 100.0]],
            [[2.0, 4.0], [6.0, 8.0], [0.0, 0.0]],
        ]
    )
    attention_mask = t.tensor([[1, 1, 0], [1, 1, 0]])
    pooled = mean_pool_embeddings(token_embeddings, attention_mask)
    return {"pooled": pooled.tolist()}


def retrieval_smoke_test() -> dict:
    queries = t.eye(3)
    candidates = t.eye(3)
    targets = t.tensor([0, 1, 2])
    similarity = cosine_similarity_matrix(queries, candidates)
    ranks = retrieval_ranks(similarity, targets)
    report = embedding_retrieval_report(queries, candidates, targets)
    return {**report.__dict__, "ranks": ranks.tolist()}


def centroid_probe_smoke_test() -> dict:
    train = t.tensor(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ]
    )
    train_labels = t.tensor([0, 0, 1, 1])
    test = t.tensor([[0.95, 0.05], [0.05, 0.95]])
    test_labels = t.tensor([0, 1])
    probe = fit_centroid_probe(train, train_labels)
    predictions = predict_centroid_probe(test, probe)
    return {
        "predictions": predictions.tolist(),
        "accuracy": centroid_probe_accuracy(test, test_labels, probe),
    }


def tool_masking_smoke_test() -> dict:
    logits = t.tensor([[1.0, 10.0, 2.0]])
    allowed_tools = t.tensor([True, False, True])
    masked = mask_disallowed_tools(logits, allowed_tools)
    return {
        "masked_disallowed": bool(t.isneginf(masked[0, 1])),
        "prediction": int(masked.argmax(dim=-1).item()),
    }


def function_call_smoke_test() -> dict:
    logits = t.tensor(
        [
            [5.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
            [0.0, 3.0, 1.0],
            [0.0, 1.0, 5.0],
        ]
    )
    labels = t.tensor([0, 1, 2, 2])
    return function_call_report(logits, labels, no_call_id=2).__dict__


def router_smoke_test() -> dict:
    similarity = t.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [2**-0.5, 0.0, 2**-0.5],
        ]
    )
    labels = t.tensor([0, 1, 2, 3, 3])
    predictions = route_with_abstention(
        similarity,
        threshold=0.6,
        min_margin=0.15,
        no_call_id=3,
    )
    return {
        "predictions": predictions.tolist(),
        **routing_report(predictions, labels, no_call_id=3).__dict__,
    }


def schema_attribution_smoke_test() -> dict:
    hidden_states = t.tensor([[1.0, 0.0], [0.0, 1.0]])
    schema_vectors = t.tensor([[0.0, 1.0], [1.0, 0.0]])
    attribution = schema_token_attribution(hidden_states, schema_vectors)
    return {
        "attribution": attribution.tolist(),
        "top_schema_ids": attribution.argmax(dim=-1).tolist(),
    }


def _load_mobile_actions_eval_rows(example_count: int) -> list[dict]:
    from huggingface_hub import hf_hub_download

    dataset_path = hf_hub_download(
        MOBILE_ACTIONS_DATASET_ID,
        "dataset.jsonl",
        repo_type="dataset",
        revision=MOBILE_ACTIONS_DATASET_REVISION,
    )
    rows: list[dict] = []
    with open(dataset_path) as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("metadata") == "eval":
                rows.append(row)
                if len(rows) >= example_count:
                    break
    if len(rows) != example_count:
        raise RuntimeError(
            f"expected {example_count} Mobile Actions eval rows, found {len(rows)}"
        )
    return rows


def _expected_first_tool_call(row: dict) -> tuple[str, dict[str, str]]:
    assistant_message = row["messages"][-1]
    function = assistant_message["tool_calls"][0]["function"]
    arguments = {key: str(value) for key, value in function.get("arguments", {}).items()}
    return function["name"], arguments


def _required_argument_names(row: dict, function_name: str) -> list[str]:
    for tool in row["tools"]:
        function = tool["function"]
        if function["name"] == function_name:
            parameters = function.get("parameters", {})
            return list(parameters.get("required", []) or [])
    return []


def functiongemma_mobile_actions_preflight(
    max_vram_gb: float = 24.0,
    *,
    example_count: int = FUNCTIONGEMMA_EVAL_EXAMPLE_COUNT,
) -> dict:
    """Evaluate a pinned public FunctionGemma checkpoint on real Mobile Actions rows."""

    if not t.cuda.is_available():
        raise RuntimeError("FunctionGemma Mobile Actions preflight requires CUDA.")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rows = _load_mobile_actions_eval_rows(example_count)
    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        REAL_FUNCTIONGEMMA_MODEL_ID,
        revision=REAL_FUNCTIONGEMMA_REVISION,
    )
    model = AutoModelForCausalLM.from_pretrained(
        REAL_FUNCTIONGEMMA_MODEL_ID,
        revision=REAL_FUNCTIONGEMMA_REVISION,
        attn_implementation="eager",
        dtype=t.bfloat16,
    ).to(device)
    model.eval()
    end_call_token_id = tokenizer.encode("<end_function_call>", add_special_tokens=False)[0]

    parse_correct = 0
    function_correct = 0
    exact_arg_correct = 0
    required_arg_correct = 0
    prompt_token_counts: list[int] = []
    generated_token_counts: list[int] = []
    failures: list[dict] = []
    examples: list[dict] = []
    function_distribution: dict[str, int] = {}

    with t.inference_mode():
        for index, row in enumerate(rows):
            expected_name, expected_args = _expected_first_tool_call(row)
            function_distribution[expected_name] = function_distribution.get(expected_name, 0) + 1
            prompt = tokenizer.apply_chat_template(
                row["messages"][:-1],
                tools=row["tools"],
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output = model.generate(
                **inputs,
                max_new_tokens=96,
                do_sample=False,
                eos_token_id=[tokenizer.eos_token_id, end_call_token_id],
                pad_token_id=tokenizer.pad_token_id,
            )
            generated_ids = output[0, inputs["input_ids"].shape[1] :]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False).strip()
            parsed = parse_function_call_text(generated_text)

            parse_ok = parsed.name is not None
            function_ok = parsed.name == expected_name
            exact_args_ok = parsed.arguments == expected_args
            required_args = _required_argument_names(row, expected_name)
            required_args_ok = all(
                parsed.arguments.get(name) == expected_args.get(name)
                for name in required_args
            )
            parse_correct += int(parse_ok)
            function_correct += int(function_ok)
            exact_arg_correct += int(exact_args_ok)
            required_arg_correct += int(required_args_ok)
            prompt_token_counts.append(int(inputs["input_ids"].shape[-1]))
            generated_token_counts.append(int(generated_ids.shape[-1]))

            case = {
                "index": index,
                "expected_function": expected_name,
                "predicted_function": parsed.name,
                "required_arguments": required_args,
                "required_arguments_match": required_args_ok,
                "exact_arguments_match": exact_args_ok,
            }
            if len(examples) < 6:
                examples.append(case)
            if not (function_ok and required_args_ok):
                failures.append(
                    {
                        **case,
                        "expected_arguments": expected_args,
                        "predicted_arguments": parsed.arguments,
                    }
                )

    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    del model
    t.cuda.empty_cache()

    parse_accuracy = parse_correct / example_count
    function_name_accuracy = function_correct / example_count
    exact_argument_accuracy = exact_arg_correct / example_count
    required_argument_accuracy = required_arg_correct / example_count
    preflight_passed = (
        parse_accuracy == 1.0
        and function_name_accuracy == 1.0
        and exact_argument_accuracy >= 0.85
        and required_argument_accuracy >= 0.85
        and peak_vram_gb <= max_vram_gb
    )

    return {
        "cuda_available": True,
        "model_id": REAL_FUNCTIONGEMMA_MODEL_ID,
        "revision": REAL_FUNCTIONGEMMA_REVISION,
        "base_model": "google/functiongemma-270m-it",
        "dataset_id": MOBILE_ACTIONS_DATASET_ID,
        "dataset_revision": MOBILE_ACTIONS_DATASET_REVISION,
        "eval_example_count": example_count,
        "function_distribution": function_distribution,
        "parse_accuracy": parse_accuracy,
        "function_name_accuracy": function_name_accuracy,
        "exact_argument_accuracy": exact_argument_accuracy,
        "required_argument_accuracy": required_argument_accuracy,
        "failure_count": len(failures),
        "failure_indices": [failure["index"] for failure in failures],
        "failure_examples": failures[:4],
        "example_predictions": examples,
        "mean_prompt_tokens": sum(prompt_token_counts) / len(prompt_token_counts),
        "mean_generated_tokens": sum(generated_token_counts) / len(generated_token_counts),
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "preflight_passed": preflight_passed,
    }


def bge_embedding_retrieval_preflight(max_vram_gb: float = 24.0) -> dict:
    """Run a pinned public embedding model on controlled retrieval pairs."""

    if not t.cuda.is_available():
        raise RuntimeError("BGE embedding retrieval preflight requires CUDA.")

    from sentence_transformers import SentenceTransformer

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    model = SentenceTransformer(
        REAL_BGE_MODEL_ID,
        revision=REAL_BGE_REVISION,
        device=str(device),
    )
    with t.inference_mode():
        query_embeddings = model.encode(
            list(REAL_BGE_QUERIES),
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        ).to(device)
        document_embeddings = model.encode(
            list(REAL_BGE_DOCUMENTS),
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        ).to(device)
    targets = t.arange(len(REAL_BGE_QUERIES), device=device)
    similarity = cosine_similarity_matrix(query_embeddings, document_embeddings)
    ranks = retrieval_ranks(similarity, targets)
    report = embedding_retrieval_report(query_embeddings, document_embeddings, targets)

    permuted_targets = t.tensor([1, 2, 3, 0], device=device)
    permuted_report = embedding_retrieval_report(
        query_embeddings,
        document_embeddings,
        permuted_targets,
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    embedding_dim = int(query_embeddings.shape[-1])

    del query_embeddings, document_embeddings, similarity, model
    t.cuda.empty_cache()

    preflight_passed = (
        report.top1_accuracy == 1.0
        and report.mean_margin >= 0.1
        and permuted_report.top1_accuracy == 0.0
        and peak_vram_gb <= max_vram_gb
    )
    return {
        "cuda_available": True,
        "model_id": REAL_BGE_MODEL_ID,
        "revision": REAL_BGE_REVISION,
        "claim_scope": "pinned_public_bge_embedding_retrieval_comparison_preflight",
        "query_count": len(REAL_BGE_QUERIES),
        "document_count": len(REAL_BGE_DOCUMENTS),
        "embedding_dim": embedding_dim,
        "ranks": ranks.tolist(),
        "top1_accuracy": report.top1_accuracy,
        "mean_reciprocal_rank": report.mean_reciprocal_rank,
        "mean_positive_similarity": report.mean_positive_similarity,
        "mean_hard_negative_similarity": report.mean_hard_negative_similarity,
        "mean_margin": report.mean_margin,
        "permuted_top1_accuracy": permuted_report.top1_accuracy,
        "permuted_mean_margin": permuted_report.mean_margin,
        "permuted_control_fails": permuted_report.top1_accuracy == 0.0,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "preflight_passed": preflight_passed,
    }


def embeddinggemma_retrieval_preflight(max_vram_gb: float = 24.0) -> dict:
    """Run authenticated EmbeddingGemma on controlled retrieval pairs."""

    if not t.cuda.is_available():
        raise RuntimeError("EmbeddingGemma retrieval preflight requires CUDA.")

    access = hf_model_artifact_access_report(EMBEDDINGGEMMA_300M)
    if not access["ready_for_direct_loading"]:
        raise RuntimeError(f"EmbeddingGemma is not ready for direct loading: {access}")

    from sentence_transformers import SentenceTransformer

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    model = SentenceTransformer(
        REAL_EMBEDDINGGEMMA_MODEL_ID,
        revision=REAL_EMBEDDINGGEMMA_REVISION,
        device=str(device),
    )
    with t.inference_mode():
        query_embeddings = model.encode(
            list(REAL_BGE_QUERIES),
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        ).to(device)
        document_embeddings = model.encode(
            list(REAL_BGE_DOCUMENTS),
            normalize_embeddings=True,
            convert_to_tensor=True,
            show_progress_bar=False,
        ).to(device)
    targets = t.arange(len(REAL_BGE_QUERIES), device=device)
    similarity = cosine_similarity_matrix(query_embeddings, document_embeddings)
    ranks = retrieval_ranks(similarity, targets)
    report = embedding_retrieval_report(query_embeddings, document_embeddings, targets)
    permuted_report = embedding_retrieval_report(
        query_embeddings,
        document_embeddings,
        t.tensor([1, 2, 3, 0], device=device),
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    embedding_dim = int(query_embeddings.shape[-1])

    del query_embeddings, document_embeddings, similarity, model
    t.cuda.empty_cache()

    preflight_passed = (
        report.top1_accuracy == 1.0
        and report.mean_margin >= 0.1
        and permuted_report.top1_accuracy == 0.0
        and peak_vram_gb <= max_vram_gb
    )
    return {
        "cuda_available": True,
        "model_id": REAL_EMBEDDINGGEMMA_MODEL_ID,
        "revision": REAL_EMBEDDINGGEMMA_REVISION,
        "claim_scope": "authenticated_embeddinggemma_retrieval_preflight",
        "query_count": len(REAL_BGE_QUERIES),
        "document_count": len(REAL_BGE_DOCUMENTS),
        "embedding_dim": embedding_dim,
        "ranks": ranks.tolist(),
        "top1_accuracy": report.top1_accuracy,
        "mean_reciprocal_rank": report.mean_reciprocal_rank,
        "mean_positive_similarity": report.mean_positive_similarity,
        "mean_hard_negative_similarity": report.mean_hard_negative_similarity,
        "mean_margin": report.mean_margin,
        "permuted_top1_accuracy": permuted_report.top1_accuracy,
        "permuted_mean_margin": permuted_report.mean_margin,
        "permuted_control_fails": permuted_report.top1_accuracy == 0.0,
        "access_report": access,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "preflight_passed": preflight_passed,
    }


def functiongemma_base_cuda_preflight(max_vram_gb: float = 24.0) -> dict:
    """Directly load the gated base FunctionGemma model and run a benign forward pass."""

    if not t.cuda.is_available():
        raise RuntimeError("FunctionGemma base preflight requires CUDA.")

    access = hf_model_artifact_access_report(FUNCTIONGEMMA_270M_IT_BASE)
    if not access["ready_for_direct_loading"]:
        raise RuntimeError(f"FunctionGemma base is not ready for direct loading: {access}")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(
        FUNCTIONGEMMA_270M_IT_BASE.repo_id,
        revision=FUNCTIONGEMMA_270M_IT_BASE.revision,
    )
    model = AutoModelForCausalLM.from_pretrained(
        FUNCTIONGEMMA_270M_IT_BASE.repo_id,
        revision=FUNCTIONGEMMA_270M_IT_BASE.revision,
        dtype=t.bfloat16,
        device_map="cuda",
    )
    model.eval()
    inputs = tokenizer(
        "Return a JSON function-call schema for opening a settings screen.",
        return_tensors="pt",
    ).to(device)
    with t.inference_mode():
        outputs = model(**inputs)
    logits = outputs.logits
    forward_passed = bool(
        logits.ndim == 3
        and logits.shape[0] == 1
        and logits.shape[1] == inputs["input_ids"].shape[1]
        and t.isfinite(logits[..., :1024].float()).all().item()
    )
    t.cuda.synchronize()
    peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    del model, outputs, logits
    t.cuda.empty_cache()

    return {
        "cuda_available": True,
        "model_id": FUNCTIONGEMMA_270M_IT_BASE.repo_id,
        "revision": FUNCTIONGEMMA_270M_IT_BASE.revision,
        "prompt_token_count": int(inputs["input_ids"].shape[-1]),
        "vocab_size": int(tokenizer.vocab_size),
        "forward_passed": forward_passed,
        "access_report": access,
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": peak_vram_gb <= max_vram_gb,
        "preflight_passed": forward_passed and peak_vram_gb <= max_vram_gb,
    }


def run_smoke_test(cpu: bool = True) -> dict:
    _ = cpu
    return {
        "pooling": embedding_pooling_smoke_test(),
        "retrieval": retrieval_smoke_test(),
        "centroid_probe": centroid_probe_smoke_test(),
        "tool_masking": tool_masking_smoke_test(),
        "function_call": function_call_smoke_test(),
        "router": router_smoke_test(),
        "schema_attribution": schema_attribution_smoke_test(),
    }


def run_gpu_test(max_vram_gb: float = 24.0) -> dict:
    if not t.cuda.is_available():
        raise RuntimeError("Section 5.6 GPU verification requires CUDA.")

    device = t.device("cuda")
    t.cuda.reset_peak_memory_stats()
    queries = t.eye(3, device=device)
    candidates = t.eye(3, device=device)
    targets = t.tensor([0, 1, 2], device=device)
    retrieval = embedding_retrieval_report(queries, candidates, targets)
    logits = t.tensor([[1.0, 10.0, 2.0]], device=device)
    allowed_tools = t.tensor([True, False, True], device=device)
    masked = mask_disallowed_tools(logits, allowed_tools)
    t.cuda.synchronize()
    synthetic_peak_vram_gb = t.cuda.max_memory_allocated() / 1024**3
    bge = bge_embedding_retrieval_preflight(max_vram_gb=max_vram_gb)
    embeddinggemma = embeddinggemma_retrieval_preflight(max_vram_gb=max_vram_gb)
    functiongemma = functiongemma_mobile_actions_preflight(max_vram_gb=max_vram_gb)
    functiongemma_base = functiongemma_base_cuda_preflight(max_vram_gb=max_vram_gb)
    embeddinggemma_access = hf_model_artifact_access_report(EMBEDDINGGEMMA_300M)
    functiongemma_base_access = hf_model_artifact_access_report(FUNCTIONGEMMA_270M_IT_BASE)
    peak_vram_gb = max(
        synthetic_peak_vram_gb,
        bge["peak_vram_gb"],
        embeddinggemma["peak_vram_gb"],
        functiongemma["peak_vram_gb"],
        functiongemma_base["peak_vram_gb"],
    )
    return {
        "cuda_available": True,
        "device": t.cuda.get_device_name(0),
        "retrieval_top1_accuracy": retrieval.top1_accuracy,
        "synthetic_retrieval_top1_accuracy": retrieval.top1_accuracy,
        "retrieval_mean_margin": retrieval.mean_margin,
        "masked_disallowed": bool(t.isneginf(masked[0, 1]).item()),
        "synthetic_tool_masking_passed": bool(t.isneginf(masked[0, 1]).item()),
        "tool_prediction": int(masked.argmax(dim=-1).item()),
        "bge_preflight_passed": bge["preflight_passed"],
        "bge_retrieval_top1_accuracy": bge["top1_accuracy"],
        "bge_mean_reciprocal_rank": bge["mean_reciprocal_rank"],
        "bge_mean_margin": bge["mean_margin"],
        "bge_permuted_top1_accuracy": bge["permuted_top1_accuracy"],
        "bge_permuted_control_fails": bge["permuted_control_fails"],
        "bge_peak_vram_gb": bge["peak_vram_gb"],
        "bge_preflight": bge,
        "embeddinggemma_preflight_passed": embeddinggemma["preflight_passed"],
        "embeddinggemma_model_id": embeddinggemma["model_id"],
        "embeddinggemma_revision": embeddinggemma["revision"],
        "embeddinggemma_retrieval_top1_accuracy": embeddinggemma["top1_accuracy"],
        "embeddinggemma_mean_reciprocal_rank": embeddinggemma["mean_reciprocal_rank"],
        "embeddinggemma_mean_margin": embeddinggemma["mean_margin"],
        "embeddinggemma_permuted_top1_accuracy": embeddinggemma[
            "permuted_top1_accuracy"
        ],
        "embeddinggemma_permuted_control_fails": embeddinggemma[
            "permuted_control_fails"
        ],
        "embeddinggemma_peak_vram_gb": embeddinggemma["peak_vram_gb"],
        "embeddinggemma_preflight": embeddinggemma,
        "embeddinggemma_access_report": embeddinggemma_access,
        "embeddinggemma_authenticated": embeddinggemma_access["authenticated"],
        "embeddinggemma_repo_listed": embeddinggemma_access["repo_listed"],
        "embeddinggemma_remote_download_ready": embeddinggemma_access[
            "remote_download_ready"
        ],
        "embeddinggemma_missing_local_patterns": embeddinggemma_access[
            "missing_local_patterns"
        ],
        "embeddinggemma_missing_remote_patterns": embeddinggemma_access[
            "missing_remote_patterns"
        ],
        "embeddinggemma_local_non_ref_file_count": embeddinggemma_access[
            "local_non_ref_file_count"
        ],
        "embeddinggemma_ready_for_direct_loading": embeddinggemma_access[
            "ready_for_direct_loading"
        ],
        "embeddinggemma_gated_unavailable": not embeddinggemma_access[
            "ready_for_direct_loading"
        ],
        "embeddinggemma_auth_error_type": embeddinggemma_access["auth_error_type"],
        "embeddinggemma_access_error_type": embeddinggemma_access["access_error_type"],
        "functiongemma_preflight_passed": functiongemma["preflight_passed"],
        "functiongemma_model_id": functiongemma["model_id"],
        "functiongemma_revision": functiongemma["revision"],
        "functiongemma_dataset_id": functiongemma["dataset_id"],
        "functiongemma_dataset_revision": functiongemma["dataset_revision"],
        "functiongemma_eval_example_count": functiongemma["eval_example_count"],
        "functiongemma_parse_accuracy": functiongemma["parse_accuracy"],
        "functiongemma_function_name_accuracy": functiongemma["function_name_accuracy"],
        "functiongemma_exact_argument_accuracy": functiongemma["exact_argument_accuracy"],
        "functiongemma_required_argument_accuracy": functiongemma[
            "required_argument_accuracy"
        ],
        "functiongemma_failure_count": functiongemma["failure_count"],
        "functiongemma_failure_indices": functiongemma["failure_indices"],
        "functiongemma_mean_prompt_tokens": functiongemma["mean_prompt_tokens"],
        "functiongemma_mean_generated_tokens": functiongemma["mean_generated_tokens"],
        "functiongemma_peak_vram_gb": functiongemma["peak_vram_gb"],
        "functiongemma_preflight": functiongemma,
        "functiongemma_base_preflight_passed": functiongemma_base[
            "preflight_passed"
        ],
        "functiongemma_base_forward_passed": functiongemma_base["forward_passed"],
        "functiongemma_base_peak_vram_gb": functiongemma_base["peak_vram_gb"],
        "functiongemma_base_preflight": functiongemma_base,
        "functiongemma_base_access_report": functiongemma_base_access,
        "functiongemma_base_authenticated": functiongemma_base_access["authenticated"],
        "functiongemma_base_repo_listed": functiongemma_base_access["repo_listed"],
        "functiongemma_base_remote_download_ready": functiongemma_base_access[
            "remote_download_ready"
        ],
        "functiongemma_base_missing_local_patterns": functiongemma_base_access[
            "missing_local_patterns"
        ],
        "functiongemma_base_missing_remote_patterns": functiongemma_base_access[
            "missing_remote_patterns"
        ],
        "functiongemma_base_local_non_ref_file_count": functiongemma_base_access[
            "local_non_ref_file_count"
        ],
        "functiongemma_base_ready_for_direct_loading": functiongemma_base_access[
            "ready_for_direct_loading"
        ],
        "functiongemma_base_auth_error_type": functiongemma_base_access["auth_error_type"],
        "functiongemma_base_access_error_type": functiongemma_base_access[
            "access_error_type"
        ],
        "peak_vram_gb": peak_vram_gb,
        "within_vram_budget": (
            peak_vram_gb <= max_vram_gb
            and bge["within_vram_budget"]
            and embeddinggemma["within_vram_budget"]
            and functiongemma["within_vram_budget"]
            and functiongemma_base["within_vram_budget"]
        ),
        "full_path": (
            "Validated synthetic retrieval and schema/tool masking, authenticated "
            "EmbeddingGemma retrieval, pinned public BGE retrieval comparison, pinned "
            "public FunctionGemma Mobile Actions generation, and direct gated base "
            "FunctionGemma CUDA loading."
        ),
    }


def run_full_experiment(max_vram_gb: float = 24.0) -> dict:
    """Run the validated experiment path used by the verification report."""

    return run_gpu_test(max_vram_gb=max_vram_gb)


if MAIN:
    print(run_smoke_test())

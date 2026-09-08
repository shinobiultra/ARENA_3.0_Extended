"""Pinned prompt/model I/O and a fixed benign capability set for 9.1.

Sensitive benchmark text stays in memory. Persist IDs, complete hashes, labels
and aggregate outcomes; benign examples may be shown in the learner.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import torch as t

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
DATASET_ID = "josephmayo/refusal-compliance-pairs"
DATASET_REVISION = "b6ed3432f1d4a695e13be1c373bf7fb5af43f376"
DATASET_FILE = "data/train-00000-of-00001.parquet"
DATASET_SHA256 = "deb1db310fe45a0193ac09cca91cc3973f3db2145733df84a9d0ab48962dacfb"
SPLITS = {"train": (0, 32), "development": (32, 64), "final": (64, 100)}


@dataclass(frozen=True)
class PromptRecord:
    row: int
    class_index: int
    split: str
    refusal: bool
    text: str = field(repr=False)

    def public_metadata(self):
        return {"row": self.row, "class_index": self.class_index, "split": self.split,
                "refusal_label": self.refusal, "prompt_sha256": hashlib.sha256(self.text.encode()).hexdigest(),
                "display": f"Refusal-category request {self.row}" if self.refusal else self.text}


def load_prompt_split(split: str) -> list[PromptRecord]:
    """Class-wise pinned source-order split, with no duplicated normalized text."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    start, stop = SPLITS[split]
    path = Path(hf_hub_download(DATASET_ID, repo_type="dataset", revision=DATASET_REVISION, filename=DATASET_FILE))
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATASET_SHA256:
        raise ValueError("Pinned prompt file hash changed")
    rows = pq.read_table(path).to_pylist()
    normalized = [" ".join(row["prompt"].lower().split()) for row in rows]
    if len(rows) != 201 or len(set(normalized)) != len(rows):
        raise ValueError("Expected 201 unique prompts in the pinned source")
    selected = []
    for label in ("refusal", "compliance"):
        class_rows = [(row_index, row) for row_index, row in enumerate(rows) if row["label"] == label]
        for class_index in range(start, stop):
            row_index, row = class_rows[class_index]
            selected.append(PromptRecord(row_index, class_index, split, label == "refusal", row["prompt"]))
    return selected


def prompt_labels(records):
    return t.tensor([row.refusal for row in records], dtype=t.bool)


def encode_chat_prompts(tokenizer, prompts: list[str]):
    formatted = [tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True) for prompt in prompts]
    encoded = tokenizer(formatted, padding=True, add_special_tokens=False, return_tensors="pt")
    if encoded["input_ids"].shape[1] > 256:
        raise ValueError("This course protocol rejects long prompts instead of silently truncating them")
    return {key: value for key, value in encoded.items() if key in ("input_ids", "attention_mask")}


def load_qwen_model(*, device="cuda", dtype=t.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION,
                                                dtype=dtype, attn_implementation="sdpa").to(device).eval()
    return model, tokenizer


# These are independent exact-answer tasks, not refusal-marker absence scores.
_CAPITALS = (("France", "Paris"), ("Japan", "Tokyo"), ("Italy", "Rome"), ("Spain", "Madrid"),
             ("Germany", "Berlin"), ("Canada", "Ottawa"), ("Egypt", "Cairo"), ("Australia", "Canberra"))
_OPPOSITES = (("hot", "cold"), ("up", "down"), ("big", "small"), ("fast", "slow"),
              ("happy", "sad"), ("early", "late"), ("open", "closed"), ("day", "night"))


def benign_capability_examples():
    rows = []
    for a, b in ((2, 3), (4, 5), (6, 2), (3, 7), (8, 4), (9, 6), (12, 5), (14, 8)):
        rows.append({"task": "addition", "prompt": f"What is {a} + {b}? Answer with only the number.", "answer": str(a+b)})
    for country, capital in _CAPITALS:
        rows.append({"task": "capital", "prompt": f"What is the capital of {country}? Answer with only the city name.", "answer": capital})
    for word, opposite in _OPPOSITES:
        rows.append({"task": "opposite", "prompt": f"Give the opposite of {word}. Answer with only one word.", "answer": opposite})
    for word in ("apple", "banana", "cherry", "lemon", "mango", "peach", "pear", "plum"):
        rows.append({"task": "copy", "prompt": f"Copy this word exactly and write nothing else: {word}", "answer": word})
    return [{"id": f"benign-{index:02d}", **row} for index, row in enumerate(rows)]


def normalized_exact_answer(completion, answer):
    # Strip peripheral punctuation, not explanatory clauses or additional answers.
    def normalize(text):
        return re.sub(r"^[\W_]+|[\W_]+$", "", text.strip().casefold())
    return normalize(completion) == normalize(answer)

"""Predeclared name-role and unrelated-object prompts for concept erasure.

Only train/development may be used to choose a site or covariance regularizer.
Final templates and all rows are defined before the first native model run.
"""
from __future__ import annotations

MODEL_ID = "gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
NAMES = ("Mary", "John", "Tom", "James", "Sarah", "Daniel", "David", "Emily",
         "Michael", "Robert", "Lisa", "Alice", "Anne", "Paul", "Mark", "Susan",
         "Emma", "Jake", "Lucy", "Henry", "Peter", "Mike", "Anna", "Joe")
# These sentence frames differ from the final templates in 11.2 and 11.3.
MAIN_TEMPLATES = {
    "train": (
        "At the station, {a} met {b}, and {subject} handed a ticket to",
        "At the office, {a} met {b}, and {subject} passed a note to",
        "At the park, {a} met {b}, and {subject} handed a bottle to",
        "At the shop, {a} met {b}, and {subject} passed a bag to",
    ),
    "probe_train": (
        "After work, {a} talked with {b}, and {subject} gave a coin to",
        "After dinner, {a} talked with {b}, and {subject} gave a cup to",
        "Before class, {a} talked with {b}, and {subject} gave a pencil to",
        "Before breakfast, {a} talked with {b}, and {subject} gave a plate to",
    ),
    "development": (
        "During lunch, {a} spoke with {b}, and {subject} handed a letter to",
        "During a meeting, {a} spoke with {b}, and {subject} passed a folder to",
    ),
    "final": (
        "Outside the theater, {a} spoke to {b}. Afterwards, {subject} handed a key to",
        "Outside the hotel, {a} spoke to {b}. Afterwards, {subject} passed a parcel to",
        "Outside the hospital, {a} spoke to {b}. Afterwards, {subject} handed a card to",
    ),
}
SIDE_TEMPLATES = {
    "train": ("{a} and {b} had a {object}. Then {subject} put the {object} away. The object was a",),
    "probe_train": ("{a} and {b} bought a {object}. Then {subject} moved the {object}. They bought a",),
    "development": ("{a} and {b} found a {object}. Then {subject} picked up the {object}. They found a",),
    "final": ("{a} and {b} saw a {object}. Later, {subject} carried the {object} outside. The object they saw was a",),
}
OBJECTS = ("book", "ball", "map", "pen")
SPLITS = {"train": (16, (1, 3, 5, 7)), "probe_train": (20, (4, 9)), "development": (20, (2, 7)), "final": (24, (5, 11))}


def prompt_rows(split: str, task: str = "role") -> list[dict]:
    if split not in SPLITS or task not in {"role", "object"}:
        raise ValueError("Choose a declared split and role or object task.")
    count, offsets = SPLITS[split]
    rows = []
    templates = MAIN_TEMPLATES[split] if task == "role" else SIDE_TEMPLATES[split]
    for first in range(count):
        for offset in offsets:
            a, b = NAMES[first], NAMES[(first + offset) % count]
            for template_id, template in enumerate(templates):
                for role in (0, 1):
                    for obj in ((None,) if task == "role" else OBJECTS):
                        subject = (a, b)[role]
                        prompt = template.format(a=a, b=b, subject=subject, object=obj)
                        # The subject after the introduction is always the final occurrence of that name.
                        start = prompt.rfind(subject)
                        rows.append(dict(id=f"{split}/{task}/{first}/{offset}/{template_id}/{role}/{obj}",
                                         prompt=prompt, span=(start, start + len(subject)), a=a, b=b,
                                         role=role, object=obj, task=task, template=template_id,
                                         answer=(b, a)[role] if task == "role" else obj,
                                         new_name=first >= 16 or (first + offset) % count >= 16))
    return rows


def answer_token_ids(tokenizer, rows):
    names = set(NAMES) | set(OBJECTS)
    mapping = {}
    for name in sorted(names):
        ids = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"The prescribed answer {name!r} is not a single token: {ids}")
        mapping[name] = ids[0]
    return mapping

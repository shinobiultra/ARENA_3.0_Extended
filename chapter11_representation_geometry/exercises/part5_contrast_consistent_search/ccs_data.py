"""Frozen controlled-review splits, defined before native CCS feature access.

Labels describe the customer's explicit evaluative adjective. The unrelated
reader's opinion is crossed with both labels, not correlated with the target.
Repeated opinions and prompt wrappers are paired measurements of a base review.
"""

from __future__ import annotations

import hashlib
import itertools
import json

ITEMS = ("film", "book", "game", "show")
TRAIN_ADJECTIVES = (
    ("awful", "wonderful"),
    ("terrible", "excellent"),
    ("dreadful", "delightful"),
    ("horrible", "fantastic"),
    ("boring", "enjoyable"),
    ("disappointing", "brilliant"),
    ("unpleasant", "impressive"),
    ("poor", "amazing"),
)
FINAL_ADJECTIVES = (
    ("lousy", "superb"),
    ("abysmal", "marvelous"),
    ("tedious", "outstanding"),
    ("inferior", "satisfying"),
)
FRAMES = {
    "train": (
        "I thought the {item} was {adjective}.",
        "After trying the {item}, I found it {adjective}.",
        "In my opinion, this {item} is {adjective}.",
        "The {item} left me with one impression: {adjective}.",
    ),
    "calibration": (
        "My verdict on this {item}: {adjective}.",
        "I would describe the {item} as {adjective}.",
    ),
    "development": (
        "This {item} turned out to be {adjective}.",
        "Having finished the {item}, I can say it was {adjective}.",
    ),
    "final": (
        "Looking back on the {item}, the word I would choose is {adjective}.",
        "This is my personal assessment of the {item}: {adjective}.",
    ),
}
VARIANTS = ("plain", "neutral_reader", "distracting_reader")
WRAPPERS = ("customer", "quoted_review")
ANSWER_SUFFIXES = (" negative", " positive")


def base_reviews(split: str) -> list[dict]:
    if split not in FRAMES:
        raise ValueError(f"Unknown split: {split}")
    adjectives = FINAL_ADJECTIVES if split == "final" else TRAIN_ADJECTIVES
    if split == "calibration":
        adjectives = TRAIN_ADJECTIVES[:4]
    elif split == "development":
        adjectives = TRAIN_ADJECTIVES[4:]
    result = []
    for template_index, item_index, adjective_index, label in itertools.product(
        range(len(FRAMES[split])), range(len(ITEMS)), range(len(adjectives)), (0, 1)
    ):
        item, adjective = ITEMS[item_index], adjectives[adjective_index][label]
        result.append({
            "base_id": f"{split}:{template_index}:{item_index}:{adjective_index}:{label}",
            "split": split,
            "template_index": template_index,
            "item": item,
            "adjective": adjective,
            "label": label,
            "review": FRAMES[split][template_index].format(item=item, adjective=adjective),
        })
    return result


def render_prefix(review: str, variant: str, wrapper: str, distractor: int | None) -> str:
    if variant not in VARIANTS or wrapper not in WRAPPERS:
        raise ValueError("Unknown prescribed variant or wrapper.")
    if variant == "distracting_reader":
        if distractor not in (0, 1):
            raise ValueError("The unrelated reader must have an explicit binary opinion.")
        opinion = "positive" if distractor else "negative"
        other = f'\nA different reader, Morgan, gives this opinion: "{opinion}".'
    elif variant == "neutral_reader":
        other = "\nA different reader, Morgan, has not tried this item and gives no opinion."
    else:
        other = ""
    if wrapper == "customer":
        return f'Customer review: "{review}"{other}\nSentiment of the customer review:'
    return f'The customer wrote the following review: "{review}"{other}\nThe sentiment expressed by the customer is'


def paired_reviews(split: str, variant: str, wrapper: str) -> list[dict]:
    rows = []
    for base in base_reviews(split):
        for distractor in ((0, 1) if variant == "distracting_reader" else (None,)):
            prefix = render_prefix(base["review"], variant, wrapper, distractor)
            rows.append({**base, "variant": variant, "wrapper": wrapper, "distractor": distractor,
                         "row_id": f'{base["base_id"]}:{variant}:{wrapper}:{distractor}',
                         "prefix": prefix, "minus": prefix + ANSWER_SUFFIXES[0],
                         "plus": prefix + ANSWER_SUFFIXES[1]})
    return rows


def split_manifest() -> dict:
    descriptions = {}
    for split in FRAMES:
        records = [row for variant, wrapper in itertools.product(VARIANTS, WRAPPERS)
                   for row in paired_reviews(split, variant, wrapper)]
        encoded = json.dumps(records, sort_keys=True).encode()
        descriptions[split] = {"base_reviews": len(base_reviews(split)), "contrast_pairs": len(records),
                               "sha256": hashlib.sha256(encoded).hexdigest()}
    return descriptions


def check_data_contract() -> None:
    reviews = {split: base_reviews(split) for split in FRAMES}
    assert {key: len(value) for key, value in reviews.items()} == {
        "train": 256, "calibration": 64, "development": 64, "final": 64,
    }
    seen = set()
    for split, bases in reviews.items():
        current = {row["review"] for row in bases}
        assert len(current) == len(bases) and not current & seen
        seen.update(current)
        for variant, wrapper in itertools.product(VARIANTS, WRAPPERS):
            pairs = paired_reviews(split, variant, wrapper)
            assert len({row["row_id"] for row in pairs}) == len(pairs)
            assert sum(row["label"] for row in pairs) * 2 == len(pairs)
            assert all(row["plus"] != row["minus"] for row in pairs)
            if variant == "distracting_reader":
                for label, distractor in itertools.product((0, 1), repeat=2):
                    assert sum(row["label"] == label and row["distractor"] == distractor for row in pairs) * 4 == len(pairs)
    assert not set(itertools.chain.from_iterable(TRAIN_ADJECTIVES)) & set(itertools.chain.from_iterable(FINAL_ADJECTIVES))


if __name__ == "__main__":
    check_data_contract()
    print(json.dumps(split_manifest(), indent=2))

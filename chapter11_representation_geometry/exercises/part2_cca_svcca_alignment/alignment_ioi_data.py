"""Small frozen IOI prompt families, extending the original ARENA task."""

MODEL_ID = "gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
NAMES = tuple(
    "Mary John Tom James Sarah Daniel David Emily Michael Robert Lisa Alice Anne Paul Mark Susan".split()
)
TRAIN_TEMPLATES = (
    "When {a} and {b} went to the store, {b} gave a book to",
    "After {a} and {b} visited the park, {b} handed a drink to",
    "{a} and {b} were at the office. {b} gave a letter to",
)
DEVELOPMENT_TEMPLATES = (
    "While {a} and {b} were at the school, {b} gave a pen to",
    "When {a} and {b} arrived at the station, {b} handed a ticket to",
)
FINAL_TEMPLATES = (
    "After {a} and {b} went to the cafe, {b} gave a cup to",
    "{a} and {b} met at the library. {b} handed a notebook to",
    "When {a} and {b} visited the garden, {b} gave a flower to",
)


def paired_rows(split):
    if split == "train":
        templates = TRAIN_TEMPLATES
        count = 12
    elif split == "development":
        templates = DEVELOPMENT_TEMPLATES
        count = 12
    elif split == "final":
        templates = FINAL_TEMPLATES
        count = 16
    else:
        raise ValueError("Choose train, development or final.")
    rows = []
    for template_index, template in enumerate(templates):
        for pair in range(count):
            left = NAMES[pair]
            right = NAMES[(pair + 1) % count]
            for orientation in [0, 1]:
                a, b = (left, right) if orientation == 0 else (right, left)
                prompt = template.format(a=a, b=b)
                position = prompt.rindex(b)
                rows.append(
                    {
                        "id": f"{split}/template{template_index}/pair{pair}/orientation{orientation}",
                        "split": split,
                        "template": template_index,
                        "pair": pair,
                        "orientation": orientation,
                        "word": b,
                        "recipient": a,
                        "subject": b,
                        "prompt": prompt,
                        "span": [position, position + len(b)],
                        "word_seen_in_training": a in NAMES[:12] and b in NAMES[:12],
                    }
                )
    return rows


def donor_indices(rows):
    lookup = {(r["template"], r["pair"], r["orientation"]): i for i, r in enumerate(rows)}
    if len(lookup) != len(rows):
        raise ValueError("IOI rows must be unique.")
    return [lookup[r["template"], r["pair"], r["orientation"] ^ 1] for r in rows]

"""Fixed paired examples for a small, CPU-runnable cross-layer alignment study."""

from pathlib import Path
import json

MODEL_ID = "EleutherAI/pythia-70m-deduped"
REVISION = "e93a9faa9c77e5d09219f6c868bfc7a1bd65593c"
WORDS = tuple(
    "hot cold big small fast slow light dark young old good bad near far tall short wet dry clean dirty hard soft open closed high low happy sad early late true false".split()
)
TRAIN_TEMPLATES = (
    "The opposite of {word} is",
    "A word with the opposite meaning to {word} is",
    "If something is not {word}, it is",
)
DEV_TEMPLATES = ("An antonym of {word} is", "The antonym for {word} is")
FINAL_TEMPLATES = (
    "Vocabulary question: the opposite of {word} is",
    "In this exercise, an antonym of {word} is",
    "Complete the contrast: the opposite of {word} is",
)


def paired_rows(split):
    if split == "train":
        templates = TRAIN_TEMPLATES
        indices = range(24)
    elif split == "development":
        templates = DEV_TEMPLATES
        indices = range(24)
    elif split == "final":
        templates = FINAL_TEMPLATES
        indices = range(32)
    else:
        raise ValueError("Choose train, development or final.")
    rows = []
    for template_index, template in enumerate(templates):
        for index in indices:
            word = WORDS[index]
            prompt = template.format(word=word)
            left = prompt.index(word)
            rows.append(
                {
                    "id": f"{split}/template{template_index}/word{index}",
                    "split": split,
                    "template": template_index,
                    "word_index": index,
                    "word": word,
                    "opposite": WORDS[index ^ 1],
                    "prompt": prompt,
                    "span": [left, left + len(word)],
                    "word_seen_in_training": index < 24,
                }
            )
    return rows


def donor_indices(rows):
    lookup = {(r["template"], r["word_index"]): i for i, r in enumerate(rows)}
    if len(lookup) != len(rows):
        raise ValueError("Paired input rows must be unique.")
    return [lookup[r["template"], r["word_index"] ^ 1] for r in rows]

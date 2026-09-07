"""Prespecified name-role interventions for the native DAS experiment.

All three splits are defined before the first native DAS run. Final prompt
families and name pairings must stay closed until development selection ends.
The high-level variable R identifies which of the first two names repeats;
the answer is the other base name, not necessarily the source's answer.
"""

MODEL_ID = "gpt2"
REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
NAMES = tuple(
    "Mary John Tom James Sarah Daniel David Emily Michael Robert Lisa Alice Anne Paul Mark Susan".split()
)
TEMPLATES = {
    "train": (
        "When {a} and {b} went to the store, {repeat} gave a book to",
        "After {a} and {b} visited the park, {repeat} handed a drink to",
        "{a} and {b} were at the office. {repeat} gave a letter to",
    ),
    "development": (
        "While {a} and {b} were at the school, {repeat} gave a pen to",
        "When {a} and {b} arrived at the station, {repeat} handed a ticket to",
    ),
    # These are distinct from the already inspected 11.2 final families.
    "final": (
        "After {a} and {b} entered the museum, {repeat} gave a map to",
        "{a} and {b} met at the restaurant. {repeat} handed a menu to",
        "When {a} and {b} visited the beach, {repeat} gave a towel to",
    ),
}
FINAL_NAME_ORDER = (0, 4, 1, 5, 2, 6, 3, 7, 8, 12, 9, 13, 10, 14, 11, 15)


def prompt_rows(split):
    """72 train, 48 development, or 96 final prompts, without model filtering."""
    if split not in TEMPLATES:
        raise ValueError("Choose train, development or final.")
    order = FINAL_NAME_ORDER if split == "final" else tuple(range(12))
    pairs = [(NAMES[order[i]], NAMES[order[i + 1]]) for i in range(0, len(order), 2)]
    rows = []
    for template_id, template in enumerate(TEMPLATES[split]):
        for pair_id, pair in enumerate(pairs):
            for orientation in (0, 1):
                a, b = pair if orientation == 0 else pair[::-1]
                for role in (0, 1):
                    repeat = (a, b)[role]
                    prompt = template.format(a=a, b=b, repeat=repeat)
                    start = prompt.rindex(repeat)
                    rows.append(
                        dict(
                            id=f"{split}/t{template_id}/p{pair_id}/o{orientation}/r{role}",
                            split=split,
                            template=template_id,
                            pair=pair_id,
                            orientation=orientation,
                            a=a,
                            b=b,
                            role=role,
                            answer=(a, b)[1 - role],
                            prompt=prompt,
                            span=[start, start + len(repeat)],
                            names_seen_in_training=a in NAMES[:12] and b in NAMES[:12],
                        )
                    )
    return rows


def interchange_pairs(rows):
    """Four interventions per base: same/cross-name donor × role 0/1.

    Cross-name donors are the next disjoint pair, with the same template and
    orientation. The same-name, same-role case is an exact identity control.
    The explicit counterfactual prompt is present in the unfiltered row set.
    """
    lookup = {
        (r["template"], r["pair"], r["orientation"], r["role"]): i for i, r in enumerate(rows)
    }
    if len(lookup) != len(rows):
        raise ValueError("The prompt identifiers must be unique.")
    count = len({r["pair"] for r in rows})
    pairs = []
    for base, row in enumerate(rows):
        for cross_name in (False, True):
            source_pair = (row["pair"] + int(cross_name)) % count
            for source_role in (0, 1):
                source = lookup[row["template"], source_pair, row["orientation"], source_role]
                counterfactual = lookup[
                    row["template"], row["pair"], row["orientation"], source_role
                ]
                pairs.append(
                    dict(
                        id=f"{row['id']}/cross{int(cross_name)}/sourceR{source_role}",
                        base=base,
                        source=source,
                        counterfactual=counterfactual,
                        target=(row["a"], row["b"])[1 - source_role],
                        changed_role=source_role != row["role"],
                        cross_name=cross_name,
                    )
                )
    return pairs


def name_token_ids(tokenizer, rows):
    """Require whole, single-token names in the output vocabulary."""
    encoded = {name: tokenizer.encode(" " + name, add_special_tokens=False) for name in NAMES}
    if any(len(value) != 1 for value in encoded.values()):
        raise ValueError("Every prescribed name must be exactly one space-prefixed token.")
    return {name: value[0] for name, value in encoded.items()}

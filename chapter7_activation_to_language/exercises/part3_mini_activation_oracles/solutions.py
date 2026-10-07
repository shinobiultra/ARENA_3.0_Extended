# %%


import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter7_activation_to_language"
section = "part3_mini_activation_oracles"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part3_mini_activation_oracles.tests as tests
import part3_mini_activation_oracles.utils as utils

MAIN = __name__ == "__main__"

# %%

ANIMALS = {  # singular -> plural
    "dog": "dogs", "horse": "horses", "bird": "birds", "rabbit": "rabbits", "cow": "cows", "pig": "pigs",
    "lion": "lions", "wolf": "wolves", "duck": "ducks", "frog": "frogs", "monkey": "monkeys", "goat": "goats",
    "chicken": "chickens", "bee": "bees", "whale": "whales", "shark": "sharks", "rat": "rats", "mouse": "mice",
    "elephant": "elephants", "turtle": "turtles", "dolphin": "dolphins", "crab": "crabs",
    # held out: never seen in training
    "cat": "cats", "tiger": "tigers", "bear": "bears", "snake": "snakes", "spider": "spiders", "bull": "bulls",
}
HELDOUT_ANIMALS = ["cat", "tiger", "bear", "snake", "spider", "bull"]
TRAIN_ANIMALS = [a for a in ANIMALS if a not in HELDOUT_ANIMALS]

CITIES = [
    "Paris", "London", "Tokyo", "Berlin", "Rome", "Madrid", "Boston", "Chicago", "Sydney", "Vienna", "Cairo",
    "Denver", "Toronto", "Athens", "Prague", "Mumbai", "Beijing", "Amsterdam", "Oslo", "Lisbon", "Warsaw",
    "Seoul", "Bangkok", "Munich", "Venice", "Glasgow",
    # held out: never seen in training
    "Moscow", "Miami", "Dallas", "Dublin", "Seattle", "Istanbul",
]
HELDOUT_CITIES = ["Moscow", "Miami", "Dallas", "Dublin", "Seattle", "Istanbul"]
TRAIN_CITIES = [c for c in CITIES if c not in HELDOUT_CITIES]

POSITIVE_VERBS = ["loved", "enjoyed", "liked", "admired", "welcomed"]
NEGATIVE_VERBS = ["hated", "feared", "disliked", "dreaded", "despised"]
OBJECTS = ["rain", "snow", "food", "music", "park", "river", "garden", "beach", "noise", "bath", "car", "storm",
           "crowd", "market", "train", "boat"]
SINGULAR_DETERMINERS = ["the", "my", "a", "our", "her"]
PLURAL_DETERMINERS = ["the", "my", "two", "our", "some"]

# {D}/{d} = capitalised/lowercase determiner, {A} = animal word, {C} = city, {V} = verb, {O} = object
TRAIN_TEMPLATES = [
    "{D} {A} in {C} {V} the {O}.",
    "In {C}, {d} {A} {V} the {O}.",
    "{D} {A} from {C} {V} the {O}.",
    "Last year in {C}, {d} {A} {V} the {O}.",
    "Everyone in {C} knew that {d} {A} {V} the {O}.",
    "{D} {A} {V} the {O} in {C}.",
]
HELDOUT_TEMPLATES = [
    "We heard that {d} {A} in {C} {V} the {O}.",
    "During a visit to {C}, {d} {A} {V} the {O}.",
    "Back home in {C}, {d} {A} {V} the {O} a lot.",
]

FAMILIES = ["animal", "city", "sentiment", "number"]
CHANCE = {"animal": 1 / len(TRAIN_ANIMALS), "city": 1 / len(TRAIN_CITIES), "sentiment": 0.5, "number": 0.5}

# %%

@dataclass
class SourceExample:
    text: str
    animal: str  # singular form, e.g. "dog" (this is the answer to the animal question)
    city: str
    sentiment: str  # "positive" or "negative"
    number: str  # "singular" or "plural"
    animal_word: str  # the animal as it appears in the text, e.g. "dogs"
    verb: str
    template: str
    tokens: list[int] = field(default_factory=list)  # BOS + the text's tokens
    available: dict[str, int] = field(default_factory=dict)  # family -> first position where it's determined


def sample_source_example(
    rng: random.Random, templates: list[str], animals: list[str], cities: list[str]
) -> SourceExample:
    """Samples a sentence with independently, uniformly chosen attributes."""
    animal, city = rng.choice(animals), rng.choice(cities)
    plural, positive = rng.random() < 0.5, rng.random() < 0.5
    verb = rng.choice(POSITIVE_VERBS if positive else NEGATIVE_VERBS)
    animal_word = ANIMALS[animal] if plural else animal
    det = rng.choice(PLURAL_DETERMINERS if plural else SINGULAR_DETERMINERS)
    if det == "a" and animal_word[0] in "aeiou":
        det = "an"
    template = rng.choice(templates)
    text = template.format(D=det.capitalize(), d=det, A=animal_word, C=city, V=verb, O=rng.choice(OBJECTS))
    return SourceExample(
        text=text,
        animal=animal,
        city=city,
        sentiment="positive" if positive else "negative",
        number="plural" if plural else "singular",
        animal_word=animal_word,
        verb=verb,
        template=template,
    )


rng = random.Random(0)
for _ in range(4):
    ex = sample_source_example(rng, TRAIN_TEMPLATES, TRAIN_ANIMALS, TRAIN_CITIES)
    print(f"{ex.text:<55} animal={ex.animal}, city={ex.city}, sentiment={ex.sentiment}, number={ex.number}")

# %%

if MAIN:
    target = HookedTransformer.from_pretrained("gpt2-small", device=device)
    target.requires_grad_(False)
    
    print(f"{target.cfg.n_layers} layers, d_model={target.cfg.d_model}")

# %%

def annotate_example(model: HookedTransformer, example: SourceExample) -> SourceExample:
    """Fills in example.tokens (with BOS) and example.available (family -> first determining position)."""
    example.tokens = model.to_tokens(example.text)[0].tolist()
    position = lambda word: example.tokens.index(model.to_single_token(" " + word))
    example.available = {
        "animal": position(example.animal_word),
        "city": position(example.city),
        "sentiment": position(example.verb),
        "number": position(example.animal_word),
    }
    return example


if MAIN:
    tests.test_annotate_example(annotate_example, SourceExample, target)

# %%

def make_dataset(
    model: HookedTransformer, n: int, templates: list[str], animals: list[str], cities: list[str], seed: int
) -> list[SourceExample]:
    rng = random.Random(seed)
    return [annotate_example(model, sample_source_example(rng, templates, animals, cities)) for _ in range(n)]

# %%

if MAIN:
    datasets = {
        "train": make_dataset(target, 6000, TRAIN_TEMPLATES, TRAIN_ANIMALS, TRAIN_CITIES, seed=1),
        "test": make_dataset(target, 600, TRAIN_TEMPLATES, TRAIN_ANIMALS, TRAIN_CITIES, seed=2),
        "heldout_templates": make_dataset(target, 600, HELDOUT_TEMPLATES, TRAIN_ANIMALS, TRAIN_CITIES, seed=3),
        "heldout_entities": make_dataset(target, 600, TRAIN_TEMPLATES, HELDOUT_ANIMALS, HELDOUT_CITIES, seed=4),
        "sweep": make_dataset(target, 600, TRAIN_TEMPLATES[:1], TRAIN_ANIMALS, TRAIN_CITIES, seed=5),
    }
    for name, examples in datasets.items():
        print(f"{name:<18} {len(examples)} sentences, e.g. {examples[0].text!r}")
    
    ex = datasets["sweep"][0]
    print("\n", [target.to_string(tok) for tok in ex.tokens], ex.available)

# %%

def resid_hook_name(model: HookedTransformer, layer: int) -> str:
    """Name of the residual stream after `layer` blocks (layer 0 = embeddings, layer n_layers = final)."""
    if layer < model.cfg.n_layers:
        return f"blocks.{layer}.hook_resid_pre"
    return f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"


@t.no_grad()
def get_resid(
    model: HookedTransformer, examples: list[SourceExample], layer: int, batch_size: int = 256
) -> list[Float[Tensor, "seq d_model"]]:
    """Returns the residual stream after `layer` blocks, at every position of every example."""
    hook_name = resid_hook_name(model, layer)
    out = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        lengths = [len(ex.tokens) for ex in batch]
        tokens = t.zeros((len(batch), max(lengths)), dtype=t.long)
        for i, ex in enumerate(batch):
            tokens[i, : lengths[i]] = t.tensor(ex.tokens)
        _, cache = model.run_with_cache(tokens.to(model.cfg.device), names_filter=hook_name)
        out += [cache[hook_name][i, :length].clone() for i, length in enumerate(lengths)]
    return out


if MAIN:
    tests.test_get_resid(get_resid, target)

# %%

SOURCE_LAYER = 6


@t.no_grad()
def last_token_resid_all_layers(
    model: HookedTransformer, examples: list[SourceExample], batch_size: int = 256
) -> Float[Tensor, "layer n d_model"]:
    """Residual stream at the final token, after every number of blocks 0..n_layers."""
    names = [resid_hook_name(model, layer) for layer in range(model.cfg.n_layers + 1)]
    out = []
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        lengths = t.tensor([len(ex.tokens) for ex in batch])
        tokens = t.zeros((len(batch), int(lengths.max())), dtype=t.long)
        for i, ex in enumerate(batch):
            tokens[i, : lengths[i]] = t.tensor(ex.tokens)
        _, cache = model.run_with_cache(tokens.to(model.cfg.device), names_filter=lambda name: name in names)
        rows = t.arange(len(batch))
        out.append(t.stack([cache[name][rows, lengths - 1] for name in names]))
    return t.cat(out, dim=1)

# %%

if MAIN:
    acts = {name: get_resid(target, examples, SOURCE_LAYER) for name, examples in datasets.items()}
    last_all = {name: last_token_resid_all_layers(target, datasets[name]) for name in ["train", "test"]}
    # The final-token vectors at layer 6, which we'll use most
    last = {name: t.stack([a[-1] for a in acts[name]]) for name in datasets}
    
    print({name: tuple(x.shape) for name, x in last.items()})
    print(f"Mean norm of layer-{SOURCE_LAYER} final-token vectors: {last['train'].norm(dim=-1).mean():.1f}")
    t.testing.assert_close(last_all["test"][SOURCE_LAYER], last["test"])

# %%

CLASSES = {
    "animal": sorted(TRAIN_ANIMALS),
    "city": sorted(TRAIN_CITIES),
    "sentiment": ["negative", "positive"],
    "number": ["plural", "singular"],
}


def family_labels(examples: list[SourceExample], family: str) -> Int[Tensor, " n"]:
    """Class index of each example's answer for `family`, or -1 if it isn't one of the training classes."""
    classes = CLASSES[family]
    answers = [getattr(ex, family) for ex in examples]
    return t.tensor([classes.index(a) if a in classes else -1 for a in answers])


class LinearProbe(nn.Module):
    def __init__(self, mean: Float[Tensor, " d"], std: Float[Tensor, " d"], n_classes: int):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        self.linear = nn.Linear(mean.shape[0], n_classes)

    def forward(self, x: Float[Tensor, "n d"]) -> Float[Tensor, "n n_classes"]:
        return self.linear((x - self.mean) / self.std)

    @t.no_grad()
    def accuracy(self, x: Float[Tensor, "n d"], labels: Int[Tensor, " n"]) -> float:
        return (self(x).argmax(-1) == labels.to(x.device)).float().mean().item()

# %%

def train_linear_probe(
    x: Float[Tensor, "n d"],
    labels: Int[Tensor, " n"],
    n_classes: int,
    n_steps: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
) -> LinearProbe:
    """Trains a standardised logistic-regression probe with full-batch Adam."""
    probe = LinearProbe(x.mean(0), x.std(0) + 1e-6, n_classes).to(x.device)
    optimizer = t.optim.Adam(probe.parameters(), lr=lr, weight_decay=weight_decay)
    labels = labels.to(x.device)
    for _ in range(n_steps):
        loss = F.cross_entropy(probe(x), labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return probe


if MAIN:
    tests.test_train_linear_probe(train_linear_probe)

# %%

if MAIN:
    t.manual_seed(0)
    probe_acc = {family: [] for family in FAMILIES}
    shuffled_acc = {family: [] for family in FAMILIES}
    probes = {}  # family -> layer-6 probe
    for family in FAMILIES:
        y_train, y_test = family_labels(datasets["train"], family), family_labels(datasets["test"], family)
        y_shuffled = y_train[t.randperm(len(y_train))]
        n_classes = len(CLASSES[family])
        for layer in range(target.cfg.n_layers + 1):
            probe = train_linear_probe(last_all["train"][layer], y_train, n_classes)
            probe_acc[family].append(probe.accuracy(last_all["test"][layer], y_test))
            control = train_linear_probe(last_all["train"][layer], y_shuffled, n_classes)
            shuffled_acc[family].append(control.accuracy(last_all["test"][layer], y_test))
            if layer == SOURCE_LAYER:
                probes[family] = probe
    
    display(pd.DataFrame(probe_acc).T.round(2))
    utils.plot_probe_layers(
        probe_acc, shuffled_acc, CHANCE, "Linear probes on GPT-2's final-token residual stream (held-out sentences)"
    )

# %%

if MAIN:
    def mean_pairwise_cosine(x: Float[Tensor, "n d"]) -> float:
        x = F.normalize(x, dim=-1)
        sims = x @ x.T
        n = x.shape[0]
        return ((sims.sum() - n) / (n * (n - 1))).item()
    
    
    center = last["train"].mean(0)
    print(f"Mean cosine similarity between different sentences' vectors: {mean_pairwise_cosine(last['train'][:1000]):.3f}")
    print(f"Mean norm of a vector v:          {last['train'].norm(dim=-1).mean():.1f}")
    print(f"Norm of the mean vector mu:       {center.norm():.1f}")
    print(f"Mean norm of the part v - mu:     {(last['train'] - center).norm(dim=-1).mean():.1f}")

# %%

PLACEHOLDER = " ?"
INJECT_LAYER = 1  # the oracle layer whose resid_pre we inject into


def oracle_prompt(question: str) -> str:
    """The oracle's input text (BOS is prepended separately when tokenizing)."""
    return f"{PLACEHOLDER}\nQuestion: {question}\nAnswer:"


TRAIN_QUESTIONS = {
    "animal": ["Which animal is this about?", "What animal is mentioned?", "Name the animal.", "What kind of animal was it?"],
    "city": ["Which city is mentioned?", "Where did this happen?", "Name the city.", "In which city was it?"],
    "sentiment": ["Is the feeling positive or negative?", "Was the reaction positive or negative?",
                  "Positive or negative?", "Is the sentiment positive or negative?"],
    "number": ["Is the subject singular or plural?", "Singular or plural?", "Is the animal singular or plural?",
               "Is there one animal (singular) or several (plural)?"],
}
HELDOUT_QUESTIONS = {
    "animal": ["Which creature does the text describe?", "What species is involved?"],
    "city": ["What place is the text about?", "Which town is it set in?"],
    "sentiment": ["Is the tone positive or negative?", "Good feelings or bad: positive or negative?"],
    "number": ["Is the noun singular or plural?", "Grammatical number: singular or plural?"],
}


def answer_token_ids(model: HookedTransformer, answers: list[str]) -> Int[Tensor, " batch"]:
    """Token id of each answer, with a leading space (every answer in this section is a single token)."""
    return t.tensor([model.to_single_token(" " + answer) for answer in answers], device=model.cfg.device)

# %%

def tokenize_oracle_prompts(
    model: HookedTransformer, questions: list[str]
) -> tuple[Int[Tensor, "batch seq"], Int[Tensor, " batch"], Int[Tensor, " batch"]]:
    """Returns (right-padded tokens with BOS, placeholder positions, positions of the last real token)."""
    rows = [model.to_tokens(oracle_prompt(q))[0] for q in questions]
    lengths = t.tensor([len(row) for row in rows])
    tokens = t.zeros((len(rows), int(lengths.max())), dtype=t.long)
    for i, row in enumerate(rows):
        tokens[i, : len(row)] = row
    placeholder_id = model.to_single_token(PLACEHOLDER)
    placeholder_pos = t.tensor([row.tolist().index(placeholder_id) for row in rows])
    device = model.cfg.device
    return tokens.to(device), placeholder_pos.to(device), (lengths - 1).to(device)


if MAIN:
    tests.test_tokenize_oracle_prompts(tokenize_oracle_prompts, target, oracle_prompt, PLACEHOLDER)

# %%

def make_injection_hook(
    vectors: Float[Tensor, "batch d_model"] | None,
    placeholder_pos: Int[Tensor, " batch"],
    center: Float[Tensor, " d_model"] | None = None,
    coef: float = 1.0,
) -> Callable:
    """Returns a hook that adds the (centred, norm-matched) vectors at the placeholder positions."""
    def hook_fn(resid: Float[Tensor, "batch seq d_model"], hook) -> Float[Tensor, "batch seq d_model"]:
        if vectors is None:
            return resid
        v = vectors.to(resid.device, resid.dtype)
        if center is not None:
            v = v - center.to(resid.device, resid.dtype)
        rows = t.arange(resid.shape[0], device=resid.device)
        h = resid[rows, placeholder_pos]
        delta = t.zeros_like(resid)
        delta[rows, placeholder_pos] = coef * h.norm(dim=-1, keepdim=True) * F.normalize(v, dim=-1)
        return resid + delta

    return hook_fn


if MAIN:
    tests.test_make_injection_hook(make_injection_hook)

# %%

def oracle_logits(
    oracle: HookedTransformer,
    vectors: Float[Tensor, "batch d_model"] | None,
    questions: list[str],
    center: Float[Tensor, " d_model"] | None,
    inject_layer: int = INJECT_LAYER,
) -> Float[Tensor, "batch d_vocab"]:
    """The oracle's next-token logits after 'Answer:', with `vectors` injected at the placeholder."""
    tokens, placeholder_pos, answer_pos = tokenize_oracle_prompts(oracle, questions)
    hook = make_injection_hook(vectors, placeholder_pos, center)
    resid = oracle.run_with_hooks(
        tokens, fwd_hooks=[(f"blocks.{inject_layer}.hook_resid_pre", hook)], stop_at_layer=oracle.cfg.n_layers
    )
    final = resid[t.arange(len(questions), device=resid.device), answer_pos]
    return oracle.unembed(oracle.ln_final(final[:, None]))[:, 0]


if MAIN:
    tests.test_oracle_logits(oracle_logits, tokenize_oracle_prompts, make_injection_hook, target)

# %%

if MAIN:
    def show_oracle_answers(
        oracle: HookedTransformer, examples: list[SourceExample], vectors: Tensor, center: Tensor | None
    ) -> pd.DataFrame:
        """Asks the first training paraphrase of every family about each example; shows answer vs truth."""
        rows = []
        with t.no_grad():
            for family in FAMILIES:
                logits = oracle_logits(oracle, vectors, [TRAIN_QUESTIONS[family][0]] * len(examples), center)
                for ex, tok in zip(examples, logits.argmax(-1).tolist()):
                    rows.append({"sentence": ex.text, "family": family, "oracle says": oracle.to_string(tok), "truth": getattr(ex, family)})
        df = pd.DataFrame(rows)
        df["oracle says"] = df["oracle says"].map(repr)
        return df.pivot(index="sentence", columns="family", values=["oracle says", "truth"])
    
    
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(show_oracle_answers(target, datasets["test"][:4], last["test"][:4], center))

# %%

def sample_oracle_batch(
    examples: list[SourceExample],
    acts: list[Float[Tensor, "seq d_model"]],
    batch_size: int,
    rng: random.Random,
    questions: dict[str, list[str]] = TRAIN_QUESTIONS,
    position: str = "last",
) -> tuple[Float[Tensor, "batch d_model"], list[str], list[str], list[str]]:
    """Samples (activation, question, answer, family) training triples."""
    assert position in ("last", "random")
    vectors, qs, answers, families = [], [], [], []
    for _ in range(batch_size):
        i = rng.randrange(len(examples))
        example = examples[i]
        family = rng.choice(list(questions))
        last_pos = len(example.tokens) - 1
        pos = last_pos if position == "last" else rng.randint(example.available[family], last_pos)
        vectors.append(acts[i][pos])
        qs.append(rng.choice(questions[family]))
        answers.append(getattr(example, family))
        families.append(family)
    return t.stack(vectors), qs, answers, families


if MAIN:
    tests.test_sample_oracle_batch(sample_oracle_batch, datasets["test"][:50], acts["test"][:50], TRAIN_QUESTIONS)

# %%

def text_only_loss_floor(examples: list[SourceExample], families: list[str] = FAMILIES) -> float:
    """Mean over families of the entropy (in nats) of that family's answer distribution."""
    entropies = []
    for family in families:
        counts = t.tensor(list(Counter(getattr(ex, family) for ex in examples).values()), dtype=t.float)
        p = counts / counts.sum()
        entropies.append(-(p * p.log()).sum().item())
    return sum(entropies) / len(entropies)


if MAIN:
    tests.test_text_only_loss_floor(text_only_loss_floor, SourceExample)

# %%

if MAIN:
    floor = text_only_loss_floor(datasets["train"])
    print(f"Best possible loss without the activation: {floor:.3f} nats")

# %%

def train_oracle(
    oracle: HookedTransformer,
    examples: list[SourceExample],
    acts: list[Float[Tensor, "seq d_model"]],
    center: Float[Tensor, " d_model"] | None,
    n_steps: int = 800,
    batch_size: int = 64,
    lr: float = 1e-4,
    position: str = "last",
    seed: int = 0,
) -> list[float]:
    """Fine-tunes `oracle` in place to answer questions about injected activations. Returns the losses."""
    rng = random.Random(seed)
    optimizer = t.optim.AdamW(oracle.parameters(), lr=lr, weight_decay=0.0)
    losses = []
    for _ in tqdm(range(n_steps)):
        vectors, questions, answers, _ = sample_oracle_batch(examples, acts, batch_size, rng, position=position)
        logits = oracle_logits(oracle, vectors, questions, center)
        loss = F.cross_entropy(logits, answer_token_ids(oracle, answers))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    oracle.zero_grad(set_to_none=True)  # free the gradients' memory (we'll keep this model around for evaluation)
    return losses

# %%

if MAIN:
    t.manual_seed(0)
    oracle = HookedTransformer.from_pretrained("gpt2-small", device=device)
    losses = train_oracle(oracle, datasets["train"], acts["train"], center, n_steps=800 if device.type != "cpu" else 200)
    print(f"Final loss (mean of last 50 steps): {np.mean(losses[-50:]):.3f}")
    
    utils.plot_training_loss({"oracle (final-token activations)": losses}, floor, "Training the oracle")

# %%

if MAIN:
    with pd.option_context("display.width", 200, "display.max_colwidth", 60):
        print(show_oracle_answers(oracle, datasets["test"][:4], last["test"][:4], center))

# %%

@t.inference_mode()
def evaluate_oracle(
    oracle: HookedTransformer,
    vectors: Float[Tensor, "n d_model"] | None,
    examples: list[SourceExample],
    center: Float[Tensor, " d_model"] | None,
    questions: dict[str, list[str]] = TRAIN_QUESTIONS,
    batch_size: int = 256,
) -> dict[str, float]:
    """Top-1 accuracy per family, averaged over all examples and all paraphrases in `questions`."""
    results = {}
    for family, paraphrases in questions.items():
        n_correct = n_total = 0
        for question in paraphrases:
            for start in range(0, len(examples), batch_size):
                batch = examples[start : start + batch_size]
                v = None if vectors is None else vectors[start : start + batch_size]
                logits = oracle_logits(oracle, v, [question] * len(batch), center)
                answers = answer_token_ids(oracle, [getattr(ex, family) for ex in batch])
                n_correct += (logits.argmax(-1) == answers).sum().item()
                n_total += len(batch)
        results[family] = n_correct / n_total
    return results


if MAIN:
    tests.test_evaluate_oracle(evaluate_oracle, oracle_logits, answer_token_ids, oracle, datasets["test"][:40], last["test"][:40], center)

# %%

def control_vectors(
    vectors: Float[Tensor, "n d_model"], kind: str, center: Float[Tensor, " d_model"], seed: int = 0
) -> Float[Tensor, "n d_model"] | None:
    """Returns the input vectors for a control condition: real, text-only, shuffled or random."""
    assert kind in ("real", "text-only", "shuffled", "random")
    generator = t.Generator().manual_seed(seed)
    if kind == "real":
        return vectors
    if kind == "text-only":
        return None
    if kind == "shuffled":
        n = vectors.shape[0]
        perm = t.randperm(n, generator=generator)
        out = t.empty_like(vectors)
        out[perm.roll(-1)] = vectors[perm]
        return out
    centered = vectors - center
    noise = t.randn(vectors.shape, generator=generator).to(vectors.device, vectors.dtype)
    return center + noise / noise.norm(dim=-1, keepdim=True) * centered.norm(dim=-1, keepdim=True)


if MAIN:
    tests.test_control_vectors(control_vectors)

# %%

if MAIN:
    test_vectors = last["test"]
    conditions = {"oracle": evaluate_oracle(oracle, test_vectors, datasets["test"], center)}
    conditions["linear probe (same vectors)"] = {
        family: probes[family].accuracy(test_vectors, family_labels(datasets["test"], family)) for family in FAMILIES
    }
    conditions["untrained GPT-2 + injection"] = evaluate_oracle(target, test_vectors, datasets["test"], center)
    for kind in ["text-only", "shuffled", "random"]:
        conditions[f"oracle, {kind}"] = evaluate_oracle(
            oracle, control_vectors(test_vectors, kind, center), datasets["test"], center
        )
    
    display(pd.DataFrame(conditions).T.round(3))
    utils.plot_condition_bars(conditions, CHANCE, "Oracle vs probe vs controls (test sentences, layer-6 final-token activations)")

# %%

if MAIN:
    oracle_table, probe_table = {}, {}
    for split, qs in [("test", TRAIN_QUESTIONS), ("held-out paraphrases", HELDOUT_QUESTIONS),
                      ("heldout_templates", TRAIN_QUESTIONS), ("heldout_entities", TRAIN_QUESTIONS)]:
        data = "test" if split == "held-out paraphrases" else split
        name = split.replace("heldout_", "held-out ")
        oracle_table[name] = evaluate_oracle(oracle, last[data], datasets[data], center, questions=qs)
        probe_table[name] = {
            family: float("nan") if split == "held-out paraphrases"
            else probes[family].accuracy(last[data], family_labels(datasets[data], family))
            for family in FAMILIES
        }
    
    oracle_df, probe_df = pd.DataFrame(oracle_table).T, pd.DataFrame(probe_table).T
    display(pd.concat({"oracle": oracle_df, "probe": probe_df}).round(3))
    utils.plot_generalization_heatmaps(
        {"oracle": oracle_df, "linear probe bank": probe_df}, "Generalisation from layer-6 final-token activations"
    )

# %%

@t.inference_mode()
def oracle_top_answers(
    oracle: HookedTransformer, vectors: Tensor, question: str, center: Tensor | None, batch_size: int = 256
) -> list[str]:
    """The oracle's top-1 answer (as a string, leading space stripped) for each vector."""
    out = []
    for start in range(0, len(vectors), batch_size):
        v = vectors[start : start + batch_size]
        logits = oracle_logits(oracle, v, [question] * len(v), center)
        out += [oracle.to_string(tok).strip() for tok in logits.argmax(-1).tolist()]
    return out

# %%

if MAIN:
    heldout = datasets["heldout_entities"]
    for family, question, values in [("animal", TRAIN_QUESTIONS["animal"][0], HELDOUT_ANIMALS),
                                     ("city", TRAIN_QUESTIONS["city"][0], HELDOUT_CITIES)]:
        said = oracle_top_answers(oracle, last["heldout_entities"], question, center)
        print(f"Q: {question}")
        for value in values:
            counts = Counter(s for s, ex in zip(said, heldout) if getattr(ex, family) == value)
            total = sum(counts.values())
            print(f"  true {value:<9} -> " + ", ".join(f"{a} ({c / total:.0%})" for a, c in counts.most_common(3)))

# %%

if MAIN:
    first_questions = {family: qs[:1] for family, qs in TRAIN_QUESTIONS.items()}
    layer_acc = {family: [] for family in FAMILIES}
    for layer in tqdm(range(target.cfg.n_layers + 1)):
        layer_center = last_all["train"][layer].mean(0)
        accs = evaluate_oracle(oracle, last_all["test"][layer], datasets["test"], layer_center, questions=first_questions)
        for family in FAMILIES:
            layer_acc[family].append(accs[family])
    
    display(pd.DataFrame(layer_acc).T.round(2))
    utils.plot_layer_transfer(
        layer_acc, probe_acc, CHANCE, SOURCE_LAYER, "An oracle trained on layer 6, reading other layers"
    )

# %%

def accuracy_by_position(
    oracle: HookedTransformer,
    examples: list[SourceExample],
    acts: list[Float[Tensor, "seq d_model"]],
    center: Float[Tensor, " d_model"] | None,
    questions: dict[str, list[str]],
) -> dict[str, list[float]]:
    """Oracle accuracy per family when reading each token position in turn (examples must have equal length)."""
    seq_len = len(examples[0].tokens)
    assert all(len(ex.tokens) == seq_len for ex in examples), "all examples must have the same length"
    results = {family: [] for family in questions}
    for pos in range(seq_len):
        vectors = t.stack([a[pos] for a in acts])
        accs = evaluate_oracle(oracle, vectors, examples, center, questions=questions)
        for family in questions:
            results[family].append(accs[family])
    return results

# %%

if MAIN:
    sweep, sweep_acts = datasets["sweep"], acts["sweep"]
    seq_len = len(sweep[0].tokens)
    token_labels = ["BOS", "D", "ANIMAL", " in", "CITY", "VERB", " the", "OBJECT", "."]
    
    # Per-position probes, trained on the training sentences that use the same template
    same_template = [i for i, ex in enumerate(datasets["train"]) if ex.template == TRAIN_TEMPLATES[0]]
    t.manual_seed(0)
    probe_by_position = {family: [] for family in FAMILIES}
    for family in FAMILIES:
        y_train = family_labels([datasets["train"][i] for i in same_template], family)
        y_test = family_labels(sweep, family)
        for pos in range(seq_len):
            x_train = t.stack([acts["train"][i][pos] for i in same_template])
            x_test = t.stack([a[pos] for a in sweep_acts])
            probe = train_linear_probe(x_train, y_train, len(CLASSES[family]))
            probe_by_position[family].append(probe.accuracy(x_test, y_test))
    
    oracle_by_position = accuracy_by_position(oracle, sweep, sweep_acts, center, first_questions)
    print(f"{len(same_template)} same-template training sentences for the probes")
    display(pd.concat({"oracle": pd.DataFrame(oracle_by_position, index=token_labels).T,
                       "probe": pd.DataFrame(probe_by_position, index=token_labels).T}).round(2))
    utils.plot_position_sweep(
        token_labels,
        {"oracle (trained on final token)": oracle_by_position, "probe trained at each position": probe_by_position},
        sweep[0].available,
        CHANCE,
        "Reading every position of '{D} {ANIMAL} in {CITY} {VERB} the {OBJECT}.'",
    )

# %%

if MAIN:
    t.manual_seed(0)
    center_all = t.cat([a[1:] for a in acts["train"]]).mean(0)
    oracle_rp = HookedTransformer.from_pretrained("gpt2-small", device=device)
    losses_rp = train_oracle(
        oracle_rp, datasets["train"], acts["train"], center_all, n_steps=800 if device.type != "cpu" else 200, position="random"
    )
    oracle_rp_by_position = accuracy_by_position(oracle_rp, sweep, sweep_acts, center_all, first_questions)
    display(pd.DataFrame(oracle_rp_by_position, index=token_labels).T.round(2))
    utils.plot_position_sweep(
        token_labels,
        {
            "oracle (trained on final token)": oracle_by_position,
            "oracle (trained on random positions)": oracle_rp_by_position,
            "probe trained at each position": probe_by_position,
        },
        sweep[0].available,
        CHANCE,
        "Positional randomness in training: coverage across positions",
    )

# %%

if MAIN:
    own_position_rows = {}
    for name, model_, mu in [("oracle (final token)", oracle, center), ("oracle (random positions)", oracle_rp, center_all)]:
        row = {}
        for family in ["animal", "city"]:
            vectors = t.stack([a[ex.available[family]] for a, ex in zip(acts["heldout_entities"], heldout)])
            row[f"{family}: at its own token"] = evaluate_oracle(
                model_, vectors, heldout, mu, questions={family: TRAIN_QUESTIONS[family]}
            )[family]
            row[f"{family}: at final '.'"] = evaluate_oracle(
                model_, last["heldout_entities"], heldout, mu, questions={family: TRAIN_QUESTIONS[family]}
            )[family]
        own_position_rows[name] = row
    display(pd.DataFrame(own_position_rows).T.round(3))

# %%

if MAIN:
    confidences = {}
    question = TRAIN_QUESTIONS["animal"][0]
    n = len(test_vectors)
    with t.inference_mode():
        for kind in ["real", "shuffled", "random", "text-only"]:
            v = control_vectors(test_vectors, kind, center)
            logits = oracle_logits(oracle, v, [question] * n, center)
            confidences[kind] = logits.softmax(-1).max(-1).values.cpu().numpy()
    
    utils.plot_confidence(confidences, f"Oracle confidence on '{question}'")

# %%

def mean_difference_direction(vectors: Float[Tensor, "n d"], mask: Tensor) -> Float[Tensor, " d"]:
    """Mean of vectors[mask] minus mean of vectors[~mask]."""
    mask = mask.to(vectors.device).bool()
    return vectors[mask].mean(0) - vectors[~mask].mean(0)


@t.inference_mode()
def steering_sweep(
    oracle: HookedTransformer,
    vectors: Float[Tensor, "n d_model"],
    direction: Float[Tensor, " d_model"],
    alphas: list[float],
    question: str,
    answer: str,
    center: Float[Tensor, " d_model"] | None,
) -> list[float]:
    """For each alpha, the fraction of (vectors + alpha * direction) for which the oracle's top answer is `answer`."""
    answer_id = oracle.to_single_token(" " + answer)
    fractions = []
    for alpha in alphas:
        logits = oracle_logits(oracle, vectors + alpha * direction, [question] * len(vectors), center)
        fractions.append((logits.argmax(-1) == answer_id).float().mean().item())
    return fractions


if MAIN:
    tests.test_mean_difference_direction(mean_difference_direction)
    tests.test_steering_sweep(steering_sweep, oracle_logits, oracle, last["test"][:20], center)

# %%

if MAIN:
    positive_mask = t.tensor([ex.sentiment == "positive" for ex in datasets["train"]])
    sentiment_dir = mean_difference_direction(last["train"], positive_mask)
    random_dir = t.randn(sentiment_dir.shape, generator=t.Generator().manual_seed(0)).to(device)
    random_dir = random_dir / random_dir.norm() * sentiment_dir.norm()
    print(f"|d_sentiment| = {sentiment_dir.norm():.2f}, compared to a typical |v - mu| = {(last['train'] - center).norm(dim=-1).mean():.2f}")
    
    negatives = [i for i, ex in enumerate(datasets["test"]) if ex.sentiment == "negative"]
    neg_examples, neg_vectors = [datasets["test"][i] for i in negatives], last["test"][negatives]
    alphas = [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    question = TRAIN_QUESTIONS["sentiment"][0]
    
    curves = {
        "says 'positive' (sentiment direction)": steering_sweep(oracle, neg_vectors, sentiment_dir, alphas, question, "positive", center),
        "says 'positive' (random direction)": steering_sweep(oracle, neg_vectors, random_dir, alphas, question, "positive", center),
    }
    for family in ["city", "number"]:
        for name, direction in [("sentiment direction", sentiment_dir), ("random direction", random_dir)]:
            curves[f"{family} still correct ({name})"] = [
                evaluate_oracle(oracle, neg_vectors + a * direction, neg_examples, center, {family: first_questions[family]})[family]
                for a in alphas
            ]
    
    display(pd.DataFrame(curves, index=alphas).T.round(2))
    utils.plot_steering(
        alphas,
        curves,
        f"Steering negative sentences' activations (n={len(negatives)}) and asking the oracle",
        r"steering strength $\alpha$  (vector = $v + \alpha d$)",
    )

# %%

@t.inference_mode()
def patched_final_vectors(
    model: HookedTransformer,
    clean_tokens: Int[Tensor, "batch seq"],
    corrupt_tokens: Int[Tensor, "batch seq"],
    patch_pos: int,
    patch_layer: int,
    read_layer: int,
) -> Float[Tensor, "batch d_model"]:
    """Final-token residual stream at read_layer, after patching corrupt -> clean at (patch_layer, patch_pos)."""
    patch_name, read_name = resid_hook_name(model, patch_layer), resid_hook_name(model, read_layer)
    _, corrupt_cache = model.run_with_cache(corrupt_tokens, names_filter=patch_name)
    stored = {}

    def patch_hook(resid: Tensor, hook) -> Tensor:
        resid[:, patch_pos] = corrupt_cache[patch_name][:, patch_pos]
        return resid

    def read_hook(resid: Tensor, hook) -> None:
        stored["final"] = resid[:, -1].clone()

    model.run_with_hooks(clean_tokens, fwd_hooks=[(patch_name, patch_hook), (read_name, read_hook)])
    return stored["final"]


if MAIN:
    tests.test_patched_final_vectors(patched_final_vectors, resid_hook_name, target)

# %%

if MAIN:
    rng = random.Random(0)
    clean_examples = sweep[:300]
    corrupt_examples = []
    for ex in clean_examples:
        new_city = rng.choice([c for c in TRAIN_CITIES if c != ex.city])
        corrupt = SourceExample(**{**ex.__dict__, "text": ex.text.replace(f" {ex.city} ", f" {new_city} "), "city": new_city})
        corrupt_examples.append(annotate_example(target, corrupt))
    
    clean_tokens = t.tensor([ex.tokens for ex in clean_examples], device=device)
    corrupt_tokens = t.tensor([ex.tokens for ex in corrupt_examples], device=device)
    city_pos = clean_examples[0].available["city"]
    assert (clean_tokens != corrupt_tokens).sum(0).nonzero().flatten().tolist() == [city_pos]
    
    old_ids = answer_token_ids(oracle, [ex.city for ex in clean_examples])
    new_ids = answer_token_ids(oracle, [ex.city for ex in corrupt_examples])
    new_labels = family_labels(corrupt_examples, "city").to(device)
    city_question = TRAIN_QUESTIONS["city"][0]
    patch_layers = list(range(SOURCE_LAYER + 1))
    patch_curves = {k: [] for k in ["oracle says new city", "oracle says original city", "probe says new city", "oracle: animal still correct"]}
    with t.inference_mode():
        for layer in patch_layers:
            v = patched_final_vectors(target, clean_tokens, corrupt_tokens, city_pos, layer, SOURCE_LAYER)
            pred = oracle_logits(oracle, v, [city_question] * len(v), center).argmax(-1)
            patch_curves["oracle says new city"].append((pred == new_ids).float().mean().item())
            patch_curves["oracle says original city"].append((pred == old_ids).float().mean().item())
            patch_curves["probe says new city"].append(probes["city"].accuracy(v, new_labels))
            patch_curves["oracle: animal still correct"].append(
                evaluate_oracle(oracle, v, clean_examples, center, {"animal": first_questions["animal"]})["animal"]
            )
    
    display(pd.DataFrame(patch_curves, index=patch_layers).T.round(2))
    utils.plot_patching(patch_layers, patch_curves, "Patching the city token in GPT-2, read out at the final token (layer 6)")

# %%

@t.inference_mode()
def ask_oracle(
    oracle: HookedTransformer,
    target: HookedTransformer,
    sentence: str,
    question: str,
    center: Tensor | None,
    position: int = -1,
    layer: int = SOURCE_LAYER,
    k: int = 3,
) -> str:
    """Runs `sentence` through the target, injects the activation at `position`, and returns the oracle's top-k answers."""
    tokens = target.to_tokens(sentence)
    _, cache = target.run_with_cache(tokens, names_filter=resid_hook_name(target, layer))
    vector = cache[resid_hook_name(target, layer)][0, position][None]
    probs = oracle_logits(oracle, vector, [question], center)[0].softmax(-1)
    top = probs.topk(k)
    return ", ".join(f"{oracle.to_string(i)!r} ({p:.2f})" for p, i in zip(top.values.tolist(), top.indices.tolist()))

# %%

if MAIN:
    for sentence, question, position in [
        ("My cat in Dublin hated the snow.", "Which animal is this about?", -1),
        ("My cat in Dublin hated the snow.", "Which animal is this about?", 2),
        ("My cat in Dublin hated the snow.", "Is the feeling positive or negative?", -1),
        ("Yesterday the two old horses in Rome happily ate all the food in the market.", "Which city is mentioned?", -1),
        ("The dogs in Paris did not love the rain.", "Is the feeling positive or negative?", -1),
        ("The dogs in Paris did not hate the rain.", "Is the feeling positive or negative?", -1),
        ("The dogs in Paris loved the rain.", "What colour is the animal?", -1),
    ]:
        print(f"{sentence!r} [pos {position}]\n  Q: {question}\n  A: {ask_oracle(oracle, target, sentence, question, center, position)}")

# %%

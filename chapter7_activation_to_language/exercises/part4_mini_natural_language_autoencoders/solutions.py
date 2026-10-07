# %%


import copy
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
pd.set_option("display.width", 200)

# Make sure exercises are in the path
chapter = "chapter7_activation_to_language"
section = "part4_mini_natural_language_autoencoders"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part4_mini_natural_language_autoencoders.tests as tests
import part4_mini_natural_language_autoencoders.utils as utils

MAIN = __name__ == "__main__"

# %%

if MAIN:
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.requires_grad_(False)
    
    print(f"{model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

# %%

NOUNS = [
    ("key", "keys"), ("book", "books"), ("dog", "dogs"), ("car", "cars"), ("tree", "trees"), ("house", "houses"),
    ("door", "doors"), ("girl", "girls"), ("boy", "boys"), ("cat", "cats"), ("bird", "birds"), ("table", "tables"),
    ("chair", "chairs"), ("teacher", "teachers"), ("student", "students"), ("window", "windows"),
]  # fmt: skip
TIME_ADVERBS = ["Today", "Now", "Currently", "Yesterday", "Earlier", "Previously"]  # first three present, last three past
PREPOSITIONS = ["near", "behind", "beside", "with"]
VERBS = [" is", " are", " was", " were"]
LAYER = 10  # the layer whose activations we'll verbalise (the residual stream *before* block 10)


def make_dataset(n: int, seed: int = 0) -> pd.DataFrame:
    rows = []
    for time_idx, time_word in enumerate(TIME_ADVERBS):
        for i, j in [(i, j) for i in range(len(NOUNS)) for j in range(len(NOUNS)) if i != j]:
            for subject_plural in [0, 1]:
                for object_plural in [0, 1]:
                    for prep in PREPOSITIONS:
                        past = int(time_idx >= 3)
                        subject, obj = NOUNS[i][subject_plural], NOUNS[j][object_plural]
                        rows.append(
                            dict(
                                sentence=f"{time_word}, the {subject} {prep} the {obj} really",
                                time_word=time_word.lower(),
                                tense=["present", "past"][past],
                                subject=subject,
                                number=["singular", "plural"][subject_plural],
                                preposition=prep,
                                object=obj,
                                object_number=["singular", "plural"][object_plural],
                                verb=2 * past + subject_plural,
                            )
                        )
    random.Random(seed).shuffle(rows)
    return pd.DataFrame(rows[:n])

# %%

if MAIN:
    df = make_dataset(5000)
    N_TRAIN = 4000
    train_idx, test_idx = t.arange(N_TRAIN), t.arange(N_TRAIN, len(df))
    
    tokens = model.to_tokens(df["sentence"].tolist())
    verb_ids = t.tensor([model.to_single_token(v) for v in VERBS])
    labels = t.tensor(df["verb"].values)
    
    print(f"tokens: {tuple(tokens.shape)}")
    print([model.to_string(tok) for tok in tokens[0]])
    display(df.head(6))

# %%

if MAIN:
    with t.inference_mode():
        clean_verb_logits = t.cat(
            [model(tokens[i : i + 250].to(device))[:, -1, verb_ids.to(device)].cpu() for i in range(0, len(tokens), 250)]
        )
    clean_preds = clean_verb_logits.argmax(-1)
    print(f"GPT-2 picks the right verb on {(clean_preds == labels).float().mean():.1%} of sentences")
    for name, mask in [
        ("subject and object agree in number", df["number"].values == df["object_number"].values),
        ("subject and object disagree in number", df["number"].values != df["object_number"].values),
    ]:
        mask = t.tensor(mask)
        print(f"  {name:<40}: {(clean_preds[mask] == labels[mask]).float().mean():.1%}")

# %%

@t.inference_mode()
def get_final_activations(
    model: HookedTransformer, tokens: Int[Tensor, "batch seq"], layers: list[int], batch_size: int = 250
) -> Float[Tensor, "layer batch d_model"]:
    """Returns resid_pre at the final position for each layer in `layers`, as a CPU tensor."""
    names = [f"blocks.{layer}.hook_resid_pre" for layer in layers]
    chunks = []
    for start in range(0, len(tokens), batch_size):
        _, cache = model.run_with_cache(
            tokens[start : start + batch_size].to(model.cfg.device),
            names_filter=lambda name: name in names,
            stop_at_layer=max(layers) + 1,
        )
        chunks.append(t.stack([cache["resid_pre", layer][:, -1] for layer in layers]).cpu())
    return t.cat(chunks, dim=1)


if MAIN:
    tests.test_get_final_activations(get_final_activations, model)

# %%

if MAIN:
    ANALYSIS_LAYERS = [2, 4, 6, 8, 10]
    acts_by_layer = get_final_activations(model, tokens, ANALYSIS_LAYERS)
    acts = acts_by_layer[ANALYSIS_LAYERS.index(LAYER)]
    acts_unit = F.normalize(acts, dim=-1)  # unit-norm versions, which is what the NLA will reconstruct
    act_scale = acts[train_idx].norm(dim=-1).mean().item()  # used to rescale reconstructions before patching
    
    for layer, layer_acts in zip(ANALYSIS_LAYERS, acts_by_layer):
        norms = layer_acts.norm(dim=-1)
        print(f"layer {layer:>2}: mean norm {norms.mean():6.1f}, std of norm {norms.std():5.1f}")

# %%

def fve(h: Float[Tensor, "batch d_model"], h_hat: Float[Tensor, "batch d_model"]) -> float:
    """Fraction of variance of the unit-normalised activations `h` explained by the reconstructions `h_hat`."""
    h_unit = F.normalize(h.float(), dim=-1)
    h_hat = h_hat.float().to(h_unit.device)
    residual = (h_unit - h_hat).pow(2).sum()
    total = (h_unit - h_unit.mean(dim=0)).pow(2).sum()
    return (1 - residual / total).item()


if MAIN:
    tests.test_fve(fve)

# %%

def class_mean_reconstruction(
    train_h: Float[Tensor, "n_train d_model"], train_labels: list, test_labels: list
) -> Float[Tensor, "n_test d_model"]:
    """Predicts each test activation as the mean unit-normalised training activation with the same label."""
    train_unit = F.normalize(train_h.float(), dim=-1)
    groups: dict = {}
    for i, label in enumerate(train_labels):
        groups.setdefault(label, []).append(i)
    means = {label: train_unit[idx].mean(dim=0) for label, idx in groups.items()}
    fallback = train_unit.mean(dim=0)
    return t.stack([means.get(label, fallback) for label in test_labels])


if MAIN:
    tests.test_class_mean_reconstruction(class_mean_reconstruction)

# %%

if MAIN:
    DESCRIPTION_KINDS = {
        "number": ["number"],
        "tense": ["tense"],
        "number + tense": ["number", "tense"],
        "subject": ["subject"],
        "time word": ["time_word"],
        "preposition": ["preposition"],
        "object": ["object"],
        "subject + time word": ["subject", "time_word"],
    }
    
    
    def description_labels(df: pd.DataFrame, columns: list[str]) -> list[tuple]:
        return list(df[columns].itertuples(index=False, name=None))
    
    
    rng = np.random.default_rng(0)
    random_labels = rng.integers(0, 32, len(df)).tolist()
    oracle_table = {}
    for name, columns in DESCRIPTION_KINDS.items():
        labels_ = description_labels(df, columns)
        train_labels, test_labels = [labels_[i] for i in train_idx], [labels_[i] for i in test_idx]
        oracle_table[name] = [
            fve(layer_acts[test_idx], class_mean_reconstruction(layer_acts[train_idx], train_labels, test_labels))
            for layer_acts in acts_by_layer
        ]
    oracle_table["control: random labels"] = [
        fve(
            layer_acts[test_idx],
            class_mean_reconstruction(layer_acts[train_idx], random_labels[:N_TRAIN], random_labels[N_TRAIN:]),
        )
        for layer_acts in acts_by_layer
    ]
    oracle_df = pd.DataFrame(oracle_table, index=ANALYSIS_LAYERS).T
    utils.plot_factor_heatmap(
        oracle_df,
        "Description oracles: held-out FVE of the best reconstruction from each kind of description",
        control_row="control: random labels",
    )

# %%

@t.inference_mode()
def patch_final_position(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    vectors: Float[Tensor, "batch d_model"],
    layer: int,
    verb_ids: Int[Tensor, " 4"],
    batch_size: int = 250,
) -> Float[Tensor, "batch 4"]:
    """Runs `tokens` with resid_pre[layer][:, -1] replaced by `vectors`; returns the four verb logits."""
    out = []
    for start in range(0, len(tokens), batch_size):
        batch_vectors = vectors[start : start + batch_size].to(model.cfg.device)

        def hook(resid: Float[Tensor, "batch seq d_model"], hook) -> Float[Tensor, "batch seq d_model"]:
            resid[:, -1] = batch_vectors
            return resid

        logits = model.run_with_hooks(
            tokens[start : start + batch_size].to(model.cfg.device),
            fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook)],
        )
        out.append(logits[:, -1, verb_ids.to(model.cfg.device)].cpu())
    return t.cat(out)


if MAIN:
    tests.test_patch_final_position(patch_final_position, model, verb_ids)

# %%

if MAIN:
    mean_vector = acts_unit[train_idx].mean(dim=0)
    mean_patch = (F.normalize(mean_vector, dim=-1) * act_scale).expand(len(test_idx), -1)
    in_place_logits = patch_final_position(model, tokens[test_idx], mean_patch, LAYER, verb_ids)
    print(f"Mean activation spliced into its own sentence: right verb {(in_place_logits.argmax(-1) == labels[test_idx]).float().mean():.1%}")

# %%

if MAIN:
    source, target = "Yesterday, the keys behind the door really", "Today, the dog near the trees really"
    source_act = get_final_activations(model, model.to_tokens(source), [LAYER])[0]
    for name, sentence, patch in [
        ("source", source, None),
        ("target", target, None),
        ("target, with the source's layer-10 activation", target, source_act),
    ]:
        sentence_tokens = model.to_tokens(sentence)
        if patch is None:
            with t.inference_mode():
                logits = model(sentence_tokens)[:, -1, verb_ids.to(device)].cpu()
        else:
            logits = patch_final_position(model, sentence_tokens, patch, LAYER, verb_ids)
        probs = logits.softmax(-1)[0]
        print(f"{name:<46} " + "  ".join(f"{v.strip()}: {p:.2f}" for v, p in zip(VERBS, probs)))

# %%

def make_partners(labels: Int[Tensor, " n"], idx: Int[Tensor, " k"], seed: int = 0) -> Int[Tensor, " k"]:
    """For each index in `idx`, a random other index in `idx` whose label is different."""
    generator = t.Generator().manual_seed(seed)
    partners = []
    for i in idx.tolist():
        while True:
            j = idx[t.randint(0, len(idx), (1,), generator=generator)].item()
            if labels[j] != labels[i]:
                partners.append(j)
                break
    return t.tensor(partners)

# %%

def transplant_accuracy(
    model: HookedTransformer,
    tokens: Int[Tensor, "n seq"],
    labels: Int[Tensor, " n"],
    h_hat: Float[Tensor, "k d_model"],
    src_idx: Int[Tensor, " k"],
    dst_idx: Int[Tensor, " k"],
    scale: float,
    layer: int = LAYER,
) -> float:
    """Patches each reconstruction (rescaled) into its partner sentence; returns how often the source's verb wins."""
    vectors = F.normalize(h_hat.float(), dim=-1) * scale
    logits = patch_final_position(model, tokens[dst_idx], vectors, layer, verb_ids)
    return (logits.argmax(dim=-1) == labels[src_idx]).float().mean().item()


if MAIN:
    tests.test_transplant_accuracy(transplant_accuracy, model, verb_ids)

# %%

if MAIN:
    partners = make_partners(labels, test_idx)
    shuffled_sources = test_idx[t.randperm(len(test_idx), generator=t.Generator().manual_seed(1))]
    
    
    def transplant(h_hat: Tensor) -> float:
        return transplant_accuracy(model, tokens, labels, h_hat, test_idx, partners, act_scale)
    
    
    transplant_controls = {
        "A's own activation": transplant(acts_unit[test_idx]),
        "mean activation": transplant(mean_vector.expand(len(test_idx), -1)),
        "another sentence's activation": transplant(acts_unit[shuffled_sources]),
        "random direction": transplant(t.randn(len(test_idx), model.cfg.d_model, generator=t.Generator().manual_seed(2))),
    }
    utils.plot_bars(
        transplant_controls,
        "Transplant test at layer 10: patch A's activation into a sentence B with a different verb",
        ylabel="transplant accuracy\n(model predicts A's verb)",
        chance=0.25,
    )

# %%

if MAIN:
    ceiling_by_layer, mean_by_layer = [], []
    for layer, layer_acts in zip(ANALYSIS_LAYERS, acts_by_layer):
        scale = layer_acts[train_idx].norm(dim=-1).mean().item()
        layer_unit = F.normalize(layer_acts, dim=-1)
        ceiling_by_layer.append(transplant_accuracy(model, tokens, labels, layer_unit[test_idx], test_idx, partners, scale, layer))
        mean_by_layer.append(transplant_accuracy(model, tokens, labels, layer_unit[train_idx].mean(0).expand(len(test_idx), -1), test_idx, partners, scale, layer))
    utils.plot_curves(
        {"A's own activation": (ANALYSIS_LAYERS, ceiling_by_layer), "control: mean activation": (ANALYSIS_LAYERS, mean_by_layer)},
        "Transplant accuracy by layer: when does the final position decide the verb?",
        xlabel="layer of the patched activation (residual stream before block)",
        ylabel="transplant accuracy",
        hlines={"chance (25%)": 0.25},
        ylim=(0, 1.02),
    )

# %%

if MAIN:
    oracle_points = {}
    for name in ["number + tense", "subject", "time word", "subject + time word"]:
        labels_ = description_labels(df, DESCRIPTION_KINDS[name])
        h_hat = class_mean_reconstruction(acts[train_idx], [labels_[i] for i in train_idx], [labels_[i] for i in test_idx])
        oracle_points[name] = (fve(acts[test_idx], h_hat), transplant(h_hat))
    display(pd.DataFrame(oracle_points, index=["FVE", "transplant accuracy"]).T.round(3))

# %%

NOUN_WORDS = [form for pair in NOUNS for form in pair]
TIME_WORDS = [word.lower() for word in TIME_ADVERBS]
ABSTRACT_WORDS = ["singular", "plural", "present", "past"]
WORDS = NOUN_WORDS + TIME_WORDS + ABSTRACT_WORDS + PREPOSITIONS + ["."]
EOS = WORDS.index(".")
MAX_WORDS = 4
FACT_ORDER = ["subject", "number", "tense", "time", "object"]


def true_facts(row) -> dict[str, list[str]]:
    """The five true facts about a sentence (a row of the dataset), each as a list of words."""
    return {
        "subject": [row.subject],
        "number": [row.number],
        "tense": [row.tense],
        "time": [row.time_word],
        "object": [row.preposition, row.object],
    }


def teacher_description(facts: dict[str, list[str]], rng: random.Random, max_words: int = MAX_WORDS) -> list[str]:
    """A random non-empty subset of the true facts, in canonical order, of at most `max_words` words."""
    while True:
        words = [word for fact in FACT_ORDER if rng.random() < 0.5 for word in facts[fact]]
        if 1 <= len(words) <= max_words:
            return words


def encode_descriptions(descriptions: list[list[str]], max_words: int = MAX_WORDS) -> Int[Tensor, "batch slots"]:
    """Word lists -> ids, padded with EOS to max_words + 1 slots (so the last slot is always EOS)."""
    ids = t.full((len(descriptions), max_words + 1), EOS, dtype=t.long)
    for i, words in enumerate(descriptions):
        assert len(words) <= max_words, f"description too long: {words}"
        if words:
            ids[i, : len(words)] = t.tensor([WORDS.index(word) for word in words])
    return ids


def decode_descriptions(ids: Int[Tensor, "batch slots"]) -> list[str]:
    """Ids -> strings, stopping at the first EOS."""
    texts = []
    for row in ids.tolist():
        words = []
        for i in row:
            if i == EOS:
                break
            words.append(WORDS[i])
        texts.append(" ".join(words))
    return texts

# %%

if MAIN:
    facts = [true_facts(row) for row in df.itertuples()]
    teacher_rng = random.Random(0)
    print(f"{len(WORDS)} words in the bottleneck vocabulary\n")
    for i in range(4):
        examples = [" ".join(teacher_description(facts[i], teacher_rng)) for _ in range(3)]
        print(f"{df.sentence[i]!r:<52} -> {examples}")
    print()
    print(encode_descriptions([["houses", "earlier"], ["plural", "past", "behind", "window"]]))

# %%

class ActivationVerbaliser(nn.Module):
    def __init__(
        self, d_act: int = 768, n_words: int = len(WORDS), d_model: int = 128, n_layers: int = 2,
        n_heads: int = 4, max_words: int = MAX_WORDS,
    ):  # fmt: skip
        super().__init__()
        self.max_words = max_words
        self.act_in = nn.Linear(d_act, d_model)
        self.word_embed = nn.Embedding(n_words, d_model)
        self.pos_embed = nn.Parameter(t.randn(max_words + 1, d_model) * 0.02)
        block = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.0, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(block, n_layers, enable_nested_tensor=False)
        self.unembed = nn.Linear(d_model, n_words)

    def forward(
        self,
        h: Float[Tensor, "batch d_act"],
        prev_ids: Int[Tensor, "batch t"] | None = None,
        prev_embeds: Float[Tensor, "batch t d_model"] | None = None,
    ) -> Float[Tensor, "batch t_plus_1 n_words"]:
        """Logits for the next word at every position: position 0 is the activation, position i > 0 is word i."""
        if prev_embeds is None:
            prev_embeds = self.word_embed(prev_ids)
        x = t.cat([self.act_in(F.normalize(h, dim=-1))[:, None], prev_embeds], dim=1)
        x = x + self.pos_embed[: x.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        return self.unembed(self.blocks(x, mask=mask, is_causal=True))


if MAIN:
    tests.test_verbaliser_forward(ActivationVerbaliser)

# %%

def description_logprobs(
    av: ActivationVerbaliser, h: Float[Tensor, "batch d_act"], ids: Int[Tensor, "batch slots"]
) -> tuple[Float[Tensor, "batch slots n_words"], Float[Tensor, "batch slots"]]:
    """Returns (log-probabilities over the vocabulary at every slot, log-probability of each word in `ids`)."""
    logprobs = av(h, ids[:, :-1]).log_softmax(dim=-1)
    token_logprobs = logprobs.gather(dim=-1, index=ids[..., None])[..., 0]
    return logprobs, token_logprobs


def description_mask(ids: Int[Tensor, "batch slots"]) -> Bool[Tensor, "batch slots"]:
    """True for every slot up to and including the first EOS."""
    is_eos = (ids == EOS).long()
    eos_before = is_eos.cumsum(dim=-1) - is_eos  # number of EOS strictly before each slot
    return eos_before == 0


if MAIN:
    tests.test_description_logprobs(description_logprobs, ActivationVerbaliser)
    tests.test_description_mask(description_mask, EOS)

# %%

@t.no_grad()
def sample_descriptions(
    av: ActivationVerbaliser, h: Float[Tensor, "batch d_act"], greedy: bool = False, temperature: float = 1.0
) -> Int[Tensor, "batch slots"]:
    """Autoregressively samples (or greedily decodes) one description per activation."""
    ids = t.zeros(h.shape[0], 0, dtype=t.long, device=h.device)
    for _ in range(av.max_words):
        logits = av(h, ids)[:, -1] / temperature
        if greedy:
            next_ids = logits.argmax(dim=-1)
        else:
            next_ids = t.distributions.Categorical(logits=logits).sample()
        ids = t.cat([ids, next_ids[:, None]], dim=1)
    ids = t.cat([ids, t.full_like(ids[:, :1], EOS)], dim=1)
    ids[~description_mask(ids)] = EOS
    return ids


if MAIN:
    tests.test_sample_descriptions(sample_descriptions, ActivationVerbaliser, EOS)

# %%

def train_verbaliser(
    av: ActivationVerbaliser,
    h_train: Float[Tensor, "n d_act"],
    facts_train: list[dict],
    n_steps: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> list[float]:
    """Supervised warm start: maximise the log-likelihood of random true (teacher) descriptions."""
    rng = random.Random(seed)
    generator = t.Generator().manual_seed(seed)
    optimizer = t.optim.Adam(av.parameters(), lr=lr)
    device = next(av.parameters()).device
    losses = []
    for _ in tqdm(range(n_steps), desc="warm start"):
        idx = t.randint(0, len(h_train), (batch_size,), generator=generator)
        descriptions = [teacher_description(facts_train[i], rng, av.max_words) for i in idx.tolist()]
        ids = encode_descriptions(descriptions, av.max_words).to(device)
        _, token_logprobs = description_logprobs(av, h_train[idx].to(device), ids)
        loss = -(token_logprobs * description_mask(ids)).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# %%

if MAIN:
    t.manual_seed(0)
    facts_train = [facts[i] for i in train_idx]
    h_train = acts_unit[train_idx].to(device)
    h_test = acts_unit[test_idx].to(device)
    
    av_warm = ActivationVerbaliser().to(device)
    warm_losses = train_verbaliser(av_warm, h_train, facts_train)
    print(f"loss: first step {warm_losses[0]:.2f}, last 100 steps {np.mean(warm_losses[-100:]):.2f}")

# %%

if MAIN:
    t.manual_seed(0)
    show = test_idx[:8]
    greedy_texts = decode_descriptions(sample_descriptions(av_warm, acts_unit[show].to(device), greedy=True))
    sampled = [decode_descriptions(sample_descriptions(av_warm, acts_unit[show].to(device))) for _ in range(3)]
    display(pd.DataFrame(
        {"sentence": df.sentence[show.tolist()].values, "greedy": greedy_texts, "samples": [" | ".join(s) for s in zip(*sampled)]}
    ))

# %%

if MAIN:
    WORD_TOKEN_IDS = t.tensor([model.to_single_token(" " + word) for word in WORDS[:-1]] + [model.to_single_token(".")])
    print(f"'houses earlier . . .' as GPT-2 sees it: {model.to_string(t.cat([t.tensor([model.tokenizer.bos_token_id]), WORD_TOKEN_IDS[encode_descriptions([['houses', 'earlier']])[0]]]))!r}")

# %%

@t.no_grad()
def read_with_gpt2(
    model: HookedTransformer, gpt2_tokens: Int[Tensor, "batch seq"], layer: int
) -> Float[Tensor, "batch d_model"]:
    """resid_pre of block `layer` at the final position."""
    _, cache = model.run_with_cache(
        gpt2_tokens.to(model.cfg.device), names_filter=f"blocks.{layer}.hook_resid_pre", stop_at_layer=layer + 1
    )
    return cache["resid_pre", layer][:, -1]


def text_features(
    model: HookedTransformer, ids: Int[Tensor, "batch slots"], layer: int = LAYER
) -> Float[Tensor, "batch d_model"]:
    """Reads descriptions (ids over WORDS) with GPT-2: BOS, then one token per slot (EOS slots become '.')."""
    gpt2_tokens = WORD_TOKEN_IDS.to(ids.device)[ids]
    bos = t.full_like(gpt2_tokens[:, :1], model.tokenizer.bos_token_id)
    return read_with_gpt2(model, t.cat([bos, gpt2_tokens], dim=1), layer)


class ActivationReconstructor(nn.Module):
    def __init__(self, d_features: int = 768, d_act: int = 768):
        super().__init__()
        self.head = nn.Linear(d_features, d_act)

    def forward(self, features: Float[Tensor, "batch d_features"]) -> Float[Tensor, "batch d_act"]:
        """Affine map applied to the normalised GPT-2 features; returns a prediction of the unit-norm activation."""
        return self.head(F.normalize(features, dim=-1))


if MAIN:
    tests.test_text_features(text_features, model, WORD_TOKEN_IDS, encode_descriptions)
    tests.test_reconstructor(ActivationReconstructor)

# %%

def train_reconstructor(
    ar: ActivationReconstructor,
    optimizer: t.optim.Optimizer,
    av: ActivationVerbaliser,
    h_train: Float[Tensor, "n d_act"],
    n_steps: int = 400,
    batch_size: int = 256,
    seed: int = 0,
) -> list[float]:
    """Trains the reconstructor by regression on descriptions sampled from a fixed verbaliser."""
    generator = t.Generator().manual_seed(seed)
    losses = []
    for _ in tqdm(range(n_steps), desc="reconstructor"):
        idx = t.randint(0, len(h_train), (batch_size,), generator=generator)
        h = h_train[idx]
        ids = sample_descriptions(av, h)
        loss = (ar(text_features(model, ids)) - h).pow(2).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# %%

if MAIN:
    t.manual_seed(0)
    ar_warm = ActivationReconstructor().to(device)
    ar_warm_opt = t.optim.Adam(ar_warm.parameters(), lr=1e-3)
    ar_losses = train_reconstructor(ar_warm, ar_warm_opt, av_warm, h_train)
    total_variance = (h_train - h_train.mean(0)).pow(2).sum(-1).mean().item()
    print(f"AR loss: first step {ar_losses[0]:.3f}, last 50 steps {np.mean(ar_losses[-50:]):.4f}")
    print(f"(total variance of the unit-norm activations, i.e. the loss of predicting the mean: {total_variance:.4f})")

# %%

def claim_accuracy(descriptions: list[str], facts: list[dict[str, list[str]]]) -> tuple[float, float]:
    """Returns (fraction of claims that are true, mean number of claims per description)."""
    n_claims, n_true = 0, 0
    for text, fact in zip(descriptions, facts):
        words = text.split()
        k = 0
        while k < len(words):
            word = words[k]
            if word in PREPOSITIONS:
                n_claims += 1
                if k + 1 < len(words) and words[k + 1] in NOUN_WORDS:
                    n_true += [word, words[k + 1]] == fact["object"]
                    k += 2
                    continue
            elif word in NOUN_WORDS:
                n_claims += 1
                n_true += word == fact["subject"][0]
            elif word in ABSTRACT_WORDS[:2]:
                n_claims += 1
                n_true += word == fact["number"][0]
            elif word in ABSTRACT_WORDS[2:]:
                n_claims += 1
                n_true += word == fact["tense"][0]
            elif word in TIME_WORDS:
                n_claims += 1
                n_true += word == fact["time"][0]
            k += 1
    return n_true / max(n_claims, 1), n_claims / len(descriptions)


if MAIN:
    tests.test_claim_accuracy(claim_accuracy)

# %%

facts_test = None  # set in the main cells below


@t.no_grad()
def evaluate_nla(av: ActivationVerbaliser, ar: ActivationReconstructor, with_behaviour: bool = False) -> dict:
    ids = sample_descriptions(av, h_test, greedy=True)
    h_hat = ar(text_features(model, ids)).cpu()
    texts = decode_descriptions(ids)
    claims_true, claims_per = claim_accuracy(texts, facts_test)
    result = dict(fve=fve(acts[test_idx], h_hat), claims_true=claims_true, claims_per_description=claims_per)
    if with_behaviour:
        result["transplant"] = transplant(h_hat)
    return result

# %%

if MAIN:
    facts_test = [facts[i] for i in test_idx]
    warm_eval = evaluate_nla(av_warm, ar_warm, with_behaviour=True)
    print({k: round(v, 3) for k, v in warm_eval.items()})

# %%

if MAIN:
    def facts_mentioned(text: str) -> str:
        words = text.split()
        kinds = []
        for word, previous in zip(words, [None] + words[:-1]):
            if word in PREPOSITIONS:
                kinds.append("object")
            elif word in NOUN_WORDS and previous not in PREPOSITIONS:
                kinds.append("subject")
            elif word in ABSTRACT_WORDS:
                kinds.append("number" if word in ABSTRACT_WORDS[:2] else "tense")
            elif word in TIME_WORDS:
                kinds.append("time")
        return " + ".join(kind for kind in FACT_ORDER if kind in kinds)
    
    
    t.manual_seed(0)
    total_var_test = (h_test - h_test.mean(0)).pow(2).sum(-1).mean().item()
    errors, groups = [], []
    random_errors, wrong_errors = [], []
    with t.no_grad():
        for _ in range(8):
            ids = sample_descriptions(av_warm, h_test)
            h_hat = ar_warm(text_features(model, ids))
            errors.append((h_hat - h_test).pow(2).sum(-1).cpu())
            groups += [facts_mentioned(text) for text in decode_descriptions(ids)]
            # controls: random words in the same slots, and the description of a different sentence
            random_ids = t.where(ids == EOS, ids, t.randint_like(ids, 0, EOS))
            random_errors.append((ar_warm(text_features(model, random_ids)) - h_test).pow(2).sum(-1).cpu())
            wrong_errors.append((h_hat.roll(1, dims=0) - h_test).pow(2).sum(-1).cpu())
    errors = t.cat(errors)
    group_stats = pd.Series(errors.numpy()).groupby(groups).agg(["mean", "size"])
    shown = ["subject + time + object", "subject + time", "subject", "time", "object", "number + tense", "tense", "number"]
    bars = {name: 1 - group_stats.loc[name, "mean"] / total_var_test for name in shown}
    print(group_stats.loc[shown, "size"].to_dict())
    bars["control: random words"] = 1 - t.cat(random_errors).mean().item() / total_var_test
    bars["control: wrong sentence"] = 1 - t.cat(wrong_errors).mean().item() / total_var_test
    utils.plot_bars(
        bars,
        "Warm-started NLA: FVE of sampled descriptions, grouped by which facts they mention (8 samples per sentence)",
        ylabel="FVE",
        ylim=(-1.0, 1.0),
        n_highlight=len(bars) - 2,
        fmt="{:.2f}",
    )

# %%

def group_advantages(rewards: Float[Tensor, " n"], group_size: int, eps: float = 1e-6) -> Float[Tensor, " n"]:
    """Normalises each consecutive group of rewards to zero mean and unit standard deviation."""
    grouped = rewards.view(-1, group_size)
    advantages = (grouped - grouped.mean(dim=1, keepdim=True)) / (grouped.std(dim=1, keepdim=True) + eps)
    return advantages.view(-1)


if MAIN:
    tests.test_group_advantages(group_advantages)

# %%

def kl_to_reference(
    logprobs: Float[Tensor, "batch slots n_words"],
    ref_logprobs: Float[Tensor, "batch slots n_words"],
    mask: Bool[Tensor, "batch slots"],
) -> Float[Tensor, " batch"]:
    """Exact KL(policy || reference) summed over the unmasked slots of each description."""
    per_slot = (logprobs.exp() * (logprobs - ref_logprobs)).sum(dim=-1)
    return (per_slot * mask).sum(dim=-1)


def policy_gradient_loss(
    token_logprobs: Float[Tensor, "batch slots"], mask: Bool[Tensor, "batch slots"], advantages: Float[Tensor, " batch"]
) -> Float[Tensor, ""]:
    """REINFORCE loss: minus the advantage-weighted log-probability of each description, averaged."""
    return -(advantages * (token_logprobs * mask).sum(dim=-1)).mean()


if MAIN:
    tests.test_kl_to_reference(kl_to_reference)
    tests.test_policy_gradient_loss(policy_gradient_loss)

# %%

def train_nla(
    av: ActivationVerbaliser,
    ar: ActivationReconstructor,
    ar_optimizer: t.optim.Optimizer,
    h_train: Float[Tensor, "n d_act"],
    n_steps: int = 600,
    batch_size: int = 64,
    group_size: int = 8,
    beta: float = 0.05,
    av_ref: ActivationVerbaliser | None = None,
    lr_av: float = 3e-4,
    train_av: bool = True,
    eval_every: int = 50,
    seed: int = 0,
) -> list[dict]:
    """End-to-end NLA training: GRPO-style policy gradient for the verbaliser, regression for the reconstructor."""
    generator = t.Generator().manual_seed(seed)
    av_optimizer = t.optim.Adam(av.parameters(), lr=lr_av)
    history = []
    for step in tqdm(range(n_steps), desc="NLA training"):
        if step % eval_every == 0:
            history.append({"step": step, **evaluate_nla(av, ar)})
        idx = t.randint(0, len(h_train), (batch_size,), generator=generator)
        h = h_train[idx].repeat_interleave(group_size, dim=0)
        ids = sample_descriptions(av, h)
        errors = (ar(text_features(model, ids)) - h).pow(2).sum(dim=-1)

        # Reconstructor: one regression step on the sampled descriptions
        ar_optimizer.zero_grad()
        errors.mean().backward()
        ar_optimizer.step()

        # Verbaliser: policy gradient with group baseline, plus KL to the reference
        if train_av:
            rewards = -t.log(errors.detach() + 1e-6)
            advantages = group_advantages(rewards, group_size)
            logprobs, token_logprobs = description_logprobs(av, h, ids)
            mask = description_mask(ids)
            loss = policy_gradient_loss(token_logprobs, mask, advantages)
            if beta > 0:
                with t.no_grad():
                    ref_logprobs, _ = description_logprobs(av_ref, h, ids)
                loss = loss + beta * kl_to_reference(logprobs, ref_logprobs, mask).mean()
            av_optimizer.zero_grad()
            loss.backward()
            av_optimizer.step()
    history.append({"step": n_steps, **evaluate_nla(av, ar)})
    return history

# %%

def clone_reconstructor(
    ar: ActivationReconstructor, optimizer: t.optim.Optimizer
) -> tuple[ActivationReconstructor, t.optim.Optimizer]:
    ar_copy = copy.deepcopy(ar)
    optimizer_copy = type(optimizer)(ar_copy.parameters(), **optimizer.defaults)
    optimizer_copy.load_state_dict(optimizer.state_dict())
    return ar_copy, optimizer_copy

# %%

if MAIN:
    t.manual_seed(0)
    av_ref = copy.deepcopy(av_warm).requires_grad_(False)
    av_rl = copy.deepcopy(av_warm)
    ar_rl, ar_rl_opt = clone_reconstructor(ar_warm, ar_warm_opt)
    history_rl = train_nla(av_rl, ar_rl, ar_rl_opt, h_train, av_ref=av_ref)
    print({k: round(v, 3) for k, v in history_rl[-1].items()})

# %%

if MAIN:
    with t.no_grad():
        rl_texts = decode_descriptions(sample_descriptions(av_rl, acts_unit[show].to(device), greedy=True))
    display(pd.DataFrame({"sentence": df.sentence[show.tolist()].values, "warm start": greedy_texts, "after RL": rl_texts}))
    
    rl_test_texts = decode_descriptions(sample_descriptions(av_rl, h_test, greedy=True))
    print("Most common fact combinations in held-out descriptions after RL:")
    print(Counter(facts_mentioned(text) for text in rl_test_texts).most_common(4))
    wrong = [i for i, (text, fact) in enumerate(zip(rl_test_texts, facts_test)) if claim_accuracy([text], [fact])[0] < 1]
    print(f"\n{len(wrong)} of {len(rl_test_texts)} descriptions contain a false claim, e.g.:")
    for i in wrong[:5]:
        print(f"  {df.sentence[int(test_idx[i])]!r:<50} -> {rl_test_texts[i]!r}")

# %%

if MAIN:
    t.manual_seed(0)
    ar_only, ar_only_opt = clone_reconstructor(ar_warm, ar_warm_opt)
    history_ar_only = train_nla(copy.deepcopy(av_warm), ar_only, ar_only_opt, h_train, n_steps=300, train_av=False)
    
    t.manual_seed(0)
    av_cold = ActivationVerbaliser().to(device)
    ar_cold = ActivationReconstructor().to(device)
    ar_cold_opt = t.optim.Adam(ar_cold.parameters(), lr=1e-3)
    history_cold = train_nla(av_cold, ar_cold, ar_cold_opt, h_train, n_steps=300, beta=0.0)
    with t.no_grad():
        cold_texts = decode_descriptions(sample_descriptions(av_cold, h_test, greedy=True))
    print(f"Cold start: {len(set(cold_texts))} distinct descriptions on the test set; most common: {Counter(cold_texts).most_common(2)}")

# %%

def straight_through_gumbel(logits: Float[Tensor, "batch n"], tau: float = 1.0) -> Float[Tensor, "batch n"]:
    """Hard one-hot sample in the forward pass; gradient of the Gumbel-softmax in the backward pass."""
    u = t.rand_like(logits).clamp(1e-9, 1 - 1e-9)
    gumbel = -t.log(-t.log(u))
    soft = ((logits + gumbel) / tau).softmax(dim=-1)
    hard = F.one_hot(soft.argmax(dim=-1), logits.shape[-1]).to(soft.dtype)
    return hard - soft.detach() + soft


if MAIN:
    tests.test_straight_through_gumbel(straight_through_gumbel)

# %%

def gumbel_sample_descriptions(
    av: ActivationVerbaliser, h: Float[Tensor, "batch d_act"], tau: float = 1.0
) -> Float[Tensor, "batch slots n_words"]:
    n_words = av.unembed.out_features
    eos = F.one_hot(t.tensor(EOS, device=h.device), n_words).float()
    done = t.zeros(h.shape[0], 1, device=h.device)
    words = []
    for _ in range(av.max_words):
        prev_embeds = t.stack([y @ av.word_embed.weight for y in words], dim=1) if words else h.new_zeros(h.shape[0], 0, av.word_embed.embedding_dim)
        y = straight_through_gumbel(av(h, prev_embeds=prev_embeds)[:, -1], tau)
        y = done * eos + (1 - done) * y  # after an EOS, every slot is EOS
        done = t.maximum(done, y[:, EOS : EOS + 1].detach())
        words.append(y)
    words.append(eos.expand(h.shape[0], -1))
    return t.stack(words, dim=1)


def soft_text_features(
    model: HookedTransformer, word_onehots: Float[Tensor, "batch slots n_words"], layer: int = LAYER
) -> Float[Tensor, "batch d_model"]:
    soft_embeds = word_onehots @ model.W_E[WORD_TOKEN_IDS.to(word_onehots.device)]
    placeholder = t.full((word_onehots.shape[0], word_onehots.shape[1] + 1), model.tokenizer.bos_token_id)
    store = {}

    def embed_hook(embed: Tensor, hook) -> Tensor:
        return t.cat([embed[:, :1], soft_embeds], dim=1)  # keep BOS, replace the rest

    def grab_hook(resid: Tensor, hook) -> None:
        store["resid"] = resid

    model.run_with_hooks(
        placeholder.to(word_onehots.device),
        fwd_hooks=[("hook_embed", embed_hook), (f"blocks.{layer}.hook_resid_pre", grab_hook)],
        stop_at_layer=layer + 1,
    )
    return store["resid"][:, -1]

# %%

def train_nla_gumbel(
    av: ActivationVerbaliser,
    ar: ActivationReconstructor,
    ar_optimizer: t.optim.Optimizer,
    h_train: Float[Tensor, "n d_act"],
    n_steps: int = 300,
    batch_size: int = 256,
    tau: float = 1.0,
    lr_av: float = 3e-4,
    eval_every: int = 50,
    seed: int = 0,
) -> list[dict]:
    generator = t.Generator().manual_seed(seed)
    av_optimizer = t.optim.Adam(av.parameters(), lr=lr_av)
    history = []
    for step in tqdm(range(n_steps), desc="Gumbel NLA training"):
        if step % eval_every == 0:
            history.append({"step": step, **evaluate_nla(av, ar)})
        h = h_train[t.randint(0, len(h_train), (batch_size,), generator=generator)]
        word_onehots = gumbel_sample_descriptions(av, h, tau)
        loss = (ar(soft_text_features(model, word_onehots)) - h).pow(2).sum(dim=-1).mean()
        av_optimizer.zero_grad()
        ar_optimizer.zero_grad()
        loss.backward()
        av_optimizer.step()
        ar_optimizer.step()
    history.append({"step": n_steps, **evaluate_nla(av, ar)})
    return history

# %%

if MAIN:
    t.manual_seed(0)
    av_gumbel = copy.deepcopy(av_warm)
    ar_gumbel, ar_gumbel_opt = clone_reconstructor(ar_warm, ar_warm_opt)
    history_gumbel = train_nla_gumbel(av_gumbel, ar_gumbel, ar_gumbel_opt, h_train)
    with t.no_grad():
        gumbel_texts = decode_descriptions(sample_descriptions(av_gumbel, acts_unit[show].to(device), greedy=True))
    display(pd.DataFrame({"sentence": df.sentence[show.tolist()].values, "REINFORCE": rl_texts, "Gumbel straight-through": gumbel_texts}))

# %%

if MAIN:
    runs = {
        "REINFORCE from warm start (β=0.05)": history_rl,
        "control: reconstructor only": history_ar_only,
        "control: REINFORCE from cold start": history_cold,
        "Gumbel straight-through from warm start": history_gumbel,
    }
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.5))
    for (name, history), style in zip(runs.items(), ["-o", "--s", ":^", "-.v"]):
        steps = [entry["step"] for entry in history]
        ax0.plot(steps, [entry["fve"] for entry in history], style, label=name, markersize=4)
        ax1.plot(steps, [entry["claims_true"] for entry in history], style, label=name, markersize=4)
    ax0.axhline(oracle_points["subject + time word"][0], color="gray", ls="--", lw=1, label="oracle: subject + time word")
    ax0.set_ylim(-0.3, 1)
    ax0.set_ylabel("held-out FVE (greedy descriptions)")
    ax0.set_title("Reconstruction: what the objective rewards", fontsize=10)
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("fraction of claims that are true")
    ax1.set_title("Truthfulness: what the objective doesn't directly reward", fontsize=10)
    for ax in (ax0, ax1):
        ax.set_xlabel("training step")
        ax.grid(alpha=0.3)
    handles, names = ax0.get_legend_handles_labels()
    fig.legend(handles, names, loc="lower center", ncol=3, fontsize=9)
    plt.tight_layout(rect=(0, 0.12, 1, 1))
    plt.show()

# %%

if MAIN:
    t.manual_seed(0)
    av_nokl = copy.deepcopy(av_warm)
    ar_nokl, ar_nokl_opt = clone_reconstructor(ar_warm, ar_warm_opt)
    history_nokl = train_nla(av_nokl, ar_nokl, ar_nokl_opt, h_train, beta=0.0, eval_every=600)
    
    
    @t.no_grad()
    def mean_kl_to_warm_start(av: ActivationVerbaliser) -> float:
        ids = sample_descriptions(av, h_test)
        logprobs, _ = description_logprobs(av, h_test, ids)
        ref_logprobs, _ = description_logprobs(av_ref, h_test, ids)
        return kl_to_reference(logprobs, ref_logprobs, description_mask(ids)).mean().item()
    
    
    for name, av_, history in [("beta = 0.05", av_rl, history_rl), ("beta = 0", av_nokl, history_nokl), ("Gumbel (no KL)", av_gumbel, history_gumbel)]:
        final = history[-1]
        with t.no_grad():
            examples = decode_descriptions(sample_descriptions(av_, acts_unit[show[:3]].to(device), greedy=True))
        print(f"{name:<15} FVE {final['fve']:.3f} | claims true {final['claims_true']:.3f} | KL from warm start {mean_kl_to_warm_start(av_):5.2f} nats | {examples}")

# %%

def pca_reconstruction(
    train_h: Float[Tensor, "n_train d"], test_h: Float[Tensor, "n_test d"], k: int
) -> Float[Tensor, "n_test d"]:
    """Projects the unit-normalised test activations onto the top-k principal components of the training set."""
    train_unit, test_unit = F.normalize(train_h.float(), dim=-1), F.normalize(test_h.float(), dim=-1)
    mean = train_unit.mean(dim=0)
    _, _, Vh = t.linalg.svd(train_unit - mean, full_matrices=False)
    V_k = Vh[:k].T
    return mean + (test_unit - mean) @ V_k @ V_k.T


if MAIN:
    tests.test_pca_reconstruction(pca_reconstruction)

# %%

if MAIN:
    def train_full_nla(max_words: int, rl_steps: int = 400, seed: int = 0) -> tuple[ActivationVerbaliser, ActivationReconstructor]:
        t.manual_seed(seed)
        av = ActivationVerbaliser(max_words=max_words).to(device)
        train_verbaliser(av, h_train, facts_train, seed=seed)
        ar = ActivationReconstructor().to(device)
        ar_opt = t.optim.Adam(ar.parameters(), lr=1e-3)
        train_reconstructor(ar, ar_opt, av, h_train, seed=seed)
        train_nla(av, ar, ar_opt, h_train, n_steps=rl_steps, av_ref=copy.deepcopy(av).requires_grad_(False), eval_every=rl_steps, seed=seed)
        return av, ar
    
    
    nlas = {1: train_full_nla(1), 2: train_full_nla(2), 4: (av_rl, ar_rl)}
    nla_results = {k: evaluate_nla(av, ar, with_behaviour=True) for k, (av, ar) in nlas.items()}
    pca_results = {}
    for k in [1, 2, 4, 8, 16]:
        h_hat = pca_reconstruction(acts[train_idx], acts[test_idx], k)
        pca_results[k] = dict(fve=fve(acts[test_idx], h_hat), transplant=transplant(h_hat))
    for k, (av, _) in nlas.items():
        with t.no_grad():
            texts = decode_descriptions(sample_descriptions(av, acts_unit[show[:3]].to(device), greedy=True))
        print(f"{k} word(s): FVE {nla_results[k]['fve']:.3f}, transplant {nla_results[k]['transplant']:.3f}, claims true {nla_results[k]['claims_true']:.3f}, e.g. {texts}")
    for k, result in pca_results.items():
        print(f"PCA {k:>2} dims: FVE {result['fve']:.3f}, transplant {result['transplant']:.3f}")

# %%

if MAIN:
    sizes = [1, 2, 4, 8, 16]
    utils.plot_budget_sweep(
        sizes,
        fve_curves={
            "NLA (k words)": [nla_results[k]["fve"] for k in [1, 2, 4]],
            "PCA (k dimensions)": [pca_results[k]["fve"] for k in sizes],
        },
        behaviour_curves={
            "NLA (k words)": [nla_results[k]["transplant"] for k in [1, 2, 4]],
            "PCA (k dimensions)": [pca_results[k]["transplant"] for k in sizes],
            "A's own activation": [transplant_controls["A's own activation"]] * len(sizes),
        },
        title="Text bottleneck vs linear bottleneck of the same size (held-out sentences)",
    )

# %%

if MAIN:
    LEMMA = {form: pair[0] for pair in NOUNS for form in pair}
    
    for name, av in [("warm start", av_warm), ("1 word", nlas[1][0]), ("2 words", nlas[2][0]), ("4 words", av_rl)]:
        with t.no_grad():
            texts = decode_descriptions(sample_descriptions(av, h_test, greedy=True))
        pairs = []  # (true subject, claimed subject)
        for text, fact in zip(texts, facts_test):
            words = text.split()
            pairs += [(fact["subject"][0], w) for j, w in enumerate(words) if w in NOUN_WORDS and (j == 0 or words[j - 1] not in PREPOSITIONS)]
        number_right = np.mean([(true in NOUN_WORDS[1::2]) == (said in NOUN_WORDS[1::2]) for true, said in pairs])
        noun_right = np.mean([LEMMA[true] == LEMMA[said] for true, said in pairs])
        swaps = Counter((LEMMA[true], LEMMA[said]) for true, said in pairs if LEMMA[true] != LEMMA[said])
        print(f"{name:<10} mentions {dict(Counter(facts_mentioned(x) for x in texts).most_common(2))}")
        print(f"{'':<10} {len(pairs)} subject claims: number right {number_right:.1%}, noun right {noun_right:.1%}; "
              f"most common swaps (true → said): {[f'{a} → {b}' for (a, b), _ in swaps.most_common(4)]}")

# %%

def bag_of_words(sentences: list[str], vocab: list[str]) -> Float[Tensor, "n vocab"]:
    """Word counts (order discarded), over a fixed vocabulary."""
    index = {word: i for i, word in enumerate(vocab)}
    counts = t.zeros(len(sentences), len(vocab))
    for row, sentence in enumerate(sentences):
        for word in sentence.replace(",", "").split():
            if word in index:
                counts[row, index[word]] += 1
    return counts


def fit_ridge(X: Float[Tensor, "n f"], Y: Float[Tensor, "n d"], alpha: float = 1.0):
    """Closed-form ridge regression with an unpenalised intercept. Returns a prediction function."""
    X, Y = X.double(), Y.double()
    mu_X, mu_Y = X.mean(dim=0), Y.mean(dim=0)
    Xc, Yc = X - mu_X, Y - mu_Y
    W = t.linalg.solve(Xc.T @ Xc + alpha * t.eye(X.shape[1], dtype=X.dtype), Xc.T @ Yc)

    def predict(X_new: Tensor) -> Tensor:
        return ((X_new.double() - mu_X) @ W + mu_Y).float()

    return predict


if MAIN:
    tests.test_bag_of_words(bag_of_words)
    tests.test_fit_ridge(fit_ridge)

# %%

if MAIN:
    sentences = df.sentence.tolist()
    input_vocab = sorted({word for s in [sentences[i] for i in train_idx] for word in s.replace(",", "").split()})
    bow = bag_of_words(sentences, input_vocab)
    ordered = t.cat(
        [bag_of_words([s.replace(",", "").split()[p] for s in sentences], input_vocab) for p in range(7)], dim=1
    )
    baselines = {}
    for name, X in [("bag of words of the input", bow), ("ordered words of the input", ordered)]:
        predict = fit_ridge(X[train_idx], acts_unit[train_idx])
        h_hat = predict(X[test_idx])
        baselines[name] = dict(fve=fve(acts[test_idx], h_hat), transplant=transplant(h_hat), h_hat=h_hat)
        print(f"{name:<28}: {X.shape[1]:>3} features, FVE {baselines[name]['fve']:.3f}, transplant {baselines[name]['transplant']:.3f}")

# %%

if MAIN:
    same_number = t.tensor((df.number.values == df.object_number.values)[test_idx.numpy()])
    with t.no_grad():
        nla4_h_hat = ar_rl(text_features(model, sample_descriptions(av_rl, h_test, greedy=True))).cpu()
    split_results = {}
    for name, h_hat in [
        ("A's own activation", acts_unit[test_idx]),
        ("NLA, 4 words", nla4_h_hat),
        ("bag of words of the input", baselines["bag of words of the input"]["h_hat"]),
        ("ordered words of the input", baselines["ordered words of the input"]["h_hat"]),
    ]:
        split_results[name] = {
            f"subject & object same number": transplant_accuracy(model, tokens, labels, h_hat[same_number], test_idx[same_number], partners[same_number], act_scale),
            f"subject & object differ": transplant_accuracy(model, tokens, labels, h_hat[~same_number], test_idx[~same_number], partners[~same_number], act_scale),
        }
    split_df = pd.DataFrame(split_results)
    display(split_df.round(3))
    split_df.T.plot.bar(figsize=(9, 4), rot=0, color=["tab:blue", "tab:orange"])
    plt.axhline(0.25, color="k", ls=":", lw=1, label="chance")
    plt.ylabel("transplant accuracy")
    plt.ylim(0, 1.2)
    plt.title("Transplant accuracy, split by whether the source's subject and object agree in number", fontsize=10)
    plt.legend(fontsize=8, loc="upper center", ncol=3)
    plt.tight_layout()
    plt.show()

# %%

if MAIN:
    with t.no_grad():
        nla_ids = sample_descriptions(av_rl, h_test, greedy=True)
    nla_texts = decode_descriptions(nla_ids)
    shuffle_rng = random.Random(0)
    shuffled_texts = [shuffle_rng.sample(text.split(), len(text.split())) for text in nla_texts]
    with t.no_grad():
        shuffled_h_hat = ar_rl(text_features(model, encode_descriptions(shuffled_texts).to(device))).cpu()
        random_ids = t.where(nla_ids == EOS, nla_ids, t.randint_like(nla_ids, 0, EOS))
        random_h_hat = ar_rl(text_features(model, random_ids)).cpu()
    damaged = {
        "NLA, 4 words": nla4_h_hat,
        "same words, shuffled": shuffled_h_hat,
        "another sentence's description": nla4_h_hat[shuffled_sources - N_TRAIN],
        "random words": random_h_hat,
    }
    damaged_results = {name: dict(fve=fve(acts[test_idx], h_hat), transplant=transplant(h_hat)) for name, h_hat in damaged.items()}
    print("e.g.", [f"{a!r} -> {' '.join(b)!r}" for a, b in zip(nla_texts[:3], shuffled_texts[:3])])
    display(pd.DataFrame(damaged_results).T.round(3))

# %%

if MAIN:
    points = {
        "NLA, 4 words": (nla_results[4]["fve"], nla_results[4]["transplant"], "nla", (6, -10, "left")),
        "NLA, 2 words": (nla_results[2]["fve"], nla_results[2]["transplant"], "nla", (-8, -14, "right")),
        "NLA, 1 word": (nla_results[1]["fve"], nla_results[1]["transplant"], "nla", (6, -12, "left")),
        "NLA warm start (before RL)": (warm_eval["fve"], warm_eval["transplant"], "nla", (-6, -13, "right")),
        "oracle: number + tense": (*oracle_points["number + tense"], "oracle"),
        "oracle: subject": (*oracle_points["subject"], "oracle"),
        "oracle: subject + time word": (*oracle_points["subject + time word"], "oracle", (-6, 6, "right")),
        "bag of words of input": (baselines["bag of words of the input"]["fve"], baselines["bag of words of the input"]["transplant"], "baseline"),
        "ordered words of input": (baselines["ordered words of the input"]["fve"], baselines["ordered words of the input"]["transplant"], "baseline", (7, -3, "left")),
        "PCA, 4 dims": (pca_results[4]["fve"], pca_results[4]["transplant"], "baseline"),
        "PCA, 16 dims": (pca_results[16]["fve"], pca_results[16]["transplant"], "baseline", (0, 9, "center")),
        "NLA words shuffled": (damaged_results["same words, shuffled"]["fve"], damaged_results["same words, shuffled"]["transplant"], "control", (-6, 6, "right")),
        "another sentence's description": (damaged_results["another sentence's description"]["fve"], damaged_results["another sentence's description"]["transplant"], "control"),
        "random words": (damaged_results["random words"]["fve"], damaged_results["random words"]["transplant"], "control"),
        "mean activation": (0.0, transplant_controls["mean activation"], "control"),
    }
    utils.plot_fve_vs_behaviour(
        points,
        "Reconstruction quality (FVE) is not behaviour: GPT-2 Small, layer 10, 1,000 held-out sentences",
        ceiling=transplant_controls["A's own activation"],
    )

# %%

if MAIN:
    def edit_and_patch(description: str, base: int, reference_sentence: str | None = None) -> None:
        words = description.split()
        gpt2_tokens = [model.tokenizer.bos_token_id] + [model.to_single_token(" " + w) for w in words]
        gpt2_tokens += [model.to_single_token(".")] * max(1, MAX_WORDS + 1 - len(words))  # pad with "." like text_features
        with t.no_grad():
            h_hat = ar_rl(read_with_gpt2(model, t.tensor([gpt2_tokens]), LAYER)).cpu()
        base_tokens = tokens[base : base + 1]
        rows = {"reconstruction from your text": h_hat}
        if reference_sentence is not None:
            rows["real activation of the reference"] = get_final_activations(model, model.to_tokens(reference_sentence), [LAYER])[0]
        print(f"description: {description!r:<34} base sentence: {df.sentence[base]!r}")
        for name, vector in rows.items():
            logits = patch_final_position(model, base_tokens, F.normalize(vector, dim=-1) * act_scale, LAYER, verb_ids)
            probs = logits.softmax(-1)[0]
            cos = F.cosine_similarity(vector, acts[base : base + 1]).item()
            print(f"  {name:<34} cos to base activation {cos:+.2f} | " + "  ".join(f"{v.strip()}: {p:.2f}" for v, p in zip(VERBS, probs)))
    
    
    base = int(test_idx[0])
    print(f"The NLA's own description of the base sentence: {nla_texts[0]!r}\n")
    edit_and_patch("tables currently with keys", base)
    edit_and_patch("table currently with keys", base, reference_sentence="Currently, the table with the keys really")
    edit_and_patch("tables yesterday with keys", base, reference_sentence="Yesterday, the tables with the keys really")
    edit_and_patch("tables formerly with keys", base)
    edit_and_patch("singular past", base)

# %%

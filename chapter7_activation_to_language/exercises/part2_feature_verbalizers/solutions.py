# %%


import sys
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch as t
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Float, Int
from sae_lens import SAE
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter7_activation_to_language"
section = "part2_feature_verbalizers"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part2_feature_verbalizers.tests as tests
import part2_feature_verbalizers.utils as utils

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 250)

MAIN = __name__ == "__main__"

# %%

HOOK = "blocks.7.hook_resid_pre"  # the SAE's hook point (and where we steer, in section 5)
LAYER = 7

# %%

if MAIN:
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.requires_grad_(False)
    
    sae = SAE.from_pretrained("gpt2-small-res-jb", HOOK, device=str(device))
    sae.requires_grad_(False)
    
    print(f"SAE: {sae.cfg.d_in} -> {sae.cfg.d_sae} latents, hook = {sae.cfg.metadata.hook_name}")

# %%

if MAIN:
    tokens = utils.load_pile_tokens(model, n_sequences=4000, seq_len=64)
    explain_tokens, score_tokens = tokens[:2000], tokens[2000:]
    print(f"explain split: {tuple(explain_tokens.shape)}, score split: {tuple(score_tokens.shape)}")
    print(repr(model.to_string(explain_tokens[0, 1:30])))

# %%

def sae_latent_acts(resid: Float[Tensor, "... d_model"], sae: SAE, latents: list[int]) -> Float[Tensor, "... n_latents"]:
    """Activations of the chosen SAE latents: ReLU((resid - b_dec) @ W_enc[:, latents] + b_enc[latents])."""
    return F.relu((resid - sae.b_dec) @ sae.W_enc[:, latents] + sae.b_enc[latents])


if MAIN:
    tests.test_sae_latent_acts(sae_latent_acts, sae)

# %%

@t.inference_mode()
def collect_activations(
    model: HookedTransformer,
    tokens: Int[Tensor, "n_seq seq"],
    hook_name: str,
    act_fn,
    batch_size: int = 100,
) -> Float[Tensor, "n_seq seq n_units"]:
    """Runs the model on `tokens` in batches and returns act_fn(activation at hook_name), with position 0 zeroed."""
    layer = int(hook_name.split(".")[1])
    out = []
    for batch in tokens.split(batch_size):
        _, cache = model.run_with_cache(batch.to(model.cfg.device), names_filter=hook_name, stop_at_layer=layer + 1)
        acts = act_fn(cache[hook_name]).float()
        acts[:, 0] = 0.0
        out.append(acts.cpu())
    return t.cat(out)


if MAIN:
    tests.test_collect_activations(collect_activations, model)

# %%

if MAIN:
    with t.inference_mode():
        counts = t.zeros(sae.cfg.d_sae, device=device)
        for batch in tokens.split(100):
            _, cache = model.run_with_cache(batch.to(device), names_filter=HOOK, stop_at_layer=LAYER + 1)
            counts += (sae.encode(cache[HOOK])[:, 1:] > 0).float().sum((0, 1))
        density = (counts / (tokens.shape[0] * (tokens.shape[1] - 1))).cpu()
    
    candidates = ((density > 5e-4) & (density < 1e-2)).nonzero().squeeze(-1)
    LATENTS = candidates[t.randperm(len(candidates), generator=t.Generator().manual_seed(0))[:30]].tolist()
    N_LATENTS = len(LATENTS)
    
    print(f"{(density == 0).sum()} latents never fire, {len(candidates)} have density in [0.05%, 1%]")
    print(f"Our {N_LATENTS} latents: {LATENTS}")

# %%

if MAIN:
    latent_fn = partial(sae_latent_acts, sae=sae, latents=LATENTS)
    explain_acts = collect_activations(model, explain_tokens, HOOK, latent_fn)
    score_acts = collect_activations(model, score_tokens, HOOK, latent_fn)
    print(explain_acts.shape, score_acts.shape)

# %%

WINDOW_BEFORE, WINDOW_AFTER = 15, 4


def window_bounds(pos: int, seq_len: int) -> tuple[int, int]:
    """The [start, end) token range of the example window centred on `pos` (never including BOS)."""
    return max(1, pos - WINDOW_BEFORE), min(seq_len, pos + WINDOW_AFTER + 1)


def get_window(
    model: HookedTransformer, tokens: Int[Tensor, "n_seq seq"], acts: Float[Tensor, "n_seq seq"], seq: int, pos: int
) -> tuple[list[str], Float[Tensor, " n_tok"]]:
    """Returns the string tokens and the activations of the example window centred on (seq, pos)."""
    start, end = window_bounds(pos, tokens.shape[1])
    return [model.to_string(tok) for tok in tokens[seq, start:end]], acts[seq, start:end]

# %%

def top_activating_examples(acts: Float[Tensor, "n_seq seq"], k: int) -> list[tuple[int, int]]:
    """The (seq, pos) of the max activation in each of the k sequences with the largest max activation (> 0)."""
    max_acts, max_pos = acts.max(dim=-1)
    order = t.argsort(max_acts, descending=True, stable=True)[:k]
    return [(int(seq), int(max_pos[seq])) for seq in order if max_acts[seq] > 0]


if MAIN:
    tests.test_top_activating_examples(top_activating_examples)

# %%

def random_examples(acts: Float[Tensor, "n_seq seq"], k: int, generator: t.Generator) -> list[tuple[int, int]]:
    """k random (seq, pos) pairs, with positions chosen so that the example window is never clipped."""
    n_seq, seq_len = acts.shape
    seqs = t.randint(0, n_seq, (k,), generator=generator)
    positions = t.randint(WINDOW_BEFORE + 1, seq_len - WINDOW_AFTER, (k,), generator=generator)
    return list(zip(seqs.tolist(), positions.tolist()))


def quantile_examples(
    acts: Float[Tensor, "n_seq seq"], n_bins: int, k_per_bin: int, generator: t.Generator
) -> list[list[tuple[int, int]]]:
    """
    Splits the activating sequences (sorted by max activation, highest first) into n_bins equal bins, and returns
    k_per_bin random examples (seq, max position) from each bin.
    """
    max_acts, max_pos = acts.max(dim=-1)
    active = (max_acts > 0).nonzero().squeeze(-1)
    active = active[t.argsort(max_acts[active], descending=True, stable=True)]
    bins = []
    for bin_seqs in active.tensor_split(n_bins):
        chosen = bin_seqs[t.randperm(len(bin_seqs), generator=generator)[:k_per_bin]]
        bins.append([(int(seq), int(max_pos[seq])) for seq in chosen])
    return bins


if MAIN:
    tests.test_random_examples(random_examples, WINDOW_BEFORE, WINDOW_AFTER)
    tests.test_quantile_examples(quantile_examples)

# %%

def format_example(str_toks: list[str], acts: Float[Tensor, " n_tok"], threshold: float) -> str:
    """Joins the tokens, wrapping each run of tokens with activation > threshold in << >>, and replacing \\n by ↵."""
    text = "".join(f"<<{tok}>>" if act > threshold else tok for tok, act in zip(str_toks, acts.tolist()))
    return text.replace(">><<", "").replace("\n", "↵")


if MAIN:
    tests.test_format_example(format_example)

# %%

if MAIN:
    def show_examples(acts: Float[Tensor, "n_seq seq"], tokens: Int[Tensor, "n_seq seq"], examples: list[tuple[int, int]], threshold: float):
        for seq, pos in examples:
            str_toks, window_acts = get_window(model, tokens, acts, seq, pos)
            print(f"  {window_acts.max().item():5.1f} | {format_example(str_toks, window_acts, threshold)}")
    
    
    FOCUS = LATENTS.index(11803)
    focus_acts = explain_acts[..., FOCUS]
    focus_threshold = 0.25 * focus_acts.max().item()
    gen = t.Generator().manual_seed(0)
    focus_top = top_activating_examples(focus_acts, 10)
    focus_quantile = quantile_examples(focus_acts, n_bins=5, k_per_bin=5, generator=gen)
    focus_random = random_examples(focus_acts, 5, gen)
    
    n_active = (focus_acts.max(dim=-1).values > 0).sum().item()
    print(f"Latent 11803 fires on {(focus_acts[:, 1:] > 0).sum()} tokens, in {n_active} of {len(focus_acts)} explain-split sequences\n")
    print("Top 10 examples (max activation in window | excerpt):")
    show_examples(focus_acts, explain_tokens, focus_top, focus_threshold)
    print("\n5 examples from the 3rd of 5 quantile bins:")
    show_examples(focus_acts, explain_tokens, focus_quantile[2], focus_threshold)
    print("\n5 random examples:")
    show_examples(focus_acts, explain_tokens, focus_random, focus_threshold)

# %%

if MAIN:
    window_max = lambda examples: [get_window(model, explain_tokens, focus_acts, s, p)[1].max().item() for s, p in examples]
    utils.plot_activation_distribution(
        focus_acts,
        top_values=window_max(focus_top),
        quantile_values=window_max([ex for b in quantile_examples(focus_acts, 5, 4, gen) for ex in b]),
        random_values=[v for v in window_max(random_examples(focus_acts, 10, gen)) if v > 0],
        title="Latent 11803 on the explain split: where do our examples come from?",
    )

# %%

if MAIN:
    EXPLAINER_NAME = "Qwen/Qwen3-1.7B"
    explainer_tok, explainer = utils.load_explainer(EXPLAINER_NAME, device)
    print(f"Loaded {EXPLAINER_NAME}: {sum(p.numel() for p in explainer.parameters()) / 1e9:.2f}B parameters")

# %%

EXPLAINER_SYSTEM = (
    "You are an interpretability researcher studying features in a language model. You will see text excerpts where "
    "a feature is active. The tokens where the feature is active are marked with << >>. Your job is to explain what "
    "the feature responds to, in one short phrase.\n\n"
    "Focus on the MARKED tokens, not on the general topic of the excerpts. Ask yourself: what do the marked tokens "
    "have in common? Often a feature fires on one particular word (e.g. 'the word \"however\"'), on a kind of word "
    "(e.g. 'names of cities'), or on a word in a particular context (e.g. '\"bank\" meaning the side of a river'). "
    "Answer with the phrase only, at most 15 words."
)
FEWSHOT_EXAMPLES = [
    "I went to the<< store>> yesterday and",
    "the nearest<< supermarket>> is closed",
    "buy it at the local<< shop>>, she said",
]
FEWSHOT_MARKED = ["store", "supermarket", "shop"]
FEWSHOT_EXPLANATION = 'nouns for places where you buy things, such as "store" and "shop"'

# %%

def build_explainer_messages(examples: list[str], marked_tokens: list[str]) -> list[dict[str, str]]:
    """The explainer conversation: system prompt, one few-shot example, then the real excerpts."""
    def query(examples: list[str], marked_tokens: list[str]) -> str:
        lines = [f"{i + 1}. {example}" for i, example in enumerate(examples)]
        return "Excerpts:\n" + "\n".join(lines) + "\nMarked tokens: " + ", ".join(marked_tokens)

    return [
        {"role": "system", "content": EXPLAINER_SYSTEM},
        {"role": "user", "content": query(FEWSHOT_EXAMPLES, FEWSHOT_MARKED)},
        {"role": "assistant", "content": FEWSHOT_EXPLANATION},
        {"role": "user", "content": query(examples, marked_tokens)},
    ]


if MAIN:
    tests.test_build_explainer_messages(build_explainer_messages)

# %%

def clean_explanation(reply: str) -> str:
    """Keeps the first line of the reply, without surrounding quotes or a final full stop."""
    line = reply.strip().split("\n")[0].strip()
    if len(line) > 1 and line[0] == line[-1] == '"':
        line = line[1:-1].strip()
    return line.rstrip(".").strip()


def explain_units(
    model: HookedTransformer,
    tokens: Int[Tensor, "n_seq seq"],
    acts: Float[Tensor, "n_seq seq n_units"],
    tokenizer,
    llm,
    k: int = 10,
    threshold_frac: float = 0.25,
) -> list[str]:
    """One explanation per unit, written by `llm` from the unit's top-k examples in `tokens`."""
    conversations = []
    for j in range(acts.shape[-1]):
        unit_acts = acts[..., j]
        threshold = threshold_frac * unit_acts.max().item()
        examples, marked = [], []
        for seq, pos in top_activating_examples(unit_acts, k):
            str_toks, window_acts = get_window(model, tokens, unit_acts, seq, pos)
            examples.append(format_example(str_toks, window_acts, threshold))
            marked.append(model.to_string(tokens[seq, pos]).replace("\n", "↵").strip())
        conversations.append(build_explainer_messages(examples, marked))
    replies = utils.generate_chat(tokenizer, llm, conversations, max_new_tokens=40)
    return [clean_explanation(reply) for reply in replies]

# %%

if MAIN:
    explanations = explain_units(model, explain_tokens, explain_acts, explainer_tok, explainer)
    
    top_example_text = []
    for j in range(N_LATENTS):
        seq, pos = top_activating_examples(explain_acts[..., j], 1)[0]
        str_toks, window_acts = get_window(model, explain_tokens, explain_acts[..., j], seq, pos)
        top_example_text.append(format_example(str_toks, window_acts, 0.25 * explain_acts[..., j].max().item()))
    
    for latent, explanation, example in zip(LATENTS, explanations, top_example_text):
        print(f"{latent:>5}  {explanation}")
        print(f"{'':>5}    strongest example: {example}")

# %%

if MAIN:
    NAIVE_SYSTEM = (
        "We're studying features in a neural network. Each feature activates on particular words or concepts in short "
        "text excerpts. In each excerpt below, the tokens where the feature activates are marked with << >>. Summarize in "
        "one short phrase what the feature activates on. Answer with the phrase only, at most 15 words."
    )
    naive_ids = [LATENTS.index(latent) for latent in [16436, 8232, 13638, 11803, 4412, 10483]]
    naive_conversations = []
    for j in naive_ids:
        unit_acts = explain_acts[..., j]
        examples = [
            format_example(*get_window(model, explain_tokens, unit_acts, s, p), 0.25 * unit_acts.max().item())
            for s, p in top_activating_examples(unit_acts, 10)
        ]
        excerpts = "Excerpts:\n" + "\n".join(f"{i + 1}. {ex}" for i, ex in enumerate(examples))
        naive_conversations.append([{"role": "system", "content": NAIVE_SYSTEM}, {"role": "user", "content": excerpts}])
    naive_explanations = [clean_explanation(r) for r in utils.generate_chat(explainer_tok, explainer, naive_conversations)]
    
    for j, naive in zip(naive_ids, naive_explanations):
        print(f"latent {LATENTS[j]}\n  naive prompt: {naive}\n  our prompt:   {explanations[j]}")

# %%

def shuffle_activations(acts: Float[Tensor, "n_seq seq n_units"], generator: t.Generator) -> Float[Tensor, "n_seq seq n_units"]:
    """Independently permutes each unit's activations across all non-BOS token positions."""
    n_seq, seq_len, n_units = acts.shape
    flat = acts[:, 1:].reshape(-1, n_units)
    shuffled = t.stack([flat[t.randperm(len(flat), generator=generator), j] for j in range(n_units)], dim=-1)
    out = t.zeros_like(acts)
    out[:, 1:] = shuffled.reshape(n_seq, seq_len - 1, n_units)
    return out


if MAIN:
    tests.test_shuffle_activations(shuffle_activations)

# %%

if MAIN:
    noise_explain_acts = shuffle_activations(explain_acts, t.Generator().manual_seed(1))
    noise_score_acts = shuffle_activations(score_acts, t.Generator().manual_seed(2))
    noise_explanations = explain_units(model, explain_tokens, noise_explain_acts, explainer_tok, explainer)
    
    print("Top 3 examples of the shuffled version of latent 11803:")
    show_examples(noise_explain_acts[..., FOCUS], explain_tokens, top_activating_examples(noise_explain_acts[..., FOCUS], 3),
                  0.25 * noise_explain_acts[..., FOCUS].max().item())
    print("\nExplanations of noise features:")
    for j in list(range(6)) + [FOCUS]:
        print(f"  shuffled latent {LATENTS[j]}: {noise_explanations[j]}")

# %%

SIM_SYSTEM = (
    "We're studying a feature in a language model. You will be given an explanation of what the feature responds to, "
    "and a text excerpt. For each token of the excerpt, predict how strongly the feature activates on it, from 0 (not "
    "active) to 9 (maximally active). Most tokens have activation 0."
)
SIM_FEWSHOT_TOKENS = ["I", " went", " to", " the", " store", " and", " then", " the", " bakery", "."]
SIM_FEWSHOT_ACTS = [0, 0, 0, 0, 9, 0, 0, 0, 6, 0]


def clean_token(text: str) -> str:
    """Newlines would break the one-token-per-line format, and tabs would break the token/activation separator."""
    return text.replace("\n", "↵").replace("\t", " ")

# %%

def build_simulation_input(tokenizer, explanation: str, str_toks: list[str]) -> tuple[list[int], list[int]]:
    """Returns (input_ids, slots): the simulator prompt for this excerpt, and the position of the tab after each token."""
    messages = [
        {"role": "system", "content": SIM_SYSTEM},
        {"role": "user", "content": f"Explanation: {FEWSHOT_EXPLANATION}\nExcerpt: {''.join(SIM_FEWSHOT_TOKENS)}"},
        {
            "role": "assistant",
            "content": "".join(f"{clean_token(tok)}\t{act}\n" for tok, act in zip(SIM_FEWSHOT_TOKENS, SIM_FEWSHOT_ACTS)),
        },
        {"role": "user", "content": f"Explanation: {explanation}\nExcerpt: {clean_token(''.join(str_toks))}"},
    ]
    ids = tokenizer(utils.chat_prompt(tokenizer, messages), add_special_tokens=False).input_ids
    slots = []
    for tok in str_toks:
        ids = ids + tokenizer(f"{clean_token(tok)}\t", add_special_tokens=False).input_ids
        slots.append(len(ids) - 1)
        ids = ids + tokenizer("unknown\n", add_special_tokens=False).input_ids
    return ids, slots


if MAIN:
    tests.test_build_simulation_input(build_simulation_input, explainer_tok)

# %%

if MAIN:
    example_ids, example_slots = build_simulation_input(explainer_tok, explanations[FOCUS], [" the", " capital", " of", " Bihar", "."])
    print(explainer_tok.decode(example_ids)[-420:])
    print(f"slots: {example_slots}, tokens at slots: {[explainer_tok.decode(example_ids[s]) for s in example_slots]}")

# %%

def simulate_activations(tokenizer, llm, requests: list[tuple[str, list[str]]], batch_size: int = 32) -> list[Float[Tensor, " n_tok"]]:
    """For each (explanation, str_toks) request, the simulator's expected activation (0-9) for every token."""
    inputs = [build_simulation_input(tokenizer, explanation, str_toks) for explanation, str_toks in requests]
    digit_ids = [tokenizer.convert_tokens_to_ids(str(d)) for d in range(10)]
    logits = utils.logits_at_positions(
        tokenizer, llm, [ids for ids, _ in inputs], [slots for _, slots in inputs], digit_ids, batch_size
    )
    return [slot_logits.softmax(dim=-1) @ t.arange(10.0) for slot_logits in logits]


if MAIN:
    tests.test_simulate_activations(simulate_activations, build_simulation_input, explainer_tok, explainer)

# %%

def correlation_score(pred: Float[Tensor, " n"], true: Float[Tensor, " n"]) -> float:
    """Pearson correlation of pred and true, or 0.0 if either is constant."""
    pred = pred.double() - pred.double().mean()
    true = true.double() - true.double().mean()
    denom = pred.norm() * true.norm()
    if denom < 1e-8:
        return 0.0
    return (pred @ true / denom).item()


if MAIN:
    tests.test_correlation_score(correlation_score)

# %%

def scoring_examples(acts: Float[Tensor, "n_seq seq"], generator: t.Generator, n_top: int = 5, n_random: int = 5) -> list[tuple[int, int]]:
    """The "top-and-random" scoring set: the n_top top-activating examples plus n_random random examples."""
    return top_activating_examples(acts, n_top) + random_examples(acts, n_random, generator)


def simulation_scores(
    model: HookedTransformer,
    tokens: Int[Tensor, "n_seq seq"],
    units: list[Float[Tensor, "n_seq seq"]],
    explanations: list[str],
    examples: list[list[tuple[int, int]]],
    tokenizer,
    llm,
) -> tuple[Float[Tensor, " n_pairs"], list[Tensor], list[Tensor]]:
    """
    Simulation score of explanations[k] for the unit with activations units[k], on the examples examples[k].
    Returns (scores, predictions, true activations), where the last two are lists of concatenated windows.
    """
    requests, trues, owner = [], [], []
    for k, (acts, explanation, exs) in enumerate(zip(units, explanations, examples)):
        for seq, pos in exs:
            str_toks, true = get_window(model, tokens, acts, seq, pos)
            requests.append((explanation, str_toks))
            trues.append(true)
            owner.append(k)
    preds = simulate_activations(tokenizer, llm, requests)
    pred_cat = [t.cat([p for p, o in zip(preds, owner) if o == k]) for k in range(len(units))]
    true_cat = [t.cat([a for a, o in zip(trues, owner) if o == k]) for k in range(len(units))]
    scores = t.tensor([correlation_score(p, a) for p, a in zip(pred_cat, true_cat)])
    return scores, pred_cat, true_cat


if MAIN:
    tests.test_simulation_scores(simulation_scores, get_window, correlation_score, model, explainer_tok, explainer)

# %%

if MAIN:
    score_units = [score_acts[..., j] for j in range(N_LATENTS)]
    SCORE_EXAMPLES = [scoring_examples(score_units[j], t.Generator().manual_seed(j)) for j in range(N_LATENTS)]
    other_explanations = explanations[1:] + explanations[:1]
    
    own_scores, own_preds, own_trues = simulation_scores(model, score_tokens, score_units, explanations, SCORE_EXAMPLES, explainer_tok, explainer)
    other_scores, _, _ = simulation_scores(model, score_tokens, score_units, other_explanations, SCORE_EXAMPLES, explainer_tok, explainer)
    noise_expl_scores, _, _ = simulation_scores(model, score_tokens, score_units, noise_explanations, SCORE_EXAMPLES, explainer_tok, explainer)
    
    print(f"Mean simulation score over {N_LATENTS} latents:")
    print(f"  own explanation:              {own_scores.mean():.3f}")
    print(f"  another latent's explanation: {other_scores.mean():.3f}")
    print(f"  explanation of noise:         {noise_expl_scores.mean():.3f}")
    print(f"Own explanation beats both controls for {((own_scores > other_scores) & (own_scores > noise_expl_scores)).sum()} / {N_LATENTS} latents")

# %%

if MAIN:
    def simulation_panels(j: int, n_windows: int = 2) -> list[tuple[list[str], Tensor, Tensor]]:
        """The first n_windows scoring windows for latent index j: (tokens, true activations, simulated activations)."""
        panels, offset = [], 0
        for seq, pos in SCORE_EXAMPLES[j][:n_windows]:
            str_toks, true = get_window(model, score_tokens, score_units[j], seq, pos)
            panels.append((str_toks, true, own_preds[j][offset : offset + len(str_toks)]))
            offset += len(str_toks)
        return panels
    
    
    SIM_EXAMPLES = [LATENTS.index(11803), LATENTS.index(4412)]
    panels, titles = [], []
    for j in SIM_EXAMPLES:
        for k, panel in enumerate(simulation_panels(j)):
            panels.append(panel)
            titles.append(f"latent {LATENTS[j]}, top-activating scoring excerpt {k + 1}.  Explanation: {explanations[j]!r}  (score on all 10 excerpts: {own_scores[j]:.2f})")
    utils.plot_simulation(panels, titles, "True vs simulated activations")

# %%

def permutation_null(pred: Float[Tensor, " n"], true: Float[Tensor, " n"], n_perm: int, generator: t.Generator) -> Float[Tensor, " n_perm"]:
    """Correlations between pred and n_perm random permutations of true."""
    perms = t.stack([t.randperm(len(true), generator=generator) for _ in range(n_perm)])
    shuffled = true.double()[perms]
    shuffled = shuffled - shuffled.mean(dim=-1, keepdim=True)
    centred_pred = pred.double() - pred.double().mean()
    denom = centred_pred.norm() * shuffled.norm(dim=-1)
    corr = (shuffled @ centred_pred) / denom.clamp(min=1e-12)
    return t.where(denom > 1e-8, corr, t.zeros_like(corr)).float()


if MAIN:
    tests.test_permutation_null(permutation_null, correlation_score)

# %%

if MAIN:
    null_gen = t.Generator().manual_seed(0)
    null95 = t.tensor([permutation_null(p, a, 1000, null_gen).quantile(0.95).item() for p, a in zip(own_preds, own_trues)])
    significant = own_scores > null95
    print(f"{significant.sum()} / {N_LATENTS} explanations beat the 95th percentile of their permutation null")
    print(f"Median 95th percentile of the null: {null95.median():.3f}")
    
    utils.plot_score_dots(
        [str(latent) for latent in LATENTS],
        {
            "own explanation": own_scores.tolist(),
            "another latent's explanation": other_scores.tolist(),
            "explanation of shuffled (noise) feature": noise_expl_scores.tolist(),
        },
        null95.tolist(),
        title=f"Simulation scores of Qwen3-1.7B explanations for {N_LATENTS} random GPT-2 SAE latents (held-out excerpts)",
        ylabel="simulation score (correlation)",
        xlabel="SAE latent (sorted by the score of its own explanation)",
    )

# %%

if MAIN:
    HUMAN_EXPLANATIONS = {
        16436: 'the word "of" right after a title noun such as "Book", "Handbook", "Board" or "Chart" (as in "Book of")',
        10483: 'the word "first" after "the", as in "for the first time"',
        4412: 'the word "to" directly after "similar"',
        18120: 'the word "lot" in "a lot of", or "deal" in "a great deal of"',
        11803: "names of Indian states such as Kerala, Karnataka, Bihar and Gujarat",
        23294: 'the word "of" right after "round" (as in "second round of voting")',
        613: 'the word "one" right after "in" (as in "all in one" or "in one day")',
        1040: 'the word "from" after "data" or "taken" (as in "data from the study")',
        9212: 'the word "argument" or "arguments", especially in programming questions',
        22850: "brand or product names in a list, such as Honda, Sony, Xbox or iPhone",
        13945: 'the word "is" right after "that" in a relative clause (as in "a tool that is")',
        18988: 'the abbreviation "prob" (short for probability)',
    }
    HUMAN_IDS = [LATENTS.index(latent) for latent in HUMAN_EXPLANATIONS]
    
    human_scores, _, _ = simulation_scores(
        model, score_tokens, [score_units[j] for j in HUMAN_IDS], list(HUMAN_EXPLANATIONS.values()),
        [SCORE_EXAMPLES[j] for j in HUMAN_IDS], explainer_tok, explainer,
    )
    for j, (latent, human) in zip(HUMAN_IDS, HUMAN_EXPLANATIONS.items()):
        print(f"{latent:>5}  LLM   {own_scores[j]:5.2f}  {explanations[j]}")
        print(f"{'':>5}  human {human_scores[HUMAN_IDS.index(j)]:5.2f}  {human}")
    print(f"\nMean over these 12 latents: LLM {own_scores[HUMAN_IDS].mean():.3f}, human {human_scores.mean():.3f}")
    print(f"Human explanation scores higher for {(human_scores > own_scores[HUMAN_IDS]).sum()} / 12 latents")

# %%

if MAIN:
    pairs = [(i, j) for i in HUMAN_IDS for j in HUMAN_IDS]
    swap_scores, _, _ = simulation_scores(
        model, score_tokens, [score_units[j] for _, j in pairs], [explanations[i] for i, _ in pairs],
        [SCORE_EXAMPLES[j] for _, j in pairs], explainer_tok, explainer,
    )
    swap_matrix = swap_scores.reshape(len(HUMAN_IDS), len(HUMAN_IDS))
    diag = swap_matrix.diag()
    off_diag = swap_matrix[~t.eye(len(HUMAN_IDS), dtype=t.bool)]
    print(f"diagonal mean {diag.mean():.3f}, off-diagonal mean {off_diag.mean():.3f}")
    print(f"the diagonal entry is the best in its column for {(swap_matrix.argmax(0) == t.arange(len(HUMAN_IDS))).sum()} / {len(HUMAN_IDS)} latents")
    
    labels = [f"{LATENTS[j]}: {explanations[j][:38]}" for j in HUMAN_IDS]
    utils.plot_matrix(
        swap_matrix, row_labels=labels, col_labels=[str(LATENTS[j]) for j in HUMAN_IDS],
        title="Simulation score of each latent's explanation (rows) on each latent's excerpts (columns)",
        xlabel="latent being simulated", ylabel="explanation used", cbar_label="simulation score",
        vmin=-0.1, vmax=0.7, figsize=(10, 7),
    )

# %%

if MAIN:
    NEURONS = t.randperm(model.cfg.d_mlp, generator=t.Generator().manual_seed(1))[:30].tolist()
    neuron_fn = lambda x: x[..., NEURONS]
    neuron_explain_acts = collect_activations(model, explain_tokens, "blocks.7.mlp.hook_post", neuron_fn)
    neuron_score_acts = collect_activations(model, score_tokens, "blocks.7.mlp.hook_post", neuron_fn)
    
    mean_resid = collect_activations(model, explain_tokens[:200], HOOK, lambda x: x)[:, 1:].reshape(-1, model.cfg.d_model).mean(0)
    RANDOM_DIRS = t.randn(model.cfg.d_model, 30, generator=t.Generator().manual_seed(2))
    RANDOM_DIRS = RANDOM_DIRS / RANDOM_DIRS.norm(dim=0)
    direction_fn = lambda x: F.relu((x - mean_resid.to(x.device)) @ RANDOM_DIRS.to(x.device))
    direction_explain_acts = collect_activations(model, explain_tokens, HOOK, direction_fn)
    direction_score_acts = collect_activations(model, score_tokens, HOOK, direction_fn)
    
    unit_type_scores = {"SAE latents": own_scores.tolist()}
    for name, (e_acts, s_acts, e_expl) in {
        "MLP neurons": (neuron_explain_acts, neuron_score_acts, None),
        "random directions": (direction_explain_acts, direction_score_acts, None),
        "noise features (shuffled latents)": (noise_explain_acts, noise_score_acts, noise_explanations),
    }.items():
        unit_expl = e_expl if e_expl is not None else explain_units(model, explain_tokens, e_acts, explainer_tok, explainer)
        units = [s_acts[..., j] for j in range(s_acts.shape[-1])]
        examples = [scoring_examples(units[j], t.Generator().manual_seed(j)) for j in range(len(units))]
        scores, _, _ = simulation_scores(model, score_tokens, units, unit_expl, examples, explainer_tok, explainer)
        unit_type_scores[name] = scores.tolist()
        print(f"{name:<34} mean score {scores.mean():.3f}   e.g. {unit_expl[0]!r}")
    
    utils.plot_unit_types(
        unit_type_scores, title="The same autointerp pipeline applied to different kinds of unit (30 each)",
        ylabel="simulation score (correlation)",
    )

# %%

DETECTION_TEMPLATE = 'Text: "{text}"\n\nDoes this text contain {explanation}? Answer Yes or No.'


def detection_scores(
    tokenizer, llm, requests: list[tuple[str, str]], template: str = DETECTION_TEMPLATE, batch_size: int = 64
) -> Float[Tensor, " n"]:
    """logit("Yes") - logit("No") for each (explanation, text) request, asked using `template`."""
    inputs = []
    for explanation, text in requests:
        messages = [{"role": "user", "content": template.format(text=clean_token(text), explanation=explanation)}]
        inputs.append(tokenizer(utils.chat_prompt(tokenizer, messages), add_special_tokens=False).input_ids)
    yes_no = [tokenizer.convert_tokens_to_ids("Yes"), tokenizer.convert_tokens_to_ids("No")]
    logits = utils.logits_at_positions(tokenizer, llm, inputs, [[len(ids) - 1] for ids in inputs], yes_no, batch_size)
    return t.stack([lg[0, 0] - lg[0, 1] for lg in logits])


if MAIN:
    tests.test_detection_scores(detection_scores, explainer_tok, explainer)

# %%

def auroc(pos_scores: Float[Tensor, " n_pos"], neg_scores: Float[Tensor, " n_neg"]) -> float:
    """P(random positive scores higher than random negative), with ties counting 1/2."""
    diff = pos_scores[:, None].double() - neg_scores[None, :].double()
    return ((diff > 0).double() + 0.5 * (diff == 0).double()).mean().item()


if MAIN:
    tests.test_auroc(auroc)

# %%

def detection_examples(
    acts: Float[Tensor, "n_seq seq"], generator: t.Generator, n_bins: int = 5, k_per_bin: int = 4, n_negative: int = 20
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]]:
    """(quantile bins of activating examples, random examples from sequences where the unit never fires)."""
    bins = quantile_examples(acts, n_bins, k_per_bin, generator)
    silent = (acts.max(dim=-1).values == 0).nonzero().squeeze(-1)
    chosen = silent[t.randperm(len(silent), generator=generator)[:n_negative]]
    positions = t.randint(WINDOW_BEFORE + 1, acts.shape[1] - WINDOW_AFTER, (len(chosen),), generator=generator)
    return bins, list(zip(chosen.tolist(), positions.tolist()))


def detection_auroc(
    model: HookedTransformer,
    tokens: Int[Tensor, "n_seq seq"],
    explanations: list[str],
    example_sets: list[tuple[list[list[tuple[int, int]]], list[tuple[int, int]]]],
    tokenizer,
    llm,
) -> Float[Tensor, "n_pairs n_bins"]:
    """AUROC of each positive bin vs the negatives, for each (explanation, example set) pair."""
    requests, owners = [], []
    for k, (explanation, (bins, negatives)) in enumerate(zip(explanations, example_sets)):
        for b, examples in enumerate(bins + [negatives]):
            for seq, pos in examples:
                start, end = window_bounds(pos, tokens.shape[1])
                requests.append((explanation, model.to_string(tokens[seq, start:end])))
                owners.append((k, b))
    scores = detection_scores(tokenizer, llm, requests)
    n_bins = len(example_sets[0][0])
    result = t.zeros(len(explanations), n_bins)
    for k in range(len(explanations)):
        negative = scores[[i for i, o in enumerate(owners) if o == (k, n_bins)]]
        for b in range(n_bins):
            result[k, b] = auroc(scores[[i for i, o in enumerate(owners) if o == (k, b)]], negative)
    return result


if MAIN:
    tests.test_detection_auroc(detection_auroc, detection_scores, auroc, window_bounds, model, explainer_tok, explainer)

# %%

if MAIN:
    top_sets = []
    for j in HUMAN_IDS:
        _, negatives = detection_examples(score_units[j], t.Generator().manual_seed(j))
        top_sets.append(([top_activating_examples(score_units[j], 10)], negatives))
    
    llm_detection = detection_auroc(model, score_tokens, [explanations[j] for j in HUMAN_IDS], top_sets, explainer_tok, explainer)[:, 0]
    human_detection = detection_auroc(model, score_tokens, list(HUMAN_EXPLANATIONS.values()), top_sets, explainer_tok, explainer)[:, 0]
    other_detection = detection_auroc(model, score_tokens, [other_explanations[j] for j in HUMAN_IDS], top_sets, explainer_tok, explainer)[:, 0]
    
    print(f"{'':<34}{'detection AUROC':>18}{'simulation score':>18}")
    print(f"{'LLM explanations':<34}{llm_detection.mean():>18.3f}{own_scores[HUMAN_IDS].mean():>18.3f}")
    print(f"{'human explanations':<34}{human_detection.mean():>18.3f}{human_scores.mean():>18.3f}")
    print(f"{'another latent' + chr(39) + 's explanation':<34}{other_detection.mean():>18.3f}{other_scores[HUMAN_IDS].mean():>18.3f}")
    
    sem = lambda x: (x.std() / len(x) ** 0.5).item()
    utils.plot_grouped_bars(
        ["detection AUROC\n(top-10 excerpts vs 20 silent ones)", "simulation score\n(correlation on 5 top + 5 random)"],
        {
            "LLM explanation": [llm_detection.mean().item(), own_scores[HUMAN_IDS].mean().item()],
            "human explanation": [human_detection.mean().item(), human_scores.mean().item()],
            "another latent's LLM explanation": [other_detection.mean().item(), other_scores[HUMAN_IDS].mean().item()],
        },
        errors={
            "LLM explanation": [sem(llm_detection), sem(own_scores[HUMAN_IDS])],
            "human explanation": [sem(human_detection), sem(human_scores)],
            "another latent's LLM explanation": [sem(other_detection), sem(other_scores[HUMAN_IDS])],
        },
        title="Two scores for the same 12 latents (mean ± s.e.m.)",
        ylabel="score", ylim=(-0.1, 1.1), hline=0.5, hline_label="chance level for AUROC",
    )

# %%

if MAIN:
    quantile_sets = [detection_examples(score_units[j], t.Generator().manual_seed(100 + j)) for j in range(N_LATENTS)]
    own_by_bin = detection_auroc(model, score_tokens, explanations, quantile_sets, explainer_tok, explainer)
    other_by_bin = detection_auroc(model, score_tokens, other_explanations, quantile_sets, explainer_tok, explainer)
    
    print("mean AUROC by quantile bin (bin 1 = strongest activations):")
    print("  own explanation:    ", own_by_bin.mean(0).numpy().round(3))
    print("  other explanation:  ", other_by_bin.mean(0).numpy().round(3))
    
    utils.plot_lines(
        list(range(1, 6)),
        {"own explanation": own_by_bin.mean(0).tolist(), "another latent's explanation": other_by_bin.mean(0).tolist()},
        errors={"own explanation": (own_by_bin.std(0) / N_LATENTS**0.5).tolist(),
                "another latent's explanation": (other_by_bin.std(0) / N_LATENTS**0.5).tolist()},
        title=f"Detection AUROC by activation quantile ({N_LATENTS} latents, mean ± s.e.m.)",
        xlabel="quantile bin of the activating excerpts", ylabel="AUROC vs silent excerpts",
        xticklabels=["1 (top 20%)", "2", "3", "4", "5 (weakest 20%)"], hline=0.5, hline_label="chance",
    )

# %%

if MAIN:
    neuron_j = NEURONS.index(2307)
    neuron_acts = neuron_explain_acts[..., neuron_j]
    neuron_explanation = explain_units(model, explain_tokens, neuron_explain_acts[..., [neuron_j]], explainer_tok, explainer)[0]
    print(f"Neuron 2307. Explanation: {neuron_explanation!r}\n")
    max_acts, max_pos = neuron_acts.max(dim=-1)
    order = max_acts.argsort(descending=True)
    for name, ranks in [("top", range(0, 4)), ("~1% quantile", range(20, 24)), ("~10% quantile", range(200, 204))]:
        print(f"{name}:")
        show_examples(neuron_acts, explain_tokens, [(int(order[r]), int(max_pos[order[r]])) for r in ranks], 0.5 * neuron_acts.max().item())

# %%

FUZZ_TEMPLATE = (
    'Text: "{text}"\n\nThe tokens marked with << >> are supposed to be examples of {explanation}. '
    "Is that correct? Answer Yes or No."
)


def fuzzing_auroc(
    model: HookedTransformer,
    tokens: Int[Tensor, "n_seq seq"],
    units: list[Float[Tensor, "n_seq seq"]],
    explanations: list[str],
    tokenizer,
    llm,
    generator: t.Generator,
    k: int = 10,
    threshold_frac: float = 0.25,
) -> Float[Tensor, " n_pairs"]:
    """AUROC of correctly vs randomly highlighted top-k excerpts, for each (unit, explanation) pair."""
    requests = []
    for acts, explanation in zip(units, explanations):
        threshold = threshold_frac * acts.max().item()
        for correct in [True, False]:
            for seq, pos in top_activating_examples(acts, k):
                str_toks, window_acts = get_window(model, tokens, acts, seq, pos)
                if correct:
                    text = format_example(str_toks, window_acts, threshold)
                else:
                    active = window_acts > threshold
                    candidates = (~active).nonzero().squeeze(-1)
                    fake = t.zeros(len(str_toks))
                    fake[candidates[t.randperm(len(candidates), generator=generator)[: int(active.sum())]]] = 1.0
                    text = format_example(str_toks, fake, 0.5)
                requests.append((explanation, text))
    scores = detection_scores(tokenizer, llm, requests, template=FUZZ_TEMPLATE).reshape(len(units), 2, k)
    return t.tensor([auroc(s[0], s[1]) for s in scores])

# %%

if MAIN:
    fuzz_gen = t.Generator().manual_seed(0)
    fuzz_own = fuzzing_auroc(model, score_tokens, score_units, explanations, explainer_tok, explainer, fuzz_gen)
    fuzz_other = fuzzing_auroc(model, score_tokens, score_units, other_explanations, explainer_tok, explainer, fuzz_gen)
    fuzz_human = fuzzing_auroc(model, score_tokens, [score_units[j] for j in HUMAN_IDS], list(HUMAN_EXPLANATIONS.values()), explainer_tok, explainer, fuzz_gen)
    print(f"Fuzzing AUROC, all {N_LATENTS} latents: own explanation {fuzz_own.mean():.3f}, another latent's explanation {fuzz_other.mean():.3f}")
    print(f"Fuzzing AUROC, the 12 latents with human explanations: LLM {fuzz_own[HUMAN_IDS].mean():.3f}, human {fuzz_human.mean():.3f}")

# %%

TEST_SENTENCE_TEMPLATE = (
    "A feature in a language model responds to {explanation}.\n\n"
    "Write {n} different, varied sentences that the feature should respond to. Write natural sentences, like ones you "
    "might find on a web page, not sentences about the feature. Put each sentence on its own line, with no numbering."
)


def parse_sentences(reply: str, n: int) -> list[str]:
    """Splits a reply into lines, strips bullets/numbering, and keeps the first n lines longer than 10 characters."""
    lines = [line.strip().lstrip("-*•0123456789.) ").strip() for line in reply.split("\n")]
    return [line for line in lines if len(line) > 10][:n]


def generate_test_sentences(tokenizer, llm, explanations: list[str], n: int = 6) -> list[list[str]]:
    """For each explanation, n sentences written by the LLM that the explanation says should activate the unit."""
    conversations = [
        [{"role": "user", "content": TEST_SENTENCE_TEMPLATE.format(explanation=explanation, n=n)}]
        for explanation in explanations
    ]
    replies = utils.generate_chat(tokenizer, llm, conversations, max_new_tokens=300)
    return [parse_sentences(reply, n) for reply in replies]


def max_activations_on_texts(model: HookedTransformer, texts: list[str], hook_name: str, act_fn) -> Float[Tensor, "n_texts n_units"]:
    """Max activation of each unit over the token positions of each text."""
    return t.stack([collect_activations(model, model.to_tokens(text), hook_name, act_fn)[0].max(dim=0).values for text in texts])


if MAIN:
    tests.test_max_activations_on_texts(max_activations_on_texts, model)

# %%

if MAIN:
    test_sentences = generate_test_sentences(explainer_tok, explainer, explanations)
    human_test_sentences = generate_test_sentences(explainer_tok, explainer, list(HUMAN_EXPLANATIONS.values()))
    
    def hit_rate(j: int, sentences: list[str]) -> tuple[float, Tensor]:
        """Fraction of sentences on which latent j fires above 10% of its explain-split max, and the activations / max."""
        acts = max_activations_on_texts(model, sentences, HOOK, partial(sae_latent_acts, sae=sae, latents=[LATENTS[j]]))[:, 0]
        rel = acts / explain_acts[..., j].max()
        return (rel > 0.1).float().mean().item(), rel
    
    own_hits = [hit_rate(j, test_sentences[j])[0] for j in range(N_LATENTS)]
    other_hits = [hit_rate(j, test_sentences[(j + 1) % N_LATENTS])[0] for j in range(N_LATENTS)]
    human_hits = [hit_rate(j, sents)[0] for j, sents in zip(HUMAN_IDS, human_test_sentences)]
    
    print(f"Fraction of test sentences that activate the latent (mean over {N_LATENTS} latents):")
    print(f"  sentences written from the latent's own explanation:  {sum(own_hits) / N_LATENTS:.3f}")
    print(f"  sentences written from another latent's explanation:  {sum(other_hits) / N_LATENTS:.3f}")
    print(f"For the 12 latents with human explanations: LLM explanation {sum(own_hits[j] for j in HUMAN_IDS) / 12:.3f}, human explanation {sum(human_hits) / 12:.3f}")
    
    utils.plot_score_dots(
        [str(latent) for latent in LATENTS],
        {"sentences from own explanation": own_hits, "sentences from another latent's explanation": other_hits},
        None,
        title="How often does the latent fire on new sentences written to fit its explanation? (6 sentences per latent)",
        ylabel="fraction of sentences that activate the latent",
        xlabel="SAE latent (sorted by the hit rate of sentences from its own explanation)",
    )

# %%

if MAIN:
    print("For reference, the top 6 examples of latent 18120 on the explain split:")
    show_examples(explain_acts[..., LATENTS.index(18120)], explain_tokens, top_activating_examples(explain_acts[..., LATENTS.index(18120)], 6),
                  0.25 * explain_acts[..., LATENTS.index(18120)].max().item())
    print()
    for latent in [4412, 18120, 11803]:
        j = LATENTS.index(latent)
        h = HUMAN_IDS.index(j)
        for name, explanation, sentences in [("LLM", explanations[j], test_sentences[j]), ("human", list(HUMAN_EXPLANATIONS.values())[h], human_test_sentences[h])]:
            _, rel = hit_rate(j, sentences)
            print(f"latent {latent}, {name} explanation: {explanation!r}")
            for sentence, r in zip(sentences, rel.tolist()):
                print(f"   {r:4.2f} | {sentence}")
        print()

# %%

def steering_hook(resid: Float[Tensor, "batch seq d_model"], hook, vector: Float[Tensor, " d_model"]) -> Float[Tensor, "batch seq d_model"]:
    return resid + vector


@t.inference_mode()
def steered_generations(
    model: HookedTransformer, vector: Float[Tensor, " d_model"], prompt_tokens: Int[Tensor, "n_prompts seq"], max_new_tokens: int = 24
) -> list[str]:
    """Greedy continuations of each prompt, while adding `vector` to the residual stream at HOOK."""
    with model.hooks(fwd_hooks=[(HOOK, partial(steering_hook, vector=vector))]):
        out = model.generate(prompt_tokens, max_new_tokens=max_new_tokens, do_sample=False, verbose=False, stop_at_eos=False)
    return [model.to_string(row[prompt_tokens.shape[1] :]) for row in out]


if MAIN:
    tests.test_steered_generations(steered_generations, model, HOOK)

# %%

def intervention_curve(
    model: HookedTransformer,
    tokenizer,
    llm,
    direction: Float[Tensor, " d_model"],
    explanations: list[str],
    coefs: list[float],
    prompt_tokens: Int[Tensor, "n_prompts seq"],
) -> tuple[Float[Tensor, "n_explanations n_coefs"], list[list[str]]]:
    """Mean detection score of each explanation on text generated while steering with c * direction, for each c."""
    scores = t.zeros(len(explanations), len(coefs))
    texts_by_coef = []
    for c, coef in enumerate(coefs):
        texts = steered_generations(model, coef * direction, prompt_tokens)
        texts_by_coef.append(texts)
        requests = [(explanation, text) for explanation in explanations for text in texts]
        scores[:, c] = detection_scores(tokenizer, llm, requests).reshape(len(explanations), len(texts)).mean(dim=-1)
    return scores, texts_by_coef


if MAIN:
    tests.test_intervention_curve(intervention_curve, model, explainer_tok, explainer)

# %%

if MAIN:
    STEERING_PROMPTS = [
        "I think that the", "The weather today is", "My favourite thing is", "In the beginning there",
        "She told me that", "Here is what happened", "The best way to", "Yesterday we went to",
    ]
    steer_prompt_tokens = model.to_tokens(STEERING_PROMPTS)
    assert all(len(model.to_tokens(p)[0]) == steer_prompt_tokens.shape[1] for p in STEERING_PROMPTS), "prompts must have equal length"
    
    COEFS = [0.0, 1.0, 2.0, 4.0]
    random_steer_dirs = t.randn(N_LATENTS, model.cfg.d_model, generator=t.Generator().manual_seed(3))
    random_steer_dirs = (random_steer_dirs / random_steer_dirs.norm(dim=-1, keepdim=True)).to(device)
    
    curves = {"latent direction, own explanation": [], "random direction, own explanation": [], "latent direction, another latent's explanation": []}
    steered_texts = {}
    for j in tqdm(range(N_LATENTS)):
        scale = explain_acts[..., j].max().item()
        latent_scores, steered_texts[j] = intervention_curve(
            model, explainer_tok, explainer, scale * sae.W_dec[LATENTS[j]], [explanations[j], other_explanations[j]], COEFS, steer_prompt_tokens
        )
        random_scores, _ = intervention_curve(
            model, explainer_tok, explainer, scale * random_steer_dirs[j], [explanations[j]], COEFS, steer_prompt_tokens
        )
        curves["latent direction, own explanation"].append(latent_scores[0])
        curves["latent direction, another latent's explanation"].append(latent_scores[1])
        curves["random direction, own explanation"].append(random_scores[0])
    
    # change relative to no steering (coefficient 0), mean +- s.e.m. over latents
    changes = {name: t.stack(c) - t.stack(c)[:, :1] for name, c in curves.items()}
    for name, change in changes.items():
        print(f"{name:<48} mean change: {change.mean(0).numpy().round(2)}")
    gap = changes["latent direction, own explanation"][:, 1:] - changes["random direction, own explanation"][:, 1:]
    print(f"latents where steering raises the judge score more than random steering (at c=1, 2, 4): {(gap > 0).sum(0).tolist()} / {N_LATENTS}")
    
    utils.plot_lines(
        COEFS,
        {name: change.mean(0).tolist() for name, change in changes.items()},
        errors={name: (change.std(0) / N_LATENTS**0.5).tolist() for name, change in changes.items()},
        title=f"Intervention scoring: steer GPT-2 with c x max activation x decoder direction ({N_LATENTS} latents)",
        xlabel="steering coefficient c", ylabel="change in judge score vs. c = 0\n(logit Yes - logit No)",
        hline=0.0,
    )

# %%

if MAIN:
    for latent in [11803, 10483]:
        j = LATENTS.index(latent)
        top_logit_tokens = (sae.W_dec[latent] @ model.W_U).topk(10).indices
        print(f"latent {latent}. Explanation: {explanations[j]!r}")
        print(f"  logit lens of decoder direction: {[model.to_string(tok) for tok in top_logit_tokens]}")
        for c, coef in enumerate(COEFS):
            print(f"  c = {coef}: {STEERING_PROMPTS[0]}{steered_texts[j][c][0]!r}")
            print(f"  {'':8}{STEERING_PROMPTS[1]}{steered_texts[j][c][1]!r}")
        print()

# %%

if MAIN:
    ts_tokens = utils.load_tinystories_tokens(model, n_sequences=4000, seq_len=64)
    ts_explain_tokens, ts_score_tokens = ts_tokens[:2000], ts_tokens[2000:]
    ts_explain_acts = collect_activations(model, ts_explain_tokens, HOOK, latent_fn)
    ts_score_acts = collect_activations(model, ts_score_tokens, HOOK, latent_fn)
    
    strength = ts_explain_acts.amax((0, 1)) / explain_acts.amax((0, 1))
    print(f"Max activation on TinyStories, as a fraction of max activation on the Pile (median over latents): {strength.median():.2f}")
    print(f"Latents whose TinyStories max is below 25% of their Pile max: {(strength < 0.25).sum()} / {N_LATENTS}")
    
    ts_explanations = explain_units(model, ts_explain_tokens, ts_explain_acts, explainer_tok, explainer)

# %%

def cross_dataset_scores(
    model: HookedTransformer,
    datasets: dict[str, tuple[Int[Tensor, "n_seq seq"], Float[Tensor, "n_seq seq n_units"]]],
    explanation_sets: dict[str, list[str]],
    tokenizer,
    llm,
    seed: int = 0,
) -> dict[tuple[str, str], Float[Tensor, " n_units"]]:
    """Simulation scores of each set of explanations on each dataset's score split."""
    results = {}
    for data_name, (tokens, acts) in datasets.items():
        units = [acts[..., j] for j in range(acts.shape[-1])]
        examples = [scoring_examples(units[j], t.Generator().manual_seed(seed + j)) for j in range(len(units))]
        for source, explanation_list in explanation_sets.items():
            results[(source, data_name)] = simulation_scores(model, tokens, units, explanation_list, examples, tokenizer, llm)[0]
    return results

# %%

if MAIN:
    cross = cross_dataset_scores(
        model,
        {"Pile": (score_tokens, score_acts), "TinyStories": (ts_score_tokens, ts_score_acts)},
        {"Pile": explanations, "TinyStories": ts_explanations},
        explainer_tok,
        explainer,
    )
    names = ["Pile", "TinyStories"]
    cross_matrix = t.tensor([[cross[(source, data)].mean().item() for data in names] for source in names])
    print(cross_matrix)
    utils.plot_matrix(
        cross_matrix, row_labels=[f"explanation written from\n{n} examples" for n in names],
        col_labels=[f"scored on {n}" for n in names],
        title=f"Mean simulation score over {N_LATENTS} latents", xlabel="scoring data (held-out split)",
        ylabel="explanation source", cbar_label="mean simulation score", vmin=0, vmax=0.35, fmt="{:.3f}", figsize=(6, 4.5),
    )

# %%

if MAIN:
    drop = cross[("Pile", "Pile")] - cross[("TinyStories", "Pile")]
    for j in drop.argsort(descending=True)[:5].tolist():
        seq, pos = top_activating_examples(ts_explain_acts[..., j], 1)[0]
        str_toks, window_acts = get_window(model, ts_explain_tokens, ts_explain_acts[..., j], seq, pos)
        print(f"latent {LATENTS[j]} (TinyStories max = {strength[j]:.0%} of Pile max)")
        print(f"  Pile explanation:        {explanations[j]!r}   (Pile score {cross[('Pile', 'Pile')][j]:.2f})")
        print(f"  TinyStories explanation: {ts_explanations[j]!r}   (Pile score {cross[('TinyStories', 'Pile')][j]:.2f})")
        print(f"  top TinyStories example: {format_example(str_toks, window_acts, 0.25 * ts_explain_acts[..., j].max().item())}")

# %%

if MAIN:
    MY_LATENT = 4412
    MY_EXPLANATION = 'the word "to" directly after "similar"'
    
    my_fn = partial(sae_latent_acts, sae=sae, latents=[MY_LATENT])
    my_explain_acts = collect_activations(model, explain_tokens, HOOK, my_fn)
    my_score_acts = collect_activations(model, score_tokens, HOOK, my_fn)[..., 0]
    print(f"Latent {MY_LATENT}: fires on {(my_explain_acts[:, 1:] > 0).float().mean():.3%} of tokens. Top 5 examples:")
    show_examples(my_explain_acts[..., 0], explain_tokens, top_activating_examples(my_explain_acts[..., 0], 5), 0.25 * my_explain_acts.max().item())
    
    llm_explanation = explain_units(model, explain_tokens, my_explain_acts, explainer_tok, explainer)[0]
    candidates = {"LLM": llm_explanation, "mine": MY_EXPLANATION}
    my_examples = scoring_examples(my_score_acts, t.Generator().manual_seed(0))
    my_sim, _, _ = simulation_scores(model, score_tokens, [my_score_acts] * 2, list(candidates.values()), [my_examples] * 2, explainer_tok, explainer)
    _, my_negatives = detection_examples(my_score_acts, t.Generator().manual_seed(0))
    my_det = detection_auroc(model, score_tokens, list(candidates.values()), [([top_activating_examples(my_score_acts, 10)], my_negatives)] * 2, explainer_tok, explainer)[:, 0]
    print()
    for (name, explanation), sim, det in zip(candidates.items(), my_sim, my_det):
        print(f"{name:>5}: simulation {sim:5.2f}   detection AUROC {det:4.2f}   {explanation!r}")

# %%

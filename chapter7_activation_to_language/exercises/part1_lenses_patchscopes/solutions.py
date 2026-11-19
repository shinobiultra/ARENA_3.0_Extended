# %%


import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Float, Int
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import ActivationCache, HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter7_activation_to_language"
section = "part1_lenses_patchscopes"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part1_lenses_patchscopes.tests as tests
import part1_lenses_patchscopes.utils as utils

MAIN = __name__ == "__main__"

# %%

if MAIN:
    GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
    hf_model = AutoModelForCausalLM.from_pretrained("gpt2", revision=GPT2_REVISION)
    hf_tokenizer = AutoTokenizer.from_pretrained("gpt2", revision=GPT2_REVISION)
    model = HookedTransformer.from_pretrained("gpt2-small", hf_model=hf_model, tokenizer=hf_tokenizer, device=device)
    model.requires_grad_(False)
    
    print(f"{model.cfg.n_layers} layers, d_model={model.cfg.d_model}, vocab size={model.cfg.d_vocab}")

# %%

FEW_SHOT_PREFIX = "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is Rome."

CANDIDATE_CAPITALS = [
    ("Germany", "Berlin"), ("Spain", "Madrid"), ("Russia", "Moscow"), ("China", "Beijing"),
    ("England", "London"), ("Egypt", "Cairo"), ("Greece", "Athens"), ("Canada", "Ottawa"),
    ("Australia", "Canberra"), ("Ireland", "Dublin"), ("Austria", "Vienna"), ("Poland", "Warsaw"),
    ("Sweden", "Stockholm"), ("Norway", "Oslo"), ("Turkey", "Ankara"), ("Thailand", "Bangkok"),
    ("Portugal", "Lisbon"), ("Cuba", "Havana"), ("Peru", "Lima"), ("Iran", "Tehran"),
    ("Iraq", "Baghdad"), ("Syria", "Damascus"), ("Hungary", "Budapest"), ("Denmark", "Copenhagen"),
    ("Finland", "Helsinki"), ("Belgium", "Brussels"), ("Netherlands", "Amsterdam"), ("Korea", "Seoul"),
    ("Ukraine", "Kiev"), ("Chile", "Santiago"), ("Mexico", "Mexico"), ("India", "New"),
    ("Israel", "Jerusalem"), ("Pakistan", "Islamabad"), ("Brazil", "Bras"), ("Scotland", "Edinburgh"),
    ("Wales", "Cardiff"), ("Switzerland", "Bern"), ("Afghanistan", "Kabul"), ("Lebanon", "Beirut"),
    ("Philippines", "Manila"), ("Indonesia", "Jakarta"), ("Malaysia", "Kuala"), ("Singapore", "Singapore"),
    ("Czech", "Prague"), ("Libya", "Tripoli"), ("Tunisia", "Tunis"), ("Jamaica", "Kingston"),
]


def is_single_token(model: HookedTransformer, s: str) -> bool:
    return model.to_tokens(s, prepend_bos=False).shape[-1] == 1


def make_prompt(country: str) -> str:
    return f"{FEW_SHOT_PREFIX} The capital of {country} is"

# %%

if MAIN:
    CAPITALS = [
        (country, capital)
        for country, capital in CANDIDATE_CAPITALS
        if is_single_token(model, " " + country) and is_single_token(model, " " + capital)
    ]
    countries = [country for country, _ in CAPITALS]
    prompts = [make_prompt(country) for country in countries]
    tokens = model.to_tokens(prompts)
    answer_tokens = model.to_tokens([" " + capital for _, capital in CAPITALS], prepend_bos=False)[:, 0]
    
    print(f"{len(CAPITALS)} country/capital pairs, all prompts have length {tokens.shape[1]}")
    print([model.to_string(tok) for tok in tokens[0]])

# %%

if MAIN:
    logits, cache = model.run_with_cache(tokens)
    predictions = logits[:, -1].argmax(-1)
    correct = predictions == answer_tokens
    
    print(f"Top-1 accuracy: {correct.float().mean():.1%}\n")
    print("Mistakes:")
    for (country, capital), pred, ok in zip(CAPITALS, predictions, correct):
        if not ok:
            print(f"  {country:<12} expected {capital!r:<12} got {model.to_string(pred)!r}")

# %%

def logit_lens(
    model: HookedTransformer, cache: ActivationCache, pos: int = -1
) -> Float[Tensor, "layer batch d_vocab"]:
    """
    Returns the logit lens decoding of the residual stream at sequence position `pos`, for the
    stream before every block plus the stream after the final block (so n_layers + 1 rows).
    """
    n_layers = model.cfg.n_layers
    resid_stack = t.stack(
        [cache["resid_pre", layer][:, pos] for layer in range(n_layers)] + [cache["resid_post", n_layers - 1][:, pos]]
    )
    return model.unembed(model.ln_final(resid_stack))


if MAIN:
    tests.test_logit_lens(logit_lens, model)

# %%

if MAIN:
    LAYER_LABELS = ["embed"] + [f"pre {layer}" for layer in range(1, model.cfg.n_layers)] + ["final"]
    lens_logits = logit_lens(model, cache, pos=-1)
    print(lens_logits.shape)

# %%

def top_tokens_table(
    model: HookedTransformer, logits: Float[Tensor, "layer d_vocab"], k: int = 5, row_labels: list[str] | None = None
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per layer, where column j contains the j-th most likely token
    at that layer along with its probability, formatted like "' Berlin' (0.42)".
    """
    probs = logits.softmax(dim=-1)
    top_probs, top_ids = probs.topk(k, dim=-1)
    rows = [
        [f"{model.to_string(tok)!r} ({p:.2f})" for tok, p in zip(ids.tolist(), ps.tolist())]
        for ids, ps in zip(top_ids, top_probs)
    ]
    return pd.DataFrame(rows, index=row_labels, columns=[f"top {j + 1}" for j in range(k)])


if MAIN:
    tests.test_top_tokens_table(top_tokens_table, model)

# %%

if MAIN:
    germany_idx = countries.index("Germany")
    display(top_tokens_table(model, lens_logits[:, germany_idx], k=4, row_labels=LAYER_LABELS))

# %%

def answer_ranks(
    logits: Float[Tensor, "layer batch d_vocab"], answer_tokens: Int[Tensor, " batch"]
) -> Int[Tensor, "layer batch"]:
    """
    Returns the rank (1 = top) of answer_tokens[b] in logits[layer, b], for every layer and batch index.
    """
    index = answer_tokens.to(logits.device)[None, :, None].expand(logits.shape[0], -1, 1)
    answer_logits = logits.gather(dim=-1, index=index)
    return (logits > answer_logits).sum(dim=-1) + 1


if MAIN:
    tests.test_answer_ranks(answer_ranks)

# %%

def other_answer_ranks(
    logits: Float[Tensor, "layer batch d_vocab"], answer_tokens: Int[Tensor, " batch"]
) -> Float[Tensor, "layer batch"]:
    """
    For each layer and prompt b, returns the median rank of the answers of all prompts other than b.
    """
    answer_tokens = answer_tokens.to(logits.device)
    batch = logits.shape[1]
    # all_answer_logits[l, b, j] = logit of answer j in prompt b's logits at layer l
    all_answer_logits = logits[:, :, answer_tokens]
    # ranks[l, b, j] = rank of answer j in prompt b's logits at layer l
    ranks = (logits[:, :, None, :] > all_answer_logits[..., None]).sum(dim=-1) + 1
    off_diagonal = ~t.eye(batch, dtype=t.bool, device=logits.device)
    other_ranks = ranks[:, off_diagonal].reshape(logits.shape[0], batch, batch - 1)
    return other_ranks.float().median(dim=-1).values


if MAIN:
    tests.test_other_answer_ranks(other_answer_ranks, answer_ranks)

# %%

def random_vectors_like(x: Float[Tensor, "... d_model"], seed: int = 0) -> Float[Tensor, "... d_model"]:
    """Random Gaussian vectors with the same norm as each vector in x."""
    generator = t.Generator(device="cpu").manual_seed(seed)
    noise = t.randn(x.shape, generator=generator).to(x.device)
    return noise / noise.norm(dim=-1, keepdim=True) * x.norm(dim=-1, keepdim=True)

# %%

if MAIN:
    right = correct.nonzero().squeeze(-1)
    ranks = answer_ranks(lens_logits[:, right], answer_tokens[right])
    other_ranks = other_answer_ranks(lens_logits[:, right], answer_tokens[right])
    
    resid_stack = t.stack(
        [cache["resid_pre", layer][right, -1] for layer in range(model.cfg.n_layers)]
        + [cache["resid_post", model.cfg.n_layers - 1][right, -1]]
    )
    random_logits = model.unembed(model.ln_final(random_vectors_like(resid_stack)))
    random_ranks = answer_ranks(random_logits, answer_tokens[right])
    
    utils.plot_lens_ranks(
        {
            "correct capital": ranks.float().median(dim=1).values.tolist(),
            "other countries' capitals": other_ranks.median(dim=1).values.tolist(),
            "random vector (norm-matched)": random_ranks.float().median(dim=1).values.tolist(),
        },
        LAYER_LABELS,
        title=f"Logit lens on GPT-2 Small: median rank of the answer ({len(right)} prompts)",
    )

# %%

def kl_to_final(logits: Float[Tensor, "layer batch d_vocab"]) -> Float[Tensor, " layer"]:
    """
    Returns KL(final || layer), averaged over the batch, for every layer. The last layer is treated as final.
    """
    logprobs = logits.log_softmax(dim=-1)
    final = logprobs[-1:]
    return (final.exp() * (final - logprobs)).sum(dim=-1).mean(dim=-1)


if MAIN:
    tests.test_kl_to_final(kl_to_final)

# %%

if MAIN:
    kl = kl_to_final(lens_logits)
    for label, value in zip(LAYER_LABELS, kl.tolist()):
        print(f"{label:>6}: {value:6.2f} nats")

# %%

def attention_lens(
    model: HookedTransformer, cache: ActivationCache, answer_tokens: Int[Tensor, " batch"]
) -> Float[Tensor, "layer head batch"]:
    """
    Returns each attention head's direct contribution to the logit of answer_tokens[b] at the final position,
    read through the final layernorm's own scale (so that all contributions add up to the real logit).
    """
    z = t.stack([cache["z", layer][:, -1] for layer in range(model.cfg.n_layers)])  # (layer, batch, head, d_head)
    head_out = t.einsum("lbhd,lhdm->lhbm", z, model.W_O)  # each head's write to the residual stream
    head_out = head_out - head_out.mean(dim=-1, keepdim=True)
    scale = cache["ln_final.hook_scale"][:, -1]  # (batch, 1)
    answer_directions = model.W_U[:, answer_tokens.to(model.W_U.device)]  # (d_model, batch)
    return t.einsum("lhbm,mb->lhb", head_out / scale, answer_directions)


if MAIN:
    tests.test_attention_lens(attention_lens, model)

# %%

if MAIN:
    head_effects = attention_lens(model, cache, answer_tokens)
    mean_effect = head_effects[:, :, right].mean(dim=-1)
    print("Heads that add most to the correct capital's logit (mean over the prompts the model gets right):")
    for value, flat in zip(*mean_effect.flatten().topk(5)):
        layer, head = divmod(flat.item(), model.cfg.n_heads)
        print(f"  L{layer}H{head}: {value:+.2f}")
    final_answer_logit = logits[right, -1].gather(-1, answer_tokens[right, None])[:, 0].mean()
    print(f"All {model.cfg.n_layers * model.cfg.n_heads} heads together: {mean_effect.sum():+.2f}.  The correct capital's final logit: {final_answer_logit:.2f}")

# %%

class TunedLens(nn.Module):
    def __init__(self, n_layers: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(t.zeros(n_layers, d_model, d_model))
        self.bias = nn.Parameter(t.zeros(n_layers, d_model))

    def forward(self, resid: Float[Tensor, "... d_model"], layer: int) -> Float[Tensor, "... d_model"]:
        return resid + resid @ self.weight[layer] + self.bias[layer]

    def logits(self, model: HookedTransformer, resid: Float[Tensor, "... d_model"], layer: int) -> Float[Tensor, "... d_vocab"]:
        return model.unembed(model.ln_final(self.forward(resid, layer)))


if MAIN:
    tests.test_tuned_lens_init(TunedLens, model)

# %%

def tuned_lens_loss(
    lens: TunedLens,
    model: HookedTransformer,
    resid_stack: Float[Tensor, "layer batch seq d_model"],
    final_logprobs: Float[Tensor, "batch seq d_vocab"],
) -> Float[Tensor, ""]:
    """
    Returns the mean over layers of KL(final || tuned lens at that layer), averaged over batch and over
    sequence positions 1 onwards.
    """
    n_layers = resid_stack.shape[0]
    target = final_logprobs[:, 1:]
    total = 0.0
    for layer in range(n_layers):
        lens_logprobs = lens.logits(model, resid_stack[layer, :, 1:], layer).log_softmax(dim=-1)
        total = total + (target.exp() * (target - lens_logprobs)).sum(dim=-1).mean()
    return total / n_layers


if MAIN:
    tests.test_tuned_lens_loss(tuned_lens_loss, TunedLens, model)

# %%

if MAIN:
    all_tokens = utils.load_tinystories_tokens(model, n_sequences=1800, seq_len=64)
    train_tokens, test_tokens = all_tokens[:1600], all_tokens[1600:]
    print(train_tokens.shape, test_tokens.shape)
    print(repr(model.to_string(train_tokens[0, 1:40])))

# %%

@t.no_grad()
def get_resid_and_logprobs(
    model: HookedTransformer, tokens: Int[Tensor, "batch seq"]
) -> tuple[Float[Tensor, "layer batch seq d_model"], Float[Tensor, "batch seq d_vocab"]]:
    logits, cache = model.run_with_cache(tokens.to(model.cfg.device), names_filter=lambda name: name.endswith("resid_pre"))
    resid_stack = t.stack([cache["resid_pre", layer] for layer in range(model.cfg.n_layers)])
    return resid_stack, logits.log_softmax(dim=-1)

# %%

def train_tuned_lens(
    lens: TunedLens,
    model: HookedTransformer,
    tokens: Int[Tensor, "n_seq seq"],
    n_steps: int = 300,
    batch_size: int = 8,
    lr: float = 1e-3,
    seed: int = 0,
) -> list[float]:
    """Trains `lens` in place, and returns the loss at each step."""
    generator = t.Generator().manual_seed(seed)
    optimizer = t.optim.Adam(lens.parameters(), lr=lr)
    losses = []
    for _ in tqdm(range(n_steps)):
        idx = t.randint(0, len(tokens), (batch_size,), generator=generator)
        resid_stack, final_logprobs = get_resid_and_logprobs(model, tokens[idx])
        loss = tuned_lens_loss(lens, model, resid_stack, final_logprobs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses

# %%

if MAIN:
    tests.test_train_tuned_lens(train_tuned_lens)

# %%

if MAIN:
    t.manual_seed(0)
    tuned_lens = TunedLens(model.cfg.n_layers, model.cfg.d_model).to(device)
    losses = train_tuned_lens(tuned_lens, model, train_tokens, n_steps=300 if device.type != "cpu" else 100)
    
    plt.figure(figsize=(6, 3))
    plt.plot(losses)
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("mean KL over layers (nats)")
    plt.title("Tuned lens training loss")
    plt.grid(alpha=0.3)
    plt.show()

# %%

@t.inference_mode()
def evaluate_lenses(
    model: HookedTransformer, lens: TunedLens, tokens: Int[Tensor, "n_seq seq"], batch_size: int = 20
) -> dict[str, list[float]]:
    if batch_size <= 0 or len(tokens) == 0 or tokens.shape[1] < 2:
        raise ValueError("Need a positive batch size and nonempty sequences with non-BOS positions")
    n_layers = model.cfg.n_layers
    metrics = {key: t.zeros(n_layers, dtype=t.float64) for key in ["logit_kl", "tuned_kl", "logit_agree", "tuned_agree"]}
    n_tokens = 0
    for start in range(0, len(tokens), batch_size):
        resid_stack, final_logprobs = get_resid_and_logprobs(model, tokens[start : start + batch_size])
        target = final_logprobs[:, 1:]
        target_top = target.argmax(dim=-1)
        for layer in range(n_layers):
            resid = resid_stack[layer, :, 1:]
            for name, lp in [
                ("logit", model.unembed(model.ln_final(resid)).log_softmax(dim=-1)),
                ("tuned", lens.logits(model, resid, layer).log_softmax(dim=-1)),
            ]:
                metrics[f"{name}_kl"][layer] += (target.exp() * (target - lp)).sum(dim=-1).float().sum().item()
                metrics[f"{name}_agree"][layer] += (lp.argmax(dim=-1) == target_top).sum().item()
        n_tokens += target_top.numel()
    return {key: (value / n_tokens).tolist() for key, value in metrics.items()}

# %%

if MAIN:
    tests.test_evaluate_lenses(evaluate_lenses)

# %%

if MAIN:
    results = evaluate_lenses(model, tuned_lens, test_tokens)
    display(pd.DataFrame(results).rename_axis("layer").round(3).T)
    utils.plot_lens_comparison(
        results["logit_kl"],
        results["tuned_kl"],
        results["logit_agree"],
        results["tuned_agree"],
        floor_agree=results["tuned_agree"][0],
    )

# %%

@t.inference_mode()
def shifted_translator_agreement(
    model: HookedTransformer, lens: TunedLens, tokens: Int[Tensor, "n_seq seq"], shift: int, batch_size: int = 20
) -> dict[int, float]:
    if batch_size <= 0 or len(tokens) == 0 or tokens.shape[1] < 2:
        raise ValueError("Need a positive batch size and nonempty sequences with non-BOS positions")
    n_layers = model.cfg.n_layers
    layers = [layer for layer in range(n_layers) if 0 <= layer + shift < n_layers]
    if not layers:
        return {}
    totals = {layer: 0 for layer in layers}
    n_tokens = 0
    for start in range(0, len(tokens), batch_size):
        resid_stack, final_logprobs = get_resid_and_logprobs(model, tokens[start : start + batch_size])
        target_top = final_logprobs[:, 1:].argmax(dim=-1)
        for layer in layers:
            lp = lens.logits(model, resid_stack[layer, :, 1:], layer + shift)
            totals[layer] += (lp.argmax(dim=-1) == target_top).sum().item()
        n_tokens += target_top.numel()
    return {layer: total / n_tokens for layer, total in totals.items()}

# %%

if MAIN:
    tests.test_shifted_translator_agreement(shifted_translator_agreement)

# %%

if MAIN:
    shifted = {shift: shifted_translator_agreement(model, tuned_lens, test_tokens, shift) for shift in [-3, 3]}
    rows = []
    for layer in range(model.cfg.n_layers):
        rows.append({
            "layer": layer,
            "logit lens": results["logit_agree"][layer],
            "tuned lens (correct translator)": results["tuned_agree"][layer],
            "translator from layer - 3": shifted[-3].get(layer, float("nan")),
            "translator from layer + 3": shifted[3].get(layer, float("nan")),
        })
    display(pd.DataFrame(rows).set_index("layer").style.format("{:.2f}").background_gradient(cmap="viridis", vmin=0, vmax=1))

# %%

if MAIN:
    with t.inference_mode():
        capital_resid = t.stack([cache["resid_pre", layer][right, -1] for layer in range(model.cfg.n_layers)])
        tuned_capital_logits = t.stack([tuned_lens.logits(model, capital_resid[layer], layer) for layer in range(model.cfg.n_layers)])
        tuned_ranks = answer_ranks(tuned_capital_logits, answer_tokens[right])
        tuned_other = other_answer_ranks(tuned_capital_logits, answer_tokens[right])
    
    utils.plot_lens_ranks(
        {
            "logit lens: correct capital": ranks[:-1].float().median(dim=1).values.tolist(),
            "logit lens: other capitals": other_ranks[:-1].median(dim=1).values.tolist(),
            "tuned lens: correct capital": tuned_ranks.float().median(dim=1).values.tolist(),
            "tuned lens: other capitals": tuned_other.median(dim=1).values.tolist(),
        },
        LAYER_LABELS[:-1],
        title="Capitals: logit lens vs tuned lens (median rank)",
    )

# %%

def normalize_rows(x: Float[Tensor, "... d"], eps: float = 1e-8) -> Float[Tensor, "... d"]:
    """Scales every vector along the last dimension to length 1 (all-zero vectors stay zero)."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


if MAIN:
    tests.test_normalize_rows(normalize_rows)

# %%

def contextual_topk(
    queries: Float[Tensor, "n_queries d"], bank: Float[Tensor, "n_bank d"], k: int = 1, chunk_size: int = 1024
) -> tuple[Float[Tensor, "n_queries k"], Int[Tensor, "n_queries k"]]:
    """
    Returns (cosine similarities, bank indices) of the k nearest bank vectors for every query, best first.
    Ties go to the lower bank index. Queries are processed `chunk_size` at a time.
    """
    bank = normalize_rows(bank)
    all_scores, all_indices = [], []
    for start in range(0, len(queries), chunk_size):
        scores = normalize_rows(queries[start : start + chunk_size]) @ bank.T  # (chunk, n_bank)
        order = scores.sort(dim=-1, descending=True, stable=True)
        all_scores.append(order.values[:, :k])
        all_indices.append(order.indices[:, :k])
    return t.cat(all_scores), t.cat(all_indices)


if MAIN:
    tests.test_contextual_topk(contextual_topk)

# %%

def layer_alignment_grid(
    query_states: Float[Tensor, "layer n_queries d"],
    bank_states: Float[Tensor, "layer n_bank d"],
    query_targets: Int[Tensor, " n_queries"],
    bank_labels: Int[Tensor, " n_bank"],
    chunk_size: int = 1024,
) -> Float[Tensor, "query_layer bank_layer"]:
    """
    grid[i, j] = fraction of layer-i queries whose nearest layer-j bank vector has a label equal to the query's target.
    """
    n_query_layers, n_bank_layers = query_states.shape[0], bank_states.shape[0]
    grid = t.zeros(n_query_layers, n_bank_layers)
    for i in range(n_query_layers):
        for j in range(n_bank_layers):
            _, nearest = contextual_topk(query_states[i], bank_states[j], k=1, chunk_size=chunk_size)
            grid[i, j] = (bank_labels[nearest[:, 0]] == query_targets).float().mean().item()
    return grid


if MAIN:
    tests.test_layer_alignment_grid(layer_alignment_grid)

# %%

@t.no_grad()
def collect_states(
    model: HookedTransformer, tokens: Int[Tensor, "n_seq seq"], batch_size: int = 100
) -> tuple[Float[Tensor, "layer n_seq pos d_model"], Int[Tensor, "n_seq pos"]]:
    """
    The residual stream before every block at positions 1 onwards (position 0 is BOS), and the model's final top-1
    prediction at the same positions.
    """
    states, tops = [], []
    for start in range(0, len(tokens), batch_size):
        resid_stack, final_logprobs = get_resid_and_logprobs(model, tokens[start : start + batch_size])
        states.append(resid_stack[:, :, 1:])
        tops.append(final_logprobs[:, 1:].argmax(dim=-1))
    return t.cat(states, dim=1), t.cat(tops)


def bigram_predictions(
    bank_current: Int[Tensor, " n_bank"], bank_next: Int[Tensor, " n_bank"], query_current: Int[Tensor, " n_queries"], vocab_size: int
) -> Int[Tensor, " n_queries"]:
    """For each query token, the token that most often followed it in the bank (-1 if it never appeared there)."""
    pairs = (bank_current * vocab_size + bank_next).cpu()
    unique, counts = pairs.unique(return_counts=True)
    best = {}
    for pair, count in zip(unique.tolist(), counts.tolist()):
        current, following = divmod(pair, vocab_size)
        if count > best.get(current, (-1, 0))[1]:
            best[current] = (following, count)
    return t.tensor([best.get(current, (-1, 0))[0] for current in query_current.cpu().tolist()])

# %%

if MAIN:
    n_bank = 600 if device.type != "cpu" else 200  # a smaller bank keeps the CPU run short
    bank_tokens, query_tokens = train_tokens[:n_bank], test_tokens[:100]
    bank_states, _ = collect_states(model, bank_tokens)
    query_states, query_top = collect_states(model, query_tokens)
    
    # A position p holds the state that predicts token p + 1, so the bank skips each story's last position.
    bank = bank_states[:, :, :-1].flatten(1, 2)  # (layer, n_bank * 62, d_model)
    bank_current, bank_next = bank_tokens[:, 1:-1].flatten().to(device), bank_tokens[:, 2:].flatten().to(device)
    queries = query_states.flatten(1, 2)  # (layer, 100 * 63, d_model)
    query_current, query_top = query_tokens[:, 1:].flatten().to(device), query_top.flatten()
    print(f"bank: {bank.shape[1]:,} vectors per layer from {n_bank} training stories.  queries: {queries.shape[1]:,} held-out states per layer.")

# %%

if MAIN:
    generator = t.Generator().manual_seed(0)
    bigram = bigram_predictions(bank_current, bank_next, query_current, model.cfg.d_vocab).to(device)
    bigram_floor = (bigram == query_top).float().mean().item()
    shuffled_next = bank_next[t.randperm(len(bank_next), generator=generator).to(device)]
    
    def match(predicted: Tensor) -> float:
        return (predicted == query_top).float().mean().item()
    
    rows = []
    with t.no_grad():
        for layer in range(model.cfg.n_layers):
            q, b = queries[layer], bank[layer]
            real_cos, real_nearest = contextual_topk(q, b)
            random_cos, random_nearest = contextual_topk(random_vectors_like(q), b)
            random_pairs = b[t.randint(len(b), (len(q),), generator=generator).to(device)]
            rows.append({
                "logit lens": match(model.unembed(model.ln_final(q)).argmax(-1)),
                "tuned lens": match(tuned_lens.logits(model, q, layer).argmax(-1)),
                "contextual lens": match(bank_next[real_nearest[:, 0]]),
                "shuffled bank labels": match(shuffled_next[real_nearest[:, 0]]),
                "random queries": match(bank_next[random_nearest[:, 0]]),
                "neighbour is the same token": (bank_current[real_nearest[:, 0]] == query_current).float().mean().item(),
                "cosine: real nearest": real_cos.mean().item(),
                "cosine: random-query nearest": random_cos.mean().item(),
                "cosine: two random states": (normalize_rows(q) * normalize_rows(random_pairs)).sum(-1).mean().item(),
            })
    lens_table = pd.DataFrame(rows).rename_axis("layer")
    layer0_floor = lens_table["tuned lens"][0]
    print(f"bigram floor: {bigram_floor:.2f}   layer-0 floor: {layer0_floor:.2f}")
    display(lens_table.iloc[:, :6].round(2).T)
    display(lens_table.iloc[:, 6:].round(2).T)

# %%

if MAIN:
    subset = t.randperm(queries.shape[1], generator=generator)[:1000].to(device)
    alignment = layer_alignment_grid(queries[:, subset], bank, query_current[subset], bank_current)
    utils.plot_lens_signature(
        lens_table[["logit lens", "tuned lens", "contextual lens"]].to_dict("list"),
        controls=lens_table[["shuffled bank labels", "random queries"]].to_dict("list"),
        floors={"bigram floor": bigram_floor, "layer-0 floor": layer0_floor},
        alignment=alignment,
    )

# %%

@t.inference_mode()
def get_source_vectors(
    model: HookedTransformer, prompts: list[str], pos: int, layer: int
) -> Float[Tensor, "batch d_model"]:
    _, cache = model.run_with_cache(model.to_tokens(prompts), names_filter=f"blocks.{layer}.hook_resid_pre")
    return cache["resid_pre", layer][:, pos].clone()


if MAIN:
    tests.test_get_source_vectors(get_source_vectors, model)

# %%

@t.inference_mode()
def get_all_layer_source_vectors(
    model: HookedTransformer, prompts: list[str], pos: int
) -> Float[Tensor, "layer batch d_model"]:
    _, cache = model.run_with_cache(model.to_tokens(prompts), names_filter=lambda name: name.endswith("resid_pre"))
    return t.stack([cache["resid_pre", layer][:, pos] for layer in range(model.cfg.n_layers)]).clone()

# %%

@t.inference_mode()
def run_patchscope(
    model: HookedTransformer,
    source_vectors: Float[Tensor, "batch d_model"],
    target_tokens: Int[Tensor, "1 seq"],
    target_pos: int,
    target_layer: int,
) -> Float[Tensor, "batch d_vocab"]:
    batch = source_vectors.shape[0]

    def patch_hook(resid: Float[Tensor, "batch seq d_model"], hook) -> Float[Tensor, "batch seq d_model"]:
        resid[:, target_pos] = source_vectors
        return resid

    logits = model.run_with_hooks(
        target_tokens.expand(batch, -1),
        fwd_hooks=[(f"blocks.{target_layer}.hook_resid_pre", patch_hook)],
    )
    return logits[:, -1]


if MAIN:
    tests.test_run_patchscope(run_patchscope, model)

# %%

if MAIN:
    target_tokens = model.to_tokens(make_prompt("x"))
    target_pos = target_tokens.shape[1] - 2
    assert model.to_string(target_tokens[0, target_pos]) == " x"
    
    source_prompts = [" " + country for country in countries]
    source_vectors = get_all_layer_source_vectors(model, source_prompts, pos=-1)
    
    patched_logits = run_patchscope(model, source_vectors[1], target_tokens, target_pos, target_layer=1)
    for country, capital, pred in list(zip(countries, [c for _, c in CAPITALS], patched_logits.argmax(-1)))[:10]:
        print(f"{country:<12} -> {model.to_string(pred)!r:<14} (true capital: {capital!r})")

# %%

@t.inference_mode()
def patchscope_accuracy_grid(
    model: HookedTransformer,
    source_vectors: Float[Tensor, "source_layer batch d_model"],
    target_tokens: Int[Tensor, "1 seq"],
    target_pos: int,
    answer_tokens: Int[Tensor, " batch"],
    target_layers: list[int],
) -> Float[Tensor, "source_layer target_layer"]:
    grid = t.zeros(source_vectors.shape[0], len(target_layers))
    for i, vectors in enumerate(source_vectors):
        for j, target_layer in enumerate(target_layers):
            logits = run_patchscope(model, vectors, target_tokens, target_pos, target_layer)
            grid[i, j] = (logits.argmax(dim=-1) == answer_tokens.to(logits.device)).float().mean()
    return grid


if MAIN:
    tests.test_patchscope_accuracy_grid(patchscope_accuracy_grid)

# %%

if MAIN:
    grid = patchscope_accuracy_grid(
        model, source_vectors, target_tokens, target_pos, answer_tokens, target_layers=list(range(model.cfg.n_layers))
    )
    utils.plot_patchscope_grid(
        grid, "Patchscope: which capital is this country?", ceiling=correct.float().mean().item()
    )

# %%

@t.inference_mode()
def patchscope_controls(
    model: HookedTransformer,
    source_prompts: list[str],
    source_layer: int,
    target_tokens: Int[Tensor, "1 seq"],
    target_pos: int,
    target_layer: int,
    answer_tokens: Int[Tensor, " batch"],
) -> dict[str, float]:
    answer_tokens = answer_tokens.to(model.cfg.device)

    def accuracy(logits: Tensor) -> float:
        return (logits.argmax(dim=-1) == answer_tokens).float().mean().item()

    vectors = get_source_vectors(model, source_prompts, pos=-1, layer=source_layer)
    bos_vectors = get_source_vectors(model, source_prompts, pos=0, layer=source_layer)
    text_only = model(target_tokens.expand(len(source_prompts), -1))[:, -1]
    return {
        "patchscope": accuracy(run_patchscope(model, vectors, target_tokens, target_pos, target_layer)),
        "text only (no patch)": accuracy(text_only),
        "random vector (norm-matched)": accuracy(
            run_patchscope(model, random_vectors_like(vectors), target_tokens, target_pos, target_layer)
        ),
        "wrong position (BOS)": accuracy(run_patchscope(model, bos_vectors, target_tokens, target_pos, target_layer)),
    }

# %%

if MAIN:
    tests.test_patchscope_controls(patchscope_controls)

# %%

if MAIN:
    controls = patchscope_controls(model, source_prompts, 4, target_tokens, target_pos, 1, answer_tokens)
    utils.plot_bars(controls, "Patchscope (source layer 4 → target layer 1) vs controls")

# %%

if MAIN:
    def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
        """95% Wilson score interval for a rate measured on n cases."""
        p = successes / n
        centre = (p + z**2 / (2 * n)) / (1 + z**2 / n)
        half = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / (1 + z**2 / n)
        return centre - half, centre + half
    
    
    n_countries = len(CAPITALS)
    for name, hits in [("Patchscope (source 4 -> target 1)", round(controls["patchscope"] * n_countries)), ("model on the real prompts", int(correct.sum()))]:
        low, high = wilson_interval(hits, n_countries)
        print(f"{name:<34} {hits}/{n_countries} = {hits / n_countries:.1%}   95% interval {low:.1%} to {high:.1%}")

# %%

if MAIN:
    LANGUAGE_PROMPT = "In France people speak French. In Germany people speak German. In Japan people speak Japanese. In x people speak"
    language_target = model.to_tokens(LANGUAGE_PROMPT)
    language_pos = [model.to_string(tok) for tok in language_target[0]].index(" x")
    
    play_countries = ["Spain", "Russia", "China", "Italy", "Greece", "Poland", "Sweden", "Turkey", "Iran", "Brazil"]
    play_vectors = get_source_vectors(model, [" " + c for c in play_countries], pos=-1, layer=3)
    play_logits = run_patchscope(model, play_vectors, language_target, language_pos, target_layer=1)
    for country, logits_row in zip(play_countries, play_logits):
        top = logits_row.softmax(-1).topk(3)
        print(f"{country:<8} -> " + ", ".join(f"{model.to_string(i)!r} ({p:.2f})" for p, i in zip(top.values, top.indices)))
    
    languages = [" Spanish", " Russian", " Chinese", " Italian", " Greek", " Polish", " Swedish", " Turkish", " Persian", " Portuguese"]
    language_answers = model.to_tokens(languages, prepend_bos=False)[:, 0]
    text_only_prediction = model(language_target)[:, -1].argmax(-1)  # nothing patched: one answer for every country
    print(f"\nPatchscope: {(play_logits.argmax(-1) == language_answers).sum().item()} of {len(play_countries)} languages right.")
    print(f"Text only (nothing patched): says {model.to_string(text_only_prediction[0])!r} for every country, "
          f"{(text_only_prediction == language_answers).sum().item()} of {len(play_countries)} right.")

# %%

if MAIN:
    sys.path.append(str(root_dir / "infrastructure" / "extension" / "learner"))
    from quiz import Interval, Quiz
    
    # Fill in every answer before you open any Reveal below.
    my_answers = {
        "Q1": "",  # "a" to "e"
        "Q2": "",
        "Q3": "",
        "Q4": "",
        "Q5": "",
    }
    my_reasons = {  # one sentence each: which result from this lesson are you relying on?
        "Q1": "",
        "Q2": "",
        "Q3": "",
        "Q4": "",
        "Q5": "",
    }

# %%

if MAIN:
    quiz = Quiz({
        "Q1": Interval("c", points=3),
        "Q2": Interval("c", points=4),
        "Q3": Interval("c", points=4),
        "Q4": Interval("c", points=4),
        "Q5": Interval("c", points=5),
    })
    quiz.score(my_answers)

# %%

if MAIN:
    from bisect import bisect_right
    
    
    def interval_of(value: float, edges: list[float]) -> str:
        """Which interval (a to e) a measured value falls in, given the four edges between the intervals."""
        return "abcde"[bisect_right(edges, value)]
    
    
    # Q1: the "X's capital city is" template. Rows of the logit lens: 0 = embed, 1-11 = pre 1 ... pre 11, 12 = final.
    def possessive_prompt(country: str) -> str:
        return f"France's capital city is Paris. Japan's capital city is Tokyo. Italy's capital city is Rome. {country}'s capital city is"
    
    
    q1_tokens = model.to_tokens([possessive_prompt(country) for country in countries])
    q1_logits, q1_cache = model.run_with_cache(q1_tokens)
    q1_right = (q1_logits[:, -1].argmax(-1) == answer_tokens).nonzero().squeeze(-1)
    q1_median = answer_ranks(logit_lens(model, q1_cache)[:, q1_right], answer_tokens[q1_right]).float().median(dim=1).values
    q1_first = (q1_median == 1).nonzero()[0].item()
    base_median = ranks.float().median(dim=1).values
    base_first = (base_median == 1).nonzero()[0].item()
    print(f"Q1: the model gets {len(q1_right)} of {len(countries)} right with this template (Part 1: {len(right)}).")
    print(f"    median rank of the correct capital by row: {[int(r) for r in q1_median]}")
    print(f"    first row with median rank 1: {LAYER_LABELS[q1_first]!r} (Part 1 template: {LAYER_LABELS[base_first]!r})")
    
    # Q2: a tuned lens trained for 30 steps, three seeds
    q2_seeds = []
    for seed in range(3):
        lens_30 = TunedLens(model.cfg.n_layers, model.cfg.d_model).to(device)
        train_tuned_lens(lens_30, model, train_tokens, n_steps=30, seed=seed)
        q2_seeds.append(evaluate_lenses(model, lens_30, test_tokens)["tuned_agree"])
    q2_mean = [sum(curve[layer] for curve in q2_seeds) / len(q2_seeds) for layer in range(model.cfg.n_layers)]
    print(f"Q2: layer-6 agreement with 30 steps: {[round(curve[6], 3) for curve in q2_seeds]} (mean {q2_mean[6]:.3f}); "
          f"300 steps: {results['tuned_agree'][6]:.3f}; logit lens: {results['logit_agree'][6]:.3f}")
    
    # Q3: a one-shot inspection prompt, source layer 4 -> target layer 1
    one_shot_target = model.to_tokens("The capital of France is Paris. The capital of x is")
    one_shot_pos = one_shot_target.shape[1] - 2
    q3_logits = run_patchscope(model, source_vectors[4], one_shot_target, one_shot_pos, target_layer=1)
    q3_accuracy = (q3_logits.argmax(-1) == answer_tokens).float().mean().item()
    q3_text_only = (model(one_shot_target)[:, -1].argmax(-1) == answer_tokens).float().mean().item()
    print(f"Q3: one-shot Patchscope {q3_accuracy:.3f} ({round(q3_accuracy * 48)}/48); three-shot {controls['patchscope']:.3f}; one-shot text only {q3_text_only:.3f}")
    
    # Q4: layer-9 queries against a layer-3 bank (next-token agreement with the model's own prediction)
    q4_match = layer_alignment_grid(queries[9:10], bank[3:4], query_top, bank_next)[0, 0].item()
    print(f"Q4: layer-9 queries, layer-3 bank: {q4_match:.3f}. Same layer: 3 -> 3: {lens_table['contextual lens'][3]:.3f}, 9 -> 9: {lens_table['contextual lens'][9]:.3f}. Bigram floor {bigram_floor:.3f}")
    
    # Q5: the plain logit lens on Pythia-160M (a zero-initialised tuned lens is the logit lens)
    PYTHIA_REVISION = "50f5173d932e8e61f858120bcb800b97af589f46"
    hf_pythia = AutoModelForCausalLM.from_pretrained("EleutherAI/pythia-160m", revision=PYTHIA_REVISION)
    pythia_tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-160m", revision=PYTHIA_REVISION)
    pythia = HookedTransformer.from_pretrained("pythia-160m", hf_model=hf_pythia, tokenizer=pythia_tokenizer, device=device)
    pythia.requires_grad_(False)
    pythia_tokens = utils.load_tinystories_tokens(pythia, n_sequences=200, seq_len=64)
    pythia_logit = evaluate_lenses(pythia, TunedLens(pythia.cfg.n_layers, pythia.cfg.d_model).to(device), pythia_tokens)["logit_agree"]
    print(f"Q5: Pythia-160M logit lens agreement by layer: {[round(x, 3) for x in pythia_logit]}")
    print(f"    layer 6: Pythia {pythia_logit[6]:.3f}, GPT-2 Small {results['logit_agree'][6]:.3f}")
    del pythia, hf_pythia
    if device.type == "cuda":
        t.cuda.empty_cache()
    
    utils.plot_quiz_runs(
        q1={"base": base_median.tolist(), "new": q1_median.tolist()},
        q2={"logit": results["logit_agree"], "tuned_300": results["tuned_agree"], "tuned_30": q2_mean, "tuned_30_seeds": q2_seeds},
        q3={"3 examples": controls["patchscope"], "1 example": q3_accuracy, "1 example, text only": q3_text_only},
        q4={"layer 3 -> 3": lens_table["contextual lens"][3], "layer 9 -> 9": lens_table["contextual lens"][9], "layer 9 -> 3": q4_match, "bigram floor": bigram_floor},
        q5={"gpt2": results["logit_agree"], "pythia": pythia_logit},
    )
    
    # The written key must match what these runs produced.
    EDGES = {"Q1": [7.5, 8.5, 9.5, 10.5], "Q2": [0.3, 0.4, 0.5, 0.6], "Q3": [0.2, 0.4, 0.6, 0.8], "Q4": [0.2, 0.3, 0.4, 0.5], "Q5": [0.05, 0.1, 0.2, 0.4]}
    measured = {"Q1": q1_first, "Q2": q2_mean[6], "Q3": q3_accuracy, "Q4": q4_match, "Q5": pythia_logit[6]}
    for name, value in measured.items():
        letter = interval_of(value, EDGES[name])
        assert quiz.questions[name].key == letter, f"{name}: the key says {quiz.questions[name].key}, but {value:.3f} falls in interval {letter}"
    print("The key matches the runs:", {name: quiz.questions[name].key for name in measured})

# %%

@t.inference_mode()
def ablate_blocks_recall(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    answer_tokens: Int[Tensor, " batch"],
    blocks: list[int],
    pos: int = -2,
) -> float:
    """
    Top-1 accuracy at the final position after replacing the MLP output of each block in `blocks`, at position `pos`,
    by its mean over the batch (the mean is taken from a clean run).
    """
    names = [f"blocks.{block}.hook_mlp_out" for block in blocks]
    _, clean_cache = model.run_with_cache(tokens, names_filter=lambda name: name in names)

    def make_hook(name: str):
        mean = clean_cache[name][:, pos].mean(dim=0)

        def hook(mlp_out: Float[Tensor, "batch seq d_model"], hook) -> Float[Tensor, "batch seq d_model"]:
            mlp_out[:, pos] = mean
            return mlp_out

        return hook

    logits = model.run_with_hooks(tokens, fwd_hooks=[(name, make_hook(name)) for name in names])
    return (logits[:, -1].argmax(dim=-1) == answer_tokens.to(logits.device)).float().mean().item()


if MAIN:
    tests.test_ablate_blocks_recall(ablate_blocks_recall, model)

# %%

if MAIN:
    necessity = {
        "no ablation": ablate_blocks_recall(model, tokens, answer_tokens, blocks=[]),
        "MLPs of blocks 1-5": ablate_blocks_recall(model, tokens, answer_tokens, blocks=list(range(1, 6))),
        "MLPs of blocks 6-10": ablate_blocks_recall(model, tokens, answer_tokens, blocks=list(range(6, 11))),
    }
    for name, accuracy in necessity.items():
        print(f"{name:<22} capital accuracy {accuracy:.1%}")

# %%

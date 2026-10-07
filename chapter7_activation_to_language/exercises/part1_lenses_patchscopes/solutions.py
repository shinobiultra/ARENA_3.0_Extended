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
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
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
    n_layers = model.cfg.n_layers
    metrics = {key: t.zeros(n_layers) for key in ["logit_kl", "tuned_kl", "logit_agree", "tuned_agree"]}
    n_batches = 0
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
                metrics[f"{name}_kl"][layer] += (target.exp() * (target - lp)).sum(dim=-1).mean().item()
                metrics[f"{name}_agree"][layer] += (lp.argmax(dim=-1) == target_top).float().mean().item()
        n_batches += 1
    return {key: (value / n_batches).tolist() for key, value in metrics.items()}

# %%

if MAIN:
    results = evaluate_lenses(model, tuned_lens, test_tokens)
    display(pd.DataFrame(results).rename_axis("layer").round(3).T)
    utils.plot_lens_comparison(
        results["logit_kl"],
        results["tuned_kl"],
        results["logit_agree"],
        results["tuned_agree"],
        bigram_agree=results["tuned_agree"][0],
    )

# %%

@t.inference_mode()
def shifted_translator_agreement(
    model: HookedTransformer, lens: TunedLens, tokens: Int[Tensor, "n_seq seq"], shift: int, batch_size: int = 20
) -> dict[int, float]:
    n_layers = model.cfg.n_layers
    layers = [layer for layer in range(n_layers) if 0 <= layer + shift < n_layers]
    totals = {layer: 0.0 for layer in layers}
    n_batches = 0
    for start in range(0, len(tokens), batch_size):
        resid_stack, final_logprobs = get_resid_and_logprobs(model, tokens[start : start + batch_size])
        target_top = final_logprobs[:, 1:].argmax(dim=-1)
        for layer in layers:
            lp = lens.logits(model, resid_stack[layer, :, 1:], layer + shift)
            totals[layer] += (lp.argmax(dim=-1) == target_top).float().mean().item()
        n_batches += 1
    return {layer: total / n_batches for layer, total in totals.items()}

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
    tests.test_patchscope_accuracy_grid(patchscope_accuracy_grid, model)

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
    controls = patchscope_controls(model, source_prompts, 4, target_tokens, target_pos, 1, answer_tokens)
    utils.plot_bars(controls, "Patchscope (source layer 4 → target layer 1) vs controls")

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

# %%

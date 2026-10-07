# %%


import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
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
section = "part5_predictive_concept_decoders"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part5_predictive_concept_decoders.tests as tests
import part5_predictive_concept_decoders.utils as utils

MAIN = __name__ == "__main__"

# %%

D_MODEL = 48
N_CONCEPTS = 6  # quiet concepts: they determine behaviour
N_LOUD = 8  # loud features: high variance, no effect on behaviour
N_QUESTIONS = 4
N_ANSWERS = 8
QUESTION_DEPENDS_ON = [[0, 1], [2, 3], [4, 5], [0, 3, 5]]
TRUE_LABELS = [f"c{i}" for i in range(N_CONCEPTS)] + [f"loud{j}" for j in range(N_LOUD)]


@dataclass
class PlantedWorld:
    concept_dirs: Float[Tensor, "n_concepts d_model"]
    loud_dirs: Float[Tensor, "n_loud d_model"]
    W_behaviour: Float[Tensor, "n_questions n_answers n_concepts"]
    b_behaviour: Float[Tensor, "n_questions n_answers"]


def make_planted_world(seed: int = 0) -> PlantedWorld:
    """Random orthonormal concept/loud directions, and a random sparse linear behaviour readout."""
    generator = t.Generator().manual_seed(seed)
    basis, _ = t.linalg.qr(t.randn(D_MODEL, D_MODEL, generator=generator))
    directions = basis[:, : N_CONCEPTS + N_LOUD].T.contiguous()
    W_behaviour = t.zeros(N_QUESTIONS, N_ANSWERS, N_CONCEPTS)
    for q, concepts in enumerate(QUESTION_DEPENDS_ON):
        W_behaviour[q][:, concepts] = 3.0 * t.randn(N_ANSWERS, len(concepts), generator=generator)
    b_behaviour = t.randn(N_QUESTIONS, N_ANSWERS, generator=generator)
    return PlantedWorld(directions[:N_CONCEPTS], directions[N_CONCEPTS:], W_behaviour, b_behaviour)


def sample_planted(
    world: PlantedWorld,
    n: int,
    seed: int,
    p_concept: float = 0.25,
    p_loud: float = 0.5,
    loud_scale: tuple[float, float] = (2.0, 4.0),
    noise: float = 0.05,
) -> tuple[Float[Tensor, "n d_model"], Float[Tensor, "n n_concepts"], Float[Tensor, "n n_loud"]]:
    """Returns (activations, binary concept labels, loud feature magnitudes)."""
    generator = t.Generator().manual_seed(seed)
    concepts = (t.rand(n, N_CONCEPTS, generator=generator) < p_concept).float()
    lo, hi = loud_scale
    loud_on = (t.rand(n, N_LOUD, generator=generator) < p_loud).float()
    loud = loud_on * (lo + (hi - lo) * t.rand(n, N_LOUD, generator=generator))
    acts = concepts @ world.concept_dirs + loud @ world.loud_dirs + noise * t.randn(n, D_MODEL, generator=generator)
    return acts, concepts, loud


def kl(p_log: Tensor, q_log: Tensor) -> float:
    """Mean KL(p || q) between log-prob tensors, summed over the last dimension and averaged over the rest."""
    return (p_log.exp() * (p_log - q_log)).sum(-1).mean().item()

# %%

def planted_behaviour(
    world: PlantedWorld, acts: Float[Tensor, "batch d_model"]
) -> Float[Tensor, "batch n_questions n_answers"]:
    """The planted model's answer log-probs for every question: log_softmax(W[q] @ (U a) + b[q])."""
    concept_proj = acts @ world.concept_dirs.to(acts.device).T  # (batch, n_concepts)
    logits = t.einsum("qvc,bc->bqv", world.W_behaviour.to(acts.device), concept_proj)
    return (logits + world.b_behaviour.to(acts.device)).log_softmax(dim=-1)


if MAIN:
    tests.test_planted_behaviour(planted_behaviour, make_planted_world)

# %%

if MAIN:
    world = make_planted_world(seed=0)
    train_acts, train_concepts, train_loud = sample_planted(world, 20_000, seed=1)
    test_acts, test_concepts, test_loud = sample_planted(world, 4_000, seed=2)
    train_behaviour = planted_behaviour(world, train_acts)
    test_behaviour = planted_behaviour(world, test_acts)
    
    total_var = (train_acts - train_acts.mean(0)).pow(2).sum(-1).mean()
    concept_var = (train_concepts @ world.concept_dirs).var(0).sum()
    loud_var = (train_loud @ world.loud_dirs).var(0).sum()
    print(f"Share of activation variance from the 6 quiet concepts: {concept_var / total_var:.1%}")
    print(f"Share of activation variance from the 8 loud features: {loud_var / total_var:.1%}\n")
    
    
    no_loud = planted_behaviour(world, train_acts - train_loud @ world.loud_dirs)
    no_concepts = planted_behaviour(world, train_acts - train_concepts @ world.concept_dirs)
    print(f"KL(behaviour || behaviour with loud features deleted): {kl(train_behaviour, no_loud):.4f} nats")
    print(f"KL(behaviour || behaviour with concepts deleted):      {kl(train_behaviour, no_concepts):.4f} nats")
    
    train_acts, test_acts = train_acts.to(device), test_acts.to(device)
    train_behaviour, test_behaviour = train_behaviour.to(device), test_behaviour.to(device)

# %%

def topk_relu(pre: Float[Tensor, "... m"], k: int) -> Float[Tensor, "... m"]:
    """Keeps the k largest entries of each row (then applies ReLU to them); zeroes everything else."""
    values, indices = pre.topk(k, dim=-1)
    return t.zeros_like(pre).scatter(-1, indices, F.relu(values))


if MAIN:
    tests.test_topk_relu(topk_relu)

# %%

class PCD(nn.Module):
    def __init__(self, d_model: int, n_latents: int, k: int, n_questions: int, n_answers: int, d_hidden: int = 128):
        super().__init__()
        self.k = k
        W_enc = t.randn(d_model, n_latents)
        self.W_enc = nn.Parameter(W_enc / W_enc.norm(dim=0, keepdim=True))  # unit-norm reading directions
        self.b_enc = nn.Parameter(t.zeros(n_latents))
        self.b_pre = nn.Parameter(t.zeros(d_model))
        self.W_emb = nn.Parameter(t.randn(n_latents, d_hidden) / n_latents**0.5)  # concept embeddings
        self.question_emb = nn.Parameter(0.1 * t.randn(n_questions, d_hidden))
        self.mlp = nn.Sequential(nn.GELU(), nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Linear(d_hidden, n_answers))

    def encode(self, acts: Float[Tensor, "batch d_model"]) -> Float[Tensor, "batch n_latents"]:
        return topk_relu((acts - self.b_pre) @ self.W_enc + self.b_enc, self.k)

    def decode(self, z: Float[Tensor, "batch n_latents"]) -> Float[Tensor, "batch n_questions n_answers"]:
        soft_token = z @ self.W_emb  # (batch, d_hidden)
        return self.mlp(soft_token[:, None, :] + self.question_emb[None, :, :])

    def forward(self, acts: Float[Tensor, "batch d_model"]) -> Float[Tensor, "batch n_questions n_answers"]:
        return self.decode(self.encode(acts))


if MAIN:
    tests.test_pcd_module(PCD)

# %%

def pcd_loss(
    model: nn.Module, acts: Float[Tensor, "batch d_model"], target_logprobs: Float[Tensor, "batch n_questions n_answers"]
) -> Float[Tensor, ""]:
    """Mean over batch and questions of KL(target || model(acts)), summed over answers."""
    pred_logprobs = model(acts).log_softmax(dim=-1)
    return (target_logprobs.exp() * (target_logprobs - pred_logprobs)).sum(dim=-1).mean()


if MAIN:
    tests.test_pcd_loss(pcd_loss, PCD)

# %%

def train_bottleneck(
    model: nn.Module,
    loss_fn: Callable,
    acts: Float[Tensor, "n d_model"],
    targets: Tensor | None = None,
    n_steps: int = 1500,
    batch_size: int = 256,
    lr: float = 3e-3,
    seed: int = 0,
) -> list[float]:
    """Trains `model` in place with Adam on random minibatches; returns the loss at every step."""
    generator = t.Generator().manual_seed(seed)
    optimizer = t.optim.Adam(model.parameters(), lr=lr)
    losses = []
    for _ in tqdm(range(n_steps), leave=False):
        idx = t.randint(0, len(acts), (batch_size,), generator=generator).to(acts.device)
        loss = loss_fn(model, acts[idx], None if targets is None else targets[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


if MAIN:
    tests.test_train_bottleneck(train_bottleneck, PCD, pcd_loss)

# %%

if MAIN:
    N_LATENTS, K = 12, 4
    
    t.manual_seed(0)
    pcd = PCD(D_MODEL, N_LATENTS, K, N_QUESTIONS, N_ANSWERS).to(device)
    t.manual_seed(0)
    untrained_pcd = PCD(D_MODEL, N_LATENTS, K, N_QUESTIONS, N_ANSWERS).to(device)  # control: same init, never trained
    pcd_losses = train_bottleneck(pcd, pcd_loss, train_acts, train_behaviour)
    
    marginal = train_behaviour.exp().mean(0, keepdim=True).log()  # always predict the average answer distribution
    marginal_kl = kl(test_behaviour, marginal.expand_as(test_behaviour))
    utils.plot_training_curves(
        {"PCD (train batches)": pcd_losses},
        title="Training a PCD on the planted world",
        ylabel="KL(model || PCD)  [nats, log scale]",
        baselines={"marginal baseline (held-out)": marginal_kl},
    )

# %%

if MAIN:
    with t.no_grad():
        print(f"Held-out KL, trained PCD:   {pcd_loss(pcd, test_acts, test_behaviour).item():.4f} nats")
        print(f"Held-out KL, untrained PCD: {pcd_loss(untrained_pcd, test_acts, test_behaviour).item():.4f} nats")
        print(f"Held-out KL, marginal:      {marginal_kl:.4f} nats\n")
        z_test = pcd.encode(test_acts)
    print(f"Active concepts per example: {(z_test > 0).sum(-1).float().mean():.2f} on average (k = {K})")
    print("Firing frequency of each latent:", [round(f, 2) for f in (z_test > 0).float().mean(0).tolist()])

# %%

def latent_alignment(
    W_enc: Float[Tensor, "d_model n_latents"], true_dirs: Float[Tensor, "n_true d_model"]
) -> Float[Tensor, "n_true n_latents"]:
    """Cosine similarity between each ground-truth direction and each latent's encoder direction."""
    return F.normalize(true_dirs, dim=-1) @ F.normalize(W_enc, dim=0)


if MAIN:
    tests.test_latent_alignment(latent_alignment)

# %%

if MAIN:
    true_dirs = t.cat([world.concept_dirs, world.loud_dirs]).to(device)
    utils.plot_alignment(
        {
            "untrained PCD (control)": latent_alignment(untrained_pcd.W_enc.detach(), true_dirs),
            "trained PCD": latent_alignment(pcd.W_enc.detach(), true_dirs),
        },
        TRUE_LABELS,
        N_CONCEPTS,
        title="Which planted directions does each PCD latent read?",
    )

# %%

class TopKSAE(nn.Module):
    def __init__(self, d_model: int, n_latents: int, k: int):
        super().__init__()
        self.k = k
        W = t.randn(d_model, n_latents)
        W = W / W.norm(dim=0, keepdim=True)
        self.W_enc = nn.Parameter(W.clone())
        self.W_dec = nn.Parameter(W.T.clone())
        self.b_enc = nn.Parameter(t.zeros(n_latents))
        self.b_pre = nn.Parameter(t.zeros(d_model))

    def encode(self, acts: Float[Tensor, "batch d_model"]) -> Float[Tensor, "batch n_latents"]:
        return topk_relu((acts - self.b_pre) @ self.W_enc + self.b_enc, self.k)

    def forward(self, acts: Float[Tensor, "batch d_model"]) -> Float[Tensor, "batch d_model"]:
        return self.encode(acts) @ self.W_dec + self.b_pre


def sae_loss(model: nn.Module, acts: Float[Tensor, "batch d_model"], targets: None = None) -> Float[Tensor, ""]:
    """Squared reconstruction error, summed over d_model and averaged over the batch."""
    return (model(acts) - acts).pow(2).sum(dim=-1).mean()


if MAIN:
    tests.test_topk_sae(TopKSAE, sae_loss)

# %%

if MAIN:
    t.manual_seed(0)
    sae = TopKSAE(D_MODEL, N_LATENTS, K).to(device)
    sae_losses = train_bottleneck(sae, sae_loss, train_acts)
    
    with t.no_grad():
        fvu = (sae(test_acts) - test_acts).pow(2).sum() / (test_acts - test_acts.mean(0)).pow(2).sum()
    print(f"SAE fraction of variance unexplained on held-out activations: {fvu:.3f}")

# %%

if MAIN:
    utils.plot_alignment(
        {
            "PCD (trained to predict behaviour)": latent_alignment(pcd.W_enc.detach(), true_dirs),
            "TopK SAE (trained to reconstruct)": latent_alignment(sae.W_enc.detach(), true_dirs),
        },
        TRUE_LABELS,
        N_CONCEPTS,
        title=f"Same bottleneck (m = {N_LATENTS}, k = {K}), different objective",
    )

# %%

class FrozenEncoderPCD(PCD):
    """A PCD whose encoder is replaced by another model's frozen `encode` method (e.g. an SAE's)."""

    def __init__(self, frozen_encoder: nn.Module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.frozen_encoder = [frozen_encoder]  # stored in a list, so its parameters aren't registered or trained

    def encode(self, acts: Float[Tensor, "batch d_model"]) -> Float[Tensor, "batch n_latents"]:
        with t.no_grad():
            return self.frozen_encoder[0].encode(acts)

# %%

if MAIN:
    t.manual_seed(0)
    sae_decoder = FrozenEncoderPCD(sae, D_MODEL, N_LATENTS, K, N_QUESTIONS, N_ANSWERS).to(device)
    train_bottleneck(sae_decoder, pcd_loss, train_acts, train_behaviour)
    
    t.manual_seed(0)
    shuffled_pcd = PCD(D_MODEL, N_LATENTS, K, N_QUESTIONS, N_ANSWERS).to(device)
    perm = t.randperm(len(train_acts), generator=t.Generator().manual_seed(0)).to(device)
    train_bottleneck(shuffled_pcd, pcd_loss, train_acts, train_behaviour[perm])  # activation i paired with behaviour perm[i]
    
    with t.no_grad():
        planted_kls = {
            "PCD": pcd_loss(pcd, test_acts, test_behaviour).item(),
            "SAE code + trained decoder": pcd_loss(sae_decoder, test_acts, test_behaviour).item(),
            "PCD on shuffled pairs": pcd_loss(shuffled_pcd, test_acts, test_behaviour).item(),
            "untrained PCD": pcd_loss(untrained_pcd, test_acts, test_behaviour).item(),
            "marginal baseline": marginal_kl,
        }
    utils.plot_bars(
        planted_kls, "Planted world: held-out behaviour prediction (lower is better)", "KL(model || decoder)  [nats]", logy=True
    )

# %%

def auroc(scores: Float[Tensor, " n"], labels: Float[Tensor, " n"]) -> float:
    """AUROC of `scores` for binary `labels`, via average ranks (ties count as half)."""
    scores, labels = scores.detach().float().cpu(), labels.detach().cpu().bool()
    order = scores.argsort()
    sorted_scores = scores[order]
    _, inverse, counts = t.unique_consecutive(sorted_scores, return_inverse=True, return_counts=True)
    # The tied block j occupies ranks (end_j - count_j + 1) ... end_j, whose average is end_j - (count_j - 1) / 2
    block_end = t.cumsum(counts, dim=0).float()
    avg_rank = block_end - (counts.float() - 1) / 2
    ranks = t.empty_like(scores)
    ranks[order] = avg_rank[inverse]
    n_pos, n_neg = labels.sum().item(), (~labels).sum().item()
    return ((ranks[labels].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


if MAIN:
    tests.test_auroc(auroc)

# %%

def best_latent_auroc(
    z_train: Float[Tensor, "n_train n_latents"],
    y_train: Float[Tensor, " n_train"],
    z_test: Float[Tensor, "n_test n_latents"],
    y_test: Float[Tensor, " n_test"],
) -> tuple[int, float]:
    """Index of the latent with the best train AUROC for the label, and that latent's test AUROC."""
    train_aurocs = [auroc(z_train[:, j], y_train) for j in range(z_train.shape[1])]
    best = max(range(len(train_aurocs)), key=lambda j: train_aurocs[j])
    return best, auroc(z_test[:, best], y_test)


if MAIN:
    tests.test_best_latent_auroc(best_latent_auroc)

# %%

def readout_accuracy(
    X_train: Float[Tensor, "n_train d"],
    y_train: Float[Tensor, " n_train"],
    X_test: Float[Tensor, "n_test d"],
    y_test: Float[Tensor, " n_test"],
    l2: float = 1.0,
) -> float:
    """Held-out balanced accuracy of a ridge-regression linear classifier trained on (X_train, y_train)."""
    X_train, X_test = X_train.detach().float().cpu(), X_test.detach().float().cpu()
    mean, std = X_train.mean(0), X_train.std(0) + 1e-6
    X_train, X_test = (X_train - mean) / std, (X_test - mean) / std
    y = 2 * y_train.detach().float().cpu() - 1
    w = t.linalg.solve(X_train.T @ X_train + l2 * t.eye(X_train.shape[1]), X_train.T @ (y - y.mean()))
    pred = (X_test @ w + y.mean()) > 0
    truth = y_test.detach().cpu().bool()
    tpr = (pred & truth).sum() / truth.sum()
    tnr = (~pred & ~truth).sum() / (~truth).sum()
    return (0.5 * (tpr + tnr)).item()


if MAIN:
    tests.test_readout_accuracy(readout_accuracy)

# %%

def shuffled_label_readout(
    X_train: Float[Tensor, "n_train d"],
    y_train: Float[Tensor, " n_train"],
    X_test: Float[Tensor, "n_test d"],
    y_test: Float[Tensor, " n_test"],
    n_shuffles: int = 5,
    seed: int = 0,
) -> float:
    """Mean held-out balanced accuracy of readouts trained on permuted training labels."""
    generator = t.Generator().manual_seed(seed)
    accs = []
    for _ in range(n_shuffles):
        perm = t.randperm(len(y_train), generator=generator)
        accs.append(readout_accuracy(X_train, y_train.cpu()[perm], X_test, y_test))
    return sum(accs) / len(accs)


if MAIN:
    tests.test_shuffled_label_readout(shuffled_label_readout)

# %%

if MAIN:
    encoders = {"PCD": pcd, "SAE": sae, "untrained PCD": untrained_pcd, "PCD on shuffled pairs": shuffled_pcd}
    with t.no_grad():
        codes = {name: (enc.encode(train_acts).cpu(), enc.encode(test_acts).cpu()) for name, enc in encoders.items()}
    
    concept_names = [f"c{i}" for i in range(N_CONCEPTS)]
    auroc_table = pd.DataFrame(index=concept_names, columns=list(encoders), dtype=float)
    readout_table = pd.DataFrame(index=concept_names, dtype=float)
    for i, concept in enumerate(concept_names):
        y_tr, y_te = train_concepts[:, i], test_concepts[:, i]
        for name, (z_tr, z_te) in codes.items():
            auroc_table.loc[concept, name] = best_latent_auroc(z_tr[:5000], y_tr[:5000], z_te, y_te)[1]
            readout_table.loc[concept, name] = readout_accuracy(z_tr, y_tr, z_te, y_te)
        readout_table.loc[concept, "dense probe (raw activation)"] = readout_accuracy(train_acts, y_tr, test_acts, y_te)
        readout_table.loc[concept, "PCD, shuffled labels"] = shuffled_label_readout(codes["PCD"][0], y_tr, codes["PCD"][1], y_te)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    utils.plot_grouped_bars(auroc_table, "Best single latent (chosen on train): held-out AUROC", "AUROC", ylim=(0.3, 1.05), chance=0.5, ax=axes[0])
    utils.plot_grouped_bars(readout_table, "Linear readout from the whole code: held-out balanced accuracy", "balanced accuracy", ylim=(0.3, 1.05), chance=0.5, ax=axes[1])
    plt.tight_layout()
    plt.show()
    display(pd.concat({"best-latent AUROC": auroc_table, "readout balanced acc.": readout_table}, axis=1).round(2))

# %%

if MAIN:
    def subspace_energy(W_enc: Tensor, world: PlantedWorld, alive: Tensor) -> tuple[float, float]:
        """Mean (over live latents) squared norm of the unit encoder direction inside the concept / loud subspaces."""
        cos = latent_alignment(W_enc.detach(), t.cat([world.concept_dirs, world.loud_dirs]).to(W_enc.device))[:, alive]
        return cos[:N_CONCEPTS].pow(2).sum(0).mean().item(), cos[N_CONCEPTS:].pow(2).sum(0).mean().item()
    
    
    # Change these settings and rerun the cell
    PLAY_LATENTS, PLAY_K = N_LATENTS, K
    play_train, _, _ = sample_planted(world, 20_000, seed=1, loud_scale=(0.3, 0.6), p_concept=0.25)
    play_train = play_train.to(device)
    play_behaviour = planted_behaviour(world, play_train)
    
    for name, model in [("PCD", pcd), ("SAE", sae)]:
        with t.no_grad():
            alive = (model.encode(train_acts) > 0).any(0)
        quiet, loud = subspace_energy(model.W_enc, world, alive)
        print(f"original world, {name}: {alive.sum().item():2d} live latents | energy in concept subspace {quiet:.2f} | in loud subspace {loud:.2f}")
    
    for name, make_model, loss_fn, targets in [
        ("PCD", lambda: PCD(D_MODEL, PLAY_LATENTS, PLAY_K, N_QUESTIONS, N_ANSWERS), pcd_loss, play_behaviour),
        ("SAE", lambda: TopKSAE(D_MODEL, PLAY_LATENTS, PLAY_K), sae_loss, None),
    ]:
        t.manual_seed(0)
        model = make_model().to(device)
        train_bottleneck(model, loss_fn, play_train, targets, n_steps=800)
        with t.no_grad():
            alive = (model.encode(play_train) > 0).any(0)
        quiet, loud = subspace_energy(model.W_enc, world, alive)
        print(f"quiet loud features, {name}: {alive.sum().item():2d} live latents | energy in concept subspace {quiet:.2f} | in loud subspace {loud:.2f}")

# %%

if MAIN:
    gpt2 = HookedTransformer.from_pretrained("gpt2-small", device=device)
    gpt2.requires_grad_(False)
    
    READ_LAYER = 10
    PROBES = ["", " After that,", " That made", " Everyone was so", " It was a"]
    
    train_prompts, train_labels = utils.make_template_prompts("train", 10_000, seed=0)
    test_prompts, test_labels = utils.make_template_prompts("test", 1_000, seed=1)
    display(
        pd.DataFrame(train_labels[:6].int().numpy(), columns=utils.CONCEPT_NAMES, index=train_prompts[:6])
    )
    print(f"{len(train_prompts)} train stories, {len(test_prompts)} test stories (with held-out words)")
    print("A test story:", test_prompts[0])

# %%

@t.no_grad()
def last_token_logprobs_and_acts(
    model: HookedTransformer, prompts: list[str], probe: str, read_layer: int
) -> tuple[Float[Tensor, "batch d_vocab"], Float[Tensor, "batch d_model"]]:
    """
    Runs the model on [p + probe for p in prompts]. Returns the next-token log-probs at the final token of
    each prompt+probe, and resid_pre[read_layer] at the final token of each prompt (the read position).
    """
    pad_id = model.tokenizer.pad_token_id
    read_pos = (model.to_tokens(prompts)[:, 1:] != pad_id).sum(1)  # index of the last real prompt token
    tokens = model.to_tokens([p + probe for p in prompts])
    last_pos = (tokens[:, 1:] != pad_id).sum(1)
    rows = t.arange(len(prompts), device=tokens.device)
    stored = {}

    def store_hook(resid: Float[Tensor, "batch seq d_model"], hook) -> None:
        stored["acts"] = resid[rows, read_pos].clone()

    with model.hooks(fwd_hooks=[(f"blocks.{read_layer}.hook_resid_pre", store_hook)]):
        resid_final = model(tokens, stop_at_layer=model.cfg.n_layers)
    logprobs = model.unembed(model.ln_final(resid_final[rows, last_pos])).log_softmax(dim=-1)
    return logprobs, stored["acts"]


if MAIN:
    tests.test_last_token_logprobs_and_acts(last_token_logprobs_and_acts, gpt2)

# %%

def restrict_to_answer_vocab(
    logprobs: Float[Tensor, "... d_vocab"], answer_ids: Int[Tensor, " n_answers"]
) -> Float[Tensor, "... n_answers_plus_1"]:
    """Log-probs of the answer tokens, plus a final 'other' bucket with the remaining probability."""
    probs = logprobs.exp()[..., answer_ids.to(logprobs.device)]
    other = (1 - probs.sum(dim=-1, keepdim=True)).clamp_min(1e-6)
    return t.cat([probs, other], dim=-1).clamp_min(1e-8).log()


if MAIN:
    tests.test_restrict_to_answer_vocab(restrict_to_answer_vocab)

# %%

if MAIN:
    calibration = [last_token_logprobs_and_acts(gpt2, train_prompts[:500], probe, READ_LAYER)[0].exp().mean(0) for probe in PROBES]
    answer_ids = []
    for mean_probs in calibration:
        answer_ids += [i for i in mean_probs.topk(30).indices.tolist() if i not in answer_ids]
    answer_ids = t.tensor(answer_ids, device=device)
    ANSWER_STRS = [gpt2.to_string(i) for i in answer_ids] + ["<other>"]
    N_GPT2_ANSWERS = len(ANSWER_STRS)
    
    
    @t.no_grad()
    def get_acts_and_behaviour(prompts: list[str], batch_size: int = 500) -> tuple[Tensor, Tensor]:
        """Read activations (n, d_model) and restricted behaviour log-probs (n, n_probes, n_answers + 1)."""
        acts, behaviour = [], []
        for start in tqdm(range(0, len(prompts), batch_size), leave=False):
            batch = prompts[start : start + batch_size]
            per_probe = []
            for probe in PROBES:
                logprobs, batch_acts = last_token_logprobs_and_acts(gpt2, batch, probe, READ_LAYER)
                per_probe.append(restrict_to_answer_vocab(logprobs, answer_ids))
            acts.append(batch_acts)  # the read activation doesn't depend on the probe
            behaviour.append(t.stack(per_probe, dim=1))
        return t.cat(acts), t.cat(behaviour)
    
    
    raw_train_acts, gpt2_train_behaviour = get_acts_and_behaviour(train_prompts)
    raw_test_acts, gpt2_test_behaviour = get_acts_and_behaviour(test_prompts)
    
    # Normalise activations: subtract the training mean and divide by a single scalar so the mean squared
    # norm per dimension is 1. (One scalar, so directions and ratios between directions are unchanged.)
    act_mean = raw_train_acts.mean(0)
    act_scale = (raw_train_acts - act_mean).norm(dim=-1).mean() / raw_train_acts.shape[1] ** 0.5
    gpt2_train_acts = (raw_train_acts - act_mean) / act_scale
    gpt2_test_acts = (raw_test_acts - act_mean) / act_scale
    
    print(f"Answer vocabulary: {len(answer_ids)} tokens + <other>")
    print(f"Activations: train {tuple(gpt2_train_acts.shape)}, test {tuple(gpt2_test_acts.shape)}")
    print(f"Behaviour:   train {tuple(gpt2_train_behaviour.shape)}, test {tuple(gpt2_test_behaviour.shape)}")
    mass = gpt2_train_behaviour[..., :-1].exp().sum(-1).mean(0)
    print("Mean probability inside the answer vocabulary, per question:", [round(m, 2) for m in mass.tolist()])

# %%

if MAIN:
    GPT2_LATENTS, GPT2_K, N_PROBES = 12, 3, len(PROBES)
    d_gpt2 = gpt2.cfg.d_model
    train_kwargs = dict(n_steps=2000, lr=1e-3)
    
    t.manual_seed(0)
    gpt2_pcd = PCD(d_gpt2, GPT2_LATENTS, GPT2_K, N_PROBES, N_GPT2_ANSWERS).to(device)
    gpt2_pcd_losses = train_bottleneck(gpt2_pcd, pcd_loss, gpt2_train_acts, gpt2_train_behaviour, **train_kwargs)
    
    t.manual_seed(0)
    gpt2_sae = TopKSAE(d_gpt2, GPT2_LATENTS, GPT2_K).to(device)
    train_bottleneck(gpt2_sae, sae_loss, gpt2_train_acts, **train_kwargs)
    t.manual_seed(0)
    gpt2_sae_decoder = FrozenEncoderPCD(gpt2_sae, d_gpt2, GPT2_LATENTS, GPT2_K, N_PROBES, N_GPT2_ANSWERS).to(device)
    train_bottleneck(gpt2_sae_decoder, pcd_loss, gpt2_train_acts, gpt2_train_behaviour, **train_kwargs)
    
    t.manual_seed(0)
    gpt2_dense = PCD(d_gpt2, d_gpt2, d_gpt2, N_PROBES, N_GPT2_ANSWERS).to(device)  # k = m = d_model: no bottleneck
    train_bottleneck(gpt2_dense, pcd_loss, gpt2_train_acts, gpt2_train_behaviour, **train_kwargs)
    
    t.manual_seed(0)
    gpt2_shuffled = PCD(d_gpt2, GPT2_LATENTS, GPT2_K, N_PROBES, N_GPT2_ANSWERS).to(device)
    gpt2_perm = t.randperm(len(gpt2_train_acts), generator=t.Generator().manual_seed(0)).to(device)
    train_bottleneck(gpt2_shuffled, pcd_loss, gpt2_train_acts, gpt2_train_behaviour[gpt2_perm], **train_kwargs)
    
    
    def per_question_kl(target: Tensor, pred_logits: Tensor) -> list[float]:
        """Mean KL(target || pred) for each question."""
        pred = pred_logits.log_softmax(-1)
        return (target.exp() * (target - pred)).sum(-1).mean(0).tolist()
    
    
    gpt2_marginal = gpt2_train_behaviour.exp().mean(0, keepdim=True).log()
    with t.no_grad():
        gpt2_kls = pd.DataFrame(
            {
                "PCD (k=3)": per_question_kl(gpt2_test_behaviour, gpt2_pcd(gpt2_test_acts)),
                "SAE code + decoder (k=3)": per_question_kl(gpt2_test_behaviour, gpt2_sae_decoder(gpt2_test_acts)),
                "no bottleneck": per_question_kl(gpt2_test_behaviour, gpt2_dense(gpt2_test_acts)),
                "PCD on shuffled pairs": per_question_kl(gpt2_test_behaviour, gpt2_shuffled(gpt2_test_acts)),
                "marginal baseline": per_question_kl(gpt2_test_behaviour, gpt2_marginal.expand_as(gpt2_test_behaviour)),
            },
            index=[repr(p) for p in PROBES],
        )
    gpt2_kls.loc["mean"] = gpt2_kls.mean()
    utils.plot_grouped_bars(gpt2_kls, "GPT-2: held-out KL(GPT-2 || decoder) per question (lower is better)", "KL [nats]")
    display(gpt2_kls.round(3))

# %%

def top_answers(logprobs: Float[Tensor, " n_answers"], n: int = 3) -> str:
    """Formats the n most likely answers of a restricted distribution, e.g. "' she' (0.41), ' the' (0.16)"."""
    probs = logprobs.exp()
    top = probs.topk(n)
    return ", ".join(f"{ANSWER_STRS[i]!r} ({p:.2f})" for p, i in zip(top.values.tolist(), top.indices.tolist()))


def describe(z_row: Tensor) -> str:
    """Formats the active concepts of one code, e.g. "#3=2.1, #7=0.4"."""
    active = z_row.nonzero().squeeze(-1).tolist()
    return ", ".join(f"#{j}={z_row[j]:.1f}" for j in sorted(active, key=lambda j: -z_row[j].item())) or "(none)"

# %%

if MAIN:
    with t.no_grad():
        gpt2_z_test = gpt2_pcd.encode(gpt2_test_acts)
        pcd_pred = gpt2_pcd(gpt2_test_acts).log_softmax(-1)
    for i in [0, 1, 2]:
        print(f"{test_prompts[i]}\n  active concepts: {describe(gpt2_z_test[i])}")
        for q, probe in enumerate(PROBES[1:4], start=1):
            print(f"  {probe!r:<19} GPT-2: {top_answers(gpt2_test_behaviour[i, q])}")
            print(f"  {'':<19} PCD:   {top_answers(pcd_pred[i, q])}")
        print()

# %%

def latent_dashboard(z: Float[Tensor, "n n_latents"], prompts: list[str], n_top: int = 3) -> pd.DataFrame:
    """One row per latent: firing frequency and the n_top most strongly activating prompts."""
    z = z.detach().cpu()
    rows = []
    for j in range(z.shape[1]):
        row = {"freq": (z[:, j] > 0).float().mean().item()}
        for rank, i in enumerate(z[:, j].topk(n_top).indices.tolist()):
            row[f"top {rank + 1}"] = prompts[i]
        rows.append(row)
    return pd.DataFrame(rows).rename_axis("latent")


if MAIN:
    tests.test_latent_dashboard(latent_dashboard)

# %%

if MAIN:
    with pd.option_context("display.max_colwidth", 70, "display.width", 250):
        display(latent_dashboard(gpt2_z_test, test_prompts, n_top=2).round(2))

# %%

def latent_group_aurocs(z: Float[Tensor, "n n_latents"], groups: dict[str, Float[Tensor, " n"]]) -> Float[Tensor, "n_latents n_groups"]:
    """AUROC of every latent's activation for every group label."""
    return t.tensor([[auroc(z[:, j], labels) for labels in groups.values()] for j in range(z.shape[1])])


if MAIN:
    tests.test_latent_group_aurocs(latent_group_aurocs, auroc)

# %%

def make_groups(labels: Float[Tensor, "n 4"]) -> dict[str, Tensor]:
    """Derived concept groups from the (female, plural, positive, outdoors) labels."""
    female, plural, positive, outdoors = labels.unbind(-1)
    return {
        "she": female * (1 - plural),
        "he": (1 - female) * (1 - plural),
        "they": plural,
        "positive": positive,
        "negative": 1 - positive,
        "outdoors": outdoors,
        "indoors": 1 - outdoors,
    }

# %%

if MAIN:
    train_groups, test_groups = make_groups(train_labels), make_groups(test_labels)
    with t.no_grad():
        gpt2_z_train = gpt2_pcd.encode(gpt2_train_acts)
        sae_z_train, sae_z_test = gpt2_sae.encode(gpt2_train_acts), gpt2_sae.encode(gpt2_test_acts)
    utils.plot_heatmaps(
        {
            "PCD latents": latent_group_aurocs(gpt2_z_test.cpu(), test_groups),
            "SAE latents": latent_group_aurocs(sae_z_test.cpu(), test_groups),
        },
        row_label="latent",
        col_labels=list(test_groups),
        title="GPT-2 layer 10: held-out AUROC of each latent for each concept group",
    )

# %%

if MAIN:
    t.manual_seed(0)
    gpt2_untrained = PCD(d_gpt2, GPT2_LATENTS, GPT2_K, N_PROBES, N_GPT2_ANSWERS).to(device)
    with t.no_grad():
        gpt2_codes = {
            "dense probe (768-d activation)": (gpt2_train_acts, gpt2_test_acts),
            "PCD (k=3)": (gpt2_z_train, gpt2_z_test),
            "SAE (k=3)": (sae_z_train, sae_z_test),
            "untrained PCD": (gpt2_untrained.encode(gpt2_train_acts), gpt2_untrained.encode(gpt2_test_acts)),
            "PCD on shuffled pairs": (gpt2_shuffled.encode(gpt2_train_acts), gpt2_shuffled.encode(gpt2_test_acts)),
        }
    
    gpt2_readout = pd.DataFrame(index=utils.CONCEPT_NAMES, dtype=float)
    for c, concept in enumerate(utils.CONCEPT_NAMES):
        for name, (X_tr, X_te) in gpt2_codes.items():
            gpt2_readout.loc[concept, name] = readout_accuracy(X_tr, train_labels[:, c], X_te, test_labels[:, c])
        gpt2_readout.loc[concept, "PCD, shuffled labels"] = shuffled_label_readout(gpt2_z_train, train_labels[:, c], gpt2_z_test, test_labels[:, c])
    
    singular = test_labels[:, 1] == 0
    gender_split = pd.DataFrame(index=["singular subjects", "plural subjects"], dtype=float)
    for name in ["dense probe (768-d activation)", "PCD (k=3)", "SAE (k=3)"]:
        X_tr, X_te = gpt2_codes[name]
        for row, mask in [("singular subjects", singular), ("plural subjects", ~singular)]:
            gender_split.loc[row, name] = readout_accuracy(X_tr, train_labels[:, 0], X_te[mask.to(X_te.device)], test_labels[mask, 0])
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 4), gridspec_kw={"width_ratios": [2.2, 1]})
    utils.plot_grouped_bars(gpt2_readout, "Held-out readout of labelled concepts (unseen words)", "balanced accuracy", ylim=(0.3, 1.05), chance=0.5, ax=axes[0])
    utils.plot_grouped_bars(gender_split, "'female' readout, split by subject number", "balanced accuracy", ylim=(0.3, 1.05), chance=0.5, ax=axes[1])
    plt.tight_layout()
    plt.show()
    display(gpt2_readout.round(2))
    display(gender_split.round(2))

# %%

if MAIN:
    group_aurocs = latent_group_aurocs(gpt2_z_test.cpu(), test_groups)
    LATENT_NAMES = [
        f"#{j} {list(test_groups)[group_aurocs[j].argmax()]}" if group_aurocs[j].max() > 0.75 else f"#{j} ?"
        for j in range(GPT2_LATENTS)
    ]
    print("Latent names:", LATENT_NAMES, "\n")
    
    my_stories = [
        "The nurse lost her keys at the station.",
        "The team celebrated the victory downtown.",
        "Sadly, the king won the lottery.",
    ]
    my_acts, my_behaviour = get_acts_and_behaviour(my_stories)
    with t.no_grad():
        my_z = gpt2_pcd.encode((my_acts - act_mean) / act_scale)
        my_pred = gpt2_pcd.decode(my_z).log_softmax(-1)
    for i, story in enumerate(my_stories):
        active = sorted(my_z[i].nonzero().squeeze(-1).tolist(), key=lambda j: -my_z[i, j].item())
        print(f"{story}\n  concepts: " + ", ".join(f"{LATENT_NAMES[j]} ({my_z[i, j]:.1f})" for j in active))
        for q in [1, 2, 3]:
            print(f"  {PROBES[q]!r:<19} GPT-2: {top_answers(my_behaviour[i, q])}")
            print(f"  {'':<19} PCD:   {top_answers(my_pred[i, q])}")
        print()

# %%

def concept_vector(
    acts: Float[Tensor, "n d_model"], z: Float[Tensor, "n n_latents"], j: int
) -> Float[Tensor, " d_model"]:
    """Mean activation where latent j is active, minus mean activation where it isn't."""
    on = z[:, j] > 0
    return acts[on].mean(0) - acts[~on].mean(0)


if MAIN:
    tests.test_concept_vector(concept_vector)

# %%

@t.no_grad()
def planted_effects(
    decoder: nn.Module,
    behaviour_fn: Callable[[Tensor], Tensor],
    acts: Float[Tensor, "batch d_model"],
    vec: Float[Tensor, " d_model"],
) -> tuple[Float[Tensor, " n"], Float[Tensor, " n"]]:
    """Flattened (predicted, actual) changes in answer log-probs when `vec` is subtracted from `acts`."""
    edited = acts - vec
    predicted = decoder(edited).log_softmax(-1) - decoder(acts).log_softmax(-1)
    actual = behaviour_fn(edited) - behaviour_fn(acts)
    return predicted.flatten(), actual.flatten()


if MAIN:
    tests.test_planted_effects(planted_effects)

# %%

if MAIN:
    behaviour_fn = lambda acts: planted_behaviour(world, acts)
    with t.no_grad():
        z_train_planted, z_test_planted = pcd.encode(train_acts), pcd.encode(test_acts)
    
    results = {name: ([], []) for name in ["PCD", "SAE code + decoder", "PCD on shuffled pairs", "PCD, random edit"]}
    for j in range(N_LATENTS):
        on = z_test_planted[:, j] > 0
        if on.sum() < 20 or (z_train_planted[:, j] > 0).all():
            continue  # skip latents that (almost) never fire or always fire: their mean difference is undefined/noisy
        v = concept_vector(train_acts, z_train_planted, j)
        r = F.normalize(t.randn(D_MODEL, generator=t.Generator().manual_seed(j)), dim=0).to(device) * v.norm()
        for name, decoder, vec in [
            ("PCD", pcd, v),
            ("SAE code + decoder", sae_decoder, v),
            ("PCD on shuffled pairs", shuffled_pcd, v),
            ("PCD, random edit", pcd, r),
        ]:
            predicted, actual = planted_effects(decoder, behaviour_fn, test_acts[on], vec)
            results[name][0].append(predicted)
            results[name][1].append(actual)
    
    points = {name: (t.cat(p), t.cat(a)) for name, (p, a) in results.items()}
    for name, (p, a) in points.items():
        print(f"{name:<22} mean |actual change| {a.abs().mean():.3f}   mean |predicted change| {p.abs().mean():.3f}")
    utils.plot_faithfulness_scatter(points, "Planted world: removing each PCD concept's effect vector")

# %%

def make_edits(
    pcd: PCD, acts_train: Float[Tensor, "n d_model"], z_train: Float[Tensor, "n n_latents"], j: int, seed: int = 0
) -> dict[str, Callable[[Tensor], Tensor]]:
    """Three ways of removing latent j from (normalised) activations of shape (..., d_model)."""
    off = z_train[:, j] == 0
    b_pre = pcd.b_pre.detach()
    w_hat = F.normalize(pcd.W_enc[:, j].detach(), dim=0)
    off_value = ((acts_train[off] - b_pre) @ w_hat).mean()
    v = concept_vector(acts_train, z_train, j)
    generator = t.Generator().manual_seed(seed)
    r = F.normalize(t.randn(v.shape, generator=generator), dim=0).to(v.device) * v.norm()

    def encoder_direction(a: Tensor) -> Tensor:
        return a + (off_value - (a - b_pre) @ w_hat)[..., None] * w_hat

    return {
        "encoder direction": encoder_direction,
        "effect vector": lambda a: a - v,
        "random vector": lambda a: a - r,
    }


if MAIN:
    tests.test_make_edits(make_edits, PCD, concept_vector)

# %%

def in_raw_space(edit: Callable[[Tensor], Tensor], mean: Tensor, scale: Tensor) -> Callable[[Tensor], Tensor]:
    """Turns an edit of normalised activations ((raw - mean) / scale) into an edit of raw activations."""
    return lambda raw: edit((raw - mean) / scale) * scale + mean

# %%

@t.no_grad()
def behaviour_under_edit(
    model: HookedTransformer,
    prompts: list[str],
    edit_raw: Callable[[Tensor], Tensor] | None,
    positions: str = "read",
    layer: int = 10,
    batch_size: int = 150,
) -> Float[Tensor, "n_prompts n_probes n_answers_plus_1"]:
    """Restricted next-token log-probs for every probe, with `edit_raw` applied to resid_pre[layer] at `positions`."""
    assert positions in ("read", "all")
    pad_id = model.tokenizer.pad_token_id
    out = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        read_pos = (model.to_tokens(batch)[:, 1:] != pad_id).sum(1)
        rows = t.arange(len(batch), device=read_pos.device)

        def edit_hook(resid: Float[Tensor, "batch seq d_model"], hook) -> Float[Tensor, "batch seq d_model"]:
            if positions == "all":
                return edit_raw(resid)
            resid[rows, read_pos] = edit_raw(resid[rows, read_pos])
            return resid

        hooks = [] if edit_raw is None else [(f"blocks.{layer}.hook_resid_pre", edit_hook)]
        per_probe = []
        for probe in PROBES:
            with model.hooks(fwd_hooks=hooks):
                logprobs, _ = last_token_logprobs_and_acts(model, batch, probe, layer)
            per_probe.append(restrict_to_answer_vocab(logprobs, answer_ids))
        out.append(t.stack(per_probe, dim=1))
    return t.cat(out)


if MAIN:
    tests.test_behaviour_under_edit(behaviour_under_edit, gpt2, test_prompts[:4], gpt2_test_behaviour[:4])

# %%

def faithfulness_summary(
    clean_actual: Tensor, edited_actual: Tensor, clean_pred: Tensor, edited_pred: Tensor, min_prob: float = 0.02
) -> dict[str, float]:
    """KL of actual and predicted behaviour changes, and the correlation of per-token log-prob changes."""
    def mean_kl(p: Tensor, q: Tensor) -> float:
        return (p.exp() * (p - q)).sum(-1).mean().item()

    mask = clean_actual[..., :-1].exp() > min_prob
    d_pred = (edited_pred - clean_pred)[..., :-1][mask]
    d_actual = (edited_actual - clean_actual)[..., :-1][mask]
    return {
        "actual KL": mean_kl(clean_actual, edited_actual),
        "predicted KL": mean_kl(clean_pred, edited_pred),
        "r": t.corrcoef(t.stack([d_pred, d_actual]))[0, 1].item(),
    }


if MAIN:
    tests.test_faithfulness_summary(faithfulness_summary)

# %%

if MAIN:
    CONDITIONS = [
        ("encoder direction", "read"),
        ("effect vector", "read"),
        ("effect vector", "all"),
        ("random vector", "all"),
    ]
    gpt2_mean, gpt2_scale = act_mean.to(device), act_scale.to(device)
    faith_rows, scatter_points = {}, {}
    for group in ["she", "he", "they"]:
        j, test_auc = best_latent_auroc(gpt2_z_train.cpu(), train_groups[group], gpt2_z_test.cpu(), test_groups[group])
        active = (gpt2_z_test[:, j] > 0).nonzero().squeeze(-1)[:150]
        prompts = [test_prompts[i] for i in active.tolist()]
        edits = make_edits(gpt2_pcd, gpt2_train_acts, gpt2_z_train, j)
        clean_actual = behaviour_under_edit(gpt2, prompts, None)
        with t.no_grad():
            clean_pred = gpt2_pcd(gpt2_test_acts[active]).log_softmax(-1)
        label = f"#{j} ({group})"
        print(f"latent {label}: held-out AUROC {test_auc:.2f}, active on {len(active)} of the stories used")
        for edit_name, positions in CONDITIONS:
            edited_actual = behaviour_under_edit(gpt2, prompts, in_raw_space(edits[edit_name], gpt2_mean, gpt2_scale), positions)
            with t.no_grad():
                edited_pred = gpt2_pcd(edits[edit_name](gpt2_test_acts[active])).log_softmax(-1)
            faith_rows[(label, f"{edit_name} @ {positions}")] = faithfulness_summary(clean_actual, edited_actual, clean_pred, edited_pred)
            if (edit_name, positions) == ("effect vector", "all"):
                mask = clean_actual[..., :-1].exp() > 0.02
                scatter_points[label] = ((edited_pred - clean_pred)[..., :-1][mask], (edited_actual - clean_actual)[..., :-1][mask])
    
    faith_df = pd.DataFrame(faith_rows).T
    utils.plot_faithfulness_bars(faith_df, "GPT-2: does removing a PCD concept change GPT-2 as much as the PCD predicts?")
    display(faith_df.round(3))

# %%

if MAIN:
    utils.plot_faithfulness_scatter(scatter_points, "GPT-2: effect vector removed at all positions (layer 10)")

# %%

if MAIN:
    story, group = "The nurse lost her keys at the station.", "she"
    j = best_latent_auroc(gpt2_z_train.cpu(), train_groups[group], gpt2_z_test.cpu(), test_groups[group])[0]
    v = concept_vector(gpt2_train_acts, gpt2_z_train, j)
    remove = lambda a: a - v
    
    story_acts = (get_acts_and_behaviour([story])[0] - act_mean) / act_scale
    clean = behaviour_under_edit(gpt2, [story], None)
    edited = behaviour_under_edit(gpt2, [story], in_raw_space(remove, gpt2_mean, gpt2_scale), "all")
    with t.no_grad():
        pred_clean, pred_edited = gpt2_pcd(story_acts).log_softmax(-1), gpt2_pcd(remove(story_acts)).log_softmax(-1)
    print(f"{story}   (removing latent #{j}, the '{group}' concept, at all positions)\n")
    for q in [1, 2]:
        print(f"{PROBES[q]!r}")
        print(f"  GPT-2 before: {top_answers(clean[0, q])}\n  GPT-2 after:  {top_answers(edited[0, q])}")
        print(f"  PCD before:   {top_answers(pred_clean[0, q])}\n  PCD after:    {top_answers(pred_edited[0, q])}")

# %%

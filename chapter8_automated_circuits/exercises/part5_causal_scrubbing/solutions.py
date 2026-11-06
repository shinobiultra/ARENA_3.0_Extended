# %%


import math
import statistics
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch as t
from jaxtyping import Float, Int
from torch import Tensor
from transformer_lens import HookedTransformer

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter8_automated_circuits"
section = "part5_causal_scrubbing"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part5_causal_scrubbing.tests as tests
import part5_causal_scrubbing.utils as utils

MAIN = __name__ == "__main__"

# %%

def sample_toy(n: int, generator: t.Generator) -> Int[Tensor, "n 3"]:
    """n inputs (a, b, c), each digit uniform in 0..9."""
    return t.randint(0, 10, (n, 3), generator=generator)


def toy_nodes(x: Int[Tensor, "n 3"]) -> dict[str, Tensor]:
    a, b, c = x.float().unbind(-1)
    return {"a": a, "b": b, "c": c, "s": a + b, "d": c - 5}


def toy_output(s: Tensor, d: Tensor) -> Tensor:
    return t.cos(math.pi * s) + 0.1 * t.tanh(d)


def toy_label(x: Int[Tensor, "n 3"]) -> Tensor:
    return (-1.0) ** (x[:, 0] + x[:, 1]).float()


def mse(out: Tensor, y: Tensor) -> float:
    return ((out - y) ** 2).mean().item()


toy_generator = t.Generator().manual_seed(0)
TOY_POOL = sample_toy(20_000, toy_generator)  # the dataset we resample from
toy_x = sample_toy(4000, toy_generator)  # the inputs we evaluate on
nodes = toy_nodes(toy_x)
toy_original = mse(toy_output(nodes["s"], nodes["d"]), toy_label(toy_x))
print(f"Original loss: {toy_original:.4f}")

# %%

def resample_matching(x: Tensor, feature: Callable[[Tensor], Tensor], pool: Tensor, generator: t.Generator) -> Tensor:
    """For each row of x, a uniformly random row of pool with the same feature value."""
    x_feat, pool_feat = feature(x), feature(pool)
    out = t.empty_like(x)
    for value in x_feat.unique():
        rows = (x_feat == value).nonzero().flatten()
        candidates = (pool_feat == value).nonzero().flatten()
        picks = candidates[t.randint(0, len(candidates), (len(rows),), generator=generator)]
        out[rows] = pool[picks]
    return out


def loss_recovered(scrubbed: float, original: float, random: float) -> float:
    return (random - scrubbed) / (random - original)


if MAIN:
    tests.test_resample_matching(resample_matching)
    tests.test_loss_recovered(loss_recovered)

# %%

NO_CONSTRAINT = lambda rows: t.zeros(len(rows), dtype=t.long)
PARITY_OF_SUM = lambda rows: (rows[:, 0] + rows[:, 1]) % 2
SAME_A = lambda rows: rows[:, 0]
PARITY_OF_A = lambda rows: rows[:, 0] % 2
PARITY_OF_B = lambda rows: rows[:, 1] % 2
SAME_C = lambda rows: rows[:, 2]


def scrubbed_toy_loss(x: Tensor, s_from: Tensor, d_from: Tensor) -> float:
    """Loss when node s is computed on the inputs s_from and node d on d_from (the label always comes from x)."""
    return mse(toy_output(toy_nodes(s_from)["s"], toy_nodes(d_from)["d"]), toy_label(x))


toy_random = scrubbed_toy_loss(
    toy_x,
    resample_matching(toy_x, NO_CONSTRAINT, TOY_POOL, toy_generator),
    resample_matching(toy_x, NO_CONSTRAINT, TOY_POOL, toy_generator),
)
print(f"Random loss: {toy_random:.4f}")

# %%

def resample(feature: Callable[[Tensor], Tensor]) -> Tensor:
    return resample_matching(toy_x, feature, TOY_POOL, toy_generator)


toy_hypotheses = {
    "d doesn't matter": (toy_x, resample(NO_CONSTRAINT)),
    "s only via its parity": (resample(PARITY_OF_SUM), resample(NO_CONSTRAINT)),
    "s only via a": (resample(SAME_A), resample(NO_CONSTRAINT)),
    "d only via c": (toy_x, resample(SAME_C)),
}
toy_scores = {}
for name, (s_from, d_from) in toy_hypotheses.items():
    scrubbed = scrubbed_toy_loss(toy_x, s_from, d_from)
    toy_scores[name] = (scrubbed, loss_recovered(scrubbed, toy_original, toy_random))

for name, (scrubbed, recovered) in toy_scores.items():
    print(f"{name:<24} scrubbed loss {scrubbed:.4f}   loss recovered {recovered:+.3f}")

# %%

d_random = toy_nodes(resample(NO_CONSTRAINT))["d"]
for name, b_feature in [("s = parity(a) + parity(b)", PARITY_OF_B), ("s = parity(a) + anything", NO_CONSTRAINT)]:
    a = toy_nodes(resample(PARITY_OF_A))["a"]
    b = toy_nodes(resample(b_feature))["b"]
    scrubbed = mse(toy_output(a + b, d_random), toy_label(toy_x))
    toy_scores[name] = (scrubbed, loss_recovered(scrubbed, toy_original, toy_random))

for name in ["s = parity(a) + parity(b)", "s = parity(a) + anything"]:
    print(f"{name:<28} scrubbed loss {toy_scores[name][0]:.4f}   loss recovered {toy_scores[name][1]:+.3f}")

# %%

s_real = nodes["s"]
zero_d = mse(toy_output(s_real, t.zeros_like(nodes["d"])), toy_label(toy_x))
mean_d = mse(toy_output(s_real, nodes["d"].mean().expand_as(nodes["d"])), toy_label(toy_x))
resampled_d = toy_scores["d doesn't matter"][0]
losses = {"original": toy_original, "resample d": resampled_d, "mean-ablate d": mean_d, "zero-ablate d": zero_d}
for name, value in losses.items():
    print(f"{name:<14} loss {value:.4f}   loss recovered {loss_recovered(value, toy_original, toy_random):+.3f}")
utils.plot_toy_losses(losses, f"Three ways to remove d, on the toy (for scale: the random loss is {toy_random:.2f})")

# %%

if MAIN:
    model = utils.load_attn_only_2l(device)
    BOS, HALF_LEN = model.tokenizer.bos_token_id, 20
    POS = slice(HALF_LEN + 1, 2 * HALF_LEN)
    N_HEADS = model.cfg.n_heads
    
    data_generator = t.Generator().manual_seed(0)
    tokens = utils.repeated_random_tokens(BOS, (64,), HALF_LEN, data_generator, device)
    SEQ = tokens.shape[1]
    
    
    def induction_loss(logits: Float[Tensor, "batch seq d_vocab"], tokens: Int[Tensor, "batch seq"]) -> float:
        """Mean next-token cross-entropy at the induction positions."""
        logprobs = logits[:, POS].log_softmax(dim=-1)
        return -logprobs.gather(-1, tokens[:, POS.start + 1 : POS.stop + 1, None]).mean().item()
    
    
    with t.no_grad():
        ORIGINAL = induction_loss(model(tokens), tokens)
    print(f"{tokens.shape[0]} prompts of length {SEQ}; original induction loss {ORIGINAL:.3f}")

# %%

@t.no_grad()
def layer0_outputs(model: HookedTransformer, tokens: Int[Tensor, "... seq"]) -> dict[str, Float[Tensor, "... seq d_model"]]:
    batch_dims, seq = tokens.shape[:-1], tokens.shape[-1]
    flat = tokens.reshape(-1, seq)
    _, cache = model.run_with_cache(flat, names_filter=lambda name: name in ("hook_embed", "blocks.0.attn.hook_z"), stop_at_layer=1)
    z = cache["blocks.0.attn.hook_z"]
    outputs = {"embed": cache["hook_embed"]}
    for head in range(model.cfg.n_heads):
        outputs[f"0.{head}"] = z[:, :, head] @ model.W_O[0, head]
    return {name: out.reshape(*batch_dims, seq, -1) for name, out in outputs.items()}


if MAIN:
    tests.test_layer0_outputs(layer0_outputs, model, tokens[:4])

# %%

def head1_forward(
    model: HookedTransformer,
    head: int,
    q_in: Float[Tensor, "batch seq d_model"],
    k_in: Float[Tensor, "batch seq d_model"],
    v_in: Float[Tensor, "batch seq d_model"],
) -> Float[Tensor, "batch seq d_model"]:
    seq = q_in.shape[1]
    pos = model.W_pos[:seq]
    q = (q_in + pos) @ model.W_Q[1, head] + model.b_Q[1, head]
    k = (k_in + pos) @ model.W_K[1, head] + model.b_K[1, head]
    v = v_in @ model.W_V[1, head] + model.b_V[1, head]
    scores = q @ k.transpose(-1, -2) / model.cfg.d_head**0.5
    causal = t.triu(t.ones(seq, seq, dtype=t.bool, device=q_in.device), diagonal=1)
    pattern = scores.masked_fill(causal, float("-inf")).softmax(dim=-1)
    return (pattern @ v) @ model.W_O[1, head]


if MAIN:
    tests.test_head1_forward(head1_forward, model, tokens[:8])

# %%

@t.no_grad()
def sender_restricted_input(
    model: HookedTransformer, tokens: Int[Tensor, "... seq"], senders: list[str] | None, generator: t.Generator
) -> Float[Tensor, "... seq d_model"]:
    real = layer0_outputs(model, tokens)
    if senders is None:
        return model.b_O[0] + sum(real.values())
    resample = utils.repeated_random_tokens(BOS, tokens.shape[:-1], HALF_LEN, generator, tokens.device)
    other = layer0_outputs(model, resample)
    return model.b_O[0] + sum(real[name] if name in senders else other[name] for name in real)


if MAIN:
    tests.test_sender_restricted_input(sender_restricted_input, layer0_outputs, model, tokens[:4], HALF_LEN)

# %%

@dataclass
class Claim:
    """What one input of a layer-1 head depends on."""

    senders: list[str] | None = None  # which senders it reads from the real input (None: all of them)
    offsets: list[int] | None = None  # which token positions, relative to the reader's position, the information comes from (None: any) - used from 3️⃣ on


@t.no_grad()
def layer1_outputs(model: HookedTransformer, tokens: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch seq head d_model"]:
    """The real outputs of all layer-1 heads (after W_O), on the given tokens."""
    _, cache = model.run_with_cache(tokens, names_filter="blocks.1.attn.hook_z")
    return t.einsum("bphe,hed->bphd", cache["blocks.1.attn.hook_z"], model.W_O[1])

# %%

def claimed_input(model: HookedTransformer, tokens: Tensor, claim: Claim, generator: t.Generator) -> Tensor:
    """The input of a layer-1 head under a claim. (In 3️⃣ you'll extend this to claims about positions.)"""
    assert claim.offsets is None, "Claims about positions come in section 3️⃣"
    return sender_restricted_input(model, tokens, claim.senders, generator)


@t.no_grad()
def scrubbed_loss(model: HookedTransformer, tokens: Int[Tensor, "batch seq"], hypothesis: dict, generator: t.Generator) -> float:
    kept = hypothesis.get("logits", [])
    real0 = layer0_outputs(model, tokens)
    other0 = layer0_outputs(model, utils.repeated_random_tokens(BOS, tokens.shape[:-1], HALF_LEN, generator, tokens.device))
    other1 = layer1_outputs(model, utils.repeated_random_tokens(BOS, tokens.shape[:-1], HALF_LEN, generator, tokens.device))
    resid = model.b_O[0] + model.b_O[1] + sum(real0[name] if name in kept else other0[name] for name in real0)
    for head in range(model.cfg.n_heads):
        if f"1.{head}" in kept:
            inputs = [claimed_input(model, tokens, hypothesis.get((head, c), Claim()), generator) for c in "qkv"]
            resid = resid + head1_forward(model, head, *inputs)
        else:
            resid = resid + other1[:, :, head]
    return induction_loss(resid @ model.W_U + model.b_U, tokens)


if MAIN:
    tests.test_scrubbed_loss(scrubbed_loss, model, tokens, ORIGINAL)

# %%

if MAIN:
    N_RESAMPLES = 5
    scrub_generator = t.Generator().manual_seed(1)
    
    
    def score(hypothesis: dict, n: int = N_RESAMPLES, data: Tensor | None = None) -> tuple[float, float]:
        """Mean and standard deviation of loss recovered over n resamples."""
        data = tokens if data is None else data
        values = [loss_recovered(scrubbed_loss(model, data, hypothesis, scrub_generator), ORIGINAL, RANDOM) for _ in range(n)]
        return statistics.mean(values), statistics.stdev(values)
    
    
    RANDOM = statistics.mean(scrubbed_loss(model, tokens, {}, scrub_generator) for _ in range(N_RESAMPLES))
    EVERYTHING = {"logits": ["embed"] + [f"{layer}.{h}" for layer in range(2) for h in range(N_HEADS)]}
    print(f"original loss {ORIGINAL:.3f}, random loss {RANDOM:.3f}")
    print(f"keep everything: loss recovered {score(EVERYTHING, n=2)[0]:.3f}")

# %%

if MAIN:
    INDUCTION = [4, 10]
    INDUCTION_LOGITS = ["embed", "1.4", "1.10"]
    
    
    def induction_hypothesis(q: Claim = Claim(), k: Claim = Claim(), v: Claim = Claim()) -> dict:
        """The logits read the direct path and the two induction heads; both heads' inputs follow the given claims."""
        hypothesis = {"logits": INDUCTION_LOGITS}
        for head in INDUCTION:
            hypothesis[(head, "q")], hypothesis[(head, "k")], hypothesis[(head, "v")] = q, k, v
        return hypothesis
    
    
    component_results = {
        "only the induction heads (+ direct path)": score(induction_hypothesis()),
        "[1.2] circuit: q<-embed, k<-0.7, v<-embed": score(induction_hypothesis(q=Claim(["embed"]), k=Claim(["0.7"]), v=Claim(["embed"]))),
        "[1.2] circuit + embed into the keys": score(induction_hypothesis(q=Claim(["embed"]), k=Claim(["embed", "0.7"]), v=Claim(["embed"]))),
    }
    for name, (mean, std) in component_results.items():
        print(f"{name:<44} loss recovered {mean:.3f} ± {std:.3f}")

# %%

if MAIN:
    single_claim_results = {
        "q reads only the embedding": score(induction_hypothesis(q=Claim(["embed"]))),
        "k reads only head 0.7": score(induction_hypothesis(k=Claim(["0.7"]))),
        "v reads only the embedding": score(induction_hypothesis(v=Claim(["embed"]))),
    }
    for name, (mean, std) in single_claim_results.items():
        print(f"{name:<28} loss recovered {mean:.3f} ± {std:.3f}")

# %%

def per_position_resample(
    tokens: Int[Tensor, "batch seq"], offsets: list[int], bos: int, half_len: int, generator: t.Generator
) -> Int[Tensor, "copy batch seq"]:
    batch, seq = tokens.shape
    resamples = utils.repeated_random_tokens(bos, (seq, batch), half_len, generator, tokens.device)
    for j in range(seq):
        for d in offsets:
            if j - d >= 0:
                resamples[j, :, j - d] = tokens[:, j - d]
    return resamples


if MAIN:
    tests.test_per_position_resample(per_position_resample, tokens[:8], BOS, HALF_LEN)

# %%

def claimed_input(model: HookedTransformer, tokens: Tensor, claim: Claim, generator: t.Generator) -> Tensor:
    """The input of a layer-1 head under a claim about which senders it reads and which positions they carry."""
    if claim.offsets is None:
        return sender_restricted_input(model, tokens, claim.senders, generator)
    copies = per_position_resample(tokens, claim.offsets, BOS, HALF_LEN, generator)
    inputs = sender_restricted_input(model, copies, claim.senders, generator)  # (copy, batch, seq, d_model)
    positions = t.arange(tokens.shape[1], device=tokens.device)
    return inputs[positions, :, positions].transpose(0, 1)

# %%

if MAIN:
    textbook = induction_hypothesis(q=Claim(offsets=[0]), k=Claim(offsets=[1]), v=Claim(offsets=[0]))
    info_results = {
        "textbook: q=[0], k=[1], v=[0]": score(textbook),
        "only: v depends on token j": score(induction_hypothesis(v=Claim(offsets=[0]))),
        "only: k depends on token j-1": score(induction_hypothesis(k=Claim(offsets=[1]))),
        "only: q depends on token i": score(induction_hypothesis(q=Claim(offsets=[0]))),
    }
    for name, (mean, std) in info_results.items():
        print(f"{name:<32} loss recovered {mean:.3f} ± {std:.3f}")

# %%

if MAIN:
    refined_results = {
        "k=[0,1]": score(induction_hypothesis(k=Claim(offsets=[0, 1]))),
        "k=[1,2]": score(induction_hypothesis(k=Claim(offsets=[1, 2]))),
        "k=[1,2,3]": score(induction_hypothesis(k=Claim(offsets=[1, 2, 3]))),
        "q=[0,1]": score(induction_hypothesis(q=Claim(offsets=[0, 1]))),
        "q=[0,1,2]": score(induction_hypothesis(q=Claim(offsets=[0, 1, 2]))),
        "refined: q=[0,1,2], k=[1,2,3], v=[0]": score(induction_hypothesis(q=Claim(offsets=[0, 1, 2]), k=Claim(offsets=[1, 2, 3]), v=Claim(offsets=[0]))),
        "refined+: q=[0..3], k=[0..4], v=[0]": score(induction_hypothesis(q=Claim(offsets=[0, 1, 2, 3]), k=Claim(offsets=[0, 1, 2, 3, 4]), v=Claim(offsets=[0]))),
        "control: k=[5]": score(induction_hypothesis(k=Claim(offsets=[5]))),
        "control: v=[1]": score(induction_hypothesis(v=Claim(offsets=[1]))),
    }
    for name, (mean, std) in refined_results.items():
        print(f"{name:<38} loss recovered {mean:.3f} ± {std:.3f}")

# %%

if MAIN:
    ceiling = component_results["only the induction heads (+ direct path)"][0]
    ladder = {
        "[1.2] circuit (components)": component_results["[1.2] circuit: q<-embed, k<-0.7, v<-embed"],
        "textbook: q=[0], k=[1], v=[0]": info_results["textbook: q=[0], k=[1], v=[0]"],
        "  only k=[1]": info_results["only: k depends on token j-1"],
        "  only q=[0]": info_results["only: q depends on token i"],
        "  only v=[0]": info_results["only: v depends on token j"],
        "keys: k=[1,2,3]": refined_results["k=[1,2,3]"],
        "queries: q=[0,1,2]": refined_results["q=[0,1,2]"],
        "refined: q=[0,1,2], k=[1,2,3], v=[0]": refined_results["refined: q=[0,1,2], k=[1,2,3], v=[0]"],
        "refined+: q=[0..3], k=[0..4], v=[0]": refined_results["refined+: q=[0..3], k=[0..4], v=[0]"],
        "control: k=[5]": refined_results["control: k=[5]"],
        "control: v=[1]": refined_results["control: v=[1]"],
    }
    colours = {name: "tab:red" for name in ladder if name.startswith("control")}
    colours |= {name: "tab:green" for name in ladder if name.startswith("refined")}
    colours |= {"textbook: q=[0], k=[1], v=[0]": "tab:orange", "[1.2] circuit (components)": "tab:gray"}
    utils.plot_hypothesis_ladder(
        ladder, "Causal scrubbing of induction hypotheses (2-layer model)",
        reference={"only the induction heads": ceiling}, colours=colours,
    )

# %%

if MAIN:
    ALL_BUT_04 = ["embed"] + [f"0.{h}" for h in range(N_HEADS) if h != 4]
    sender_results = {
        "q, k read only embed, 0.4 and 0.7": score(induction_hypothesis(q=Claim(["embed", "0.4", "0.7"]), k=Claim(["embed", "0.4", "0.7"]))),
        "q, k read every sender except 0.4": score(induction_hypothesis(q=Claim(ALL_BUT_04), k=Claim(ALL_BUT_04))),
        "q, k read every sender except 0.7": score(induction_hypothesis(q=Claim(["embed"] + [f"0.{h}" for h in range(N_HEADS) if h != 7]), k=Claim(["embed"] + [f"0.{h}" for h in range(N_HEADS) if h != 7]))),
    }
    for name, (mean, std) in sender_results.items():
        print(f"{name:<36} loss recovered {mean:.3f} ± {std:.3f}")

# %%

if MAIN:
    def unrepeated_random_tokens(bos, batch_shape, half_len, generator, device):
        """Random sequences of the same length with no repetition: not the task distribution."""
        toks = t.randint(1000, 10000, (*batch_shape, 2 * half_len + 1), generator=generator)
        toks[..., 0] = bos
        return toks.to(device)
    
    
    to_compare = {
        "only the induction heads": induction_hypothesis(),
        "[1.2] circuit (components)": induction_hypothesis(q=Claim(["embed"]), k=Claim(["0.7"]), v=Claim(["embed"])),
        "textbook": textbook,
        "refined": induction_hypothesis(q=Claim(offsets=[0, 1, 2]), k=Claim(offsets=[1, 2, 3]), v=Claim(offsets=[0])),
    }
    task_distribution = {name: score(h)[0] for name, h in to_compare.items()}
    original_sampler = utils.repeated_random_tokens
    utils.repeated_random_tokens = unrepeated_random_tokens
    try:
        RANDOM_TASK = RANDOM
        RANDOM = statistics.mean(scrubbed_loss(model, tokens, {}, scrub_generator) for _ in range(N_RESAMPLES))
        wrong_distribution = {name: score(h)[0] for name, h in to_compare.items()}
        print(f"random loss: task distribution {RANDOM_TASK:.3f}, unrepeated sequences {RANDOM:.3f}\n")
    finally:
        utils.repeated_random_tokens = original_sampler
        RANDOM = RANDOM_TASK
    
    print(f"{'hypothesis':<28} {'task distribution':>18} {'unrepeated':>11}")
    for name in to_compare:
        print(f"{name:<28} {task_distribution[name]:>18.3f} {wrong_distribution[name]:>11.3f}")

# %%

if MAIN:
    def per_position_induction_loss(logits: Tensor, tokens: Tensor) -> Tensor:
        logprobs = logits[:, POS].log_softmax(dim=-1)
        return -logprobs.gather(-1, tokens[:, POS.start + 1 : POS.stop + 1, None])[..., 0].mean(0)
    
    
    mean_loss = induction_loss
    induction_loss = per_position_induction_loss
    try:
        curves = {
            name: sum(scrubbed_loss(model, tokens, h, scrub_generator) for _ in range(N_RESAMPLES)) / N_RESAMPLES
            for name, h in [("full model", EVERYTHING), ("refined", to_compare["refined"]), ("textbook", textbook), ("everything resampled", {})]
        }
    finally:
        induction_loss = mean_loss
    
    positions = list(range(POS.start, POS.stop))
    utils.plot_per_position(curves, positions, "Loss at each induction position under scrubbing")
    for name, curve in curves.items():
        print(f"{name:<22} first 3 positions {curve[:3].mean():.2f}   last 3 positions {curve[-3:].mean():.2f}")

# %%

if MAIN:
    my_hypothesis = induction_hypothesis(q=Claim(offsets=[0, 1]), k=Claim(senders=["embed", "0.4", "0.7"], offsets=[1, 2, 3]), v=Claim(offsets=[0]))
    mean, std = score(my_hypothesis)
    print(f"my hypothesis: loss recovered {mean:.3f} ± {std:.3f}")

# %%

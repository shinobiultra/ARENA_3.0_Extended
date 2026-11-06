# %%


import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
chapter = "chapter17_training_dynamics"
section = "part2_induction_head_emergence"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))
import part2_induction_head_emergence.tests as tests
import part2_induction_head_emergence.utils as utils
MAIN = __name__ == "__main__"
np.set_printoptions(precision=4, suppress=True)

# %%

def repeat_tokens(period, repeats=3):
    if period < 2 or repeats < 2:
        raise ValueError("Need a nontrivial repeated block")
    return np.tile(np.arange(period), repeats)


if MAIN:
    tests.test_repeat_tokens(repeat_tokens)

# %%

def previous_attention(length):
    return np.eye(length, k=-1)


if MAIN:
    tests.test_previous_attention(previous_attention)

# %%

def prefix_keys(tokens, vocab, strength=1.0):
    one_hot = np.eye(vocab)[tokens]
    return strength * previous_attention(len(tokens)) @ one_hot


if MAIN:
    tests.test_prefix_keys(prefix_keys)

# %%

def causal_softmax(scores):
    mask = np.tril(np.ones_like(scores, dtype=bool))
    x = np.where(mask, scores, -np.inf)
    x = np.exp(x - x.max(axis=-1, keepdims=True))
    return x / x.sum(axis=-1, keepdims=True)


if MAIN:
    tests.test_causal_softmax(causal_softmax)

# %%

if MAIN:
    small_tokens=repeat_tokens(3,2)
    small_keys=prefix_keys(small_tokens,3)
    print("position token previous-key-token ablated-key-norm")
    for i,(token,key) in enumerate(zip(small_tokens,small_keys)):
        previous=int(key.argmax()) if key.sum() else "none"
        print(i,int(token),previous,0.)

# %%

def induction_pattern(tokens, vocab, strength=1.0, beta=12.0):
    q = np.eye(vocab)[tokens]
    k = prefix_keys(tokens, vocab, strength)
    return causal_softmax(beta * q @ k.T)


if MAIN:
    tests.test_induction_pattern(induction_pattern)

# %%

def prefix_score(attention, period):
    dest = np.arange(period, 2*period-1)
    return float(attention[dest, dest-period+1].mean())


if MAIN:
    tests.test_prefix_score(prefix_score)

# %%

def copy_logits(attention, tokens, vocab, scale=8.0):
    return scale * attention @ np.eye(vocab)[tokens]


if MAIN:
    tests.test_copy_logits(copy_logits)

# %%

def next_token_loss(logits, tokens):
    x = logits[:-1]
    m = x.max(axis=-1)
    log_z = m + np.log(np.exp(x-m[:,None]).sum(axis=-1))
    return log_z - x[np.arange(len(tokens)-1), tokens[1:]]


if MAIN:
    tests.test_next_token_loss(next_token_loss)

# %%

def icl_gain(losses, period):
    return float(losses[:period-1].mean() - losses[period:2*period-1].mean())


if MAIN:
    tests.test_icl_gain(icl_gain)

# %%

if MAIN:
    worked_tokens=repeat_tokens(4,3)
    worked_attention=induction_pattern(worked_tokens,4)
    worked_logits=copy_logits(worked_attention,worked_tokens,4)
    wrong_logits=copy_logits(worked_attention,(worked_tokens+1)%4,4)
    print("destination current expected clean-top1 wrong-value-top1")
    for i in range(4,7):
        print(i,int(worked_tokens[i]),int(worked_tokens[i+1]),int(worked_logits[i].argmax()),int(wrong_logits[i].argmax()))

# %%

def ablate_previous(keys):
    return np.zeros_like(keys)


if MAIN:
    tests.test_ablate_previous(ablate_previous)

# %%

def composition_score(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.linalg.norm(a @ b) / denom) if denom else 0.0


if MAIN:
    tests.test_composition_score(composition_score)

# %%

def first_crossing(coordinates, scores, threshold):
    hits = np.flatnonzero(np.asarray(scores) >= threshold)
    return coordinates[int(hits[0])] if len(hits) else None


if MAIN:
    tests.test_first_crossing(first_crossing)

# %%

if MAIN:
    period = 4
    tokens = repeat_tokens(period, 3)
    strengths = np.array([0., .05, .1, .2, .4, .7, 1.])
    scores, gains, clean_loss, wrong_loss, ablated_loss = [], [], [], [], []
    for strength in strengths:
        attn = induction_pattern(tokens, period, strength)
        loss = next_token_loss(copy_logits(attn, tokens, period), tokens)
        wrong = next_token_loss(copy_logits(attn, (tokens+1)%period, period), tokens)
        keys = ablate_previous(prefix_keys(tokens, period, strength))
        ablated = causal_softmax(12 * np.eye(period)[tokens] @ keys.T)
        abl_loss = next_token_loss(copy_logits(ablated, tokens, period), tokens)
        scores.append(prefix_score(attn, period)); gains.append(icl_gain(loss, period))
        clean_loss.append(loss[period:2*period-1].mean())
        wrong_loss.append(wrong[period:2*period-1].mean())
        ablated_loss.append(abl_loss[period:2*period-1].mean())
    print("strength prefix repeated_loss wrong_value_loss ablated_loss")
    for row in zip(strengths, scores, clean_loss, wrong_loss, ablated_loss):
        print(" ".join(f"{v:.4f}" for v in row))
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    axes[0].imshow(induction_pattern(tokens,period),vmin=0,vmax=1,cmap="magma")
    axes[0].set(title="Clean fixed copying circuit",xlabel="Source position",ylabel="Destination position")
    axes[1].plot(strengths,scores,"o-",label="Prefix match")
    axes[1].axhline(prefix_score(induction_pattern(tokens,period,0),period),ls="--",color="gray",label="Previous-token ablation")
    axes[1].set(title="Mechanism coordinate, not training",xlabel="Chosen previous-token strength",ylabel="Mean stripe attention")
    axes[1].legend()
    for vals,label in [(clean_loss,"Clean"),(wrong_loss,"Wrong values"),(ablated_loss,"Previous-token ablation")]:
        axes[2].plot(strengths,vals,"o-",label=label)
    axes[2].set(title="Same tokens and scoring offsets",xlabel="Chosen previous-token strength",ylabel="Repeated-block NLL (nats/token)")
    axes[2].legend(); utils.finish_figure(fig)

# %%

if MAIN:
    print("composition scores at strength 0 and 1:",
          composition_score(np.zeros((4,4)),np.eye(4)),
          composition_score(np.eye(4),np.eye(4)))

# %%

if MAIN:
    play_period = 6
    play_tokens = repeat_tokens(play_period, 3)
    play_attn = induction_pattern(play_tokens, play_period)
    play_uniform = induction_pattern(play_tokens, play_period,0)
    fig, ax = plt.subplots(figsize=(7,3))
    for attn,label in [(play_attn,"Prefix-copy fixture"),(play_uniform,"Ablated keys")]:
        losses = next_token_loss(copy_logits(attn,play_tokens,play_period),play_tokens)
        ax.plot(np.arange(len(losses)),losses,"o-",label=label)
    ax.set(xlabel="Destination token position",ylabel="NLL (nats)",title="Play: more competing keys")
    ax.legend();utils.finish_figure(fig)
    print("prefix score",round(prefix_score(play_attn,play_period),4))

# %%

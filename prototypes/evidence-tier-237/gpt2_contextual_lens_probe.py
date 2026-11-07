"""PROTOTYPE (throwaway, never merged), ticket #237: a tier-1 preview for the 7.1 bridge.

Real GPT-2 small residual streams over TinyStories (the corpus 7.1 already loads), CPU only,
offline (both were already in the local HF cache; HF_HUB_OFFLINE=1 forbids any download).

For held-out query tokens we compare three readouts of the residual stream at layer L:
  logit lens       ln_f(h) @ W_U  -> does its top-1 equal the NEXT token?
  embedding lens   nearest row of W_E by cosine -> does it equal the CURRENT token?
  contextual lens  nearest bank vector (same layer, other stories) -> its token = current token?
                   and the token that followed it in its story = the query's next token?
Controls: norm-matched random queries, a bank with shuffled token labels, and the cross-layer
heatmap (query layer x bank layer), which is the LatentLens signature figure in text-only form.

Run: HF_HUB_OFFLINE=1 <checkout>/.venv/bin/python -I gpt2_contextual_lens_probe.py > gpt2_probe_output.txt
"""

import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

HERE = Path(__file__).parent
torch.manual_seed(237)
torch.set_grad_enabled(False)
t0 = time.time()

tok = GPT2TokenizerFast.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
path = hf_hub_download(
    "roneneldan/TinyStories", "data/validation-00000-of-00001-869c898b519ad725.parquet", repo_type="dataset"
)
texts = pd.read_parquet(path)["text"].tolist()

SEQ, N_BANK, N_QUERY = 64, 600, 60
ids = []
for t in texts:
    x = tok(t)["input_ids"]
    if len(x) >= SEQ + 1:
        ids.append(x[: SEQ + 1])
    if len(ids) == N_BANK + N_QUERY:
        break
ids = torch.tensor(ids)
inp, nxt = ids[:, :SEQ], ids[:, 1 : SEQ + 1]

hs = []
for i in range(0, len(inp), 50):
    out = model.transformer(inp[i : i + 50], output_hidden_states=True)
    hs.append(torch.stack(out.hidden_states[:-1], 0))  # resid_pre of layers 0..11 (last one has ln_f)
H = torch.cat(hs, 1)  # (12 layers, n_seq, SEQ, 768)
n_layers = H.shape[0]
POS0 = 4  # skip the first positions (attention-sink / BOS-like states)
bank_H, query_H = H[:, :N_BANK, POS0:], H[:, N_BANK:, POS0:]
bank_cur, bank_nxt = inp[:N_BANK, POS0:].reshape(-1), nxt[:N_BANK, POS0:].reshape(-1)
q_cur, q_nxt = inp[N_BANK:, POS0:].reshape(-1), nxt[N_BANK:, POS0:].reshape(-1)
print(f"bank: {bank_cur.numel()} contextual vectors per layer ({N_BANK} stories x {SEQ - POS0} positions)")
print(f"queries: {q_cur.numel()} tokens from {N_QUERY} held-out stories; layers 0..{n_layers - 1}")

W_U = model.lm_head.weight  # tied to W_E in GPT-2
W_E = model.transformer.wte.weight
ln_f = model.transformer.ln_f


def unit(x):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def nn_index(q, bank, chunk=4096):
    best_s, best_i = [], []
    b = unit(bank)
    for i in range(0, len(q), chunk):
        s = unit(q[i : i + chunk]) @ b.T
        v, j = s.max(1)
        best_s.append(v)
        best_i.append(j)
    return torch.cat(best_i), torch.cat(best_s)


# Bigram floor (7.1 already uses one for the tuned lens): predict the most common follower of the
# current token in the bank. If the contextual lens's "next" score is no better, it is just bigram lookup.
V = 50257
pair = bank_cur * V + bank_nxt
uniq, counts = pair.unique(return_counts=True)
best_follow = {}
for code, c in zip(uniq.tolist(), counts.tolist()):
    a, b = divmod(code, V)
    if c > best_follow.get(a, (None, 0))[1]:
        best_follow[a] = (b, c)
bigram_pred = torch.tensor([best_follow.get(t, (-1, 0))[0] for t in q_cur.tolist()])
bigram_floor = (bigram_pred == q_nxt).float().mean().item()
print(f"bigram floor (most common follower of the current token in the bank): next-token match {bigram_floor:.2f}")

rows = []
print(
    f"\n{'layer':>5s} {'logit->next':>12s} {'embed->cur':>11s} {'ctx->cur':>9s} {'ctx->next':>10s} "
    f"{'rand q cos':>11s} {'rand pair cos':>14s} {'top-1 cos':>10s} {'shuffled->cur':>14s} {'centred->cur':>13s}"
)
perm = torch.randperm(bank_cur.numel())
for L in range(n_layers):
    q = query_H[L].reshape(-1, 768)
    bnk = bank_H[L].reshape(-1, 768)
    logit_top = (ln_f(q) @ W_U.T).argmax(-1)
    emb_top = (unit(q) @ unit(W_E).T).argmax(-1)
    j, s = nn_index(q, bnk)
    r = torch.randn_like(q)
    r = r * (q.norm(dim=-1, keepdim=True) / r.norm(dim=-1, keepdim=True))
    _, s_rand = nn_index(r, bnk)
    pair_cos = (unit(q) * unit(bnk[torch.randint(len(bnk), (len(q),))])).sum(-1)  # anisotropy baseline
    mu = bnk.mean(0, keepdim=True)
    jc, _ = nn_index(q - mu, bnk - mu)
    row = dict(
        layer=L,
        logit_next=(logit_top == q_nxt).float().mean().item(),
        embed_cur=(emb_top == q_cur).float().mean().item(),
        ctx_cur=(bank_cur[j] == q_cur).float().mean().item(),
        ctx_next=(bank_nxt[j] == q_nxt).float().mean().item(),
        rand_cos=s_rand.mean().item(),
        pair_cos=pair_cos.mean().item(),
        real_cos=s.mean().item(),
        shuffled_cur=(bank_cur[perm][j] == q_cur).float().mean().item(),
        centred_cur=(bank_cur[jc] == q_cur).float().mean().item(),
    )
    rows.append(row)
    print(
        f"{L:5d} {row['logit_next']:12.2f} {row['embed_cur']:11.2f} {row['ctx_cur']:9.2f} {row['ctx_next']:10.2f} "
        f"{row['rand_cos']:11.2f} {row['pair_cos']:14.2f} {row['real_cos']:10.2f} {row['shuffled_cur']:14.3f} "
        f"{row['centred_cur']:13.2f}"
    )
unigram_next = (bank_nxt.bincount(minlength=50257).argmax() == q_nxt).float().mean().item()
print(f"\nnext-token floor: always predicting the bank's most common next token = {unigram_next:.2f}")

# Cross-layer heatmap: query layer x bank layer, current-token recovery (subsample queries for speed)
sub = torch.randperm(q_cur.numel())[:1500]
heat = np.zeros((n_layers, n_layers))
for Lq in range(n_layers):
    q = query_H[Lq].reshape(-1, 768)[sub]
    for Lb in range(n_layers):
        j, _ = nn_index(q, bank_H[Lb].reshape(-1, 768))
        heat[Lq, Lb] = (bank_cur[j] == q_cur[sub]).float().mean().item()
best = heat.argmax(1)
print("\ncross-layer current-token recovery, best bank layer per query layer:")
print("  " + "  ".join(f"q{Lq}->b{best[Lq]} ({heat[Lq, best[Lq]]:.2f} vs same-layer {heat[Lq, Lq]:.2f})" for Lq in range(n_layers)))

# A few qualitative examples at layer 6
L = 6
q = query_H[L].reshape(-1, 768)
j, s = nn_index(q, bank_H[L].reshape(-1, 768))
logit_top = (ln_f(q) @ W_U.T).topk(3, dim=-1).indices
bank_ctx = inp[:N_BANK]
q_ctx = inp[N_BANK:]
print(f"\nexamples at layer {L} (query in context | logit-lens top-3 | nearest bank token in its context):")
n_pos = SEQ - POS0
for qi in [17, 140, 333, 701, 1260, 2000]:
    si, pi = divmod(qi, n_pos)
    pi += POS0
    bj = j[qi].item()
    bs, bp = divmod(bj, n_pos)
    bp += POS0
    qtxt = tok.decode(q_ctx[si, max(0, pi - 6) : pi]) + "[" + tok.decode(q_ctx[si, pi]) + "]"
    btxt = tok.decode(bank_ctx[bs, max(0, bp - 6) : bp]) + "[" + tok.decode(bank_ctx[bs, bp]) + "]"
    lt = ", ".join(repr(tok.decode(t)) for t in logit_top[qi])
    print(f"  ...{qtxt!r} | {lt} | ...{btxt!r} (cos {s[qi]:.2f})")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
Ls = [r["layer"] for r in rows]
for key, lab in [
    ("logit_next", "logit lens: top-1 = next token"),
    ("embed_cur", "embedding lens: = current token"),
    ("ctx_cur", "contextual lens: = current token"),
    ("ctx_next", "contextual lens: neighbour's next = next"),
    ("shuffled_cur", "control: shuffled bank labels"),
]:
    ax[0].plot(Ls, [r[key] for r in rows], marker="o", label=lab)
ax[0].axhline(unigram_next, color="grey", ls=":", label="most-common-next floor")
ax[0].axhline(bigram_floor, color="black", ls="--", label="bigram floor (next)")
ax[0].set_xlabel("layer (resid_pre)")
ax[0].set_ylabel("match rate")
ax[0].set_title("GPT-2 small, TinyStories, held-out tokens")
ax[0].legend(fontsize=7)
im = ax[1].imshow(heat, origin="lower", cmap="viridis", vmin=0, vmax=1)
ax[1].set_xlabel("bank layer")
ax[1].set_ylabel("query layer")
ax[1].set_title("contextual lens across layers: current-token recovery")
plt.colorbar(im, ax=ax[1])
fig.tight_layout()
fig.savefig(HERE / "gpt2_contextual_lens_probe.png", dpi=110)
print(f"\nsaved gpt2_contextual_lens_probe.png; wall time {time.time() - t0:.0f} s on CPU")

"""PROTOTYPE (throwaway, never merged), ticket #237: what a tier-0 CPU toy can and cannot show.

Two planted toys, numpy only, a few seconds on CPU:

A. Contextual retrieval (the LatentLens mechanism). "Visual tokens" are described by their
   nearest neighbours in a bank of contextual text vectors. The toy shows the traps a learner
   must control for: a phrase-level "any related word in the top 5" rubric that random vectors
   also pass, background patches that pass the broad rubric with no local grounding, and
   shuffled metadata as the control that should fail.

B. A readable weight edit (the Sparks mechanism). A tiny MLP "solver" usually loops instead of
   answering. A rank-1 output-direction projection edit removes the loop logit. Accuracy jumps,
   but accuracy among finished answers does not: the gain is completion rescue, not capability.
   The toy also runs the projector-sign identity (u -> -u gives the same edit), the signed
   counterfactual (-dW is not a mirror image), a same-norm random edit, a decoding-only
   comparator and a function-preserving hidden-unit permutation.

Every number in tier_outlines.md for the toys comes from run_output.txt.
Run: <checkout>/.venv/bin/python -I evidence_tier_toy.py > run_output.txt
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
rng = np.random.default_rng(237)

# ---------------------------------------------------------------------------------------------
# A. Contextual retrieval toy
# ---------------------------------------------------------------------------------------------
print("=" * 90)
print("A. CONTEXTUAL RETRIEVAL TOY (planted: every result here is true by construction)")
print("=" * 90)

COLOURS = "red blue green yellow black white orange purple brown grey".split()
OBJECTS = "car dog cat boat bird horse chair kite bus apple".split()
SCENES = "beach street park lake field kitchen road sky snow garden".split()
FUNCTION = "a the on in near".split()
VOCAB = COLOURS + OBJECTS + SCENES + FUNCTION
W2I = {w: i for i, w in enumerate(VOCAB)}
d = 64
E = rng.normal(size=(len(VOCAB), d))
E /= np.linalg.norm(E, axis=1, keepdims=True)
mu_text = 1.5 * rng.normal(size=d) / np.sqrt(d)  # shared offset of text states (anisotropy)
mu_vis = 1.5 * rng.normal(size=d) / np.sqrt(d)  # a different offset for visual tokens (modality gap)


def text_phrase(c, o, s):
    return ["a", c, o, "on", "the", s]


def contextual_states(words, beta=0.6, noise=0.25):
    """Toy 'mid-layer' text state: own token + beta * mean of the other tokens + offset + noise."""
    ids = np.array([W2I[w] for w in words])
    own = E[ids]
    ctx = (E[ids].sum(0, keepdims=True) - own) / (len(ids) - 1)
    return own + beta * ctx + mu_text + noise * rng.normal(size=own.shape) / np.sqrt(d)


def build_bank(n_phrases):
    vecs, tok, phrase_id, phrases = [], [], [], []
    for p in range(n_phrases):
        c, o, s = rng.choice(COLOURS), rng.choice(OBJECTS), rng.choice(SCENES)
        words = text_phrase(c, o, s)
        vecs.append(contextual_states(words))
        tok += words
        phrase_id += [p] * len(words)
        phrases.append(words)
    return np.concatenate(vecs), np.array(tok), np.array(phrase_id), phrases


def visual_patch(c, o, s, kind, noise=0.35):
    """Object patches mix object, colour and a little scene; background patches carry only the scene."""
    if kind == "object":
        v = 0.9 * E[W2I[o]] + 0.6 * E[W2I[c]] + 0.3 * E[W2I[s]]
    else:
        v = 0.7 * E[W2I[s]]
    return v + mu_vis + noise * rng.normal(size=d) / np.sqrt(d)


def normalize_rows(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True).clip(1e-12)


def contextual_topk(queries, bank, k):
    sims = normalize_rows(queries) @ normalize_rows(bank).T
    idx = np.argsort(-sims, axis=1, kind="stable")[:, :k]
    return idx, np.take_along_axis(sims, idx, axis=1)


bank, bank_tok, bank_pid, bank_phrases = build_bank(3000)
content = np.isin(bank_tok, COLOURS + OBJECTS + SCENES)
bank_c, tok_c, pid_c = bank[content], bank_tok[content], bank_pid[content]  # LatentLens-style: content tokens
print(f"bank: {len(bank)} contextual text vectors from 3000 phrases; {content.sum()} content-token vectors kept")

n_img = 400
imgs = [(rng.choice(COLOURS), rng.choice(OBJECTS), rng.choice(SCENES)) for _ in range(n_img)]
q_obj = np.stack([visual_patch(*im, "object") for im in imgs])
q_bg = np.stack([visual_patch(*im, "background") for im in imgs])
q_rand = rng.normal(size=q_obj.shape)
q_rand *= np.linalg.norm(q_obj, axis=1, keepdims=True) / np.linalg.norm(q_rand, axis=1, keepdims=True)


def scores(queries, imgs, k=5, kind="object", shuffle=False):
    """Judge-free scores. local: a top-k token IS the patch's object. broad-token: a top-k token is
    any word of the image. broad-phrase: any word of a top-k token's whole phrase is a word of the image
    (the description shown to a judge is the token in its context)."""
    toks, pids = tok_c, pid_c
    if shuffle:  # break the vector -> (token, phrase) pairing; the vectors themselves are untouched
        perm = rng.permutation(len(toks))
        toks, pids = toks[perm], pids[perm]
    idx, sims = contextual_topk(queries, bank_c, k)
    local, btok, bphr = [], [], []
    for qi, (c, o, s) in enumerate(imgs):
        related = {c, o, s}
        got = toks[idx[qi]]
        local.append(o in got if kind == "object" else False)
        btok.append(bool(related & set(got)))
        words = set().union(*[set(bank_phrases[p]) for p in pids[idx[qi]]])
        bphr.append(bool(related & words))
    return np.mean(local), np.mean(btok), np.mean(bphr), sims[:, 0].mean()


rows = {
    "object patches": scores(q_obj, imgs, kind="object"),
    "background patches": scores(q_bg, imgs, kind="background"),
    "norm-matched random": scores(q_rand, imgs, kind="object"),
    "object, shuffled metadata": scores(q_obj, imgs, kind="object", shuffle=True),
}
print(f"\n{'query (k=5)':28s} {'local':>7s} {'broad/token':>12s} {'broad/phrase':>13s} {'top-1 cos':>10s}")
for name, (lo, bt, bp, cs) in rows.items():
    print(f"{name:28s} {lo:7.2f} {bt:12.2f} {bp:13.2f} {cs:10.2f}")

# Static embedding lens comparator: nearest input-embedding row (one candidate per word).
content_ids = np.array([W2I[w] for w in COLOURS + OBJECTS + SCENES])
sims_static = normalize_rows(q_obj) @ normalize_rows(E[content_ids]).T
top_static = np.array(COLOURS + OBJECTS + SCENES)[np.argsort(-sims_static, axis=1)[:, :5]]
print(
    f"\nstatic embedding lens, object patches: local {np.mean([im[1] in t for im, t in zip(imgs, top_static)]):.2f}"
    "  (same planted content, one candidate per word: the toy cannot say which lens is better)"
)

ks = [1, 2, 3, 5, 10, 20]
k_sweep = {name: [] for name in ["random / token", "random / phrase", "object / token"]}
for k in ks:
    k_sweep["random / token"].append(scores(q_rand, imgs, k=k)[1])
    k_sweep["random / phrase"].append(scores(q_rand, imgs, k=k)[2])
    k_sweep["object / token"].append(scores(q_obj, imgs, k=k)[1])
print("\nbroad-rubric pass rate against k (candidates shown per query):")
for name, v in k_sweep.items():
    print(f"  {name:16s} " + "  ".join(f"k={k}:{x:.2f}" for k, x in zip(ks, v)))

# Hand-check tests a learner's functions would face (the tier-0 'immediate tests').
q = np.array([[1.0, 0.0], [0.0, 0.0]])
b = np.array([[2.0, 0.0], [1.0, 1.0], [0.0, 3.0], [2.0, 0.0]])
i1, _ = contextual_topk(q[:1], b, 2)
assert list(i1[0]) == [0, 3], "stable ties: first of two identical vectors wins"
i2, _ = contextual_topk(5 * q[:1], b, 2)
assert (i1 == i2).all(), "positive rescaling of the query does not change neighbours"
print("\nunit checks passed: stable ties, positive-scale invariance (zero-vector policy: clip norm, rank arbitrary)")

# ---------------------------------------------------------------------------------------------
# B. Weight-edit toy
# ---------------------------------------------------------------------------------------------
print("\n" + "=" * 90)
print("B. WEIGHT-EDIT TOY (planted: a solver that usually loops instead of answering)")
print("=" * 90)

n_in, n_hid, n_cls = 8, 20, 10
LOOP = n_cls  # output index 10 = "keep generating" (an unfinished answer)
T = rng.normal(size=(n_cls, n_in))
# hidden = relu([T x; -T x]), so relu(z) - relu(-z) = z lets the answer head read T x exactly
W1 = np.vstack([T, -T])
b1 = np.zeros(n_hid)
W2 = np.zeros((n_cls + 1, n_hid))
W2[:n_cls, :n_cls] = np.eye(n_cls)
W2[:n_cls, n_cls:] = -np.eye(n_cls)  # answer logits = T x
W2[:n_cls] += 0.3 * rng.normal(size=(n_cls, n_hid))  # imperfect "skill"
b2 = np.zeros(n_cls + 1)
W2[LOOP] = 0.25 * rng.normal(size=n_hid)
b2[LOOP] = 6.0  # strong habit of looping


def forward(x, W1, b1, W2, b2):
    h = np.maximum(0, x @ W1.T + b1)
    return h @ W2.T + b2


X = rng.normal(size=(5000, n_in))
y = np.argmax(X @ T.T, axis=1)


def behaviour(logits, ban_loop=False):
    pred = np.argmax(logits[:, :n_cls], 1) if ban_loop else np.argmax(logits, 1)
    answered = pred != LOOP
    correct = answered & (pred == y)
    return answered.mean(), correct.mean(), correct.sum() / max(answered.sum(), 1)


def output_direction_edit(W, b, u, strength=1.0):
    """Remove `strength` of the output component along unit u: [W|b] <- (I - s u u^T)[W|b]."""
    u = u / np.linalg.norm(u)
    P = np.eye(len(u)) - strength * np.outer(u, u)
    return P @ W, P @ b


u = np.zeros(n_cls + 1)
u[LOOP] = 1.0
W2e, b2e = output_direction_edit(W2, b2, u)
W2s, b2s = output_direction_edit(W2, b2, -u)
dW, db = W2e - W2, b2e - b2
assert np.allclose(W2e, W2s) and np.allclose(b2e, b2s)
print("projector-sign identity: editing along -u gives the identical matrix (u u^T is unchanged). PASS")

n_par = W1.size + b1.size + W2.size + b2.size
support = int((np.abs(np.c_[dW, db]) > 0).sum())
sv = np.linalg.svd(np.c_[dW, db], compute_uv=False)
print(
    f"diff: {support} of {n_par} parameters changed ({support / n_par:.1%}), "
    f"singular values of [dW|db]: {sv[0]:.2f}, then max {sv[1:].max():.1e} (rank 1); "
    f"loop bias {b2[LOOP]:.1f} -> {b2e[LOOP]:.1f}"
)

# Random controls with the edit's Frobenius norm, 20 seeds each:
#  same support = a random change to the loop row only; random rank-1 = a random outer product anywhere.
edit_norm = np.linalg.norm(np.c_[dW, db])


def random_edit(kind, seed):
    g = np.random.default_rng(seed)
    D = np.zeros((n_cls + 1, n_hid + 1))
    if kind == "same support":
        D[LOOP] = g.normal(size=n_hid + 1)
    else:
        D = np.outer(g.normal(size=n_cls + 1), g.normal(size=n_hid + 1))
    D *= edit_norm / np.linalg.norm(D)
    return W2 + D[:, :-1], b2 + D[:, -1]


# wrong-row control: same edit on an answer row
uw = np.zeros(n_cls + 1)
uw[3] = 1.0
Ww, bw = output_direction_edit(W2, b2, uw)
# function-preserving permutation of hidden units, then the same edit
perm = rng.permutation(n_hid)
W1p, b1p, W2p = W1[perm], b1[perm], W2[:, perm]
assert np.abs(forward(X, W1p, b1p, W2p, b2) - forward(X, W1, b1, W2, b2)).max() < 1e-12
W2pe, b2pe = output_direction_edit(W2p, b2, u)

conds = {
    "base": forward(X, W1, b1, W2, b2),
    "edit (remove loop dir)": forward(X, W1, b1, W2e, b2e),
    "signed counterfactual -dW": forward(X, W1, b1, W2 - dW, b2 - db),
    "wrong row (class 3)": forward(X, W1, b1, Ww, bw),
    "permuted units + edit": forward(X, W1p, b1p, W2pe, b2pe),
}
print(f"\n{'condition':28s} {'answered':>9s} {'accuracy':>9s} {'acc|answered':>13s}")
results = {}
for name, lg in conds.items():
    results[name] = behaviour(lg)
    print(f"{name:28s} {results[name][0]:9.2f} {results[name][1]:9.2f} {results[name][2]:13.2f}")
for kind in ["same support", "random rank-1"]:
    runs = np.array([behaviour(forward(X, W1, b1, *random_edit(kind, s))) for s in range(20)])
    results[f"random edit, {kind}"] = tuple(runs.mean(0))
    m, sd = runs.mean(0), runs.std(0)
    print(
        f"{'random edit, ' + kind:28s} {m[0]:5.2f}±{sd[0]:.2f} {m[1]:5.2f}±{sd[1]:.2f} {m[2]:9.2f}±{sd[2]:.2f}"
        "   (mean±sd, 20 seeds, same norm)"
    )
results["decoding only (ban loop)"] = behaviour(conds["base"], ban_loop=True)
a, c, ca = results["decoding only (ban loop)"]
print(f"{'decoding only (ban loop)':28s} {a:9.2f} {c:9.2f} {ca:13.2f}   <- no weight changed")
print(
    "\npermutation: outputs identical (max diff < 1e-12), yet "
    f"{(np.abs(W2p - W2) > 1e-12).sum()} of {W2.size} entries of W2 moved: a coordinate-based reading changes."
)

# ---------------------------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
names = list(rows)
xx = np.arange(len(names))
for j, (lab, col) in enumerate([("local", 0), ("broad / token", 1), ("broad / phrase", 2)]):
    ax[0].bar(xx + (j - 1) * 0.27, [rows[n][col] for n in names], 0.27, label=lab)
ax[0].set_xticks(xx, [n.replace(", ", ",\n").replace(" patches", "\npatches") for n in names], fontsize=8)
ax[0].set_ylabel("pass rate (top-5)")
ax[0].set_title("A. Which queries 'pass'? (planted toy)")
ax[0].legend(fontsize=8)
for name, v in k_sweep.items():
    ax[1].plot(ks, v, marker="o", label=name)
ax[1].set_xscale("log")
ax[1].set_xticks(ks, ks)
ax[1].set_xlabel("candidates shown per query (k)")
ax[1].set_ylabel("broad-rubric pass rate")
ax[1].set_title("A. Random vectors pass a phrase rubric")
ax[1].legend(fontsize=8)
cn = list(results)
vals = np.array([results[n] for n in cn])
xx = np.arange(len(cn))
for j, lab in enumerate(["answered", "accuracy", "acc | answered"]):
    ax[2].bar(xx + (j - 1) * 0.27, vals[:, j], 0.27, label=lab)
ax[2].set_xticks(xx, [n.replace(" (", "\n(").replace(" + ", "\n+ ") for n in cn], fontsize=7, rotation=30, ha="right")
ax[2].set_title("B. Completion rescue, not capability")
ax[2].legend(fontsize=8)
im = ax[3].imshow(np.c_[dW, db], cmap="RdBu", vmin=-3.5, vmax=3.5, aspect="auto")
ax[3].set_title("B. The edit [dW | db]: one row, rank 1")
ax[3].set_xlabel("hidden unit (last column: bias)")
ax[3].set_ylabel("output (10 = loop)")
plt.colorbar(im, ax=ax[3])
fig.tight_layout()
fig.savefig(HERE / "evidence_tier_toy.png", dpi=110)
print(f"\nsaved {HERE / 'evidence_tier_toy.png'}")

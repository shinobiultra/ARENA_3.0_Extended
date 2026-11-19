# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
[
    {"title": "The Logit Lens", "icon": "1-circle-fill", "subtitle": "(25%)"},
    {"title": "The Tuned Lens", "icon": "2-circle-fill", "subtitle": "(30%)"},
    {"title": "Patchscopes", "icon": "3-circle-fill", "subtitle": "(25%)"},
    {"title": "Held-out Quiz", "icon": "bullseye", "subtitle": "(10%)"},
    {"title": "Bonus", "icon": "star", "subtitle": "(10%)"},
]
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# [7.1] Logit Lens, Tuned Lens & Patchscopes
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<img src="https://raw.githubusercontent.com/info-arena/ARENA_img/main/misc/headers/header-14-1.png" width="350">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# Introduction
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
If you give GPT-2 Small the prompt

```
The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is Rome. The capital of Germany is
```

it will answer `" Berlin"`. That answer is computed by 12 transformer blocks, each of which reads from and writes to the **residual stream**. So here's a natural question: *at
which layer does the model "know" the answer is Berlin?* Is the answer assembled gradually, or does it appear all at once in the last block? Does the model first figure out that
the answer is *some* capital city, and only later figure out *which* one?

Three numbers from this section show how much there is to find out:

- GPT-2 Small names the right capital for 77% of 48 countries.
- A **tuned lens** reads the model's own next-token prediction 37% of the time straight from the token embeddings, before any block has run. The free **logit lens** manages 0%.
- A **Patchscope** that reads a country's layer-4 state names its capital 79% of the time. Three controls that should fail (a text-only prompt, a random vector and the BOS token) all get 0%.

We'll study three methods which all try to **read the residual stream in the model's own language**:

- The **logit lens** (nostalgebraist, 2020) takes an intermediate residual stream vector and decodes it as if it were the final layer, by applying the model's own final layernorm and unembedding. It costs nothing to compute, needs no training, and works surprisingly well on GPT-2.
- The **tuned lens** (Belrose et al., 2023) points out that intermediate layers don't necessarily use the same "basis" as the final layer, and fixes this by learning a small affine *translator* for each layer, trained to make the decoded distribution match the model's final output.
- **Patchscopes** (Ghandeharioun et al., 2024) take a different approach: instead of decoding a vector with a fixed readout, they *patch* it into a different prompt (an "inspection prompt" like `"The capital of x is"`) and let the model itself tell us what information the vector carries.

We study GPT-2 Small through TransformerLens, plus about 20 MB of TinyStories validation text (1,600 train and 200 test sequences of 64 tokens). Nothing is trained except the tuned lens (about 7 million parameters, 300 steps).

A recurring theme will be **controls**. Decoding methods are very good at producing plausible-looking tokens, so every result in this notebook is paired with a baseline that should fail. When a control *doesn't* fail, that's often the most interesting result of all.

In this section you will:

- read GPT-2's hidden states with the logit lens and watch Berlin climb;
- train a tuned lens and beat the logit lens on held-out text;
- find each state's nearest neighbours in a bank of real contexts, and put all three lenses on one figure, next to a bigram table and a shuffled bank that should lose;
- make the model decode its own states with Patchscopes;
- check each result against a control that should fail: mismatched capitals, random vectors, a bigram table, shuffled bank labels, text-only and BOS inputs.

Prerequisites: [1.2] [Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/) (hooks, `run_with_cache`, the residual stream) and [0.3] [Optimization](https://learn.arena.education/chapter0_fundamentals/03_optimization/) (a short PyTorch training loop).
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Content & Learning Objectives

### 1️⃣ The Logit Lens

You'll implement the logit lens by applying GPT-2's final layernorm and unembedding to every layer of the residual stream, and use it to watch the answer to a factual question
emerge. You'll compare the rank of the correct answer against the ranks of *other* plausible answers and against random vectors, to separate "the model knows the answer" from "the
model knows what kind of answer goes here". Then you'll read the same logit the other way round, as a sum of contributions from individual attention heads.

> ##### Learning Objectives
>
> - Understand why the residual stream can be decoded at any layer, and what assumption the logit lens makes when it does so
> - Implement the logit lens and use it to produce top-token tables and answer-rank curves
> - Use mismatched-answer and random-vector controls to interpret a rank curve correctly
> - Measure how far each layer's decoded distribution is from the final one (KL divergence)
> - Implement the attention lens, and check that heads, MLPs and embeddings add up to the final logit

### 2️⃣ The Tuned Lens

The logit lens assumes every layer speaks the same language as the last one. You'll test that assumption by training a tuned lens: one affine translator per layer, optimised to
match the model's output distribution on held-out text. You'll then ask what the improvement actually consists of, by swapping translators between layers and by comparing it with a
different kind of reader: the nearest neighbours of each state in a bank of real contexts. The lens-comparison figure at the end of this part puts all of them side by side.

> ##### Learning Objectives
>
> - Understand representational drift, and why a learned per-layer translator can fix it
> - Implement and train a tuned lens with a KL-divergence objective
> - Evaluate lenses on held-out data with KL and top-1 agreement, rather than on hand-picked prompts
> - Implement a contextual lens (cosine nearest neighbours in a bank of real contexts) and compare it across layers
> - Use a layer-0 floor, a bigram table, shuffled bank labels and random queries to avoid over-reading a trained probe

### 3️⃣ Patchscopes

Rather than decoding a vector with a fixed readout, you'll patch it into an inspection prompt and let the model decode it for you. You'll build a Patchscope that extracts a
country's capital from the country's hidden state, map how well it works for every (source layer, target layer) pair, and check it against text-only, random-vector and
wrong-position controls.

> ##### Learning Objectives
>
> - Understand the Patchscope framework (source prompt, source layer, target prompt, target layer) and how the logit lens is a special case of it
> - Implement activation patching across prompts with TransformerLens hooks
> - Produce a source-layer × target-layer accuracy map and say what it does and doesn't show
> - Separate information *carried by the patched vector* from information supplied by the inspection prompt itself
> - Put an interval on a rate measured on 48 countries

### 🎯 Held-out Quiz

Five experiments we ran on GPT-2 Small and didn't show you, one of them on a model you never loaded. Each changes one thing in a run you made. You'll predict each result and write
down why, then see what really happened and what a good mental model would have said.

> ##### Learning Objectives
>
> - Predict the result of a change to a lens or Patchscope run from results you've already measured
> - Name the result your prediction rests on, and notice where your picture breaks down
> - Check a number against the run that produced it

### ☆ Bonus

Suggestions for going further: watching the model "change its mind" between a country's largest city and its capital, an anomaly hunt in the Patchscope grid, a necessity test for the
blocks that grid points at, the claim of the LatentLens paper, what this lesson doesn't show, and the papers that extend these methods.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Setup code

The whole lesson takes about 3 minutes on a GPU. It works on CPU with a shorter tuned-lens run, and the downloads add up to under 1 GB: GPT-2 Small (about 0.5 GB), the TinyStories validation file (about 20 MB) and, for the quiz, Pythia-160M (about 0.3 GB). The saved outputs and figures below were produced on a CUDA GPU.
'''

# ! CELL TYPE: code
# ! FILTERS: [~]
# ! TAGS: []

from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 1️⃣ The Logit Lens
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Reading the residual stream early

Recall the structure of a GPT-2 style transformer. The token embedding (plus positional embedding) is written into the residual stream, and then each block reads from the stream
and *adds* its output back in:

$$
x_0 = W_E[\text{token}] + W_{pos}[\text{position}], \qquad x_{\ell+1} = x_\ell + \text{Attn}_\ell(x_\ell) + \text{MLP}_\ell(x_\ell + \text{Attn}_\ell(x_\ell))
$$

At the very end, the model produces logits by applying a final layernorm and the unembedding matrix:

$$
\text{logits} = \text{LN}_f(x_L)\, W_U + b_U
$$

Because every block only ever *adds* to the stream, the final residual $x_L$ is literally the sum of the embedding and every block's output. The idea of the **logit lens** is to
stop early: take $x_\ell$ for some $\ell < L$, and apply exactly the same readout,

$$
\text{LogitLens}(x_\ell) = \text{LN}_f(x_\ell)\, W_U + b_U .
$$

If you think of the residual stream as the model's working memory, the logit lens asks: *if the model had to stop thinking right now and answer, what would it say?*

Notice the assumption hiding in that equation. We're using the readout that was trained for $x_L$ on a vector $x_\ell$ from somewhere in the middle of the network. That's only
sensible if intermediate layers represent "what the next token will be" in roughly the same directions as the final layer does. For GPT-2 this turns out to be approximately true,
which is why the logit lens was discovered on GPT-2. In section 2️⃣ we'll see how much this assumption costs us, and fix it.

A note on layernorm: TransformerLens folds the learned layernorm weights into the following linear layers by default, so `model.ln_final` just centres and normalises its input, and
`model.unembed` applies the (folded) $W_U$ and $b_U$. We apply `ln_final` *separately* to each layer's residual stream vector, so each one is normalised by its own scale. This
matters, because the residual stream norm grows a lot over the course of the model.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Let's load GPT-2 Small. We pin the exact Hugging Face revision of its weights and tokenizer, so that everyone gets the same model. We won't train it at any point in this section, so we turn off gradients for its parameters.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

GPT2_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
hf_model = AutoModelForCausalLM.from_pretrained("gpt2", revision=GPT2_REVISION)
hf_tokenizer = AutoTokenizer.from_pretrained("gpt2", revision=GPT2_REVISION)
model = HookedTransformer.from_pretrained("gpt2-small", hf_model=hf_model, tokenizer=hf_tokenizer, device=device)
model.requires_grad_(False)

print(f"{model.cfg.n_layers} layers, d_model={model.cfg.d_model}, vocab size={model.cfg.d_vocab}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## The dataset: capital cities

Our running example is a simple factual recall task. GPT-2 Small is a small model, and if you ask it `"The capital of Germany is"` cold, it will usually say something like `" a"`
or `" now"`. But if you show it a few examples first, it does much better. We'll use a few-shot prompt with three examples, then ask about a new country.

We also need every country and every capital to be a **single token**, for two reasons. First, it means every prompt has exactly the same length, so we can batch them without
padding and talk about "the last position" unambiguously. Second, it means "the answer" is a single entry of the logit vector, which makes ranks and probabilities easy to define.
The cell below filters our candidate list down to the pairs which satisfy this.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

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


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">48 country/capital pairs, all prompts have length 27
['&lt;|endoftext|&gt;', 'The', ' capital', ' of', ' France', ' is', ' Paris', '.', ' The', ' capital', ' of', ' Japan', ' is', ' Tokyo', '.', ' The', ' capital', ' of', ' Italy', ' is', ' Rome', '.', ' The', ' capital', ' of', ' Germany', ' is']</pre>

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Before we look inside the model, let's see how well it actually does. It's important to know this *before* interpreting any internal decoding: if the model doesn't get the answer
right at the output, we shouldn't expect any lens to find it in the middle.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

logits, cache = model.run_with_cache(tokens)
predictions = logits[:, -1].argmax(-1)
correct = predictions == answer_tokens

print(f"Top-1 accuracy: {correct.float().mean():.1%}\n")
print("Mistakes:")
for (country, capital), pred, ok in zip(CAPITALS, predictions, correct):
    if not ok:
        print(f"  {country:<12} expected {capital!r:<12} got {model.to_string(pred)!r}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Top-1 accuracy: 77.1%

Mistakes:
  China        expected 'Beijing'    got ' Shanghai'
  Canada       expected 'Ottawa'     got ' Toronto'
  Australia    expected 'Canberra'   got ' Sydney'
  Turkey       expected 'Ankara'     got ' Istanbul'
  Syria        expected 'Damascus'   got ' Aleppo'
  Mexico       expected 'Mexico'     got ' San'
  India        expected 'New'        got ' Mumbai'
  Pakistan     expected 'Islamabad'  got ' Karachi'
  Switzerland  expected 'Bern'       got ' Zurich'
  Libya        expected 'Tripoli'    got ' Benghazi'
  Jamaica      expected 'Kingston'   got ' Jamaica'</pre>

</details>

The mistakes are not random! In almost every case the model names the country's *largest or most famous city* instead of its capital. Keep this in mind; we'll come back to it in
the bonus section, where we'll use the lenses to watch the model weighing up "Shanghai" against "Beijing".
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement the logit lens

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement `logit_lens`, which takes a cache from a forward pass and returns the logit lens decoding of the residual stream at a single sequence position, for every layer.

We'll decode $n_{layers} + 1$ residual stream vectors per prompt:

- `resid_pre` for each of the $n_{layers}$ blocks (so the first one is just the embeddings, and the $\ell$-th one is the stream *before* block $\ell$),
- plus `resid_post` of the final block, which is the stream the model actually unembeds.

So the output should have shape `(n_layers + 1, batch, d_vocab)`, and its last row should be exactly the model's real output logits at that position. For each vector, apply
`model.ln_final` and then `model.unembed` (both of these act on the last dimension, so you can apply them to a whole stack of vectors at once).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def logit_lens(
    model: HookedTransformer, cache: ActivationCache, pos: int = -1
) -> Float[Tensor, "layer batch d_vocab"]:
    """
    Returns the logit lens decoding of the residual stream at sequence position `pos`, for the
    stream before every block plus the stream after the final block (so n_layers + 1 rows).
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    n_layers = model.cfg.n_layers
    resid_stack = t.stack(
        [cache["resid_pre", layer][:, pos] for layer in range(n_layers)] + [cache["resid_post", n_layers - 1][:, pos]]
    )
    return model.unembed(model.ln_final(resid_stack))
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_logit_lens(logit_lens, model)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

You can get the residual stream before block `layer` with `cache["resid_pre", layer]`, which has shape `(batch, seq, d_model)`. Index the position, stack the results along a new
first dimension, then apply `model.ln_final` followed by `model.unembed`.
</details>

<details>
<summary>Help - my output is close to the expected one but the test fails.</summary>

Make sure you apply `ln_final` to each layer's vector *separately* (which happens automatically if you apply it to the stacked tensor, since layernorm acts on the last dimension),
rather than e.g. normalising by the final layer's scale. Also check you're using `resid_post` of the *last* block for the final row, not `resid_pre`.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
The test compares your function against TransformerLens's `cache.accumulated_resid()`, which returns the same stack of residual stream vectors.

<details>
<summary>Aside - why not just use <code>cache.accumulated_resid(apply_ln=True)</code>?</summary>

You might have seen `accumulated_resid(apply_ln=True)` used in [1.2] or [1.4.1] for direct logit attribution. It looks like it should give the logit lens, but it doesn't quite: it
divides *every* layer's vector by the layernorm scale that was computed on the **final** residual stream. That makes the readout a linear function of the residual stream, which is
exactly what you want for attribution (so that the contributions of different components add up to the final logit). The logit lens instead asks "what would the model say if it
stopped here?", and if the model stopped here, the final layernorm would normalise *this* vector by *its own* scale.

Since the residual stream norm grows a lot through the network, the two versions give very different results at early layers. Try decoding with both and comparing the top tokens!
</details>

We'll use these labels for the $n_{layers}+1$ rows throughout:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

LAYER_LABELS = ["embed"] + [f"pre {layer}" for layer in range(1, model.cfg.n_layers)] + ["final"]
lens_logits = logit_lens(model, cache, pos=-1)
print(lens_logits.shape)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - make a top-tokens table

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 5-10 minutes on this exercise.
> ```

Numbers are hard to read; tokens are easy. Implement `top_tokens_table`, which takes logit-lens logits for a **single prompt** (shape `(layer, d_vocab)`) and returns a
`pandas.DataFrame` with one row per layer and one column per rank. Each entry should be a string containing the token (use `repr` so you can see leading spaces) and its probability
to 2 decimal places, e.g. `"' Berlin' (0.42)"`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def top_tokens_table(
    model: HookedTransformer, logits: Float[Tensor, "layer d_vocab"], k: int = 5, row_labels: list[str] | None = None
) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per layer, where column j contains the j-th most likely token
    at that layer along with its probability, formatted like "' Berlin' (0.42)".
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    probs = logits.softmax(dim=-1)
    top_probs, top_ids = probs.topk(k, dim=-1)
    rows = [
        [f"{model.to_string(tok)!r} ({p:.2f})" for tok, p in zip(ids.tolist(), ps.tolist())]
        for ids, ps in zip(top_ids, top_probs)
    ]
    return pd.DataFrame(rows, index=row_labels, columns=[f"top {j + 1}" for j in range(k)])
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_top_tokens_table(top_tokens_table, model)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Let's look at the prompt about Germany:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

germany_idx = countries.index("Germany")
display(top_tokens_table(model, lens_logits[:, germany_idx], k=4, row_labels=LAYER_LABELS))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

The exact probabilities may differ very slightly on different hardware, but you should see:

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                   top 1                top 2                top 3              top 4
embed   ' destro' (0.42)     ' arrang' (0.28)   ' mathemat' (0.15)    ' corrid' (0.09)
pre 1      ' not' (0.26)        ' now' (0.25)      ' still' (0.09)      ' also' (0.08)
pre 2      ' not' (0.36)        ' now' (0.20)       ' also' (0.13)     ' still' (0.03)
pre 3      ' not' (0.32)        ' now' (0.25)       ' also' (0.14)     ' still' (0.06)
pre 4     ' also' (0.43)        ' not' (0.29)        ' now' (0.10)     ' still' (0.03)
pre 5     ' also' (0.35)        ' not' (0.35)        ' now' (0.07)     ' still' (0.03)
pre 6      ' not' (0.40)       ' also' (0.09)    ' neither' (0.08)      ' none' (0.07)
pre 7     ' also' (0.18)        ' not' (0.17)        ' the' (0.09)   ' neither' (0.07)
pre 8     ' Rome' (0.25)     ' London' (0.17)      ' Paris' (0.06)    ' Venice' (0.04)
pre 9   ' Berlin' (0.60)     ' Munich' (0.24)  ' Frankfurt' (0.04)   ' Hamburg' (0.03)
pre 10  ' Berlin' (0.81)     ' Munich' (0.11)  ' Frankfurt' (0.05)   ' Germany' (0.01)
pre 11  ' Berlin' (0.50)  ' Frankfurt' (0.16)    ' Hamburg' (0.16)    ' Munich' (0.11)
final   ' Berlin' (0.25)    ' Dresden' (0.11)  ' Frankfurt' (0.11)    ' Munich' (0.10)</pre>

</details>

<details>
<summary>Question - what story does this table tell? Try to describe each phase before reading on.</summary>

Reading from top to bottom:

- **Embedding row: junk.** The decoded tokens (`' destro'`, `' arrang'`, ...) are word fragments with nothing
  to do with the prompt. The embedding row is the token embedding of `" is"` plus the positional embedding for
  position 26, and after layernorm it happens to point towards some rare tokens. It isn't a prediction of
  anything, which is the logit lens's core assumption failing in the most obvious way.
- **Layers 1-7: "what usually follows *is*".** Generic continuations like `' not'`, `' now'` and `' also'`. The model hasn't (visibly) used the context yet.
- **Layer 8: "a capital city".** Suddenly the top tokens are capitals, and two of them (`' Rome'`, `' Paris'`) appear *in the prompt*. The model has worked out what type of answer goes here, and its best guesses at this point lean on the in-context examples.
- **Layers 9-11: "a German city, probably Berlin".** `' Berlin'` takes over, and the runners-up are all German cities (`' Munich'`, `' Frankfurt'`, `' Hamburg'`). So by now the model is using the information that the country is Germany.
- **Final row: less confident than layer 10.** The final layer actually *spreads out* probability (Berlin goes from 0.81 to 0.25). The logit lens decodes intermediate layers as if they were final, but the last block evidently does more than just sharpen the answer.

We'll check whether this "type first, then specific answer" story holds across all 48 prompts in the next exercise.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - compute answer ranks across layers

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Tables are good for one prompt; for 48 prompts we want a single number per layer. The **rank** of the correct answer is a good choice: it's 1 when the answer is the top prediction,
and it's comparable across layers even when the overall scale of the logits changes a lot (which, as we'll see, it does).

Implement `answer_ranks`, which takes logits of shape `(layer, batch, d_vocab)` and a tensor of answer token ids of shape `(batch,)`, and returns the rank of each answer at each
layer. Define the rank as one plus the number of tokens with a *strictly larger* logit, so a token tied for first place still has rank 1.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def answer_ranks(
    logits: Float[Tensor, "layer batch d_vocab"], answer_tokens: Int[Tensor, " batch"]
) -> Int[Tensor, "layer batch"]:
    """
    Returns the rank (1 = top) of answer_tokens[b] in logits[layer, b], for every layer and batch index.
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    index = answer_tokens.to(logits.device)[None, :, None].expand(logits.shape[0], -1, 1)
    answer_logits = logits.gather(dim=-1, index=index)
    return (logits > answer_logits).sum(dim=-1) + 1
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_answer_ranks(answer_ranks)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Use `logits.gather(dim=-1, index=...)` to pick out each answer's logit (you'll need an index of shape `(layer, batch, 1)`), then compare it against the full logit vector with
broadcasting.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - the "which capital?" control

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Suppose the rank of `" Berlin"` drops from 20,000 to 1 over the layers. Does that mean the model has worked out that the capital of Germany is Berlin? Not necessarily. It might
only have worked out that *the next token is a capital city*, in which case `" Madrid"` and `" Oslo"` would also be climbing the ranks. To tell these apart, we need a
**mismatched-answer control**: for each prompt, how highly ranked are the *other* prompts' answers?

Implement `other_answer_ranks`. For prompt `b` and each layer, compute the rank of every *other* prompt's answer token in prompt `b`'s logits, and return the median of those ranks.
(You can reuse `answer_ranks` if you like, though a direct implementation is also fine.)
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def other_answer_ranks(
    logits: Float[Tensor, "layer batch d_vocab"], answer_tokens: Int[Tensor, " batch"]
) -> Float[Tensor, "layer batch"]:
    """
    For each layer and prompt b, returns the median rank of the answers of all prompts other than b.
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    answer_tokens = answer_tokens.to(logits.device)
    batch = logits.shape[1]
    # all_answer_logits[l, b, j] = logit of answer j in prompt b's logits at layer l
    all_answer_logits = logits[:, :, answer_tokens]
    # ranks[l, b, j] = rank of answer j in prompt b's logits at layer l
    ranks = (logits[:, :, None, :] > all_answer_logits[..., None]).sum(dim=-1) + 1
    off_diagonal = ~t.eye(batch, dtype=t.bool, device=logits.device)
    other_ranks = ranks[:, off_diagonal].reshape(logits.shape[0], batch, batch - 1)
    return other_ranks.float().median(dim=-1).values
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_other_answer_ranks(other_answer_ranks, answer_ranks)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`logits[:, :, answer_tokens]` has shape `(layer, batch, batch)`, where entry `[l, b, j]` is the logit that prompt `b` assigns to prompt `j`'s answer. Compare this against the full
logit vector (careful with memory: `(layer, batch, batch, d_vocab)` booleans for 13 layers and 48 prompts is about 1.5GB, which is fine on most machines, but you can loop over
layers if it isn't). Then drop the diagonal `j == b` and take a median.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
We'll add one more baseline: **random vectors**. For each prompt and layer, we replace the residual stream vector with a random Gaussian vector *of the same norm*, and decode that.
If the answer's rank improves for random vectors too, then the "improvement" comes from the readout and not from anything the model computed. We've written this one for you, since
it doesn't teach anything new.

Now let's make the plot. We'll only use prompts that the model gets right, since there's no "answer emerging" to look at otherwise.

**Make a prediction first:** at which layer will the median rank of the correct capital first reach 1, and will the other countries' capitals (the control) get anywhere near it?
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def random_vectors_like(x: Float[Tensor, "... d_model"], seed: int = 0) -> Float[Tensor, "... d_model"]:
    """Random Gaussian vectors with the same norm as each vector in x."""
    generator = t.Generator(device="cpu").manual_seed(seed)
    noise = t.randn(x.shape, generator=generator).to(x.device)
    return noise / noise.norm(dim=-1, keepdim=True) * x.norm(dim=-1, keepdim=True)


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/logit_lens_ranks.png" width="800">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret this plot?</summary>

There are four phases, and they match what we saw in the Germany table.

**Embedding to layer 5: nothing.** The correct capital, the other capitals and the random baseline all sit around rank 10,000 or worse (the vocabulary has 50,257 tokens). Whatever
the early blocks are doing, it isn't visible to the final readout. That doesn't mean they're doing nothing; it means that what they're doing isn't expressed in the directions the
unembedding reads.

**Layers 6 to 8: "it's a capital city".** The correct capital *and* the other countries' capitals improve together, by more than an order of magnitude, and the two curves are
almost on top of each other. The model has worked out that the next token should be a capital city, but at this point it doesn't prefer the right one. If we'd only plotted the
correct answer, we'd have wrongly concluded that the model "knows Berlin" by layer 7.

**Layer 9: "it's Berlin".** The correct capital jumps from a median rank of a few hundred to rank 1 in a single block, while the other capitals stay around rank 150.
Country-specific recall becomes visible in the output basis all at once.

**Final layers: suppressing the competition.** Look at the other capitals' curve after layer 9: it goes back *up*, to rank ~700 at the output. The last blocks don't just promote
the right answer, they push the wrong capitals back down.

The random-vector baseline never improves, which tells us the readout itself isn't somehow favouring capital cities. The improvement comes from the vectors the model computed.

Bear in mind that all of this is a statement about what's *linearly readable by the final unembedding*. Information about "Berlin" could exist earlier in some other basis. That's
precisely the worry the tuned lens addresses.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Measuring the gap between two distributions: KL divergence

Ranks only look at one token. To compare *whole* next-token distributions we need a distance between two probability vectors. The standard one is the **KL divergence**
$D_{KL}(P \| Q) = \sum_v P(v) \log \frac{P(v)}{Q(v)}$: the average surprise you'd feel if you expected $Q$ and the world followed $P$.

Take a fair coin $Q = (0.5, 0.5)$ and a lopsided one $P = (0.9, 0.1)$. Then $D_{KL}(P \| Q) = 0.9 \ln\frac{0.9}{0.5} + 0.1 \ln\frac{0.1}{0.5} = 0.37$ nats. Swap them and you get
$D_{KL}(Q \| P) = 0.51$ nats: **KL is not symmetric**, and the order matters. The first argument says whose probabilities do the weighting. We'll always put the model's final distribution first, so that the lens is
punished for missing the tokens the model really puts weight on.

Go deeper: 0.7 Foundations Toolkit, Part 1.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - KL divergence to the final distribution

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

A complete measure of "how close is layer $\ell$ to the final answer?" is the KL divergence between the final distribution $p_L$ and the decoded
distribution $p_\ell$:

$$
D_{KL}(p_L \,\|\, p_\ell) = \sum_{v} p_L(v) \left(\log p_L(v) - \log p_\ell(v)\right)
$$

We put $p_L$ first because we care about how well the lens predicts *the tokens the model actually puts weight on*. This is also the quantity we'll train the tuned lens to minimise
in the next section.

Implement `kl_to_final`, which takes logits of shape `(layer, batch, d_vocab)` (where the last layer is the model's actual output), and returns the mean KL divergence over the
batch, for each layer.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def kl_to_final(logits: Float[Tensor, "layer batch d_vocab"]) -> Float[Tensor, " layer"]:
    """
    Returns KL(final || layer), averaged over the batch, for every layer. The last layer is treated as final.
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    logprobs = logits.log_softmax(dim=-1)
    final = logprobs[-1:]
    return (final.exp() * (final - logprobs)).sum(dim=-1).mean(dim=-1)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_kl_to_final(kl_to_final)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Use `log_softmax` rather than `softmax` followed by `log`, for numerical stability. Keep a dimension of size 1 on the final layer's log-probabilities so that it broadcasts against
every layer.
</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

kl = kl_to_final(lens_logits)
for label, value in zip(LAYER_LABELS, kl.tolist()):
    print(f"{label:>6}: {value:6.2f} nats")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> embed:  36.85 nats
 pre 1:  13.21 nats
 pre 2:  13.00 nats
 pre 3:  13.57 nats
 pre 4:  13.71 nats
 pre 5:  12.34 nats
 pre 6:   9.57 nats
 pre 7:   8.06 nats
 pre 8:   6.76 nats
 pre 9:   2.48 nats
pre 10:   5.14 nats
pre 11:   2.96 nats
 final:   0.00 nats</pre>

</details>

<details>
<summary>Question - why is the embedding layer's KL so enormous, and why does layer 10 do <i>worse</i> than layer 9?</summary>

**The embedding layer.** As we saw in the table, the logit lens confidently predicts junk tokens at the embedding layer. The final distribution puts essentially no probability on
`' destro'`, and the lens puts essentially no probability on the tokens the model actually predicts. KL divergence punishes confident disagreement extremely heavily.

**Layer 10.** At layer 10 the lens puts probability 0.81 on `' Berlin'`, but the final distribution only puts 0.25 on it, with the rest spread over other German cities. The lens is
*more* confident than the model ends up being, and that over-confidence costs KL even though the top token is correct. KL and rank measure different things: rank asks "is the
answer on top?", KL asks "does the whole distribution match?". You'll usually want to look at both.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## The attention lens: which heads write the answer?

The logit lens decodes the *whole* residual stream. But the stream is a sum: the embedding, plus the output of every attention head, plus the output of every MLP, plus a few
biases. The final layernorm centres that sum and divides it by one scale, and the unembedding is linear, so the final logit of a token is itself a sum with one term per thing that
wrote to the stream. Read a single head's output through the unembedding and you get the **attention lens**: how much that head, acting directly, adds to the logit of the answer.
It's the direct logit attribution from [1.2], one head at a time.

There's one choice to get right, and it's the *opposite* of the logit lens. A logit lens asks "what would the model say if it stopped here?", so each vector is divided by its own
scale. Here we want the pieces to add up to the logit the model really produced, so every head is divided by the **final** layernorm's scale at the last position, which the cache has
saved as `ln_final.hook_scale`.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement the attention lens

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15 minutes on this exercise.
> ```

Implement `attention_lens`, which takes the cache of a forward pass and one answer token per prompt, and returns a tensor of shape `(n_layers, n_heads, batch)`: how much each
head adds to that prompt's answer logit at the **final position**.

For each head, take its output at the last position (`cache["z", layer][:, -1]` has shape `(batch, head, d_head)`), push it through that head's `W_O` to get a vector in the residual
stream, centre it over `d_model`, divide by the final layernorm's scale `cache["ln_final.hook_scale"][:, -1]`, and dot it with the answer's column of `W_U`.

The test checks the contract that makes this useful: your heads, plus every MLP, plus the embeddings and the biases, must add up to the model's real logit for the answer.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def attention_lens(
    model: HookedTransformer, cache: ActivationCache, answer_tokens: Int[Tensor, " batch"]
) -> Float[Tensor, "layer head batch"]:
    """
    Returns each attention head's direct contribution to the logit of answer_tokens[b] at the final position,
    read through the final layernorm's own scale (so that all contributions add up to the real logit).
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    z = t.stack([cache["z", layer][:, -1] for layer in range(model.cfg.n_layers)])  # (layer, batch, head, d_head)
    head_out = t.einsum("lbhd,lhdm->lhbm", z, model.W_O)  # each head's write to the residual stream
    head_out = head_out - head_out.mean(dim=-1, keepdim=True)
    scale = cache["ln_final.hook_scale"][:, -1]  # (batch, 1)
    answer_directions = model.W_U[:, answer_tokens.to(model.W_U.device)]  # (d_model, batch)
    return t.einsum("lhbm,mb->lhb", head_out / scale, answer_directions)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_attention_lens(attention_lens, model)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`t.einsum("lbhd,lhdm->lhbm", z, model.W_O)` applies every head's `W_O` at once (`model.W_O` has shape `(layer, head, d_head, d_model)`). After centring over the last dimension, divide by the
scale (shape `(batch, 1)`, so it broadcasts over the `(layer, head, batch, d_model)` tensor) and contract `d_model` with `model.W_U[:, answer_tokens]`.
</details>

<details>
<summary>Help - the test says the pieces don't add up to the final logit.</summary>

Check the scale first. Each head's output is much smaller than the whole residual stream, so dividing it by *its own* norm (or not dividing at all) gives numbers of the wrong size. The only
scale that makes the terms add up is the one the final layernorm really used: `cache["ln_final.hook_scale"]`, taken at the last position.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Which heads write the right capital most strongly, averaged over the prompts the model gets right?
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

head_effects = attention_lens(model, cache, answer_tokens)
mean_effect = head_effects[:, :, right].mean(dim=-1)
print("Heads that add most to the correct capital's logit (mean over the prompts the model gets right):")
for value, flat in zip(*mean_effect.flatten().topk(5)):
    layer, head = divmod(flat.item(), model.cfg.n_heads)
    print(f"  L{layer}H{head}: {value:+.2f}")
final_answer_logit = logits[right, -1].gather(-1, answer_tokens[right, None])[:, 0].mean()
print(f"All {model.cfg.n_layers * model.cfg.n_heads} heads together: {mean_effect.sum():+.2f}.  The correct capital's final logit: {final_answer_logit:.2f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">FILL_ATTN</pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

TODO_ATTN_READING
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 1️⃣

The logit lens is free: no training, no extra parameters, and it often gives very readable results on GPT-2. We've seen that:

- The correct answer only becomes readable in the second half of the network, and then very suddenly (at layer 9).
- Before the specific answer appears, the *type* of the answer (a capital city) appears. Without the mismatched-answer control, we'd have conflated the two.
- The final layers actively push competing capitals back down.
- Random vectors with matched norms never decode to the answer, so the readout isn't doing the work.

But we've also seen its main weakness: the embedding layer decodes to junk and the early layers decode to generic filler, and we can't tell whether that's because the information
isn't there yet, or because it's there in a different basis. The next section addresses this directly.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 2️⃣ The Tuned Lens
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Representational drift

The logit lens assumes that every layer's residual stream lives in the same coordinate system as the final layer. There are good reasons to expect this to be only approximately
true:

- Some directions in the residual stream are used as *scratch space* that later layers read and overwrite, but which never directly affect the output.
- The model may store information about the next token in a format that is easy for the next few blocks to use, rather than in the format the unembedding reads. For example, the residual stream might encode "the answer is the capital of the country three tokens back", which the final layers then *resolve* into a specific token.
- The residual stream norm grows over the network, and different layers can have systematically different means (bias directions) which the final layernorm wasn't trained to remove.

Belrose et al. ([*Eliciting Latent Predictions from Transformers with the Tuned Lens*](https://arxiv.org/abs/2303.08112), 2023) call this **representational drift**, and they correct for it with a learned **translator** for each layer: an affine map applied *before* the logit lens readout.

$$
\text{TunedLens}_\ell(x_\ell) = \text{LogitLens}\left(x_\ell + A_\ell x_\ell + b_\ell\right)
$$

$A_\ell$ is a $d_{model} \times d_{model}$ matrix and $b_\ell$ is a bias, and both are initialised to zero, so the tuned lens *starts* as the logit lens and only moves away from it
as much as the training signal demands. The translators are trained to make each layer's decoded distribution match the model's actual output, by minimising

$$
\mathcal{L} = \frac{1}{L} \sum_{\ell=0}^{L-1} \mathbb{E}_{\text{tokens}} \left[ D_{KL}\left(p_L \,\|\, \text{TunedLens}_\ell(x_\ell)\right) \right]
$$

over a corpus of ordinary text. Importantly, the model itself is frozen: we only train the translators.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Aside - isn't this just a probe? Why is it called a lens?</summary>

It is a probe, in the sense that it's a learned linear map from activations to something we care about. The differences are in the target and the constraint:

- A typical probe is trained to predict a *label we chose* (e.g. "is this sentence true?"). The tuned lens is trained to predict *the model's own output distribution*, so it only ever reports information in terms the model itself uses.
- The translator is initialised at the identity and composed with the model's own unembedding, rather than being a fresh classifier. That keeps it close to the logit lens and makes it much harder for it to "learn the task itself" from scratch.

That said, the probe worry doesn't go away entirely. With $d_{model}^2 \approx 590{,}000$ parameters per layer, a translator could learn to extract things the model doesn't
actually use downstream. We'll build a control for exactly this below.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement the `TunedLens` module

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement the `TunedLens` module. It should have two parameters:

- `weight`, of shape `(n_layers, d_model, d_model)`, initialised to zeros,
- `bias`, of shape `(n_layers, d_model)`, initialised to zeros,

and its `forward` method should take a residual stream tensor of any shape `(..., d_model)` plus a layer index, and return `resid + resid @ weight[layer] + bias[layer]`. (We use
the convention `resid @ W` rather than `W @ resid`, to match the rest of TransformerLens.)

The `logits` method, which applies the translator followed by the logit lens readout, is written for you.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class TunedLens(nn.Module):
    def __init__(self, n_layers: int, d_model: int):
        super().__init__()
        # EXERCISE
        # raise NotImplementedError()
        # END EXERCISE
        # SOLUTION
        self.weight = nn.Parameter(t.zeros(n_layers, d_model, d_model))
        self.bias = nn.Parameter(t.zeros(n_layers, d_model))
        # END SOLUTION

    def forward(self, resid: Float[Tensor, "... d_model"], layer: int) -> Float[Tensor, "... d_model"]:
        # EXERCISE
        # raise NotImplementedError()
        # END EXERCISE
        # SOLUTION
        return resid + resid @ self.weight[layer] + self.bias[layer]
        # END SOLUTION

    def logits(self, model: HookedTransformer, resid: Float[Tensor, "... d_model"], layer: int) -> Float[Tensor, "... d_vocab"]:
        return model.unembed(model.ln_final(self.forward(resid, layer)))


# HIDE
if MAIN:
    tests.test_tuned_lens_init(TunedLens, model)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - why the residual parameterisation (resid + resid @ A) rather than just resid @ A?</summary>

With $A_\ell$ initialised at zero, `resid + resid @ A` starts as the identity map, so the tuned lens starts exactly at the logit lens. The optimiser then only has to learn a
*correction*. If we used `resid @ A` with $A$ initialised at the identity matrix, the function would be the same at initialisation, but weight decay and the optimiser's dynamics
would pull it towards zero rather than towards the identity. The residual form is also what the original paper uses.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement the tuned lens loss

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15 minutes on this exercise.
> ```

Implement `tuned_lens_loss`. It takes:

- `resid_stack` of shape `(n_layers, batch, seq, d_model)`, the residual stream before every block for a batch of text,
- `final_logprobs` of shape `(batch, seq, d_vocab)`, the model's actual output log-probabilities on the same text,

and returns the KL divergence $D_{KL}(p_L \| p_\ell)$, **summed over the vocabulary, averaged over batch and sequence positions, then averaged over layers**.

Exclude sequence position 0 from the average. Position 0 is always the BOS token, and the model's prediction there doesn't depend on any context, so it would just add noise.

Memory tip: the decoded logits for one layer have shape `(batch, seq, d_vocab)`, which for a batch of 8 sequences of length 64 is already about 100MB. Loop over layers and add up
the losses, rather than decoding all layers at once.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

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
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    n_layers = resid_stack.shape[0]
    target = final_logprobs[:, 1:]
    total = 0.0
    for layer in range(n_layers):
        lens_logprobs = lens.logits(model, resid_stack[layer, :, 1:], layer).log_softmax(dim=-1)
        total = total + (target.exp() * (target - lens_logprobs)).sum(dim=-1).mean()
    return total / n_layers
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_tuned_lens_loss(tuned_lens_loss, TunedLens, model)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

For each layer, compute `lens.logits(model, resid_stack[layer, :, 1:], layer).log_softmax(-1)`, then use the same KL formula as in `kl_to_final`, averaging over both batch and
position with `.mean()`.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Training data

We need a corpus of ordinary text to train on. We'll use the validation split of [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), which is a single small file. It's simpler text than GPT-2 was trained on, but that doesn't matter much here: we're not training the model, only learning to *read* it, and the target is always whatever GPT-2 itself predicts.

We take 1,600 sequences of 64 tokens for training and a separate 200 for evaluation. The evaluation sequences are never used for training, so all the numbers we report about the
tuned lens are held-out numbers.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

all_tokens = utils.load_tinystories_tokens(model, n_sequences=1800, seq_len=64)
train_tokens, test_tokens = all_tokens[:1600], all_tokens[1600:]
print(train_tokens.shape, test_tokens.shape)
print(repr(model.to_string(train_tokens[0, 1:40])))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
We also need a helper which runs the model on a batch of tokens and returns the residual stream stack and the final log-probabilities. We only need `resid_pre`, so we use
`names_filter` to avoid caching everything else. (We use `t.no_grad()` rather than `t.inference_mode()` here: the tensors this returns become inputs to the lens's training graph,
and PyTorch doesn't allow inference-mode tensors to be saved for backward.)
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.no_grad()
def get_resid_and_logprobs(
    model: HookedTransformer, tokens: Int[Tensor, "batch seq"]
) -> tuple[Float[Tensor, "layer batch seq d_model"], Float[Tensor, "batch seq d_vocab"]]:
    logits, cache = model.run_with_cache(tokens.to(model.cfg.device), names_filter=lambda name: name.endswith("resid_pre"))
    resid_stack = t.stack([cache["resid_pre", layer] for layer in range(model.cfg.n_layers)])
    return resid_stack, logits.log_softmax(dim=-1)


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - train the tuned lens

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Write a standard training loop. For each step, sample a random batch of `batch_size` sequences from `tokens`, get the residual stream and final log-probabilities with
`get_resid_and_logprobs`, compute `tuned_lens_loss`, and take an optimiser step. Use Adam with the given learning rate. Return the list of losses.

The defaults (300 steps, batch size 8) take about a minute on a GPU. If you're on a CPU, use around 100 steps: the lens will be a bit worse, but all the qualitative conclusions
below still hold.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

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
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
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
    # END SOLUTION


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# HIDE
if MAIN:
    tests.test_train_tuned_lens(train_tuned_lens)
# END HIDE

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/tuned_lens_training.png" width="500">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - my loss starts at around 7 and plateaus near 0.6. Is that right?</summary>

Yes. At initialisation the tuned lens *is* the logit lens, so the initial loss is the mean logit-lens KL over layers, which is dominated by the huge KL at the embedding layer. The
loss drops very quickly in the first ~50 steps (mostly from fixing the embedding layer) and then decreases slowly. You won't get it to zero: the early layers genuinely don't
contain enough information to predict what the final layer will say.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - evaluate both lenses on held-out text

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15 minutes on this exercise.
> ```

Now for the main comparison. Implement `evaluate_lenses`, which goes through `tokens` in batches and returns a dictionary of per-layer metrics (each a list with one float per layer
`0..n_layers-1`):

- `"logit_kl"` and `"tuned_kl"`: mean KL from the final distribution, as in `kl_to_final`,
- `"logit_agree"` and `"tuned_agree"`: the fraction of positions where the lens's top-1 token equals the model's final top-1 token.

As before, exclude sequence position 0. Average over **all evaluated tokens**, counting every non-BOS position equally. The final batch can be shorter than `batch_size`: sum the
tokenwise KL values and correct predictions, then divide each total by the number of evaluated tokens. An unweighted average of batch means would over-count a short final batch.
Require nonempty sequences with at least two positions and a positive batch size.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.inference_mode()
def evaluate_lenses(
    model: HookedTransformer, lens: TunedLens, tokens: Int[Tensor, "n_seq seq"], batch_size: int = 20
) -> dict[str, list[float]]:
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
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
    # END SOLUTION


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# HIDE
if MAIN:
    tests.test_evaluate_lenses(evaluate_lenses)
# END HIDE

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

results = evaluate_lenses(model, tuned_lens, test_tokens)
display(pd.DataFrame(results).rename_axis("layer").round(3).T)
utils.plot_lens_comparison(
    results["logit_kl"],
    results["tuned_kl"],
    results["logit_agree"],
    results["tuned_agree"],
    floor_agree=results["tuned_agree"][0],
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/lens_comparison.png" width="900">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output (approximately)</summary>

On held-out text, the tuned lens should be dramatically better at every layer. Your numbers will vary a little with the random seed and the number of training steps, but should be
close to these (from a 300-step run on a GPU):

| layer | logit lens KL | tuned lens KL | logit lens top-1 | tuned lens top-1 |
|---|---|---|---|---|
| 0 (embed) | ~45 | ~2.1 | 0.00 | ~0.37 |
| 1 | ~5.8 | ~1.15 | ~0.06 | ~0.46 |
| 6 | ~4.3 | ~0.65 | ~0.10 | ~0.60 |
| 11 | ~1.1 | ~0.06 | ~0.62 | ~0.86 |

If your numbers are in this ballpark, your implementation is working.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### What did the tuned lens actually learn?

The improvement is huge: at layer 1, the logit lens agrees with the model's final prediction about 6% of the time, and the tuned lens about 46% of the time. It's tempting to
conclude that "GPT-2 already knows the next token half the time after one block". Before we believe that, let's look at the dashed line.

The **layer-0 tuned lens** reads the residual stream before *any* block has run. That vector is just the embedding of the current token plus its position embedding. So whatever the
layer-0 tuned lens predicts is a function of the current token and its position alone, with no context at all. It reaches roughly 37% agreement with GPT-2's final predictions. We'll
call this the **layer-0 floor**: what a trained reader can get from the current token and position, before the model has done anything.

That reframes the result. The tuned lens at layer 1 isn't "46% from nothing", it's "46% against a 37% floor". The gap between the dashed line and the tuned lens curve is the
part that requires *context*, and that grows steadily through the network. This is the honest version of the claim: the tuned lens shows that next-token information accumulates
gradually over the layers, while the logit lens makes it look like it all appears at the end.

(The layer-0 floor sees positions too, so it isn't quite a bigram model. Later in this part we'll build a real bigram table and compare.)
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - the wrong-layer control

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Another way to check that the translators have learned something specific to *each layer's* representation (rather than one generic "make the logit lens better" correction) is to
apply them to the wrong layer.

Implement `shifted_translator_agreement`, which evaluates top-1 agreement (on held-out tokens, excluding position 0) when the residual stream at layer $\ell$ is decoded with the
translator for layer $\ell + \text{shift}$. Only evaluate layers where $\ell + \text{shift}$ is a valid layer index; return a dict mapping each evaluated layer $\ell$ to its
agreement. Count correct predictions and evaluated tokens across batches, including a short final batch, before dividing. Use the same input requirements as `evaluate_lenses`.

If the translators really are layer-specific, the shifted versions should do noticeably worse than the correct ones, though probably still better than the logit lens, since
adjacent layers have similar representations.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.inference_mode()
def shifted_translator_agreement(
    model: HookedTransformer, lens: TunedLens, tokens: Int[Tensor, "n_seq seq"], shift: int, batch_size: int = 20
) -> dict[int, float]:
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
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
    # END SOLUTION


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# HIDE
if MAIN:
    tests.test_shifted_translator_agreement(shifted_translator_agreement)
# END HIDE

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret this table?</summary>

The correct translator is the best at every layer, typically by 0.15-0.25. Shifted translators still beat the logit lens in the middle of the network (neighbouring layers really do
share a lot of structure), but they're clearly worse than the correct one.

The most striking entries involve the embedding layer. The layer-3 translator applied to the embeddings gets **0.00** agreement (the correct layer-0 translator gets 0.37), and the
layer-0 translator applied to layer 3 also gets 0.00. The embedding layer's representation is simply in a different place from everything after it, so a translator that was fit to
one is useless on the other.

If a shifted translator had done *as well as* the correct one everywhere, that would suggest the tuned lens was mostly learning one generic correction (e.g. a fixed bias) rather
than layer-specific structure, and we'd be suspicious of reading too much into its layer-by-layer curve.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Back to the capitals

Our tuned lens was trained on children's stories, not on factual recall. Let's see what it makes of the capitals prompts anyway: does it find `" Berlin"` earlier than the logit
lens did? Make a prediction before running the cell.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/capitals_tuned_vs_logit.png" width="800">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret this plot?</summary>

The tuned lens curves lie almost exactly on top of the logit lens curves. On this task, the tuned lens **doesn't find the answer any earlier**: both lenses see nothing until layer
5, both see "a capital" over layers 6-8, and both see the specific capital appear at layer 9.

That might be surprising after the big improvement on held-out TinyStories text, so it's worth thinking about why.

- **Most of the tuned lens's gains are on common tokens.** Averaged over a corpus, top-1 agreement is dominated by frequent, predictable tokens (punctuation, `" the"`, `" was"`, the second half of a common word). The layer-0 floor tells us how much of that is available from the current token and position alone. A rare, fact-dependent answer like `" Canberra"` is a very different kind of prediction.
- **Distribution shift.** The translators were fit on children's stories. There are no capital cities in TinyStories, so nothing pushed the translators to read out capital-city information early, even if it's there.
- **Maybe the information really isn't there.** It's entirely possible that GPT-2 Small simply hasn't retrieved the specific capital before layer 9, and that both lenses are correctly reporting this.

We can't distinguish these explanations with lenses alone. That's a good reason to try a method which doesn't rely on reading out a next-token prediction at all, which is what
Patchscopes do. On the positive side, the agreement between the two lenses means the "type first, then specific answer" story from section 1️⃣ isn't an artefact of the logit lens's
mismatched basis.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## A third reader: nearest neighbours in a bank of real contexts

Both lenses so far read a state by pushing it through the model's unembedding, with or without a translator. Here's a reader with no unembedding at all: **describe a state by its
nearest neighbours.** Collect a *bank* of residual stream vectors from 600 TinyStories stories (about 37,000 vectors per layer), and for each one remember the token it came from and the token
that followed it in its story. To read a new state, find the bank vector that points most nearly the same way, and say: "this state is like that one". We'll call it the
**contextual lens**. It is the idea behind LatentLens, which we meet in the Bonus.

## Cosine nearest neighbours

**Cosine similarity** compares the *directions* of two vectors and ignores their lengths: $\cos(a, b) = \frac{a \cdot b}{\|a\|\,\|b\|}$. It is 1 for the same direction, 0 at right angles and
$-1$ for opposite directions. Take a query $(3, 4)$. The bank vector $(6, 8)$ has cosine 1, even though it is twice as long. $(4, -3)$ has cosine 0 and $(-3, -4)$ has cosine $-1$. Because lengths
don't matter, the search isn't fooled by the residual stream's growing norm.

The trap is that GPT-2's states all lean the same way: the cosine between *any two* states is about 0.6. So "the nearest neighbour has cosine 0.9" means much less than it sounds, and the right
yardstick is a search with random vectors of the same norm. We'll measure all of this below.

Go deeper: 0.7 Foundations Toolkit, Part 3.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - normalise rows

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 5 minutes on this exercise.
> ```

Implement `normalize_rows`, which divides every row of a matrix (every vector along the last dimension) by its length, so that every row has length 1. A row of zeros has no direction: it
should stay zeros rather than become `NaN`, so clamp the length from below by `eps`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def normalize_rows(x: Float[Tensor, "... d"], eps: float = 1e-8) -> Float[Tensor, "... d"]:
    """Scales every vector along the last dimension to length 1 (all-zero vectors stay zero)."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_normalize_rows(normalize_rows)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`x.norm(dim=-1, keepdim=True)` gives each row's length with shape `(..., 1)`, which broadcasts against `x`. Use `.clamp_min(eps)` on the length before dividing.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - find the nearest bank vectors

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Implement `contextual_topk`, which takes `queries` of shape `(n_queries, d)` and a `bank` of shape `(n_bank, d)`, and returns the cosine similarity and the index of each query's `k` most
similar bank vectors, best first. Two details matter:

- **Chunk the queries.** We'll search 6,000 queries against 37,000 bank vectors, and one score matrix for that is almost 1 GB. Process `chunk_size` queries at a time.
- **Break ties the same way every time.** When two bank vectors are equally similar, the one with the lower index wins. (`topk` doesn't promise this, but a stable sort does.)
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def contextual_topk(
    queries: Float[Tensor, "n_queries d"], bank: Float[Tensor, "n_bank d"], k: int = 1, chunk_size: int = 1024
) -> tuple[Float[Tensor, "n_queries k"], Int[Tensor, "n_queries k"]]:
    """
    Returns (cosine similarities, bank indices) of the k nearest bank vectors for every query, best first.
    Ties go to the lower bank index. Queries are processed `chunk_size` at a time.
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    bank = normalize_rows(bank)
    all_scores, all_indices = [], []
    for start in range(0, len(queries), chunk_size):
        scores = normalize_rows(queries[start : start + chunk_size]) @ bank.T  # (chunk, n_bank)
        order = scores.sort(dim=-1, descending=True, stable=True)
        all_scores.append(order.values[:, :k])
        all_indices.append(order.indices[:, :k])
    return t.cat(all_scores), t.cat(all_indices)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_contextual_topk(contextual_topk)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

For each chunk, normalise the queries and compute `scores = chunk @ bank.T` (the bank can be normalised once, before the loop). Then `scores.sort(dim=-1, descending=True, stable=True)`
returns `.values` and `.indices`; keep the first `k` columns of each, and concatenate the chunks at the end.
</details>

<details>
<summary>Help - my answer is right except when two bank vectors tie.</summary>

`scores.topk(k)` is allowed to return tied entries in any order. A stable descending sort keeps tied entries in their original order, which puts the lower bank index first.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - compare every query layer with every bank layer

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Do the states of different layers share a language? Test it by searching a bank built at layer $j$ with queries taken from layer $i$. Implement `layer_alignment_grid`, which takes
`query_states` of shape `(n_layers, n_queries, d)`, `bank_states` of shape `(n_layers, n_bank, d)`, a target label for each query (`query_targets`, shape `(n_queries,)`) and a label for
each bank vector (`bank_labels`, shape `(n_bank,)`). Entry `[i, j]` of the returned `(n_layers, n_layers)` grid is the fraction of layer-`i` queries whose single nearest layer-`j` bank vector
carries a label equal to the query's target.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

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
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    n_query_layers, n_bank_layers = query_states.shape[0], bank_states.shape[0]
    grid = t.zeros(n_query_layers, n_bank_layers)
    for i in range(n_query_layers):
        for j in range(n_bank_layers):
            _, nearest = contextual_topk(query_states[i], bank_states[j], k=1, chunk_size=chunk_size)
            grid[i, j] = (bank_labels[nearest[:, 0]] == query_targets).float().mean().item()
    return grid
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_layer_alignment_grid(layer_alignment_grid)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Two nested loops over layers, and inside them one call to `contextual_topk` with `k=1`. The nearest bank index is `nearest[:, 0]`; look up its label with `bank_labels[...]` and compare with
`query_targets`.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Now we need the bank and the queries. We write two plumbing helpers for you. `collect_states` runs the model and keeps the residual stream before every block, together with the
model's own final top-1 prediction at each position. `bigram_predictions` builds a real bigram table from the bank: for each query token, the token that most often followed it in the
bank's stories. That is our **bigram floor**, the best you can do from the current token alone, with no position and no state.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

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


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Every query is a held-out state, and the question we ask of each reader is the same one we asked of the lenses: **does it predict the token the model itself will predict?** For a lens, that's
its top-1 token. For the contextual lens it's the token that followed the nearest bank vector in its own story. We'll score all of them against the model's final top-1 on the same
6,300 held-out states.

The controls sit on the same axes:

- **bigram floor**: predict the token that most often followed the current token in the bank;
- **layer-0 floor**: the layer-0 tuned lens from before;
- **shuffled bank labels**: the real neighbours, but with the bank's "next token" labels shuffled among the bank vectors;
- **norm-matched random queries**: random vectors with the same norms as the queries, searched against the same bank.

**Make a prediction first:** will the neighbour's next token match the model's prediction better than the bigram floor does, and will the contextual lens beat the tuned lens?
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">FILL_BRIDGE</pre>

</details>

Now the heatmap. For each layer of queries and each layer of the bank we ask a simpler question than next-token prediction: is the nearest neighbour *the same token* as the query? (Searching
every layer pair with all 6,300 queries would take a while, so we use 1,000 of them.) Then we put everything in one figure.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

subset = t.randperm(queries.shape[1], generator=generator)[:1000].to(device)
alignment = layer_alignment_grid(queries[:, subset], bank, query_current[subset], bank_current)
utils.plot_lens_signature(
    lens_table[["logit lens", "tuned lens", "contextual lens"]].to_dict("list"),
    controls=lens_table[["shuffled bank labels", "random queries"]].to_dict("list"),
    floors={"bigram floor": bigram_floor, "layer-0 floor": layer0_floor},
    alignment=alignment,
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/lens_signature.png" width="1000">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret this figure?</summary>

TODO_SIGNATURE_READING
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 2️⃣

- The tuned lens is a cheap, principled fix for representational drift: one affine translator per layer, initialised at the logit lens, trained to match the model's own output.
- On held-out text it beats the logit lens at every layer, by a large margin early on.
- But a large part of the early-layer improvement is bigram statistics, which you can read off the layer-0 lens. The interesting signal is the gap above that floor.
- Translators are layer-specific: using the wrong layer's translator hurts, and the embedding layer's translator is useless anywhere else.
- Corpus-level improvements don't automatically transfer to a specific task: on the capitals prompts, the tuned lens tells the same story as the logit lens.

Both lenses share one limitation: they only tell us what can be read out *as a next-token prediction*. A hidden state might carry lots of information (that the subject is a
European country, that it's a monarchy, that its capital is Madrid) which never shows up in the next-token distribution at that position. Patchscopes give us a way to ask about
that information directly.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 3️⃣ Patchscopes
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Letting the model decode its own activations

Ghandeharioun et al.
([*Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models*](https://arxiv.org/abs/2401.06102), 2024)
observed that we already have a very powerful decoder for hidden states: the model itself. If we want to know
what a hidden state $h$ "means", we can put it into a forward pass on a carefully chosen **inspection
prompt**, and read off what the model says.

A Patchscope is specified by:

- a **source**: a prompt $S$, a position $i$ and a layer $\ell$, which together determine the hidden state $h = x_\ell^{S}[i]$ that we want to inspect;
- a **target**: a prompt $T$, a position $i^*$ and a layer $\ell^*$, where we *overwrite* the target's residual stream with $h$, i.e. set $x_{\ell^*}^{T}[i^*] \leftarrow h$, and then let the forward pass continue;
- a **readout**: what we measure from the patched forward pass, e.g. the next-token distribution at the final position.

For example, to ask "what capital city does this hidden state represent?", we can use the target prompt

```
The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is Rome. The capital of x is
```

and patch $h$ into the position of `" x"`. If $h$ is a representation of Germany, a model which can do the few-shot task should now say `" Berlin"`. The placeholder `" x"` is
deliberately meaningless: without any patching, the model has no way to know which country we mean.

You met this idea applied to SAE latents in [1.3.3] [patch scoping](https://learn.arena.education/chapter1_transformer_interp/13_saes/2-understanding-latents-a-deeper-dive/#patch-scoping).

<details>
<summary>Aside - the logit lens is a Patchscope</summary>

The logit lens is the Patchscope with $T = S$, $i^* = i$, $\ell^* = L$ (i.e. patching into the very end of the network, so no further computation happens) and next-token readout.
Patchscopes generalise this in two ways: the target prompt can be different from the source (so we can ask a *specific question* of the hidden state), and the target layer can be
earlier than $L$ (so the model gets to *process* the hidden state before answering, rather than us applying a fixed linear readout).
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - extract source vectors

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 5 minutes on this exercise.
> ```

Implement `get_source_vectors`, which runs the model on a list of source prompts (all with the same number of tokens) and returns the residual stream *before* block `layer`
(`resid_pre`) at position `pos`, as a `(batch, d_model)` tensor.

Our source prompts will be as simple as possible: just the country name with a leading space, e.g. `" Germany"` (plus the BOS token which `to_tokens` prepends). So the source
vector at position `-1` is the model's representation of the country, with no other context.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.inference_mode()
def get_source_vectors(
    model: HookedTransformer, prompts: list[str], pos: int, layer: int
) -> Float[Tensor, "batch d_model"]:
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    _, cache = model.run_with_cache(model.to_tokens(prompts), names_filter=f"blocks.{layer}.hook_resid_pre")
    return cache["resid_pre", layer][:, pos].clone()
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_get_source_vectors(get_source_vectors, model)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
We'll want the source vectors at every layer, so here's a helper that stacks them (using one forward pass rather than twelve):
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.inference_mode()
def get_all_layer_source_vectors(
    model: HookedTransformer, prompts: list[str], pos: int
) -> Float[Tensor, "layer batch d_model"]:
    _, cache = model.run_with_cache(model.to_tokens(prompts), names_filter=lambda name: name.endswith("resid_pre"))
    return t.stack([cache["resid_pre", layer][:, pos] for layer in range(model.cfg.n_layers)]).clone()


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - run a Patchscope

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Implement `run_patchscope`. It takes a batch of source vectors of shape `(batch, d_model)`, a **single** target prompt as a token tensor of shape `(1, seq)`, a target position and
a target layer. It should:

1. Repeat the target tokens so there's one copy per source vector.
2. Run the model with a hook on `blocks.{target_layer}.hook_resid_pre` which overwrites position `target_pos` of copy `b` with source vector `b`.
3. Return the logits at the **final** position, of shape `(batch, d_vocab)`.

Use `model.run_with_hooks`, which removes the hooks automatically after the forward pass (the test checks that you don't leave a hook attached).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.inference_mode()
def run_patchscope(
    model: HookedTransformer,
    source_vectors: Float[Tensor, "batch d_model"],
    target_tokens: Int[Tensor, "1 seq"],
    target_pos: int,
    target_layer: int,
) -> Float[Tensor, "batch d_vocab"]:
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    batch = source_vectors.shape[0]

    def patch_hook(resid: Float[Tensor, "batch seq d_model"], hook) -> Float[Tensor, "batch seq d_model"]:
        resid[:, target_pos] = source_vectors
        return resid

    logits = model.run_with_hooks(
        target_tokens.expand(batch, -1),
        fwd_hooks=[(f"blocks.{target_layer}.hook_resid_pre", patch_hook)],
    )
    return logits[:, -1]
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_run_patchscope(run_patchscope, model)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Your hook function should take `(resid, hook)` and modify `resid[:, target_pos]` in place (then return it). Since `source_vectors` has shape `(batch, d_model)` and `resid[:,
target_pos]` also has shape `(batch, d_model)`, the assignment lines up one source vector per batch element. `target_tokens.expand(batch, -1)` gives you the repeated target without
copying memory.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Now let's set up the experiment. The target prompt is the few-shot prefix with `" x"` in place of the country, and the target position is the position of `" x"`. Our source prompts
are just the bare country names.

As a first sanity check, let's patch every country's layer-1 representation into layer 1 of the target, and see what comes out.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

target_tokens = model.to_tokens(make_prompt("x"))
target_pos = target_tokens.shape[1] - 2
assert model.to_string(target_tokens[0, target_pos]) == " x"

source_prompts = [" " + country for country in countries]
source_vectors = get_all_layer_source_vectors(model, source_prompts, pos=-1)

patched_logits = run_patchscope(model, source_vectors[1], target_tokens, target_pos, target_layer=1)
for country, capital, pred in list(zip(countries, [c for _, c in CAPITALS], patched_logits.argmax(-1)))[:10]:
    print(f"{country:<12} -> {model.to_string(pred)!r:<14} (true capital: {capital!r})")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output (approximately)</summary>

Most rows should name the right capital, and the mistakes should be the same "largest city" mistakes that the model makes on the real prompts:

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Germany      -> ' Berlin'      (true capital: 'Berlin')
Spain        -> ' Madrid'      (true capital: 'Madrid')
Russia       -> ' Moscow'      (true capital: 'Moscow')
China        -> ' Shanghai'    (true capital: 'Beijing')
...</pre>

</details>

That's remarkable when you think about it. The target prompt says nothing about Germany. The only way the model can produce `" Berlin"` is from the information in the 768 numbers
we patched in, and the model's own machinery decodes them into an answer to the question the target prompt asks.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - map accuracy across source and target layers

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15 minutes on this exercise.
> ```

Now let's be systematic. Implement `patchscope_accuracy_grid`, which takes source vectors for every source layer (shape `(n_source_layers, batch, d_model)`), a target prompt and
position, the answer tokens, and a list of target layers, and returns a `(n_source_layers, n_target_layers)` tensor where entry `[i, j]` is the top-1 accuracy of patching source
layer `i` into target layer `target_layers[j]`.

The test runs your function against a fake `run_patchscope` and checks it, entry by entry, against a plain loop.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.inference_mode()
def patchscope_accuracy_grid(
    model: HookedTransformer,
    source_vectors: Float[Tensor, "source_layer batch d_model"],
    target_tokens: Int[Tensor, "1 seq"],
    target_pos: int,
    answer_tokens: Int[Tensor, " batch"],
    target_layers: list[int],
) -> Float[Tensor, "source_layer target_layer"]:
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    grid = t.zeros(source_vectors.shape[0], len(target_layers))
    for i, vectors in enumerate(source_vectors):
        for j, target_layer in enumerate(target_layers):
            logits = run_patchscope(model, vectors, target_tokens, target_pos, target_layer)
            grid[i, j] = (logits.argmax(dim=-1) == answer_tokens.to(logits.device)).float().mean()
    return grid
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_patchscope_accuracy_grid(patchscope_accuracy_grid)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Before you run the grid, here is what it will show. Rows are the layers the country's state is read from, columns are the layers of the inspection prompt it is patched into.

**Make a prediction first:** which target layers will still decode the capital: all twelve, only the first few, or only the last few?
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

grid = patchscope_accuracy_grid(
    model, source_vectors, target_tokens, target_pos, answer_tokens, target_layers=list(range(model.cfg.n_layers))
)
utils.plot_patchscope_grid(
    grid, "Patchscope: which capital is this country?", ceiling=correct.float().mean().item()
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/patchscope_grid.png" width="600">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret this grid?</summary>

**What the grid shows.** Patching into target layers 1, 3, 4 and 5 works well for early source layers, reaching 75-79%, which matches what the model does on the real prompts (77%). Patching
into target layer 6 or later almost never works, whatever the source. The target layer matters more than the source layer.

**What the cutoff does and doesn't mean.** Patching at target layer $\ell^*$ means only blocks $\ell^*, \dots, L-1$ get to process the patched vector. In *this inspection prompt*, a country
patched in after block 5 can no longer be turned into its capital. So the grid is a map of where a donor vector can still be read, and it *suggests* that the recall happens in blocks 1-5.
It does not show that blocks 1-5 are *necessary* for the model's own recall. Here the model is handed a state from another prompt and asked a different question; in the native prompt the
later blocks might be able to do the lookup themselves. (Compare the logit lens, which only saw the answer at layer 9: it tells us when the answer arrives at the final position in the
output basis, which is a different thing from where it was computed.) The Bonus has an exercise that tests necessity directly.

**Target layer 0 is special.** Only the layer-0 source works at target layer 0. The residual stream before block 0 is just token plus position embeddings, which live in a different
region of space from processed residual vectors, so a processed vector patched in there is off-distribution for block 0.

**Later source layers work less well.** Source vectors from layers 1-5 decode best. From layer 6 onwards accuracy drops, and by layer 9-11 only target layer 2 still works
reasonably. The country's identity is still *there* in late source vectors (some target layer can read it), but it's gradually being reformatted into something the early blocks of
the target prompt don't expect.

**Target layer 2 is strange.** Look at the column for target layer 2: early source vectors (1-4) almost completely fail there (4-12%), even though they work at target layers 1 and
3 on either side. Yet late source vectors (8-11) work *better* at target layer 2 than anywhere else. We don't have an explanation for this. It's exactly the kind of anomaly that's
worth chasing (see the Bonus), and a good reminder that a Patchscope is an intervention on a real network, not a clean measuring device.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - controls for the Patchscope

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15-20 minutes on this exercise.
> ```

A Patchscope could look good for boring reasons. Maybe the target prompt alone already biases the model towards some capitals; maybe *any* vector with the right norm gets decoded
as a capital; maybe we'd get the same result from any position of the source prompt. Implement `patchscope_controls`, which returns the top-1 accuracy (against `answer_tokens`) for
each of these four conditions, at a single source layer and target layer:

- `"patchscope"`: the real thing, patching `source_vectors`.
- `"text only (no patch)"`: run the target prompt with nothing patched, so every country gets the same output.
- `"random vector (norm-matched)"`: patch `random_vectors_like(source_vectors)` instead.
- `"wrong position (BOS)"`: patch the source prompt's position-0 (BOS) vector, at the same source layer, instead of the country position. You'll need `get_source_vectors` with `pos=0`.

You'll need to pass in the source prompts so you can compute the last one.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

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
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
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
    # END SOLUTION


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

# HIDE
if MAIN:
    tests.test_patchscope_controls(patchscope_controls)
# END HIDE

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

controls = patchscope_controls(model, source_prompts, 4, target_tokens, target_pos, 1, answer_tokens)
utils.plot_bars(controls, "Patchscope (source layer 4 → target layer 1) vs controls")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/patchscope_controls.png" width="650">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret this?</summary>

The Patchscope gets 79%, which is 38 of 48 countries and about the same as the model's own 77% on the real prompts (we'll put an interval on that in a moment), and all three controls get 0%:

- **Text only**: with nothing patched, the model sees `"The capital of x is"`, and it predicts the same token for every row (often one of the in-context capitals, like `" Tokyo"`). That's correct for at most one country.
- **Random vector**: a vector with the right norm but no content doesn't decode to the right capital. So it's not that "any vector in that slot" works.
- **Wrong position**: the BOS position has the same layer and comes from the same prompt, but it carries no information about the country (it can't, since it comes before the country token in a causal model!). This is the cleanest control of all, because it rules out anything about the source prompt or layer other than the information at the country position.

Together these show that the answer comes from information carried by the patched vector itself.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### How different are 79% and 77%?

Both numbers are rates measured on only 48 countries, so each could easily have been a few points higher or lower on a different list. A **Wilson interval** is the honest error bar for a
rate like that: the range of true rates that would plausibly produce the number we saw. Go deeper: 0.7 Foundations Toolkit, Part 5.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">FILL_WILSON</pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

TODO_WILSON_READING
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Try it yourself - ask the hidden state a different question

The power of Patchscopes is that the target prompt is a *question*, and you get to choose the question. Below, we've set up a second inspection prompt which asks about a country's
language instead of its capital. Run it, then try writing your own. The same cell also computes the **text-only baseline**: what the inspection prompt says with nothing patched in. Some ideas:

- Continents (`"France is in Europe. Japan is in Asia. Brazil is in South America. x is in"`). Which countries does GPT-2 Small get right?
- Use a source vector from a *contextual* prompt, e.g. `"Last summer I flew to Spain"`, and inspect the `" Spain"` position. Does context change which layers work?
- Use the token-identity prompt from the Patchscopes paper, `"cat -> cat; 1135 -> 1135; hello -> hello; x ->"`, and patch in the hidden state of the *last* token of a multi-token name like `" Leonardo DiCaprio"`. At which source layers does the model "say" the full name?

There's no right answer here; the point is to get a feel for how much information the model's own forward pass can pull out of a single hidden state.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">FILL_LANG</pre>

TODO_TRYIT_READING
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 3️⃣

- A Patchscope decodes a hidden state by placing it into an inspection prompt and letting the model's own forward pass answer a question about it.
- GPT-2 Small can recover a country's capital from the country's hidden state about as well as it answers the real question (38 and 37 of 48, with overlapping intervals), as long as the vector comes from an early-to-middle source layer and is patched into target layers 1-5.
- Text-only, random-vector and wrong-position controls all get 0%, so the answer genuinely comes from the patched vector.
- The target-layer cutoff shows where a donor vector can still be read *in this inspection prompt*. It suggests where recall happens, but it doesn't show that blocks 1-5 are necessary for the model's own recall: the Bonus tests that directly.
- The grid also contains an anomaly (target layer 2) that we can't yet explain. Reporting it is better than smoothing it over.

Compared to the lenses, Patchscopes are much more flexible (you can ask different questions) but also easier to fool yourself with: the inspection prompt is doing a lot of work,
and a cleverly chosen prompt can make the model "find" things by itself. That's why the text-only control is the first thing to check, every time.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 🎯 Held-out Quiz
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
You've read GPT-2 Small with three kinds of lens, trained a tuned lens, searched a bank of real contexts, and patched states into an inspection prompt. We ran five more experiments on GPT-2 Small
(and one on a model you never loaded) and haven't shown you any of them. Before you see the results, predict them.

This is how Stanford's CS 312 *[Deep Learning Alchemy](https://deep-learning-alchemy.github.io/)* tests understanding: if you understand a system, you can say what it will do in a setting you
haven't tried. A wrong answer here isn't a failure. It's the most useful thing that can happen in this lesson, because it shows you exactly which part of your picture is off.

**How it works.**

1. Each question shows its change as a diff against a call you made in this lesson.
2. Write your answers, and one sentence of reasoning for each, in the answer cell. The reasoning should name a result from this lesson. Don't run any new experiments first.
3. Run `quiz.score(my_answers)`, then open the **Reveal** dropdowns. Each has the real numbers and what a good mental model would have said.

**Difficulty** says how far a question takes you from what you ran:

- **Easy** means something you built, used on a case it was designed for.
- **Medium** means one knob turned beyond the range you ran.
- **Hard** means a regime you never saw, or a number rather than a direction.

**Scoring.** The right interval earns full points and a neighbouring interval earns half. There are 20 points in total. The score is for you: it isn't saved or sent anywhere. Expect to get around half.

> ```yaml
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 25 minutes on this quiz.
> ```

**The baseline.** Every question changes one call from this lesson. In your runs: GPT-2 Small gets @@BASE_ACC@@ of the 48 capitals right; the median rank of the correct capital first reaches 1 at @@BASE_RANK_LAYER@@
of the logit lens; the 300-step tuned lens agrees with the model's final prediction @@BASE_TUNED6@@ of the time at layer 6; the Patchscope from source layer 4 into target layer 1 names @@BASE_PATCH@@ of the capitals;
and a layer-3 contextual lens agrees @@BASE_CTX3@@ of the time, a layer-9 one @@BASE_CTX9@@.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Q1. A different template &nbsp;·&nbsp; Easy &nbsp;·&nbsp; 3 points

**From the lesson.** In Part 1 you used the logit lens and `answer_ranks` to see the correct capital's median rank reach 1 in one block.

**This question changes.** The prompt template. The three examples and the question all use "X's capital city is" instead of "The capital of X is". Everything else stays: the same 48 countries, and
the median over the prompts the model gets right.

```diff
- def make_prompt(country: str) -> str:
-     return f"{FEW_SHOT_PREFIX} The capital of {country} is"
+ def make_prompt(country: str) -> str:
+     return f"France's capital city is Paris. Japan's capital city is Tokyo. Italy's capital city is Rome. {country}'s capital city is"
```

**In which row of the logit lens does the median rank of the correct capital first reach 1?** (Rows are labelled as in Part 1: `embed`, `pre 1`, ..., `pre 11`, `final`.)

| a | b | c | d | e |
|---|---|---|---|---|
| @@Q1A@@ | @@Q1B@@ | @@Q1C@@ | @@Q1D@@ | @@Q1E@@ |
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Q2. A tuned lens with a tenth of the training &nbsp;·&nbsp; Medium &nbsp;·&nbsp; 4 points

**From the lesson.** You trained the tuned lens for 300 steps, and it agreed with the model's final prediction far more often than the logit lens at every layer.

**This question changes.** Only the number of training steps. Batch size 8 and learning rate 1e-3 stay. We trained three lenses, with seeds 0, 1 and 2, and averaged.

```diff
- losses = train_tuned_lens(tuned_lens, model, train_tokens, n_steps=300)
+ losses = train_tuned_lens(tuned_lens, model, train_tokens, n_steps=30, seed=seed)   # seeds 0, 1, 2
```

**What is the tuned lens's top-1 agreement with the model's final prediction at layer 6, on the held-out text?**

| a | b | c | d | e |
|---|---|---|---|---|
| @@Q2A@@ | @@Q2B@@ | @@Q2C@@ | @@Q2D@@ | @@Q2E@@ |
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Q3. One example in the inspection prompt &nbsp;·&nbsp; Medium &nbsp;·&nbsp; 4 points

**From the lesson.** In Part 3 your Patchscope used an inspection prompt with three worked examples, and patching the layer-4 state of each country into target layer 1 got most capitals right.

**This question changes.** The inspection prompt has one worked example instead of three. The source layer (4), the target layer (1) and the 48 countries stay.

```diff
- FEW_SHOT_PREFIX = "The capital of France is Paris. The capital of Japan is Tokyo. The capital of Italy is Rome."
+ ONE_SHOT_PREFIX = "The capital of France is Paris."
  target_tokens = model.to_tokens(f"{PREFIX} The capital of x is")
```

**What top-1 accuracy does the Patchscope get?**

| a | b | c | d | e |
|---|---|---|---|---|
| @@Q3A@@ | @@Q3B@@ | @@Q3C@@ | @@Q3D@@ | @@Q3E@@ |
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Q4. A bank from another layer &nbsp;·&nbsp; Medium &nbsp;·&nbsp; 4 points

**From the lesson.** In Part 2 you searched a bank of real contexts with states from the *same* layer, and the heatmap showed that a neighbour from almost any layer is the same token as the query.

**This question changes.** The bank is built from layer-3 states, and queried with layer-9 states. We score the neighbour's *next* token, as in the left panel of your figure: how often does it equal
the model's final top-1 prediction at the query? The 6,300 queries and the 37,000-vector bank are the ones you used.

```diff
- _, nearest = contextual_topk(queries[layer], bank[layer])       # same layer
+ _, nearest = contextual_topk(queries[9], bank[3])               # layer-9 queries, layer-3 bank
```

**What match rate do you get?**

| a | b | c | d | e |
|---|---|---|---|---|
| @@Q4A@@ | @@Q4B@@ | @@Q4C@@ | @@Q4D@@ | @@Q4E@@ |
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Q5. A model you never loaded &nbsp;·&nbsp; Hard &nbsp;·&nbsp; 5 points

**From the lesson.** You compared the logit lens with the tuned lens on GPT-2 Small, where the logit lens already works.

**This question changes.** The model. Pythia-160M has the same depth (12 blocks) and width (768) as GPT-2 Small, but it was trained differently, and its embedding and unembedding are not tied. We
read the same 200 held-out TinyStories sequences (tokenized with Pythia's own tokenizer) with the plain logit lens, no training at all.

```diff
- model = HookedTransformer.from_pretrained("gpt2-small", ...)
+ model = HookedTransformer.from_pretrained("pythia-160m", ...)
```

**What is the logit lens's top-1 agreement with Pythia's final prediction at layer 6, the middle of the network?**

| a | b | c | d | e |
|---|---|---|---|---|
| @@Q5A@@ | @@Q5B@@ | @@Q5C@@ | @@Q5D@@ | @@Q5E@@ |
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Your answers

Fill in every answer before you open any Reveal below.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Now the key, and your score. (The key is written out in the cell below. If you'd rather not see it before you've answered, answer first.)
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

quiz = Quiz({
    "Q1": Interval("c", points=3),
    "Q2": Interval("c", points=4),
    "Q3": Interval("c", points=4),
    "Q4": Interval("c", points=4),
    "Q5": Interval("c", points=5),
})
quiz.score(my_answers)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Reveal

<details>
<summary>Reveal - Q1: a different template</summary>

@@REVEAL1@@
</details>

<details>
<summary>Reveal - Q2: a tuned lens with a tenth of the training</summary>

@@REVEAL2@@
</details>

<details>
<summary>Reveal - Q3: one example in the inspection prompt</summary>

@@REVEAL3@@
</details>

<details>
<summary>Reveal - Q4: a bank from another layer</summary>

@@REVEAL4@@
</details>

<details>
<summary>Reveal - Q5: a model you never loaded</summary>

@@REVEAL5@@
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Try it yourself - reproduce one result

Pick the question you got most wrong and reproduce it. Q3 is the cheapest: it takes a few seconds, using the functions and the source vectors you already have.

```python
one_shot_target = model.to_tokens("The capital of France is Paris. The capital of x is")
one_shot_pos = one_shot_target.shape[1] - 2
logits_one_shot = run_patchscope(model, source_vectors[4], one_shot_target, one_shot_pos, target_layer=1)
print(f"one-shot Patchscope accuracy: {(logits_one_shot.argmax(-1) == answer_tokens).float().mean():.1%}")
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## The runs behind this quiz

Every number in the reveals above comes from the next cell, which reruns all the key runs and draws the figure below. It takes about 2 minutes, most of it downloading and running Pythia. You don't need
to run it. It's here so that you can check any answer, and so that the quiz can never drift from the code: the cell ends by checking that the key above matches what the runs produced.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/quiz_runs.png" width="1100">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I read this figure?</summary>

TODO_QUIZ_FIGURE
</details>

## What this quiz tested

TODO_QUIZ_TESTED
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# ☆ Bonus
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Watching the model change its mind

Recall that GPT-2 Small answers `" Shanghai"` for China, `" Toronto"` for Canada and `" Sydney"` for Australia. A nice exercise is to plot, for each of these prompts, the logit
difference between the capital and the "famous city" (e.g. `Beijing - Shanghai`) at every layer, using both lenses. Questions to investigate:

- Does the correct capital ever lead, and then get overtaken? Or does the famous city win from the start?
- For the prompts the model gets *right*, is there a competing famous city (e.g. `Berlin - Munich`, `Madrid - Barcelona`) that leads at some intermediate layer?
- Can you find a Patchscope target layer at which China's hidden state decodes to `" Beijing"` rather than `" Shanghai"`?

## Anomaly hunt: what's wrong with target layer 2?

In the Patchscope grid, early source vectors fail when patched into target layer 2, but work at target layers 1 and 3. Some things you could try:

- Patch into target layer 2 but *also* patch the same vector into target layer 1 (so block 1 has already processed it). Does accuracy recover?
- Look at what the model predicts instead when source layer 2 is patched into target layer 2. Is it a single default answer (e.g. one of the in-context capitals), or noise?
- Compare the norm of the patched vector with the norm of the target's own residual stream at that position and layer. Is the patched vector unusually small or large there?
- Use the logit lens at the final position of the *patched* run, to see at which layer the answer goes wrong.

## Are blocks 1-5 necessary?

The Patchscope grid says a country's state can still be read when it is patched into target layers up to 5. That tells us where recall *can be done in the inspection prompt*. It doesn't say the
model *needs* those blocks to recall a capital in its own prompt. To ask that, break the blocks inside the native prompt and see whether recall survives.

We'll **mean-ablate** the MLP outputs at the country's position. For a chosen block, replace the MLP output at the country token by its average over all 48 prompts. That removes whatever is
specific to each country but leaves a typical activation behind, so the rest of the network isn't shown something wildly off-distribution. (It's a common first guess that factual recall lives in
MLPs. We ablate MLPs only, and only at one position, so this is one test among several you could run.)

'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - mean-ablate some blocks and measure recall

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Implement `ablate_blocks_recall`. It takes the native prompts `tokens` (shape `(batch, seq)`), the answer tokens, a list of `blocks` and a position `pos` (the country sits at position `-2`, just before
`" is"`). First run the model once on the clean prompts and cache each chosen block's `hook_mlp_out`; take its mean over the batch at position `pos`. Then run the model again with hooks that replace
`mlp_out[:, pos]` of those blocks by their means, and return the fraction of prompts whose final top-1 token is the answer.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

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
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
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
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_ablate_blocks_recall(ablate_blocks_recall, model)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`model.run_with_cache(tokens, names_filter=lambda name: name in names)` caches only the hooks you ask for. A hook that edits `mlp_out[:, pos]` in place and returns `mlp_out` is all
`model.run_with_hooks` needs. Make one hook per block, each closing over its own mean.
</details>

<details>
<summary>Help - my accuracy with no blocks ablated isn't the model's accuracy.</summary>

With `blocks=[]` there are no hooks, so you should get exactly the model's own top-1 accuracy on the prompts. If you don't, check that you compare the final position's logits
(`logits[:, -1]`) with the answer tokens.
</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

necessity = {
    "no ablation": ablate_blocks_recall(model, tokens, answer_tokens, blocks=[]),
    "MLPs of blocks 1-5": ablate_blocks_recall(model, tokens, answer_tokens, blocks=list(range(1, 6))),
    "MLPs of blocks 6-10": ablate_blocks_recall(model, tokens, answer_tokens, blocks=list(range(6, 11))),
}
for name, accuracy in necessity.items():
    print(f"{name:<22} capital accuracy {accuracy:.1%}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">FILL_NECESSITY</pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

TODO_NECESSITY_READING
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Claim card: LatentLens

**The claim.** Describe a vision-language model's visual tokens by their nearest *contextual* text vectors, which is the contextual lens you built in Part 2 applied to image tokens, and an
LLM judge finds the description interpretable for 68% of visual tokens. The same judge finds 32% interpretable for EmbeddingLens (nearest static token embedding) and 24% for LogitLens. These
are the authors' numbers across 15 models, from [LatentLens](https://arxiv.org/abs/2602.00462); we didn't re-run them.

**What our version shows.** On GPT-2's text states, a state's nearest neighbour is very often the same token in another story, and its next token carries some information about what the model
will say (Part 2's figure). We used no judge and no images.

**What neither shows.** Retrieval is description, not causal use. A nearby context tells you what a state *looks like*. It doesn't show that the model uses the state the way that context would
suggest. Testing that takes an intervention, like the ablation above.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## What this lesson doesn't show

- **One model, one task.** Everything about capitals is GPT-2 Small on 48 single-token countries. The Wilson intervals say how much a rate measured on 48 cases can move.
- **A tuned lens trained on children's stories.** Its gains on TinyStories didn't carry over to the capitals prompts, and we can't tell from lenses alone whether that is distribution shift or an absent answer.
- **A bank of 600 stories.** The contextual lens only describes a state by what is in the bank. A bigger or different bank would give different neighbours.
- **Readable isn't necessary, and similar isn't used.** The Patchscope grid shows where a donor vector can still be read; the ablation tests only the MLPs, at one position. Neither says how the model computes the answer, and a nearest neighbour is a description, not a cause.
- **A few controls, not all of them.** We checked random vectors, shuffled labels, text-only prompts and the BOS token. We did not check, for example, other inspection prompts for the same question.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Further reading

- nostalgebraist (2020), [*interpreting GPT: the logit lens*](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). The original blog post.
- Belrose et al. (2023), [*Eliciting Latent Predictions from Transformers with the Tuned Lens*](https://arxiv.org/abs/2303.08112). Includes a causal-basis analysis showing tuned-lens features are used by the model, and a prompt-injection detection application.
- Ghandeharioun et al. (2024), [*Patchscopes*](https://arxiv.org/abs/2401.06102). Includes multi-hop reasoning correction and cross-model Patchscopes (decoding a small model's hidden states with a bigger model).
- Pal et al. (2023), [*Future Lens*](https://arxiv.org/abs/2311.04897). Decoding tokens several positions ahead from a single hidden state.
- Chen et al. (2024), [*SelfIE*](https://arxiv.org/abs/2403.10949). Using the model to produce free-text descriptions of its own hidden states, a natural extension of the Patchscope idea that leads into the next few sections of this chapter.
'''

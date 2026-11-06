# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
[
    {"title": "The Logit Lens", "icon": "1-circle-fill", "subtitle": "(30%)"},
    {"title": "The Tuned Lens", "icon": "2-circle-fill", "subtitle": "(30%)"},
    {"title": "Patchscopes", "icon": "3-circle-fill", "subtitle": "(30%)"},
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

Questions like these are what this section is about. We'll study three methods which all try to **read the residual stream in the model's own language**:

- The **logit lens** (nostalgebraist, 2020) takes an intermediate residual stream vector and decodes it as if it were the final layer, by applying the model's own final layernorm and unembedding. It costs nothing to compute, needs no training, and works surprisingly well on GPT-2.
- The **tuned lens** (Belrose et al., 2023) points out that intermediate layers don't necessarily use the same "basis" as the final layer, and fixes this by learning a small affine *translator* for each layer, trained to make the decoded distribution match the model's final output.
- **Patchscopes** (Ghandeharioun et al., 2024) take a different approach: instead of decoding a vector with a fixed readout, they *patch* it into a different prompt (an "inspection prompt" like `"The capital of x is"`) and let the model itself tell us what information the vector carries.

By the end of this section you'll have built all three from scratch, and used them to produce three pictures: the layer at which GPT-2 converges on each capital city, how much
better a trained lens is than the free one (and how much of that is just bigram statistics), and a map of *where* in the model entity information can be read out and *where* it can
be re-injected.

A recurring theme will be **controls**. Decoding methods are very good at producing plausible-looking tokens, so every result in this notebook is paired with a baseline that should
fail: random vectors with matched norms, the wrong layer's translator, inspection prompts with nothing patched in, and source vectors taken from the wrong position. When a control
*doesn't* fail, that's often the most interesting result of all.

This section assumes you're comfortable with TransformerLens (hooks, `run_with_cache`, the residual stream) at the level of [1.2] Intro to Mech Interp. The tuned lens part uses a
short PyTorch training loop; nothing more than [0.3] Optimization.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Content & Learning Objectives

### 1️⃣ The Logit Lens

You'll implement the logit lens by applying GPT-2's final layernorm and unembedding to every layer of the residual stream, and use it to watch the answer to a factual question
emerge. You'll compare the rank of the correct answer against the ranks of *other* plausible answers and against random vectors, to separate "the model knows the answer" from "the
model knows what kind of answer goes here".

> ##### Learning Objectives
>
> - Understand why the residual stream can be decoded at any layer, and what assumption the logit lens makes when it does so
> - Implement the logit lens and use it to produce top-token tables and answer-rank curves
> - Use mismatched-answer and random-vector controls to interpret a rank curve correctly
> - Measure how far each layer's decoded distribution is from the final one (KL divergence)

### 2️⃣ The Tuned Lens

The logit lens assumes every layer speaks the same language as the last one. You'll test that assumption by training a tuned lens: one affine translator per layer, optimised to
match the model's output distribution on held-out text. You'll then ask what the improvement actually consists of, by comparing against a bigram baseline and swapping translators
between layers.

> ##### Learning Objectives
>
> - Understand representational drift, and why a learned per-layer translator can fix it
> - Implement and train a tuned lens with a KL-divergence objective
> - Evaluate lenses on held-out data with KL and top-1 agreement, rather than on hand-picked prompts
> - Use a bigram baseline and a wrong-layer control to avoid over-reading a trained probe

### 3️⃣ Patchscopes

Rather than decoding a vector with a fixed readout, you'll patch it into an inspection prompt and let the model decode it for you. You'll build a Patchscope that extracts a
country's capital from the country's hidden state, map how well it works for every (source layer, target layer) pair, and check it against text-only, random-vector and
wrong-position controls.

> ##### Learning Objectives
>
> - Understand the Patchscope framework (source prompt, source layer, target prompt, target layer) and how the logit lens is a special case of it
> - Implement activation patching across prompts with TransformerLens hooks
> - Produce a source-layer × target-layer accuracy map and explain its shape
> - Separate information *carried by the patched vector* from information supplied by the inspection prompt itself

### ☆ Bonus

Suggestions for going further: watching the model "change its mind" between a country's largest city and its capital, the attention lens, lenses on models where the logit lens
fails, and the papers that extend these methods.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Setup code

The saved outputs and figures below were produced on a CUDA GPU.
'''

# ! CELL TYPE: code
# ! FILTERS: [~]
# ! TAGS: []

from IPython import get_ipython

ipython = get_ipython()
ipython.run_line_magic("load_ext", "autoreload")
ipython.run_line_magic("autoreload", "2")

# ! CELL TYPE: code
# ! FILTERS: [colab]
# ! TAGS: [master-comment]

# import os
# import sys
#
# IN_COLAB = "google.colab" in sys.modules
#
# if IN_COLAB:
#     # Install dependencies, and download this chapter's exercise files from the ARENA extension repo
#     %pip install transformer_lens==2.16.1 jaxtyping
#     repo, branch, chapter = "ARENA_3.0_Extended", "main", "chapter7_activation_to_language"
#     if not os.path.exists(f"/content/{chapter}"):
#         !wget -P /content https://github.com/shinobiultra/{repo}/archive/refs/heads/{branch}.zip
#         !unzip -q /content/{branch}.zip '{repo}-{branch}/{chapter}/exercises/*' -d /content
#         !mv /content/{repo}-{branch}/{chapter} /content/{chapter}
#         !rm -r /content/{branch}.zip /content/{repo}-{branch}
#     os.chdir(f"/content/{chapter}/exercises")

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
Let's load GPT-2 Small. We won't train it at any point in this section, so we turn off gradients for its parameters.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

model = HookedTransformer.from_pretrained("gpt2-small", device=device)
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

- **Embedding row: junk.** The decoded tokens (`' destro'`, `' arrang'`, ...) are word fragments with nothing to do with the prompt. The embedding row is the token embedding of `" is"` plus the positional embedding for position 26, and after layernorm it happens to point towards some rare tokens. It isn't a prediction of anything, which is the logit lens's core assumption failing in the most obvious way.
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
### Exercise - KL divergence to the final distribution

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Ranks only look at one token. A more complete measure of "how close is layer $\ell$ to the final answer?" is the KL divergence between the final distribution $p_L$ and the decoded
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
    bigram_agree=results["tuned_agree"][0],
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

The **layer-0 tuned lens** reads the residual stream before *any* block has run. That vector is just the embedding of the current token plus its position. So whatever the layer-0
tuned lens predicts is a function of the current token (and position) alone: it's a learned **bigram model**. It reaches roughly 37% agreement with GPT-2's final predictions,
purely from bigram statistics.

That reframes the result. The tuned lens at layer 1 isn't "46% from nothing", it's "46% versus a 37% bigram floor". The gap between the dashed line and the tuned lens curve is the
part that requires *context*, and that grows steadily through the network. This is the honest version of the claim: the tuned lens shows that next-token information accumulates
gradually over the layers, while the logit lens makes it look like it all appears at the end.
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

- **Most of the tuned lens's gains are on common tokens.** Averaged over a corpus, top-1 agreement is dominated by frequent, predictable tokens (punctuation, `" the"`, `" was"`, the second half of a common word). The bigram floor tells us how much of that is available from the current token alone. A rare, fact-dependent answer like `" Canberra"` is a very different kind of prediction.
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

Ghandeharioun et al. ([*Patchscopes: A Unifying Framework for Inspecting Hidden Representations of Language Models*](https://arxiv.org/abs/2401.06102), 2024) observed that we already have a very powerful decoder for hidden states: the model itself. If we want to know what a hidden state $h$ "means", we can put it into a forward pass on a carefully chosen **inspection prompt**, and read off what the model says.

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
    tests.test_patchscope_accuracy_grid(patchscope_accuracy_grid, model)
# END HIDE

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

**The target layer matters more than the source layer.** Patching into target layers 1, 3, 4 and 5 works well for early source layers, reaching 75-79%, which is as good as the
model does on the real prompts (77%). Patching into target layer 6 or later almost never works, whatever the source.

Why the cutoff? Patching at target layer $\ell^*$ means only blocks $\ell^*, \dots, L-1$ get to process the patched vector. Retrieving "the capital of country X" is a
factual-recall computation, and attention then has to move the result to the final position. If we inject the country after the blocks that do this have already run, nothing
downstream looks up the capital. The grid is effectively a map of *where the recall computation lives*: somewhere in blocks 1-5. (Compare this with the logit lens, which only saw
the answer at layer 9. The logit lens tells us when the answer arrives *at the final position in the output basis*; the Patchscope tells us which blocks are needed to *produce*
it.)

**Target layer 0 is special.** Only the layer-0 source works at target layer 0. The residual stream before block 0 is just token plus position embeddings, which live in a different
region of space from processed residual vectors, so a processed vector patched in there is off-distribution for block 0.

**Later source layers work less well.** Source vectors from layers 1-5 decode best. From layer 6 onwards accuracy drops, and by layer 9-11 only target layer 2 still works
reasonably. The country's identity is still *there* in late source vectors (some target layer can read it), but it's gradually being reformatted into something the early blocks of
the target prompt don't expect.

**Target layer 2 is strange.** Look at the column for target layer 2: early source vectors (1-4) almost completely fail there (4-12%), even though they work at target layers 1 and
3 on either side. Yet late source vectors (8-11) work *better* at target layer 2 than anywhere else. We don't have an explanation for this. It's exactly the kind of anomaly that's
worth chasing (see the bonus section), and a good reminder that a Patchscope is an intervention on a real network, not a clean measuring device.
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

The Patchscope gets 79% (slightly better than the model's own 77% on the real prompts!), and all three controls get 0%:

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
## Try it yourself - ask the hidden state a different question

The power of Patchscopes is that the target prompt is a *question*, and you get to choose the question. Below, we've set up a second inspection prompt which asks about a country's
language instead of its capital. Run it, then try writing your own. Some ideas:

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

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Spain    -> ' Spanish' (0.49), ' English' (0.14), ' Portuguese' (0.11)
Russia   -> ' Russian' (0.43), ' English' (0.15), ' Ukrainian' (0.02)
China    -> ' Chinese' (0.29), ' Mandarin' (0.25), ' Korean' (0.10)
Italy    -> ' Italian' (0.46), ' Spanish' (0.11), ' English' (0.10)
Greece   -> ' Greek' (0.38), ' English' (0.08), ' Spanish' (0.06)
Poland   -> ' Polish' (0.41), ' English' (0.08), ' Russian' (0.06)
Sweden   -> ' Swedish' (0.37), ' English' (0.20), ' Norwegian' (0.04)
Turkey   -> ' Turkish' (0.37), ' English' (0.10), ' Greek' (0.06)
Iran     -> ' Persian' (0.14), ' English' (0.10), ' Turkish' (0.08)
Brazil   -> ' Portuguese' (0.46), ' English' (0.13), ' Spanish' (0.10)</pre>

All ten are right, including Brazil → Portuguese (not Spanish) and Iran → Persian. The same source vectors that answered "what's the capital?" also answer "what language do they
speak?". Before you get too excited: what's the text-only baseline here? Run the language prompt with nothing patched and see how often `' English'` would have been "correct" by
default.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 3️⃣

- A Patchscope decodes a hidden state by placing it into an inspection prompt and letting the model's own forward pass answer a question about it.
- GPT-2 Small can recover a country's capital from the country's hidden state as well as it answers the real question, as long as the vector comes from an early-to-middle source layer and is patched into target layers 1-5. The target-layer cutoff tells us which blocks the recall computation needs.
- Text-only, random-vector and wrong-position controls all get 0%, so the answer genuinely comes from the patched vector.
- The grid also contains an anomaly (target layer 2) that we can't yet explain. Reporting it is better than smoothing it over.

Compared to the lenses, Patchscopes are much more flexible (you can ask different questions) but also easier to fool yourself with: the inspection prompt is doing a lot of work,
and a cleverly chosen prompt can make the model "find" things by itself. That's why the text-only control is the first thing to check, every time.
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

## The attention lens

The logit lens decodes the whole residual stream; you can just as well decode the output of a single attention head (`cache["result", layer][:, :, head]`, after setting
`model.set_use_attn_result(True)`) through `ln_final` and $W_U$. This is sometimes called the *attention lens*, and it's closely related to the direct logit attribution you built
in [1.2]. Which heads write `" Berlin"` directly? Do they attend to the `" Germany"` token? (Compare with the "mover heads" in [1.4.1] IOI.)

## When the logit lens fails

GPT-2 is a well-behaved case for the logit lens. Load `pythia-160m` or `pythia-410m` and repeat the layer-by-layer evaluation from section 2️⃣. You should find that the logit lens
is much worse (often decoding to nonsense until the last couple of layers), while the tuned lens still produces a smooth curve. Belrose et al. discuss why: models trained without
tied embeddings, or with different normalisation schemes, drift much more.

## Further reading

- nostalgebraist (2020), [*interpreting GPT: the logit lens*](https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens). The original blog post.
- Belrose et al. (2023), [*Eliciting Latent Predictions from Transformers with the Tuned Lens*](https://arxiv.org/abs/2303.08112). Includes a causal-basis analysis showing tuned-lens features are used by the model, and a prompt-injection detection application.
- Ghandeharioun et al. (2024), [*Patchscopes*](https://arxiv.org/abs/2401.06102). Includes multi-hop reasoning correction and cross-model Patchscopes (decoding a small model's hidden states with a bigger model).
- Pal et al. (2023), [*Future Lens*](https://arxiv.org/abs/2311.04897). Decoding tokens several positions ahead from a single hidden state.
- Chen et al. (2024), [*SelfIE*](https://arxiv.org/abs/2403.10949). Using the model to produce free-text descriptions of its own hidden states, a natural extension of the Patchscope idea that leads into the next few sections of this chapter.
'''

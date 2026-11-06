# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
[
    {"title": "Activations, reconstructions and behaviour", "icon": "1-circle-fill", "subtitle": "(25%)"},
    {"title": "A warm-started NLA", "icon": "2-circle-fill", "subtitle": "(25%)"},
    {"title": "Training through a discrete bottleneck", "icon": "3-circle-fill", "subtitle": "(25%)"},
    {"title": "What can text carry?", "icon": "4-circle-fill", "subtitle": "(20%)"},
    {"title": "Bonus", "icon": "star", "subtitle": "(5%)"},
]
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# [7.4] Mini Natural Language Autoencoders
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
Here is a sentence, and the verb GPT-2 Small picks to continue it, if we make it choose between `" is"`, `" are"`, `" was"` and `" were"`:

```
Yesterday, the keys behind the door really   →   " were"   (probability 0.90 among the four)
```

To get this right the model has to work out two things, neither of which is written next to the verb. The subject is *keys*, which is plural, even though the closest noun (*door*)
is singular. And the sentence is about the past. By the time the residual stream at the final position reaches layer 10, both facts are stored in that one 768-dimensional vector.
If you cut that vector out of this sentence and paste it into a completely different one, like `"Today, the dog near the trees really"` (where GPT-2 would otherwise pick `" is"`),
GPT-2 switches to `" were"`.

So the vector "knows" something like *plural subject, past tense*. Here is the question this section is about: **can we get a model to tell us, in words, what an activation
contains, and can we check that the words are right?**

The idea we'll study is the **natural language autoencoder** (NLA), introduced by Anthropic in 2026 ([Fraser-Taliente, Kantamneni, Ong et al.](https://transformer-circuits.pub/2026/nla/)). An NLA is an autoencoder whose bottleneck is text:

- an **activation verbaliser** (AV) reads an activation $h$ and writes a description $z$, a short piece of text;
- an **activation reconstructor** (AR) reads the description $z$ and tries to reproduce $h$.

The two are trained so that $\text{AR}(\text{AV}(h)) \approx h$. Nobody tells the verbaliser what to say. The only pressure is that the reconstructor must be able to rebuild the
activation from the text, so the text has to carry the information that is in the activation. And because the bottleneck is text, a human can read it.

That's the promise. The catch is in the details. Text is discrete, so you can't backpropagate through it. A verbaliser that's free to write anything might invent a private code
that the reconstructor understands and we don't. And "the reconstruction is close to the activation" is not the same as "the reconstruction does what the activation does": a
description could capture most of the variance in an activation while dropping the one detail the model actually uses.

In this section you'll build a miniature NLA that trains in a couple of minutes, on GPT-2 Small activations from a dataset where we *know* the ground truth, and use it to see each
of these issues concretely. By the end you'll have:

- mapped what GPT-2's final-position activation contains, layer by layer, and built a behavioural test that actually tests something;
- trained a verbaliser that turns an activation into four words like `tables currently with keys`, and a reconstructor (built from GPT-2 itself) that turns the words back into an activation which GPT-2 uses almost as well as the original;
- trained the pair end-to-end through the discrete bottleneck with a policy gradient, and seen why it only works from a warm start;
- compared the text bottleneck against honest baselines (a bag of words of the input sentence, PCA with the same number of dimensions, random text), and produced the section's signature picture: **reconstruction quality versus behaviour**, where the bag of words of the input reconstructs 70% of the variance but gets the verb wrong most of the time the subject and object disagree, while the best reconstruction from a two-word description like `plural past` explains only 17% of the variance and gets the verb right 99% of the time.

This section assumes you're comfortable with TransformerLens hooks and activation patching ([1.2] Intro to Mech Interp, and the Patchscopes part of [7.1]). Part 3️⃣ uses a policy
gradient with a KL penalty to a reference policy; you've seen both in [2.3] PPO and [2.4] RLHF, but we re-derive everything we need.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Content & Learning Objectives

### 1️⃣ Activations, reconstructions and behaviour

Before building an autoencoder, we need to know what we're encoding and how to score a reconstruction. You'll collect GPT-2 activations on a templated dataset with known latent
factors, implement the fraction of variance explained (FVE) metric used in the NLA paper, and use "description oracles" to map which facts the activation contains at each layer.
You'll then build a behavioural test, discover that the obvious one is useless, and replace it with one that isn't.

> ##### Learning Objectives
>
> - Implement the NLA paper's FVE metric, and understand what FVE = 0 and FVE = 1 mean
> - Use class-mean "description oracles" to measure how much of an activation a given description could possibly explain
> - See how the content of the final-position residual stream changes across layers
> - Design a behavioural test whose controls fail (transplant patching), after seeing one whose controls pass

### 2️⃣ A warm-started NLA

You'll build both halves of a miniature NLA: a small transformer verbaliser that reads an activation and writes up to four words, and a reconstructor made of frozen GPT-2 plus an
affine head. You'll warm-start the verbaliser on a teacher that writes random *true* facts about each sentence, like the paper's supervised warm start, and evaluate the whole loop
on held-out sentences.

> ##### Learning Objectives
>
> - Understand the NLA architecture: activation verbaliser (AV) and activation reconstructor (AR)
> - Implement an autoregressive verbaliser conditioned on an activation, with log-probabilities, masking and sampling
> - Implement a reconstructor that reads text with the target model itself and maps it back to an activation
> - Measure FVE, behaviour and the truthfulness of the generated descriptions

### 3️⃣ Training through a discrete bottleneck

The warm-started verbaliser says true things, but not necessarily the *useful* true things. You'll train it end-to-end with a group-baseline policy gradient (the GRPO-style recipe
from the paper), with the reconstructor trained alongside by regression and a KL penalty to the warm start. You'll compare against controls that only train the reconstructor, or
start from scratch, and look at a differentiable alternative (Gumbel-softmax).

> ##### Learning Objectives
>
> - Understand why you can't backpropagate through sampled text, and how the score-function (REINFORCE) estimator gets around it
> - Implement group-normalised advantages, a KL penalty to a reference policy, and the policy-gradient loss
> - Train an NLA end-to-end and see what it learns to say
> - Understand why end-to-end training collapses from a cold start, and what the warm start is for

### 4️⃣ What can text carry?

You'll put the NLA next to honest baselines: PCA bottlenecks with the same number of dimensions as the NLA has words, a decoder that sees a bag of words of the *input sentence*,
shuffled and random text. You'll measure both FVE and behaviour for all of them, and find out which parts of the activation survive the trip through words and which don't. You'll
finish by editing the text in the bottleneck by hand and watching GPT-2's prediction change.

> ##### Learning Objectives
>
> - Compare a text bottleneck against a continuous bottleneck of the same size
> - Build a bag-of-words bypass and understand why high FVE can coexist with broken behaviour
> - Test whether the reconstructor depends on word order
> - Understand what changes (and what doesn't) between this miniature and a full-scale NLA

### ☆ Bonus

Training at a different layer, rewarding behaviour instead of variance, provoking the verbaliser into lying, and further reading.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Setup code
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

import copy
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from IPython.display import display
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
pd.set_option("display.width", 200)

# Make sure exercises are in the path
chapter = "chapter7_activation_to_language"
section = "part4_mini_natural_language_autoencoders"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part4_mini_natural_language_autoencoders.tests as tests
import part4_mini_natural_language_autoencoders.utils as utils

MAIN = __name__ == "__main__"

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 1️⃣ Activations, reconstructions and behaviour
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## The task: subject-verb agreement across an attractor

We need a setting where we know the right answer: what *should* a description of the activation say? So instead of natural text, we'll use a templated dataset with a handful of
known **latent factors**. Every sentence looks like this:

```
{Time}, the {subject} {preposition} the {object} really
```

for example `"Earlier, the house behind the windows really"`. The factors are:

- the **time adverb**: one of `Today`, `Now`, `Currently` (present tense) or `Yesterday`, `Earlier`, `Previously` (past tense);
- the **subject**: one of 16 nouns, singular or plural (`key`/`keys`, `dog`/`dogs`, ...);
- the **preposition**: `near`, `behind`, `beside` or `with`;
- the **object**: a different one of the 16 nouns, singular or plural.

The next word is a verb, and GPT-2 has to pick between `" is"`, `" are"`, `" was"` and `" were"`. The right choice depends on exactly two of the factors: the **number of the subject** and the **tense**. The object noun is an *attractor*: in `"the house behind the windows"`, the noun closest to the verb is plural, but the verb must agree with the singular subject. Getting this right requires the model to use word order, not just which words are present. (Subject-verb agreement across attractors is a classic test of syntax in language models; see [Linzen et al., 2016](https://arxiv.org/abs/1611.01368), and [Finlayson et al., 2021](https://arxiv.org/abs/2106.06087) for a causal analysis in GPT-2.)

Why end every sentence with `" really"`? Because we're going to study the activation at the **final position**, and we want everything in that activation to have been put there by
the model from context. If the sentence ended with the object noun, the final-position activation would be dominated by the identity of that word. (As we'll see, the object still
dominates the early layers even with `really` at the end.) With a fixed final token, the token itself contributes nothing that varies across the dataset, so any variation is
information the model *moved* there.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Let's load GPT-2 Small. We'll never train it in this section.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

model = HookedTransformer.from_pretrained("gpt2-small", device=device)
model.requires_grad_(False)

print(f"{model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Here's the dataset. We build every combination of the factors (6 time adverbs × 16 × 15 subject/object pairs × 2 × 2 numbers × 4 prepositions = 23,040 sentences), shuffle them, and
keep 5,000: 4,000 for training and 1,000 held out. Every word is a single GPT-2 token, so every sentence has the same length and the final position is always the same index.

The `verb` column is the index of the correct verb in `VERBS`: `2 * past + plural`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

NOUNS = [
    ("key", "keys"), ("book", "books"), ("dog", "dogs"), ("car", "cars"), ("tree", "trees"), ("house", "houses"),
    ("door", "doors"), ("girl", "girls"), ("boy", "boys"), ("cat", "cats"), ("bird", "birds"), ("table", "tables"),
    ("chair", "chairs"), ("teacher", "teachers"), ("student", "students"), ("window", "windows"),
]  # fmt: skip
TIME_ADVERBS = ["Today", "Now", "Currently", "Yesterday", "Earlier", "Previously"]  # first three present, last three past
PREPOSITIONS = ["near", "behind", "beside", "with"]
VERBS = [" is", " are", " was", " were"]
LAYER = 10  # the layer whose activations we'll verbalise (the residual stream *before* block 10)


def make_dataset(n: int, seed: int = 0) -> pd.DataFrame:
    rows = []
    for time_idx, time_word in enumerate(TIME_ADVERBS):
        for i, j in [(i, j) for i in range(len(NOUNS)) for j in range(len(NOUNS)) if i != j]:
            for subject_plural in [0, 1]:
                for object_plural in [0, 1]:
                    for prep in PREPOSITIONS:
                        past = int(time_idx >= 3)
                        subject, obj = NOUNS[i][subject_plural], NOUNS[j][object_plural]
                        rows.append(
                            dict(
                                sentence=f"{time_word}, the {subject} {prep} the {obj} really",
                                time_word=time_word.lower(),
                                tense=["present", "past"][past],
                                subject=subject,
                                number=["singular", "plural"][subject_plural],
                                preposition=prep,
                                object=obj,
                                object_number=["singular", "plural"][object_plural],
                                verb=2 * past + subject_plural,
                            )
                        )
    random.Random(seed).shuffle(rows)
    return pd.DataFrame(rows[:n])


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

df = make_dataset(5000)
N_TRAIN = 4000
train_idx, test_idx = t.arange(N_TRAIN), t.arange(N_TRAIN, len(df))

tokens = model.to_tokens(df["sentence"].tolist())
verb_ids = t.tensor([model.to_single_token(v) for v in VERBS])
labels = t.tensor(df["verb"].values)

print(f"tokens: {tuple(tokens.shape)}")
print([model.to_string(tok) for tok in tokens[0]])
display(df.head(6))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:DATASET -->tokens: (5000, 9)
['&lt;|endoftext|&gt;', 'Now', ',', ' the', ' girl', ' near', ' the', ' key', ' really']
                                         sentence  time_word    tense   subject    number preposition    object object_number  verb
0               Now, the girl near the key really        now  present      girl  singular        near       key      singular     0
1     Yesterday, the book behind the chair really  yesterday     past      book  singular      behind     chair      singular     2
2        Earlier, the key beside the trees really    earlier     past       key  singular      beside     trees        plural     2
3      Yesterday, the doors with the girls really  yesterday     past     doors    plural        with     girls        plural     3
4  Earlier, the houses behind the teachers really    earlier     past    houses    plural      behind  teachers        plural     3
5  Today, the teachers behind the students really      today  present  teachers    plural      behind  students        plural     1</pre>

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Before looking inside the model, let's check it can actually do the task. We compare the logits of the four verbs at the final position, and count how often the largest one is the
grammatically correct verb. (We don't care whether a verb is GPT-2's overall top prediction; after `"really"` it often prefers some other word. We only care which of the four verbs
it prefers.)
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

with t.inference_mode():
    clean_verb_logits = t.cat(
        [model(tokens[i : i + 250].to(device))[:, -1, verb_ids.to(device)].cpu() for i in range(0, len(tokens), 250)]
    )
clean_preds = clean_verb_logits.argmax(-1)
print(f"GPT-2 picks the right verb on {(clean_preds == labels).float().mean():.1%} of sentences")
for name, mask in [
    ("subject and object agree in number", df["number"].values == df["object_number"].values),
    ("subject and object disagree in number", df["number"].values != df["object_number"].values),
]:
    mask = t.tensor(mask)
    print(f"  {name:<40}: {(clean_preds[mask] == labels[mask]).float().mean():.1%}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:ACCURACY -->GPT-2 picks the right verb on 100.0% of sentences
  subject and object agree in number      : 100.0%
  subject and object disagree in number   : 100.0%</pre>

</details>

GPT-2 Small gets every sentence right, including the ones where the attractor disagrees with the subject. So whatever it is doing, it reliably works out the subject's number and
the tense. Our question is where that information lives, and whether we can get it out as words.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - collect final-position activations

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 5-10 minutes on this exercise.
> ```

Implement `get_final_activations`, which returns the residual stream *before* each block in `layers` (i.e. `resid_pre`), at the **final** sequence position, for every sentence. The
output should have shape `(len(layers), batch, d_model)` and live on the CPU (5,000 × 768 floats per layer is small, and we'll do a lot of CPU-side analysis on it).

Process the sentences in batches of `batch_size`, and don't run the model further than you need to: `run_with_cache` takes a `stop_at_layer` argument (the forward pass stops
*before* block `stop_at_layer`, so to cache `resid_pre` of block `layer` you need `stop_at_layer=layer + 1`), and a `names_filter` so that only the activations you ask for are
stored.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.inference_mode()
def get_final_activations(
    model: HookedTransformer, tokens: Int[Tensor, "batch seq"], layers: list[int], batch_size: int = 250
) -> Float[Tensor, "layer batch d_model"]:
    """Returns resid_pre at the final position for each layer in `layers`, as a CPU tensor."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    names = [f"blocks.{layer}.hook_resid_pre" for layer in layers]
    chunks = []
    for start in range(0, len(tokens), batch_size):
        _, cache = model.run_with_cache(
            tokens[start : start + batch_size].to(model.cfg.device),
            names_filter=lambda name: name in names,
            stop_at_layer=max(layers) + 1,
        )
        chunks.append(t.stack([cache["resid_pre", layer][:, -1] for layer in layers]).cpu())
    return t.cat(chunks, dim=1)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_get_final_activations(get_final_activations, model)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

For each batch, call `model.run_with_cache(batch_tokens, names_filter=..., stop_at_layer=max(layers) + 1)`, then stack `cache["resid_pre", layer][:, -1]` over the layers you want.
Concatenate the batches along dimension 1.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
We'll collect activations at five layers, so we can see how the final position's content changes through the network, and keep layer 10 as our main object of study. We'll call the
layer-10 activations `acts`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

ANALYSIS_LAYERS = [2, 4, 6, 8, 10]
acts_by_layer = get_final_activations(model, tokens, ANALYSIS_LAYERS)
acts = acts_by_layer[ANALYSIS_LAYERS.index(LAYER)]
acts_unit = F.normalize(acts, dim=-1)  # unit-norm versions, which is what the NLA will reconstruct
act_scale = acts[train_idx].norm(dim=-1).mean().item()  # used to rescale reconstructions before patching

for layer, layer_acts in zip(ANALYSIS_LAYERS, acts_by_layer):
    norms = layer_acts.norm(dim=-1)
    print(f"layer {layer:>2}: mean norm {norms.mean():6.1f}, std of norm {norms.std():5.1f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:NORMS -->layer  2: mean norm   58.5, std of norm   0.4
layer  4: mean norm   69.7, std of norm   1.3
layer  6: mean norm   83.2, std of norm   1.7
layer  8: mean norm  109.0, std of norm   2.8
layer 10: mean norm  156.1, std of norm   2.4</pre>

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Scoring a reconstruction: fraction of variance explained

An autoencoder needs a reconstruction metric. The NLA paper uses the **fraction of variance explained** (FVE). All activations are first normalised to unit L2 norm, $\tilde h = h /
\|h\|$; the reconstructor is trained to predict $\tilde h$; and over a set of $N$ held-out activations,

$$
\text{FVE} = 1 - \frac{\sum_{i=1}^N \|\tilde h_i - \hat h_i\|^2}{\sum_{i=1}^N \|\tilde h_i - \bar h\|^2}, \qquad \bar h = \frac{1}{N}\sum_{i=1}^N \tilde h_i .
$$

The denominator is the total variance of the (normalised) activations around their mean. So:

- **FVE = 0** is what you get by always predicting the mean activation, i.e. by ignoring the input entirely. It's the score of a description that says nothing.
- **FVE = 1** is perfect reconstruction.
- **FVE < 0** is possible, and means the reconstruction is *worse* than the mean. Confidently wrong reconstructions (like decoding the description of a different sentence) will do this.

Why normalise? The norm of the residual stream varies a lot between layers and somewhat between inputs, and we mostly care about its *direction*. It also makes FVE comparable
across layers. When we patch a reconstruction back into the model, we'll rescale it to the typical norm at that layer (`act_scale`).
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `fve`

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 5-10 minutes on this exercise.
> ```

Implement `fve`. The first argument is the raw activations (normalise them inside the function); the second is the reconstruction, which is already a prediction of the *normalised*
activation (so don't normalise it). Return a Python float.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def fve(h: Float[Tensor, "batch d_model"], h_hat: Float[Tensor, "batch d_model"]) -> float:
    """Fraction of variance of the unit-normalised activations `h` explained by the reconstructions `h_hat`."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    h_unit = F.normalize(h.float(), dim=-1)
    h_hat = h_hat.float().to(h_unit.device)
    residual = (h_unit - h_hat).pow(2).sum()
    total = (h_unit - h_unit.mean(dim=0)).pow(2).sum()
    return (1 - residual / total).item()
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_fve(fve)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`F.normalize(h, dim=-1)` gives unit-norm rows. Then compute the summed squared error, and the summed squared deviation from the column means (`h_unit.mean(dim=0)`), and return one
minus their ratio.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Description oracles: what could a perfect description explain?

Before building any NLA, let's ask a simpler question. Suppose a description says *exactly* "the subject is plural and the tense is past", and nothing else. What's the best
reconstruction any decoder could produce from it?

Under squared error, the best guess given a description is the **average activation of all sentences with that description**. So if we group the training sentences by what the
description would say, and predict each held-out activation by the mean of its group, we get (an estimate of) the best FVE that any reconstructor reading *that kind of description*
could achieve. We'll call this a **description oracle**: it tells us how much of the activation a description *could* carry, before we worry about whether a verbaliser can write it
or a reconstructor can read it.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `class_mean_reconstruction`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement `class_mean_reconstruction`. It takes the training activations, a label for each training activation (any hashable value: a string, or a tuple of strings when a
description mentions several facts), and a label for each held-out activation. It should return, for each held-out activation, the mean of the **unit-normalised** training
activations that have the same label. If a held-out label never occurs in training, fall back to the mean of all training activations.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def class_mean_reconstruction(
    train_h: Float[Tensor, "n_train d_model"], train_labels: list, test_labels: list
) -> Float[Tensor, "n_test d_model"]:
    """Predicts each test activation as the mean unit-normalised training activation with the same label."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    train_unit = F.normalize(train_h.float(), dim=-1)
    groups: dict = {}
    for i, label in enumerate(train_labels):
        groups.setdefault(label, []).append(i)
    means = {label: train_unit[idx].mean(dim=0) for label, idx in groups.items()}
    fallback = train_unit.mean(dim=0)
    return t.stack([means.get(label, fallback) for label in test_labels])
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_class_mean_reconstruction(class_mean_reconstruction)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Build a dictionary from label to the list of training indices with that label (`dict.setdefault` is handy), then a dictionary from label to the mean of those (normalised) rows.
Look up each test label, using `.get(label, fallback)`.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Now we can map what the final-position activation contains. For each of five layers and each kind of description, we compute the held-out FVE of the description oracle. We also
include a **control**: random labels with 32 classes (the same number of classes as the `subject` description). Random labels carry no information about the sentence, so this
should score about zero; it tells us how much FVE we'd get just from overfitting group means.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

DESCRIPTION_KINDS = {
    "number": ["number"],
    "tense": ["tense"],
    "number + tense": ["number", "tense"],
    "subject": ["subject"],
    "time word": ["time_word"],
    "preposition": ["preposition"],
    "object": ["object"],
    "subject + time word": ["subject", "time_word"],
}


def description_labels(df: pd.DataFrame, columns: list[str]) -> list[tuple]:
    return list(df[columns].itertuples(index=False, name=None))


rng = np.random.default_rng(0)
random_labels = rng.integers(0, 32, len(df)).tolist()
oracle_table = {}
for name, columns in DESCRIPTION_KINDS.items():
    labels_ = description_labels(df, columns)
    train_labels, test_labels = [labels_[i] for i in train_idx], [labels_[i] for i in test_idx]
    oracle_table[name] = [
        fve(layer_acts[test_idx], class_mean_reconstruction(layer_acts[train_idx], train_labels, test_labels))
        for layer_acts in acts_by_layer
    ]
oracle_table["control: random labels"] = [
    fve(
        layer_acts[test_idx],
        class_mean_reconstruction(layer_acts[train_idx], random_labels[:N_TRAIN], random_labels[N_TRAIN:]),
    )
    for layer_acts in acts_by_layer
]
oracle_df = pd.DataFrame(oracle_table, index=ANALYSIS_LAYERS).T
utils.plot_factor_heatmap(
    oracle_df,
    "Description oracles: held-out FVE of the best reconstruction from each kind of description",
    control_row="control: random labels",
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/description_oracles.png" width="700">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret this heatmap?</summary>

Read the table column by column, i.e. moving through the network.

**Layers 2-4: the final position is mostly about the words just before it.** At layer 4, knowing the *object* noun (the word right before `really`) explains 54% of the variance,
more than any other single fact, while the subject explains only 8%. Early on, a position mostly carries information about its own and nearby tokens, and `really` follows the
object. (At layer 2 the time word, the very first word of the sentence, also explains 29%.)

**Layers 6-10: the subject moves in, the object moves out.** By layer 10, the subject explains 46% and the object only 7%. The model has gathered what it needs to predict the verb,
which noun the subject is, into the position where the verb gets predicted. The time word explains 23%, and `subject + time word` together explain 71%, roughly the sum of the two:
they seem to be stored largely independently.

**The two facts that decide the verb explain very little variance.** `number + tense` is *everything* GPT-2 needs to choose between is/are/was/were, and it explains only 18% of the
variance at layer 10. The activation carries much more than the verb decision: *which* noun the subject is (not just whether it's plural), *which* of the three past-tense adverbs
was used, and so on. Hold on to this; it will matter a lot.

**The control fails, as it should.** Random labels with 32 classes explain -1% at every layer: group means overfit slightly, and that's all. So every clearly positive number in the
table is real information about the sentence.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Does the model *use* what's in the activation?

FVE measures how much of the activation's variance a reconstruction captures. But an interpretability tool should also tell us about what the model *does* with the activation. The
NLA paper checks this kind of thing in several ways; the simplest version is to **patch the reconstruction back into the model** and see whether the model behaves as it did with
the original activation.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `patch_final_position`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement `patch_final_position`. It runs the model on `tokens`, overwriting the residual stream before block `layer` at the **final** position with `vectors` (one vector per
sentence, already at the right scale), and returns the logits of the four verbs at the final position, shape `(batch, 4)`. Process the batch in chunks of `batch_size`, and return a
CPU tensor.

Use `model.run_with_hooks` with a hook on `blocks.{layer}.hook_resid_pre`, as in the Patchscopes exercises in [7.1].
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.inference_mode()
def patch_final_position(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    vectors: Float[Tensor, "batch d_model"],
    layer: int,
    verb_ids: Int[Tensor, " 4"],
    batch_size: int = 250,
) -> Float[Tensor, "batch 4"]:
    """Runs `tokens` with resid_pre[layer][:, -1] replaced by `vectors`; returns the four verb logits."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    out = []
    for start in range(0, len(tokens), batch_size):
        batch_vectors = vectors[start : start + batch_size].to(model.cfg.device)

        def hook(resid: Float[Tensor, "batch seq d_model"], hook) -> Float[Tensor, "batch seq d_model"]:
            resid[:, -1] = batch_vectors
            return resid

        logits = model.run_with_hooks(
            tokens[start : start + batch_size].to(model.cfg.device),
            fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", hook)],
        )
        out.append(logits[:, -1, verb_ids.to(model.cfg.device)].cpu())
    return t.cat(out)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_patch_final_position(patch_final_position, model, verb_ids)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Inside the loop over chunks, define a hook function `hook(resid, hook)` that sets `resid[:, -1] = batch_vectors` and returns `resid`. Python closures capture `batch_vectors` by
name, so define the hook *inside* the loop (or pass the vectors in some other way).
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### A behavioural test that doesn't test anything

The obvious test is to splice a reconstruction back into the *same* sentence and check that GPT-2 still picks the same verb. Before using it on a real reconstruction, let's try it
with the most useless reconstruction there is: the **mean activation**, the same vector for every sentence (FVE = 0 by definition). Make a prediction: how often will GPT-2 still
pick the right verb?
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

mean_vector = acts_unit[train_idx].mean(dim=0)
mean_patch = (F.normalize(mean_vector, dim=-1) * act_scale).expand(len(test_idx), -1)
in_place_logits = patch_final_position(model, tokens[test_idx], mean_patch, LAYER, verb_ids)
print(f"Mean activation spliced into its own sentence: right verb {(in_place_logits.argmax(-1) == labels[test_idx]).float().mean():.1%}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:INPLACE -->Mean activation spliced into its own sentence: right verb 90.5%</pre>

</details>

The mean activation, which contains no information about the sentence at all, "preserves behaviour" 90.5% of the time. That's because the layers after the patch (blocks 10 and 11)
can still attend to the earlier positions, where the subject and the time adverb are, and recover most of what they need from there. A behavioural test that a content-free vector
passes isn't measuring anything about the reconstruction. Any reconstruction would score at least this well.

This is worth remembering beyond NLAs: **always run the null intervention through your behavioural metric first.** If the control passes, change the test.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### The transplant test

Here's a test the mean activation can't pass. For each held-out sentence $A$, pick a partner sentence $B$ whose correct verb is *different*. Run GPT-2 on $B$, but replace $B$'s
final-position activation at layer 10 with (a reconstruction of) $A$'s. If the patched vector carries $A$'s subject number and tense strongly enough, GPT-2 will predict $A$'s verb
even though $B$'s context says otherwise.

We score the **transplant accuracy**: the fraction of transplants where the patched model predicts $A$'s verb. Because $B$'s verb is always different from $A$'s, a vector that
carries nothing (so the model just follows $B$'s context) scores 0%, and picking a random verb scores 25%. And the real activation of $A$ gives us a ceiling.

Here's a single transplant, the one from the introduction. We print GPT-2's probabilities over the four verbs (renormalised to sum to one).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

source, target = "Yesterday, the keys behind the door really", "Today, the dog near the trees really"
source_act = get_final_activations(model, model.to_tokens(source), [LAYER])[0]
for name, sentence, patch in [
    ("source", source, None),
    ("target", target, None),
    ("target, with the source's layer-10 activation", target, source_act),
]:
    sentence_tokens = model.to_tokens(sentence)
    if patch is None:
        with t.inference_mode():
            logits = model(sentence_tokens)[:, -1, verb_ids.to(device)].cpu()
    else:
        logits = patch_final_position(model, sentence_tokens, patch, LAYER, verb_ids)
    probs = logits.softmax(-1)[0]
    print(f"{name:<46} " + "  ".join(f"{v.strip()}: {p:.2f}" for v, p in zip(VERBS, probs)))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:DEMO -->source                                         is: 0.00  are: 0.07  was: 0.03  were: 0.90
target                                         is: 0.75  are: 0.00  was: 0.25  were: 0.00
target, with the source's layer-10 activation  is: 0.07  are: 0.22  was: 0.18  were: 0.53</pre>

</details>

On its own, the target sentence gets `is` (0.75). With the source's layer-10 activation pasted into its final position, and nothing else changed, GPT-2 prefers `were` (0.53):
plural and past, like the source, even though the target's own subject is singular and its time adverb is present tense. It's not a complete takeover (the target's context still
pulls some probability towards the other three verbs), but one vector at one position is enough to flip the decision.

Here's a helper that picks a partner for every held-out sentence:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def make_partners(labels: Int[Tensor, " n"], idx: Int[Tensor, " k"], seed: int = 0) -> Int[Tensor, " k"]:
    """For each index in `idx`, a random other index in `idx` whose label is different."""
    generator = t.Generator().manual_seed(seed)
    partners = []
    for i in idx.tolist():
        while True:
            j = idx[t.randint(0, len(idx), (1,), generator=generator)].item()
            if labels[j] != labels[i]:
                partners.append(j)
                break
    return t.tensor(partners)


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `transplant_accuracy`

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement `transplant_accuracy`. `h_hat` holds one reconstruction per source sentence (rows aligned with `src_idx`), as a prediction of the *unit-normalised* activation. Rescale
each row to unit norm and then multiply by `scale`, patch it into the partner sentences `tokens[dst_idx]` with `patch_final_position`, and return the fraction of rows where the
predicted verb (argmax of the four verb logits) equals `labels[src_idx]`.

Why rescale every reconstruction to the same norm? A reconstruction that hedges between several possibilities will usually have a smaller norm than a real activation, and patching
a short vector into the residual stream has a weaker effect for reasons unrelated to its content. Rescaling makes the comparison about *direction*, which is also what FVE measures.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def transplant_accuracy(
    model: HookedTransformer,
    tokens: Int[Tensor, "n seq"],
    labels: Int[Tensor, " n"],
    h_hat: Float[Tensor, "k d_model"],
    src_idx: Int[Tensor, " k"],
    dst_idx: Int[Tensor, " k"],
    scale: float,
    layer: int = LAYER,
) -> float:
    """Patches each reconstruction (rescaled) into its partner sentence; returns how often the source's verb wins."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    vectors = F.normalize(h_hat.float(), dim=-1) * scale
    logits = patch_final_position(model, tokens[dst_idx], vectors, layer, verb_ids)
    return (logits.argmax(dim=-1) == labels[src_idx]).float().mean().item()
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_transplant_accuracy(transplant_accuracy, model, verb_ids)
# END HIDE


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Now let's run the transplant test with the real activations and three controls:

- **A's own activation**: the ceiling. How strongly does the layer-10 final-position vector determine the verb?
- **the mean activation**: carries no information, so GPT-2 should follow B's context and score about 0%.
- **a random other sentence's activation**: a real activation, so it pushes strongly towards *some* verb, but the wrong sentence's. It should score around chance.
- **a random direction**: a Gaussian vector with the same norm.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

partners = make_partners(labels, test_idx)
shuffled_sources = test_idx[t.randperm(len(test_idx), generator=t.Generator().manual_seed(1))]


def transplant(h_hat: Tensor) -> float:
    return transplant_accuracy(model, tokens, labels, h_hat, test_idx, partners, act_scale)


transplant_controls = {
    "A's own activation": transplant(acts_unit[test_idx]),
    "mean activation": transplant(mean_vector.expand(len(test_idx), -1)),
    "another sentence's activation": transplant(acts_unit[shuffled_sources]),
    "random direction": transplant(t.randn(len(test_idx), model.cfg.d_model, generator=t.Generator().manual_seed(2))),
}
utils.plot_bars(
    transplant_controls,
    "Transplant test at layer 10: patch A's activation into a sentence B with a different verb",
    ylabel="transplant accuracy\n(model predicts A's verb)",
    chance=0.25,
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/transplant_controls.png" width="650">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret this plot?</summary>

**A's own activation: 92%.** Everything else in sentence B (its subject, its time adverb) points to a different verb, yet replacing a single vector at a single position makes GPT-2
predict A's verb 92% of the time. So by layer 10 the verb decision lives almost entirely at the final position; the last two blocks only occasionally overrule it from context. This
is the ceiling for any reconstruction.

**Mean activation: 2%.** With a content-free vector, GPT-2 falls back on B's context, which by construction points to a different verb. This is the control that "passed" the
in-place test with 90.5%; here it fails completely, which is exactly what we want from a test.

**Another sentence's activation: 25%.** A real activation from a random third sentence pushes towards *that* sentence's verb, which happens to be A's verb about a quarter of the
time. That's chance.

**Random direction: 20%.** A random vector of the right norm doesn't carry A's verb either.

So a reconstruction that carries nothing scores near 0%, one that carries the wrong thing scores near 25%, and a perfect one scores 92%.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Why did we pick layer 10? Let's run the same test with the real activations from each of our five layers (patching layer $\ell$'s activation into block $\ell$ of the partner
sentence), alongside the mean-activation control.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

ceiling_by_layer, mean_by_layer = [], []
for layer, layer_acts in zip(ANALYSIS_LAYERS, acts_by_layer):
    scale = layer_acts[train_idx].norm(dim=-1).mean().item()
    layer_unit = F.normalize(layer_acts, dim=-1)
    ceiling_by_layer.append(transplant_accuracy(model, tokens, labels, layer_unit[test_idx], test_idx, partners, scale, layer))
    mean_by_layer.append(transplant_accuracy(model, tokens, labels, layer_unit[train_idx].mean(0).expand(len(test_idx), -1), test_idx, partners, scale, layer))
utils.plot_curves(
    {"A's own activation": (ANALYSIS_LAYERS, ceiling_by_layer), "control: mean activation": (ANALYSIS_LAYERS, mean_by_layer)},
    "Transplant accuracy by layer: when does the final position decide the verb?",
    xlabel="layer of the patched activation (residual stream before block)",
    ylabel="transplant accuracy",
    hlines={"chance (25%)": 0.25},
    ylim=(0, 1.02),
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/transplant_by_layer.png" width="650">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
At layers 2, 4 and 6, transplanting the real activation does essentially nothing: GPT-2 predicts B's verb, so transplant accuracy is 0-2%. The verb decision hasn't reached the
final position yet, and the later blocks take the subject's number and the tense from B's context. At layer 8 the final position starts to matter (35%), and by layer 10 it
dominates (92%). The mean-activation control stays at 0-2% throughout.

This is why we'll verbalise layer 10. It's also a warning: at early layers, a transplant test can't tell a good reconstruction from a bad one, because even the real activation
fails it. To evaluate an early-layer NLA by its behavioural effect, you'd need a different test.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Two axes, not one

We now have two ways to score a reconstruction: how much variance it explains (FVE), and whether the model uses it like the original (transplant accuracy). Let's score a few
description oracles on both. Before running the cell, guess: which of these descriptions will have the highest FVE, and which the highest transplant accuracy?
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

oracle_points = {}
for name in ["number + tense", "subject", "time word", "subject + time word"]:
    labels_ = description_labels(df, DESCRIPTION_KINDS[name])
    h_hat = class_mean_reconstruction(acts[train_idx], [labels_[i] for i in train_idx], [labels_[i] for i in test_idx])
    oracle_points[name] = (fve(acts[test_idx], h_hat), transplant(h_hat))
display(pd.DataFrame(oracle_points, index=["FVE", "transplant accuracy"]).T.round(3))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:TWO_AXES -->                       FVE  transplant accuracy
number + tense       0.175                0.995
subject              0.463                0.345
time word            0.235                0.285
subject + time word  0.715                0.929</pre>

</details>

<details>
<summary>Help - why are these two columns so different?</summary>

**`number + tense`** explains only 17.5% of the variance. But its reconstruction, which is just one of four average vectors, makes GPT-2 predict the right verb in 99.5% of
transplants: *more* often than the real activation (92%). The class mean is a distilled version of the verb information, without the sentence-specific details that occasionally let
B's context win.

**`subject`** explains 46% of the variance, far more, but only gets 34.5% of transplants right. The noun form tells you the number (`tables` is plural) but not the tense. Remember
that B's verb always differs from A's, in number, in tense, or in both. If the transplanted vector fixes the number and the model takes the tense from B's context, the result is
A's verb only when B differs from A in number alone: about a third of the time. **`time word`** has the mirror-image problem, and scores 28.5%.

**`subject + time word`** gets both: 71.5% of the variance and 92.9% of transplants, matching the real activation.

The lesson is that FVE measures *how much* of the activation is reconstructed, weighted by variance, while transplant accuracy measures whether the *particular directions the model
reads* are reconstructed. These can be very different. An autoencoder trained to maximise FVE will spend its capacity on the high-variance facts (which noun, which time word),
which may or may not be the ones the model uses. We'll come back to this in part 4️⃣.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 1️⃣

- GPT-2 Small gets subject-verb agreement right on every sentence in our dataset, including the ones where the attractor noun disagrees with the subject.
- We'll score reconstructions with the NLA paper's FVE: unit-normalise the activations, and compare the squared error to the variance around the mean. FVE = 0 is "says nothing", FVE = 1 is perfect.
- **Description oracles** (class means) tell us how much of the activation a given kind of description could explain. At layer 10, the final position mostly encodes the subject (46%) and the time word (23%); the two facts that decide the verb (number and tense) explain only 18%. At layer 4, it mostly encodes the object.
- A behavioural test that the mean activation passes (in-place splicing: 90.5%) is useless. The **transplant test** has a ceiling of 92% (the real activation), and its controls score 2-25%.
- FVE and behaviour are different axes: `number + tense` has an FVE of 0.18 and a transplant accuracy of 99.5%; `subject` has 0.46 and 34.5%.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 2️⃣ A warm-started NLA
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Natural language autoencoders

A natural language autoencoder has two halves:

- The **activation verbaliser** (AV) is a language model that reads an activation $h$ and writes a description $z$. In the paper, the AV is a copy of the target model itself: the activation (normalised, then scaled by a fixed constant) is inserted into a fixed prompt in place of a token embedding, and the model generates an explanation by sampling at temperature 1.
- The **activation reconstructor** (AR) reads the description $z$ and outputs a reconstruction $\hat h$. In the paper, the AR is the target model *truncated* to its first $\ell$ layers (where $\ell$ is the layer the activation came from), reading $z$ wrapped in a fixed prompt, followed by a learned affine map applied to the layer-$\ell$ activation at the final token.

The objective is reconstruction: make $\|\tilde h - \text{AR}(z)\|^2$ small, for $z \sim \text{AV}(\cdot \mid h)$. Nothing in the objective says the description must be *true*, or
*readable*, only that it must let the AR rebuild $h$. The bet is that if you start from a model that writes sensible text, and don't let it drift too far, the cheapest way to carry
the information is to describe it in plain language.

Training happens in two stages:

1. **Warm start.** Train the AV and AR with supervised learning on pairs of activations and descriptions. In the paper, the descriptions are summaries of the text *preceding* the activation, written by another Claude model. These summaries describe the context, not the activation, so they're only a proxy; the paper reports that the warm-started NLAs reach an FVE of about 0.3-0.4.
2. **End-to-end training.** Sample several descriptions per activation from the AV, reward each one by how well the AR reconstructs the activation from it, and update the AV with a policy gradient (GRPO), plus a KL penalty that keeps the AV close to its warm-started self. The AR is trained by plain regression on the AV's samples. After this stage, the paper's NLAs reach an FVE of about 0.6-0.8.

Here's how our miniature version compares:

| | NLA paper | this notebook |
|---|---|---|
| target model | Claude models (and open models) | GPT-2 Small |
| activation | residual stream at a middle-to-late layer, unit-normalised | `resid_pre` of block 10 at the final position, unit-normalised |
| AV | a copy of the target model, activation injected as a token embedding; writes long explanations | a 2-layer transformer trained from scratch, activation as the first position; writes up to 4 words from a 47-word vocabulary |
| AR | target model truncated to layer $\ell$, plus an affine map on the final token | *frozen* GPT-2 Small truncated to layer 10, plus a trained affine map on the final token |
| warm start | supervised training on model-written summaries of the context | supervised training on random lists of true facts about the sentence |
| end-to-end | GRPO, reward $-\log \|h - \hat h\|^2$, KL penalty to the warm start, AR trained by regression | the same recipe, at toy scale |

Our reconstructor really is GPT-2 reading English. That will matter: it means the reconstructor can make sense of words it was never trained on, which you'll exploit at the end of
the section.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## The bottleneck language

The verbaliser writes from a fixed vocabulary of 47 words: the 32 noun forms, the 6 time words (lower-cased), the four "abstract" words `singular`, `plural`, `present`, `past`, the
4 prepositions, and `"."`, which ends a description. A description is at most `MAX_WORDS = 4` words, followed by `"."`.

For the warm start we need a **teacher**: something that, given a sentence, writes descriptions in this language. Like the paper's summaries, our teacher describes the *sentence*,
not the activation: it doesn't know which facts GPT-2 has moved to the final position. For each sentence there are five true facts:

| fact | words | example |
|---|---|---|
| subject | the subject noun | `houses` |
| number | `singular` / `plural` | `plural` |
| tense | `present` / `past` | `past` |
| time | the time word | `earlier` |
| object | preposition + object noun | `behind window` |

The teacher includes each fact independently with probability ½ (in the order above), and rejects lists that are empty or longer than four words. So every teacher description is
**true**, but which true things it says is random. A good NLA should learn which of them are worth saying.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

NOUN_WORDS = [form for pair in NOUNS for form in pair]
TIME_WORDS = [word.lower() for word in TIME_ADVERBS]
ABSTRACT_WORDS = ["singular", "plural", "present", "past"]
WORDS = NOUN_WORDS + TIME_WORDS + ABSTRACT_WORDS + PREPOSITIONS + ["."]
EOS = WORDS.index(".")
MAX_WORDS = 4
FACT_ORDER = ["subject", "number", "tense", "time", "object"]


def true_facts(row) -> dict[str, list[str]]:
    """The five true facts about a sentence (a row of the dataset), each as a list of words."""
    return {
        "subject": [row.subject],
        "number": [row.number],
        "tense": [row.tense],
        "time": [row.time_word],
        "object": [row.preposition, row.object],
    }


def teacher_description(facts: dict[str, list[str]], rng: random.Random, max_words: int = MAX_WORDS) -> list[str]:
    """A random non-empty subset of the true facts, in canonical order, of at most `max_words` words."""
    while True:
        words = [word for fact in FACT_ORDER if rng.random() < 0.5 for word in facts[fact]]
        if 1 <= len(words) <= max_words:
            return words


def encode_descriptions(descriptions: list[list[str]], max_words: int = MAX_WORDS) -> Int[Tensor, "batch slots"]:
    """Word lists -> ids, padded with EOS to max_words + 1 slots (so the last slot is always EOS)."""
    ids = t.full((len(descriptions), max_words + 1), EOS, dtype=t.long)
    for i, words in enumerate(descriptions):
        assert len(words) <= max_words, f"description too long: {words}"
        if words:
            ids[i, : len(words)] = t.tensor([WORDS.index(word) for word in words])
    return ids


def decode_descriptions(ids: Int[Tensor, "batch slots"]) -> list[str]:
    """Ids -> strings, stopping at the first EOS."""
    texts = []
    for row in ids.tolist():
        words = []
        for i in row:
            if i == EOS:
                break
            words.append(WORDS[i])
        texts.append(" ".join(words))
    return texts


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

facts = [true_facts(row) for row in df.itertuples()]
teacher_rng = random.Random(0)
print(f"{len(WORDS)} words in the bottleneck vocabulary\n")
for i in range(4):
    examples = [" ".join(teacher_description(facts[i], teacher_rng)) for _ in range(3)]
    print(f"{df.sentence[i]!r:<52} -> {examples}")
print()
print(encode_descriptions([["houses", "earlier"], ["plural", "past", "behind", "window"]]))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:TEACHER -->47 words in the bottleneck vocabulary

'Now, the girl near the key really'                  -&gt; ['present now', 'girl present now', 'present']
'Yesterday, the book behind the chair really'        -&gt; ['book', 'book behind chair', 'book singular']
'Earlier, the key beside the trees really'           -&gt; ['key past', 'key past', 'singular earlier']
'Yesterday, the doors with the girls really'         -&gt; ['plural past yesterday', 'plural', 'doors past with girls']

tensor([[11, 36, 46, 46, 46],
        [39, 41, 43, 30, 46]])</pre>

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## The activation verbaliser

Our verbaliser is a tiny decoder-only transformer, built the way the paper's AV uses its language model: the activation goes in where a token embedding would go. Concretely, for an
activation $h$ and the words written so far $w_1, \dots, w_t$, the input sequence is

$$
\big[\; W_{in} \tilde h,\;\; E[w_1],\;\; \dots,\;\; E[w_t] \;\big] + P
$$

where $W_{in}$ is a linear map from $d_{model}=768$ to the verbaliser's width (128), $E$ is a learned word embedding, and $P$ is a learned positional embedding. This goes through
two causal transformer blocks and an unembedding, so the output at position $i$ is the distribution over the word in slot $i+1$. In particular, the output at position 0 (the
activation itself) is the distribution of the first word.

A description has `max_words + 1` slots (the last of which is always `"."`), so when we score a full description we feed in the first `max_words` slots and get `max_words + 1`
predictions, one for each slot.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `ActivationVerbaliser.forward`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

We've written `__init__` for you. Implement `forward`: normalise `h`, map it to the first position with `act_in`, concatenate the previous words' embeddings after it, add the
positional embeddings, run `self.blocks` with a causal mask, and apply `self.unembed`. The output should have shape `(batch, t + 1, n_words)`.

The forward pass accepts either word ids (`prev_ids`) or word embeddings (`prev_embeds`). The ids version is what we'll use almost everywhere; the embeddings version lets part 3️⃣
feed in *soft* words. The first line, which handles this, is written for you.

For the causal mask, use `nn.Transformer.generate_square_subsequent_mask(seq_len, device=...)` and pass it to `self.blocks(x, mask=mask, is_causal=True)`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class ActivationVerbaliser(nn.Module):
    def __init__(
        self, d_act: int = 768, n_words: int = len(WORDS), d_model: int = 128, n_layers: int = 2,
        n_heads: int = 4, max_words: int = MAX_WORDS,
    ):  # fmt: skip
        super().__init__()
        self.max_words = max_words
        self.act_in = nn.Linear(d_act, d_model)
        self.word_embed = nn.Embedding(n_words, d_model)
        self.pos_embed = nn.Parameter(t.randn(max_words + 1, d_model) * 0.02)
        block = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=4 * d_model, dropout=0.0, batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(block, n_layers, enable_nested_tensor=False)
        self.unembed = nn.Linear(d_model, n_words)

    def forward(
        self,
        h: Float[Tensor, "batch d_act"],
        prev_ids: Int[Tensor, "batch t"] | None = None,
        prev_embeds: Float[Tensor, "batch t d_model"] | None = None,
    ) -> Float[Tensor, "batch t_plus_1 n_words"]:
        """Logits for the next word at every position: position 0 is the activation, position i > 0 is word i."""
        if prev_embeds is None:
            prev_embeds = self.word_embed(prev_ids)
        # EXERCISE
        # raise NotImplementedError()
        # END EXERCISE
        # SOLUTION
        x = t.cat([self.act_in(F.normalize(h, dim=-1))[:, None], prev_embeds], dim=1)
        x = x + self.pos_embed[: x.shape[1]]
        mask = nn.Transformer.generate_square_subsequent_mask(x.shape[1], device=x.device)
        return self.unembed(self.blocks(x, mask=mask, is_causal=True))
        # END SOLUTION


# HIDE
if MAIN:
    tests.test_verbaliser_forward(ActivationVerbaliser)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`self.act_in(F.normalize(h, dim=-1))` has shape `(batch, d_model)`; add a sequence dimension with `[:, None]` and `t.cat` it with `prev_embeds` along dimension 1. Slice
`self.pos_embed[:seq_len]` so it broadcasts over the batch.
</details>

<details>
<summary>Help - the test says my output at position k depends on words after k.</summary>

You've forgotten the causal mask (or passed it without `is_causal=True`, which some PyTorch versions need to use the mask correctly in the fast path). Without it, each position can
attend to later words, and teacher-forced training would let the model "see the answer".
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - log-probabilities and masks for descriptions

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Both the warm start (maximum likelihood) and the policy gradient need $\log p(z \mid h) = \sum_i \log p(z_i \mid h, z_{<i})$ for a given description $z$. Implement two functions:

- `description_logprobs(av, h, ids)` takes descriptions as ids of shape `(batch, slots)` (where `slots = max_words + 1`), runs the verbaliser on the first `slots - 1` ids, and returns **both** the full log-probability tensor of shape `(batch, slots, n_words)` and the log-probability of each actual word, shape `(batch, slots)`. (We'll need the full distributions for the KL penalty in part 3️⃣.)
- `description_mask(ids)` returns a boolean mask of shape `(batch, slots)` which is `True` for every slot up to **and including** the first EOS, and `False` after it. The words after the first EOS are padding: we never sampled them, so they shouldn't count towards the log-probability.

For example, for the description `houses earlier . . .`, the mask is `[True, True, True, False, False]`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def description_logprobs(
    av: ActivationVerbaliser, h: Float[Tensor, "batch d_act"], ids: Int[Tensor, "batch slots"]
) -> tuple[Float[Tensor, "batch slots n_words"], Float[Tensor, "batch slots"]]:
    """Returns (log-probabilities over the vocabulary at every slot, log-probability of each word in `ids`)."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    logprobs = av(h, ids[:, :-1]).log_softmax(dim=-1)
    token_logprobs = logprobs.gather(dim=-1, index=ids[..., None])[..., 0]
    return logprobs, token_logprobs
    # END SOLUTION


def description_mask(ids: Int[Tensor, "batch slots"]) -> Bool[Tensor, "batch slots"]:
    """True for every slot up to and including the first EOS."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    is_eos = (ids == EOS).long()
    eos_before = is_eos.cumsum(dim=-1) - is_eos  # number of EOS strictly before each slot
    return eos_before == 0
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_description_logprobs(description_logprobs, ActivationVerbaliser)
    tests.test_description_mask(description_mask, EOS)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint (logprobs)</summary>

`av(h, ids[:, :-1])` gives logits of shape `(batch, slots, n_words)`, where position $i$ predicts slot $i$. Apply `log_softmax`, then `gather` along the last dimension with
`ids[..., None]`.
</details>

<details>
<summary>Hint (mask)</summary>

A slot should be masked out if there is at least one EOS *strictly before* it. `cumsum` of the EOS indicator counts the EOS tokens up to and including each slot; subtract the
indicator itself to get the count strictly before.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - sample descriptions

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15 minutes on this exercise.
> ```

Implement `sample_descriptions`, which generates a description for each activation, one word at a time:

- Start with an empty `(batch, 0)` tensor of ids. At each of the `av.max_words` steps, run the verbaliser on the ids so far, take the logits at the last position, and either sample from them (at the given `temperature`) or take the argmax if `greedy=True`. Append the chosen word.
- Then append a final EOS, so the output has `av.max_words + 1` slots.
- Finally, overwrite every slot *after* the first EOS with EOS, so that a description ends cleanly (this way two descriptions that say the same thing have the same ids).

Return the ids, of shape `(batch, av.max_words + 1)`. This function doesn't need gradients, so it's decorated with `@t.no_grad()`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.no_grad()
def sample_descriptions(
    av: ActivationVerbaliser, h: Float[Tensor, "batch d_act"], greedy: bool = False, temperature: float = 1.0
) -> Int[Tensor, "batch slots"]:
    """Autoregressively samples (or greedily decodes) one description per activation."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    ids = t.zeros(h.shape[0], 0, dtype=t.long, device=h.device)
    for _ in range(av.max_words):
        logits = av(h, ids)[:, -1] / temperature
        if greedy:
            next_ids = logits.argmax(dim=-1)
        else:
            next_ids = t.distributions.Categorical(logits=logits).sample()
        ids = t.cat([ids, next_ids[:, None]], dim=1)
    ids = t.cat([ids, t.full_like(ids[:, :1], EOS)], dim=1)
    ids[~description_mask(ids)] = EOS
    return ids
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_sample_descriptions(sample_descriptions, ActivationVerbaliser, EOS)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`t.distributions.Categorical(logits=logits).sample()` samples one index per row. For the final step, `ids[~description_mask(ids)] = EOS` uses the mask you just wrote: everything
that isn't "up to and including the first EOS" becomes EOS.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - warm-start the verbaliser

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Now train the verbaliser to imitate the teacher. Implement `train_verbaliser`: for each step, sample a batch of training indices, draw a **fresh** teacher description for each (so
the verbaliser sees many different true descriptions of the same sentence over training), encode them, and minimise the negative log-likelihood of the description, summed over the
masked slots and averaged over the batch. Return the list of losses.

The activations `h_train` and `facts_train` are aligned (row $i$ of one belongs with entry $i$ of the other). The defaults take about 20 seconds on a GPU.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def train_verbaliser(
    av: ActivationVerbaliser,
    h_train: Float[Tensor, "n d_act"],
    facts_train: list[dict],
    n_steps: int = 1000,
    batch_size: int = 256,
    lr: float = 1e-3,
    seed: int = 0,
) -> list[float]:
    """Supervised warm start: maximise the log-likelihood of random true (teacher) descriptions."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    rng = random.Random(seed)
    generator = t.Generator().manual_seed(seed)
    optimizer = t.optim.Adam(av.parameters(), lr=lr)
    device = next(av.parameters()).device
    losses = []
    for _ in tqdm(range(n_steps), desc="warm start"):
        idx = t.randint(0, len(h_train), (batch_size,), generator=generator)
        descriptions = [teacher_description(facts_train[i], rng, av.max_words) for i in idx.tolist()]
        ids = encode_descriptions(descriptions, av.max_words).to(device)
        _, token_logprobs = description_logprobs(av, h_train[idx].to(device), ids)
        loss = -(token_logprobs * description_mask(ids)).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
    # END SOLUTION


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

t.manual_seed(0)
facts_train = [facts[i] for i in train_idx]
h_train = acts_unit[train_idx].to(device)
h_test = acts_unit[test_idx].to(device)

av_warm = ActivationVerbaliser().to(device)
warm_losses = train_verbaliser(av_warm, h_train, facts_train)
print(f"loss: first step {warm_losses[0]:.2f}, last 100 steps {np.mean(warm_losses[-100:]):.2f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - what loss should I expect?</summary>

The loss starts around 15 nats (about $\ln 47 \approx 3.85$ nats for each of the four-or-so slots, i.e. a uniform guess) and falls to about 3.5 nats per description.

It can't go to zero, because the teacher picks which facts to mention at random. Each of the five facts is included with probability ½, and lists that are empty or longer than four
words are rejected. That rules out 6 of the 32 subsets (the empty one, the four 5-word ones, and the 6-word one), leaving 26 equally likely subsets, so even a verbaliser that knew
all the facts perfectly would have a loss of $\ln 26 \approx 3.26$ nats. The remaining gap of about 0.2 nats is uncertainty about the facts themselves: the verbaliser can't always
read every fact from the activation (probably mostly the object, which is only weakly present at layer 10).
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Let's see what it says about some held-out sentences. We print the greedy description, plus three samples:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

t.manual_seed(0)
show = test_idx[:8]
greedy_texts = decode_descriptions(sample_descriptions(av_warm, acts_unit[show].to(device), greedy=True))
sampled = [decode_descriptions(sample_descriptions(av_warm, acts_unit[show].to(device))) for _ in range(3)]
display(pd.DataFrame(
    {"sentence": df.sentence[show.tolist()].values, "greedy": greedy_texts, "samples": [" | ".join(s) for s in zip(*sampled)]}
))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:WARM_SAMPLES -->                                       sentence                    greedy                                            samples
0    Currently, the tables with the keys really     tables plural present                   tables present | tables | tables
1   Earlier, the house behind the window really       house singular past  house past | house singular behind windows | past
2        Earlier, the boys with the tree really          boys plural past     boys plural earlier | boys past | boys earlier
3            Now, the cat with the chair really      cat singular present  singular present with chair | present with cha...
4       Earlier, the house with the dogs really       house singular past    house past | with dogs | past earlier with dogs
5  Previously, the tables with the birds really        tables plural past         with birds | tables past | past previously
6    Yesterday, the tree near the tables really        tree singular past  past yesterday near tables | yesterday near ta...
7        Now, the window behind the door really  singular now behind door   now | singular behind door | present behind door</pre>

</details>

Almost everything here is true (one sample says `behind windows` where the object is a single `window`): the verbaliser has learned to read the subject, its number, the tense, the
time word and the object out of the layer-10 activation, well enough to describe held-out sentences. The samples show how random the choice of facts is. The greedy descriptions
mostly come out as subject, number, tense (e.g. `tables plural present`), the first three facts in the teacher's order. That's true, but not efficient: `plural` is redundant once
you've said `tables`, and `present` carries much less information than `currently` would.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## The activation reconstructor

The reconstructor reads a description with GPT-2 itself. For the description `houses earlier`, padded to five slots, GPT-2 sees

```
<|endoftext|> houses earlier...
```

(each word with a leading space, and each EOS slot as a `"."` token), and we take the residual stream before block 10 at the **final** position, which is always a `"."`. That's the
paper's design: a fixed final token, so that the reconstructor must gather the description's content with attention, just as the target model had to gather the sentence's content
into its final position. Then a trained affine map turns these features into a reconstruction:

$$
\hat h = W \frac{f(z)}{\|f(z)\|} + b, \qquad f(z) = \text{GPT-2}_{\text{blocks } 0..9}(z)\,[\text{final position}] .
$$

(We normalise the features first because the residual stream norm can vary between descriptions, and it's the direction we want the affine map to read.) GPT-2 stays frozen, so the
only trained parameters are $W$ and $b$: 590,592 numbers.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `text_features` and `ActivationReconstructor`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15 minutes on this exercise.
> ```

Implement two things:

- `read_with_gpt2(model, gpt2_tokens, layer)`: run GPT-2 on a batch of token sequences (already including the BOS token) and return `resid_pre` of block `layer` at the final position, shape `(batch, d_model)`. Stop the forward pass as early as you can.
- `text_features(model, ids, layer)`: convert description ids (over `WORDS`) into GPT-2 tokens using `WORD_TOKEN_IDS` (defined below: it maps each word to its GPT-2 token, and EOS to the `"."` token), prepend a BOS column, and call `read_with_gpt2`.

Then implement `ActivationReconstructor.forward`, which normalises the features and applies `self.head`.

We split `read_with_gpt2` out because at the end of the section you'll use it to read descriptions containing words that *aren't* in the verbaliser's vocabulary.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

WORD_TOKEN_IDS = t.tensor([model.to_single_token(" " + word) for word in WORDS[:-1]] + [model.to_single_token(".")])
print(f"'houses earlier . . .' as GPT-2 sees it: {model.to_string(t.cat([t.tensor([model.tokenizer.bos_token_id]), WORD_TOKEN_IDS[encode_descriptions([['houses', 'earlier']])[0]]]))!r}")

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.no_grad()
def read_with_gpt2(
    model: HookedTransformer, gpt2_tokens: Int[Tensor, "batch seq"], layer: int
) -> Float[Tensor, "batch d_model"]:
    """resid_pre of block `layer` at the final position."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    _, cache = model.run_with_cache(
        gpt2_tokens.to(model.cfg.device), names_filter=f"blocks.{layer}.hook_resid_pre", stop_at_layer=layer + 1
    )
    return cache["resid_pre", layer][:, -1]
    # END SOLUTION


def text_features(
    model: HookedTransformer, ids: Int[Tensor, "batch slots"], layer: int = LAYER
) -> Float[Tensor, "batch d_model"]:
    """Reads descriptions (ids over WORDS) with GPT-2: BOS, then one token per slot (EOS slots become '.')."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    gpt2_tokens = WORD_TOKEN_IDS.to(ids.device)[ids]
    bos = t.full_like(gpt2_tokens[:, :1], model.tokenizer.bos_token_id)
    return read_with_gpt2(model, t.cat([bos, gpt2_tokens], dim=1), layer)
    # END SOLUTION


class ActivationReconstructor(nn.Module):
    def __init__(self, d_features: int = 768, d_act: int = 768):
        super().__init__()
        self.head = nn.Linear(d_features, d_act)

    def forward(self, features: Float[Tensor, "batch d_features"]) -> Float[Tensor, "batch d_act"]:
        """Affine map applied to the normalised GPT-2 features; returns a prediction of the unit-norm activation."""
        # EXERCISE
        # raise NotImplementedError()
        # END EXERCISE
        # SOLUTION
        return self.head(F.normalize(features, dim=-1))
        # END SOLUTION


# HIDE
if MAIN:
    tests.test_text_features(text_features, model, WORD_TOKEN_IDS, encode_descriptions)
    tests.test_reconstructor(ActivationReconstructor)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`WORD_TOKEN_IDS[ids]` converts a whole `(batch, slots)` tensor of word ids into GPT-2 token ids by indexing. Build a `(batch, 1)` column of `model.tokenizer.bos_token_id` and
concatenate it on the left. In `read_with_gpt2`, use `run_with_cache` with `names_filter` and `stop_at_layer=layer + 1`, as in `get_final_activations`.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - train the reconstructor

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

In an NLA, the reconstructor learns to read whatever the verbaliser writes. Implement `train_reconstructor`: for each step, sample a batch of training activations, **sample** a
description of each from the (fixed) verbaliser, compute the text features, and take a gradient step on the squared reconstruction error

$$
\mathcal{L}_{AR} = \frac{1}{B}\sum_{b} \|\tilde h_b - \text{AR}(z_b)\|^2 ,
$$

summed over the $d_{model}$ dimensions and averaged over the batch. Return the list of losses.

Note that the optimizer is passed in rather than created inside. That's deliberate: we'll keep training this same reconstructor in part 3️⃣, and Adam's first update with fresh
statistics moves *every* parameter by roughly the learning rate, which is enough to knock a converged 768×768 head badly off course. Passing the optimizer in lets us continue where
we left off.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def train_reconstructor(
    ar: ActivationReconstructor,
    optimizer: t.optim.Optimizer,
    av: ActivationVerbaliser,
    h_train: Float[Tensor, "n d_act"],
    n_steps: int = 400,
    batch_size: int = 256,
    seed: int = 0,
) -> list[float]:
    """Trains the reconstructor by regression on descriptions sampled from a fixed verbaliser."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    generator = t.Generator().manual_seed(seed)
    losses = []
    for _ in tqdm(range(n_steps), desc="reconstructor"):
        idx = t.randint(0, len(h_train), (batch_size,), generator=generator)
        h = h_train[idx]
        ids = sample_descriptions(av, h)
        loss = (ar(text_features(model, ids)) - h).pow(2).sum(dim=-1).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses
    # END SOLUTION


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

t.manual_seed(0)
ar_warm = ActivationReconstructor().to(device)
ar_warm_opt = t.optim.Adam(ar_warm.parameters(), lr=1e-3)
ar_losses = train_reconstructor(ar_warm, ar_warm_opt, av_warm, h_train)
total_variance = (h_train - h_train.mean(0)).pow(2).sum(-1).mean().item()
print(f"AR loss: first step {ar_losses[0]:.3f}, last 50 steps {np.mean(ar_losses[-50:]):.4f}")
print(f"(total variance of the unit-norm activations, i.e. the loss of predicting the mean: {total_variance:.4f})")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:AR_LOSS -->AR loss: first step 1.526, last 50 steps 0.0411
(total variance of the unit-norm activations, i.e. the loss of predicting the mean: 0.0634)</pre>

</details>

On the verbaliser's *sampled* descriptions, the reconstructor's loss ends at about 0.041, against 0.063 for predicting the mean: it explains roughly a third of the variance (1 -
0.041/0.063 ≈ 0.35) of the training activations. Sampled descriptions are often short or uninformative (`past`, `with dogs`), so this is lower than what we'll get from the
verbaliser's single best guess.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Evaluating the warm-started NLA

We'll evaluate NLAs in three ways, always on the 1,000 held-out sentences:

1. **FVE** of the reconstruction from the verbaliser's *greedy* description (its single best guess).
2. **Transplant accuracy** of the same reconstruction, from part 1️⃣.
3. **Claim accuracy**: are the descriptions *true*? This is the thing the reconstruction objective never directly asks for, so we need to check it separately. We can, because we know the ground truth.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `claim_accuracy`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement `claim_accuracy`, which parses each description into claims and checks them against the sentence's true facts. Read the words left to right:

- a **preposition followed by a noun** is one claim about the object, true if both words match the `object` fact (e.g. `behind window`);
- a **preposition not followed by a noun** is a malformed claim, and counts as false;
- any other **noun** is a claim about the subject;
- a **number word**, **tense word** or **time word** is a claim about that fact.

Return the fraction of all claims (pooled over all descriptions) that are true, and the average number of claims per description. You can use the word lists `NOUN_WORDS`,
`PREPOSITIONS`, `TIME_WORDS`, and `ABSTRACT_WORDS` (whose first two entries are number words and last two are tense words).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def claim_accuracy(descriptions: list[str], facts: list[dict[str, list[str]]]) -> tuple[float, float]:
    """Returns (fraction of claims that are true, mean number of claims per description)."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    n_claims, n_true = 0, 0
    for text, fact in zip(descriptions, facts):
        words = text.split()
        k = 0
        while k < len(words):
            word = words[k]
            if word in PREPOSITIONS:
                n_claims += 1
                if k + 1 < len(words) and words[k + 1] in NOUN_WORDS:
                    n_true += [word, words[k + 1]] == fact["object"]
                    k += 2
                    continue
            elif word in NOUN_WORDS:
                n_claims += 1
                n_true += word == fact["subject"][0]
            elif word in ABSTRACT_WORDS[:2]:
                n_claims += 1
                n_true += word == fact["number"][0]
            elif word in ABSTRACT_WORDS[2:]:
                n_claims += 1
                n_true += word == fact["tense"][0]
            elif word in TIME_WORDS:
                n_claims += 1
                n_true += word == fact["time"][0]
            k += 1
    return n_true / max(n_claims, 1), n_claims / len(descriptions)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_claim_accuracy(claim_accuracy)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Here's an evaluation helper that puts the three metrics together. It returns a dictionary, so that we can log it during training in part 3️⃣.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

facts_test = None  # set in the main cells below


@t.no_grad()
def evaluate_nla(av: ActivationVerbaliser, ar: ActivationReconstructor, with_behaviour: bool = False) -> dict:
    ids = sample_descriptions(av, h_test, greedy=True)
    h_hat = ar(text_features(model, ids)).cpu()
    texts = decode_descriptions(ids)
    claims_true, claims_per = claim_accuracy(texts, facts_test)
    result = dict(fve=fve(acts[test_idx], h_hat), claims_true=claims_true, claims_per_description=claims_per)
    if with_behaviour:
        result["transplant"] = transplant(h_hat)
    return result


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

facts_test = [facts[i] for i in test_idx]
warm_eval = evaluate_nla(av_warm, ar_warm, with_behaviour=True)
print({k: round(v, 3) for k, v in warm_eval.items()})

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:WARM_EVAL -->{'fve': 0.496, 'claims_true': 0.998, 'claims_per_description': 3.15, 'transplant': 0.855}</pre>

</details>

The warm-started NLA reconstructs about half of the variance from its greedy descriptions (the paper reports 0.3-0.4 for its warm-started NLAs, on a much harder problem). 99.8% of
its claims are true. And a transplant of its reconstruction carries the source sentence's verb 85.5% of the time, against 92% for the real activation: its descriptions say the
number and the tense, which is exactly what the verb needs.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Which descriptions reconstruct well?

The warm-started verbaliser says a random selection of true things. That gives us a natural experiment: we can sample many descriptions for each held-out sentence and ask how good
the reconstruction is, depending on *which* facts the description mentions. The cell below samples 8 descriptions per sentence, groups them by the set of facts they mention, and
computes the FVE within each group (using the variance of the whole test set as the denominator, so the numbers are comparable). As controls, we decode **random word sequences** of
the same lengths, and the descriptions of **the wrong sentence**.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

def facts_mentioned(text: str) -> str:
    words = text.split()
    kinds = []
    for word, previous in zip(words, [None] + words[:-1]):
        if word in PREPOSITIONS:
            kinds.append("object")
        elif word in NOUN_WORDS and previous not in PREPOSITIONS:
            kinds.append("subject")
        elif word in ABSTRACT_WORDS:
            kinds.append("number" if word in ABSTRACT_WORDS[:2] else "tense")
        elif word in TIME_WORDS:
            kinds.append("time")
    return " + ".join(kind for kind in FACT_ORDER if kind in kinds)


t.manual_seed(0)
total_var_test = (h_test - h_test.mean(0)).pow(2).sum(-1).mean().item()
errors, groups = [], []
random_errors, wrong_errors = [], []
with t.no_grad():
    for _ in range(8):
        ids = sample_descriptions(av_warm, h_test)
        h_hat = ar_warm(text_features(model, ids))
        errors.append((h_hat - h_test).pow(2).sum(-1).cpu())
        groups += [facts_mentioned(text) for text in decode_descriptions(ids)]
        # controls: random words in the same slots, and the description of a different sentence
        random_ids = t.where(ids == EOS, ids, t.randint_like(ids, 0, EOS))
        random_errors.append((ar_warm(text_features(model, random_ids)) - h_test).pow(2).sum(-1).cpu())
        wrong_errors.append((h_hat.roll(1, dims=0) - h_test).pow(2).sum(-1).cpu())
errors = t.cat(errors)
group_stats = pd.Series(errors.numpy()).groupby(groups).agg(["mean", "size"])
shown = ["subject + time + object", "subject + time", "subject", "time", "object", "number + tense", "tense", "number"]
bars = {name: 1 - group_stats.loc[name, "mean"] / total_var_test for name in shown}
print(group_stats.loc[shown, "size"].to_dict())
bars["control: random words"] = 1 - t.cat(random_errors).mean().item() / total_var_test
bars["control: wrong sentence"] = 1 - t.cat(wrong_errors).mean().item() / total_var_test
utils.plot_bars(
    bars,
    "Warm-started NLA: FVE of sampled descriptions, grouped by which facts they mention (8 samples per sentence)",
    ylabel="FVE",
    ylim=(-1.0, 1.0),
    n_highlight=len(bars) - 2,
    fmt="{:.2f}",
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/warm_fve_by_facts.png" width="900">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret this plot?</summary>

Descriptions that mention the subject, the time word and the object reconstruct best (0.64), followed by subject + time word (0.56) and the subject alone (0.36). Descriptions that
only mention the number and the tense get 0.18, and the number alone 0.06. The ranking matches the description oracles from part 1️⃣ at layer 10 (subject 0.46, time word 0.23,
number + tense 0.18), with somewhat lower values for most groups: the reconstructor has to decode the text, and it was trained on a mixture of all kinds of descriptions.

The controls are strongly negative: random words (-0.44) and the description of the wrong sentence (-0.40) are much worse than predicting the mean, because the reconstructor
confidently reconstructs some *other* activation.

So the warm-started NLA already contains a much better NLA: the one you'd get if the verbaliser always mentioned the subject, the time word and the object. The verbaliser just
doesn't know that it should. Teaching it which facts are worth its words is what end-to-end training is for.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 2️⃣

- An NLA is an autoencoder with a text bottleneck: an activation verbaliser (AV) writes a description, an activation reconstructor (AR) reads it back. Neither half is told what to say; the description is shaped only by what the reconstructor can use.
- Our miniature AV is a 2-layer transformer that reads the activation in place of a first token; our AR is frozen GPT-2 reading the description up to layer 10, plus an affine head.
- We warm-started the AV on a teacher that writes random true facts about the sentence, and trained the AR on the AV's samples.
- The warm-started NLA reconstructs about half the variance (FVE 0.50), says true things (99.8% of claims), and its reconstructions carry the verb in 85.5% of transplants.
- But it mostly says facts in the teacher's order rather than the most informative ones. Which facts it mentions makes a big difference: the subject and time word are worth far more than the number and tense.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 3️⃣ Training through a discrete bottleneck
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Why we can't just backpropagate

If the bottleneck were a vector, we'd train the encoder and decoder together by backpropagating the reconstruction loss through both. With text, the path from the verbaliser's
parameters $\phi$ to the loss goes through a **sample**: $z \sim \text{AV}_\phi(\cdot \mid h)$, a sequence of discrete word choices. The loss $\|\tilde h - \text{AR}(z)\|^2$ is a
function of *which* words were chosen, and an infinitesimal change in $\phi$ doesn't change which words get sampled; it only changes their probabilities. So the gradient of the
sampled loss with respect to $\phi$ is zero almost everywhere (and undefined where a sample flips).

What we actually want to minimise is the *expected* loss, $\mathbb{E}_{z \sim \text{AV}_\phi(\cdot|h)}[\ell(h, z)]$, and that *is* a smooth function of $\phi$, because the
probabilities are. There are two standard ways to estimate its gradient:

1. **The score-function estimator (REINFORCE).** Using $\nabla_\phi p_\phi(z) = p_\phi(z) \nabla_\phi \log p_\phi(z)$,
$$
\nabla_\phi \, \mathbb{E}_{z \sim p_\phi}[R(z)] = \mathbb{E}_{z \sim p_\phi}\big[ R(z)\, \nabla_\phi \log p_\phi(z) \big] .
$$
Sample some descriptions, and push up the log-probability of each in proportion to its reward. This is unbiased, needs nothing from the reconstructor except a number per sample (so
the reconstructor can be any black box, like a full language model reading text), and has high variance. It's what the NLA paper uses.
2. **Relaxations (Gumbel-softmax, straight-through).** Replace the discrete sample by something differentiable, and backpropagate through the reconstructor into the verbaliser. Lower variance, but biased, and it requires the reconstructor to accept "soft" words. We'll try this at the end of the part.

### Variance reduction: a baseline per activation

The raw REINFORCE estimator is very noisy, and here the reward also has a large activation-dependent offset: some activations are just easier to reconstruct than others, whatever
the description says. Since $\mathbb{E}_{z}[\nabla_\phi \log p_\phi(z)] = 0$, we can subtract any baseline $b(h)$ that doesn't depend on $z$ without changing the expectation:
$$
\nabla_\phi \approx \frac{1}{G}\sum_{g=1}^G \big(R(h, z_g) - b(h)\big)\, \nabla_\phi \log p_\phi(z_g \mid h) .
$$
**GRPO** (group relative policy optimisation, used by the paper) samples a *group* of $G$ descriptions for each activation and uses the group's mean reward as the baseline, also
dividing by the group's standard deviation:
$$
A_g = \frac{R_g - \text{mean}(R_{1..G})}{\text{std}(R_{1..G}) + \epsilon} .
$$
Each description is then judged only against other descriptions *of the same activation*: "was this a better or worse thing to say about $h$ than the alternatives?"

### The reward and the KL penalty

Following the paper, the reward is $R(h, z) = -\log \|\tilde h - \text{AR}(z)\|^2$. The log makes the reward care about *relative* improvements: halving the error is worth the same
whether the error was large or small.

The full verbaliser objective (to be minimised) is
$$
\mathcal{L}_{AV} = -\frac{1}{BG}\sum_{b, g} A_{b,g} \log p_\phi(z_{b,g} \mid h_b) \;+\; \beta \, \frac{1}{BG}\sum_{b,g} D_{KL}\big(p_\phi(\cdot \mid h_b) \,\|\,
p_{\text{ref}}(\cdot \mid h_b)\big),
$$
where $p_{\text{ref}}$ is the frozen warm-started verbaliser. The KL term is the same device as in RLHF ([2.4]): it stops the policy wandering too far from a model that writes
sensible text. The paper reports that it preserves the fluency of explanations. We compute the KL exactly at each slot of the sampled description (our vocabulary only has 47
words), and sum over the slots.

Meanwhile the reconstructor takes one regression step per training step on the same samples. As in the paper, the two updates are not coupled: the reconstructor's loss doesn't
backpropagate into the verbaliser.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - group-normalised advantages

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 5 minutes on this exercise.
> ```

`rewards` has shape `(batch * group_size,)`, with the `group_size` samples for each activation stored consecutively (we'll build batches with `repeat_interleave`). Return the
advantages $A$, same shape. Use the default (unbiased) `std`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def group_advantages(rewards: Float[Tensor, " n"], group_size: int, eps: float = 1e-6) -> Float[Tensor, " n"]:
    """Normalises each consecutive group of rewards to zero mean and unit standard deviation."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    grouped = rewards.view(-1, group_size)
    advantages = (grouped - grouped.mean(dim=1, keepdim=True)) / (grouped.std(dim=1, keepdim=True) + eps)
    return advantages.view(-1)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_group_advantages(group_advantages)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - KL penalty and policy-gradient loss

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement:

- `kl_to_reference(logprobs, ref_logprobs, mask)`: the per-description KL divergence $\sum_{i} \text{mask}_i \sum_v p_\phi(v)\,(\log p_\phi(v) - \log p_{\text{ref}}(v))$, where both inputs are full log-probability tensors of shape `(batch, slots, n_words)` as returned by `description_logprobs`. Return shape `(batch,)`.
- `policy_gradient_loss(token_logprobs, mask, advantages)`: the scalar $-\frac{1}{N}\sum_n A_n \sum_i \text{mask}_{n,i} \log p_\phi(z_{n,i})$.

Note that the advantages must be treated as constants: if they had been computed from something with gradients, you'd need to detach them. (Ours come from the rewards, which we'll
detach when we compute them.)
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def kl_to_reference(
    logprobs: Float[Tensor, "batch slots n_words"],
    ref_logprobs: Float[Tensor, "batch slots n_words"],
    mask: Bool[Tensor, "batch slots"],
) -> Float[Tensor, " batch"]:
    """Exact KL(policy || reference) summed over the unmasked slots of each description."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    per_slot = (logprobs.exp() * (logprobs - ref_logprobs)).sum(dim=-1)
    return (per_slot * mask).sum(dim=-1)
    # END SOLUTION


def policy_gradient_loss(
    token_logprobs: Float[Tensor, "batch slots"], mask: Bool[Tensor, "batch slots"], advantages: Float[Tensor, " batch"]
) -> Float[Tensor, ""]:
    """REINFORCE loss: minus the advantage-weighted log-probability of each description, averaged."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    return -(advantages * (token_logprobs * mask).sum(dim=-1)).mean()
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_kl_to_reference(kl_to_reference)
    tests.test_policy_gradient_loss(policy_gradient_loss)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - the NLA training loop

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 20-25 minutes on this exercise.
> ```

Put it together. Implement `train_nla`. Each step should:

1. Sample `batch_size` training activations and repeat each `group_size` times (`repeat_interleave`), so the samples for one activation are consecutive.
2. Sample one description per row from the verbaliser, and compute the reconstructions.
3. **Reconstructor step**: the squared error per row (summed over $d_{model}$), averaged; backward; step `ar_optimizer`.
4. **Verbaliser step** (only if `train_av` is true): rewards $-\log(\text{error} + 10^{-6})$ (detached), group advantages, the policy-gradient loss, plus `beta` times the mean KL to `av_ref` if `beta > 0` (compute the reference log-probabilities under `t.no_grad()`); backward; step the verbaliser's optimizer (Adam with `lr_av`, created inside the function).
5. Evaluate with `evaluate_nla(av, ar)` at the *start* of every `eval_every`-th step (so step 0 records the starting point), and once more after the last step (recorded as step `n_steps`), appending `{"step": step, **evaluate_nla(av, ar)}` to a history list.

Return the history. The `train_av=False` option gives us a control: the reconstructor keeps learning, but the verbaliser is frozen.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def train_nla(
    av: ActivationVerbaliser,
    ar: ActivationReconstructor,
    ar_optimizer: t.optim.Optimizer,
    h_train: Float[Tensor, "n d_act"],
    n_steps: int = 600,
    batch_size: int = 64,
    group_size: int = 8,
    beta: float = 0.05,
    av_ref: ActivationVerbaliser | None = None,
    lr_av: float = 3e-4,
    train_av: bool = True,
    eval_every: int = 50,
    seed: int = 0,
) -> list[dict]:
    """End-to-end NLA training: GRPO-style policy gradient for the verbaliser, regression for the reconstructor."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    generator = t.Generator().manual_seed(seed)
    av_optimizer = t.optim.Adam(av.parameters(), lr=lr_av)
    history = []
    for step in tqdm(range(n_steps), desc="NLA training"):
        if step % eval_every == 0:
            history.append({"step": step, **evaluate_nla(av, ar)})
        idx = t.randint(0, len(h_train), (batch_size,), generator=generator)
        h = h_train[idx].repeat_interleave(group_size, dim=0)
        ids = sample_descriptions(av, h)
        errors = (ar(text_features(model, ids)) - h).pow(2).sum(dim=-1)

        # Reconstructor: one regression step on the sampled descriptions
        ar_optimizer.zero_grad()
        errors.mean().backward()
        ar_optimizer.step()

        # Verbaliser: policy gradient with group baseline, plus KL to the reference
        if train_av:
            rewards = -t.log(errors.detach() + 1e-6)
            advantages = group_advantages(rewards, group_size)
            logprobs, token_logprobs = description_logprobs(av, h, ids)
            mask = description_mask(ids)
            loss = policy_gradient_loss(token_logprobs, mask, advantages)
            if beta > 0:
                with t.no_grad():
                    ref_logprobs, _ = description_logprobs(av_ref, h, ids)
                loss = loss + beta * kl_to_reference(logprobs, ref_logprobs, mask).mean()
            av_optimizer.zero_grad()
            loss.backward()
            av_optimizer.step()
    history.append({"step": n_steps, **evaluate_nla(av, ar)})
    return history
    # END SOLUTION


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

The structure of each step is:

```python
idx = t.randint(0, len(h_train), (batch_size,), generator=generator)
h = h_train[idx].repeat_interleave(group_size, dim=0)
ids = sample_descriptions(av, h)
errors = (ar(text_features(model, ids)) - h).pow(2).sum(dim=-1)
# ... reconstructor step on errors.mean() ...
# ... verbaliser step using rewards = -t.log(errors.detach() + 1e-6) ...
```

`text_features` runs under `no_grad` (GPT-2 is frozen), so `errors` only has gradients with respect to the reconstructor's head. That's exactly what we want: the verbaliser gets
its learning signal only through the rewards.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
We'll train several NLAs that start from the same place, so here's a helper which makes an independent copy of a reconstructor *together with* its optimizer state:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def clone_reconstructor(
    ar: ActivationReconstructor, optimizer: t.optim.Optimizer
) -> tuple[ActivationReconstructor, t.optim.Optimizer]:
    ar_copy = copy.deepcopy(ar)
    optimizer_copy = type(optimizer)(ar_copy.parameters(), **optimizer.defaults)
    optimizer_copy.load_state_dict(optimizer.state_dict())
    return ar_copy, optimizer_copy


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Now let's train. This is the main run: start from the warm-started verbaliser and reconstructor, with $\beta = 0.05$. It takes about a minute on a GPU.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

t.manual_seed(0)
av_ref = copy.deepcopy(av_warm).requires_grad_(False)
av_rl = copy.deepcopy(av_warm)
ar_rl, ar_rl_opt = clone_reconstructor(ar_warm, ar_warm_opt)
history_rl = train_nla(av_rl, ar_rl, ar_rl_opt, h_train, av_ref=av_ref)
print({k: round(v, 3) for k, v in history_rl[-1].items()})

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:RL_FINAL -->{'step': 600, 'fve': 0.881, 'claims_true': 0.977, 'claims_per_description': 3.0}</pre>

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
What does it say now? Here are the same eight held-out sentences as before, with the warm-started verbaliser's greedy description next to the trained one's:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

with t.no_grad():
    rl_texts = decode_descriptions(sample_descriptions(av_rl, acts_unit[show].to(device), greedy=True))
display(pd.DataFrame({"sentence": df.sentence[show.tolist()].values, "warm start": greedy_texts, "after RL": rl_texts}))

rl_test_texts = decode_descriptions(sample_descriptions(av_rl, h_test, greedy=True))
print("Most common fact combinations in held-out descriptions after RL:")
print(Counter(facts_mentioned(text) for text in rl_test_texts).most_common(4))
wrong = [i for i, (text, fact) in enumerate(zip(rl_test_texts, facts_test)) if claim_accuracy([text], [fact])[0] < 1]
print(f"\n{len(wrong)} of {len(rl_test_texts)} descriptions contain a false claim, e.g.:")
for i in wrong[:5]:
    print(f"  {df.sentence[int(test_idx[i])]!r:<50} -> {rl_test_texts[i]!r}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:RL_SAMPLES -->                                       sentence                warm start                      after RL
0    Currently, the tables with the keys really     tables plural present    tables currently with keys
1   Earlier, the house behind the window really       house singular past   house earlier behind window
2        Earlier, the boys with the tree really          boys plural past        boys earlier with tree
3            Now, the cat with the chair really      cat singular present            cat now with chair
4       Earlier, the house with the dogs really       house singular past       house earlier with dogs
5  Previously, the tables with the birds really        tables plural past  tables previously with birds
6    Yesterday, the tree near the tables really        tree singular past    tree yesterday near tables
7        Now, the window behind the door really  singular now behind door        window now behind door
Most common fact combinations in held-out descriptions after RL:
[('subject + time + object', 1000)]

65 of 1000 descriptions contain a false claim, e.g.:
  'Previously, the bird behind the boys really'      -&gt; 'cat previously behind boys'
  'Now, the tables near the dogs really'             -&gt; 'tables now near cats'
  'Previously, the boys with the girl really'        -&gt; 'girls previously with boy'
  'Yesterday, the books behind the cat really'       -&gt; 'books yesterday behind dog'
  'Currently, the chair beside the cat really'       -&gt; 'chair currently beside dog'</pre>

</details>

Every one of the 1,000 held-out descriptions now has the same form: subject, time word, preposition, object. The verbaliser has stopped using `singular`/`plural` and
`present`/`past`. They're redundant once the subject's noun form and the time word have been given (`tables` is plural, `currently` is present tense), and the words are better
spent on the object, even though the object explains only about 7% of the variance at layer 10. Nobody told it any of this. It's what the reconstruction reward selected from among
the teacher's true descriptions.

It isn't perfect: 65 of the 1,000 descriptions contain a false claim. Look at what kind of false claim: `bird` → `cat`, `dogs` → `cats`, `boys` → `girls`, `cat` → `dog`. The wrong
nouns are *similar* to the right ones, and the number and the time word are still correct. We'll come back to this in part 4️⃣.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Controls: is the verbaliser learning, and does it need the warm start?

The FVE went up a lot. But the reconstructor was also training the whole time, so maybe it just got better at reading the same descriptions. And maybe the warm start is unnecessary
decoration. Two controls:

- **Reconstructor only**: the same loop with `train_av=False`. The verbaliser stays at its warm start, the reconstructor keeps training on its samples.
- **Cold start**: a freshly initialised verbaliser and reconstructor, trained with the same loop (no KL penalty, since there's no sensible reference to stay close to).

We train each for 300 steps (the curves have stabilised by then).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

t.manual_seed(0)
ar_only, ar_only_opt = clone_reconstructor(ar_warm, ar_warm_opt)
history_ar_only = train_nla(copy.deepcopy(av_warm), ar_only, ar_only_opt, h_train, n_steps=300, train_av=False)

t.manual_seed(0)
av_cold = ActivationVerbaliser().to(device)
ar_cold = ActivationReconstructor().to(device)
ar_cold_opt = t.optim.Adam(ar_cold.parameters(), lr=1e-3)
history_cold = train_nla(av_cold, ar_cold, ar_cold_opt, h_train, n_steps=300, beta=0.0)
with t.no_grad():
    cold_texts = decode_descriptions(sample_descriptions(av_cold, h_test, greedy=True))
print(f"Cold start: {len(set(cold_texts))} distinct descriptions on the test set; most common: {Counter(cold_texts).most_common(2)}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:COLD -->Cold start: 1 distinct descriptions on the test set; most common: [('plural previously plural previously', 1000)]</pre>

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## A differentiable alternative: Gumbel-softmax with a straight-through estimator

The other classic way through a discrete bottleneck is to make it (approximately) differentiable. The **Gumbel-max trick** says that if you add independent Gumbel noise $g_v = -\log(-\log u_v)$, $u_v \sim U(0,1)$, to the logits, the argmax is an exact sample from the softmax distribution. The **Gumbel-softmax** ([Jang et al., 2016](https://arxiv.org/abs/1611.01144); [Maddison et al., 2016](https://arxiv.org/abs/1611.00712)) replaces the argmax by a softmax with temperature $\tau$, which gives a differentiable "soft one-hot" vector. The **straight-through** version uses the hard one-hot in the forward pass (so the reconstructor sees real words), but the soft one-hot's gradient in the backward pass:

$$
y = \text{onehot}(\arg\max_v (\ell_v + g_v)) \;-\; \text{sg}\big[y_{soft}\big] \;+\; y_{soft}, \qquad y_{soft} = \text{softmax}((\ell + g)/\tau),
$$

where $\text{sg}$ is stop-gradient (`.detach()`). The forward value is exactly the hard one-hot; the gradient is that of $y_{soft}$.

To use it, the reconstructor has to accept one-hot (or soft) words: we feed GPT-2 the embedding $y^\top W_E[\text{words}]$ instead of a token. The gradient of the reconstruction
loss then flows through GPT-2 into the verbaliser's logits. Note the difference from REINFORCE: the verbaliser now learns from *how the reconstructor's output would change if the
input embedding moved a little*, a local, first-order signal about embeddings, rather than from *how good the sampled description actually was*.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `straight_through_gumbel`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement it from scratch (rather than with `F.gumbel_softmax`, which does the same thing). Sample the uniform noise with `t.rand_like`, and clamp it away from 0 and 1 (e.g. to
`[1e-9, 1 - 1e-9]`) before taking logs.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def straight_through_gumbel(logits: Float[Tensor, "batch n"], tau: float = 1.0) -> Float[Tensor, "batch n"]:
    """Hard one-hot sample in the forward pass; gradient of the Gumbel-softmax in the backward pass."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    u = t.rand_like(logits).clamp(1e-9, 1 - 1e-9)
    gumbel = -t.log(-t.log(u))
    soft = ((logits + gumbel) / tau).softmax(dim=-1)
    hard = F.one_hot(soft.argmax(dim=-1), logits.shape[-1]).to(soft.dtype)
    return hard - soft.detach() + soft
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_straight_through_gumbel(straight_through_gumbel)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
The rest is plumbing, so we've written it for you. `gumbel_sample_descriptions` generates a description word by word like `sample_descriptions`, but returns straight-through
one-hot vectors (of shape `(batch, slots, n_words)`) and feeds each chosen word back in as an embedding, so the whole thing is differentiable; once a description has produced EOS,
later slots are fixed to EOS. `soft_text_features` is `text_features` for one-hot inputs: it swaps GPT-2's token embeddings for $y^\top W_E[\text{words}]$ with a hook.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def gumbel_sample_descriptions(
    av: ActivationVerbaliser, h: Float[Tensor, "batch d_act"], tau: float = 1.0
) -> Float[Tensor, "batch slots n_words"]:
    n_words = av.unembed.out_features
    eos = F.one_hot(t.tensor(EOS, device=h.device), n_words).float()
    done = t.zeros(h.shape[0], 1, device=h.device)
    words = []
    for _ in range(av.max_words):
        prev_embeds = t.stack([y @ av.word_embed.weight for y in words], dim=1) if words else h.new_zeros(h.shape[0], 0, av.word_embed.embedding_dim)
        y = straight_through_gumbel(av(h, prev_embeds=prev_embeds)[:, -1], tau)
        y = done * eos + (1 - done) * y  # after an EOS, every slot is EOS
        done = t.maximum(done, y[:, EOS : EOS + 1].detach())
        words.append(y)
    words.append(eos.expand(h.shape[0], -1))
    return t.stack(words, dim=1)


def soft_text_features(
    model: HookedTransformer, word_onehots: Float[Tensor, "batch slots n_words"], layer: int = LAYER
) -> Float[Tensor, "batch d_model"]:
    soft_embeds = word_onehots @ model.W_E[WORD_TOKEN_IDS.to(word_onehots.device)]
    placeholder = t.full((word_onehots.shape[0], word_onehots.shape[1] + 1), model.tokenizer.bos_token_id)
    store = {}

    def embed_hook(embed: Tensor, hook) -> Tensor:
        return t.cat([embed[:, :1], soft_embeds], dim=1)  # keep BOS, replace the rest

    def grab_hook(resid: Tensor, hook) -> None:
        store["resid"] = resid

    model.run_with_hooks(
        placeholder.to(word_onehots.device),
        fwd_hooks=[("hook_embed", embed_hook), (f"blocks.{layer}.hook_resid_pre", grab_hook)],
        stop_at_layer=layer + 1,
    )
    return store["resid"][:, -1]


# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
And here's a training loop. It's simpler than `train_nla`: one loss, the reconstruction error, backpropagated into both the reconstructor head and (through GPT-2 and the
straight-through samples) the verbaliser. We start from the same warm-started verbaliser and reconstructor as the main REINFORCE run, and use no KL penalty.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def train_nla_gumbel(
    av: ActivationVerbaliser,
    ar: ActivationReconstructor,
    ar_optimizer: t.optim.Optimizer,
    h_train: Float[Tensor, "n d_act"],
    n_steps: int = 300,
    batch_size: int = 256,
    tau: float = 1.0,
    lr_av: float = 3e-4,
    eval_every: int = 50,
    seed: int = 0,
) -> list[dict]:
    generator = t.Generator().manual_seed(seed)
    av_optimizer = t.optim.Adam(av.parameters(), lr=lr_av)
    history = []
    for step in tqdm(range(n_steps), desc="Gumbel NLA training"):
        if step % eval_every == 0:
            history.append({"step": step, **evaluate_nla(av, ar)})
        h = h_train[t.randint(0, len(h_train), (batch_size,), generator=generator)]
        word_onehots = gumbel_sample_descriptions(av, h, tau)
        loss = (ar(soft_text_features(model, word_onehots)) - h).pow(2).sum(dim=-1).mean()
        av_optimizer.zero_grad()
        ar_optimizer.zero_grad()
        loss.backward()
        av_optimizer.step()
        ar_optimizer.step()
    history.append({"step": n_steps, **evaluate_nla(av, ar)})
    return history


# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

t.manual_seed(0)
av_gumbel = copy.deepcopy(av_warm)
ar_gumbel, ar_gumbel_opt = clone_reconstructor(ar_warm, ar_warm_opt)
history_gumbel = train_nla_gumbel(av_gumbel, ar_gumbel, ar_gumbel_opt, h_train)
with t.no_grad():
    gumbel_texts = decode_descriptions(sample_descriptions(av_gumbel, acts_unit[show].to(device), greedy=True))
display(pd.DataFrame({"sentence": df.sentence[show.tolist()].values, "REINFORCE": rl_texts, "Gumbel straight-through": gumbel_texts}))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:GUMBEL -->                                       sentence                     REINFORCE  Gumbel straight-through
0    Currently, the tables with the keys really    tables currently with keys  books books books books
1   Earlier, the house behind the window really   house earlier behind window  books books books books
2        Earlier, the boys with the tree really        boys earlier with tree  books books books books
3            Now, the cat with the chair really            cat now with chair  books books books books
4       Earlier, the house with the dogs really       house earlier with dogs  books books books books
5  Previously, the tables with the birds really  tables previously with birds  books books books books
6    Yesterday, the tree near the tables really    tree yesterday near tables  books books books books
7        Now, the window behind the door really        window now behind door  books books books books</pre>

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Now let's put all four runs on the same axes, showing both what the reconstruction objective rewards (FVE) and what it doesn't (whether the descriptions are true).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

runs = {
    "REINFORCE from warm start (β=0.05)": history_rl,
    "control: reconstructor only": history_ar_only,
    "control: REINFORCE from cold start": history_cold,
    "Gumbel straight-through from warm start": history_gumbel,
}
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(13, 4.5))
for (name, history), style in zip(runs.items(), ["-o", "--s", ":^", "-.v"]):
    steps = [entry["step"] for entry in history]
    ax0.plot(steps, [entry["fve"] for entry in history], style, label=name, markersize=4)
    ax1.plot(steps, [entry["claims_true"] for entry in history], style, label=name, markersize=4)
ax0.axhline(oracle_points["subject + time word"][0], color="gray", ls="--", lw=1, label="oracle: subject + time word")
ax0.set_ylim(-0.3, 1)
ax0.set_ylabel("held-out FVE (greedy descriptions)")
ax0.set_title("Reconstruction: what the objective rewards", fontsize=10)
ax1.set_ylim(0, 1.05)
ax1.set_ylabel("fraction of claims that are true")
ax1.set_title("Truthfulness: what the objective doesn't directly reward", fontsize=10)
for ax in (ax0, ax1):
    ax.set_xlabel("training step")
    ax.grid(alpha=0.3)
handles, names = ax0.get_legend_handles_labels()
fig.legend(handles, names, loc="lower center", ncol=3, fontsize=9)
plt.tight_layout(rect=(0, 0.12, 1, 1))
plt.show()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/training_runs.png" width="1000">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret these curves?</summary>

**REINFORCE from the warm start (blue)** climbs from 0.50 to 0.77 in 50 steps, and reaches 0.88 at step 600, well above the `subject + time word` oracle (0.72). Throughout, 96-98%
of its claims are true.

**Reconstructor only (orange)** stays around 0.5. The reconstructor had already learned to read the warm-started verbaliser's descriptions about as well as it could. The
improvement in the blue curve comes from the verbaliser changing *what it says*.

**Cold start (green)** never gets going. Its first point, at initialisation, is far below zero (off the bottom of the plot). Within 50 steps the verbaliser says the same thing for
every input (`plural previously plural previously`), and FVE sits just below zero: the reconstructor has learned to output roughly the mean activation. Its claim accuracy is 34%,
because `plural` is true about half the time and `previously` about a sixth of the time. This is a chicken-and-egg problem. At initialisation the descriptions are random, so they
carry no information about the activation, and the best the reconstructor can do is predict the mean. Then, for a given activation, no description is better than any other except
by noise, and the policy gradient has nothing useful to follow. Worse, the reconstructor is a little better at decoding descriptions it has seen often, so frequently sampled
descriptions earn slightly higher rewards, and the policy collapses onto one of them. (In quick experiments while writing this section, an entropy bonus and a slower reconstructor
didn't prevent the collapse either. See the bonus section.)

**Gumbel straight-through from the warm start (red)** starts where REINFORCE does, and then falls apart: by step 150 it writes `books books books books` for every sentence (its FVE
briefly drops off the bottom of the plot there), from step 200 its FVE sits just below zero, and 97% of its claims are false. The gradient it follows is the reconstructor's
sensitivity to its *input embeddings*, linearised at the current words. Through ten layers of GPT-2, that's a poor guide to what happens when you swap one word for a completely
different one, and nothing in it checks whether a sampled description actually works, let alone whether it's true. We haven't tuned it (temperature schedules, learning rates and a
KL penalty could all help), so don't read this as "Gumbel-softmax can't work here". Read it as: the straight-through gradient optimises something different from the sampled reward,
and with a reconstructor that co-adapts to whatever it's given, it's easy to slide into a degenerate solution.

So what does end-to-end training through a discrete bottleneck need? In our hands: a gradient estimator that scores the descriptions the verbaliser actually samples (REINFORCE with
a per-activation baseline), and a warm start that makes the descriptions informative and readable from step 0.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Does the KL penalty matter here?

The KL penalty is there to stop the verbaliser drifting away from the language it was warm-started on. Let's check whether it's doing anything in our setting, by repeating the main
run with $\beta = 0$, and measuring how far each trained verbaliser has moved from the warm start (the mean KL from the warm-started verbaliser, on held-out activations).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

t.manual_seed(0)
av_nokl = copy.deepcopy(av_warm)
ar_nokl, ar_nokl_opt = clone_reconstructor(ar_warm, ar_warm_opt)
history_nokl = train_nla(av_nokl, ar_nokl, ar_nokl_opt, h_train, beta=0.0, eval_every=600)


@t.no_grad()
def mean_kl_to_warm_start(av: ActivationVerbaliser) -> float:
    ids = sample_descriptions(av, h_test)
    logprobs, _ = description_logprobs(av, h_test, ids)
    ref_logprobs, _ = description_logprobs(av_ref, h_test, ids)
    return kl_to_reference(logprobs, ref_logprobs, description_mask(ids)).mean().item()


for name, av_, history in [("beta = 0.05", av_rl, history_rl), ("beta = 0", av_nokl, history_nokl), ("Gumbel (no KL)", av_gumbel, history_gumbel)]:
    final = history[-1]
    with t.no_grad():
        examples = decode_descriptions(sample_descriptions(av_, acts_unit[show[:3]].to(device), greedy=True))
    print(f"{name:<15} FVE {final['fve']:.3f} | claims true {final['claims_true']:.3f} | KL from warm start {mean_kl_to_warm_start(av_):5.2f} nats | {examples}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:KL -->beta = 0.05     FVE 0.881 | claims true 0.977 | KL from warm start  3.78 nats | ['tables currently with keys', 'house earlier behind window', 'boys earlier with tree']
beta = 0        FVE 0.879 | claims true 0.970 | KL from warm start 12.77 nats | ['tables currently with keys', 'house earlier behind window', 'boys earlier with bird']
Gumbel (no KL)  FVE -0.028 | claims true 0.031 | KL from warm start 66.87 nats | ['books books books books', 'books books books books', 'books books books books']</pre>

</details>

With or without the KL penalty, REINFORCE reaches the same FVE (0.881 vs 0.879) and the same truthfulness (97.7% vs 97.0% of claims true), and the greedy descriptions are almost
identical. The penalty does keep the verbaliser much closer to its warm start (3.8 nats vs 12.8 nats). Since the greedy descriptions barely differ, that extra distance must be in
the rest of the distribution: without the penalty, the verbaliser moves its probability mass further from where the warm start put it. Whatever that drift is, it's harmless here.

So in our toy, the KL penalty isn't what keeps the descriptions honest. What does? REINFORCE only ever reinforces descriptions that the verbaliser actually *sampled* and that the
reconstructor could actually *decode*, and starting from the warm start, those are true descriptions in the teacher's language. The Gumbel run, which follows the reconstructor's
gradient rather than the rewards of sampled descriptions, has no such protection, and ended up 67 nats from the warm start.

Our verbaliser's language is also unusually easy to stay honest in, because its honest words can already say almost everything worth saying. At scale, an activation contains far
more than any explanation can say, and the paper reports that the KL penalty is what keeps explanations fluent through training. The bonus section suggests a way to put our
verbaliser under more pressure.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 3️⃣

- You can't backpropagate through sampled text. REINFORCE estimates the gradient of the expected reward from samples; a per-activation group baseline (GRPO) makes it usable; the reward is $-\log$ of the reconstruction error; a KL penalty keeps the verbaliser near its warm start.
- From the warm start, end-to-end training raised FVE from 0.50 to 0.88 in 600 steps, while keeping 97.7% of claims true. The verbaliser learned *what* to say: every held-out description became subject, time word, preposition and object, dropping the redundant number and tense words.
- Training only the reconstructor left FVE around 0.5: the gain comes from the verbaliser changing what it says.
- From a cold start, REINFORCE collapsed to a single description for every input. The reconstructor can't learn to read text that carries no information, and the verbaliser gets no useful reward for text the reconstructor can't read. The warm start breaks this chicken-and-egg problem.
- The Gumbel straight-through relaxation, from the same warm start, collapsed too, into a description that is false 97% of the time. Following the reconstructor's local gradient is not the same as rewarding descriptions that work.
- In this toy, the KL penalty made little difference to FVE or truthfulness.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 4️⃣ What can text carry?
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Our NLA reconstructs 88% of the variance of a held-out activation from four words. Is that good? A number like this only means something next to baselines. In this part we'll
compare the text bottleneck against:

- **continuous bottlenecks of the same size**: PCA, keeping $k$ dimensions, against an NLA allowed $k$ words;
- **the input itself**, as a bag of words and with word order;
- **damaged descriptions**: the NLA's own words, shuffled.

and for each one we'll measure both FVE and transplant accuracy.

## A budget of words versus a budget of dimensions

PCA is the optimal *linear* bottleneck: for a given $k$, projecting onto the top $k$ principal components and back gives the smallest possible squared error of any rank-$k$ linear
map. An NLA with $k$ words is a very different kind of bottleneck: each word is one of 47 symbols (so at most $\log_2 47 \approx 5.6$ bits), but the decoder that reads them is a
nonlinear language model.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `pca_reconstruction`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement `pca_reconstruction`. Normalise both sets of activations to unit norm, compute the mean $\mu$ and the top $k$ principal directions $V_k$ (the first $k$ right singular
vectors of the centred training matrix, from `t.linalg.svd(..., full_matrices=False)`) **on the training set only**, and return the reconstructions $\mu + (\tilde h - \mu) V_k
V_k^\top$ of the test activations.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def pca_reconstruction(
    train_h: Float[Tensor, "n_train d"], test_h: Float[Tensor, "n_test d"], k: int
) -> Float[Tensor, "n_test d"]:
    """Projects the unit-normalised test activations onto the top-k principal components of the training set."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    train_unit, test_unit = F.normalize(train_h.float(), dim=-1), F.normalize(test_h.float(), dim=-1)
    mean = train_unit.mean(dim=0)
    _, _, Vh = t.linalg.svd(train_unit - mean, full_matrices=False)
    V_k = Vh[:k].T
    return mean + (test_unit - mean) @ V_k @ V_k.T
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_pca_reconstruction(pca_reconstruction)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Now we need NLAs with smaller budgets. We train two more, allowed **one** and **two** words, with exactly the same recipe as before (warm start on teacher descriptions of at most
$k$ words, then reconstructor training, then 400 steps of REINFORCE). This takes about two minutes.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

def train_full_nla(max_words: int, rl_steps: int = 400, seed: int = 0) -> tuple[ActivationVerbaliser, ActivationReconstructor]:
    t.manual_seed(seed)
    av = ActivationVerbaliser(max_words=max_words).to(device)
    train_verbaliser(av, h_train, facts_train, seed=seed)
    ar = ActivationReconstructor().to(device)
    ar_opt = t.optim.Adam(ar.parameters(), lr=1e-3)
    train_reconstructor(ar, ar_opt, av, h_train, seed=seed)
    train_nla(av, ar, ar_opt, h_train, n_steps=rl_steps, av_ref=copy.deepcopy(av).requires_grad_(False), eval_every=rl_steps, seed=seed)
    return av, ar


nlas = {1: train_full_nla(1), 2: train_full_nla(2), 4: (av_rl, ar_rl)}
nla_results = {k: evaluate_nla(av, ar, with_behaviour=True) for k, (av, ar) in nlas.items()}
pca_results = {}
for k in [1, 2, 4, 8, 16]:
    h_hat = pca_reconstruction(acts[train_idx], acts[test_idx], k)
    pca_results[k] = dict(fve=fve(acts[test_idx], h_hat), transplant=transplant(h_hat))
for k, (av, _) in nlas.items():
    with t.no_grad():
        texts = decode_descriptions(sample_descriptions(av, acts_unit[show[:3]].to(device), greedy=True))
    print(f"{k} word(s): FVE {nla_results[k]['fve']:.3f}, transplant {nla_results[k]['transplant']:.3f}, claims true {nla_results[k]['claims_true']:.3f}, e.g. {texts}")
for k, result in pca_results.items():
    print(f"PCA {k:>2} dims: FVE {result['fve']:.3f}, transplant {result['transplant']:.3f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:BUDGET -->1 word(s): FVE 0.510, transplant 0.508, claims true 0.835, e.g. ['table', 'house', 'earlier']
2 word(s): FVE 0.718, transplant 0.920, claims true 0.889, e.g. ['tables currently', 'house earlier', 'girls earlier']
4 word(s): FVE 0.881, transplant 0.864, claims true 0.977, e.g. ['tables currently with keys', 'house earlier behind window', 'boys earlier with tree']
PCA  1 dims: FVE 0.213, transplant 0.060
PCA  2 dims: FVE 0.334, transplant 0.075
PCA  4 dims: FVE 0.525, transplant 0.817
PCA  8 dims: FVE 0.711, transplant 0.891
PCA 16 dims: FVE 0.845, transplant 0.898</pre>

</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

sizes = [1, 2, 4, 8, 16]
utils.plot_budget_sweep(
    sizes,
    fve_curves={
        "NLA (k words)": [nla_results[k]["fve"] for k in [1, 2, 4]],
        "PCA (k dimensions)": [pca_results[k]["fve"] for k in sizes],
    },
    behaviour_curves={
        "NLA (k words)": [nla_results[k]["transplant"] for k in [1, 2, 4]],
        "PCA (k dimensions)": [pca_results[k]["transplant"] for k in sizes],
        "A's own activation": [transplant_controls["A's own activation"]] * len(sizes),
    },
    title="Text bottleneck vs linear bottleneck of the same size (held-out sentences)",
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/budget_sweep.png" width="1000">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret these plots?</summary>

**Variance (left).** Word for dimension, the text bottleneck keeps far more variance than PCA: one word reconstructs 51% (PCA with one dimension: 21%), two words 72% (about the
same as PCA with *eight* dimensions), four words 88% (more than PCA with sixteen). That's because the activation's variance is organised around categorical facts: 32 subject forms,
6 time words, and so on. A single word can pick out one of 32 subjects exactly; a single linear dimension can only place them along a line. Our reconstructor is also nonlinear
(it's GPT-2), while PCA is the best *linear* bottleneck, so this isn't an argument that words beat vectors in general; a nonlinear autoencoder with a few continuous dimensions
would do better still. It's an argument that natural-language words are an efficient code *for the kind of structure this activation has*.

**Behaviour (right).** The curves look quite different here. PCA with one or two dimensions scores *below* chance (6-8%): its reconstructions are close to the mean, so GPT-2
follows B's context, like the mean-activation control. PCA jumps to 82% at four dimensions, once it captures the directions that encode the verb. The one-word NLA is stuck at 51%:
it says the subject for 684 of the 1,000 held-out sentences and the time word for the other 316 (you'll see these counts in the next cell), so each description *states* either the
number or the tense, not both. (51% is more than the one-in-three that the `subject` and `time word` oracles got in part 1️⃣. Can you work out where the extra comes from? Hint:
look at which sentences get a time word rather than a noun.) Two words are enough for the NLA to match the real activation (92%). And four words do slightly *worse* on behaviour
(86%) than two, despite explaining more variance. Adding the object to the description makes the reconstruction more accurate overall, but slightly less effective at carrying the
verb; we don't have a full explanation for that small drop, and it's a nice illustration that the two axes really are separate.

The one-word NLA is the cleanest example. A single word *could* carry everything the verb needs, if the vocabulary had one word for each number/tense combination (ours doesn't, but
the two-word `number + tense` oracle got 99.5%). Even then, the FVE objective would prefer the subject, because that's where the variance is.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### What do the NLAs get wrong?

The budget runs reported claim accuracies well below the 4-word NLA's 97.7%. Let's look at which claims are false. For each NLA's held-out descriptions, the cell below finds the
subject claims (a noun not after a preposition) and checks the *number* and the *noun* separately. It also counts which facts each NLA chose to mention.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

LEMMA = {form: pair[0] for pair in NOUNS for form in pair}

for name, av in [("warm start", av_warm), ("1 word", nlas[1][0]), ("2 words", nlas[2][0]), ("4 words", av_rl)]:
    with t.no_grad():
        texts = decode_descriptions(sample_descriptions(av, h_test, greedy=True))
    pairs = []  # (true subject, claimed subject)
    for text, fact in zip(texts, facts_test):
        words = text.split()
        pairs += [(fact["subject"][0], w) for j, w in enumerate(words) if w in NOUN_WORDS and (j == 0 or words[j - 1] not in PREPOSITIONS)]
    number_right = np.mean([(true in NOUN_WORDS[1::2]) == (said in NOUN_WORDS[1::2]) for true, said in pairs])
    noun_right = np.mean([LEMMA[true] == LEMMA[said] for true, said in pairs])
    swaps = Counter((LEMMA[true], LEMMA[said]) for true, said in pairs if LEMMA[true] != LEMMA[said])
    print(f"{name:<10} mentions {dict(Counter(facts_mentioned(x) for x in texts).most_common(2))}")
    print(f"{'':<10} {len(pairs)} subject claims: number right {number_right:.1%}, noun right {noun_right:.1%}; "
          f"most common swaps (true → said): {[f'{a} → {b}' for (a, b), _ in swaps.most_common(4)]}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:WRONG -->warm start mentions {'subject + number + tense': 468, 'subject + number + tense + time': 243}
           834 subject claims: number right 100.0%, noun right 99.4%; most common swaps (true → said): ['boy → girl', 'student → teacher', 'girl → dog']
1 word     mentions {'subject': 684, 'time': 316}
           684 subject claims: number right 97.1%, noun right 78.8%; most common swaps (true → said): ['bird → dog', 'girl → boy', 'student → girl', 'student → boy']
2 words    mentions {'subject + time': 999, 'time': 1}
           999 subject claims: number right 100.0%, noun right 78.0%; most common swaps (true → said): ['boy → girl', 'cat → dog', 'cat → boy', 'girl → boy']
4 words    mentions {'subject + time + object': 1000}
           1000 subject claims: number right 100.0%, noun right 97.6%; most common swaps (true → said): ['bird → cat', 'boy → girl', 'dog → girl', 'dog → cat']</pre>

</details>

The warm-started verbaliser gets the subject's noun right 99.4% of the time. After RL, the 4-word NLA gets it right 97.6% of the time, and the 1- and 2-word NLAs only about 78% of
the time. The number, in contrast, stays (almost) perfect, and (as you can work out from the overall claim accuracies printed earlier) so do the time words.

The swaps aren't random. They're between *similar* nouns: `bird` → `dog`, `cat` → `dog`, `girl` → `boy`, `student` → `girl`. So the descriptions are wrong in their specifics, but
right in the facts that carry the most variance (number and tense, via the noun form and the time word) and roughly right in kind (an animal, a person). That's a miniature version
of an observation in the NLA paper: its explanations are often false in their specifics while remaining thematically faithful.

Why would end-to-end training make the verbaliser *less* accurate about the noun than its warm start, when the information is clearly readable from the activation? We don't know
for sure. A plausible explanation: similar nouns produce similar activations, so the reconstruction reward barely distinguishes `girl` from `boy`, while the reconstructor is a
little better at decoding descriptions it has seen often. That gives the policy a weak, systematic push towards merging rarer descriptions into more common ones, the same force
that collapsed the cold start, here acting only where it costs almost nothing. With only one or two words, each word has to be chosen more carefully, so the push towards the most
decodable word matters more. Testing this explanation is a good exercise.

The lesson for evaluating NLAs: **the reconstruction objective guarantees that the text carries information about the activation, not that each claim in it is true.** Claims that
matter little for reconstruction are the ones most likely to drift.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## The bag-of-words bypass

Here's a more uncomfortable baseline. An NLA's description is supposed to tell us what the *activation* contains. But the activation is a deterministic function of the input
sentence, so any decoder that sees the input could reconstruct it without any help from the verbaliser. The NLA paper flags exactly this: an AV could get good reconstructions by
reproducing the input context verbatim. (Our reconstructor *is* GPT-2 up to layer 10, so given the original sentence verbatim, it would reconstruct the activation perfectly, with
the identity map as its head.)

The verbatim input is clearly cheating. What about something cruder: a **bag of words** of the input, i.e. how many times each word occurs, with the order thrown away? That's the
kind of "description" you'd get from a verbaliser that just lists salient input words. Let's see how far a linear decoder on a bag of words gets, and compare it to the same decoder
given the words *with their positions*.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - bag-of-words features and a ridge decoder

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15 minutes on this exercise.
> ```

Implement:

- `bag_of_words(sentences, vocab)`: split each sentence into words (remove the comma first), and return a `(n_sentences, len(vocab))` float tensor of word counts. Ignore words that aren't in `vocab`.
- `fit_ridge(X, Y, alpha)`: closed-form ridge regression with an unpenalised intercept. Centre $X$ and $Y$ with their means $\mu_X, \mu_Y$, solve $W = (X_c^\top X_c + \alpha I)^{-1} X_c^\top Y_c$ (use `t.linalg.solve`), and return a function `predict(X_new) = (X_new - mu_X) @ W + mu_Y`.

We'll fit the decoders to the unit-normalised training activations.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def bag_of_words(sentences: list[str], vocab: list[str]) -> Float[Tensor, "n vocab"]:
    """Word counts (order discarded), over a fixed vocabulary."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    index = {word: i for i, word in enumerate(vocab)}
    counts = t.zeros(len(sentences), len(vocab))
    for row, sentence in enumerate(sentences):
        for word in sentence.replace(",", "").split():
            if word in index:
                counts[row, index[word]] += 1
    return counts
    # END SOLUTION


def fit_ridge(X: Float[Tensor, "n f"], Y: Float[Tensor, "n d"], alpha: float = 1.0):
    """Closed-form ridge regression with an unpenalised intercept. Returns a prediction function."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    X, Y = X.double(), Y.double()
    mu_X, mu_Y = X.mean(dim=0), Y.mean(dim=0)
    Xc, Yc = X - mu_X, Y - mu_Y
    W = t.linalg.solve(Xc.T @ Xc + alpha * t.eye(X.shape[1], dtype=X.dtype), Xc.T @ Yc)

    def predict(X_new: Tensor) -> Tensor:
        return ((X_new.double() - mu_X) @ W + mu_Y).float()

    return predict
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_bag_of_words(bag_of_words)
    tests.test_fit_ridge(fit_ridge)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
The "ordered words" baseline uses the same decoder on the same words, but gives each word slot its own bag (so `dogs` as the 3rd word and `dogs` as the 6th word are different
features). It's the bag of words plus word order, and nothing else.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

sentences = df.sentence.tolist()
input_vocab = sorted({word for s in [sentences[i] for i in train_idx] for word in s.replace(",", "").split()})
bow = bag_of_words(sentences, input_vocab)
ordered = t.cat(
    [bag_of_words([s.replace(",", "").split()[p] for s in sentences], input_vocab) for p in range(7)], dim=1
)
baselines = {}
for name, X in [("bag of words of the input", bow), ("ordered words of the input", ordered)]:
    predict = fit_ridge(X[train_idx], acts_unit[train_idx])
    h_hat = predict(X[test_idx])
    baselines[name] = dict(fve=fve(acts[test_idx], h_hat), transplant=transplant(h_hat), h_hat=h_hat)
    print(f"{name:<28}: {X.shape[1]:>3} features, FVE {baselines[name]['fve']:.3f}, transplant {baselines[name]['transplant']:.3f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:BOW -->bag of words of the input   :  44 features, FVE 0.698, transplant 0.612
ordered words of the input  : 308 features, FVE 0.881, transplant 0.909</pre>

</details>

The bag of words reconstructs 70% of the variance, roughly as much as `subject + time word` (72%), from features that contain no information about word order at all. A decoder that
sees the input words *with* their positions gets 88%, the same as the four-word NLA. So far, so unsurprising.

But look at the transplant accuracy: the ordered words get 91%, about the same as the real activation, and the bag of words gets only 61%. Something the model uses is missing.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Where does the bag of words fail?

Let's split the transplant test by whether the source sentence's subject and object have the *same* number (`the dogs near the cats`) or *different* numbers (`the dogs near the
cat`). Make a prediction first. Then run it for the real activation, the NLA, and both input baselines.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

same_number = t.tensor((df.number.values == df.object_number.values)[test_idx.numpy()])
with t.no_grad():
    nla4_h_hat = ar_rl(text_features(model, sample_descriptions(av_rl, h_test, greedy=True))).cpu()
split_results = {}
for name, h_hat in [
    ("A's own activation", acts_unit[test_idx]),
    ("NLA, 4 words", nla4_h_hat),
    ("bag of words of the input", baselines["bag of words of the input"]["h_hat"]),
    ("ordered words of the input", baselines["ordered words of the input"]["h_hat"]),
]:
    split_results[name] = {
        f"subject & object same number": transplant_accuracy(model, tokens, labels, h_hat[same_number], test_idx[same_number], partners[same_number], act_scale),
        f"subject & object differ": transplant_accuracy(model, tokens, labels, h_hat[~same_number], test_idx[~same_number], partners[~same_number], act_scale),
    }
split_df = pd.DataFrame(split_results)
display(split_df.round(3))
split_df.T.plot.bar(figsize=(9, 4), rot=0, color=["tab:blue", "tab:orange"])
plt.axhline(0.25, color="k", ls=":", lw=1, label="chance")
plt.ylabel("transplant accuracy")
plt.ylim(0, 1.2)
plt.title("Transplant accuracy, split by whether the source's subject and object agree in number", fontsize=10)
plt.legend(fontsize=8, loc="upper center", ncol=3)
plt.tight_layout()
plt.show()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/bow_split.png" width="750">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret this?</summary>

When the subject and object have the same number (`the dogs near the cats`), the bag of words is fine: 91%, as good as anything. When they differ (`the dogs near the cat`), it
drops to 33.5%, roughly the one-in-three we'd get by fixing the tense and taking the number from B's context.

The reason is word order. `Yesterday, the dogs near the cat really` and `Yesterday, the cat near the dogs really` have *exactly the same bag of words*, and the decoder can't tell
which noun is the subject. But GPT-2 can, and the verb depends on it (`were` versus `was`). The bag of words gets most of the variance right, because most of the variance is about
*which words* are present, but it misses the one fact the model needs, which depends on their order.

The NLA (88%/85%) and the ordered words (90%/92%) don't have this problem: the NLA's description puts the subject first, and the ordered-words decoder knows positions. This is the
bag-of-words bypass: **a decoder that only has lexical information can look strong on FVE while losing exactly the order-dependent factors that drive behaviour**. If you only
measured FVE, you'd conclude the bag of words was 80% as good as the NLA.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Does the reconstructor read word order?

The bag of words fails because it can't tell which noun came first. Does our NLA's reconstructor care about order in *its* input? Recall it reads the description with GPT-2, which
certainly can see order; but it was only ever trained on descriptions in the teacher's canonical order, so it could also have learned to ignore position. Let's shuffle the words of
each of the 4-word NLA's descriptions and decode them again. We also add two more controls: the description of **another** sentence, and **random words** of the same lengths.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

with t.no_grad():
    nla_ids = sample_descriptions(av_rl, h_test, greedy=True)
nla_texts = decode_descriptions(nla_ids)
shuffle_rng = random.Random(0)
shuffled_texts = [shuffle_rng.sample(text.split(), len(text.split())) for text in nla_texts]
with t.no_grad():
    shuffled_h_hat = ar_rl(text_features(model, encode_descriptions(shuffled_texts).to(device))).cpu()
    random_ids = t.where(nla_ids == EOS, nla_ids, t.randint_like(nla_ids, 0, EOS))
    random_h_hat = ar_rl(text_features(model, random_ids)).cpu()
damaged = {
    "NLA, 4 words": nla4_h_hat,
    "same words, shuffled": shuffled_h_hat,
    "another sentence's description": nla4_h_hat[shuffled_sources - N_TRAIN],
    "random words": random_h_hat,
}
damaged_results = {name: dict(fve=fve(acts[test_idx], h_hat), transplant=transplant(h_hat)) for name, h_hat in damaged.items()}
print("e.g.", [f"{a!r} -> {' '.join(b)!r}" for a, b in zip(nla_texts[:3], shuffled_texts[:3])])
display(pd.DataFrame(damaged_results).T.round(3))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:SHUFFLE -->e.g. ["'tables currently with keys' -&gt; 'keys currently tables with'", "'house earlier behind window' -&gt; 'window earlier behind house'", "'boys earlier with tree' -&gt; 'with tree boys earlier'"]
                                  fve  transplant
NLA, 4 words                    0.881       0.864
same words, shuffled            0.452       0.504
another sentence's description -0.883       0.243
random words                   -0.745       0.163</pre>

</details>

Shuffling the words of the NLA's descriptions drops FVE from 0.88 to 0.45 and transplant accuracy from 86% to 50%. So the reconstructor does read word order: `keys currently tables
with` is not the same description as `tables currently with keys`. In the shuffled version, the first noun is often the *object*, so the reconstructor is likely to build an
activation for the wrong subject.

The other two controls do what they should. The description of a different sentence gives a large *negative* FVE (-0.88): a confident reconstruction of the wrong activation is much
worse than predicting the mean. Its transplant accuracy is 24%, chance. Random words are similar (FVE -0.75, transplant 16%).
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## The whole picture: reconstruction versus behaviour

Let's put everything on one plot: every NLA, every description oracle, every baseline and every control, placed by its FVE (x-axis) and its transplant accuracy (y-axis). A perfect
reconstruction would sit in the top-right corner, next to the real activation's ceiling.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

points = {
    "NLA, 4 words": (nla_results[4]["fve"], nla_results[4]["transplant"], "nla", (6, -10, "left")),
    "NLA, 2 words": (nla_results[2]["fve"], nla_results[2]["transplant"], "nla", (-8, -14, "right")),
    "NLA, 1 word": (nla_results[1]["fve"], nla_results[1]["transplant"], "nla", (6, -12, "left")),
    "NLA warm start (before RL)": (warm_eval["fve"], warm_eval["transplant"], "nla", (-6, -13, "right")),
    "oracle: number + tense": (*oracle_points["number + tense"], "oracle"),
    "oracle: subject": (*oracle_points["subject"], "oracle"),
    "oracle: subject + time word": (*oracle_points["subject + time word"], "oracle", (-6, 6, "right")),
    "bag of words of input": (baselines["bag of words of the input"]["fve"], baselines["bag of words of the input"]["transplant"], "baseline"),
    "ordered words of input": (baselines["ordered words of the input"]["fve"], baselines["ordered words of the input"]["transplant"], "baseline", (7, -3, "left")),
    "PCA, 4 dims": (pca_results[4]["fve"], pca_results[4]["transplant"], "baseline"),
    "PCA, 16 dims": (pca_results[16]["fve"], pca_results[16]["transplant"], "baseline", (0, 9, "center")),
    "NLA words shuffled": (damaged_results["same words, shuffled"]["fve"], damaged_results["same words, shuffled"]["transplant"], "control", (-6, 6, "right")),
    "another sentence's description": (damaged_results["another sentence's description"]["fve"], damaged_results["another sentence's description"]["transplant"], "control"),
    "random words": (damaged_results["random words"]["fve"], damaged_results["random words"]["transplant"], "control"),
    "mean activation": (0.0, transplant_controls["mean activation"], "control"),
}
utils.plot_fve_vs_behaviour(
    points,
    "Reconstruction quality (FVE) is not behaviour: GPT-2 Small, layer 10, 1,000 held-out sentences",
    ceiling=transplant_controls["A's own activation"],
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/fve_vs_behaviour.png" width="850">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret this plot?</summary>

The diagonal from bottom-left to top-right is where you'd expect things to lie if FVE and behaviour went together. Several points are far from it.

**Top left: `number + tense`.** FVE 0.18, transplant 99.5%. It describes exactly the two facts the model uses, and nothing else. A "worse" reconstruction by the autoencoder's own
objective is the best one behaviourally.

**Middle right: the bag of words of the input.** FVE 0.70, transplant 61%. It gets most of the variance from which words are present, and loses the order-dependent fact (which noun
is the subject) that decides the verb.

**Bottom middle: `subject` and the one-word NLA.** FVE about 0.5, transplant 35-51%. When the budget is one word, the FVE objective spends it on the subject, the highest-variance
fact.

**Top right: the good bottlenecks.** The two- and four-word NLAs, `subject + time word`, the ordered input words and PCA with 16 dimensions all reach FVE 0.7-0.88 and transplant
accuracy 86-93%. Getting *all* the variance-carrying facts also gets the verb, because the verb is a function of them.

**Bottom left: the controls.** The mean activation, random words, and the description of a different sentence all have FVE at or below zero and transplant accuracy between 2% and
24%. Shuffling the NLA's words lands it in the middle.

The general point: **FVE tells you how much of an activation's variance a description captures; it doesn't tell you whether the description captures what the model uses.** These
can come apart in both directions. If you're evaluating an NLA (or a sparse autoencoder, or any reconstruction-based method), measure both.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## What the text can't carry

Let's be precise about what our NLA's text did and didn't carry.

**It carried every fact it had words for, in the right roles.** With four words, the descriptions say the subject, the time word, the preposition and the object, and 97.7% of those
claims are true. That's enough to pin down the sentence completely (the remaining words, `the`, `,` and `really`, never change), so in principle a perfect reconstructor could
rebuild the activation exactly from the description. The remaining 12% of unexplained variance is mostly the reconstructor's fault (plus the few false claims), not a limit of the
text: a linear head on frozen GPT-2 features isn't a perfect decoder. (The ordered-words ridge decoder, which has the same information, gets the same 88%.)

**It carried word order, but only because the reconstructor reads it.** The same words in a different order gave half the FVE.

**It didn't carry what it wasn't allowed to say.** With one word, the NLA reconstructs half the variance and gets the verb right only half the time; with two words, it gets the
verb but not the object. The text can only carry as much as its budget allows, and the objective decides what to keep based on variance, not on what the model uses.

**It doesn't carry anything we don't have words for.** In this toy, every latent factor has a name in the vocabulary. That's the biggest difference from the real setting. An
activation in a real model contains features nobody has named, combinations that are awkward to express in words, and graded quantities (how confident, how formal, how close to the
end of the document) that a short description can only approximate. The paper's NLAs explain 60-80% of the variance of their activations; the rest is, by definition, what their
explanations don't say. And an NLA's description, like any description, is chosen to explain variance, so the parts of an activation that are low-variance but important (like
`number + tense` here) are the parts most at risk of being left out.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Try it yourself - edit the bottleneck

Because the bottleneck is text, you can edit it. The function below takes any short description you like (any words that are single GPT-2 tokens, whether or not the verbaliser ever
uses them), reconstructs an activation from it with the 4-word NLA's reconstructor, patches that activation into a base sentence, and prints GPT-2's distribution over the four
verbs. As a reference, it also patches in the *real* activation of the sentence your description is "about", if you tell it what that sentence is.

Some things to try:

- Change the number of the subject (`tables` → `table`) or the time word (`currently` → `yesterday`). Does the verb follow?
- Use the abstract words: `plural past`, `singular present`.
- Use words the verbaliser never writes and the reconstructor never saw in training: `formerly`, `recently`, `puppies`, `vehicles`. GPT-2's embeddings know these words are related to ones it did see. Does the reconstructor generalise?
- Contradict yourself: `table plural`, or `yesterday present`. Which word wins?
- Write something long: the reconstructor was trained on at most four words, but GPT-2 doesn't mind.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

def edit_and_patch(description: str, base: int, reference_sentence: str | None = None) -> None:
    words = description.split()
    gpt2_tokens = [model.tokenizer.bos_token_id] + [model.to_single_token(" " + w) for w in words]
    gpt2_tokens += [model.to_single_token(".")] * max(1, MAX_WORDS + 1 - len(words))  # pad with "." like text_features
    with t.no_grad():
        h_hat = ar_rl(read_with_gpt2(model, t.tensor([gpt2_tokens]), LAYER)).cpu()
    base_tokens = tokens[base : base + 1]
    rows = {"reconstruction from your text": h_hat}
    if reference_sentence is not None:
        rows["real activation of the reference"] = get_final_activations(model, model.to_tokens(reference_sentence), [LAYER])[0]
    print(f"description: {description!r:<34} base sentence: {df.sentence[base]!r}")
    for name, vector in rows.items():
        logits = patch_final_position(model, base_tokens, F.normalize(vector, dim=-1) * act_scale, LAYER, verb_ids)
        probs = logits.softmax(-1)[0]
        cos = F.cosine_similarity(vector, acts[base : base + 1]).item()
        print(f"  {name:<34} cos to base activation {cos:+.2f} | " + "  ".join(f"{v.strip()}: {p:.2f}" for v, p in zip(VERBS, probs)))


base = int(test_idx[0])
print(f"The NLA's own description of the base sentence: {nla_texts[0]!r}\n")
edit_and_patch("tables currently with keys", base)
edit_and_patch("table currently with keys", base, reference_sentence="Currently, the table with the keys really")
edit_and_patch("tables yesterday with keys", base, reference_sentence="Yesterday, the tables with the keys really")
edit_and_patch("tables formerly with keys", base)
edit_and_patch("singular past", base)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><!-- out:PLAY -->The NLA's own description of the base sentence: 'tables currently with keys'

description: 'tables currently with keys'       base sentence: 'Currently, the tables with the keys really'
  reconstruction from your text      cos to base activation +1.00 | is: 0.01  are: 0.95  was: 0.00  were: 0.04
description: 'table currently with keys'        base sentence: 'Currently, the tables with the keys really'
  reconstruction from your text      cos to base activation +0.99 | is: 0.45  are: 0.50  was: 0.02  were: 0.03
  real activation of the reference   cos to base activation +0.99 | is: 0.60  are: 0.36  was: 0.02  were: 0.02
description: 'tables yesterday with keys'       base sentence: 'Currently, the tables with the keys really'
  reconstruction from your text      cos to base activation +0.95 | is: 0.00  are: 0.56  was: 0.01  were: 0.42
  real activation of the reference   cos to base activation +0.91 | is: 0.00  are: 0.40  was: 0.02  were: 0.58
description: 'tables formerly with keys'        base sentence: 'Currently, the tables with the keys really'
  reconstruction from your text      cos to base activation +0.98 | is: 0.00  are: 0.45  was: 0.01  were: 0.54
description: 'singular past'                    base sentence: 'Currently, the tables with the keys really'
  reconstruction from your text      cos to base activation +0.91 | is: 0.43  are: 0.02  was: 0.51  were: 0.04</pre>

</details>

The NLA's own description reconstructs the base activation almost exactly (cosine similarity 1.00 to two decimal places), and GPT-2 says `are` (0.95).

Changing `tables` to `table` moves the reconstruction only slightly (cosine 0.99), and moves GPT-2 from `are` to an even split (`is` 0.45, `are` 0.50). That might look like a weak
effect, but compare it with the *real* activation of `"Currently, the table with the keys really"`, patched into the same base sentence: it only gets to `is` 0.60. The base
sentence's context (`the tables`) still pulls towards `are` from the other positions. The edited text gets most of the way to what the real activation does. Changing `currently` to
`yesterday` is similar: `were` 0.42 from the text, 0.58 from the real activation.

`formerly` is not in the verbaliser's vocabulary, so the reconstructor's head never saw it in training. But GPT-2 reads it, and represents it similarly enough to the past-tense
time words that the reconstruction pushes towards `were` (0.54), slightly *more* than `yesterday` did. That's the benefit of a reconstructor that reads text with a pretrained
language model: the bottleneck language is open-ended, rather than a fixed codebook.

`singular past`, in the abstract words that the trained verbaliser no longer uses, also works: the number flips completely (`is` + `was` = 0.94), and the tense about half-way
(`was` 0.51), with the base sentence's `Currently` pulling towards the present.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## What changes at full scale

Our NLA is a toy, and it's worth being explicit about which of its lessons should transfer to the real thing and which shouldn't.

**What's the same.** The architecture (a verbaliser that reads the activation in place of a token embedding; a reconstructor that is the target model truncated at the activation's
layer, plus an affine map on the final token), the metric (FVE of unit-normalised activations), the training recipe (a supervised warm start, then GRPO with a $-\log$ MSE reward
and a KL penalty to the warm start, with the reconstructor trained by regression on the verbaliser's samples), and the central tension: the objective rewards text that is
*informative about the activation*, which is not the same as text that is true, readable, or about what the model *uses*.

**What's different.**

- **The verbaliser is a full language model.** In the paper both halves are copies of the target model (Claude Haiku 3.5, Haiku 4.5 and Opus 4.6, and some open models), and the explanations are free-form text, several hundred tokens long. Ours writes at most four words from a 47-word vocabulary that only contains facts. A full language model can do things ours can't: it can *infer* things that aren't in the activation (the paper lists this "excessive expressivity" as a limitation), and it can confabulate plausible but false specifics. The paper reports that explanations are typically thematically faithful even when their specifics are wrong.
- **There's no ground truth.** We checked every claim against the sentence's known factors. At scale, nobody knows what an activation "really" contains, so claims have to be validated indirectly: by checking them against the context, by noticing that claims which recur across several adjacent tokens are more likely to be true, and by downstream tasks. The paper evaluates NLAs by using them: NLA-equipped agents do better than baselines on an automated auditing task (finding out what's wrong with an intentionally misaligned model), NLA explanations surface evaluation awareness more often on evaluation transcripts than on real deployment traffic, and they reveal the model's plan for an upcoming rhyme in poetry, which the authors confirm with steering experiments.
- **The activation contains far more than any short text.** In our toy, four words pin down the sentence completely; whatever the 4-word NLA misses is mostly the reconstructor's fault, not a limit of the text. At scale, an activation in the middle of a long document carries information about a great many things, and any finite explanation has to choose. That makes the budget experiment above the important one: what an NLA chooses to say is determined by what explains the most *variance*, and part 1️⃣ showed that this can be quite different from what the model *uses*.
- **The numbers.** The paper's warm-started NLAs reach an FVE of about 0.3-0.4, and RL brings them to about 0.6-0.8, with FVE growing roughly linearly in the logarithm of the number of training steps. Ours goes from 0.5 to almost 0.9 in 600 steps, because the problem is tiny.
- **Cost.** Joint RL on two full language models, sampling hundreds of tokens for every activation, is expensive. Our whole section runs in a few minutes on one GPU.

**What the toy suggests checking at scale.** Each of our controls has a natural full-scale analogue. *Behaviour*: patch the reconstruction back in; does the model do what it did
with the original activation, on a test where the mean activation fails? *Order and meaning*: shuffle, paraphrase or translate an explanation; does the reconstruction survive
changes that preserve meaning, and break under changes that don't? *Drift*: on inputs where some ground truth is available, are the explanation's claims true, and does that change
over training? *Bypasses*: how close does a decoder that just sees (a summary of) the input context get?
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 4️⃣

- Word for dimension, the NLA's text bottleneck keeps far more variance than PCA (one word: 0.51 vs 0.21; four words: 0.88 vs 0.53), because the activation's variance is organised around nameable categorical facts.
- With a small budget, the FVE objective spends words on high-variance facts: the one-word NLA mostly says the subject (for 684 of 1,000 sentences) and gets the verb right only about half the time, while a two-word oracle description of number and tense gets it right 99.5% of the time.
- A bag of words of the input reaches FVE 0.70 but gets only 61% of transplants right, and 33.5% when the subject and object differ in number, because it can't represent which noun is the subject. The same decoder with word order gets 91%.
- The reconstructor reads word order: shuffling the NLA's words halves its FVE.
- End-to-end training makes the NLA say what helps reconstruction, not necessarily what's true: its false claims swap similar nouns (`bird` → `dog`, `girl` → `boy`) while keeping the high-variance facts right, and they get more common as the word budget shrinks.
- FVE and behaviour are different axes, and a reconstruction-based interpretability method should be evaluated on both.
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
## Train an NLA at a different layer

The description-oracle heatmap says the final position is about the *object* at layer 4, and about the *subject and time word* at layer 10. If NLAs describe what's in the
activation, an NLA trained at layer 4 should start talking about the object.

- Set `LAYER = 4`, recompute `acts`, and retrain the warm start, reconstructor and RL. (The reconstructor should read GPT-2 at layer 4 too, as in the paper.) What does the trained verbaliser say? Does it match the heatmap's prediction?
- The transplant-by-layer plot in part 1️⃣ says that at layer 4 the verb decision hasn't reached the final position yet, so the transplant test can't tell good reconstructions from bad ones there. How would you evaluate whether a layer-4 NLA preserves what the model *uses* from that activation? (One option: measure the effect on predictions further downstream, or on a different token.)

## Reward behaviour instead of variance

Part 4️⃣ showed that the one-word NLA mostly spends its word on the subject, because that's the most variance, even though `plural past` style descriptions would preserve behaviour
far better. What if the reward cared about behaviour?

- Replace the reward with the negative KL divergence between GPT-2's next-token distribution with the original activation patched in and with the reconstruction patched in (in the activation's own sentence, or better, in a transplant partner). This is analogous to training sparse autoencoders end-to-end on downstream KL rather than on reconstruction MSE ([Braun et al., 2024](https://arxiv.org/abs/2405.12241)).
- With a one-word budget, what does the verbaliser learn to say now? Can you find a mix of the two rewards which gets a good FVE *and* good behaviour?

## Make the verbaliser lie

With four words, REINFORCE from a warm start stayed (mostly) truthful with or without the KL penalty. With one or two words, it started swapping similar nouns even with the
penalty. Explore when the verbaliser lies, and whether the KL penalty helps:

- For the 2-word NLA, compare $\beta = 0$, $0.05$ and $0.5$. How do noun accuracy, FVE and transplant accuracy change? Is there a $\beta$ that keeps the nouns right without hurting FVE?
- Remove the prepositions and object nouns from what the teacher says (the verbaliser can still *write* the nouns), so that the object is unnameable in the warm-start language. The object still carries some variance. With $\beta = 0$, does the verbaliser start encoding it, e.g. by adding a second noun, or by repurposing the redundant `singular`/`plural`/`present`/`past` words? What happens to claim accuracy? Does $\beta > 0$ prevent it?
- Can you get the Gumbel straight-through run to stay truthful by adding the KL penalty to its loss?

## Rescue the cold start

The cold start collapsed to a single description. What's the smallest change that avoids the collapse? Some ideas: an entropy bonus; freezing the reconstructor at a random
initialisation for the first few hundred steps; a much smaller learning rate for the reconstructor; a curriculum that starts with a vocabulary of two words. If you get a
cold-started NLA to reconstruct well, look at its descriptions. Are its claims true? (This is the classic "emergent communication" setting; see the reading list.)

## A real verbaliser

Replace our 2-layer verbaliser with GPT-2 Small itself: inject the (projected) activation as a token embedding into a fixed prompt, and fine-tune it to write the teacher's
descriptions, as the paper does with the target model (and as in SelfIE and LatentQA). Does a pretrained language model make a better warm start? Does it make the cold-start
problem go away?

## Further reading

- Fraser-Taliente, Kantamneni, Ong et al. (2026), [*Natural Language Autoencoders Produce Unsupervised Explanations of LLM Activations*](https://transformer-circuits.pub/2026/nla/), and the accompanying [Anthropic post](https://www.anthropic.com/research/natural-language-autoencoders). The method this section is a miniature of.
- Williams (1992), *Simple statistical gradient-following algorithms for connectionist reinforcement learning*. The REINFORCE estimator.
- Shao et al. (2024), [*DeepSeekMath*](https://arxiv.org/abs/2402.03300). Introduces GRPO, the group-baseline policy gradient used by the NLA paper.
- Jang, Gu & Poole (2016), [*Categorical Reparameterization with Gumbel-Softmax*](https://arxiv.org/abs/1611.01144); Maddison, Mnih & Teh (2016), [*The Concrete Distribution*](https://arxiv.org/abs/1611.00712); Bengio, Léonard & Courville (2013), [*Estimating or Propagating Gradients Through Stochastic Neurons*](https://arxiv.org/abs/1308.3432) (the straight-through estimator).
- Havrylov & Titov (2017), [*Emergence of Language with Multi-agent Games*](https://arxiv.org/abs/1705.11192), and Kottur et al. (2017), [*Natural Language Does Not Emerge 'Naturally' in Multi-Agent Dialog*](https://arxiv.org/abs/1706.08502). Two agents that communicate through discrete symbols, trained end-to-end: the cold-start version of an NLA. The symbols they invent are usually not human language.
- Lewis et al. (2017), [*Deal or No Deal? End-to-End Learning for Negotiation Dialogues*](https://arxiv.org/abs/1706.05125), and Lu et al. (2020), [*Countering Language Drift with Seeded Iterated Learning*](https://arxiv.org/abs/2003.12694). Language models fine-tuned with RL drift away from human language unless something anchors them; the NLA paper's KL penalty is one such anchor.
- Chen et al. (2024), [*SelfIE*](https://arxiv.org/abs/2403.10949), and Pan et al. (2024), [*LatentQA*](https://arxiv.org/abs/2412.08686). Supervised methods for getting a model to describe its own activations in words; compare with the activation oracles in [7.3].
- Braun et al. (2024), [*Identifying Functionally Important Features with End-to-End Sparse Dictionary Learning*](https://arxiv.org/abs/2405.12241). The same "variance is not function" point for sparse autoencoders.
- Linzen, Dupoux & Goldberg (2016), [*Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies*](https://arxiv.org/abs/1611.01368), and Finlayson et al. (2021), [*Causal Analysis of Syntactic Agreement Mechanisms in Neural Language Models*](https://arxiv.org/abs/2106.06087). Where our task comes from.
'''

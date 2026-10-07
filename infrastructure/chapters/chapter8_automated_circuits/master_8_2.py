# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
[
    {"title": "Attribution Patching", "icon": "1-circle-fill", "subtitle": "(25%)"},
    {"title": "Where the Linear Approximation Breaks", "icon": "2-circle-fill", "subtitle": "(20%)"},
    {"title": "Edge Attribution Patching", "icon": "3-circle-fill", "subtitle": "(25%)"},
    {"title": "EAP-IG and Faithfulness", "icon": "4-circle-fill", "subtitle": "(25%)"},
    {"title": "Bonus", "icon": "star", "subtitle": "(5%)"},
]
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# [8.2] Attribution Patching and EAP
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
GPT-2 Small completes

```
When Mary and John went to the store, John gave a drink to
```

with `" Mary"`. In [1.4.1] you found *how*, by activation patching: run the model on a corrupted prompt where the names are different, splice one component's activation from one run into the other, and measure how much the logit difference between `" Mary"` and `" John"` moves. Do that for every attention head and you get the famous IOI circuit map.

Now count the forward passes. One per head is 144. One per (layer, position) of the residual stream is 180. And if you want to know not just *which components* matter but *which connections between them* matter (does head 9.9 get its query from the S-inhibition heads or from somewhere else?), you need one forward pass per **edge** of the model's computational graph. For GPT-2 Small, with heads, MLPs and separate query/key/value inputs, that's 32,491 forward passes for one batch of prompts. For a 7B-parameter model the count runs into the millions.

**Attribution patching** (Nanda, 2023) replaces all of those forward passes with *one* forward pass and *one* backward pass. The idea is a first-order Taylor expansion: if you know the gradient of the metric with respect to an activation, you can estimate what would happen if you changed that activation, without actually changing it. **Edge attribution patching** (EAP; Syed, Rager & Conmy, 2023) applies the same trick to every edge of the graph at once. And **EAP-IG** (Hanna, Pezzelle & Belinkov, 2024) replaces the single gradient with an average of gradients along the path from the corrupted input to the clean one, i.e. integrated gradients.

Approximations are only useful if you know when to trust them. So the question running through this whole section is:

> *When can a gradient stand in for an intervention, and what goes wrong when it can't?*

By the end you'll have built all three methods from scratch on GPT-2 Small and produced four pictures: attribution patching agreeing with exact patching head by head (in a fraction of the time); a gallery of components where the first-order approximation gets the *sign* of the effect wrong, and why; the IOI circuit drawn from 32,491 edge scores that took a couple of seconds to compute; and faithfulness curves showing that the circuit EAP finds for IOI barely works at all, while the one EAP-IG finds recovers a large fraction of the behaviour. Explaining that last gap leads to the most important idea in the section: approximating single-edge patching well is not the same thing as finding a faithful circuit.

As usual, every headline result comes with a control that should fail: gradient-only and activation-only baselines, shuffled scores, randomly chosen edges, held-out prompts, and a second task.

This section assumes you've done [1.4.1] Indirect Object Identification (at least the activation patching parts), and are comfortable with TransformerLens hooks and PyTorch autograd. Section [8.1] covers noising vs denoising and patching metrics in more depth; you don't need it, but it helps.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Content & Learning Objectives

### 1️⃣ Attribution Patching

You'll derive attribution patching from a first-order Taylor expansion, implement it with one forward and one backward pass, and compare it head by head against exact activation patching on IOI. You'll check that both halves of the formula matter, using activation-only, gradient-only and shuffled baselines.

> ##### Learning Objectives
>
> - Understand attribution patching as a linear approximation to activation patching, and what "noising" and "denoising" mean for it
> - Implement attribution patching with TransformerLens forward and backward hooks
> - Compare attribution patching against exact patching with scatter plots and rank correlation
> - Use baselines to show that the activation difference and the gradient are both necessary

### 2️⃣ Where the Linear Approximation Breaks

Attribution patching works beautifully for attention head outputs. You'll then find places where it fails badly, including components where it gets the sign of the effect wrong, and look at the metric along the straight line from the clean to the corrupted activation to see why: saturating attention softmaxes, and large, nonlinear changes like swapping out a token's identity. Integrating the gradient along that line fixes the estimate, at a price.

> ##### Learning Objectives
>
> - Identify the components and corruptions for which attribution patching is unreliable
> - Interpret attribution patching geometrically, as a tangent line to the metric along a path
> - Understand attention saturation and why it causes both overestimates and false negatives
> - Implement integrated gradients for a single component and watch it converge to the exact effect

### 3️⃣ Edge Attribution Patching

Because the residual stream is a sum, the input to any component is a sum of contributions from earlier components, and those contributions are the *edges* of the model's computational graph. You'll build the graph for GPT-2 Small (32,491 edges), implement EAP, check it against exact edge patching, and draw the IOI circuit from the top-scoring edges.

> ##### Learning Objectives
>
> - Understand the residual-stream computational graph, with separate query, key and value inputs
> - Implement EAP: activation differences at every sender, gradients at every receiver, one einsum
> - Implement exact single-edge patching and use it to check EAP
> - Read an IOI circuit off a graph of edge scores

### 4️⃣ EAP-IG and Faithfulness

A ranked list of edges isn't yet a circuit. You'll implement circuit evaluation (corrupting every edge outside the circuit) and use it to draw faithfulness curves. You'll find that EAP's top edges barely recover IOI at all, implement EAP-IG, which does much better, and track down why: the missing edges belong to *redundant* mechanisms whose single-edge effects are tiny. You'll also check everything on held-out prompts and on a second task (greater-than).

> ##### Learning Objectives
>
> - Implement edge-level circuit evaluation and normalised faithfulness
> - Implement EAP-IG and check its completeness property
> - Compare circuits found by EAP, EAP-IG and random edge sets of the same size
> - Understand why single-edge effects are the wrong target for circuit discovery when mechanisms are redundant

### ☆ Bonus

Denoising-direction attribution, other integration paths, AtP*'s fixes for attention saturation (section [8.3]), and further reading.
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
#     %pip install transformer_lens==2.16.1 jaxtyping einops
#     repo, branch, chapter = "ARENA_3.0_Extended", "main", "chapter8_automated_circuits"
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
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

import einops
import numpy as np
import pandas as pd
import torch as t
from IPython.display import display
from jaxtyping import Bool, Float, Int
from torch import Tensor
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# Make sure exercises are in the path
chapter = "chapter8_automated_circuits"
section = "part2_attribution_patching_eap"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part2_attribution_patching_eap.tests as tests
import part2_attribution_patching_eap.utils as utils

MAIN = __name__ == "__main__"

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 1️⃣ Attribution Patching
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## The model

We'll use GPT-2 Small throughout. We never train it, so we turn off gradients for its parameters (we'll still take gradients with respect to its *activations*, which is the whole point of this section).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

model = HookedTransformer.from_pretrained("gpt2-small", device=device)
model.requires_grad_(False)

print(f"{model.cfg.n_layers} layers, {model.cfg.n_heads} heads per layer, d_model={model.cfg.d_model}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## The task: IOI, a corruption and a metric

Our main task is indirect object identification. We generate prompts from four templates, with random names, places and objects, and with the indirect object (IO) first in half the prompts and second in the other half:

```
When Mary and John went to the store, John gave a drink to       ->  " Mary"   (IO first, "ABBA")
When John and Mary went to the store, John gave a drink to       ->  " Mary"   (IO second, "BABA")
```

Every template has the same length (15 tokens including BOS), with the names at positions 2, 4 and 10, so we can batch prompts without padding.

For each clean prompt we also need a **corrupted** prompt. We'll use the corruption Wang et al. call the **ABC distribution**: replace all three names with three fresh, unrelated names, keeping everything else fixed:

```
clean:    When Mary and John went to the store, John gave a drink to
corrupt:  When Laura and Steve went to the store, Peter gave a drink to
```

The corrupted prompt contains nothing about Mary or John, so the model's preference between `" Mary"` and `" John"` there is roughly zero. Any component whose activation, swapped from the corrupted run into the clean one, destroys the preference must be carrying information the model uses for the task.

Our **metric** is the logit difference, $\text{logit}(\text{IO}) - \text{logit}(\text{S})$, at the final position, averaged over prompts.

We'll package a task as a small `Task` object holding the clean and corrupted tokens, the answers, and a function computing the metric *per prompt*. It can be sliced (`task[:32]` is a task with the first 32 prompts), and its `chunks` method iterates over it in mini-batches, which keeps memory use down when we take gradients.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@dataclass
class Task:
    clean_tokens: Int[Tensor, "batch seq"]
    corrupt_tokens: Int[Tensor, "batch seq"]
    answers: Tensor  # task-specific, e.g. (batch, 2) token ids of the correct and incorrect answer
    metric: Callable[[Tensor, Tensor], Tensor]  # (logits, answers) -> per-prompt metric of shape (batch,)

    def __len__(self) -> int:
        return len(self.clean_tokens)

    def __getitem__(self, sl: slice) -> "Task":
        """A sub-task containing the prompts selected by the slice `sl`, e.g. task[:32]."""
        return Task(self.clean_tokens[sl], self.corrupt_tokens[sl], self.answers[sl], self.metric)

    def chunks(self, batch_size: int) -> Iterator["Task"]:
        """Yields consecutive sub-tasks of at most `batch_size` prompts."""
        for i in range(0, len(self), batch_size):
            yield self[i : i + batch_size]

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement the IOI metric

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 5 minutes on this exercise.
> ```

Implement `ioi_metric`, which takes logits of shape `(batch, seq, d_vocab)` and answers of shape `(batch, 2)` (column 0 is the IO token id, column 1 the subject token id), and returns the logit difference $\text{logit}(\text{IO}) - \text{logit}(\text{S})$ at the final position, for each prompt separately (shape `(batch,)`).

Returning a per-prompt metric rather than the mean is deliberate: when we split a task into chunks later, we'll want to add up per-prompt values across chunks.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def ioi_metric(logits: Float[Tensor, "batch seq d_vocab"], answers: Int[Tensor, "batch 2"]) -> Float[Tensor, " batch"]:
    """Returns logit(IO) - logit(S) at the final position, for each prompt."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    final_logits = logits[:, -1]
    return final_logits.gather(-1, answers[:, :1])[:, 0] - final_logits.gather(-1, answers[:, 1:])[:, 0]
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_ioi_metric(ioi_metric)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Take `logits[:, -1]` (shape `(batch, d_vocab)`), then use `.gather(-1, answers[:, :1])` to pick out each prompt's IO logit (and similarly for S).
</details>

Now let's build the dataset. We make two independent sets of 128 prompts: `ioi`, which we'll use to compute every attribution score in this section, and `ioi_heldout`, which we'll only use in 4️⃣ to evaluate circuits on prompts they weren't selected on.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def make_ioi_task(model: HookedTransformer, n: int, seed: int, corruption: str = "abc") -> Task:
    clean, corrupt, io_names, s_names = utils.make_ioi_prompts(n, seed=seed, corruption=corruption)
    answers = t.stack(
        [model.to_tokens(io_names, prepend_bos=False)[:, 0], model.to_tokens(s_names, prepend_bos=False)[:, 0]], dim=1
    )
    return Task(model.to_tokens(clean), model.to_tokens(corrupt), answers, ioi_metric)


@t.no_grad()
def mean_metric(model: HookedTransformer, task: Task, tokens: Tensor) -> float:
    """The task's metric on `tokens` (either task.clean_tokens or task.corrupt_tokens), averaged over prompts."""
    return task.metric(model(tokens), task.answers).mean().item()

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

ioi = make_ioi_task(model, 128, seed=0)
ioi_heldout = make_ioi_task(model, 128, seed=1)

for i in range(2):
    print("clean:  ", model.to_string(ioi.clean_tokens[i, 1:]))
    print("corrupt:", model.to_string(ioi.corrupt_tokens[i, 1:]))
    print("answers:", [model.to_string(tok) for tok in ioi.answers[i]], "\n")

with t.no_grad():
    clean_ld = ioi.metric(model(ioi.clean_tokens), ioi.answers)
    corrupt_ld = ioi.metric(model(ioi.corrupt_tokens), ioi.answers)
print(f"Clean logit diff:   {clean_ld.mean():.3f}  (IO preferred in {(clean_ld > 0).float().mean():.0%} of prompts)")
print(f"Corrupt logit diff: {corrupt_ld.mean():.3f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">clean:   When George and Mark went to the store, Mark gave a computer to
corrupt: When Grace and Sam went to the store, Jack gave a computer to
answers: [' George', ' Mark']

clean:   After Mike and Emma went to the hospital, Emma gave a snack to
corrupt: After Mark and Max went to the hospital, Peter gave a snack to
answers: [' Mike', ' Emma']

Clean logit diff:   3.180  (IO preferred in 99% of prompts)
Corrupt logit diff: 0.063</pre>

</details>

On the clean prompts the model prefers the indirect object by a logit difference of 3.18 (odds of about 24 to 1), and gets 99% of prompts right. On the corrupted prompts its preference between the two clean names is essentially zero (0.06), as intended: the corrupted prompts don't mention either name. So in the noising direction, "complete damage" means a drop of about 3.1 in logit difference.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## From activation patching to a Taylor expansion

Pick a component, say the output of one attention head, and call its activation $a$ (for all prompts and positions at once). Write $M(a)$ for the value of the metric when we run the model on the clean prompts but force that component's activation to be $a$, letting everything downstream of it recompute as normal. Then $M(a_{\text{clean}})$ is just the clean metric, and **activation patching** in the *noising* direction measures

$$
\Delta M_{\text{exact}} = M(a_{\text{corrupt}}) - M(a_{\text{clean}}),
$$

the damage done by corrupting this one component. A large negative value means the component carries information the model needs. This costs one forward pass per component.

Now suppose $M$ were linear in $a$, at least between $a_{\text{clean}}$ and $a_{\text{corrupt}}$. Then a first-order Taylor expansion around the clean activation would be exact:

$$
M(a_{\text{corrupt}}) \approx M(a_{\text{clean}}) + \left(a_{\text{corrupt}} - a_{\text{clean}}\right) \cdot \left.\frac{\partial M}{\partial a}\right|_{a_{\text{clean}}}
$$

so

$$
\boxed{\;\Delta M_{\text{AP}} = \left(a_{\text{corrupt}} - a_{\text{clean}}\right) \cdot \left.\frac{\partial M}{\partial a}\right|_{\text{clean run}}\;}
$$

This is **attribution patching** (AP). Here's why it's so much cheaper. The activation difference $a_{\text{corrupt}} - a_{\text{clean}}$ needs one forward pass on each input. The gradient $\partial M / \partial a$ needs one backward pass from the metric on the clean run, and that single backward pass gives us the gradient with respect to *every* activation in the model simultaneously. So the cost is two forward passes and one backward pass, whether we score 144 heads or every neuron in the model.

A few remarks before we implement it:

- **Total, not direct, effect.** The gradient $\partial M / \partial a$ flows back through every path from $a$ to the output, so AP estimates the same *total* effect as activation patching, not the direct effect you get from direct logit attribution.
- **The dot product is over everything.** $a$ has shape `(batch, pos, ...)`. The dot product sums over every position and every hidden dimension of the component. And because our metric is the *mean* over the batch, the gradient for each prompt already carries a factor of $1/\text{batch}$, so summing over the batch dimension too gives the estimated change in the *mean* metric.
- **Noising vs denoising.** We expanded around the clean run and moved towards the corrupted activation: this is the *noising* direction, which asks "how much does corrupting this component hurt?". You could equally expand around the corrupted run and move towards the clean activation (*denoising*, which is what Neel Nanda's original post does). We use noising throughout because it's the direction circuit discovery uses in 3️⃣ and 4️⃣: a circuit is a set of edges you can keep clean while corrupting everything else.
- **Gradient × input.** If you've seen "gradient × input" saliency maps, AP is the same idea with a different baseline: instead of comparing an activation with zero, it compares it with the activation on a corrupted input. Choosing the baseline is choosing the question.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - cache activations and gradients

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement `get_cache_fwd_and_bwd`. It should run the model on `tokens`, compute `metric_fn(logits)` (a scalar), backpropagate from it, and return:

- the metric value as a float,
- a dict mapping each hook name in `names` to the activation at that hook,
- a dict mapping each hook name in `names` to the gradient of the metric with respect to that activation.

Use `model.hooks(fwd_hooks=..., bwd_hooks=...)`, a context manager which attaches hooks and removes them again afterwards. Forward hook functions have the signature `hook_fn(activation, hook)`, and backward hook functions have the signature `hook_fn(gradient, hook)`, where `gradient` is the gradient of whatever you call `.backward()` on with respect to that hook's activation. In both cases `hook.name` tells you which hook you're in. Store detached copies (`.detach().clone()`), so the stored tensors don't hold on to the graph.

There's one wrinkle. We froze the model's parameters, and token ids aren't differentiable, so as it stands nothing in the forward pass requires a gradient and `.backward()` will fail. The fix is to make the very first activation require grad: add a forward hook on `"hook_embed"` (the token embeddings) that returns `embed.requires_grad_(True)`. Everything downstream will then be tracked by autograd.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def get_cache_fwd_and_bwd(
    model: HookedTransformer, tokens: Int[Tensor, "batch seq"], metric_fn: Callable[[Tensor], Tensor], names: list[str]
) -> tuple[float, dict[str, Tensor], dict[str, Tensor]]:
    """
    Runs the model on `tokens`, backpropagates from metric_fn(logits), and returns (metric value, activations,
    gradients), where the dicts map each hook name in `names` to the activation / gradient at that hook.
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    activations, gradients = {}, {}

    def make_differentiable(embed: Tensor, hook) -> Tensor:
        return embed.requires_grad_(True)

    def forward_hook(act: Tensor, hook) -> None:
        activations[hook.name] = act.detach().clone()

    def backward_hook(grad: Tensor, hook) -> None:
        gradients[hook.name] = grad.detach().clone()

    in_names = lambda name: name in names
    with model.hooks(
        fwd_hooks=[("hook_embed", make_differentiable), (in_names, forward_hook)], bwd_hooks=[(in_names, backward_hook)]
    ):
        value = metric_fn(model(tokens))
        value.backward()
    return value.item(), activations, gradients
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_get_cache_fwd_and_bwd(get_cache_fwd_and_bwd, model)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Create two empty dicts. Your forward hook stores `act.detach().clone()` in one, keyed by `hook.name`; your backward hook stores `grad.detach().clone()` in the other. You can pass a *function* of the hook name instead of a name as the first element of a hook tuple, e.g. `(lambda name: name in names, forward_hook)`. Inside the `with model.hooks(...)` block, run the model, compute the metric, and call `.backward()` on it.
</details>

<details>
<summary>Help - I get <code>RuntimeError: element 0 of tensors does not require grad</code>.</summary>

You need the `hook_embed` forward hook that calls `requires_grad_(True)` on the embeddings. Without it, no tensor in the forward pass requires a gradient (the model's parameters are frozen), so there's no graph to backpropagate through.
</details>

The test compares your gradient at the final residual stream against an independent calculation: it takes the cached final residual stream, applies the final layernorm and unembedding by hand, and differentiates that with `torch.autograd.grad`.

We'll also need the activations on the corrupted prompts, which only needs a forward pass:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.no_grad()
def get_activations(model: HookedTransformer, tokens: Int[Tensor, "batch seq"], names: list[str]) -> dict[str, Tensor]:
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in names)
    return {name: cache[name] for name in names}

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
A backward pass has to store the intermediate activations of the whole forward pass, and for 128 prompts that's a couple of GB (most of it in the MLPs' hidden layers and in the `(batch, seq, d_vocab)` logits). So we'll run `get_cache_fwd_and_bwd` on chunks of 32 prompts and concatenate the results. The one subtlety is the metric: we want gradients of the *mean* metric over the whole task, so for each chunk we backpropagate from

$$
\frac{1}{N_{\text{task}}} \sum_{\text{prompts in chunk}} M_i ,
$$

and since the mean over the task is the sum of these terms over chunks, the gradients we get for each prompt are exactly the gradients of the task's mean metric. (We'll use the same trick again in 3️⃣.)
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def get_cache_fwd_and_bwd_chunked(
    model: HookedTransformer, task: Task, names: list[str], batch_size: int = 32
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """Clean-run activations and gradients of the task's mean metric at `names`, computed chunk by chunk."""
    acts, grads = {name: [] for name in names}, {name: [] for name in names}
    for chunk in task.chunks(batch_size):
        metric_fn = lambda logits: chunk.metric(logits, chunk.answers).sum() / len(task)
        _, chunk_acts, chunk_grads = get_cache_fwd_and_bwd(model, chunk.clean_tokens, metric_fn, names)
        for name in names:
            acts[name].append(chunk_acts[name])
            grads[name].append(chunk_grads[name])
    return {name: t.cat(v) for name, v in acts.items()}, {name: t.cat(v) for name, v in grads.items()}

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - attribution patching for every attention head

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement `attribution_patch_heads`. It takes clean activations and gradients, and corrupted activations, at the hooks `blocks.{layer}.attn.hook_z` for every layer, and returns a `(n_layers, n_heads)` tensor of AP scores.

`hook_z` has shape `(batch, pos, n_heads, d_head)`: it's each head's attention-weighted sum of value vectors, just before the output projection $W_O$. A head's output in the residual stream is $z_h W_O^h$, so patching $z_h$ is the same as patching the head's output, and the gradient with respect to $z_h$ already includes the $W_O^h$ factor. For each head, compute $(z_{\text{corrupt}} - z_{\text{clean}}) \cdot \partial M / \partial z$, summed over batch, position and `d_head`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def attribution_patch_heads(
    clean_acts: dict[str, Tensor], clean_grads: dict[str, Tensor], corrupt_acts: dict[str, Tensor], n_layers: int
) -> Float[Tensor, "layer head"]:
    """AP estimate of the change in the metric from corrupting each head's output (hook_z), at every position."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    scores = []
    for layer in range(n_layers):
        name = f"blocks.{layer}.attn.hook_z"
        delta = corrupt_acts[name] - clean_acts[name]
        scores.append(einops.einsum(delta, clean_grads[name], "batch pos head d_head, batch pos head d_head -> head"))
    return t.stack(scores)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_attribution_patch_heads(attribution_patch_heads, model)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

For each layer, `(corrupt_acts[name] - clean_acts[name]) * clean_grads[name]` has shape `(batch, pos, head, d_head)`. Sum over every dimension except `head`, then stack the layers.
</details>

The test uses two independent checks: a small hand-made example, and the identity that the AP scores of all heads in a layer must add up to the AP score of the whole attention layer's output (`hook_attn_out`). That identity holds because the attention output is (up to a constant bias) the sum of the heads' outputs, so the chain rule splits its gradient exactly.

Now let's compute AP scores for every head on IOI, and time it.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

z_names = [f"blocks.{layer}.attn.hook_z" for layer in range(model.cfg.n_layers)]

start = time.time()
clean_acts, clean_grads = get_cache_fwd_and_bwd_chunked(model, ioi, z_names)
corrupt_acts = get_activations(model, ioi.corrupt_tokens, z_names)
ap_heads = attribution_patch_heads(clean_acts, clean_grads, corrupt_acts, model.cfg.n_layers).cpu()
ap_time = time.time() - start
print(f"Attribution patching: 144 head scores in {ap_time:.2f}s")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - exact activation patching for every head

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

To know whether AP is any good we need the ground truth. Implement `exact_patch_heads`: for every head, run the model on the clean prompts with that head's `hook_z` replaced (at every position) by its value from the corrupted run, and record the change in the mean metric relative to the clean run. That's 144 forward passes.

This is the same activation patching you did in [1.4.1], but in the noising direction (corrupt patched into clean). The test compares against TransformerLens's own `patching.get_act_patch_attn_head_out_all_pos`, run in the same direction.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.no_grad()
def exact_patch_heads(model: HookedTransformer, task: Task, corrupt_acts: dict[str, Tensor]) -> Float[Tensor, "layer head"]:
    """Exact change in the mean metric from patching each head's corrupted hook_z into the clean run."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    clean_value = task.metric(model(task.clean_tokens), task.answers).mean()
    results = t.zeros(model.cfg.n_layers, model.cfg.n_heads)
    for layer in range(model.cfg.n_layers):
        name = f"blocks.{layer}.attn.hook_z"
        for head in range(model.cfg.n_heads):

            def patch_head(z: Tensor, hook) -> Tensor:
                z[:, :, head] = corrupt_acts[name][:, :, head]
                return z

            logits = model.run_with_hooks(task.clean_tokens, fwd_hooks=[(name, patch_head)])
            results[layer, head] = task.metric(logits, task.answers).mean() - clean_value
    return results
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_exact_patch_heads(exact_patch_heads, model, make_ioi_task(model, 8, seed=5))
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Loop over layers and heads. For each, define a hook function on `blocks.{layer}.attn.hook_z` that sets `z[:, :, head] = corrupt_acts[name][:, :, head]` and returns `z`, and run `model.run_with_hooks(task.clean_tokens, fwd_hooks=[(name, hook_fn)])`.
</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

start = time.time()
exact_heads = exact_patch_heads(model, ioi, corrupt_acts)
exact_time = time.time() - start
print(f"Exact patching: 144 head scores in {exact_time:.2f}s ({exact_time / ap_time:.0f}x slower than AP)")

pearson = t.corrcoef(t.stack([exact_heads.flatten(), ap_heads.flatten()]))[0, 1].item()
print(f"Pearson correlation between AP and exact: {pearson:.3f}\n")

head_labels = [f"{layer}.{head}" for layer in range(model.cfg.n_layers) for head in range(model.cfg.n_heads)]
print("Largest disagreements:")
for i in (ap_heads - exact_heads).abs().flatten().topk(4).indices.tolist():
    print(f"  head {head_labels[i]:>5}: exact {exact_heads.flatten()[i]:+.3f}, AP {ap_heads.flatten()[i]:+.3f}")
utils.plot_head_heatmaps(
    {"Exact patching (144 forward passes)": exact_heads, "Attribution patching (1 fwd + 1 bwd)": ap_heads},
    title="IOI: change in logit difference when each head's output is corrupted",
)
utils.plot_scatter(
    exact_heads,
    {"attention heads": ap_heads},
    title=f"Attribution patching vs exact patching, per head (r = {pearson:.2f})",
    labels=head_labels,
    n_labels=8,
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/heads_heatmaps.png" width="900">
<img src="media/heads_scatter.png" width="450">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

The timings depend on your hardware; the rest should match closely.

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Attribution patching: 144 head scores in 0.39s
Exact patching: 144 head scores in 10.38s (27x slower than AP)
Pearson correlation between AP and exact: 0.938

Largest disagreements:
  head   5.5: exact -0.855, AP -0.175
  head   8.6: exact -1.189, AP -1.521
  head   9.9: exact -0.465, AP -0.780
  head  4.11: exact -0.287, AP +0.024</pre>

</details>

<details>
<summary>Help - how should I interpret these plots?</summary>

**The two heatmaps are nearly indistinguishable.** The same heads light up, with the same signs, and you should recognise most of them from [1.4.1]:

- The **negative name mover heads** 10.7 and 11.10 have the largest *positive* effects: they normally push *against* the IO, so corrupting them raises the logit difference.
- The **S-inhibition heads** 7.9, 8.6 and 8.10 (and, more weakly, 7.3), the **induction head** 5.5 and the **duplicate token head** 3.0 all have negative effects: they compute and pass on the "which name is duplicated?" signal.
- Among the **name movers**, 9.9 has a negative effect as expected. Head 9.6, which Wang et al. also call a name mover, has a positive effect here in *both* heatmaps. We'll leave that as an anomaly to chase in the bonus section; the point for now is that AP reproduces it.

**The scatter plot is close to the diagonal**, with a correlation of 0.94, and AP got there roughly 30 times faster than exact patching (the exact ratio depends on your hardware). With 144 heads that's a nice speed-up; with every neuron or every edge of a large model it's the difference between feasible and infeasible, because exact patching scales with the number of components and AP doesn't.

**But it's not perfect.** Look at the disagreements: AP underestimates the induction head 5.5 almost fivefold (−0.18 vs −0.86), overestimates the S-inhibition head 8.6 and the name mover 9.9 by 30-70%, and gets the sign of the previous-token head 4.11 wrong (though both values are small). A first-order approximation can only be as good as the model is linear along the way from clean to corrupt, and in 2️⃣ we'll look directly at how far from linear it can be.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - rank correlation

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 5-10 minutes on this exercise.
> ```

Pearson correlation is dominated by the few largest values. When we use attribution scores to *choose* components (keep the top 10 heads, the top 1,000 edges), what matters is whether the ordering is right, so the more relevant statistic is a rank correlation. Implement `spearman_correlation`: replace each value by its rank within its own tensor (0 for the smallest, $n-1$ for the largest), then compute the Pearson correlation of the ranks. You can ignore ties.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def spearman_correlation(x: Tensor, y: Tensor) -> float:
    """Spearman rank correlation between two tensors of the same number of elements (ties ignored)."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    rank_x = x.flatten().argsort().argsort().float()
    rank_y = y.flatten().argsort().argsort().float()
    return t.corrcoef(t.stack([rank_x, rank_y]))[0, 1].item()
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_spearman_correlation(spearman_correlation)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`v.argsort()` gives the indices that would sort `v`; applying `argsort()` again to that gives each element's rank. Then `t.corrcoef(t.stack([rank_x, rank_y]))[0, 1]` is the Pearson correlation of the ranks.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - do we need both halves of the formula?

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15 minutes on this exercise.
> ```

AP multiplies two things: how much a component's activation *changes* between clean and corrupt, and how sensitive the metric is to that component (the gradient). Is the product really needed, or would either half alone rank heads just as well? And is the agreement with exact patching specific to pairing each head's activation difference with *its own* gradient? Implement three baselines, each returning a `(n_layers, n_heads)` tensor:

- `"activation change only"`: for each head, the norm of $z_{\text{corrupt}} - z_{\text{clean}}$ over `d_head`, summed over batch and position. This is the "which heads behave differently on the corrupted prompts?" heuristic.
- `"gradient only"`: the norm of $\partial M / \partial z$ over `d_head`, summed over batch and position. This is plain gradient saliency.
- `"AP, shuffled pairing"`: the AP formula, but with each head's activation difference dotted with the gradient of a *different*, randomly chosen head (use a random permutation of the 144 heads, seeded with `seed`). This should destroy any agreement, if AP's agreement with exact patching is not an accident.

Since the first two baselines are unsigned, we'll compare everything by the Spearman correlation between $|\text{score}|$ and $|\text{exact effect}|$: "does this method rank heads by importance?".
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def baseline_head_scores(
    clean_acts: dict[str, Tensor], clean_grads: dict[str, Tensor], corrupt_acts: dict[str, Tensor], n_layers: int, seed: int = 0
) -> dict[str, Float[Tensor, "layer head"]]:
    """Returns the activation-only, gradient-only and shuffled-pairing baselines described above."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    names = [f"blocks.{layer}.attn.hook_z" for layer in range(n_layers)]
    delta = t.stack([corrupt_acts[name] - clean_acts[name] for name in names])  # (layer, batch, pos, head, d_head)
    grads = t.stack([clean_grads[name] for name in names])
    n_heads = delta.shape[3]

    delta_flat = einops.rearrange(delta, "layer batch pos head d -> batch pos (layer head) d")
    grads_flat = einops.rearrange(grads, "layer batch pos head d -> batch pos (layer head) d")
    perm = t.randperm(n_layers * n_heads, generator=t.Generator().manual_seed(seed)).to(delta.device)
    shuffled = einops.einsum(delta_flat, grads_flat[:, :, perm], "batch pos lh d, batch pos lh d -> lh")

    return {
        "activation change only": delta.norm(dim=-1).sum(dim=(1, 2)),
        "gradient only": grads.norm(dim=-1).sum(dim=(1, 2)),
        "AP, shuffled pairing": shuffled.reshape(n_layers, n_heads),
    }
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_baseline_head_scores(baseline_head_scores)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Stack the per-layer tensors into one `(layer, batch, pos, head, d_head)` tensor. The first two baselines are a `.norm(dim=-1)` followed by a sum over the batch and position dimensions. For the shuffled pairing, flatten `(layer, head)` into one dimension of size 144, index the gradients' flattened dimension with a permutation, and do the AP dot product.
</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

baselines = baseline_head_scores(clean_acts, clean_grads, corrupt_acts, model.cfg.n_layers)
rank_agreement = {"attribution patching": spearman_correlation(ap_heads.abs(), exact_heads.abs())}
for name, scores in baselines.items():
    rank_agreement[name] = spearman_correlation(scores.cpu().abs(), exact_heads.abs())
for name, rho in rank_agreement.items():
    print(f"{name:>24}: Spearman with |exact| = {rho:.3f}")
print("\nSigned agreement (Pearson correlation with the signed exact effect):")
for name, scores in [("attribution patching", ap_heads), ("AP, shuffled pairing", baselines["AP, shuffled pairing"].cpu())]:
    print(f"{name:>24}: {t.corrcoef(t.stack([scores.flatten(), exact_heads.flatten()]))[0, 1]:+.3f}")

utils.plot_bars(
    rank_agreement,
    title="Which scores rank heads by the size of their exact patching effect?",
    ylabel="Spearman correlation with |exact effect|",
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/heads_baselines.png" width="650">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">    attribution patching: Spearman with |exact| = 0.915
  activation change only: Spearman with |exact| = 0.479
           gradient only: Spearman with |exact| = -0.034
    AP, shuffled pairing: Spearman with |exact| = 0.411

Signed agreement (Pearson correlation with the signed exact effect):
    attribution patching: +0.938
    AP, shuffled pairing: -0.026</pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

**The gradient alone is useless** (Spearman −0.03). A large gradient says the metric is sensitive to this head's output *in some direction*, but says nothing about whether the corruption actually moves the head's output in that direction, or at all.

**The activation change alone is moderately informative** (0.48): heads whose output changes a lot when you swap the names tend to be heads that process names. But it's well short of AP (0.92), because plenty of heads change a lot without the change mattering for the logit difference.

**The shuffled pairing is a control that only half fails, and it's worth understanding why.** By the unsigned measure it still gets 0.41. That's because the shuffled score of head $i$ is still proportional to the size of head $i$'s own activation change, so it inherits most of the "activation change only" signal; the shuffled gradient just adds noise. On the *signed* comparison, which also asks whether the effect helps or hurts, the shuffled pairing collapses to −0.03, while AP keeps 0.94.

So both halves of the formula matter: the activation change says *how far* the component moves, and its own gradient says *whether that movement matters, and which way*. Pairing each head's change with its own gradient is what takes us from 0.48 to 0.92 and gets the signs right.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 1️⃣

- Attribution patching is a first-order Taylor expansion of activation patching: $\Delta M \approx (a_{\text{corrupt}} - a_{\text{clean}}) \cdot \nabla_a M$. One forward and one backward pass give an estimate for every component at once.
- For the attention heads on IOI it works well: correlation 0.94 with exact patching, Spearman 0.92 on the size of the effects, and about 30 times faster even for just 144 components.
- Both halves are needed. Gradient saliency on its own has no relationship with the exact effects; the activation change alone gets about half way; and pairing changes with the wrong gradients destroys the signed agreement.
- Even here, a few heads (the induction head 5.5, the S-inhibition head 8.6, the name mover 9.9) are estimated poorly. In the next section we'll find components where AP is much worse than "a bit off", and work out why.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 2️⃣ Where the Linear Approximation Breaks
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Patching the residual stream

Head outputs were a friendly case. Let's try the residual stream, the first thing you patched in [1.4.1]: for every layer and every position, corrupt `resid_pre` at that (layer, position) and see what happens to the logit difference.

This is a much harsher test, for a simple reason. At the name positions in early layers, the residual stream *is* basically the name: corrupting it replaces `" Mary"` with `" Laura"` wholesale. That is not a small perturbation of anything, and a first-order Taylor expansion has no reason to be accurate for it.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - attribution patching for the residual stream

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 5 minutes on this exercise.
> ```

Implement `attribution_patch_resid`, which takes activations and gradients at `blocks.{layer}.hook_resid_pre` for every layer and returns a `(n_layers, seq)` tensor of AP scores, one per (layer, position). It's the same formula as before, summed over batch and `d_model` but *not* over position.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def attribution_patch_resid(
    clean_acts: dict[str, Tensor], clean_grads: dict[str, Tensor], corrupt_acts: dict[str, Tensor], n_layers: int
) -> Float[Tensor, "layer pos"]:
    """AP estimate of the change in the metric from corrupting resid_pre at each (layer, position)."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    scores = []
    for layer in range(n_layers):
        name = f"blocks.{layer}.hook_resid_pre"
        delta = corrupt_acts[name] - clean_acts[name]
        scores.append(einops.einsum(delta, clean_grads[name], "batch pos d_model, batch pos d_model -> pos"))
    return t.stack(scores)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_attribution_patch_resid(attribution_patch_resid)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - exact patching for the residual stream

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵⚪⚪⚪
>
> You should spend up to 5-10 minutes on this exercise.
> ```

And the ground truth: implement `exact_patch_resid`, which for every (layer, position) runs the clean prompts with `resid_pre` at that layer and position replaced by its corrupted value, and returns the change in the mean metric. That's $12 \times 15 = 180$ forward passes. (The test checks you against TransformerLens's `patching.get_act_patch_resid_pre`.)
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.no_grad()
def exact_patch_resid(model: HookedTransformer, task: Task, corrupt_acts: dict[str, Tensor]) -> Float[Tensor, "layer pos"]:
    """Exact change in the mean metric from patching corrupted resid_pre into the clean run, per (layer, position)."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    clean_value = task.metric(model(task.clean_tokens), task.answers).mean()
    seq_len = task.clean_tokens.shape[1]
    results = t.zeros(model.cfg.n_layers, seq_len)
    for layer in range(model.cfg.n_layers):
        name = f"blocks.{layer}.hook_resid_pre"
        for pos in range(seq_len):

            def patch_resid(resid: Tensor, hook) -> Tensor:
                resid[:, pos] = corrupt_acts[name][:, pos]
                return resid

            logits = model.run_with_hooks(task.clean_tokens, fwd_hooks=[(name, patch_resid)])
            results[layer, pos] = task.metric(logits, task.answers).mean() - clean_value
    return results
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_exact_patch_resid(exact_patch_resid, model, make_ioi_task(model, 8, seed=5))
# END HIDE

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

resid_names = [f"blocks.{layer}.hook_resid_pre" for layer in range(model.cfg.n_layers)]
clean_resid, clean_resid_grads = get_cache_fwd_and_bwd_chunked(model, ioi, resid_names)
corrupt_resid = get_activations(model, ioi.corrupt_tokens, resid_names)

ap_resid = attribution_patch_resid(clean_resid, clean_resid_grads, corrupt_resid, model.cfg.n_layers).cpu()
exact_resid = exact_patch_resid(model, ioi, corrupt_resid)

print(f"Pearson:  {t.corrcoef(t.stack([ap_resid.flatten(), exact_resid.flatten()]))[0, 1]:.3f}")
print(f"Spearman: {spearman_correlation(ap_resid, exact_resid):.3f}\n")
print("Layer 0 at the name positions:")
for pos in [2, 4, 10]:
    label = utils.IOI_POSITION_LABELS[pos]
    print(f"  {label:>6}: exact {exact_resid[0, pos]:+.2f}, AP {ap_resid[0, pos]:+.2f}")

utils.plot_layer_pos_heatmaps(
    {"Exact patching (180 forward passes)": exact_resid, "Attribution patching": ap_resid},
    utils.IOI_POSITION_LABELS,
    title="IOI: change in logit difference when resid_pre is corrupted at one (layer, position)",
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/resid_heatmaps.png" width="950">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Pearson:  0.108
Spearman: 0.344

Layer 0 at the name positions:
  name 1: exact -4.67, AP +0.73
  name 2: exact -3.91, AP +0.55
      S2: exact -3.28, AP -0.18</pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

**The exact heatmap is the familiar picture from [1.4.1].** Up to about layer 9, the information the model needs sits at the three name positions; from layer 8 onwards it has been moved to the final position (`END`, the last `to`), where the answer is produced. Corrupting a name position in an early layer is devastating: at layer 0, corrupting the first name lowers the logit difference by 4.67, *more* than the whole clean logit difference of 3.18. That's not surprising once you think about it: the clean IO name is then simply no longer in the prompt, so the model has no reason to say it.

**AP gets the late layers right and the name positions badly wrong.** At the final position in layers 8-11, AP's estimates are close to the exact ones. At S2 it gets the middle layers roughly right but misses most of the effect in layers 0-4. And at the name positions it gets the **sign** wrong: at every layer from 0 to 9 for the first name, and at all of them but layer 6 for the second. At layer 0 it predicts that corrupting the first name would *help* the model (+0.73), when in fact it destroys the behaviour (−4.67). Overall the correlation is 0.11.

Why the difference? At the final position in late layers, the corruption nudges the residual stream, and only a few mostly-linear steps remain before the logits. At a name position in an early layer, the corruption replaces one name with another: the "perturbation" is as big as the vector itself, and it has to pass through a dozen layers of attention and MLPs. There's no reason for a first-order expansion to survive that. Let's look at exactly how it fails.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## The one-dimensional picture

To see what's going on, let's stop looking at two numbers (exact and AP) and look at the whole function in between. Fix one component, and define the straight-line path from its clean activation to its corrupted activation,

$$
a(\alpha) = a_{\text{clean}} + \alpha \left(a_{\text{corrupt}} - a_{\text{clean}}\right), \qquad \alpha \in [0, 1],
$$

and the metric along that path, relative to the clean run,

$$
f(\alpha) = M\big(a(\alpha)\big) - M(a_{\text{clean}}).
$$

Then $f(0) = 0$, the exact patching effect is $f(1)$, and by the chain rule

$$
f'(\alpha) = \left.\frac{\partial M}{\partial a}\right|_{a(\alpha)} \cdot \left(a_{\text{corrupt}} - a_{\text{clean}}\right),
$$

so the AP score is exactly $f'(0)$. In other words, **attribution patching draws the tangent line to $f$ at $\alpha = 0$, and reads off its value at $\alpha = 1$**. It's exact if $f$ is a straight line, and it can be arbitrarily wrong if $f$ curves.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - the metric along the path

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Implement `metric_along_path`. A component is specified by a hook name and an `index` into that hook's activation, such as `np.s_[:, :, 7]` for head 7 of a `hook_z` or `hook_q` activation (all prompts and positions), or `np.s_[:, 10]` for position 10 of a residual stream or MLP output. (`np.s_` is just a convenient way to write a tuple of slices: `np.s_[:, 10]` is `(slice(None), 10)`, and you can use it to index a torch tensor with `act[index]`.)

For each `alpha` in `alphas`, run the model on the clean prompts with a hook that replaces `act[index]` by `clean[index] + alpha * (corrupt[index] - clean[index])`, and return two lists:

- `values`: $f(\alpha)$, the mean metric minus the clean mean metric;
- `slopes`: $f'(\alpha) = \nabla_a M \cdot (a_{\text{corrupt}} - a_{\text{clean}})$, where the gradient is taken at the interpolated activation, and the dot product is over the indexed part only.

To get the gradient, make the interpolated activation a leaf that requires grad inside your hook (clone the activation, write the interpolated values into it, call `.requires_grad_(True)`, keep a reference to it, and return it), then use `t.autograd.grad(metric, interpolated_act)`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def metric_along_path(
    model: HookedTransformer,
    task: Task,
    hook_name: str,
    index: tuple,
    alphas: list[float],
    clean_acts: dict[str, Tensor],
    corrupt_acts: dict[str, Tensor],
) -> tuple[list[float], list[float]]:
    """
    Returns (values, slopes): f(alpha) and f'(alpha) for each alpha, where f is the change in the mean metric when
    act[index] at `hook_name` is set to clean + alpha * (corrupt - clean) in a run on the clean prompts.
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    clean_value = mean_metric(model, task, task.clean_tokens)
    clean, corrupt = clean_acts[hook_name], corrupt_acts[hook_name]
    delta = corrupt[index] - clean[index]
    values, slopes = [], []
    for alpha in alphas:
        stored = {}

        def interpolate(act: Tensor, hook) -> Tensor:
            act = act.clone()
            act[index] = clean[index] + alpha * delta
            act.requires_grad_(True)
            stored["act"] = act
            return act

        logits = model.run_with_hooks(task.clean_tokens, fwd_hooks=[(hook_name, interpolate)])
        value = task.metric(logits, task.answers).mean()
        (grad,) = t.autograd.grad(value, stored["act"])
        values.append(value.item() - clean_value)
        slopes.append((grad[index] * delta).sum().item())
    return values, slopes
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_metric_along_path(metric_along_path, model, make_ioi_task(model, 8, seed=5))
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Compute `delta = corrupt[index] - clean[index]` once. Inside the loop, define the hook as a closure over `alpha`, run `model.run_with_hooks(task.clean_tokens, fwd_hooks=[(hook_name, hook_fn)])`, compute the mean metric, and call `t.autograd.grad(value, stored_act)`, which returns a tuple with one gradient. The slope is `(grad[index] * delta).sum()`.
</details>

<details>
<summary>Help - I get <code>RuntimeError: One of the differentiated Tensors appears to not have been used in the graph</code>.</summary>

The tensor you differentiate with respect to must be the one the rest of the forward pass is computed from. Your hook has to *return* exactly the tensor you stored (the clone you called `requires_grad_(True)` on). If you store one tensor and return another, the stored one isn't part of the graph.
</details>

The test checks two identities: $f(1)$ must equal the exact patching effect of the component, and $f'(0)$ must equal its AP score.

We'll look at four components, chosen to show four different behaviours:

- **Head 10.7's output**, a negative name mover head with one of the largest effects in the heatmap above.
- **Head 6.9's query**, at every position. Head 6.9 is an induction head (in Wang et al.'s terms), which attends from S2 to the token after S1.
- **MLP0's output at S2.** MLP0 in GPT-2 Small acts like an extension of the embedding, so this is close to the identity of the S2 token.
- **The residual stream before layer 0 at the first name**, i.e. the token embedding of the first name.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

COMPONENTS = {
    "head 10.7 output": ("blocks.10.attn.hook_z", np.s_[:, :, 7]),
    "head 6.9 query": ("blocks.6.attn.hook_q", np.s_[:, :, 9]),
    "MLP0 output at S2": ("blocks.0.hook_mlp_out", np.s_[:, 10]),
    "resid_pre 0 at name 1": ("blocks.0.hook_resid_pre", np.s_[:, 2]),
}
ioi_small = ioi[:32]  # metric_along_path does a backward pass on all prompts at once, so we use a subset
component_hooks = sorted({hook_name for hook_name, _ in COMPONENTS.values()})
clean_component_acts = get_activations(model, ioi_small.clean_tokens, component_hooks)
corrupt_component_acts = get_activations(model, ioi_small.corrupt_tokens, component_hooks)

alphas = [i / 20 for i in range(21)]
paths = {}
for name, (hook_name, index) in COMPONENTS.items():
    values, slopes = metric_along_path(
        model, ioi_small, hook_name, index, alphas, clean_component_acts, corrupt_component_acts
    )
    paths[name] = {"alphas": alphas, "values": values, "ap": slopes[0], "exact": values[-1]}
    print(f"{name:>22}: exact {values[-1]:+.3f}   AP {slopes[0]:+.3f}")

utils.plot_paths(paths, title="The metric along the straight line from clean (α=0) to corrupted (α=1) activation")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/paths.png" width="1000">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">      head 10.7 output: exact +1.336   AP +1.031
        head 6.9 query: exact -0.485   AP -1.366
     MLP0 output at S2: exact -3.301   AP -0.116
 resid_pre 0 at name 1: exact -4.837   AP +0.695</pre>

(These are for the 32-prompt subset, so they differ slightly from the full-dataset numbers above.)
</details>

<details>
<summary>Help - how should I interpret these four curves?</summary>

In each panel the black curve is the truth, the red dashed line is what attribution patching assumes, and the black square is the exact patching effect. They show four different ways for a function to relate to its tangent.

**Head 10.7's output: nearly straight.** The curve bends slightly upwards, so the tangent underestimates the effect by about a quarter (1.03 vs 1.34), but the sign and rough size are right. Most head outputs look like this, which is why 1️⃣ went well.

**Head 6.9's query: saturating.** The metric falls steeply at first, then flattens out after about $\alpha = 0.4$: beyond that point, changing the query further does nothing. The tangent extrapolates the initial steep slope all the way to $\alpha = 1$, and overestimates the effect almost threefold (−1.37 vs −0.49). We'll see below that this is the attention softmax saturating.

**MLP0's output at S2: flat, then a cliff.** For the first third of the path almost nothing happens; then between $\alpha \approx 0.4$ and $0.9$ the logit difference collapses by more than 3. The gradient at the clean point is nearly zero, so AP reports almost nothing (−0.12 vs −3.30). This is a **false negative**: the method would tell you this component doesn't matter, when it's one of the most important in the model.

**The residual stream at the first name: wrong direction.** The metric first goes slightly *up* (to about +0.1 at $\alpha \approx 0.25$) before crashing to −4.84. The tangent at the clean point points up, so AP gets the **sign** wrong.

The general lesson: AP sees only the local slope at the clean activation. If the function bends anywhere between the clean and the corrupted activation, AP can be wrong by any amount, in either direction. Large patches (replacing a token's identity) and saturating nonlinearities (softmax, GELU, layernorm) are where the function bends.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Why attention saturates

Head 6.9's curve is the textbook attention-saturation failure, so it's worth being precise about it. An attention head's pattern is a softmax over scores $s_j = q \cdot k_j / \sqrt{d_{head}}$, and the derivative of a softmax probability with respect to its own score is

$$
\frac{\partial p_i}{\partial s_i} = p_i (1 - p_i).
$$

This is largest at $p_i = 1/2$ and goes to zero as $p_i \to 0$ or $p_i \to 1$. Moving the query along a straight line moves the scores along a straight line, but the probabilities follow an S-shaped (logistic-like) curve. So:

- If the head starts **saturated** ($p_i \approx 1$ or $\approx 0$ on the clean run), the gradient at the clean point is near zero even if a large change in the query would completely redirect the head. AP reports a **false negative**.
- If the head starts in the steep part of the curve, the gradient at the clean point is large, but the probability can only fall as far as zero. AP extrapolates the steep slope all the way to $\alpha = 1$ and **overestimates** the effect.

Head 6.9 is the second case. Let's watch its attention from S2 to the token after S1 as we move its query along the path:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

hook_name, index = COMPONENTS["head 6.9 query"]
s1_pos = (ioi_small.clean_tokens[:, :10] == ioi_small.clean_tokens[:, 10:11]).float().argmax(dim=1)  # position of S1
batch_idx = t.arange(len(ioi_small))
for alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    stored = {}

    def interpolate(q: Tensor, hook) -> Tensor:
        q[index] = clean_component_acts[hook_name][index] + alpha * (
            corrupt_component_acts[hook_name][index] - clean_component_acts[hook_name][index]
        )
        return q

    def store_pattern(pattern: Tensor, hook) -> None:
        stored["p"] = pattern[batch_idx, 9, 10, s1_pos + 1]  # head 9, query position S2, key position S1+1

    with t.no_grad():
        model.run_with_hooks(
            ioi_small.clean_tokens, fwd_hooks=[(hook_name, interpolate), ("blocks.6.attn.hook_pattern", store_pattern)]
        )
    print(f"alpha = {alpha:.1f}: head 6.9 attention from S2 to S1+1 = {stored['p'].mean():.3f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">alpha = 0.0: head 6.9 attention from S2 to S1+1 = 0.248
alpha = 0.1: head 6.9 attention from S2 to S1+1 = 0.173
alpha = 0.2: head 6.9 attention from S2 to S1+1 = 0.111
alpha = 0.3: head 6.9 attention from S2 to S1+1 = 0.068
alpha = 0.5: head 6.9 attention from S2 to S1+1 = 0.028
alpha = 0.7: head 6.9 attention from S2 to S1+1 = 0.014
alpha = 1.0: head 6.9 attention from S2 to S1+1 = 0.006</pre>

</details>

The attention to the token after S1 falls off roughly exponentially: it has more than halved by $\alpha = 0.2$ and is down to 0.03 by $\alpha = 0.5$. From there on the head has essentially stopped attending to that token, so moving its query further can't change anything, and that's exactly the plateau in the metric curve. The gradient at $\alpha = 0$ is measured where the attention is falling fastest, and the tangent line assumes it keeps falling at that rate all the way to $\alpha = 1$, which is impossible for a probability that's already near zero.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
This is the problem that **AtP\*** (Kramár et al., 2024) attacks with its "QK fix": instead of linearising through the softmax, it recomputes the attention pattern exactly for the patched query or key, and only linearises what comes after. You'll implement that in section [8.3]. Here we'll look at a more general fix.

## Integrated gradients for one component

By the fundamental theorem of calculus,

$$
f(1) = f(1) - f(0) = \int_0^1 f'(\alpha)\, d\alpha .
$$

AP approximates this integral by its integrand at a single point, $f'(0)$. The obvious improvement is to average the slope over several points along the path:

$$
\text{IG}_m = \frac{1}{m}\sum_{k=0}^{m-1} f'\!\left(\frac{k}{m}\right)
= \left(a_{\text{corrupt}} - a_{\text{clean}}\right) \cdot \frac{1}{m}\sum_{k=0}^{m-1} \left.\frac{\partial M}{\partial a}\right|_{a(k/m)} .
$$

This is a Riemann sum for the integral, so $\text{IG}_m \to f(1)$, the exact patching effect, as $m \to \infty$; and with $m = 1$ it's exactly attribution patching. In machine learning this is called **integrated gradients** (Sundararajan et al., 2017): "the activation difference times the gradient averaged along the path".
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - integrated gradients for a single component

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 5 minutes on this exercise.
> ```

Implement `integrated_gradients_component`, which returns $\text{IG}_m$ for one component, using `metric_along_path` with $\alpha_k = k/m$ for $k = 0, \dots, m-1$ (a left Riemann sum, so that $m = 1$ is AP).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def integrated_gradients_component(
    model: HookedTransformer,
    task: Task,
    hook_name: str,
    index: tuple,
    n_steps: int,
    clean_acts: dict[str, Tensor],
    corrupt_acts: dict[str, Tensor],
) -> float:
    """IG estimate of the patching effect of act[index] at hook_name, with n_steps left-Riemann steps."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    alphas = [k / n_steps for k in range(n_steps)]
    _, slopes = metric_along_path(model, task, hook_name, index, alphas, clean_acts, corrupt_acts)
    return sum(slopes) / n_steps
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_integrated_gradients_component(integrated_gradients_component, metric_along_path, model, make_ioi_task(model, 8, seed=5))
# END HIDE

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

step_counts = [1, 2, 4, 8, 16, 32]
ig_errors = {}
for name, (hook_name, index) in COMPONENTS.items():
    estimates = [
        integrated_gradients_component(model, ioi_small, hook_name, index, m, clean_component_acts, corrupt_component_acts)
        for m in step_counts
    ]
    ig_errors[name] = [abs(est - paths[name]["exact"]) for est in estimates]
    paths[name]["ig"] = estimates[step_counts.index(8)]
    print(f"{name:>22}: exact {paths[name]['exact']:+.3f}  IG: " + "  ".join(f"m={m}: {e:+.3f}" for m, e in zip(step_counts, estimates)))

utils.plot_ig_convergence(step_counts, ig_errors, title="Integrated gradients converge to the exact patching effect")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/ig_convergence.png" width="600">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">      head 10.7 output: exact +1.336  IG: m=1: +1.031  m=2: +1.189  m=4: +1.262  m=8: +1.299  m=16: +1.317  m=32: +1.327
        head 6.9 query: exact -0.485  IG: m=1: -1.366  m=2: -0.787  m=4: -0.642  m=8: -0.566  m=16: -0.526  m=32: -0.505
     MLP0 output at S2: exact -3.301  IG: m=1: -0.116  m=2: -2.320  m=4: -3.027  m=8: -3.259  m=16: -3.293  m=32: -3.298
 resid_pre 0 at name 1: exact -4.837  IG: m=1: +0.695  m=2: -6.104  m=4: -4.432  m=8: -4.648  m=16: -4.775  m=32: -4.805</pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

**Integrated gradients fixes all four failures.** With 8 steps, every estimate has the right sign and is within 0.2 of the exact effect, including the two components where AP was wrong in sign or by a factor of almost 30.

**The convergence is roughly first-order.** For the smooth curves (heads 10.7 and 6.9), the error halves every time $m$ doubles, as you'd expect from a left Riemann sum, whose error is $O(1/m)$. The two "cliff" curves start with much larger errors and behave erratically for small $m$: with $m = 2$ the residual-stream estimate *overshoots* to −6.10, because one of its two sample points ($\alpha = 0.5$) lands on the steepest part of the cliff and gets weighted as if it covered half the path. Once there are enough points to resolve the cliff, they converge at least as fast as the smooth curves.

**But there's a catch.** $\text{IG}_m$ for one component costs $m$ forward *and* backward passes, while exact patching of that component costs one forward pass. So integrated gradients *for a single component* is strictly worse than just patching it! It's useful here to understand what goes wrong with AP, not as a replacement for exact patching. The trick that makes integration cheap is to move *every* component along its path at the same time, so that $m$ backward passes give integrated gradients for everything at once. We'll do that in 4️⃣, and it will turn out to estimate something subtly different from single-component patching.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 2️⃣

- Attribution patching is the tangent line to the metric along the straight path from the clean to the corrupted activation. It's exact when that path is linear, and can be arbitrarily wrong when it isn't.
- It fails worst for **large patches**: the residual stream at the name positions and MLP0's output, where the corruption replaces a token's identity. There, AP misses huge effects entirely (MLP0 at S2: −0.12 vs −3.30) or gets the sign wrong (the first name: +0.70 vs −4.84).
- **Attention saturation** makes AP overestimate effects whose softmax runs out of probability to move (head 6.9's query: almost 3x too big). In heads that start out saturated it can hide effects entirely; we didn't meet a clean example here, but Kramár et al. (2024) document many. AtP* (section [8.3]) fixes this for queries and keys.
- Averaging the gradient along the path (integrated gradients) converges to the exact effect, and about 8 steps were enough for every component here. But doing it one component at a time costs more than exact patching.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 3️⃣ Edge Attribution Patching
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## From nodes to edges

So far we've asked "how much does this *component* matter?". Circuit discovery asks a finer question: "how much does this *connection* matter?". Does head 9.9 matter because of what its query reads from the S-inhibition heads, or because of what its value reads from the name tokens?

The residual stream makes connections easy to define. Every component reads from the residual stream, and the residual stream at any point is the sum of the outputs of everything that came before it (plus some constant biases):

$$
x_v = \sum_{u \text{ before } v} a_u + \text{biases}.
$$

So the input $x_v$ to a component $v$ splits into one contribution $a_u$ from each earlier component $u$, and we call each pair $(u \to v)$ an **edge**. To *patch an edge*, we replace $u$'s contribution to $v$'s input (and only to $v$'s input) with its corrupted value: $x_v \to x_v + (a_u^{\text{corrupt}} - a_u^{\text{clean}})$, while every other component still sees $u$'s clean output. (If you did path patching in [1.4.1], this is the same idea, for a single sender and receiver.)

We'll use the same graph as Hanna et al.'s EAP-IG code:

- **Senders** (things that write to the residual stream): the input embeddings (token plus position), each of the 144 attention heads, and each of the 12 MLPs. That's 157 senders.
- **Receivers** (things that read from it): the query, key and value input of each attention head (432 receivers), the input of each MLP (12), and the logits, i.e. the final residual stream (1). That's 445 receivers.
- There is an edge from $u$ to $v$ whenever $u$ writes before $v$ reads. Heads in layer $\ell$ read before any head in layer $\ell$ writes, so there are no edges within an attention layer; but MLP $\ell$ reads *after* the heads in layer $\ell$ have written, so those edges do exist.

Splitting a head's input into query, key and value lets the graph distinguish Q-, K- and V-composition from [1.2]'s "A Mathematical Framework" discussion. To patch or differentiate head $h$'s query input without touching any other head, we need each head to have its *own* copy of its input.

We also need hook points on every receiver's input, so we switch on two TransformerLens options now:

- `set_use_split_qkv_input(True)` gives every attention head its own copy of the residual stream as the input to its query, key and value projections, at hooks `blocks.{layer}.hook_q_input` (and `k`, `v`), each of shape `(batch, pos, n_heads, d_model)`;
- `set_use_hook_mlp_in(True)` adds a hook `blocks.{layer}.hook_mlp_in` on the input to each MLP.

Neither changes what the model computes, though the first makes forward and backward passes somewhat slower and hungrier, since every layer now carries `3 * n_heads` copies of the residual stream. That's why we only switch it on now.

(There's also a `set_use_attn_result` option, which caches each head's output in the residual stream basis, `hook_result`. We deliberately *don't* use it: TransformerLens implements it by materialising a `(batch, pos, n_heads, d_head, d_model)` tensor, which needs several GB even for our small batches. We'll compute head outputs from `hook_z` and $W_O$ instead.)

'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

model.set_use_split_qkv_input(True)
model.set_use_hook_mlp_in(True)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Here are the names and index conventions we'll use. Senders are ordered `input, a0.0, ..., a0.11, m0, a1.0, ..., m11`; receivers are ordered layer by layer as `a{layer}.{head}<q>` for all heads, then `<k>`, then `<v>`, then `m{layer}`, and finally `logits`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

N_LAYERS, N_HEADS = 12, 12  # GPT-2 Small

SENDER_NAMES = ["input"] + [
    name for layer in range(N_LAYERS) for name in [f"a{layer}.{head}" for head in range(N_HEADS)] + [f"m{layer}"]
]
RECEIVER_NAMES = [
    name
    for layer in range(N_LAYERS)
    for name in [f"a{layer}.{head}<{qkv}>" for qkv in "qkv" for head in range(N_HEADS)] + [f"m{layer}"]
] + ["logits"]
N_SENDERS, N_RECEIVERS = len(SENDER_NAMES), len(RECEIVER_NAMES)


def head_sender(layer: int, head: int) -> int:
    return 1 + layer * (N_HEADS + 1) + head


def mlp_sender(layer: int) -> int:
    return 1 + layer * (N_HEADS + 1) + N_HEADS


def head_receiver(layer: int, qkv: str, head: int) -> int:
    return layer * (3 * N_HEADS + 1) + "qkv".index(qkv) * N_HEADS + head


def mlp_receiver(layer: int) -> int:
    return layer * (3 * N_HEADS + 1) + 3 * N_HEADS


LOGITS_RECEIVER = N_LAYERS * (3 * N_HEADS + 1)

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

assert SENDER_NAMES[head_sender(9, 9)] == "a9.9" and SENDER_NAMES[mlp_sender(3)] == "m3"
assert RECEIVER_NAMES[head_receiver(5, "k", 5)] == "a5.5<k>" and RECEIVER_NAMES[mlp_receiver(0)] == "m0"
assert RECEIVER_NAMES[LOGITS_RECEIVER] == "logits"
print(f"{N_SENDERS} senders, {N_RECEIVERS} receivers")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - build the edge mask

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement `build_edge_mask`, returning a boolean tensor of shape `(N_SENDERS, N_RECEIVERS)` which is `True` exactly where an edge exists.

A clean way to do this is to give every sender a "write time" and every receiver a "read time", and keep the pairs where the write comes strictly before the read. For example: the input writes at time 0; the heads in layer $\ell$ read at time $2\ell + 1$ and write at time $2\ell + 1$; MLP $\ell$ reads and writes at time $2\ell + 2$; the logits read at time $2L + 1$. (A head can't send to another head in the same layer because they share the time $2\ell+1$.)

You should get 32,491 edges.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def build_edge_mask() -> Bool[Tensor, "sender receiver"]:
    """True where sender u writes to the residual stream before receiver v reads from it."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    write_time = [0]
    for layer in range(N_LAYERS):
        write_time += [2 * layer + 1] * N_HEADS + [2 * layer + 2]
    read_time = []
    for layer in range(N_LAYERS):
        read_time += [2 * layer + 1] * (3 * N_HEADS) + [2 * layer + 2]
    read_time += [2 * N_LAYERS + 1]
    return t.tensor(write_time)[:, None] < t.tensor(read_time)[None, :]
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_build_edge_mask(build_edge_mask, SENDER_NAMES, RECEIVER_NAMES)
# END HIDE

EDGE_MASK = build_edge_mask()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Build two lists, `write_time` of length 157 and `read_time` of length 445, following the orderings above. Then `t.tensor(write_time)[:, None] < t.tensor(read_time)[None, :]` broadcasts to the full mask.
</details>

<details>
<summary>Where does 32,491 come from?</summary>

The input sends to all 445 receivers. Each head in layer $\ell$ sends to the $3 \times 12$ q/k/v inputs of every later layer, to MLPs $\ell, \dots, 11$, and to the logits: $36(11 - \ell) + (12 - \ell) + 1$ edges. MLP $\ell$ sends to $36(11-\ell) + (11 - \ell) + 1$. Summing: $445 + 12 \sum_{\ell}(36(11-\ell) + 13 - \ell) + \sum_\ell (37(11-\ell) + 1) = 445 + 29{,}592 + 2{,}454 = 32{,}491$.
</details>

To do exact patching on every one of these edges we'd need 32,491 forward passes (per batch of prompts). Let's see how EAP gets all of them from one.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## EAP: one activation difference per sender, one gradient per receiver

Patching the edge $u \to v$ changes $v$'s input by $\Delta a_u = a_u^{\text{corrupt}} - a_u^{\text{clean}}$ and nothing else. So the first-order Taylor expansion from 1️⃣ gives

$$
\Delta M_{\text{EAP}}(u \to v) = \Delta a_u \cdot \left.\frac{\partial M}{\partial x_v}\right|_{\text{clean run}},
$$

with the dot product summed over batch, position and `d_model` (the edge is a connection at every position, and each position of $a_u$ only feeds the same position of $x_v$). This is **edge attribution patching** (Syed, Rager & Conmy, 2023).

The beautiful thing is how the computation factorises. $\Delta a_u$ depends only on the sender, and the gradient $g_v = \partial M / \partial x_v$ depends only on the receiver. So we need:

- the outputs of all 157 senders on the clean and corrupted prompts: two forward passes;
- the gradient of the metric with respect to all 445 receiver inputs on the clean prompts: one backward pass;
- and then *every* edge score at once, as a single einsum: `scores[u, v] = sum over (batch, pos, d) of delta[batch, pos, u, d] * grads[batch, pos, v, d]`.

The tensors are big: for 16 prompts of 15 tokens, the receiver gradients are $16 \times 15 \times 445 \times 768$ floats, about 330MB. That's why `Task` has a `chunks` method: we'll process prompts in mini-batches and add up the scores.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - sender activations

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement `sender_activations`, which runs the model on `tokens` and returns a tensor of shape `(batch, pos, N_SENDERS, d_model)` holding each sender's output in the residual stream, in the sender order above:

- `input`: the embeddings, i.e. `blocks.0.hook_resid_pre` (token plus positional embedding);
- head $h$ in layer $\ell$: $z_h W_O^h$, computed from `blocks.{layer}.attn.hook_z` and `model.W_O[layer]` (shape `(n_heads, d_head, d_model)`), *without* the bias $b_O$, which is the same on every input and so never differs between clean and corrupted runs;
- MLP $\ell$: `blocks.{layer}.hook_mlp_out`.

The test checks that your senders add up (together with the attention output biases) to the final residual stream.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.no_grad()
def sender_activations(model: HookedTransformer, tokens: Int[Tensor, "batch seq"]) -> Float[Tensor, "batch pos sender d_model"]:
    """Outputs of every sender (input embedding, each head, each MLP) in the residual stream basis."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    names = ["blocks.0.hook_resid_pre"]
    for layer in range(model.cfg.n_layers):
        names += [f"blocks.{layer}.attn.hook_z", f"blocks.{layer}.hook_mlp_out"]
    _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in names)

    outputs = [cache["blocks.0.hook_resid_pre"][:, :, None]]
    for layer in range(model.cfg.n_layers):
        z = cache[f"blocks.{layer}.attn.hook_z"]
        outputs.append(einops.einsum(z, model.W_O[layer], "batch pos head d_head, head d_head d_model -> batch pos head d_model"))
        outputs.append(cache[f"blocks.{layer}.hook_mlp_out"][:, :, None])
    return t.cat(outputs, dim=2)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_sender_activations(sender_activations, model)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Use `run_with_cache` with a `names_filter` for the hooks you need. For each layer, `einops.einsum(z, model.W_O[layer], "batch pos head d_head, head d_head d_model -> batch pos head d_model")` gives the per-head outputs, and `cache[f"blocks.{layer}.hook_mlp_out"][:, :, None]` gives the MLP output with a sender dimension of size 1. Concatenate everything along dimension 2.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - receiver gradients

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Implement `receiver_gradients`, which runs the model on `tokens`, backpropagates from `metric_fn(logits)`, and returns the gradients with respect to every receiver's input, as a tensor of shape `(batch, pos, N_RECEIVERS, d_model)` in the receiver order above. The receiver inputs live at these hooks:

- `blocks.{layer}.hook_q_input`, `hook_k_input`, `hook_v_input`, each of shape `(batch, pos, n_heads, d_model)`, for the heads' query, key and value inputs;
- `blocks.{layer}.hook_mlp_in`, shape `(batch, pos, d_model)`, for the MLPs;
- `blocks.{N_LAYERS - 1}.hook_resid_post`, shape `(batch, pos, d_model)`, for the logits.

To save memory, preallocate the output tensor with `t.zeros` and have your backward hooks write each gradient straight into the right slice, using `hook.layer()` (the layer number) and `hook.name` to work out which slice. Use the `hook_embed` trick from `get_cache_fwd_and_bwd` to make the forward pass differentiable.

The optional `input_embed` argument isn't needed until 4️⃣: if it's given, your `hook_embed` hook should *replace* the token embeddings with it (`input_embed.clone().requires_grad_(True)`), so the model runs on those embeddings instead of the ones for `tokens`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def receiver_gradients(
    model: HookedTransformer,
    tokens: Int[Tensor, "batch seq"],
    metric_fn: Callable[[Tensor], Tensor],
    input_embed: Float[Tensor, "batch seq d_model"] | None = None,
) -> Float[Tensor, "batch pos receiver d_model"]:
    """Gradient of metric_fn(logits) with respect to the input of every receiver."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    batch, seq = tokens.shape
    grads = t.zeros(batch, seq, N_RECEIVERS, model.cfg.d_model, device=tokens.device)

    def store_grad(grad: Tensor, hook) -> None:
        layer, hook_type = hook.layer(), hook.name.split(".")[-1]
        if hook_type == "hook_resid_post":
            grads[:, :, LOGITS_RECEIVER] = grad
        elif hook_type == "hook_mlp_in":
            grads[:, :, mlp_receiver(layer)] = grad
        else:
            qkv = hook_type[len("hook_")]  # "q", "k" or "v"
            start = head_receiver(layer, qkv, 0)
            grads[:, :, start : start + N_HEADS] = grad

    def embed_hook(embed: Tensor, hook) -> Tensor:
        if input_embed is not None:
            embed = input_embed.clone()
        return embed.requires_grad_(True)

    names = [f"blocks.{layer}.hook_{qkv}_input" for layer in range(model.cfg.n_layers) for qkv in "qkv"]
    names += [f"blocks.{layer}.hook_mlp_in" for layer in range(model.cfg.n_layers)]
    names += [f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"]
    with model.hooks(fwd_hooks=[("hook_embed", embed_hook)], bwd_hooks=[(lambda name: name in names, store_grad)]):
        metric_fn(model(tokens)).backward()
    return grads
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_receiver_gradients(receiver_gradients, model)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`hook.name.split(".")[-1]` is one of `"hook_q_input"`, `"hook_k_input"`, `"hook_v_input"`, `"hook_mlp_in"` or `"hook_resid_post"`. For the q/k/v hooks, the gradient has shape `(batch, pos, n_heads, d_model)` and goes into the 12 consecutive receivers starting at `head_receiver(layer, qkv, 0)`.
</details>

<details>
<summary>How does the test check this?</summary>

With a neat identity. The input embedding is part of the residual stream all the way to the end, and it reaches the output *only* through some receiver's input. Every receiver input is a copy of the residual stream, so by the chain rule the gradient with respect to the embeddings equals the sum of the gradients with respect to *all* 445 receiver inputs. The test compares your gradients, summed over the receiver dimension, with an independently computed gradient at `blocks.0.hook_resid_pre`.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - EAP scores

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Now put the pieces together. Implement `eap_scores`, which loops over the task in chunks of `batch_size` prompts and, for each chunk:

1. computes $\Delta a$ = sender activations on the corrupted prompts minus sender activations on the clean prompts;
2. computes the receiver gradients on the clean prompts, for the metric `chunk.metric(logits, chunk.answers).sum() / len(task)`;
3. adds `einsum(delta, grads, "batch pos sender d, batch pos receiver d -> sender receiver")` to a running total.

Finally, zero out the entries which aren't edges by multiplying with `EDGE_MASK`, and return the `(N_SENDERS, N_RECEIVERS)` scores.

(This is the same chunking trick as `get_cache_fwd_and_bwd_chunked` in 1️⃣. Since the scores are linear in the gradient, adding up the chunks' scores gives the score for the whole task.)
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def eap_scores(model: HookedTransformer, task: Task, batch_size: int = 16) -> Float[Tensor, "sender receiver"]:
    """EAP score of every edge: (corrupt - clean sender output) . (gradient at receiver input on the clean run)."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    scores = t.zeros(N_SENDERS, N_RECEIVERS, device=model.cfg.device)
    for chunk in task.chunks(batch_size):
        delta = sender_activations(model, chunk.corrupt_tokens) - sender_activations(model, chunk.clean_tokens)
        metric_fn = lambda logits: chunk.metric(logits, chunk.answers).sum() / len(task)
        grads = receiver_gradients(model, chunk.clean_tokens, metric_fn)
        scores += einops.einsum(delta, grads, "batch pos sender d, batch pos receiver d -> sender receiver")
    return scores * EDGE_MASK.to(scores.device)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_eap_scores(eap_scores, sender_activations, receiver_gradients, model, make_ioi_task(model, 6, seed=5))
# END HIDE

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

start = time.time()
ioi_eap = eap_scores(model, ioi)
print(f"EAP: {int(EDGE_MASK.sum())} edge scores for {len(ioi)} prompts in {time.time() - start:.1f}s\n")


def top_edges_table(scores: Tensor, k: int = 15) -> pd.DataFrame:
    flat = scores.abs().masked_fill(~EDGE_MASK.to(scores.device), -1).flatten()
    rows = []
    for idx in flat.topk(k).indices.tolist():
        u, v = divmod(idx, N_RECEIVERS)
        rows.append({"edge": f"{SENDER_NAMES[u]} -> {RECEIVER_NAMES[v]}", "score": round(scores[u, v].item(), 3)})
    return pd.DataFrame(rows)


display(top_edges_table(ioi_eap))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">EAP: 32491 edge scores for 128 prompts in 1.1s

                  edge  score
0       a9.9 -> logits -2.877
1      a10.7 -> logits  1.955
2   a10.7 -> a11.10&lt;q&gt; -1.161
3    a9.9 -> a11.10&lt;q&gt;  1.101
4       a9.6 -> logits -1.078
5          input -> m0  0.839
6     a11.10 -> logits  0.825
7      a5.5 -> a8.6&lt;v&gt; -0.749
8        m0 -> a6.9&lt;q&gt; -0.672
9      a5.5 -> a6.9&lt;q&gt;  0.648
10          a5.5 -> m6  0.553
11    a9.9 -> a10.7&lt;q&gt;  0.531
12     a10.0 -> logits -0.528
13            m0 -> m4  0.514
14     m0 -> a11.10&lt;k&gt;  0.489</pre>

</details>

32,491 edge scores in about a second (the timing depends on your hardware). The top of the list is the end of the IOI circuit: name movers 9.9, 9.6 and 10.0 writing to the logits (negative scores: corrupting them hurts), negative name movers 10.7 and 11.10 writing to the logits (positive scores), and 9.9 and 10.7 feeding the *query* of the negative name mover 11.10, so the negative name movers know which name to push down. Further down, the induction head 5.5 feeds the value of the S-inhibition head 8.6. Notice `input -> m0` near the top with a *positive* score: EAP thinks corrupting the input embeddings' contribution to MLP0 would *help* the model. Remember MLP0 from 2️⃣?
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Checking EAP against exact edge patching

As in 1️⃣, an approximation is only as good as our check of it. We can't afford exact patching on all 32,491 edges, but we can afford a few hundred. We'll check two groups:

- the **200 edges with the largest |EAP score|**, which are the ones circuit discovery would actually keep;
- **200 edges chosen uniformly at random**, as a control. Most edges in a model don't matter for any given task, so we expect exact patching and EAP both to put these near zero. If EAP's top edges didn't look any different from random ones under exact patching, EAP would be useless.

To patch a single edge $u \to v$ exactly, run the clean prompts with a hook on $v$'s input hook that adds $\Delta a_u$ to $v$'s input (and only to head $h$'s slice, if $v$ is a head's q/k/v input). Here's a helper that maps a receiver index to its hook name and head:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def receiver_hook(v: int) -> tuple[str, int | None]:
    """Returns (hook name, head index or None) for receiver v."""
    if v == LOGITS_RECEIVER:
        return f"blocks.{N_LAYERS - 1}.hook_resid_post", None
    layer, offset = divmod(v, 3 * N_HEADS + 1)
    if offset == 3 * N_HEADS:
        return f"blocks.{layer}.hook_mlp_in", None
    return f"blocks.{layer}.hook_{'qkv'[offset // N_HEADS]}_input", offset % N_HEADS


assert receiver_hook(head_receiver(5, "k", 5)) == ("blocks.5.hook_k_input", 5)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - exact single-edge patching

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Implement `patch_single_edges`, which takes a list of edges `(u, v)` and returns a tensor with the exact change in the mean metric from corrupting each edge on its own. Loop over the task in chunks; for each chunk compute $\Delta a$ for all senders (as in `eap_scores`) and the clean metric, then for each edge run the clean prompts with a hook on `receiver_hook(v)` that adds `delta[:, :, u]` to the receiver's input. Accumulate per-prompt metric sums across chunks, and return (patched sum − clean sum) / `len(task)` for each edge.

Be careful with the head receivers: the hook's input has shape `(batch, pos, n_heads, d_model)`, and you should only add $\Delta a_u$ to head `head`'s slice. Clone before modifying in place.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.no_grad()
def patch_single_edges(
    model: HookedTransformer, task: Task, edges: list[tuple[int, int]], batch_size: int = 32
) -> Float[Tensor, " n_edges"]:
    """Exact change in the mean metric from corrupting each edge (u, v) on its own, in the clean run."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    patched_sums = t.zeros(len(edges))
    clean_sum = 0.0
    for chunk in task.chunks(batch_size):
        delta = sender_activations(model, chunk.corrupt_tokens) - sender_activations(model, chunk.clean_tokens)
        clean_sum += chunk.metric(model(chunk.clean_tokens), chunk.answers).sum().item()
        for i, (u, v) in enumerate(edges):
            hook_name, head = receiver_hook(v)

            def add_delta(x: Tensor, hook) -> Tensor:
                if head is None:
                    return x + delta[:, :, u]
                x = x.clone()
                x[:, :, head] += delta[:, :, u]
                return x

            logits = model.run_with_hooks(chunk.clean_tokens, fwd_hooks=[(hook_name, add_delta)])
            patched_sums[i] += chunk.metric(logits, chunk.answers).sum().item()
    return (patched_sums - clean_sum) / len(task)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_patch_single_edges(patch_single_edges, sender_activations, model, make_ioi_task(model, 6, seed=5))
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Inside the edge loop, look up `hook_name, head = receiver_hook(v)` and define a hook function that closes over `u` and `head`. For MLP and logits receivers it returns `x + delta[:, :, u]`; for head receivers it clones `x`, does `x[:, :, head] += delta[:, :, u]`, and returns the clone.
</details>

The test uses an oracle for edges into the logits: corrupting $u \to$ logits just adds $\Delta a_u$ to the final residual stream, so the effect can be computed directly with `model.ln_final` and `model.unembed`.

Now let's patch the top-200 EAP edges and 200 random edges. This is 400 forward passes per chunk of prompts, so it takes a little while.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

valid_edges = EDGE_MASK.nonzero().tolist()  # list of [u, v], in a fixed order
flat_eap = ioi_eap[EDGE_MASK.to(ioi_eap.device)].cpu()  # EAP score of each valid edge, same order

top_idx = flat_eap.abs().topk(200).indices
random_idx = t.randperm(len(valid_edges), generator=t.Generator().manual_seed(0))[:200]
check_idx = t.cat([top_idx, random_idx])

start = time.time()
check_exact = patch_single_edges(model, ioi, [tuple(valid_edges[i]) for i in check_idx.tolist()])
print(f"Exact patching of {len(check_idx)} edges took {time.time() - start:.0f}s")

exact_top, exact_random = check_exact[:200], check_exact[200:]
eap_top, eap_random = flat_eap[top_idx], flat_eap[random_idx]
for name, ex, ap in [("top-200 by |EAP|", exact_top, eap_top), ("200 random edges", exact_random, eap_random)]:
    pearson = t.corrcoef(t.stack([ex, ap]))[0, 1].item()
    print(f"{name:>17}: Pearson {pearson:.3f}, Spearman {spearman_correlation(ex, ap):.3f}, "
          f"mean |exact effect| {ex.abs().mean():.4f}")

edge_labels = [f"{SENDER_NAMES[valid_edges[i][0]]}->{RECEIVER_NAMES[valid_edges[i][1]]}" for i in top_idx.tolist()]
print("\nThe five top-200 edges where EAP is furthest from the exact effect:")
for j in (eap_top - exact_top).abs().topk(5).indices.tolist():
    print(f"  {edge_labels[j]:>18}: exact {exact_top[j]:+.3f}, EAP {eap_top[j]:+.3f}")

utils.plot_scatter(
    exact_top,
    {"top-200 edges by |EAP|": eap_top, "200 random edges (control)": (exact_random, eap_random)},
    title="EAP vs exact single-edge patching (IOI)",
    xlabel="exact effect of corrupting the edge",
    ylabel="EAP estimate",
    labels=edge_labels,
    n_labels=6,
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/edges_scatter.png" width="480">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

The timing depends on your hardware.

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Exact patching of 400 edges took 66s
 top-200 by |EAP|: Pearson 0.663, Spearman 0.892, mean |exact effect| 0.2192
 200 random edges: Pearson 0.983, Spearman 0.968, mean |exact effect| 0.0014

The five top-200 edges where EAP is furthest from the exact effect:
           input->m0: exact -3.682, EAP +0.839
            a5.5->m5: exact -0.113, EAP +0.470
      a9.9->a10.7&lt;q&gt;: exact +0.980, EAP +0.531
    a10.7->a11.10&lt;q&gt;: exact -0.802, EAP -1.161
            a3.0->m5: exact +0.068, EAP +0.412</pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

**The control behaves as it should.** The 200 random edges cluster tightly at the origin: their average exact effect is 0.0014, about 150 times smaller than for EAP's top 200 (0.22). So EAP's top edges really are special, and EAP is excellent at telling apart edges that matter from edges that don't. On the random edges its estimates are also very accurate (Pearson 0.98), which is what you'd expect from 2️⃣: small effects come from small, nearly linear perturbations.

**Among the top 200, the ranking is good (Spearman 0.89) but the values have some big misses (Pearson 0.66).** Most points lie close to the diagonal, including the largest edges into the logits. The worst miss by far is `input -> m0`: exact −3.68, EAP +0.84. That's the MLP0 failure from 2️⃣ again, now at the level of an edge, with EAP getting the sign wrong. The next-worst misses are edges into mid-layer MLPs (`a5.5 -> m5` and `a3.0 -> m5`, where EAP overstates small effects several-fold, once with the wrong sign) and edges into the negative name movers' *queries*, which act through attention patterns, like head 6.9's query in 2️⃣.

So far EAP looks like a decent, cheap approximation to exact edge patching, with known failure modes. Hold that thought until 4️⃣.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Drawing the circuit

Let's look at the top EAP edges as a graph. In the plot below, heads are placed by layer (vertically) and head index (horizontally), MLPs sit in the right-hand column, and we draw the 50 edges with the largest |EAP score|. Red edges have negative scores: corrupting them *lowers* the logit difference, so they carry information the model uses. Blue edges have positive scores. We've labelled the heads that Wang et al. assigned a role in the IOI circuit, so you can compare.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

IOI_HEAD_ROLES = {
    "a9.9": "name mover", "a9.6": "name mover", "a10.0": "name mover",
    "a10.7": "neg. name mover", "a11.10": "neg. name mover",
    "a7.3": "S-inhibition", "a7.9": "S-inhibition", "a8.6": "S-inhibition", "a8.10": "S-inhibition",
    "a5.5": "induction", "a6.9": "induction", "a5.8": "induction", "a5.9": "induction",
    "a0.1": "duplicate token", "a0.10": "duplicate token", "a3.0": "duplicate token",
    "a2.2": "previous token", "a4.11": "previous token",
}

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

utils.plot_circuit_graph(
    ioi_eap, EDGE_MASK, SENDER_NAMES, RECEIVER_NAMES, top_n=50,
    title="IOI: the 50 edges with the largest |EAP score|", highlight=IOI_HEAD_ROLES,
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/eap_graph.png" width="900">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret this?</summary>

Reading from the top down, you can trace most of the IOI circuit from [1.4.1]:

- **Name movers and negative name movers.** 9.9, 9.6 and 10.0 send red edges to the logits; 10.7 and 11.10 send blue ones. The queries of the negative name movers (dashed) come from the name movers 9.9 and 9.6, and from 10.7.
- **S-inhibition.** 8.6 sends edges into the *queries* of the name movers 9.9 and 10.0 (dashed lines), which is how Wang et al. describe S-inhibition: it tells the name movers which name not to attend to. The other S-inhibition heads, 7.9 and 8.10, show up mainly as *receivers*, taking the induction head 5.5's output into their values (and 8.10 also writes to the logits directly).
- **Induction and previous-token heads.** The induction head 5.5 writes into the values of the S-inhibition heads and the query of the other induction head 6.9. The previous-token heads 2.2 and 4.11 (plus 4.7 and 5.6) feed the *key* of 6.9 (dotted lines): that's the K-composition which defines an induction head.
- **Duplicate token head 3.0 and MLP0.** 3.0 feeds the induction head 6.9 and the mid-layer MLPs, and MLP0 sends many edges up into the circuit. At the name positions, MLP0 is essentially an extended embedding of the name.

What's *missing* is as interesting. The duplicate token heads 0.1 and 0.10 (grey labels in the bottom row) don't appear at all among EAP's top 50 edges, even though Wang et al. found them to be part of the circuit. And the input embeddings connect to the circuit only through MLP0, with a blue (positive) edge whose sign we know is wrong.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 3️⃣

- Because the residual stream is a sum, a component's input splits into contributions from every earlier component. These are the edges of the computational graph: 32,491 of them in GPT-2 Small, with separate query, key and value inputs.
- EAP scores every edge as (sender's activation change) · (gradient at the receiver's input). The computation factorises into sender activations, receiver gradients and one einsum, so all 32,491 scores take about a second.
- Against exact single-edge patching, EAP ranks the important edges well (Spearman 0.89 on the top 200) and puts random edges at zero, as it should. Its biggest miss is the edge from the input to MLP0 (+0.84 vs an exact −3.68), exactly where 2️⃣ told us to expect trouble; other misses involve MLP inputs and attention queries.
- The top EAP edges trace out most of the known IOI circuit, but they leave out the duplicate token heads in layer 0.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 4️⃣ EAP-IG and Faithfulness
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Is a list of edges a circuit?

So far we've judged EAP by how well it predicts *single-edge* patching effects. But that's not what we use it for. We use the scores to pick a **circuit**, a small set of edges that's supposed to be "the part of the model that does the task", and the real question is whether that circuit *works*.

The standard test (used by ACDC, EAP and EAP-IG) is **faithfulness**. Take a circuit $C$, a set of edges. Run the model on the clean prompts, but make every edge *not* in $C$ carry its corrupted value. In terms of a receiver $v$'s input:

$$
x_v^{(C)} = \sum_{(u \to v) \in C} a_u^{(C)} \;+\; \sum_{(u \to v) \notin C} a_u^{\text{corrupt}} \;+\; \text{biases},
$$

where $a_u^{(C)}$ is sender $u$'s output *in this patched run*: if $u$'s own inputs were partly corrupted, its output changes, and that change propagates. The normalised faithfulness is

$$
F(C) = \frac{M(C) - M(\text{corrupt})}{M(\text{clean}) - M(\text{corrupt})},
$$

so the empty circuit (every edge corrupted) gives exactly the corrupted run, $F = 0$, and the full graph gives exactly the clean run, $F = 1$. A good circuit gets $F$ close to 1 with few edges.

Notice the difference from single-edge patching: here we corrupt tens of thousands of edges *at the same time*. Keep that in mind.

### Implementing circuit evaluation

There's a neat way to compute $x_v^{(C)}$ in a single forward pass. The model computes $x_v = \sum_u a_u^{(C)} + \text{biases}$ by itself (that's just the residual stream). So we only need to subtract, for each edge $u \to v$ outside the circuit, the difference between what $u$ actually output in this run and its corrupted output:

$$
x_v^{(C)} = x_v - \sum_{(u \to v) \notin C} \left(a_u^{(C)} - a_u^{\text{corrupt}}\right).
$$

So during the forward pass we keep a running tensor `diff` of shape `(batch, pos, N_SENDERS, d_model)`, holding $a_u^{(C)} - a_u^{\text{corrupt}}$ for every sender computed so far (zero for the ones not computed yet), and at each receiver's input hook we subtract `einsum(diff, not_in_circuit[:, v])`, where `not_in_circuit = EDGE_MASK & ~in_circuit`.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - run the model with a circuit

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 25-35 minutes on this exercise.
> ```

Implement `run_with_circuit`, which returns the mean metric over the task when only the edges in `in_circuit` (a boolean `(N_SENDERS, N_RECEIVERS)` tensor) carry clean values. For each chunk of prompts:

1. Compute the corrupted sender outputs `corrupt_out = sender_activations(model, chunk.corrupt_tokens)`, and a zero tensor `diff` of the same shape.
2. Run the model on the clean prompts with these hooks, in forward order:
   - **senders** (fill in `diff` as each sender is computed):
     - `blocks.0.hook_resid_pre`: `diff[:, :, 0] = x - corrupt_out[:, :, 0]`;
     - `blocks.{layer}.attn.hook_z`: compute the heads' outputs from `z` with `W_O` (as in `sender_activations`), and write their difference from the corrupted outputs into the 12 sender slots starting at `head_sender(layer, 0)`;
     - `blocks.{layer}.hook_mlp_out`: write into slot `mlp_sender(layer)`;
   - **receivers** (subtract the corrupted edges' differences):
     - `blocks.{layer}.hook_{q,k,v}_input`: subtract `einsum(diff, not_in_circuit[:, receivers of this layer and type], "batch pos sender d, sender head -> batch pos head d")`;
     - `blocks.{layer}.hook_mlp_in` and `blocks.{N_LAYERS-1}.hook_resid_post`: the same with a single receiver column.
3. Add the chunk's per-prompt metric sum to a running total, and return total / `len(task)`.

Because of the order in which TransformerLens calls hooks, the senders a receiver depends on are always filled in before that receiver's hook runs (for example, layer 3's MLP output is written before layer 4's `hook_q_input` is called). Senders that haven't run yet have `diff = 0`, and they can't send to this receiver anyway.

The test checks three things: the full graph gives the clean metric, the empty circuit gives the corrupted metric, and a circuit missing exactly one edge gives the same answer as `patch_single_edges`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.no_grad()
def run_with_circuit(
    model: HookedTransformer, task: Task, in_circuit: Bool[Tensor, "sender receiver"], batch_size: int = 32
) -> float:
    """Mean metric over the task when every edge outside `in_circuit` carries its corrupted value."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    not_in_circuit = (EDGE_MASK.to(in_circuit.device) & ~in_circuit).float().to(model.cfg.device)
    total = 0.0
    for chunk in task.chunks(batch_size):
        corrupt_out = sender_activations(model, chunk.corrupt_tokens)
        diff = t.zeros_like(corrupt_out)

        def store_input(x: Tensor, hook) -> None:
            diff[:, :, 0] = x - corrupt_out[:, :, 0]

        def store_heads(z: Tensor, hook) -> None:
            layer, start = hook.layer(), head_sender(hook.layer(), 0)
            out = einops.einsum(z, model.W_O[layer], "batch pos head d_head, head d_head d_model -> batch pos head d_model")
            diff[:, :, start : start + N_HEADS] = out - corrupt_out[:, :, start : start + N_HEADS]

        def store_mlp(x: Tensor, hook) -> None:
            u = mlp_sender(hook.layer())
            diff[:, :, u] = x - corrupt_out[:, :, u]

        def patch_head_inputs(x: Tensor, hook) -> Tensor:
            qkv = hook.name.split(".")[-1][len("hook_")]
            start = head_receiver(hook.layer(), qkv, 0)
            mask = not_in_circuit[:, start : start + N_HEADS]
            return x - einops.einsum(diff, mask, "batch pos sender d, sender head -> batch pos head d")

        def patch_single_input(x: Tensor, hook) -> Tensor:
            v = LOGITS_RECEIVER if hook.name.endswith("resid_post") else mlp_receiver(hook.layer())
            return x - einops.einsum(diff, not_in_circuit[:, v], "batch pos sender d, sender -> batch pos d")

        hooks = [("blocks.0.hook_resid_pre", store_input)]
        for layer in range(model.cfg.n_layers):
            hooks += [(f"blocks.{layer}.hook_{qkv}_input", patch_head_inputs) for qkv in "qkv"]
            hooks += [
                (f"blocks.{layer}.attn.hook_z", store_heads),
                (f"blocks.{layer}.hook_mlp_in", patch_single_input),
                (f"blocks.{layer}.hook_mlp_out", store_mlp),
            ]
        hooks += [(f"blocks.{model.cfg.n_layers - 1}.hook_resid_post", patch_single_input)]
        logits = model.run_with_hooks(chunk.clean_tokens, fwd_hooks=hooks)
        total += chunk.metric(logits, chunk.answers).sum().item()
    return total / len(task)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_run_with_circuit(run_with_circuit, patch_single_edges, model, make_ioi_task(model, 6, seed=5))
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint - the sender hooks</summary>

Sender hooks don't change the forward pass; they just record. They can return `None`. Remember that `diff` holds $a_u^{(C)} - a_u^{\text{corrupt}}$, the difference between what the sender output *in this run* and its corrupted output.
</details>

<details>
<summary>Hint - the receiver hooks</summary>

For a q/k/v input hook at layer $\ell$, the relevant columns of `not_in_circuit` are `head_receiver(layer, qkv, 0)` to `head_receiver(layer, qkv, 0) + N_HEADS`, giving a `(N_SENDERS, n_heads)` mask. `einops.einsum(diff, mask, "batch pos sender d, sender head -> batch pos head d")` then has exactly the shape of the hook's activation, `(batch, pos, n_heads, d_model)`. Return `x - that`.
</details>

<details>
<summary>Help - my full circuit doesn't give the clean metric.</summary>

With the full graph, `not_in_circuit` is all zeros, so the receiver hooks subtract nothing and the run should be exactly the clean run. If it isn't, check that your receiver hooks *return* the patched tensor rather than modifying a copy, and that `not_in_circuit` is `EDGE_MASK & ~in_circuit` (not the other way round).
</details>

<details>
<summary>Help - my empty circuit doesn't give the corrupted metric.</summary>

With the empty circuit, each receiver input becomes $x_v - \sum_u (a_u^{(C)} - a_u^{\text{corrupt}}) = \sum_u a_u^{\text{corrupt}} + \text{biases}$, exactly the corrupted run's input. If this fails, check that every sender's slot of `diff` is filled in (including the input embedding at `blocks.0.hook_resid_pre`), and that your head outputs don't include `b_O` (the corrupted outputs from `sender_activations` don't either).
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
To turn scores into circuits we keep the $k$ edges with the largest *absolute* score. (We use absolute values because an edge with a large positive score, one whose corruption *helps* the metric, like the negative name movers' edges to the logits, is also part of the mechanism.)
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def top_k_circuit(scores: Float[Tensor, "sender receiver"], k: int) -> Bool[Tensor, "sender receiver"]:
    """The k valid edges with the largest |score|."""
    flat = scores.abs().masked_fill(~EDGE_MASK.to(scores.device), -1.0).flatten()
    circuit = t.zeros_like(flat, dtype=t.bool)
    circuit[flat.topk(k).indices] = True
    return circuit.reshape(scores.shape)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - faithfulness curves

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 5-10 minutes on this exercise.
> ```

Implement `faithfulness_curve`, which returns the normalised faithfulness $F$ of `top_k_circuit(scores, k)` for each `k` in `ks`. Compute the clean and corrupted mean metrics on the same task (you can use `mean_metric`, or chunk them yourself for large tasks).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def faithfulness_curve(
    model: HookedTransformer, task: Task, scores: Float[Tensor, "sender receiver"], ks: list[int], batch_size: int = 32
) -> list[float]:
    """Normalised faithfulness of the top-k circuit (by |score|), for each k in ks."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    clean_value = mean_metric(model, task, task.clean_tokens)
    corrupt_value = mean_metric(model, task, task.corrupt_tokens)
    return [
        (run_with_circuit(model, task, top_k_circuit(scores, k), batch_size) - corrupt_value) / (clean_value - corrupt_value)
        for k in ks
    ]
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_faithfulness_curve(faithfulness_curve, model, make_ioi_task(model, 6, seed=5))
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Now the real test. We take the EAP scores computed on `ioi`, and measure the faithfulness of their top-$k$ circuits on the **held-out** prompts in `ioi_heldout`, for $k$ from 20 up to 10,000 edges. As a control, we do the same for random edge sets of the same sizes (five random seeds). Before you run it: what do you expect the curve to look like?
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

KS = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def random_faithfulness(model: HookedTransformer, task: Task, ks: list[int], n_seeds: int = 5) -> list[list[float]]:
    """Faithfulness curves of random edge sets (random scores), one curve per seed."""
    generator = t.Generator().manual_seed(0)
    return [faithfulness_curve(model, task, t.rand(N_SENDERS, N_RECEIVERS, generator=generator), ks) for _ in range(n_seeds)]


ioi_random_curves = random_faithfulness(model, ioi_heldout, KS)
ioi_eap_curve = faithfulness_curve(model, ioi_heldout, ioi_eap, KS)
print(f"{'edges':>8} {'EAP':>7} {'random (mean)':>14}")
for k, f_eap, f_rand in zip(KS, ioi_eap_curve, np.mean(ioi_random_curves, axis=0)):
    print(f"{k:>8} {f_eap:>7.2f} {f_rand:>14.2f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">   edges     EAP  random (mean)
      20    0.00          -0.00
      50    0.03          -0.00
     100   -0.03          -0.00
     200   -0.02          -0.00
     500    0.09           0.00
    1000   -0.01          -0.00
    2000    0.40          -0.00
    5000    0.82          -0.01
   10000    0.86          -0.04</pre>

</details>

The random control does what it should: random edge sets recover nothing at any size. But look at EAP. **Its top 1,000 edges recover nothing either** (−0.01). Only at 2,000 edges does its circuit start to work (0.40), and it takes 5,000 edges, 15% of the whole graph, to get to 0.82. That's strange, given that in 3️⃣ EAP ranked the important edges well against exact patching, and its top-50 graph looked like the IOI circuit. Something in the top 1,000 must be missing. Before we find out what, let's implement the alternative.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## EAP-IG

In 2️⃣ we fixed a badly-behaved single-component estimate by averaging the gradient along the path from clean to corrupt. EAP-IG (Hanna, Pezzelle & Belinkov, 2024) does the same thing for edges, with one crucial change: instead of moving one component at a time (which would cost $m$ backward passes *per edge*), it moves the **input** along a path, so that every activation in the model moves at once:

$$
e(\alpha) = e_{\text{clean}} + \alpha \left(e_{\text{corrupt}} - e_{\text{clean}}\right),
$$

where $e$ is the token embedding (the positional embeddings are the same for clean and corrupted prompts). At each of $m$ points along this path we run the model on $e(\alpha_k)$ and take the gradient at every receiver's input, then use the *average* gradient in place of the clean one:

$$
\Delta M_{\text{EAP-IG}}(u \to v) = \Delta a_u \cdot \frac{1}{m} \sum_{k=0}^{m-1} \left.\frac{\partial M}{\partial x_v}\right|_{e(k/m)} .
$$

(The activation difference $\Delta a_u$ is the same as in EAP.) That's $m$ backward passes in total, for all 32,491 edges. With $m=1$ the only point is $\alpha = 0$, the clean input, so EAP-IG with one step *is* EAP.

What does this estimate? It's no longer an approximation to single-edge patching. Instead, it has a **completeness** property: by the same fundamental-theorem-of-calculus argument as in 2️⃣, the scores of the edges leaving the input add up (as $m \to \infty$) to the *total* change from clean to corrupted,

$$
\sum_v \Delta M_{\text{EAP-IG}}(\text{input} \to v) \;\longrightarrow\; M(\text{corrupt}) - M(\text{clean}),
$$

because the gradient with respect to the input embedding is the sum of the gradients at all receivers (the identity the `receiver_gradients` test used). EAP-IG divides up the effect of corrupting *everything*, along a path where everything is partially corrupted. Hold on to that thought; it's going to matter.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - EAP-IG scores

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement `eap_ig_scores`. It's `eap_scores` with an inner loop over $k = 0, \dots, m-1$: compute the clean and corrupted token embeddings for the chunk with `model.embed(tokens)` (inside `t.no_grad()`), interpolate them with $\alpha = k/m$, call `receiver_gradients(..., input_embed=...)` on the clean tokens, and add `einsum(delta, grads) / n_steps` to the scores.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def eap_ig_scores(
    model: HookedTransformer, task: Task, n_steps: int = 5, batch_size: int = 16
) -> Float[Tensor, "sender receiver"]:
    """EAP-IG score of every edge, integrating receiver gradients along the clean -> corrupt input-embedding path."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    scores = t.zeros(N_SENDERS, N_RECEIVERS, device=model.cfg.device)
    for chunk in task.chunks(batch_size):
        delta = sender_activations(model, chunk.corrupt_tokens) - sender_activations(model, chunk.clean_tokens)
        with t.no_grad():
            clean_embed, corrupt_embed = model.embed(chunk.clean_tokens), model.embed(chunk.corrupt_tokens)
        metric_fn = lambda logits: chunk.metric(logits, chunk.answers).sum() / len(task)
        for k in range(n_steps):
            alpha = k / n_steps
            embed = clean_embed + alpha * (corrupt_embed - clean_embed)
            grads = receiver_gradients(model, chunk.clean_tokens, metric_fn, input_embed=embed)
            scores += einops.einsum(delta, grads, "batch pos sender d, batch pos receiver d -> sender receiver") / n_steps
    return scores * EDGE_MASK.to(scores.device)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_eap_ig_scores(eap_ig_scores, eap_scores, model, make_ioi_task(model, 6, seed=5))
# END HIDE

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

start = time.time()
ioi_eap_ig = eap_ig_scores(model, ioi, n_steps=5)
print(f"EAP-IG (5 steps): {time.time() - start:.1f}s\n")

clean_value, corrupt_value = mean_metric(model, ioi, ioi.clean_tokens), mean_metric(model, ioi, ioi.corrupt_tokens)
print("Completeness check: sum of scores of edges leaving the input embedding")
print(f"  M(corrupt) - M(clean) = {corrupt_value - clean_value:+.3f}")
print(f"  EAP                   = {ioi_eap[0].sum():+.3f}")
for n_steps in [5, 20]:
    scores = ioi_eap_ig if n_steps == 5 else eap_ig_scores(model, ioi, n_steps=n_steps)
    print(f"  EAP-IG ({n_steps:>2} steps)     = {scores[0].sum():+.3f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

The timing depends on your hardware.

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">EAP-IG (5 steps): 2.5s

Completeness check: sum of scores of edges leaving the input embedding
  M(corrupt) - M(clean) = -3.117
  EAP                   = +1.104
  EAP-IG ( 5 steps)     = -3.254
  EAP-IG (20 steps)     = -3.139</pre>

</details>

This is a striking check. Summed over all edges leaving the input, EAP's scores are its linear prediction of what happens when you corrupt *all* the names: it predicts the logit difference goes **up** by 1.10. In fact it goes down by 3.12. EAP-IG with only 5 steps gets −3.25, and with 20 steps −3.14, converging on the true value as completeness says it must. The first-order expansion around the clean run simply doesn't describe the journey from clean to corrupt.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

ioi_eap_ig_curve = faithfulness_curve(model, ioi_heldout, ioi_eap_ig, KS)
print(f"{'edges':>8} {'EAP':>7} {'EAP-IG':>7} {'random':>7}")
for k, f_eap, f_ig, f_rand in zip(KS, ioi_eap_curve, ioi_eap_ig_curve, np.mean(ioi_random_curves, axis=0)):
    print(f"{k:>8} {f_eap:>7.2f} {f_ig:>7.2f} {f_rand:>7.2f}")

utils.plot_faithfulness(
    {
        "IOI (held-out prompts)": {
            "ks": KS,
            "curves": {"EAP": ioi_eap_curve, "EAP-IG (5 steps)": ioi_eap_ig_curve},
            "random": ioi_random_curves,
            "n_edges": int(EDGE_MASK.sum()),
        }
    },
    title="How much of the IOI behaviour does the top-k circuit recover?",
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/faithfulness_ioi.png" width="600">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">   edges     EAP  EAP-IG  random
      20    0.00   -0.00   -0.00
      50    0.03   -0.13   -0.00
     100   -0.03   -0.02   -0.00
     200   -0.02    0.08   -0.00
     500    0.09    0.35    0.00
    1000   -0.01    0.57   -0.00
    2000    0.40    0.63   -0.00
    5000    0.82    0.84   -0.01
   10000    0.86    0.94   -0.04</pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

**EAP-IG's circuits work far better at the sizes that matter.** At 500 edges EAP-IG's circuit recovers 35% of the logit difference on held-out prompts, and at 1,000 edges 57%, while EAP's recover 9% and nothing. The curves only meet at around 5,000 edges, where both circuits contain most of what matters. Random edge sets recover nothing at any size, so neither curve is an artefact of how we evaluate circuits.

**This is the signature result of the section,** and it reproduces the main finding of Hanna et al. (2024): circuits found by EAP-IG are substantially more faithful than those found by EAP, at the same size. Remember that the scores were computed on one set of prompts and the faithfulness on another, so these are held-out numbers.

**Some honest caveats about the curves themselves:**

- Neither method reaches full faithfulness with 10,000 edges (EAP-IG: 0.94). With the ABC corruption, which changes every name, the IOI behaviour in GPT-2 Small depends on a lot of edges.
- Very small circuits can be slightly *worse* than nothing (EAP-IG's 50-edge circuit scores −0.13). That can happen when a circuit keeps some edges that work against the answer (like the negative name movers' connections) without enough of the ones that work for it.
- The curves aren't monotonic (EAP: 0.09 at 500 edges, −0.01 at 1,000). Adding edges can break a partial mechanism as easily as complete it.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Why does EAP fail on IOI?

Let's find out what EAP-IG's circuit has that EAP's doesn't. The cell below takes the top 1,000 edges by each method, finds the edges that are in EAP-IG's circuit but not EAP's, and lists the ten with the largest EAP-IG scores, along with their rank under EAP and their *exact* single-edge patching effect.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

eap_circuit_1000, ig_circuit_1000 = top_k_circuit(ioi_eap, 1000), top_k_circuit(ioi_eap_ig, 1000)
ig_only = ig_circuit_1000 & ~eap_circuit_1000
print(f"Overlap of the two 1000-edge circuits: {(eap_circuit_1000 & ig_circuit_1000).sum().item()} edges\n")

ig_only_scores = ioi_eap_ig.abs().masked_fill(~ig_only, -1).flatten()
missing_edges = [divmod(idx, N_RECEIVERS) for idx in ig_only_scores.topk(10).indices.tolist()]
eap_ranks = ioi_eap.abs().masked_fill(~EDGE_MASK.to(ioi_eap.device), -1).flatten().argsort(descending=True).argsort()
missing_exact = patch_single_edges(model, ioi, missing_edges)

display(pd.DataFrame([
    {
        "edge": f"{SENDER_NAMES[u]} -> {RECEIVER_NAMES[v]}",
        "EAP-IG score": round(ioi_eap_ig[u, v].item(), 3),
        "EAP score": round(ioi_eap[u, v].item(), 3),
        "EAP rank": eap_ranks[u * N_RECEIVERS + v].item() + 1,
        "exact single-edge effect": round(exact, 3),
    }
    for (u, v), exact in zip(missing_edges, missing_exact.tolist())
]))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Overlap of the two 1000-edge circuits: 814 edges

               edge  EAP-IG score  EAP score  EAP rank  exact single-edge effect
0  input -> a0.1&lt;k&gt;        -0.296     -0.001     10317                     0.036
1  input -> a0.1&lt;q&gt;        -0.233      0.005      2334                     0.022
2  input -> a0.3&lt;v&gt;        -0.107      0.009      1333                    -0.011
3       a0.10 -> m0        -0.097     -0.008      1459                    -0.124
4        a0.1 -> m1        -0.061     -0.004      2906                    -0.014
5    m0 -> a1.11&lt;k&gt;        -0.060     -0.006      1862                    -0.111
6     m0 -> a3.6&lt;v&gt;        -0.041      0.003      3675                    -0.011
7     m0 -> a9.6&lt;v&gt;         0.039     -0.005      2399                     0.018
8     m1 -> a3.4&lt;q&gt;         0.038     -0.001      8032                    -0.006
9   a2.2 -> a4.7&lt;v&gt;         0.038     -0.011      1175                    -0.023</pre>

</details>

The circuits share 814 of their 1,000 edges. The ten highest-scoring edges that EAP-IG keeps and EAP drops sit almost all at the bottom of the network: the input embeddings, the layer-0 heads, and MLPs 0 and 1. They come in two kinds:

- **Edges with a real single-edge effect that EAP underestimates.** `a0.10 -> m0` has an exact effect of −0.124 and `m0 -> a1.11<k>` of −0.111, but EAP gives them −0.008 and −0.006. These are the MLP0 failures from 2️⃣ again.
- **Edges with almost no single-edge effect, about which EAP is right.** The two top rows are the input embeddings feeding the key and query of head 0.1, one of Wang et al.'s duplicate token heads. Corrupting either edge on its own changes the logit difference by +0.04 and +0.02 (slightly *positive*!). EAP agrees that they barely matter, and ranks the key edge 10,317th out of 32,491. Yet EAP-IG ranks them among the most important edges in the model.

For the second kind, EAP isn't making an approximation error at all. EAP-IG is measuring something different. What?
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - joint vs individual effects

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 10 minutes on this exercise.
> ```

Here's a hypothesis. Wang et al. found *three* duplicate token heads (0.1, 0.10 and 3.0), plus induction heads that compute the same "S is duplicated" signal by a different route. If several heads compute the same thing, corrupting the input to any *one* of them does little, because the others still deliver the signal; but corrupting all of them at once breaks it. The mechanism behaves like an OR gate, and single-edge patching (and any good approximation to it, like EAP) is blind to OR gates.

Test this. Implement `joint_vs_individual`, which takes a list of edges and returns:

- a tensor with the exact effect of corrupting each edge on its own (`patch_single_edges`), and
- the exact effect of corrupting all of them *at once*: `run_with_circuit` with the full graph minus those edges, minus the clean mean metric.

If effects were additive, the joint effect would equal the sum of the individual effects.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def joint_vs_individual(
    model: HookedTransformer, task: Task, edges: list[tuple[int, int]], batch_size: int = 32
) -> tuple[Float[Tensor, " n_edges"], float]:
    """Returns (individual effects of corrupting each edge alone, effect of corrupting all of them together)."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    individual = patch_single_edges(model, task, edges, batch_size)
    circuit = EDGE_MASK.clone()
    for u, v in edges:
        circuit[u, v] = False
    joint = run_with_circuit(model, task, circuit, batch_size) - mean_metric(model, task, task.clean_tokens)
    return individual, joint
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_joint_vs_individual(joint_vs_individual, model, make_ioi_task(model, 6, seed=5))
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
We'll test three groups of edges: the edges from the input embedding into the duplicate token heads 0.1 and 0.10 (query, key and value: 6 edges); the edges from the input into *every* layer-0 head (36 edges); and, as a control, 36 random edges.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

edge_groups = {
    "input -> heads 0.1, 0.10 (q, k, v)": [(0, head_receiver(0, qkv, h)) for h in [1, 10] for qkv in "qkv"],
    "input -> all layer-0 heads (q, k, v)": [(0, head_receiver(0, qkv, h)) for h in range(N_HEADS) for qkv in "qkv"],
    "36 random edges": [tuple(valid_edges[i]) for i in t.randperm(len(valid_edges), generator=t.Generator().manual_seed(1))[:36].tolist()],
}
for name, edges in edge_groups.items():
    individual, joint = joint_vs_individual(model, ioi, edges)
    print(f"{name:>38}: sum of individual effects {individual.sum():+.3f}, joint effect {joint:+.3f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">  input -> heads 0.1, 0.10 (q, k, v): sum of individual effects -0.319, joint effect -0.779
  input -> all layer-0 heads (q, k, v): sum of individual effects +0.337, joint effect -1.035
                       36 random edges: sum of individual effects -0.042, joint effect -0.043</pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

**The effects are not additive.** Corrupting the six input edges of the duplicate token heads 0.1 and 0.10 together costs 2.4 times as much as the sum of corrupting them one at a time (−0.78 vs −0.32). For all 36 input edges into layer 0's heads it's more dramatic still: individually they add up to a small *positive* effect (+0.34), but corrupting them together costs a third of the whole logit difference (−1.04). The control behaves differently: for 36 random edges the joint effect equals the sum of the individual ones (−0.043 vs −0.042), as you'd expect for edges that don't interact.

This is what a **redundant** mechanism looks like. Several components compute the same signal (here, "this name has appeared before"), so knocking out any one input leaves the others to deliver it, like an OR gate. Single-edge patching sees each input as unimportant, and so does EAP, which approximates single-edge patching well. But a faithfulness test corrupts about 31,000 edges *at once*, so the backups are gone too, and the edges matter after all.

EAP-IG sees this because its gradients are taken along a path where *every* input is partially corrupted at once. Halfway along the path, the backups are already half-broken, and the remaining duplicate-detection edges carry real weight. That's the sense in which EAP-IG "divides up the total effect" rather than estimating single-edge effects, and it's why it matches what circuit discovery needs.

We haven't shown *which* components back each other up; that would take a more careful set of joint patches (see the bonus section). "Redundancy" is the simplest explanation that fits these numbers, not a proven mechanism.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
One last check on this explanation. If EAP's circuit fails because it's missing a handful of jointly-important edges, then adding them back should rescue it. Let's add the ten edges from the table above (the top EAP-IG edges that EAP missed) to EAP's circuits, and re-measure faithfulness on the held-out prompts.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

heldout_clean = mean_metric(model, ioi_heldout, ioi_heldout.clean_tokens)
heldout_corrupt = mean_metric(model, ioi_heldout, ioi_heldout.corrupt_tokens)

print(f"{'edges':>8} {'EAP':>6} {'EAP + 10 edges':>15} {'EAP-IG':>7}")
for k in [1000, 2000, 5000]:
    circuit = top_k_circuit(ioi_eap, k)
    for u, v in missing_edges:
        circuit[u, v] = True
    rescued = (run_with_circuit(model, ioi_heldout, circuit) - heldout_corrupt) / (heldout_clean - heldout_corrupt)
    print(f"{k:>8} {ioi_eap_curve[KS.index(k)]:>6.2f} {rescued:>15.2f} {ioi_eap_ig_curve[KS.index(k)]:>7.2f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">   edges    EAP  EAP + 10 edges  EAP-IG
    1000  -0.01            0.20    0.57
    2000   0.40            0.51    0.63
    5000   0.82            0.81    0.84</pre>

</details>

Adding just ten edges lifts EAP's 1,000-edge circuit from nothing to 0.20, and its 2,000-edge circuit from 0.40 to 0.51. So the missing edges are part of the story. But they're not all of it: the rescued circuits still fall well short of EAP-IG's (0.57 and 0.63). The two 1,000-edge circuits differ in 186 edges, and these are spread throughout the network: of EAP-IG's 186 extra edges, 52 leave the input or a layer-0 component, 67 leave layers 1-4, and 67 leave later layers. An honest summary: we've found the *kind* of edge EAP misses and why, but not a short list that accounts for the whole gap.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## A second task: greater-than

One task is an anecdote. Let's repeat the comparison on **greater-than** (Hanna, Liu & Variengien, 2023), which you may have met in [8.1]:

```
The war lasted from the year 1732 to the year 17
```

GPT-2 Small should continue with a two-digit year greater than 32. The corrupted prompt replaces the start year with `1701`, after which every year is "greater". The metric is the probability mass on years greater than the start year minus the mass on years less than or equal to it, a *probability* difference rather than a logit difference. That makes it more nonlinear (it goes through a softmax), which is another reason a first-order method might struggle.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def make_greater_than_metric(model: HookedTransformer) -> Callable[[Tensor, Tensor], Tensor]:
    year_tokens = t.tensor([model.to_single_token(f"{year:02d}") for year in range(100)], device=model.cfg.device)

    def greater_than_metric(logits: Float[Tensor, "batch seq d_vocab"], start_years: Int[Tensor, " batch"]) -> Float[Tensor, " batch"]:
        probs = logits[:, -1].softmax(dim=-1)[:, year_tokens]  # (batch, 100)
        greater = t.arange(100, device=logits.device)[None, :] > start_years[:, None]
        return (probs * greater).sum(-1) - (probs * ~greater).sum(-1)

    return greater_than_metric


def make_greater_than_task(model: HookedTransformer, n: int, seed: int) -> Task:
    clean, corrupt, years = utils.make_greater_than_prompts(model, n, seed=seed)
    return Task(
        model.to_tokens(clean), model.to_tokens(corrupt), t.tensor(years, device=model.cfg.device), make_greater_than_metric(model)
    )

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

gt, gt_heldout = make_greater_than_task(model, 128, seed=0), make_greater_than_task(model, 128, seed=1)
print(model.to_string(gt.clean_tokens[0, 1:]), "|", model.to_string(gt.corrupt_tokens[0, 1:]))
print(f"Held-out: clean metric {mean_metric(model, gt_heldout, gt_heldout.clean_tokens):.3f}, "
      f"corrupt metric {mean_metric(model, gt_heldout, gt_heldout.corrupt_tokens):.3f}\n")

gt_eap, gt_eap_ig = eap_scores(model, gt), eap_ig_scores(model, gt, n_steps=5)
gt_curves = {"EAP": faithfulness_curve(model, gt_heldout, gt_eap, KS), "EAP-IG (5 steps)": faithfulness_curve(model, gt_heldout, gt_eap_ig, KS)}
gt_random_curves = random_faithfulness(model, gt_heldout, KS)
print(f"{'edges':>8} {'EAP':>7} {'EAP-IG':>7} {'random':>7}")
for k, f_eap, f_ig, f_rand in zip(KS, gt_curves["EAP"], gt_curves["EAP-IG (5 steps)"], np.mean(gt_random_curves, axis=0)):
    print(f"{k:>8} {f_eap:>7.2f} {f_ig:>7.2f} {f_rand:>7.2f}")

utils.plot_faithfulness(
    {
        "IOI (held-out prompts)": {"ks": KS, "curves": {"EAP": ioi_eap_curve, "EAP-IG (5 steps)": ioi_eap_ig_curve}, "random": ioi_random_curves, "n_edges": int(EDGE_MASK.sum())},
        "Greater-than (held-out prompts)": {"ks": KS, "curves": gt_curves, "random": gt_random_curves, "n_edges": int(EDGE_MASK.sum()), "legend_loc": "lower right"},
    },
    title="Faithfulness of top-k circuits found by EAP and EAP-IG, vs random edge sets of the same size",
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/faithfulness_both.png" width="1000">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">The revolution lasted from the year 1455 to the year 14 | The revolution lasted from the year 1401 to the year 14
Held-out: clean metric 0.856, corrupt metric -0.486

   edges     EAP  EAP-IG  random
      20    0.16    0.18   -0.00
      50    0.41    0.56   -0.00
     100    0.70    0.87   -0.00
     200    0.90    0.94   -0.00
     500    0.96    0.96    0.00
    1000    0.98    0.98    0.00
    2000    0.96    1.02    0.00
    5000    0.98    1.00   -0.00
   10000    0.99    1.00    0.01</pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

**On greater-than, both methods work, and EAP-IG is still better at small sizes.** EAP-IG's 100-edge circuit recovers 87% of the probability difference, against 70% for EAP's; by 500 edges both are at 0.96, and the curves are nearly indistinguishable from there on. Random edge sets again recover nothing. (EAP-IG's 2,000-edge circuit scores 1.02: slightly *better* than the full model. That happens when a circuit leaves out some edges that work against the task.)

**So EAP's collapse on IOI is not a general property of EAP.** Why the difference? Here are some hypotheses to think about, and to test if you want to:

- The greater-than corruption changes a single token (the start year's last two digits), while the ABC corruption changes three names. A smaller change is closer to the linear regime.
- The IOI behaviour, as we saw, depends on redundant duplicate-detection edges near the input. Hanna et al. (2023) found the greater-than computation concentrated in a few mid-layer attention heads that move the start year to the final position, and MLPs 8-11 that compute the comparison: a more serial structure.
- Greater-than's metric is a probability difference, which is *more* nonlinear than the logit difference. Despite that, EAP does well here, which suggests the nonlinearity of the metric matters less than the structure of the circuit and the size of the corruption.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Finally, let's draw the IOI circuit according to EAP-IG, the same way we drew EAP's in 3️⃣.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

utils.plot_circuit_graph(
    ioi_eap_ig, EDGE_MASK, SENDER_NAMES, RECEIVER_NAMES, top_n=50,
    title="IOI: the 50 edges with the largest |EAP-IG score|", highlight=IOI_HEAD_ROLES,
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/eap_ig_graph.png" width="900">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - what changed compared to EAP's graph?</summary>

The top of the circuit (name movers, negative name movers, S-inhibition, induction heads) is almost the same as in EAP's graph (a few weaker edges into the logits, from 10.6, 7.9 and MLP9, are new). The main differences are at the bottom:

- `input -> m0` is now the **second-largest** edge in the model, and red: EAP-IG says corrupting the embeddings' contribution to MLP0 is very harmful, matching the exact effect (where EAP had it positive).
- The duplicate token head **0.1** now appears, with its query, key and value inputs coming from the input embeddings, and its output going into MLP0.
- MLP0 now also sends edges into MLP1 and MLP2, into the key of the duplicate token head 3.0 (EAP only had its query), and into the query of the induction head 5.5.

This is the part of the IOI circuit that detects *which name is repeated*, and it's exactly the part EAP's first-order view couldn't see.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Try it yourself - change the corruption

Everything in this section depends on the choice of corrupted prompts, and that choice is a *question*: "which information do I want to trace?". The ABC corruption we've used replaces all three names, so it traces everything the model does with names. A much gentler corruption replaces only the second mention of the subject:

```
clean:    When Mary and John went to the store, John gave a drink to
corrupt:  When Mary and John went to the store, Peter gave a drink to
```

This keeps the names in the first clause, and only removes the *duplication* of John. It asks the narrower question "how does the model use the fact that S is repeated?".

The cell below re-runs the EAP vs EAP-IG comparison with this `"s2"` corruption. Before running it, predict: will the gap between EAP and EAP-IG get bigger or smaller? (Hint: think about which edges EAP missed, and what this corruption changes at the name positions.) Then try your own variations:

- change `PLAY_STEPS` to 1 (which should reproduce EAP), 2 or 20;
- write a new corruption in `utils.make_ioi_prompts` (e.g. swap the IO and S names, or replace only the IO);
- use `make_greater_than_task` with a different corrupted start year.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

PLAY_CORRUPTION = "s2"
PLAY_STEPS = 5

play_task = make_ioi_task(model, 128, seed=0, corruption=PLAY_CORRUPTION)
play_heldout = make_ioi_task(model, 128, seed=1, corruption=PLAY_CORRUPTION)
print(f"Held-out: clean {mean_metric(model, play_heldout, play_heldout.clean_tokens):.3f}, "
      f"corrupt {mean_metric(model, play_heldout, play_heldout.corrupt_tokens):.3f}")

play_curves = {
    "EAP": faithfulness_curve(model, play_heldout, eap_scores(model, play_task), KS),
    f"EAP-IG ({PLAY_STEPS} steps)": faithfulness_curve(model, play_heldout, eap_ig_scores(model, play_task, n_steps=PLAY_STEPS), KS),
}
play_random = random_faithfulness(model, play_heldout, KS, n_seeds=2)
for name, curve in play_curves.items():
    print(f"{name:>16}: " + "  ".join(f"{k}: {f:.2f}" for k, f in zip(KS, curve)))

utils.plot_faithfulness(
    {f"IOI, corruption = {PLAY_CORRUPTION!r}": {"ks": KS, "curves": play_curves, "random": play_random, "n_edges": int(EDGE_MASK.sum())}},
    title="Try it yourself: faithfulness with a different corruption",
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/play_faithfulness.png" width="600">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output (for the "s2" corruption)</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Held-out: clean 3.298, corrupt 0.000
             EAP: 20: -0.00  50: 0.02  100: 0.12  200: 0.32  500: 0.53  1000: 0.43  2000: 0.66  5000: 0.89  10000: 0.97
EAP-IG (5 steps): 20: 0.00  50: 0.04  100: 0.21  200: 0.39  500: 0.63  1000: 0.76  2000: 0.84  5000: 0.88  10000: 0.98</pre>

</details>

<details>
<summary>Help - what should I see?</summary>

With the gentler corruption, EAP's circuits work much better: 0.53 at 500 edges, compared with 0.09 under the ABC corruption. Only one token changes, so the corruption is closer to the linear regime, and since the first two names are the same in the clean and corrupted prompts, there's no large "the name is a different name" change at those positions. EAP-IG is still better at every size from 100 to 2,000 edges, though (0.76 vs 0.43 at 1,000), and EAP's curve is non-monotonic: its 1,000-edge circuit is worse than its 500-edge one. The corrupted prompts no longer contain a repeated name, so this corruption still switches off the redundant duplicate-detection mechanism, which would explain why EAP still lags. (You can check this with the missing-edges table and `joint_vs_individual` from above.)

The general point: "which edges matter?" has no answer without "for which corruption?". Both the circuit and the reliability of the approximation depend on it.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 4️⃣

- A circuit is evaluated by corrupting every edge *outside* it. Faithfulness measures how much of the clean behaviour survives, from 0 (corrupted) to 1 (clean).
- EAP-IG integrates receiver gradients along the path from the corrupted to the clean *input embeddings*, so it costs $m$ backward passes for all edges at once. It satisfies completeness, which EAP badly fails (+1.10 predicted vs −3.12 actual for the input's edges).
- On IOI with the ABC corruption, EAP's top 1,000 edges recover nothing on held-out prompts, while EAP-IG's recover 57%. Random edge sets recover nothing. On greater-than both methods work, with EAP-IG ahead at small circuit sizes (0.87 vs 0.70 at 100 edges).
- The edges EAP misses are partly approximation errors at MLP0, and partly **redundant** edges whose single-edge effects are tiny but whose joint effect is large: the 36 input edges into layer 0's heads have individual effects summing to +0.34, but a joint effect of −1.04.
- The deepest lesson: a perfect approximation to single-edge patching would *still* miss these edges. Circuit discovery asks how edges matter when many others are corrupted too, and integrating along a path where everything is partially corrupted is closer to that question.
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
## Denoising-direction attribution

Everything in this section was in the noising direction: gradients at the clean run, activations moving towards the corrupted ones. Repeat 1️⃣ and 2️⃣ in the denoising direction (gradients at the *corrupted* run, patching clean activations in), as in Neel Nanda's original post. Some questions:

- Does AP agree better or worse with exact patching for heads? For the residual stream at the name positions?
- For a single component, the denoising AP estimate is the tangent at $\alpha = 1$ of the curves from 2️⃣. Looking at those curves, can you predict which components AP will get right in the denoising direction?
- For EAP, does using corrupted-run gradients change which IOI edges are found, and how faithful the circuits are?

## Better integration

- We used a left Riemann sum so that one step reproduces EAP. A midpoint rule ($\alpha_k = (k + \tfrac12)/m$) converges much faster for smooth curves. Try it in `integrated_gradients_component` and `eap_ig_scores`. How many steps does EAP-IG then need for the completeness check to be within 1%?
- Hanna et al.'s code also offers **EAP-IG-activations**, which interpolates every *sender's* activation (rather than the input embeddings) between its clean and corrupted value. Marks et al. (2024) use this kind of activation-level integration for SAE features in their sparse feature circuits. Implement it, and compare its circuits' faithfulness with input-level EAP-IG. (See section [8.6] for circuits over SAE features.)

## Fixing attention saturation: AtP*

Kramár et al. (2024) propose **AtP\***, with two fixes to attribution patching: recomputing the attention softmax exactly when patching queries and keys (the "QK fix", aimed at exactly the saturation we saw with head 6.9), and **GradDrop**, which guards against cancellation between paths. You'll implement both in section [8.3], and can then rerun the head 6.9 path plot from 2️⃣ with the QK fix.

## Anomaly hunts

- **Head 9.6.** Wang et al. classify 9.6 as a name mover, which should *hurt* the logit difference when corrupted. In our noising experiments its effect is positive, in both exact patching and AP. Is this specific to the ABC corruption? Look at what 9.6 attends to on the corrupted prompts, and what its output does to the IO and S logits directly.
- **Which heads back each other up?** We argued that the layer-0 duplicate-detection edges are redundant. Use `joint_vs_individual` to find a *minimal* set of edges whose joint effect is large while each individual effect is small. Do the induction heads (5.5, 6.9) provide a second route to the same signal, as Wang et al.'s circuit suggests?
- **Faithfulness above 1 and below 0.** EAP-IG's 2,000-edge greater-than circuit scores 1.02, and its 50-edge IOI circuit −0.13. Find the edges responsible.

## Metrics and selection

- ACDC and several later papers score edges with the KL divergence between the clean and patched output distributions, rather than a task-specific metric. Implement a KL metric and compare the circuits it finds.
- We selected edges by top-$k$ absolute score. Real circuit-finding tools also prune "dangling" edges (edges whose sender has no path from the input, or whose receiver has no path to the logits), or grow the circuit greedily. Implement pruning and see whether it improves the faithfulness of small circuits.
- Miller et al. (2024) show that faithfulness scores depend a lot on details like the corruption, the metric, and whether you patch at every position. How robust are our conclusions to these choices? (The try-it-yourself cell is a start.)

## Further reading

- Neel Nanda (2023), [*Attribution Patching: Activation Patching At Industrial Scale*](https://www.neelnanda.io/mechanistic-interpretability/attribution-patching). The post that introduced attribution patching, with a careful discussion of where it fails.
- Syed, Rager & Conmy (2023), [*Attribution Patching Outperforms Automated Circuit Discovery*](https://arxiv.org/abs/2310.10348). Introduces edge attribution patching and compares it with ACDC.
- Hanna, Pezzelle & Belinkov (2024), [*Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms*](https://arxiv.org/abs/2403.17806). Introduces EAP-IG, and argues that circuits should be judged by faithfulness rather than overlap with a reference circuit. Their [EAP-IG library](https://github.com/hannamw/EAP-IG) uses the same graph as this section.
- Marks, Rager, Michaud, Belinkov, Bau & Mueller (2024), [*Sparse Feature Circuits*](https://arxiv.org/abs/2403.19647). Attribution patching and integrated gradients over SAE features and error terms (section [8.6]).
- Kramár, Lieberum, Shah & Nanda (2024), [*AtP\*: An efficient and scalable method for localizing LLM behaviour to components*](https://arxiv.org/abs/2403.00745). A systematic study of attribution patching's failure modes, and fixes for them (section [8.3]).
- Sundararajan, Taly & Yan (2017), [*Axiomatic Attribution for Deep Networks*](https://arxiv.org/abs/1703.01365). The original integrated gradients paper, including the completeness axiom.
- Wang et al. (2022), [*Interpretability in the Wild*](https://arxiv.org/abs/2211.00593), and Hanna, Liu & Variengien (2023), [*How does GPT-2 compute greater-than?*](https://arxiv.org/abs/2305.00586). The two circuits we used as test cases.
- Conmy et al. (2023), [*Towards Automated Circuit Discovery for Mechanistic Interpretability*](https://arxiv.org/abs/2304.14997). ACDC, the exact-patching method EAP was designed to approximate (section [8.4]).
- Miller, Chughtai & Saunders (2024), [*Transformer Circuit Faithfulness Metrics are not Robust*](https://arxiv.org/abs/2407.08734). How much faithfulness numbers depend on evaluation choices.
'''

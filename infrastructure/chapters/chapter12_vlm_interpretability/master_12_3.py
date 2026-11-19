# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
```python
[
    {"title": "A World to Ask Questions About", "icon": "1-circle-fill", "subtitle": "(10%)"},
    {"title": "Building the VLM", "icon": "2-circle-fill", "subtitle": "(15%)"},
    {"title": "Training and Modality Controls", "icon": "3-circle-fill", "subtitle": "(15%)"},
    {"title": "Where Does the Answer Come From?", "icon": "4-circle-fill", "subtitle": "(30%)"},
    {"title": "Is a Real VLM Doing the Same?", "icon": "5-circle-fill", "subtitle": "(10%)"},
    {"title": "Held-out Quiz", "icon": "bullseye", "subtitle": "(15%)"},
    {"title": "Bonus", "icon": "star", "subtitle": "(5%)"},
]
```
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# [12.3] Build a Mini VLM on SigLIP and GPT-2
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
Here is a picture: a blue circle in the top-right corner, a yellow triangle in the bottom-left, and a green square in the bottom-right. Here are three questions about it, and the
answers of the model you'll build (you'll run this scene yourself at the end of Part 3, and we take the highest-probability valid answer):

```
Question: What colour is the circle? Answer:                  blue
Question: How many shapes are there? Answer:                  three
Question: Is the circle at the top or the bottom? Answer:     top
```

Some numbers about this model, all on held-out scenes. It names the colour of a shape correctly on essentially every question (100.0% to one decimal). Show it a blank image instead and the same colour questions
get 19% right, and the best any model could do without seeing the image is 16% (one in six, give or take). And one result won't behave: asked whether a shape is on the left or the right, the model is
right 94.5% of the time, while top or bottom is 99.7%, and we don't know why. You'll measure all of these, and find out which part of the network is responsible for the ones that work.

The model is a **vision-language model** (VLM), assembled from three pieces: a frozen image encoder (SigLIP), a small language model (GPT-2 Small), and a thin trained
**projector** that translates one into the other. The encoder and the language model are both pretrained and we don't change the encoder at all; **what you build is the projector, the
image tokens, the training and the controls.** This is the recipe of LLaVA (Liu et al., 2023), shrunk until it trains in a few minutes on one GPU. The image is cut into patches,
the patches are embedded by the vision encoder, and the projector turns them into a short sequence of **image tokens** that sit in the language model's context window *in front of*
the question. From then on the language model treats them like any other tokens: it attends to them, moves their contents around, and eventually writes an answer. Qwen-VL,
InternVL, PaliGemma and LLaVA all work this way; the differences are in the details and in the scale.

That design raises questions that a text-only transformer never has to ask:

- **Is the model looking at the image at all?** A VLM trained on question-answer pairs can score well by exploiting statistics of the questions. If you blank the image or swap in the wrong one, does the answer change?
- **If it is looking, at *what*?** The answer to "what colour is the circle?" should come from the tokens that cover the circle, not from the background. How would you check that, and what would count as evidence?
- **And *when*?** The answer is computed by twelve transformer blocks. At which layer does visual information leave the image tokens and reach the position where the answer is written?

These are exactly the questions asked by recent interpretability work on real VLMs, for example Neo et al. (2025), *Towards Interpreting Visual Information Processing in
Vision-Language Models*, and Zhang et al. (2025), *Cross-modal Information Flow in Multimodal Large Language Models*. The trouble with studying a 7B-parameter LLaVA is that you
never know the ground truth, and every experiment is slow. So in this section we build a VLM in a **world with exact ground truth**: images of 1 to 3 coloured shapes, and questions
about colour, shape, count and position whose answers we can compute from the scene description. We train the model, and then take it apart.

Since a trained model is very good at producing plausible-looking answers, most of the section is about **controls**. Every headline number comes with baselines that should fail,
and you will check that they do:

- **blank and noise images**, and **shuffled image-question pairs** (the right question with someone else's image);
- a **text-only model** trained on the same questions with no image at all, and the best an image-blind guesser could do in principle;
- a **random, untrained projector**;
- for the interpretability experiments: **random vectors**, **background tokens**, **same-size random sets of tokens**, **other objects' tokens** and the **tokens the answer position attends to most**.

You'll also find things that don't match the tidy story. The logit lens does not cleanly show colour words at the object's tokens, and background tokens carry some decodable colour
information despite having much smaller mean patching effects. Reporting those honestly is part of the exercise.

**In this section you will:**

- build a projector that turns image patches into GPT-2 tokens, and train it;
- test the VLM against blank, noise, shuffled, random-projector, text-only and image-only controls;
- find where the answer comes from with lenses, attention and patching, against same-size background controls;
- check whether Qwen3-VL shows the same pattern. The signature result is one figure: how much of the answer's contrast patching restores at each depth, for the object's tokens, the answer position and
  three controls, in your mini VLM and in Qwen3-VL on top of each other.

**Prerequisites.** You should be comfortable with TransformerLens hooks and the residual stream ([1.2 Intro to Mech Interp](https://learn.arena.education/chapter1_transformer_interp/02_intro_mech_interp/)),
a basic PyTorch training loop, and activation patching ([1.4.1 Indirect Object Identification](https://learn.arena.education/chapter1_transformer_interp/21_ioi/)). Section
[7.1](../../../chapter7_activation_to_language/exercises/part1_lenses_patchscopes/7.1_Logit_Lens_Tuned_Lens_and_Patchscopes_exercises.ipynb) (logit lens and Patchscopes) is useful background, and
[12.1](../part1_clip_siglip/12.1_CLIP_and_SigLIP_from_Scratch_exercises.ipynb) (CLIP and SigLIP from Scratch) explains the vision encoder we use here, though you don't need it: we treat SigLIP as a frozen feature extractor.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Content & Learning Objectives

### 1️⃣ A World to Ask Questions About

You'll build the world the VLM lives in: scenes of coloured shapes, five kinds of question with exactly computable answers, and the frozen SigLIP vision encoder that turns each
scene into 196 patch embeddings. You'll also work out what the best *image-blind* guesser would score on each question kind. That number is the line every later result has to beat.

> ##### Learning Objectives
>
> - Design a synthetic task where the answer genuinely depends on the image, with ground truth computed from the scene description rather than from pixels
> - Compute the best accuracy achievable without looking at the image, as a floor for every later comparison
> - Extract patch embeddings from a frozen vision encoder (SigLIP) and understand their shape, ordering and normalisation

### 2️⃣ Building the VLM

You'll implement the three pieces of a LLaVA-style VLM: a patch-merging projector that maps SigLIP's 196 patch embeddings to 49 tokens in GPT-2's embedding space, the assembly of
`[BOS] [image tokens] [question]` into the residual stream, and the forward pass that reads the answer from the last question token. You'll test each against an independent oracle,
and compare the untrained VLM with uniform-chance and image-blind baselines.

> ##### Learning Objectives
>
> - Understand the vision encoder, projector, language model decomposition of a LLaVA-style VLM, and why image tokens can be treated like word embeddings
> - Implement patch merging and an MLP projector, and verify which patches each image token depends on
> - Inject non-token embeddings into a TransformerLens model with `start_at_layer`, and read logits from a chosen position
> - Evaluate with forced choice among the valid answers, so that chance level is well defined

### 3️⃣ Training and Modality Controls

You'll train the projector alone (with GPT-2 frozen), then the projector and GPT-2 together, and compare. Then comes the question that matters: is the trained model *using the
image*? You'll evaluate it on blank images, noise images, shuffled image-question pairs, a random projector and a text-only model, and plot all of them next to the trained VLM. Last,
you'll take the question away and keep the picture, to see how much of the answer the image alone can give.

> ##### Learning Objectives
>
> - Implement a VLM training loop with frozen and trainable parameter groups
> - Build the standard modality controls (blank, noise, shuffled, random projector, text-only) and check that each one fails
> - Read a VLM's accuracy against the best image-blind baseline instead of against zero
> - See what a model does when its image carries no information, and what it can still do when the question is missing

### 4️⃣ Where Does the Answer Come From?

You'll open up the trained model with four tools of increasing strength: a logit lens on image tokens (what do they decode to?), attention from the answer position (what does it
read?), activation patching of image tokens (what is *sufficient* to change the answer?), and attention knockout (which routes are *necessary*?). Every tool is run against a
control: background tokens, same-size random token sets, random vectors, and the tokens the answer position attends to most.

> ##### Learning Objectives
>
> - Apply the logit lens to image tokens, and see why a decoding that looks at the whole vocabulary can miss information that a restricted decoding finds
> - Measure attention from the answer position to a token set, and compare to a same-size random control
> - Implement activation patching on image tokens with clean/corrupted counterfactual scenes, and read the layer-by-layer recovery curve, stopping on any pair whose clean and corrupted answers do not differ
> - Compare the object's tokens with the tokens the answer position attends to most
> - Use attention knockout to test which information routes the model needs, and tell "decodable", "attended", "sufficient" and "necessary" apart

### 5️⃣ Is a Real VLM Doing the Same?

You'll repeat the modality controls and the patching experiment on Qwen3-VL-2B-Instruct, a real VLM with a 28-layer language model, on exactly the same scenes, and draw the
lesson's signature figure: the two recovery curves on one relative-depth axis, with uncertainty bands.

> ##### Learning Objectives
>
> - Run forced-choice evaluation and modality controls on a real VLM
> - Implement activation patching on a Hugging Face model with forward pre-hooks
> - Compare a toy and a real VLM on a relative-depth axis, with a band showing how much the answer depends on which counterfactual pairs you drew, and identify the architectural details (such as DeepStack) that complicate the comparison

### 🎯 Held-out Quiz

Four experiments we ran on your mini VLM and didn't show you. Each changes one thing in a call you made. You'll predict each result and write down why, then see what really
happened and what a good mental model would have said.

> ##### Learning Objectives
>
> - Predict the result of a change to a run or an experiment from results you've already measured
> - Name the result your prediction rests on, and notice when a result you trusted stops at the edge of what you tested
> - Find the places where your picture of "the answer lives in the object's tokens" is too narrow

### ☆ Bonus

What this lesson doesn't show, and suggestions for going further: the original two-stage LLaVA recipe, why *left or right* is the hardest question for a flattened image, LoRA
instead of full fine-tuning, a frozen random projector, compositional generalisation, language priors versus evidence, a tuned lens for image tokens, scaling up, and the papers
behind everything in this section.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Setup code

This lesson needs a GPU: it took about 10 minutes end to end on an NVIDIA RTX 5090 laptop GPU, with a peak of about 12 GB of GPU memory (the quiz's key runs are included). Downloads are about 6 GB, mostly Qwen3-VL-2B (about 4.4 GB). It is not practical on CPU.
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

import copy
import sys
from collections import Counter
from pathlib import Path

import einops
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

# Make sure exercises are in the path
chapter = "chapter12_vlm_interpretability"
section = "part3_mini_vlm_from_scratch"
root_dir = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / chapter).exists())
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

import part3_mini_vlm_from_scratch.tests as tests
import part3_mini_vlm_from_scratch.utils as utils
from part3_mini_vlm_from_scratch.utils import COLOURS, N_IMAGE_TOKENS, SHAPES, TOKEN_GRID, Scene, SceneObject

MAIN = __name__ == "__main__"

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 1️⃣ A World to Ask Questions About
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Why a toy world?

A real VLM is trained on hundreds of thousands of photographs with captions and conversations, and when it answers "what is the man holding?" nobody can tell you exactly which
pixels the answer should depend on. We want the opposite: a task small enough to train in minutes, where **the answer provably depends on the image** and where we know exactly
which part of the image it depends on.

Our world is a canvas of 224 × 224 pixels (SigLIP's input size) with a grey background and **one to three coloured shapes**. Each shape has one of three forms (circle, square,
triangle) and one of six colours (red, green, blue, yellow, purple, orange). Shapes are placed one per quadrant of a 2 × 2 layout, with a little random jitter and a random size.
Two rules keep questions unambiguous: **within a scene, every shape form is different, and every colour is different.** So "the circle" and "the red object" each name exactly one
object.

Let's look at a few scenes. The helper functions that generate and draw them live in `utils.py`: this is the world we ask questions about, not part of the method.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

demo_rng = np.random.default_rng(12)
demo_scenes = [utils.sample_scene(demo_rng) for _ in range(8)]
demo_images = utils.render_scenes(demo_scenes)
quadrant_names = {(0, 0): "top left", (0, 1): "top right", (1, 0): "bottom left", (1, 1): "bottom right"}
demo_captions = [
    "\n".join(f"{o.colour} {o.shape} ({quadrant_names[o.quadrant]})" for o in scene.objects) for scene in demo_scenes
]
utils.plot_scene_gallery(demo_images, demo_captions, ncols=4, title="Eight random scenes")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/scene_gallery.png" width="900">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Five kinds of question

Each scene comes with five kinds of question. Here is the full list, with the templates we'll use (the prompt always starts with a space, and ends with `Answer:`):

| kind | prompt template | possible answers |
|---|---|---|
| `colour` | ` Question: What colour is the {shape}? Answer:` | red, green, blue, yellow, purple, orange |
| `shape` | ` Question: What shape is the {colour} object? Answer:` | circle, square, triangle |
| `count` | ` Question: How many shapes are there? Answer:` | one, two, three |
| `horizontal` | ` Question: Is the {shape} on the left or the right? Answer:` | left, right |
| `vertical` | ` Question: Is the {shape} at the top or the bottom? Answer:` | top, bottom |

Two design choices are worth pausing on.

**Every answer is a single GPT-2 token.** (Check this: `" red"`, `" circle"`, `" two"`, `" left"` and `" bottom"` are each one token, and so is every other answer.) The model's
answer is then just one entry of the next-token distribution at the end of the prompt. That makes accuracy, logit differences and activation patching all well defined: there is
exactly one place where "the answer" is written, namely the position of the final `:`.

**The labels come from the scene description, never from pixels.** The renderer takes a list of `SceneObject`s and draws them, so the answer to every question is a function of that
list. If you wanted to know what the right answer is, you would never look at the image; if the model gets it right, it must have extracted the information from the image. That is
what makes this a clean testbed.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - write the ground truth

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 5-10 minutes on this exercise.
> ```

Implement `question_and_answer`, which takes a `Scene`, a question `kind` and the index `target` of the object the question is about, and returns `(prompt, answer)` following the
table above. Each `SceneObject` has `.colour`, `.shape` and `.quadrant`, where `quadrant = (row, col)` and **row 0 is the top, col 0 is the left**. The `count` question ignores
`target` and counts `scene.objects`. Both strings start with a space, as in the table; the prompt ends with `Answer:`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

KINDS = ["colour", "shape", "count", "horizontal", "vertical"]
NUMBER_WORDS = {1: "one", 2: "two", 3: "three"}
ANSWER_WORDS = {  # the valid answers of each kind (each is a single GPT-2 token once a space is prepended)
    "colour": list(COLOURS),
    "shape": SHAPES,
    "count": ["one", "two", "three"],
    "horizontal": ["left", "right"],
    "vertical": ["top", "bottom"],
}


def question_and_answer(scene: Scene, kind: str, target: int) -> tuple[str, str]:
    """
    Returns the (prompt, answer) for a question of the given kind about scene.objects[target].
    E.g. (" Question: What colour is the circle? Answer:", " red").
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    obj = scene.objects[target]
    if kind == "colour":
        return f" Question: What colour is the {obj.shape}? Answer:", " " + obj.colour
    if kind == "shape":
        return f" Question: What shape is the {obj.colour} object? Answer:", " " + obj.shape
    if kind == "count":
        return " Question: How many shapes are there? Answer:", " " + NUMBER_WORDS[len(scene.objects)]
    if kind == "horizontal":
        return f" Question: Is the {obj.shape} on the left or the right? Answer:", " " + ("left" if obj.quadrant[1] == 0 else "right")
    if kind == "vertical":
        return f" Question: Is the {obj.shape} at the top or the bottom? Answer:", " " + ("top" if obj.quadrant[0] == 0 else "bottom")
    raise ValueError(f"Unknown question kind {kind!r}")
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_question_and_answer(question_and_answer)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

One `if` per kind. For `horizontal` the answer is `" left"` when `obj.quadrant[1] == 0` and `" right"` otherwise; for `vertical` it is `" top"` when `obj.quadrant[0] == 0` and `"
bottom"` otherwise. For `count`, use the `NUMBER_WORDS` dictionary.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Let's use it to make the datasets. We render **5,000 training scenes** and **1,000 test scenes** from different random seeds, and for every test scene we prepare one question of
each kind about a randomly chosen object. The main VQA tables use these 5,000 held-out questions; later analyses use stated subsets. Different seeds give a separate sample, not a
guarantee of new colour–shape combinations: those combinations can appear in both splits. Before we do anything else, let's check the single-token claim and look at the first few
questions.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

gpt2 = HookedTransformer.from_pretrained("gpt2-small", device=device, revision="607a30d783dfa663caf39e06633721c8d4cfcd7e")
gpt2.requires_grad_(False)
gpt2.tokenizer.pad_token = gpt2.tokenizer.eos_token  # prompts are right-padded with the end-of-text token

for kind, words in ANSWER_WORDS.items():
    n_tokens = [gpt2.to_tokens(" " + w, prepend_bos=False).shape[1] for w in words]
    assert all(n == 1 for n in n_tokens), f"Some {kind} answers are not single tokens"
print("All answers are single GPT-2 tokens.")

train_scenes = [utils.sample_scene(np.random.default_rng(seed)) for seed in range(100_000, 105_000)]
test_scenes = [utils.sample_scene(np.random.default_rng(seed)) for seed in range(200_000, 201_000)]


def make_question_set(scenes: list[Scene], seed: int) -> dict[str, tuple[list[str], list[str]]]:
    """For every kind: (prompts, answers), one question per scene about a random object."""
    rng = np.random.default_rng(seed)
    questions = {}
    for kind in KINDS:
        qa = [question_and_answer(s, kind, int(rng.integers(len(s.objects)))) for s in scenes]
        questions[kind] = ([prompt for prompt, _ in qa], [answer for _, answer in qa])
    return questions


test_questions = make_question_set(test_scenes, seed=0)
for kind in KINDS:
    prompts, answers = test_questions[kind]
    print(f"{kind:<10} {prompts[0]!r:<62} -> {answers[0]!r}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">All answers are single GPT-2 tokens.
colour     ' Question: What colour is the triangle? Answer:'              -&gt; ' blue'
shape      ' Question: What shape is the yellow object? Answer:'          -&gt; ' square'
count      ' Question: How many shapes are there? Answer:'                -&gt; ' three'
horizontal ' Question: Is the triangle on the left or the right? Answer:' -&gt; ' left'
vertical   ' Question: Is the circle at the top or the bottom? Answer:'   -&gt; ' top'</pre>

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## How well can you do without looking?

Before building anything, let's ask how much of the task can be solved *without the image*. Questions have structure: a model that has seen a few thousand of them knows that
`colour` questions are answered with colour words, that `horizontal` questions are answered with `left` or `right`, and that scenes with two objects are more common than scenes
with one. A model could exploit all of that and still be blind.

The cleanest way to quantify this is the best **constant guess**: for each question kind, always answer with the most common answer in the training set, and see how often that is
right on the test set. No model that ignores the image can beat this on average (a model that sees the question and nothing else can do no better than the best fixed answer, since
for a given kind the question text carries no further information about the answer). It's the floor that every later result has to be compared against, so we compute it first.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - the best image-blind guesser

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 5-10 minutes on this exercise.
> ```

Implement `best_constant_guess`. It takes the training answers and the test answers, each a dictionary from question kind to a list of answer strings, and returns a dictionary from
kind to a pair `(guess, test_accuracy)`: the most frequent training answer for that kind, and the fraction of test answers equal to it.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def best_constant_guess(
    train_answers: dict[str, list[str]], test_answers: dict[str, list[str]]
) -> dict[str, tuple[str, float]]:
    """For each question kind, the most common training answer and its accuracy on the test answers."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    result = {}
    for kind, answers in train_answers.items():
        guess = Counter(answers).most_common(1)[0][0]
        result[kind] = (guess, float(np.mean([a == guess for a in test_answers[kind]])))
    return result
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_best_constant_guess(best_constant_guess)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`collections.Counter(answers).most_common(1)` gives the most frequent element. The accuracy is the mean of `answer == guess` over the *test* answers.
</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

train_questions = make_question_set(train_scenes, seed=1)
baseline = best_constant_guess(
    {kind: train_questions[kind][1] for kind in KINDS}, {kind: test_questions[kind][1] for kind in KINDS}
)
IMAGE_BLIND = {kind: acc for kind, (_, acc) in baseline.items()}
display(pd.DataFrame({"best constant guess": {k: g.strip() for k, (g, _) in baseline.items()}, "test accuracy": IMAGE_BLIND}).round(3))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">           best constant guess  test accuracy
colour                  yellow          0.161
shape                   square          0.326
count                      two          0.410
horizontal                left          0.488
vertical                   top          0.491</pre>

</details>

<details>
<summary>Question - why are the numbers what they are? Which row would change if scenes had a different number of objects?</summary>

Uniform colours suggest about 1/6, shapes about 1/3, and binary position questions about 1/2; the finite test sample gives 0.161, 0.326, 0.488 and 0.491 for the training-majority
answers. Count is different: the most frequent training answer is `two`, which achieves 0.410 on test. Changing the distribution of object counts directly changes this count
baseline. It can also change which scenes and target objects are sampled, so recompute every baseline after changing the generator rather than assuming exact chance values.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## The vision encoder

Now for the eyes. We use **SigLIP** (Zhai et al., 2023), the vision tower of Google's `siglip-base-patch16-224`: a ViT-B/16 that was trained, together with a text tower we won't
use, to match images to captions with a sigmoid loss (the family of CLIP-style models from [12.1]). It works like this:

1. Cut the 224 × 224 image into 16 × 16 pixel patches: a **14 × 14 grid, 196 patches**.
2. Embed every patch linearly (and add a learned position embedding), then run 12 transformer blocks with *bidirectional* attention.
3. The output is one 768-dimensional vector per patch. (SigLIP then pools them with an attention-pooling head into a single image embedding for image-text matching. VLMs don't use that: they want the *patch* embeddings, which keep the spatial layout.)

We will treat SigLIP as a **frozen feature extractor**: we run every image through it once and never update it. This is common practice: LLaVA keeps its CLIP encoder frozen
throughout, and we get a big speed-up for free, since training only needs to touch cached patch features.

One thing to know about the preprocessing: SigLIP expects pixel values rescaled to $[0, 1]$ and then normalised with mean 0.5 and standard deviation 0.5, i.e. mapped to $[-1, 1]$.
In symbols, $x \mapsto (x/255 - 0.5)/0.5$.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - extract patch features

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement `extract_patch_features`. It takes the (frozen) vision model and an array of images of shape `(n, 224, 224, 3)` with `uint8` pixels, and returns the final hidden state of
shape `(n, 196, 768)`. Process the images in chunks of `batch_size` to bound memory, convert them to the right range, permute to channels-first, and call
`vision_model(pixel_values=...).last_hidden_state` without tracking gradients. The vision model runs in `float16`, so cast the pixel values to `vision_model.dtype`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

siglip = utils.load_siglip(device)
print(f"SigLIP vision tower: {sum(p.numel() for p in siglip.parameters()) / 1e6:.0f}M parameters, dtype {siglip.dtype}")

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.no_grad()
def extract_patch_features(
    vision_model, images: np.ndarray, batch_size: int = 128
) -> Float[Tensor, "n patch d_vision"]:
    """
    Runs the frozen vision tower on uint8 images of shape (n, 224, 224, 3) and returns its last hidden state,
    one 768-dim embedding per 16x16 patch: shape (n, 196, 768).
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    features = []
    for start in range(0, len(images), batch_size):
        pixels = t.as_tensor(images[start : start + batch_size]).to(vision_model.device)
        pixels = pixels.permute(0, 3, 1, 2).float() / 255  # (n, 3, 224, 224) in [0, 1]
        pixels = (pixels - 0.5) / 0.5  # SigLIP normalisation: mean = std = 0.5
        features.append(vision_model(pixel_values=pixels.to(vision_model.dtype)).last_hidden_state)
    return t.cat(features)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_extract_patch_features(extract_patch_features, siglip)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

For each chunk: `t.as_tensor(images[start:start+batch_size]).to(vision_model.device)` has shape `(n, 224, 224, 3)`; `.permute(0, 3, 1, 2)` makes it `(n, 3, 224, 224)`. Convert to
float, divide by 255, subtract 0.5 and divide by 0.5, cast to `vision_model.dtype`, and call the model. Concatenate the chunks with `t.cat`.
</details>

<details>
<summary>Help - the test says my features differ from Hugging Face's</summary>

The most common mistakes are forgetting the `/ 255` (so pixels are in $[0, 255]$), forgetting the `- 0.5) / 0.5` (so they're in $[0, 1]$ instead of $[-1, 1]$), and feeding
channels-last images (the model expects `(n, 3, 224, 224)`). The test compares against Hugging Face's own `SiglipImageProcessor`, which does the same preprocessing.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
Now we featurise both datasets. Rendering 5,000 scenes takes about 15 seconds on one CPU core, and SigLIP takes about 10 seconds on a GPU. We keep the features in `float16` on the
GPU: 5,000 × 196 × 768 values is about 1.5 GB.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

def featurise(scenes: list[Scene], chunk: int = 250) -> Tensor:
    """Render scenes and run them through SigLIP, chunk by chunk (so we never hold all the images at once)."""
    return t.cat([extract_patch_features(siglip, utils.render_scenes(scenes[i : i + chunk])) for i in tqdm(range(0, len(scenes), chunk))])


train_features = featurise(train_scenes)
test_features = featurise(test_scenes)
print(f"train features: {tuple(train_features.shape)}, test features: {tuple(test_features.shape)}, dtype {train_features.dtype}")
print(f"mean patch-embedding norm: {train_features[:500].float().norm(dim=-1).mean():.1f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">train features: (5000, 196, 768), test features: (1000, 196, 768), dtype torch.float16
mean patch-embedding norm: 53.7</pre>

</details>

Note the norm. SigLIP's output patch embeddings are large (a norm of about 50, which for 768 dimensions means entries of order 2) compared to GPT-2's token embeddings (a norm of
about 3-4). The projector will have to take care of that rescaling, and it can: it's trained.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### What does SigLIP see?

Each of the 196 patch embeddings is a 768-dimensional vector. To get a feel for what they hold, we colour each patch by its **top three principal components** (PCA fit jointly over
the patches of all the scenes shown, so a colour means the same thing in every panel; PCA is built from scratch in 0.7 Foundations Toolkit, Part 2). If SigLIP's patch embeddings
are doing their job, patches covering a shape should look different from background patches, and different from each other when the shapes differ.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

utils.plot_patch_pca(demo_images[:4], featurise(demo_scenes[:4]))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/siglip_patch_pca.png" width="900">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret this picture?</summary>

The false-colour patch maps have structure near object locations and across the background, but their RGB channels are the top three principal components, not the scene's colour
labels. A red object need not produce a red PCA patch. SigLIP is contextual, so even a patch geometrically outside an object can encode information about it. This exploratory
picture motivates the controlled readouts and interventions below; it is not a localization or task-accuracy test.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 1️⃣

- We have a world where **answers are computed from the scene description**, five kinds of question with single-token answers, and the best image-blind score for each kind: that is the line a model must clear to prove it uses the image.
- SigLIP turns a 224 × 224 image into **196 patch embeddings** (a 14 × 14 grid of 768-dimensional vectors), and we run it once per image and cache the result. It is frozen, so everything that follows only has to train what sits on top of it.
- The patch embeddings look *local* (patches over a shape differ from background patches), but, as we'll see in section 4️⃣, they also carry some information about the whole image. That will matter when we interpret patching experiments.

Next we build the part that makes a VLM a VLM: the projector, and the machinery that puts image tokens into a language model's context.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 2️⃣ Building the VLM
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## The LLaVA recipe

LLaVA (Liu et al., 2023, [*Visual Instruction Tuning*](https://arxiv.org/abs/2304.08485)) made a simple observation: you don't need a new architecture to give a language model eyes. Take a pretrained vision encoder $V$, a pretrained language model $L$, and connect them with a small trainable **projector** $P$ that maps the encoder's patch embeddings into the language model's input embedding space:

$$
h = V(\text{image}) \in \mathbb{R}^{196 \times 768}, \qquad z = P(h) \in \mathbb{R}^{49 \times 768}, \qquad \text{logits} = L\big([\,\text{BOS};\; z;\; E(q)\,]\big)
$$

where $E(q)$ are the ordinary token embeddings of the question. The rows of $z$ are "image tokens": vectors that live where word embeddings live, and that the language model reads
as if they were words. Nothing in the language model's architecture changes. Here is how our version compares with the original:

| | LLaVA-1.5 | our mini VLM |
|---|---|---|
| vision encoder | CLIP ViT-L/14 at 336px (frozen) | SigLIP ViT-B/16 at 224px (frozen) |
| image tokens | 576 | **49** (after merging 2 × 2 patches) |
| projector | 2-layer MLP on each patch | 2-layer MLP on each merged 2 × 2 block |
| language model | Vicuna 7B / 13B (32 / 40 layers) | **GPT-2 Small** (124M, 12 layers) |
| training data | 558k captions, then 665k instruction examples | 5,000 synthetic scenes × 5 question kinds |
| training | days on 8 A100s | about 4 minutes on one GPU |

**Why merge patches?** 196 image tokens would make every sequence 3-4 times longer, and attention cost grows quadratically with length. Several current VLMs (Qwen2-VL, InternVL,
and the Qwen3-VL we use in section 5️⃣) therefore concatenate each $2 \times 2$ block of neighbouring patch embeddings before projecting, so that one image token summarises a $32
\times 32$ pixel region. We do the same. The merge is a pure reshape: the 4 patch embeddings of a block are stacked into one vector of width $4 \times 768 = 3072$, and the
projector's first linear layer can then mix them however it likes.

**Where do the image tokens sit?** GPT-2 has learned absolute position embeddings, and we simply give the image tokens positions 1 to 49 (position 0 is the BOS token), in raster
order of the $7 \times 7$ grid, followed by the question. GPT-2 was never told that positions 1-49 form a grid. It will have to learn, during fine-tuning, how to use them. (This is
a real limitation of 1-D position embeddings for images, which is why Qwen2-VL and others use 2-D rotary embeddings. We'll run into it in the bonus section.)

Here is the whole thing:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

utils.plot_architecture()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/vlm_architecture.png" width="900">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - merge patches

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement `merge_patches`. It takes patch embeddings of shape `(batch, 196, d)` in raster order (patch `p` sits at row `p // 14`, column `p % 14`) and returns tokens of shape
`(batch, 49, merge² · d)`. Output token `7 * i + j` should be the concatenation, **in this order**, of the patches at grid positions $(2i, 2j)$, $(2i, 2j+1)$, $(2i+1, 2j)$, $(2i+1,
2j+1)$ (row, column), each a vector of length `d`. The function should work for any `merge` that divides the grid size.

This is a nice use of `einops.rearrange`: first split the flat patch index into `(row, col)`, then split each of row and column into `(block index, offset within block)`, and
finally merge the two offsets into the channel dimension.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def merge_patches(
    patch_features: Float[Tensor, "batch patch d"], merge: int = 2
) -> Float[Tensor, "batch token merged_d"]:
    """
    Concatenates every (merge x merge) block of neighbouring patches into one vector.
    (batch, 14*14, d) -> (batch, 7*7, merge*merge*d), blocks in raster order, patches inside a block in raster order.
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    grid = int(patch_features.shape[1] ** 0.5)
    x = einops.rearrange(patch_features, "b (gh gw) d -> b gh gw d", gh=grid)
    return einops.rearrange(x, "b (h m1) (w m2) d -> b (h w) (m1 m2 d)", m1=merge, m2=merge)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_merge_patches(merge_patches)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Two `rearrange` calls are easiest to read. First `"b (gh gw) d -> b gh gw d"` with `gh=grid` (where `grid = int(patch**0.5)`). Then split height and width into blocks: `"b (h m1)
(w m2) d -> b (h w) (m1 m2 d)"` with `m1=merge, m2=merge`. In einops, a group like `(h m1)` means "h blocks of m1 rows each", so `m1` is the offset inside a block, and `(m1 m2 d)`
puts the within-block offsets in front of the feature dimension, in raster order.
</details>

<details>
<summary>Help - the test says token (0, 1) has the wrong patches</summary>

Check which of the two orders you got: `(m1 m2 d)` stacks patches in raster order inside the block (what we want), while `(d m1 m2)` would interleave features of different patches.
Also check that you split `(h m1)` and not `(m1 h)`: the first factor is the *block index*.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - the projector

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement the `Projector` module: merge the patches, then apply a **two-layer MLP**, `Linear(merge² · d_vision, d_model)`, then `GELU`, then `Linear(d_model, d_model)`. This is the
connector LLaVA-1.5 found to work better than a single linear layer. In the forward pass, cast the input to `float32` (the cached features are `float16`).

The test is a good illustration of an *independent oracle* for something that has random weights: instead of comparing numbers, it checks **which input patches each output token
depends on**, using gradients. Output token 0 should depend on patches 0, 1, 14 and 15 and nothing else.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

class Projector(nn.Module):
    """Merge 2x2 blocks of vision patches, then map them to the language model's embedding space with an MLP."""

    def __init__(self, d_vision: int = 768, d_model: int = 768, merge: int = 2):
        super().__init__()
        self.merge = merge
        # EXERCISE
        # raise NotImplementedError()
        # END EXERCISE
        # SOLUTION
        self.mlp = nn.Sequential(nn.Linear(d_vision * merge**2, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        # END SOLUTION

    def forward(self, patch_features: Float[Tensor, "batch patch d_vision"]) -> Float[Tensor, "batch token d_model"]:
        # EXERCISE
        # raise NotImplementedError()
        # END EXERCISE
        # SOLUTION
        return self.mlp(merge_patches(patch_features.float(), self.merge))
        # END SOLUTION


# HIDE
if MAIN:
    tests.test_projector(Projector)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`nn.Sequential(nn.Linear(...), nn.GELU(), nn.Linear(...))` is all you need for the MLP. In `forward`, call `merge_patches(patch_features.float(), self.merge)` first.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## What GPT-2 makes of the question on its own

Before wiring the two together, let's see what the language model does with a question when it has no image. We run the prompt through GPT-2 and look at its top few predictions for
the next token.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

prompt = test_questions["colour"][0][0]
with t.no_grad():
    probs = gpt2(gpt2.to_tokens(prompt))[0, -1].softmax(-1)
print(prompt)
for p, i in zip(*probs.topk(6)):
    print(f"  {gpt2.to_string(i)!r:<12} {p:.3f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"> Question: What colour is the triangle? Answer:
  ' The'       0.204
  ' It'        0.115
  ' A'         0.040
  ' This'      0.034
  '\n'         0.031
  ' I'         0.023</pre>

</details>

GPT-2 is not about to say a colour. It has no idea what's being asked about, which is as it should be. Any ability to answer will have to come from training the VLM.

## Assembling the input

Now the central piece of plumbing. The language model expects the *residual stream* at the start of block 0: for each position, the embedding of the token there plus the position
embedding. We have two kinds of "tokens": real words (embedded with `model.embed`, which is a lookup in the embedding matrix `model.W_E`) and image tokens (already vectors,
produced by the projector). We need to build the sequence $[\text{BOS}, z_1, \dots, z_{49}, q_1, \dots, q_n]$ and add position embeddings to it ourselves, because we are about to
bypass the model's own embedding step.

TransformerLens lets us do that: `model(x, start_at_layer=0)` treats `x` as the residual stream before block 0 and skips the embedding entirely. (And
`stop_at_layer=model.cfg.n_layers` makes it return the residual stream *after* the last block, before the final layer norm. We'll use that in the next exercise.)
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - build the input embeddings

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement `build_input_embeddings`. It takes the model, image embeddings of shape `(batch, 49, d_model)` (or `None` for a text-only model) and question tokens of shape `(batch,
n_question)`, and returns the residual stream input of shape `(batch, 1 + 49 + n_question, d_model)`: the BOS embedding, then the image embeddings, then the question token
embeddings, with positional embeddings added to all of them.

- The BOS embedding is `model.W_E[model.tokenizer.bos_token_id]` (GPT-2's `<|endoftext|>`), repeated across the batch.
- The positional embeddings are `model.W_pos[:seq_len]`, shape `(seq_len, d_model)`.
- If `image_embeds is None`, build `[BOS, question]` instead.

The test uses a neat oracle: if the "image embeddings" are just the embeddings of some ordinary words, then the multimodal input must be *exactly* what TransformerLens computes for
the corresponding plain text.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def build_input_embeddings(
    model: HookedTransformer,
    image_embeds: Float[Tensor, "batch image_token d_model"] | None,
    question_tokens: Int[Tensor, "batch question"],
) -> Float[Tensor, "batch seq d_model"]:
    """
    The residual stream before block 0 for [BOS, image tokens, question tokens]:
    token (or image) embeddings plus positional embeddings. With image_embeds=None: [BOS, question tokens].
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    batch = question_tokens.shape[0]
    bos = model.W_E[model.tokenizer.bos_token_id].expand(batch, 1, -1)
    parts = [bos] + ([image_embeds] if image_embeds is not None else []) + [model.embed(question_tokens)]
    x = t.cat(parts, dim=1)
    return x + model.W_pos[: x.shape[1]]
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_build_input_embeddings(build_input_embeddings, gpt2)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Build a list `[bos, image_embeds, model.embed(question_tokens)]` (without the middle one if `image_embeds is None`), concatenate along dimension 1, then add `model.W_pos[:
x.shape[1]]` (which broadcasts over the batch).
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - the forward pass

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15 minutes on this exercise.
> ```

Now put everything together. `vlm_answer_logits` takes the model, the projector, SigLIP patch features `(batch, 196, 768)` (or `None` for a text-only model), question tokens
`(batch, n_question)` and the length of every question `(batch,)`, and returns the **logits for the answer token, shape `(batch, d_vocab)`**: the model's next-token prediction at
the last real question token, i.e. the final `:` of `Answer:`.

Steps: project the features, build the input embeddings, run `model(x, start_at_layer=0, stop_at_layer=model.cfg.n_layers)` to get the final residual stream, pick out the residual
at the answer position of each example, and apply `model.ln_final` and `model.unembed`.

Two details:

- Questions in a batch have different lengths, so they are **right-padded** with the end-of-text token. The answer position of example `b` is therefore `1 + n_image_tokens + question_lens[b] - 1` (BOS, then the image tokens, then the question). You don't need an attention mask: attention is causal, and the padding comes *after* the answer position, so it can't affect it.
- Only decode the answer position. Decoding all positions would give a `(batch, seq, 50257)` tensor: wasteful, since we only ever need one row.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def vlm_answer_logits(
    model: HookedTransformer,
    projector: nn.Module,
    features: Float[Tensor, "batch patch d_vision"] | None,
    question_tokens: Int[Tensor, "batch question"],
    question_lens: Int[Tensor, " batch"],
) -> Float[Tensor, "batch d_vocab"]:
    """
    Logits for the token that follows the last real question token, given [BOS, image tokens, question].
    Pass features=None for a text-only model.
    """
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    image_embeds = projector(features) if features is not None else None
    n_prefix = 1 + (image_embeds.shape[1] if image_embeds is not None else 0)  # BOS (+ image tokens)
    x = build_input_embeddings(model, image_embeds, question_tokens)
    resid = model(x, start_at_layer=0, stop_at_layer=model.cfg.n_layers)
    answer_resid = resid[t.arange(len(x)), n_prefix + question_lens - 1]
    return model.unembed(model.ln_final(answer_resid))
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_vlm_answer_logits(vlm_answer_logits, gpt2)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`projector(features)` has shape `(batch, 49, 768)`; the number of positions before the question is `1 + 49`. The final residual stream is `resid = model(x, start_at_layer=0,
stop_at_layer=model.cfg.n_layers)` with shape `(batch, seq, d_model)`. Index it with `resid[t.arange(batch), n_prefix + question_lens - 1]` to get `(batch, d_model)`, then
`model.unembed(model.ln_final(...))`.
</details>

<details>
<summary>Help - my logits don't match the test's plain-text oracle for the shorter question</summary>

You're probably reading the answer from the last *position* (`-1`) instead of the last *real* token. For a right-padded batch the last position of the shorter question is padding.
Use `question_lens` to compute the index per example.
</details>

The test does something satisfying: it replaces the projector with a lookup that returns the embeddings of the words " a cat sat" and checks that the VLM forward pass then gives
*exactly* the logits GPT-2 produces on the plain-text sequence `<BOS> a cat sat <question>`, even in a padded batch. So the multimodal path really is the text path with different
vectors in some slots.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Measuring the answer: forced choice

How should we score the model? The strictest option is **free generation**: the answer counts only if the model's single most likely next token, among all 50,257 tokens, is the
right one. That's what you'd want from a deployed VLM, but it makes an untrained model look like it scores 0%, and it mixes two failures (not producing a valid answer word at all,
and producing the wrong one).

Instead, we use **forced choice**: restrict the logits to the answers that are *valid for this question kind* (six colours for a colour question, `left` and `right` for a
horizontal one), and count the model as right if the correct answer has the highest logit among them. Now **chance level is well defined** (1/6 for colour, 1/2 for `horizontal`),
the controls in the next section have a meaningful floor, and a trained model that has learned to produce valid answers is scored the same as under free generation (we'll check
this).
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - forced-choice accuracy

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 5-10 minutes on this exercise.
> ```

Implement `answer_accuracy`. It takes `logits` of shape `(batch, d_vocab)`, `answer_ids` of shape `(batch,)` (the token id of the right answer for each example) and `candidate_ids`
of shape `(n_candidates,)` (the token ids of the valid answers), and returns the fraction of examples where the candidate with the largest logit is the right answer, as a Python
`float`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def answer_accuracy(
    logits: Float[Tensor, "batch d_vocab"], answer_ids: Int[Tensor, " batch"], candidate_ids: Int[Tensor, " candidate"]
) -> float:
    """Fraction of examples for which the highest-logit *candidate* token is the correct answer."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    choice = candidate_ids[logits[:, candidate_ids].argmax(dim=-1)]
    return (choice == answer_ids).float().mean().item()
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_answer_accuracy(answer_accuracy)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`logits[:, candidate_ids]` has shape `(batch, n_candidates)`. Its `argmax(dim=-1)` is an index *into the candidate list*, so convert it back to a token id with `candidate_ids[...]`
before comparing with `answer_ids`.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - evaluate on a set of questions

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement `evaluate_vlm`, which scores a VLM on a prepared question set. `questions` is a dictionary from question kind to `(prompts, answers)`, where example `i` of every kind
goes with scene `i`, so `features[i]` is its image. For each kind, tokenise the prompts with `utils.tokenize_prompts(model, prompts)` (returns right-padded tokens and the lengths),
compute the answer logits in chunks of `batch_size` with your `vlm_answer_logits`, and score them with `answer_accuracy` against the valid answers of that kind
(`ANSWER_WORDS[kind]`, each with a space prepended; use `model.to_single_token(" red")` to get a token id). Return a dictionary from kind to accuracy.

With `use_image=False`, pass `None` instead of features, so the same function can evaluate a text-only model. The **features** argument is what we'll swap out when we build
controls: blank, noise or shuffled images are just different `features`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.no_grad()
def evaluate_vlm(
    model: HookedTransformer,
    projector: nn.Module,
    features: Float[Tensor, "scene patch d_vision"],
    questions: dict[str, tuple[list[str], list[str]]],
    use_image: bool = True,
    batch_size: int = 250,
) -> dict[str, float]:
    """Forced-choice accuracy per question kind. Example i of every kind is about the image features[i]."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    results = {}
    for kind, (prompts, answers) in questions.items():
        candidate_ids = t.tensor([model.to_single_token(" " + w) for w in ANSWER_WORDS[kind]], device=model.cfg.device)
        answer_ids = t.tensor([model.to_single_token(a) for a in answers], device=model.cfg.device)
        tokens, lens = utils.tokenize_prompts(model, prompts)
        correct = 0.0
        for start in range(0, len(prompts), batch_size):
            chunk = slice(start, start + batch_size)
            logits = vlm_answer_logits(model, projector, features[chunk] if use_image else None, tokens[chunk], lens[chunk])
            correct += answer_accuracy(logits, answer_ids[chunk], candidate_ids) * len(logits)
        results[kind] = correct / len(prompts)
    return results
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_evaluate_vlm(evaluate_vlm, Projector, gpt2, vlm_answer_logits)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

For each kind, build `candidate_ids` and `answer_ids` as tensors on `model.cfg.device`. Loop over `range(0, len(prompts), batch_size)`, slice `tokens`, `lens` and `features` with
the same slice, and accumulate `answer_accuracy(...) * len(logits)` so that a smaller final chunk is weighted correctly. Divide by the total at the end.
</details>

Let's use it on the model we have right now: pretrained GPT-2, with a **randomly initialised projector**. Nothing has been trained on our task. Compare its scores with uniform
chance and the best constant guess; pretrained answer preferences can produce different scores without useful visual grounding. An unexpectedly strong score would motivate a
leakage check, not prove leakage by itself.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

t.manual_seed(0)
untrained_projector = Projector().to(device)
untrained = evaluate_vlm(gpt2, untrained_projector, test_features, test_questions)
chance = {kind: 1 / len(ANSWER_WORDS[kind]) for kind in KINDS}
display(pd.DataFrame({"untrained VLM": untrained, "uniform chance": chance, "best image-blind guess": IMAGE_BLIND}).round(3).T)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

The exact numbers depend on the random initialisation of the projector. Our run is roughly image-blind, with nonuniform answer preferences rather than exact uniform-chance
performance:

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                        colour  shape  count  horizontal  vertical
untrained VLM            0.192  0.324  0.236       0.512     0.491
uniform chance           0.167  0.333  0.333       0.500     0.500
best image-blind guess   0.161  0.326  0.410       0.488     0.491</pre>

</details>

<details>
<summary>Question - the untrained VLM is not exactly at the best-constant-guess accuracy. Why not, and is this a problem?</summary>

A random projector and pretrained GPT-2 have nonuniform preferences; neither samples answers uniformly nor necessarily chooses the training-majority answer. The untrained model
gets 0.192 colour, 0.324 shape and 0.236 count accuracy, with position accuracy near 1/2. It has no useful visual mapping yet. Deviations from an idealized chance number are
expected on finite data; the important comparison is the trained model against image-blind and mismatched-image controls on the same questions.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 2️⃣

- A LLaVA-style VLM is a vision encoder, a projector and a language model, and the only place the image enters the language model is as **a block of embedding-like vectors in the context**. Our projector merges $2 \times 2$ patches (196 → 49 tokens) and applies a two-layer MLP.
- Building the multimodal input is plumbing, but it's plumbing you can test with an oracle: if the "image" embeddings are word embeddings, the VLM must reproduce the text-only model exactly.
- We read the answer at the last question token, and score with **forced choice** among valid answers, which gives every question kind a well-defined chance level.
- The untrained VLM shows answer preferences and roughly image-blind performance, not exact uniform-chance scores. Now we train it.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 3️⃣ Training and Modality Controls
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## What do we train?

LLaVA trains in two stages. In **stage 1** (feature alignment) both the vision encoder and the language model are frozen and only the projector is trained, so that image tokens
land somewhere sensible in the language model's embedding space. In **stage 2** the language model is unfrozen too, and projector and language model are trained together on
instruction-following data.

We'll compare the two regimes directly, from the same starting point (pretrained GPT-2, freshly initialised projector):

- **(A) projector only.** GPT-2 is frozen at its pretrained weights. The projector has about 3 million parameters. This is like LLaVA's stage 1, except that we train on questions and answers instead of captions.
- **(B) projector and GPT-2 together.** About 166 million trainable parameters. That counts everything the training loop can change: the projector's 3.0M, GPT-2 Small's 124M, and a second copy of
  the 38.6M-entry unembedding matrix, because TransformerLens keeps the unembedding as its own parameter instead of tying it to the token embedding.

The loss is the cross-entropy of the **answer token only** (the score that is low when the model puts its probability on the right answer; 0.7 Foundations Toolkit, Part 1 builds it from
scratch): we feed `[BOS, image tokens, question]`, read the logits at the last question token with `vlm_answer_logits`, and
compare them with the right answer's token id. (We don't train the model to predict the question text, only the answer.) For the optimiser we use AdamW with a **one-cycle
schedule** (linear warm-up over the first 10% of steps, then cosine decay) and **different learning rates for the two parameter groups**: $10^{-3}$ for the freshly initialised
projector, $2 \times 10^{-4}$ for GPT-2 (small by projector standards, but large for a language model, because this is a short run on a tiny task). Matrix multiplies run in
bfloat16 for speed.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - parameter groups

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement `make_optimizer(model, projector, train_lm, lr_projector, lr_lm)`. It should:

- set `requires_grad` on the language model's parameters to `train_lm` (so that with `train_lm=False` it is **frozen**), and make sure the projector's parameters require grad;
- return an `AdamW` optimiser (with `weight_decay=0.0`) whose parameter groups are the projector's parameters with learning rate `lr_projector`, plus (only if `train_lm`) the language model's parameters with learning rate `lr_lm`.

(Passing frozen parameters to an optimiser is harmless but wasteful, and a frozen model must not be updated, so the cleanest design is for the optimiser to only know about the
parameters being trained.)
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def make_optimizer(
    model: HookedTransformer,
    projector: nn.Module,
    train_lm: bool,
    lr_projector: float = 1e-3,
    lr_lm: float = 2e-4,
) -> t.optim.AdamW:
    """AdamW over the projector (lr_projector) and, if train_lm, the language model (lr_lm). Freezes the LM otherwise."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    model.requires_grad_(train_lm)
    projector.requires_grad_(True)
    groups = [{"params": list(projector.parameters()), "lr": lr_projector}]
    if train_lm:
        groups.append({"params": list(model.parameters()), "lr": lr_lm})
    return t.optim.AdamW(groups, weight_decay=0.0)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_make_optimizer(make_optimizer, Projector, gpt2)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`module.requires_grad_(flag)` sets the flag on every parameter of a module at once. A parameter group is a dictionary like `{"params": [...], "lr": 1e-3}`; `AdamW` accepts a list
of them.
</details>

## The batch sampler

Each training step needs a batch of (image features, question tokens, answer ids). Because every scene can be asked five kinds of question about up to three objects, we generate
questions on the fly: draw random scenes, then for each a random kind and a random target object, and call your `question_and_answer`. We've written this for you; note that it
calls your function from Part 1.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def sample_batch(
    model: HookedTransformer,
    features: Float[Tensor, "scene patch d_vision"],
    scenes: list[Scene],
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """A random batch: (patch features, right-padded question tokens, question lengths, answer token ids)."""
    idx = rng.integers(len(scenes), size=batch_size)
    prompts, answers = [], []
    for i in idx:
        kind = KINDS[rng.integers(len(KINDS))]
        prompt, answer = question_and_answer(scenes[i], kind, int(rng.integers(len(scenes[i].objects))))
        prompts.append(prompt)
        answers.append(answer)
    tokens, lens = utils.tokenize_prompts(model, prompts)
    answer_ids = t.tensor([model.to_single_token(a) for a in answers], device=model.cfg.device)
    return features[t.as_tensor(idx, device=features.device)], tokens, lens, answer_ids

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - the training loop

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Implement the body of the training loop in `train_vlm`. The optimiser, the learning-rate schedule and the batch sampling are set up for you. For each step you should:

1. compute the answer logits with `vlm_answer_logits` (passing `feats` if `use_image` else `None`), **inside** a `t.autocast(device.type, dtype=t.bfloat16, enabled=device.type == "cuda")` context, which runs matrix multiplies in bfloat16;
2. compute the cross-entropy between the logits (cast to `float32` first) and `answer_ids`;
3. zero the gradients, backpropagate, step the optimiser **and the scheduler**;
4. append `loss.item()` to `losses`.

The test trains on 16 scenes for a few steps and checks that the loss falls and, importantly, that a frozen language model stays *exactly* as it was.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def train_vlm(
    model: HookedTransformer,
    projector: nn.Module,
    features: Float[Tensor, "scene patch d_vision"],
    scenes: list[Scene],
    n_steps: int,
    train_lm: bool,
    lr_projector: float = 1e-3,
    lr_lm: float = 2e-4,
    batch_size: int = 64,
    use_image: bool = True,
    seed: int = 0,
) -> list[float]:
    """Trains the projector (and the language model if train_lm) on random questions. Returns the loss at every step."""
    optimizer = make_optimizer(model, projector, train_lm, lr_projector, lr_lm)
    scheduler = t.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in optimizer.param_groups], total_steps=n_steps, pct_start=0.1
    )
    rng = np.random.default_rng(seed)
    losses = []
    for _ in tqdm(range(n_steps), mininterval=10):
        feats, tokens, lens, answer_ids = sample_batch(model, features, scenes, batch_size, rng)
        # EXERCISE
        # raise NotImplementedError()
        # END EXERCISE
        # SOLUTION
        with t.autocast(device.type, dtype=t.bfloat16, enabled=device.type == "cuda"):
            logits = vlm_answer_logits(model, projector, feats if use_image else None, tokens, lens)
        loss = F.cross_entropy(logits.float(), answer_ids)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        # END SOLUTION
    return losses


# HIDE
if MAIN:
    tests.test_train_vlm(train_vlm, Projector, gpt2)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

The standard five lines: `loss = F.cross_entropy(logits.float(), answer_ids)`, then `optimizer.zero_grad()`, `loss.backward()`, `optimizer.step()`, `scheduler.step()`. Only the
forward pass (not the loss) needs to be inside the `autocast` context.
</details>

<details>
<summary>Help - the test says the frozen language model changed</summary>

That means `make_optimizer` didn't freeze it, or the optimiser received the LM's parameters anyway. Re-check `make_optimizer` with `train_lm=False`: after it runs,
`any(p.requires_grad for p in model.parameters())` should be `False`.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Train

Time to train both regimes, each from pretrained GPT-2 and a fresh projector. (A) takes 600 steps with the language model frozen; (B) takes 1,500 steps with everything trainable.
Batch size 64, so (B) sees 96,000 questions: about 19 passes over the 5,000 training scenes, each time with different questions.

If you're on a CPU, reduce the number of steps (try 150 and 400) and the number of training scenes; the qualitative picture should survive, but you'll need more steps for the
left/right and top/bottom questions than you'd like.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

import time

# (A) projector only: GPT-2 frozen
t.manual_seed(0)
projector_a = Projector().to(device)
lm_a = copy.deepcopy(gpt2)
start = time.time()
losses_a = train_vlm(lm_a, projector_a, train_features, train_scenes, n_steps=600, train_lm=False)
n_trainable = sum(p.numel() for p in projector_a.parameters())
print(f"(A) projector only: {n_trainable / 1e6:.1f}M trainable parameters (projector), {time.time() - start:.0f}s, final loss {np.mean(losses_a[-50:]):.3f}")
results_a = evaluate_vlm(lm_a, projector_a, test_features, test_questions)
del lm_a
t.cuda.empty_cache()

# (B) projector + GPT-2
t.manual_seed(0)
projector = Projector().to(device)
vlm = copy.deepcopy(gpt2)
start = time.time()
losses_b = train_vlm(vlm, projector, train_features, train_scenes, n_steps=1500, train_lm=True)
n_trainable = sum(p.numel() for p in projector.parameters()) + sum(p.numel() for p in vlm.parameters())
print(f"(B) projector + GPT-2: {n_trainable / 1e6:.0f}M trainable parameters (projector + GPT-2 incl. its separate unembedding), {time.time() - start:.0f}s, final loss {np.mean(losses_b[-50:]):.4f}")
results_b = evaluate_vlm(vlm, projector, test_features, test_questions)
vlm.requires_grad_(False)
projector.requires_grad_(False)

display(
    pd.DataFrame({"(A) projector only": results_a, "(B) projector + GPT-2": results_b, "best image-blind guess": IMAGE_BLIND}).round(3).T
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

The times depend on your GPU (these are from an NVIDIA L40S).

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">(A) projector only: 3.0M trainable parameters (projector), 56s, final loss 0.373
(B) projector + GPT-2: 166M trainable parameters (projector + GPT-2 incl. its separate unembedding), 186s, final loss 0.0315
                        colour  shape  count  horizontal  vertical
(A) projector only       0.594  0.943  0.999       0.748     0.762
(B) projector + GPT-2    1.000  1.000  0.999       0.945     0.997
best image-blind guess   0.161  0.326  0.410       0.488     0.491</pre>

</details>

<details>
<summary>Help - how should I interpret this table?</summary>

Training only the 3.0M-parameter projector already solves count (99.9%) and most of shape (94.3%), but colour (59.4%) and position (74.8% horizontal, 76.2% vertical) remain imperfect.
Training the projector and GPT-2 reaches 100.0% colour, 100.0% shape, 99.9% count and 99.7% vertical; horizontal remains weaker at 94.5%. Both beat the image-blind guesses, but good
accuracy alone cannot show the model is using the right image. The following modality controls test that. The two regimes use different numbers of steps as well as different
trainable parameters, so this is a practical recipe comparison, not an isolated causal estimate of unfreezing GPT-2 (the quiz returns to this).
</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

utils.plot_training_curves(
    {"(A) projector only, GPT-2 frozen": losses_a, "(B) projector + GPT-2": losses_b}, title="Training loss of the two regimes"
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/training_curves.png" width="600">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret the loss curves?</summary>

Both losses fall sharply at first. The projector-only run then plateaus at a higher loss, while the joint run continues downward over its longer training schedule (final printed
losses 0.373 and 0.0315). These are moving averages of training cross-entropy, not held-out VQA accuracy curves, and the runs have different step counts. Use the held-out accuracy
table to assess task performance; compare equal budgets if you want to attribute the gap specifically to unfreezing the language model.
</details>

## Is the model *looking*? Modality controls

A high accuracy on held-out questions is not yet evidence that the model uses the image. It's evidence that the *combination* of image and question determines the answer well. So
we ask the question the other way round: what happens to the accuracy when we **take the information away**? We'll build six controls, each of which should drive accuracy down to
the image-blind floor:

1. **Blank image.** SigLIP features of a scene with no objects, the same for every question.
2. **Noise image.** SigLIP features of images of independent uniform pixel noise.
3. **Shuffled image-question pairs.** The question about scene $i$ is asked with the image of a *different* scene $\pi(i)$. The images are real and in-distribution, only the pairing is wrong. This is the most important control: it rules out the model answering from "something about the image" (e.g. image statistics) rather than from the specific objects the question asks about.
4. **Random projector.** The trained language model, but with a freshly initialised projector. This tests whether the projector learned something specific, or whether any image tokens will do.
5. **Text-only model.** GPT-2 fine-tuned on the same questions and answers with **no image tokens at all**, with the same optimiser and a shorter schedule (it has less to learn). This measures how far one can get from question text alone, as a trained model rather than as an analytic bound.
6. **Best image-blind guess**, which we computed in Part 1.

**Make a prediction first:** what accuracy will the shuffled image-question pairs get on `count` questions, compared with the best image-blind guess of 0.41? (Hint: if the shuffled scene
happens to contain the same number of objects, the answer is still right.)
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - a shuffle with no fixed points

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

For the shuffled control, the pairing must be wrong for *every* question. A uniformly random permutation leaves about one item in place on average, and even a few fixed points
would contaminate the control with correct pairs, so we need a **derangement**: a permutation $\pi$ with $\pi(i) \neq i$ for all $i$.

Implement `derangement(n, seed)`, returning a `LongTensor` of length `n`. Rejection sampling (draw random permutations until one has no fixed points) works but is slow for big `n`.
A neat alternative: take a random permutation `order` of `0..n-1` and let each element point to the *next* element of `order` (cyclically). The result is a single cycle, which has
no fixed points when $n \geq 2$. Use a seeded `torch.Generator` so the control is reproducible.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def derangement(n: int, seed: int = 0) -> Int[Tensor, " n"]:
    """A random permutation of 0..n-1 with no fixed points (perm[i] != i for all i)."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    order = t.randperm(n, generator=t.Generator().manual_seed(seed))
    perm = t.empty(n, dtype=t.long)
    perm[order] = order.roll(-1)  # each element of `order` points to the next one, cyclically
    return perm
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_derangement(derangement)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`order = t.randperm(n, generator=t.Generator().manual_seed(seed))` is a random ordering. If you set `perm[order[k]] = order[(k + 1) % n]` for every `k`, then `perm` is a single
cycle through all elements. In one line: `perm[order] = order.roll(-1)`.
</details>

Now we build all the controls and evaluate. For the text-only baseline we train a fresh copy of GPT-2 for 400 steps with `use_image=False` (no image tokens in the sequence at all;
the projector is passed for the API but never used).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

n_test = len(test_scenes)
blank_features = extract_patch_features(siglip, utils.render_scenes([Scene([], background=235)])).expand(n_test, -1, -1)
noise_features = extract_patch_features(siglip, utils.noise_images(n_test, seed=0))
shuffled_features = test_features[derangement(n_test, seed=0).to(device)]

t.manual_seed(0)
random_projector = Projector().to(device)
text_only_lm = copy.deepcopy(gpt2)
losses_text = train_vlm(text_only_lm, Projector().to(device), train_features, train_scenes, n_steps=400, train_lm=True, use_image=False)
text_only_lm.requires_grad_(False)

all_results = {
    "trained VLM (B)": results_b,
    "projector only (A)": results_a,
    "blank image": evaluate_vlm(vlm, projector, blank_features, test_questions),
    "noise image": evaluate_vlm(vlm, projector, noise_features, test_questions),
    "shuffled image-question pairs": evaluate_vlm(vlm, projector, shuffled_features, test_questions),
    "random projector": evaluate_vlm(vlm, random_projector, test_features, test_questions),
    "text-only model": evaluate_vlm(text_only_lm, projector, test_features, test_questions, use_image=False),
}
display(pd.DataFrame(all_results).round(3).T)
utils.plot_accuracy_by_condition(
    all_results, IMAGE_BLIND, title="The trained VLM against every modality control (1,000 held-out questions per kind)"
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/accuracy_controls.png" width="900">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                               colour  shape  count  horizontal  vertical
trained VLM (B)                 1.000  1.000  0.999       0.945     0.997
projector only (A)              0.594  0.943  0.999       0.748     0.762
blank image                     0.186  0.332  0.236       0.512     0.509
noise image                     0.168  0.353  0.236       0.509     0.509
shuffled image-question pairs   0.198  0.329  0.358       0.519     0.494
random projector                0.207  0.326  0.410       0.534     0.462
text-only model                 0.159  0.341  0.410       0.512     0.491</pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

The trained model (B) is far above every control on every kind except count, where projector-only (A) matches it. The controls sit close to the best image-blind guess, with a few instructive exceptions.
On **count**, blank and noise images score 0.236, *below* the image-blind guess of 0.410: they make the model say `one` every time, which is right only for the 23.6% of scenes with one object.
The **shuffled** pairs score 0.358 on count, between those two, as you should expect: a shuffled scene has the same number of objects as the right one about a third of the time. The random projector
scores 0.410 on count, exactly the image-blind guess. Colour, shape and position controls land near chance (1/6, 1/3 and 1/2). The weakest result of the trained model is `horizontal` (0.945),
against 0.997 for `vertical`.
</details>

## Keep the picture, take the question away

Every control so far broke the image. Here is the opposite one: keep the **true image** and take the **question** away. For every question we give the model the same fixed text,
just ` Answer:`. It knows an answer is due, but not whether you want a colour, a shape, a count or a position, and not which object you mean. We still score it afterwards with the
candidate words of the right kind (six colours for a colour question, and so on), but the model never sees those words.

What should happen? The image holds a lot, and a model that read it perfectly could still not know *which* object the question is about. With two objects of different colours, any
fixed colour answer is right half the time; with three, one time in three. The best a perfect reader of the scene could do without the target is the **target-blind oracle**: for each
scene and question kind, work out the answer for every object that could have been the target, and take the share of the most common answer. Count is special: every target gives the same
count, so the oracle is 1 and a model that read the scene perfectly would lose nothing on it. We will see whether our model does.

Two things to keep in mind when you read the result. This is the trained VLM given a prompt it never saw, not a separately trained image-only model, so some of any drop comes from the odd
prompt. And one scene gives one answer, not an average, so a model can beat the oracle on a few hundred scenes by luck.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `target_blind_oracle`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement `target_blind_oracle(scene, kind)`. Use `question_and_answer` to get the answer to the question of this `kind` for **every** object in the scene as the target, count how often
each answer occurs, and return the largest count divided by the number of objects. Answers that repeat (two objects on the left, for instance) must count twice, so don't
assume that every answer is different, and don't hard-code `1 / len(scene.objects)`. An empty scene has no targets: raise a `ValueError`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def target_blind_oracle(scene: Scene, kind: str) -> float:
    """Best target-blind answer probability with perfect knowledge of this scene."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    if not scene.objects:
        raise ValueError("The target-sampling contract requires a nonempty scene")
    counts = Counter(question_and_answer(scene, kind, target)[1] for target in range(len(scene.objects)))
    return max(counts.values()) / len(scene.objects)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_target_blind_oracle(target_blind_oracle)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Build the list of answers with a loop over `range(len(scene.objects))`, calling `question_and_answer(scene, kind, target)[1]` for each. `Counter(answers).most_common(1)` gives the
most frequent answer and how often it occurs.
</details>

<details>
<summary>Help - what does this oracle tell us?</summary>

For two objects of different colours the oracle is 1/2 for colour. If both objects are on the left it is 1 for horizontal position, and if they are on different sides it is 1/2. It is 1
for count because every target produces the same count. Our generator gives every object in a scene a different shape and colour, but two objects can share a side, which is why the
oracle has to count answers instead of assuming they are all different.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - implement `evaluate_readout_controls`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵⚪⚪⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement `evaluate_readout_controls`. It scores the trained model twice on the same images, the same targets, the same answers and the same candidate words: first with each
question's full prompt, then with the fixed ` Answer:` cue. It returns the accuracy of each condition for each question kind, and a list with one row per (condition, kind, scene) that
records the prediction, the answer, the candidate logits, the number of objects in the scene and the oracle, so that we can group the results by object count afterwards.

The questions come with the answers but not with the object each one was about, because `make_question_set` drew the targets inside. The helper `replay_question_targets` draws
them again with the same random numbers; your function should check that `question_and_answer(scene, kind, target)` really reproduces each original prompt and answer.
The model's input for the image-only condition is the cue and the image, nothing else. Count the correct answers over all batches before dividing, including a shorter last batch.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def replay_question_targets(scenes: list[Scene], seed: int) -> dict[str, list[int]]:
    """Draws make_question_set's random targets again (same seed, same order): which object each question was about."""
    rng = np.random.default_rng(seed)
    return {kind: [int(rng.integers(len(scene.objects))) for scene in scenes] for kind in KINDS}


@t.no_grad()
def evaluate_readout_controls(model, projector, features, scenes, questions, targets, batch_size=250):
    """Accuracy of the full question and of the fixed ` Answer:` cue, plus one row per (condition, kind, scene)."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    if batch_size < 1 or len(scenes) == 0 or len(features) != len(scenes):
        raise ValueError("Need nonempty aligned scenes/features and a positive batch size")
    # One image-only forward per image batch. Task/candidate information enters
    # only after these shared logits exist; it cannot change the model input.
    fixed_tokens, fixed_lens = utils.tokenize_prompts(model, [" Answer:"] * len(scenes))
    fixed_scores = {kind: [] for kind in questions}
    for start in range(0, len(scenes), batch_size):
        stop = min(start + batch_size, len(scenes))
        logits = vlm_answer_logits(model, projector, features[start:stop], fixed_tokens[start:stop], fixed_lens[start:stop])
        for kind in questions:
            ids = t.tensor([model.to_single_token(" " + word) for word in ANSWER_WORDS[kind]], device=model.cfg.device)
            fixed_scores[kind].append(logits[:, ids].float().cpu())
    fixed_scores = {kind: t.cat(rows) for kind, rows in fixed_scores.items()}
    results, rows_out = {}, []
    for condition in ["full question", "image-only (fixed readout)"]:
        results[condition] = {}
        for kind, (prompts, answers) in questions.items():
            if len(prompts) != len(scenes) or len(answers) != len(scenes) or len(targets[kind]) != len(scenes):
                raise ValueError("Original questions, labels and targets must remain aligned")
            for i, scene in enumerate(scenes):
                if question_and_answer(scene, kind, targets[kind][i]) != (prompts[i], answers[i]):
                    raise ValueError("Target replay does not reconstruct the original question and label")
            model_prompts = prompts if condition == "full question" else [" Answer:"] * len(scenes)
            tokens, lens = utils.tokenize_prompts(model, model_prompts) if condition == "full question" else (fixed_tokens, fixed_lens)
            words = [" " + word for word in ANSWER_WORDS[kind]]
            candidate_ids = t.tensor([model.to_single_token(word) for word in words], device=model.cfg.device)
            correct = 0
            for start in range(0, len(scenes), batch_size):
                stop = min(start + batch_size, len(scenes))
                if condition == "full question":
                    logits = vlm_answer_logits(model, projector, features[start:stop], tokens[start:stop], lens[start:stop])
                    scores = logits[:, candidate_ids].float().cpu()
                else:
                    scores = fixed_scores[kind][start:stop]
                for offset, row in enumerate(scores):
                    i = start + offset
                    prediction = words[int(row.argmax())]
                    hit = prediction == answers[i]
                    correct += int(hit)
                    rows_out.append(dict(condition=condition, scene_index=i, kind=kind,
                        object_count=len(scenes[i].objects), target_index=targets[kind][i],
                        original_question=prompts[i], model_prompt=model_prompts[i],
                        answer=answers[i], prediction=prediction, correct=hit,
                        candidate_words=words, candidate_ids=candidate_ids.cpu().tolist(),
                        candidate_logits=row.tolist(), target_blind_oracle=target_blind_oracle(scenes[i], kind)))
            results[condition][kind] = correct / len(scenes)
    return results, rows_out
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_evaluate_readout_controls(evaluate_readout_controls)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Run the model once on all the images with the fixed cue, and keep the candidate logits of every kind; you can score all five kinds from those same logits. For the full-question
condition, tokenise each kind's prompts and run in batches of `batch_size`. For each row of candidate logits, the prediction is the candidate word with the highest logit.
</details>

<details>
<summary>Help - is a failed image-only control enough?</summary>

No. Removing the question also changes the prompt, so the drop mixes "the model cannot tell which object we mean" with "the model has never seen this prompt". Read it together with the
shuffled-image and text-only controls above. The scorer restricts the output to the right candidate words; this is a forced choice, not free image captioning.
</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

readout_targets = replay_question_targets(test_scenes, seed=0)
readout_accuracy, readout_rows = evaluate_readout_controls(
    vlm, projector, test_features, test_scenes, test_questions, readout_targets)
assert all(abs(readout_accuracy["full question"][kind] - results_b[kind]) < 1e-6 for kind in KINDS)  # same numbers as regime (B)
display(pd.DataFrame({**all_results, "image-only (fixed readout)": readout_accuracy["image-only (fixed readout)"]}).round(3).T)
readout_table = pd.DataFrame(readout_rows)
display(readout_table.groupby(["condition", "kind", "object_count"])[["correct", "target_blind_oracle"]].mean().round(3))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                               colour  shape  count  horizontal  vertical
trained VLM (B)                 1.000  1.000  0.999       0.945     0.997
projector only (A)              0.594  0.943  0.999       0.748     0.762
blank image                     0.186  0.332  0.236       0.512     0.509
noise image                     0.168  0.353  0.236       0.509     0.509
shuffled image-question pairs   0.198  0.329  0.358       0.519     0.494
random projector                0.207  0.326  0.410       0.534     0.462
text-only model                 0.159  0.341  0.410       0.512     0.491
image-only (fixed readout)      0.469  0.547  0.444       0.721     0.722
                                                    correct  \
condition                  kind       object_count            
full question              colour     1               1.000   
                                      2               1.000   
                                      3               1.000   
                           count      1               1.000   
                                      2               1.000   
                                      3               0.997   
                           horizontal 1               1.000   
                                      2               0.976   
                                      3               0.873   
                           shape      1               1.000   
                                      2               1.000   
                                      3               1.000   
                           vertical   1               1.000   
                                      2               1.000   
                                      3               0.992   
image-only (fixed readout) colour     1               0.661   
                                      2               0.471   
                                      3               0.339   
                           count      1               1.000   
                                      2               0.080   
                                      3               0.494   
                           horizontal 1               0.983   
                                      2               0.646   
                                      3               0.633   
                           shape      1               0.953   
                                      2               0.490   
                                      3               0.342   
                           vertical   1               1.000   
                                      2               0.668   
                                      3               0.599   

                                                    target_blind_oracle  
condition                  kind       object_count                       
full question              colour     1                           1.000  
                                      2                           0.500  
                                      3                           0.333  
                           count      1                           1.000  
                                      2                           1.000  
                                      3                           1.000  
                           horizontal 1                           1.000  
                                      2                           0.654  
                                      3                           0.667  
                           shape      1                           1.000  
                                      2                           0.500  
                                      3                           0.333  
                           vertical   1                           1.000  
                                      2                           0.674  
                                      3                           0.667  
image-only (fixed readout) colour     1                           1.000  
                                      2                           0.500  
                                      3                           0.333  
                           count      1                           1.000  
                                      2                           1.000  
                                      3                           1.000  
                           horizontal 1                           1.000  
                                      2                           0.654  
                                      3                           0.667  
                           shape      1                           1.000  
                                      2                           0.500  
                                      3                           0.333  
                           vertical   1                           1.000  
                                      2                           0.674  
                                      3                           0.667  </pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

The image-only row is the trained model with the true image and the fixed ` Answer:` cue. Its accuracies are 0.469 for colour, 0.547 for shape, 0.444 for count, 0.721 for horizontal and 0.722 for vertical,
against 1.000, 1.000, 0.999, 0.945 and 0.997 with the question. So removing the question costs accuracy on every kind, and it costs the most on the kinds that need a target.

The grouped table shows why that is not simply a failure to see. For colour the image-only accuracy is 0.661 on single-object scenes, 0.471 on two-object scenes and 0.339 on three-object scenes, and the
perfect-reader oracle (no question, so a random target) is 1, 1/2 and 1/3. On two and three objects the model sits close to the oracle for colour, shape and position (for example horizontal 0.646 against
a mean oracle of 0.654 on two objects), so it behaves like a reader of the scene that has to guess which object you mean. On one object there is nothing to guess, and the model is still only right 0.661 of the time for
colour: without a question it doesn't reliably say the colour. These are expectations, not ceilings: one scene gives one answer, so a model can land above the oracle by luck (colour on three objects: 0.339 against 0.333).

**Count** is the surprise. A scene reader loses nothing there (the oracle is 1 for every object count), but the model's image-only count accuracy is 1.000 on one-object scenes, 0.080 on two-object scenes and 0.494 on
three-object scenes: 0.444 overall. With only ` Answer:` the model is not being asked for a count, and its default answers happen to be right for one and three objects and almost never for two. The fixed cue is outside
the training distribution, so this drop mixes "no instruction" with "no information". It doesn't show that the image lacks the count: the question-and-image run of regime (B) gets count right 99.9% of the time.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### What does the model say when there is nothing to see?

The controls tell us the accuracy collapses. They don't tell us *how*: does the model's answer become random, or does it fall back on a favourite? We can find out by tabulating the
model's answers on blank images. We tabulate the forced-choice answers for `colour` and `count` questions on the 1,000 test questions (the same blank image every time).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

@t.no_grad()
def predicted_answers(model, projector, features, questions, kind: str, use_image: bool = True) -> Counter:
    """Counts of the answer words the model picks (forced choice) for the questions of one kind."""
    prompts, _ = questions[kind]
    tokens, lens = utils.tokenize_prompts(model, prompts)
    candidates = [" " + w for w in ANSWER_WORDS[kind]]
    candidate_ids = t.tensor([model.to_single_token(c) for c in candidates], device=model.cfg.device)
    logits = vlm_answer_logits(model, projector, features if use_image else None, tokens, lens)
    return Counter(candidates[i] for i in logits[:, candidate_ids].argmax(-1).tolist())


for kind in ["colour", "count"]:
    print(f"{kind}:")
    print("  true image  :", dict(sorted(predicted_answers(vlm, projector, test_features, test_questions, kind).items())))
    print("  blank image :", dict(sorted(predicted_answers(vlm, projector, blank_features, test_questions, kind).items())))
    print("  noise image :", dict(sorted(predicted_answers(vlm, projector, noise_features, test_questions, kind).items())))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">colour:
  true image  : {' blue': 188, ' green': 159, ' orange': 139, ' purple': 161, ' red': 192, ' yellow': 161}
  blank image : {' purple': 321, ' red': 679}
  noise image : {' blue': 4, ' green': 52, ' purple': 119, ' red': 527, ' yellow': 298}
count:
  true image  : {' one': 236, ' three': 353, ' two': 411}
  blank image : {' one': 1000}
  noise image : {' one': 1000}</pre>

</details>

<details>
<summary>Question - what does this tell you about how the model fails?</summary>

Blank images make 679 of 1,000 colour answers `red` and the other 321 `purple`, and every count answer `one`; noise images also make every count answer `one`, and spread colour answers over mostly `red` and `yellow`. With real images, predictions span the answer classes and track
their frequencies. This is confident defaulting under missing visual evidence, not a robust `I cannot tell` response. The blank-image count accuracy is therefore the test frequency
of one-object scenes (0.236). Forced-choice normalization always distributes probability among allowed answers, so confidence does not itself show grounded knowledge.
</details>

## A look at the answers

Accuracy is a number; let's also look at some of the model's actual answers, with the probability it puts on its choice among the valid answers. The cell below picks eight test
questions at random (the same ones every time) and shows the model's answer under each scene.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

@t.no_grad()
def answer_probabilities(model, projector, features, prompts: list[str], kind: str) -> Float[Tensor, "batch candidate"]:
    """Softmax over the valid answers of `kind` (forced choice) for each prompt."""
    tokens, lens = utils.tokenize_prompts(model, prompts)
    candidate_ids = t.tensor([model.to_single_token(" " + w) for w in ANSWER_WORDS[kind]], device=model.cfg.device)
    return vlm_answer_logits(model, projector, features, tokens, lens)[:, candidate_ids].softmax(-1)


gallery_rng = np.random.default_rng(5)
gallery_idx = gallery_rng.choice(n_test, size=8, replace=False)
gallery_kinds = [KINDS[k] for k in gallery_rng.integers(0, 5, size=8)]
gallery_captions = []
for i, kind in zip(gallery_idx, gallery_kinds):
    prompt, answer = test_questions[kind][0][i], test_questions[kind][1][i]
    probs = answer_probabilities(vlm, projector, test_features[i : i + 1], [prompt], kind)[0]
    best = int(probs.argmax())
    guess = ANSWER_WORDS[kind][best]
    mark = "correct" if " " + guess == answer else f"WRONG (truth: {answer.strip()})"
    gallery_captions.append(f"{prompt[11:-8]}\nmodel: {guess} ({probs[best]:.0%}), {mark}")
utils.plot_scene_gallery(utils.render_scenes([test_scenes[i] for i in gallery_idx]), gallery_captions, ncols=4)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/vlm_gallery.png" width="900">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Try it yourself - ask your own question

Time to play with the model. The function `ask` takes a list of `SceneObject`s (colour, shape, quadrant, and optionally a jitter `dx`, `dy` in pixels and a `size`), renders the
scene, runs the VLM, and prints the model's probability for each valid answer to a question of the kind you choose. The quadrant is `(row, col)` with `(0, 0)` the top left.

Run it as it is, then try some experiments:

- **Move an object** by changing its quadrant, or by pushing it towards the middle line with a large `dx` (say 45). At what point does `horizontal` flip?
- **Resize** an object (`size=30` or `size=110`, the training range is 52-68). Does the model still know the colour?
- **Break the rules of the world.** Put two circles in the scene (the model never saw this). What does "What colour is the circle?" give?
- **An empty scene** (`objects=[]`): what does the model answer to the `count` question? (Think about what's in its answer vocabulary.)
- Can you find a scene where the model is **confidently wrong**?
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

def ask(objects: list[SceneObject], kinds: list[str] | None = None, target: int = 0) -> None:
    """Render a scene, then print the VLM's forced-choice answer distribution for one question of each kind."""
    scene = Scene(objects, background=235, noise_seed=0)
    image = utils.render_scene(scene)
    features = extract_patch_features(siglip, image[None])
    plt.figure(figsize=(3, 3))
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    for kind in kinds or KINDS:
        if kind == "count":
            prompt, truth = question_and_answer(scene, kind, 0) if objects else (" Question: How many shapes are there? Answer:", " (none)")
        else:
            prompt, truth = question_and_answer(scene, kind, target)
        probs = answer_probabilities(vlm, projector, features, [prompt], kind)[0]
        shown = ", ".join(f"{w} {p:.2f}" for w, p in zip(ANSWER_WORDS[kind], probs.tolist()))
        print(f"{prompt[11:-8]:<48} truth: {truth.strip():<8} model: {shown}")


ask([SceneObject("blue", "circle", (0, 1)), SceneObject("yellow", "triangle", (1, 0), dx=20), SceneObject("green", "square", (1, 1))])

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/ask_example.png" width="250">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">What colour is the circle?                       truth: blue     model: red 0.00, green 0.00, blue 1.00, yellow 0.00, purple 0.00, orange 0.00
What shape is the blue object?                   truth: circle   model: circle 1.00, square 0.00, triangle 0.00
How many shapes are there?                       truth: three    model: one 0.00, two 0.00, three 1.00
Is the circle on the left or the right?          truth: right    model: left 0.04, right 0.96
Is the circle at the top or the bottom?          truth: top      model: top 0.99, bottom 0.01</pre>

For this scene, the model assigns rounded probability 1.00 to blue, circle and three, 0.99 to top, and 0.96 to right. These are probabilities renormalized over the valid answers for each
question kind, not probabilities of arbitrary generated text. Change an object or its quadrant and rerun; the fixed template and synthetic images define the scope of this play
cell.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 3️⃣

- Training only the projector on a frozen GPT-2 (regime A) already teaches the model a lot, but not everything. After 1,500 joint-training steps, regime B reaches near-perfect accuracy on four kinds (colour 100.0%, shape 100.0%, count 99.9%, vertical 99.7%), while horizontal remains weaker at 94.5%.
- The modality controls perform much worse than the trained VLM: blank images, noise images, **shuffled
  image-question pairs**, a **random projector** and a **text-only model** give roughly image-blind
  performance, with some systematic defaults below the best constant guess (blank/noise count accuracy 0.236
  versus 0.410). The shuffled control preserves a real image and a real question while breaking their match,
  so it directly tests whether correct pairing matters.
- Keep the picture and drop the question, and accuracy falls on every kind (colour 0.469, count 0.444): the image-only model behaves like a reader of the scene that has to guess the target, on top of a prompt it never saw.
- When there is nothing to see, the model doesn't produce random answers: it **falls back on a favourite**. A VLM that is losing its grip on the image can look fluent and confident, so this kind of behaviour is exactly what controls are for.

So the model does use the image. That is a statement about behaviour. The next section asks what happens *inside*.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 4️⃣ Where Does the Answer Come From?
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Four kinds of evidence

We now have a VLM that answers colour questions about a held-out scene correctly every time (we'll verify that in a moment), and we know from the controls that it needs the image.
Now we ask what it does with it. For a question like *"What colour is the circle?"* the picture in our heads is:

1. The circle is covered by a handful of image tokens (say 4 of the 49).
2. Somewhere in the network, the information "this object is blue" is carried by those tokens.
3. At some layer, that information is moved to the position where the answer is written, the final `:`.

Each of these three steps can be probed, and a good interpretability result is one where **different kinds of evidence agree**. We'll use four, in increasing order of strength.
They answer different questions, and the difference between them is the main lesson of this section:

| evidence | method | what it tells you |
|---|---|---|
| **decodable** | logit lens on image tokens | what the token's residual stream *contains*, as read by the output embedding |
| **attended** | attention pattern of the answer position | what the answer position *looks at*, layer by layer |
| **sufficient** | activation patching of image tokens | whether those tokens' content can *change the answer* |
| **necessary** | attention knockout | whether the model *needs* a route to produce the answer |

This follows the structure of two recent studies on real VLMs. Neo et al. (2025,
[*Towards Interpreting Visual Information Processing in Vision-Language Models*](https://arxiv.org/abs/2410.07149))
apply the logit lens to LLaVA's image tokens, and ablate the tokens at an object's location, finding that the
answer depends on them. Zhang et al. (2025,
[*Cross-modal Information Flow in Multimodal Large Language Models*](https://arxiv.org/abs/2411.18620)) block
attention between groups of tokens in windows of layers and find that image information flows to the question
in lower and middle layers and to the last position in later ones. We'll reproduce the flavour of both, at toy
scale first and on a real model in section 5️⃣.

The answer is read at position 59 (the final `:`), and image tokens are at positions 1-49.

**Make a prediction first:** in which layers of GPT-2 will the answer position put most of its attention on the object's image tokens: the early layers, the middle ones or the last few?
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Setup: a batch of held-out colour questions

For most of this section we use colour questions, because they're the cleanest: one object, one attribute. We take the first 200 test scenes, pick a random target object in each,
and ask `"What colour is the {shape}?"`. Because every shape word is a single token, all 200 prompts have the same length: **10 question tokens**, so the sequence layout is the
same for every example, which will make life much easier:

| position | 0 | 1 - 49 | 50 - 59 |
|---|---|---|---|
| content | BOS | the 49 image tokens (raster order of the 7 × 7 grid) | ` Question`, `:`, ` What`, ` colour`, ` is`, ` the`, ` {shape}`, `?`, ` Answer`, `:` |

The **answer position** is 59. We run the trained VLM once on all 200 prompts and cache everything we need: the residual stream before each block (`resid_pre`, which has shape
`(batch, 60, 768)`), and the attention patterns.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

N_ANALYSIS = 200
analysis_rng = np.random.default_rng(3)
a_scenes = test_scenes[:N_ANALYSIS]
a_targets = [int(analysis_rng.integers(len(s.objects))) for s in a_scenes]
a_prompts, a_answers = zip(*[question_and_answer(s, "colour", i) for s, i in zip(a_scenes, a_targets)])
a_tokens, a_lens = utils.tokenize_prompts(vlm, list(a_prompts))
assert (a_lens == a_lens[0]).all(), "all colour prompts should have the same length"

SEQ_LEN = 1 + N_IMAGE_TOKENS + int(a_lens[0])
ANSWER_POS = SEQ_LEN - 1
IMAGE_SLICE = slice(1, 1 + N_IMAGE_TOKENS)
N_LAYERS = vlm.cfg.n_layers
colour_ids = t.tensor([vlm.to_single_token(" " + c) for c in COLOURS], device=device)
a_target_colour = t.tensor([list(COLOURS).index(s.objects[i].colour) for s, i in zip(a_scenes, a_targets)], device=device)

with t.no_grad():
    a_embeds = build_input_embeddings(vlm, projector(test_features[:N_ANALYSIS]), a_tokens)
    _, a_cache = vlm.run_with_cache(
        a_embeds,
        start_at_layer=0,
        stop_at_layer=N_LAYERS,
        names_filter=lambda name: name.endswith("resid_pre") or name.endswith("hook_pattern") or name == f"blocks.{N_LAYERS - 1}.hook_resid_post",
    )
    a_logits = vlm.unembed(vlm.ln_final(vlm(a_embeds, start_at_layer=0, stop_at_layer=N_LAYERS)[:, ANSWER_POS]))

print(f"sequence length {SEQ_LEN}, answer position {ANSWER_POS}")
print("question tokens:", [vlm.to_string(tok) for tok in a_tokens[0]])
print(f"forced-choice accuracy on these {N_ANALYSIS} questions: {answer_accuracy(a_logits, colour_ids[a_target_colour], colour_ids):.1%}")
print(f"free-generation accuracy (top-1 over the whole vocabulary): {(a_logits.argmax(-1) == colour_ids[a_target_colour]).float().mean():.1%}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">sequence length 60, answer position 59
question tokens: [' Question', ':', ' What', ' colour', ' is', ' the', ' triangle', '?', ' Answer', ':']
forced-choice accuracy on these 200 questions: 100.0%
free-generation accuracy (top-1 over the whole vocabulary): 100.0%</pre>

The last line answers a question we left open in section 2️⃣: for the trained VLM, forced choice and free generation agree. The model has learned to produce the valid answer words
by itself.
</details>

## Which tokens are the object's? And what are the controls?

We know where every object is, because we drew it. Image token $(i, j)$ of the $7 \times 7$ grid (position $1 + 7i + j$ in the sequence) covers a $32 \times 32$ pixel block, and we
call it one of the **object's tokens** if the object covers at least 20% of that block (`utils.object_token_mask`). A typical object has 4 or 5 such tokens. We also need three
kinds of comparison sets, and building these properly is half of the work:

- **Other objects' tokens**: tokens of the *other* shapes in the scene. They contain the right *kind* of content (a coloured shape), just not the queried one.
- **Background tokens**: tokens that no object touches.
- **A same-size random control**: *as many* background tokens as the object has, chosen at random. Comparing the object's 4 tokens against *all* 30+ background tokens would be unfair in any experiment that sums over the set (like attention mass), and the same-size set removes that confound.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - same-size random controls

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement `random_background_mask`. It takes `object_masks` of shape `(batch, 49)` (True where an image token belongs to the target object), `occupied_masks` of shape `(batch, 49)`
(True where a token belongs to *any* object), and a seed. For each row it returns a mask with **exactly as many True entries as the object mask has**, chosen uniformly at random
without replacement from the tokens where `occupied_masks` is False. Use a seeded `torch.Generator` so the control is reproducible, and make sure no token is ever picked that
overlaps an object.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def random_background_mask(
    object_masks: Bool[Tensor, "batch token"], occupied_masks: Bool[Tensor, "batch token"], seed: int = 0
) -> Bool[Tensor, "batch token"]:
    """For each row, a random set of background tokens (not in occupied_masks) as large as the object's token set."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    generator = t.Generator().manual_seed(seed)
    out = t.zeros_like(object_masks)
    for b in range(len(object_masks)):
        k = int(object_masks[b].sum())
        free = (~occupied_masks[b]).nonzero().flatten().cpu()
        chosen = free[t.randperm(len(free), generator=generator)[:k]]
        out[b, chosen.to(out.device)] = True
    return out
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_random_background_mask(random_background_mask)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Loop over rows. For row `b`: `k = int(object_masks[b].sum())`; the candidate tokens are `(~occupied_masks[b]).nonzero().flatten()`; shuffle them with `t.randperm(len(free),
generator=generator)` and keep the first `k`; set those entries of an all-False row to True. Create the generator on the CPU (`t.Generator().manual_seed(seed)`) and move indices to
the right device at the end.
</details>
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

def token_set_masks(scenes: list[Scene], targets: list[int], seed: int = 0) -> dict[str, Tensor]:
    """Boolean (n, 49) masks: the target object's tokens, the other objects' tokens, and the same-size random control."""
    object_masks = t.tensor(np.stack([utils.object_token_mask(s.objects[i]).flatten() for s, i in zip(scenes, targets)]), device=device)
    occupied = t.tensor(
        np.stack([np.any([utils.object_token_mask(o).flatten() for o in s.objects], axis=0) for s in scenes]), device=device
    )
    return {
        "object": object_masks,
        "other objects": occupied & ~object_masks,
        "occupied": occupied,
        "random background": random_background_mask(object_masks, occupied, seed),
    }


a_masks = token_set_masks(a_scenes, a_targets)
print(f"tokens per object: {a_masks['object'].sum(1).float().mean():.1f} on average (min {a_masks['object'].sum(1).min()}, max {a_masks['object'].sum(1).max()})")
print(f"background tokens: {(~a_masks['occupied']).sum(1).float().mean():.1f} on average")
show = list(range(5))
utils.plot_token_maps(
    [a_scenes[i] for i in show],
    [a_targets[i] for i in show],
    [a_masks["object"][show].float().cpu().numpy(), a_masks["random background"][show].float().cpu().numpy(), a_masks["other objects"][show].float().cpu().numpy()],
    ["the object's tokens", "same-size random\nbackground (control)", "other objects'\ntokens"],
    [f"question about the {a_scenes[i].objects[a_targets[i]].colour} {a_scenes[i].objects[a_targets[i]].shape}" for i in show],
    suptitle="The token sets we will compare (yellow = in the set; white outline = the queried object)",
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/token_sets.png" width="800">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">tokens per object: 4.4 on average (min 2, max 9)
background tokens: 39.6 on average</pre>

</details>

## The logit lens on image tokens

In [7.1](../../../chapter7_activation_to_language/exercises/part1_lenses_patchscopes/7.1_Logit_Lens_Tuned_Lens_and_Patchscopes_exercises.ipynb) we decoded the residual stream of *text* tokens by applying the final layer norm and the unembedding:

$$
\text{LogitLens}(x_\ell) = \text{LN}_f(x_\ell)\, W_U + b_U .
$$

Nothing about this formula requires the vector to come from a text token. We can apply it to the residual stream at an **image token** and ask: what words would the model "say" at
this position? If the object's tokens carry the object's colour, the lens might show a colour word at some layer. Neo et al. (2025) found exactly this in LLaVA: in later layers,
the image tokens covering an object decode to the object's name.

We can only decode *residual stream* vectors at positions 1-49, which is what we cached. We want the same 13 vectors as in 7.1 (the residual stream before each of the 12 blocks,
plus the final output), so we stack them. Let's start by looking at what two tokens decode to: one token that overlaps the queried object, and one background token, in the first
scene.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

LAYER_LABELS = ["embed"] + [f"pre {layer}" for layer in range(1, N_LAYERS)] + ["final"]
# (13, batch, 49, d_model): the residual stream at the image tokens, before every block and after the last one
image_resid = t.stack(
    [a_cache["resid_pre", layer][:, IMAGE_SLICE] for layer in range(N_LAYERS)] + [a_cache["resid_post", N_LAYERS - 1][:, IMAGE_SLICE]]
)


def top_tokens(resid: Float[Tensor, "d_model"], k: int = 3) -> str:
    probs = vlm.unembed(vlm.ln_final(resid)).softmax(-1)
    return ", ".join(f"{vlm.to_string(i)!r}" for i in probs.topk(k).indices)


b = 0
obj_token = int(a_masks["object"][b].nonzero()[0])
bg_token = int((~a_masks["occupied"][b]).nonzero()[0])
print(f"scene 0: the question is about the {a_scenes[b].objects[a_targets[b]].colour} {a_scenes[b].objects[a_targets[b]].shape}")
rows = {
    LAYER_LABELS[layer]: {"object token": top_tokens(image_resid[layer, b, obj_token]), "background token": top_tokens(image_resid[layer, b, bg_token])}
    for layer in [0, 1, 4, 8, 10, 11, 12]
}
with pd.option_context("display.max_colwidth", 60):
    display(pd.DataFrame(rows).T)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">scene 0: the question is about the blue triangle
                         object token                   background token
embed        'avorite', 'ida', 'iera'  ' dissatisf', ' inex', ' traject'
pre 1      'ially', 'alyst', ' Dhabi'           'saf', 'esters', 'itude'
pre 4       'azo', ' bubble', 'ially'                  ' the', ' a', ','
pre 8          'ulent', 'onic', 'emi'                  ' the', ' a', ','
pre 10       'ulent', 'onic', ' Finn'                  ' the', ' a', ','
pre 11  ' Finn', ' blue', ' triangle'                  ' the', ' a', ','
final       ' blue', ' triangle', '.'             ' the', ' a', ' three'</pre>

</details>

<details>
<summary>Question - what do you make of this table? Is it evidence for or against the colour being "in" the object's tokens?</summary>

The object is a blue triangle. At the embedding stage and in the early blocks its top words are fragments such as `avorite` and `ida`, and the background token's are others. Only in the last layers do the words become
related to the scene: before block 11 the object token decodes to ` Finn`, ` blue` and ` triangle`, and at the final layer to ` blue`, ` triangle` and `.`, while the background token decodes to ` the`, ` a` and `,`
from block 4 on. That is a nice picture for this one token in this one scene, and it comes from the unrestricted lens, which only shows the very last layers. It neither proves that the colour is
"in" the token at the earlier layers nor that it is used: a useful representation can fail to align with the pretrained unembedding. The next exercise quantifies the colour rank and a restricted readout over all 200 scenes before we try
interventions.
</details>

A table of three tokens for one scene can't settle anything, and decoded tokens are a notoriously good way to see what you expected to see. We need a number that can be computed
over all 200 scenes, and a control. Two metrics, both about the *true colour word* of the queried object:

1. **Rank.** At each layer, the rank of the true colour word (say `" blue"`) among all 50,257 tokens when the token's residual stream is decoded. Rank 1 means the lens's top prediction is the colour.
2. **Restricted accuracy.** The full-vocabulary rank is a harsh test: it asks the lens to put the colour above *every*
   word in the language, including plausible ones like `" the"`. A gentler test restricts attention to the six
   colour words and asks whether the right one has the highest logit among them (chance: 1/6). It asks whether
   the token carries colour information *in a form the unembedding can read*, without requiring the colour to
   beat unrelated vocabulary.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - rank of a word under the lens

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement `lens_ranks`. It takes `resid` of shape `(layer, token, d_model)` (the residual streams to decode, for one example) and a token id `word_id`, applies the logit lens
(`model.ln_final` then `model.unembed`), and returns the **rank** of `word_id` for every layer and token, shape `(layer, token)`. Rank 1 is the top prediction; define the rank as
one plus the number of tokens with a strictly higher logit.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def lens_ranks(
    model: HookedTransformer, resid: Float[Tensor, "layer token d_model"], word_id: int
) -> Int[Tensor, "layer token"]:
    """Rank (1 = top) of `word_id` in the logit-lens decoding of every residual stream vector."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    logits = model.unembed(model.ln_final(resid))
    return (logits > logits[..., word_id, None]).sum(dim=-1) + 1
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_lens_ranks(lens_ranks, vlm)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`model.unembed(model.ln_final(resid))` has shape `(layer, token, d_vocab)` (both act on the last dimension). The word's logit is `logits[..., word_id]`; keep a trailing dimension
of size 1 (`logits[..., word_id, None]`) so you can compare it against all logits with `>` and sum over the vocabulary.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - restricted lens accuracy

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement `restricted_lens_correct`. It takes the same `resid`, a tensor `candidate_ids` of the six colour-word ids, and the index `target_index` (0-5) of the right colour among
them, and returns a **boolean** tensor of shape `(layer, token)`: whether, among the candidates, the lens ranks the target highest. To save memory and time, don't compute the full
vocabulary: compute only the candidate logits, `ln_final(resid) @ W_U[:, candidate_ids] + b_U[candidate_ids]`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def restricted_lens_correct(
    model: HookedTransformer,
    resid: Float[Tensor, "layer token d_model"],
    candidate_ids: Int[Tensor, " candidate"],
    target_index: int,
) -> Bool[Tensor, "layer token"]:
    """True where the logit lens, restricted to the candidate words, puts the target candidate first."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    logits = model.ln_final(resid) @ model.W_U[:, candidate_ids] + model.b_U[candidate_ids]
    return logits.argmax(dim=-1) == target_index
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_restricted_lens_correct(restricted_lens_correct, vlm)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`model.ln_final(resid)` has shape `(layer, token, d_model)`; `model.W_U[:, candidate_ids]` has shape `(d_model, n_candidates)`. Their product has shape `(layer, token,
n_candidates)`, and the unembedding bias is `model.b_U[candidate_ids]`. Then `argmax(dim=-1) == target_index`.
</details>

Now the experiment. For every example we compute, for the 13 decoded layers: the median rank of the true colour over **the object's tokens**, over **background tokens** (a control:
nothing there should say what the queried object's colour is, if tokens were purely local), and over **random vectors with the same norm** as the object's tokens (a control for the
lens itself: it should show nothing). The same three sets for the restricted accuracy.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

lens_stats = {key: [] for key in ["rank_obj", "rank_bg", "rank_rand", "acc_obj", "acc_bg", "acc_rand"]}
for b in range(N_ANALYSIS):
    word_id, target = colour_ids[a_target_colour[b]], int(a_target_colour[b])
    obj_mask, bg_mask = a_masks["object"][b], ~a_masks["occupied"][b]
    resid = image_resid[:, b]  # (13, 49, d_model)
    noise = t.randn(resid[:, obj_mask].shape, generator=t.Generator().manual_seed(b)).to(device)
    noise = noise / noise.norm(dim=-1, keepdim=True) * resid[:, obj_mask].norm(dim=-1, keepdim=True)  # same norm as the real vectors
    ranks = lens_ranks(vlm, resid, word_id)
    correct = restricted_lens_correct(vlm, resid, colour_ids, target)
    lens_stats["rank_obj"].append(ranks[:, obj_mask].float().median(dim=1).values)
    lens_stats["rank_bg"].append(ranks[:, bg_mask].float().median(dim=1).values)
    lens_stats["rank_rand"].append(lens_ranks(vlm, noise, word_id).float().median(dim=1).values)
    lens_stats["acc_obj"].append(correct[:, obj_mask].float().mean(dim=1))
    lens_stats["acc_bg"].append(correct[:, bg_mask].float().mean(dim=1))
    lens_stats["acc_rand"].append(restricted_lens_correct(vlm, noise, colour_ids, target).float().mean(dim=1))
lens_stats = {k: t.stack(v) for k, v in lens_stats.items()}  # each (N_ANALYSIS, 13)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
utils.plot_layer_curves(
    {
        "object tokens": lens_stats["rank_obj"].median(0).values.cpu().numpy(),
        "background tokens": lens_stats["rank_bg"].median(0).values.cpu().numpy(),
        "random vectors (norm-matched)": lens_stats["rank_rand"].median(0).values.cpu().numpy(),
    },
    "Plain logit lens: rank of the true colour word",
    "median rank among 50,257 tokens (log scale)",
    controls=("background tokens", "random vectors (norm-matched)"),
    logy=True,
    ax=axes[0],
)
utils.plot_layer_curves(
    {
        "object tokens": lens_stats["acc_obj"].mean(0).cpu().numpy(),
        "background tokens": lens_stats["acc_bg"].mean(0).cpu().numpy(),
        "random vectors (norm-matched)": lens_stats["acc_rand"].mean(0).cpu().numpy(),
    },
    "Lens restricted to the six colour words",
    "fraction decoded as the true colour",
    controls=("background tokens", "random vectors (norm-matched)"),
    ylim=(0, 1.02),
    hlines={"chance (1/6)": 1 / 6},
    ax=axes[1],
)
for ax in axes:
    ax.set_xticks(range(13))
    ax.set_xticklabels(LAYER_LABELS, rotation=45, ha="right", fontsize=8)
plt.tight_layout()
plt.show()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/image_token_lens.png" width="900">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I interpret this figure?</summary>

In the unrestricted lens, the object's true colour has a median rank of a few hundred among the 50,257 words in the early layers, and about 40 at the final layer: much better than norm-matched random vectors
(about 15,000 throughout), but never first. Background tokens catch up late: their median rank falls from about 16,000 to about 450. Restricting the same lens to six colour words gives roughly 0.8 accuracy for the object's
tokens at every layer (0.79 to 0.86), versus 0.25 to 0.34 for background tokens and about 1/6 for random vectors. This supports decodable colour information in the object's tokens under a restricted readout; the
nonzero background result matters because SigLIP embeddings are contextual.
</details>

<details>
<summary>Question - given this, would you be comfortable claiming that "the logit lens shows the colour in the object's tokens"?</summary>

Only with the qualification that the six-colour restricted readout decodes the colour more accurately from object tokens than from background or random vectors. The unrestricted
top-token table is not a clean colour description, background also carries signal, and no readout establishes that GPT-2 uses these tokens to answer. The patching and knockout
tests below address that causal question.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Attention from the answer position

The lens asks what a token *contains*. A different question is what the answer position *reads*. Attention patterns give a direct view. Head $h$ in block $\ell$ writes into the
residual stream of the answer position (query position 59) a weighted sum of what it reads from every earlier position (the keys):

$$
\text{pattern}^{\ell, h}_{59, k} = \operatorname{softmax}_k\left( \frac{q_{59} \cdot k_k}{\sqrt{d_{head}}} \right), \qquad \text{sum over } k \le 59 \text{ is } 1 .
$$

We want to know how much of this attention mass lands on the object's tokens, per layer and per head, and to compare it against the same-size random control (an object has 4-5
tokens out of 49, so a head attending to *random* image tokens would put about 10% on them, and a head ignoring the image would put about 0).
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - attention mass on a token set

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 5-10 minutes on this exercise.
> ```

Implement `attention_mass`. It takes the attention patterns of **one example**, shape `(layer, head, query, key)`, a query position, and a boolean mask over key positions, and
returns the total attention that query position pays to the masked keys, for every layer and head: shape `(layer, head)`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def attention_mass(
    patterns: Float[Tensor, "layer head query key"], query_pos: int, key_mask: Bool[Tensor, " key"]
) -> Float[Tensor, "layer head"]:
    """Total attention from position `query_pos` onto the key positions where `key_mask` is True."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    return (patterns[:, :, query_pos] * key_mask).sum(dim=-1)
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_attention_mass(attention_mass)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`patterns[:, :, query_pos]` has shape `(layer, head, key)`. Multiply by the mask (it broadcasts over the last dimension) and sum over keys.
</details>

Now we compute the mass for every example and every token set. The sets are the object's tokens, the same-size random background control, the other objects' tokens, the BOS token
and the question tokens (positions 50-58, i.e. the question *excluding* the answer position itself), so that all of the answer position's attention is accounted for.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

def key_mask_for(token_mask: Tensor | None = None, positions: slice | None = None) -> Tensor:
    """A (SEQ_LEN,) boolean key mask: either image tokens given by a (49,) mask, or a slice of sequence positions."""
    mask = t.zeros(SEQ_LEN, dtype=t.bool, device=device)
    if token_mask is not None:
        mask[IMAGE_SLICE] = token_mask
    if positions is not None:
        mask[positions] = True
    return mask


mass = {name: [] for name in ["object", "same-size random background", "other objects", "BOS", "question"]}
for b in range(N_ANALYSIS):
    patterns = t.stack([a_cache["pattern", layer][b] for layer in range(N_LAYERS)])  # (layer, head, query, key)
    mass["object"].append(attention_mass(patterns, ANSWER_POS, key_mask_for(a_masks["object"][b])))
    mass["same-size random background"].append(attention_mass(patterns, ANSWER_POS, key_mask_for(a_masks["random background"][b])))
    mass["other objects"].append(attention_mass(patterns, ANSWER_POS, key_mask_for(a_masks["other objects"][b])))
    mass["BOS"].append(attention_mass(patterns, ANSWER_POS, key_mask_for(positions=slice(0, 1))))
    mass["question"].append(attention_mass(patterns, ANSWER_POS, key_mask_for(positions=slice(1 + N_IMAGE_TOKENS, ANSWER_POS))))
mass = {name: t.stack(v) for name, v in mass.items()}  # each (N_ANALYSIS, layer, head)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.5, 1]})
utils.plot_layer_curves(
    {name: mass[name].mean((0, 2)).cpu().numpy() for name in ["object", "same-size random background", "other objects", "question"]},
    "Attention from the answer position (mean over heads and 200 questions)",
    "fraction of the answer position's attention",
    xlabel="layer",
    controls=("same-size random background", "other objects"),
    ax=axes[0],
)
utils.plot_head_heatmap(mass["object"].mean(0), "Attention on the object's tokens, per head", ax=axes[1], vmax=1)
plt.tight_layout()
plt.show()

per_head = mass["object"].mean(0)
top = per_head.flatten().topk(8)
print("heads with the most attention on the object's tokens (layer.head: object / same-size random control):")
for value, index in zip(top.values.tolist(), top.indices.tolist()):
    layer, head = divmod(index, vlm.cfg.n_heads)
    control = mass["same-size random background"].mean(0)[layer, head].item()
    print(f"  L{layer}.H{head}: {value:.2f} / {control:.2f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/attention_to_object.png" width="900">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">heads with the most attention on the object's tokens (layer.head: object / same-size random control):
  L9.H6: 0.99 / 0.00
  L9.H9: 0.99 / 0.00
  L10.H1: 0.97 / 0.00
  L9.H0: 0.97 / 0.00
  L10.H10: 0.96 / 0.00
  L10.H0: 0.95 / 0.00
  L9.H8: 0.91 / 0.01
  L11.H6: 0.85 / 0.01</pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

Early on, the answer position attends substantially to question tokens while giving little mass to image tokens. Object attention rises in the last layers, with a mean peak of about 0.44 at
layer 10; the matched background control stays small. Individual heads are much more concentrated: L9.H6 and L9.H9 put about 0.99 mass on object tokens versus 0.00 on the
same-size background set. This locates a candidate reading route, but attention weights alone do not measure the information or causal importance carried by that route.
</details>

To *see* what a head does, we can draw its attention pattern from the answer position over the image tokens. Here is the head with the most attention on the object's tokens, on
four different scenes:
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

best_layer, best_head = divmod(int(per_head.flatten().argmax()), vlm.cfg.n_heads)
show = [0, 1, 2, 3, 4]
head_maps = np.stack([a_cache["pattern", best_layer][b, best_head, ANSWER_POS, IMAGE_SLICE].cpu().numpy() for b in show])
utils.plot_token_maps(
    [a_scenes[i] for i in show],
    [a_targets[i] for i in show],
    [head_maps],
    [f"attention of L{best_layer}.H{best_head}\nfrom the answer position"],
    [f"question about the {a_scenes[i].objects[a_targets[i]].colour} {a_scenes[i].objects[a_targets[i]].shape}" for i in show],
    suptitle=f"Where does the answer position look? (head L{best_layer}.H{best_head}; white outline = the queried object)",
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/attention_overlay.png" width="800">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - what is this showing?</summary>

These are answer-position attention maps for the selected head L9.H6, displayed on the 7×7 visual-token grid. In each of the five examples the bright cells are one or two tokens at the edge of
the white-outlined queried object, and the rest of the grid is dark: the head does not spread its attention over the whole object. The head was selected for high mean object attention on this analysis set, so this gallery illustrates that measured preference; it is not a new
validation set or evidence that every head localizes the object.
</details>

## The tokens the answer position reads most

One more set of tokens will serve as a control in the patching experiments below: the **image tokens that the answer position attends to most**. If attention told the whole story, patching
*those* tokens should work as well as patching the object's. If the object's tokens work better, then the position that is read most is not always the one that carries the content of the answer.
We keep as many tokens as the object has (call it $k$), so that the comparison with the object's tokens and with the same-size random background is fair.

To score each image token, we average the attention that the answer position pays to it over all layers and all heads.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - the most-attended tokens

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement `top_attention_mask`. It takes the attention patterns of **one example**, shape `(layer, head, query, key)`, a query position, a `slice` saying which key positions are image
tokens, and a number `k`. Average the attention from the query position over layers and heads for each image token, and return a boolean mask over the image tokens (one entry per
token in the slice, not over the whole sequence) with exactly the `k` best-attended tokens set to `True`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def top_attention_mask(
    patterns: Float[Tensor, "layer head query key"], query_pos: int, image_slice: slice, k: int
) -> Bool[Tensor, " image_token"]:
    """The k image tokens (keys in image_slice) with the highest attention from query_pos, averaged over layers and heads."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    scores = patterns[:, :, query_pos, image_slice].mean(dim=(0, 1))  # (image_token,)
    mask = t.zeros_like(scores, dtype=t.bool)
    mask[scores.topk(k).indices] = True
    return mask
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_top_attention_mask(top_attention_mask)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`patterns[:, :, query_pos, image_slice]` has shape `(layer, head, image_token)`. Average it over the first two dimensions, and use `scores.topk(k).indices` to find the best tokens. Only the image
tokens compete: the BOS token (position 0) usually soaks up a lot of attention, and it must not take a place in the mask.
</details>

Let's see how often the most-attended tokens are the object's own: for each of the 200 analysis questions, the fraction of the object's tokens that are among the $k$ best-attended ones.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

a_patterns = [t.stack([a_cache["pattern", layer][b] for layer in range(N_LAYERS)]) for b in range(N_ANALYSIS)]  # each (layer, head, query, key)
a_top = t.stack([top_attention_mask(a_patterns[b], ANSWER_POS, IMAGE_SLICE, int(a_masks["object"][b].sum())) for b in range(N_ANALYSIS)])
overlap = ((a_top & a_masks["object"]).sum(1) / a_masks["object"].sum(1)).float()
print(f"fraction of the object's tokens that are among the most-attended: {overlap.mean():.2f} on average (a random choice of the same size would give {(a_masks['object'].sum(1) / N_IMAGE_TOKENS).float().mean():.2f})")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">fraction of the object's tokens that are among the most-attended: 0.23 on average (a random choice of the same size would give 0.09)</pre>

</details>

On average 23% of the object's tokens are among the $k$ most-attended ones, against 9% for a random choice of the same size. So the most-attended tokens overlap the object more than chance but are mostly something else, and patching them is a real test of whether attention tells you where the answer lives.

## Activation patching on image tokens

Attention is a statement about *where information could flow*; it isn't a statement about what the answer *depends on*. (An attention head can look at a token and move nothing
useful from it, and the question gets answered by something else.) To test dependence we have to **intervene**, which is what activation patching does. You've seen it for text in
[1.4.1 Indirect Object Identification](https://learn.arena.education/chapter1_transformer_interp/21_ioi/); here the same idea applies to image tokens, and the setup is particularly clean.

**Counterfactual pairs.** We build 100 *pairs of scenes* that differ in exactly one thing: the **colour of the queried object**. Everything else (geometry, background, noise) is
identical. We ask the same question ("What colour is the circle?") on both, and the model answers correctly on each (we'll check). The *clean* scene is the one with the original
colour (say red) and the *corrupted* scene has the object recoloured (say blue). To read out the effect we use a **logit difference** between the two answers:

$$
\text{LD} = \text{logit}(\text{" red"}) - \text{logit}(\text{" blue"}) \quad \text{(at the answer position)},
$$

which is large and positive on the clean run, and large and negative on the corrupted run. Now **patch**: run the corrupted scene, but at layer $\ell$ and at a chosen set $S$ of
positions, overwrite the residual stream (`resid_pre` of block $\ell$) with its value from the clean run. If $S$ contains the information that distinguishes the two scenes (and the
model uses it from there), the logit difference should move towards the clean value. We normalise:

$$
\text{recovery}(S, \ell) = \frac{\text{LD}_{\text{patched}} - \text{LD}_{\text{corrupted}}}{\text{LD}_{\text{clean}} - \text{LD}_{\text{corrupted}}} ,
$$

so that 0 means "the patch did nothing" and 1 means "the patch fully restored the clean behaviour".

Because the two scenes differ only in the object's colour, **the object's image tokens are where the two scenes differ** (roughly: SigLIP is a transformer with global attention, so
some of the difference leaks into other tokens, a point we'll come back to). That suggests patching the object's tokens should restore the answer, patching background tokens should not,
and patching the *answer position* should only start to matter once the information has been moved there.

**Make a prediction first:** if you patch the object's tokens with the clean run's values at layer 4, what fraction of the clean-versus-corrupted logit difference will come back, and what
fraction will the same number of random background tokens restore?
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

pairs = utils.make_counterfactual_pairs(100, seed=7)
clean_scenes, corrupt_scenes = [p[0] for p in pairs], [p[1] for p in pairs]
pair_targets = [p[2] for p in pairs]
clean_features, corrupt_features = featurise(clean_scenes), featurise(corrupt_scenes)

pair_prompts = [question_and_answer(scene, "colour", target)[0] for scene, target in zip(clean_scenes, pair_targets)]
pair_tokens, pair_lens = utils.tokenize_prompts(vlm, pair_prompts)
assert (pair_lens == a_lens[0]).all()
clean_ids = t.tensor([vlm.to_single_token(" " + s.objects[i].colour) for s, i in zip(clean_scenes, pair_targets)], device=device)
corrupt_ids = t.tensor([vlm.to_single_token(" " + s.objects[i].colour) for s, i in zip(corrupt_scenes, pair_targets)], device=device)

with t.no_grad():
    clean_embeds = build_input_embeddings(vlm, projector(clean_features), pair_tokens)
    corrupt_embeds = build_input_embeddings(vlm, projector(corrupt_features), pair_tokens)
    _, clean_cache = vlm.run_with_cache(
        clean_embeds,
        start_at_layer=0,
        stop_at_layer=N_LAYERS,
        names_filter=lambda name: name.endswith("resid_pre") or name.endswith("hook_pattern"),
    )
    clean_resids = {layer: clean_cache["resid_pre", layer] for layer in range(N_LAYERS)}  # layer -> (100, 60, 768)
    clean_patterns = [t.stack([clean_cache["pattern", layer][b] for layer in range(N_LAYERS)]) for b in range(len(pairs))]  # each (layer, head, query, key)
    clean_logits = vlm.unembed(vlm.ln_final(vlm(clean_embeds, start_at_layer=0, stop_at_layer=N_LAYERS)[:, ANSWER_POS]))
    corrupt_logits = vlm.unembed(vlm.ln_final(vlm(corrupt_embeds, start_at_layer=0, stop_at_layer=N_LAYERS)[:, ANSWER_POS]))

print(f"the model gets the clean scenes right: {(clean_logits.argmax(-1) == clean_ids).float().mean():.0%}, the corrupted scenes right: {(corrupt_logits.argmax(-1) == corrupt_ids).float().mean():.0%}")
rows = t.arange(len(pairs))
ld_clean = (clean_logits[rows, clean_ids] - clean_logits[rows, corrupt_ids]).mean().item()
ld_corrupt = (corrupt_logits[rows, clean_ids] - corrupt_logits[rows, corrupt_ids]).mean().item()
print(f"mean logit difference (clean colour - corrupt colour): clean run {ld_clean:+.1f}, corrupted run {ld_corrupt:+.1f}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">the model gets the clean scenes right: 100%, the corrupted scenes right: 100%
mean logit difference (clean colour - corrupt colour): clean run +15.9, corrupted run -15.6</pre>

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - patch the residual stream

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 20 minutes on this exercise.
> ```

Implement `patched_answer_logits`. It takes the **corrupted** run's input embeddings `(batch, seq, d_model)`, the **clean** run's residual stream before block `layer`
(`clean_resid`, same shape), the layer, a boolean `patch_mask` of shape `(batch, seq)` saying which (example, position) entries to overwrite, and the answer position. It should run
the corrupted input through the model with a hook on `blocks.{layer}.hook_resid_pre` that copies `clean_resid` into the residual stream at the masked entries, and return the
answer-position logits `(batch, d_vocab)`.

Use `model.run_with_hooks(embeds, start_at_layer=0, stop_at_layer=model.cfg.n_layers, fwd_hooks=[(hook_name, hook_fn)])`, as in the forward pass of Part 2, which returns the final
residual stream (before `ln_final`). Then apply `ln_final` and `unembed` at the answer position.

**One trap:** when you start at layer 0, `blocks.0.hook_resid_pre` receives *the very tensor you passed in as the input*. If your hook modifies it in place, you will silently
change the corrupted embeddings that you later reuse for other conditions. **Clone the residual stream inside the hook** before writing to it. The test checks for exactly this.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def patched_answer_logits(
    model: HookedTransformer,
    corrupt_embeds: Float[Tensor, "batch seq d_model"],
    clean_resid: Float[Tensor, "batch seq d_model"],
    layer: int,
    patch_mask: Bool[Tensor, "batch seq"],
    answer_pos: int,
) -> Float[Tensor, "batch d_vocab"]:
    """Logits at the answer position of the corrupted run, with resid_pre of block `layer` overwritten from the clean run where patch_mask."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    def patch_hook(resid: Float[Tensor, "batch seq d_model"], hook) -> Float[Tensor, "batch seq d_model"]:
        resid = resid.clone()
        resid[patch_mask] = clean_resid[patch_mask]
        return resid

    resid = model.run_with_hooks(
        corrupt_embeds,
        start_at_layer=0,
        stop_at_layer=model.cfg.n_layers,
        fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", patch_hook)],
    )
    return model.unembed(model.ln_final(resid[:, answer_pos]))
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_patched_answer_logits(patched_answer_logits, vlm)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Inside the hook: `resid = resid.clone(); resid[patch_mask] = clean_resid[patch_mask]; return resid`. Indexing a `(batch, seq, d_model)` tensor with a `(batch, seq)` boolean mask
selects the masked `(example, position)` vectors, shape `(n_masked, d_model)`, on both sides of the assignment.
</details>

<details>
<summary>Help - the test says patching nothing changed the answer (or the clean run is not reproduced)</summary>

Two common causes: (1) forgetting to clone (an in-place edit of the input is visible to later code), and (2) applying `ln_final` to the wrong position. The test checks that an
all-False mask reproduces the corrupted run exactly, and that an all-True mask at any layer reproduces the clean run exactly.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - the recovery metric

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement `logit_diff_recovery`. Given the patched, clean and corrupted answer logits (each `(batch, d_vocab)`) and the token ids of the clean and corrupted answers (each
`(batch,)`), compute the logit difference $\text{LD} = \text{logit}[\text{clean id}] - \text{logit}[\text{corrupt id}]$ for each of the three runs and return the recovery
$(\text{LD}_{\text{patched}} - \text{LD}_{\text{corrupted}}) / (\text{LD}_{\text{clean}} - \text{LD}_{\text{corrupted}})$ per example, shape `(batch,)`.

There is a trap in the denominator. If a pair's clean run does not beat its corrupted run by a real margin (a gap of zero, or a clean run that is actually *below* the corrupted one), the
ratio is infinite, undefined or the wrong sign, and one such pair would quietly wreck the mean of the whole batch. So **before computing anything else**, raise a `ValueError` if any pair
in the batch has a clean-minus-corrupted gap that is zero, negative or not a finite number, or if any recovery comes out not finite. Stop the whole batch: don't drop the bad pairs or
replace them by zero, because then the mean no longer describes the pairs you chose. A tiny positive gap is still a gap, so don't add a threshold, and recovery above 1 or below 0 is a real result
(a patch can overshoot), so don't clip it.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def logit_diff_recovery(
    patched_logits: Float[Tensor, "batch d_vocab"],
    clean_logits: Float[Tensor, "batch d_vocab"],
    corrupt_logits: Float[Tensor, "batch d_vocab"],
    clean_ids: Int[Tensor, " batch"],
    corrupt_ids: Int[Tensor, " batch"],
) -> Float[Tensor, " batch"]:
    """0 = patched run looks like the corrupted run, 1 = it looks like the clean run (in logit difference between the two answers)."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    def logit_diff(logits: Tensor) -> Tensor:
        rows = t.arange(len(logits), device=logits.device)
        return logits[rows, clean_ids] - logits[rows, corrupt_ids]

    gap = logit_diff(clean_logits) - logit_diff(corrupt_logits)
    bad = ~(t.isfinite(gap) & (gap > 0))
    if bad.any():
        raise ValueError(f"Pairs {bad.nonzero().flatten().tolist()} have a clean-minus-corrupted gap that is zero, negative or not finite: {gap[bad].tolist()}")
    recovery = (logit_diff(patched_logits) - logit_diff(corrupt_logits)) / gap
    if not t.isfinite(recovery).all():
        raise ValueError(f"Recovery is not finite for pairs {(~t.isfinite(recovery)).nonzero().flatten().tolist()}")
    return recovery
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_logit_diff_recovery(logit_diff_recovery)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Write a small inner function `logit_diff(logits)` that returns `logits[rows, clean_ids] - logits[rows, corrupt_ids]` with `rows = t.arange(len(logits))`, and call it on all three
tensors. Compute the gap `logit_diff(clean_logits) - logit_diff(corrupt_logits)` first, and raise if `t.isfinite(gap) & (gap > 0)` is false for any pair. `t.isfinite` is also what to
use on the final recovery.
</details>

<details>
<summary>Help - the test says a bad pair did not stop the batch</summary>

Check that you raise (and don't just filter or fix). The test builds a batch of ten pairs where pair 7 has a gap of 4 and pair 9 has a gap of 0: the right behaviour is an error for the whole
batch, not eight good numbers and two repairs. It also checks that a gap of $10^{-6}$ is accepted and that a recovery of 1.5 is returned unchanged.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - a patching sweep

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 10-15 minutes on this exercise.
> ```

Implement `patching_sweep`, which produces the curves for the main figure. Given a dictionary of position masks (name → boolean `(batch, seq)`), loop over layers and over masks,
use `patched_answer_logits` and `logit_diff_recovery`, and return, for every mask name, a tensor of shape `(len(layers), batch)` with the **recovery of every pair** at each layer. (We keep
every pair's recovery, not only the mean, because the uncertainty bands in the final figure come from resampling pairs.) Run everything under `t.no_grad()`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def patching_sweep(
    model: HookedTransformer,
    corrupt_embeds: Float[Tensor, "batch seq d_model"],
    clean_resids: dict[int, Float[Tensor, "batch seq d_model"]],
    masks: dict[str, Bool[Tensor, "batch seq"]],
    clean_logits: Float[Tensor, "batch d_vocab"],
    corrupt_logits: Float[Tensor, "batch d_vocab"],
    clean_ids: Int[Tensor, " batch"],
    corrupt_ids: Int[Tensor, " batch"],
    answer_pos: int,
    layers: list[int],
) -> dict[str, Float[Tensor, "layer batch"]]:
    """For each named position mask: the recovery of every pair (shape (len(layers), batch)) from patching it at each layer in `layers`."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    curves = {name: [] for name in masks}
    with t.no_grad():
        for layer in layers:
            for name, mask in masks.items():
                patched = patched_answer_logits(model, corrupt_embeds, clean_resids[layer], layer, mask, answer_pos)
                curves[name].append(logit_diff_recovery(patched, clean_logits, corrupt_logits, clean_ids, corrupt_ids))
    return {name: t.stack(rows) for name, rows in curves.items()}
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_patching_sweep(patching_sweep, vlm)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`clean_resids[layer]` is the clean run's `resid_pre` at that layer. For each `(layer, name)`: `patched = patched_answer_logits(model, corrupt_embeds, clean_resids[layer], layer,
mask, answer_pos)`, then `logit_diff_recovery(patched, ...)` has shape `(batch,)`: append it to `curves[name]`, and at the end stack each list into a `(layer, batch)` tensor with `t.stack`.
</details>

Now the position sets. All masks have shape `(100, 60)`: one row per pair, one column per sequence position. For the image-token sets we use the same constructions as before (the
object's tokens, the same-size random control, the other objects' tokens), and we add four more: the **highest-attention tokens** (the $k$ image tokens that the answer position attends to most
in the clean run, found with your `top_attention_mask`, with $k$ the number of the object's tokens), **all image tokens** (an upper bound: this replaces the whole image, so it should
recover everything until the information has left), **the question tokens** (positions 50-58) and **the answer position** (59).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

p_masks = token_set_masks(clean_scenes, pair_targets, seed=1)
p_masks["highest attention"] = t.stack(
    [top_attention_mask(clean_patterns[b], ANSWER_POS, IMAGE_SLICE, int(p_masks["object"][b].sum())) for b in range(len(pairs))]
)


def seq_mask(token_mask: Tensor | None = None, positions: slice | None = None) -> Tensor:
    """(100, SEQ_LEN) boolean mask: image tokens given by a (100, 49) mask, and/or a slice of sequence positions."""
    mask = t.zeros(len(pairs), SEQ_LEN, dtype=t.bool, device=device)
    if token_mask is not None:
        mask[:, IMAGE_SLICE] = token_mask
    if positions is not None:
        mask[:, positions] = True
    return mask


patch_sets = {
    "object tokens": seq_mask(p_masks["object"]),
    "same-size random background": seq_mask(p_masks["random background"]),
    "other objects": seq_mask(p_masks["other objects"]),
    "highest-attention tokens": seq_mask(p_masks["highest attention"]),
    "all image tokens": seq_mask(t.ones_like(p_masks["object"])),
    "question tokens (50-58)": seq_mask(positions=slice(1 + N_IMAGE_TOKENS, ANSWER_POS)),
    "answer position (59)": seq_mask(positions=slice(ANSWER_POS, ANSWER_POS + 1)),
}
patch_recovery = patching_sweep(  # name -> (layer, pair): the recovery of every pair at every layer
    vlm, corrupt_embeds, clean_resids, patch_sets, clean_logits, corrupt_logits, clean_ids, corrupt_ids, ANSWER_POS, list(range(N_LAYERS))
)
patch_curves = {name: values.mean(dim=1).tolist() for name, values in patch_recovery.items()}  # the mean over the 100 pairs
utils.plot_layer_curves(
    patch_curves,
    "Patching clean -> corrupted: fraction of the logit difference recovered",
    "recovery (0 = corrupted answer, 1 = clean answer)",
    controls=("same-size random background", "other objects", "highest-attention tokens", "all image tokens"),
    ylim=(-0.05, 1.08),
)
print(pd.DataFrame(patch_curves, index=[f"{layer}" for layer in range(N_LAYERS)]).T.round(2).to_string())

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/patching_recovery.png" width="700">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                0     1     2     3     4     5     6     7     8     9    10    11
object tokens                0.94  0.94  0.94  0.94  0.94  0.94  0.94  0.94  0.93  0.93  0.61  0.19
same-size random background  0.01  0.01  0.01  0.01  0.01  0.01  0.01  0.01  0.01  0.01  0.01  0.00
other objects                0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
highest-attention tokens     0.49  0.48  0.49  0.49  0.49  0.49  0.49  0.49  0.48  0.48  0.34  0.14
all image tokens             1.00  1.00  1.00  1.00  1.00  1.00  1.00  1.00  0.99  0.98  0.64  0.20
question tokens (50-58)      0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.01  0.02
answer position (59)         0.00  0.00 -0.00  0.00  0.00  0.00  0.00  0.00  0.01  0.01  0.35  0.78</pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

On 100 recolouring pairs, patching the target object's tokens recovers about 0.94 of the clean-versus-corrupt logit gap through layers 0–9, against about 0.01 for same-size random
background and 0.00 for other objects. All image tokens recover 1.00 there (0.98 at layer 9). Object recovery then falls to 0.61 at layer 10 and 0.19 at layer 11 while answer-position recovery rises to 0.35 and 0.78;
question-token recovery stays near zero. This supports a late transfer into the answer-position residual stream. The new control, the $k$ image tokens that the answer position attends to most, recovers
about 0.49 through layer 9: about half of what the object's tokens do, and far more than random background. Attention points at useful tokens, but not only at the object's tokens, and the object is the better place to look. Recovery is a normalized logit contrast, not percent accuracy, and
the remaining gap between object and all-image patching leaves room for contextual information outside the geometric object mask.
</details>

This is the main result of the section: *the answer to a colour question is carried by the few image tokens that cover the object, up to a layer, after which it has been moved to
the answer position*, and same-size random background and other-object controls recover little: about 0.01 and 0.00 through the early and middle layers. These controls are
small, not literal zeros. The same figure is the first half of the lesson's signature result, which we redraw against Qwen3-VL in Part 5.

### Which token matters? Single-token patching

Patching the whole set tells us the *group* is sufficient. Is the information spread evenly over the object's tokens, or concentrated in one? Let's patch **one image token at a
time** (49 patched runs per scene) at a fixed layer, and draw the recovery for each token as a $7 \times 7$ map on top of the scene. The code is a loop over your own functions:
each of the 49 runs uses a one-hot mask.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

@t.no_grad()
def single_token_recovery(pair: int, layer: int) -> Float[Tensor, "token"]:
    """Recovery obtained by patching each of the 49 image tokens (separately) of one pair at the given layer."""
    mask = t.zeros(N_IMAGE_TOKENS, SEQ_LEN, dtype=t.bool, device=device)
    mask[t.arange(N_IMAGE_TOKENS), 1 + t.arange(N_IMAGE_TOKENS)] = True
    expand = lambda x: x[pair : pair + 1].expand(N_IMAGE_TOKENS, *x.shape[1:])
    patched = patched_answer_logits(vlm, expand(corrupt_embeds), expand(clean_resids[layer]), layer, mask, ANSWER_POS)
    return logit_diff_recovery(patched, expand(clean_logits), expand(corrupt_logits), expand(clean_ids), expand(corrupt_ids))


show = [0, 1, 2, 3, 4]
map_layers = [0, 6, 10, 11]
token_maps = [np.stack([single_token_recovery(i, layer).cpu().numpy() for i in show]) for layer in map_layers]
for layer, m in zip(map_layers, token_maps):
    in_obj = np.concatenate([m[k][p_masks["object"][i].cpu().numpy()] for k, i in enumerate(show)])
    in_bg = np.concatenate([m[k][~p_masks["occupied"][i].cpu().numpy()] for k, i in enumerate(show)])
    print(f"layer {layer:>2}: mean recovery of a single object token {in_obj.mean():.2f} (max {in_obj.max():.2f}), of a single background token {in_bg.mean():.3f} (max {in_bg.max():.3f})")
utils.plot_token_maps(
    corrupt_scenes[:5],
    pair_targets[:5],
    token_maps,
    [f"patch one token\nat layer {layer}" for layer in map_layers],
    [f"corrupted scene: {corrupt_scenes[i].objects[pair_targets[i]].colour} {corrupt_scenes[i].objects[pair_targets[i]].shape}\n(clean: {clean_scenes[i].objects[pair_targets[i]].colour})" for i in show],
    vmax=0.5,
    suptitle="Fraction of the logit difference recovered by patching ONE image token (white outline = the recoloured object)",
)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/patching_token_maps.png" width="800">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">layer  0: mean recovery of a single object token 0.25 (max 0.82), of a single background token 0.000 (max 0.024)
layer  6: mean recovery of a single object token 0.24 (max 0.78), of a single background token 0.000 (max 0.032)
layer 10: mean recovery of a single object token 0.17 (max 0.49), of a single background token 0.000 (max 0.033)
layer 11: mean recovery of a single object token 0.10 (max 0.27), of a single background token 0.000 (max 0.036)</pre>

</details>

<details>
<summary>Help - how should I interpret these maps?</summary>

Single-token effects mostly concentrate on the recoloured object, and weaken late as its information reaches the answer position. Mean object-token recovery falls from 0.25 at
layer 0 to 0.10 at layer 11; background means are 0.000 at every layer shown. The best single background token recovers at most 0.036, while the best single object token recovers 0.82 at layer 0, so avoid claiming that every background
position is exactly inert. The maps and printed means/maxima summarize the same five example pairs, not the full 100-pair sweep. Single-token effects need not add linearly to the effect of
patching a set.
</details>

## Necessary, not just sufficient: attention knockout

Patching shows that the object's tokens are **sufficient** to carry the answer: if you give the model the clean content in those positions, the answer follows. It doesn't show that
the model **needs** them via any particular route, because a transformer can often get the same information along more than one path. To test necessity, Zhang et al. (2025) use
**attention knockout**: they block attention between two groups of tokens in a window of layers, and check whether the answer survives. We'll do the same with the three groups of
positions we have: the object's image tokens, the question tokens (50-58), and the answer position (59).

Mechanically: in a chosen set of layers we set the attention *scores* from the blocked query positions to the blocked key positions to $-\infty$ before the softmax, so that those
keys receive zero attention and the remaining attention renormalises over the other keys. The model is not told; it just finds that those keys have vanished from certain heads'
view.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - block attention

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Implement `block_attention_logits`. It takes input embeddings `(batch, seq, d_model)`, a list of `layers`, a boolean tensor `block` of shape `(batch, seq, seq)` (`block[b, q, k]`
is True if query position `q` must not attend to key position `k` in example `b`), and the answer position, and returns the answer-position logits `(batch, d_vocab)`. In each layer
in `layers`, add a hook on `blocks.{layer}.attn.hook_attn_scores` (shape `(batch, head, query, key)`, already causally masked) that fills the blocked entries with `-inf`:
`scores.masked_fill(block[:, None], float("-inf"))`.

The test uses an invariance as an oracle: if the answer position and all question tokens are blocked from every image token in every layer, then the answer logits **cannot depend
on the image at all**.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def block_attention_logits(
    model: HookedTransformer,
    embeds: Float[Tensor, "batch seq d_model"],
    layers: list[int],
    block: Bool[Tensor, "batch seq seq"],
    answer_pos: int,
) -> Float[Tensor, "batch d_vocab"]:
    """Answer-position logits when, in every block in `layers`, query q may not attend to key k wherever block[b, q, k]."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    def block_hook(scores: Float[Tensor, "batch head query key"], hook) -> Float[Tensor, "batch head query key"]:
        return scores.masked_fill(block[:, None], float("-inf"))

    resid = model.run_with_hooks(
        embeds,
        start_at_layer=0,
        stop_at_layer=model.cfg.n_layers,
        fwd_hooks=[(f"blocks.{layer}.attn.hook_attn_scores", block_hook) for layer in layers],
    )
    return model.unembed(model.ln_final(resid[:, answer_pos]))
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_block_attention_logits(block_attention_logits, vlm)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

The hook receives the attention scores of one block, shape `(batch, head, query, key)`. `block` has shape `(batch, query, key)`, so `block[:, None]` (shape `(batch, 1, query,
key)`) broadcasts over heads. Return `scores.masked_fill(block[:, None], float("-inf"))`. Pass one hook per layer in `layers` in `fwd_hooks`.
</details>

We block five routes, each in a window of three consecutive layers, and in all layers at once, and measure the forced-choice accuracy on the 200 colour questions (it is 100%
unblocked). The routes:

- **answer position → object's tokens** (the route we expect to matter),
- **answer position → same-size random background tokens** (the control: the same *number* of tokens blocked, none of them the object),
- **answer position → all image tokens**,
- **question tokens → all image tokens** (can the question tokens "pre-read" the image?),
- **answer position → question tokens** (does the answer position need to read the question?).
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

def route_block(query: str, keys: str) -> Bool[Tensor, "batch seq seq"]:
    """Boolean (N_ANALYSIS, seq, seq) 'do not attend' mask for a route: queries in {answer, question} x keys in {object, ...}."""
    query_pos = {"answer": slice(ANSWER_POS, ANSWER_POS + 1), "question": slice(1 + N_IMAGE_TOKENS, ANSWER_POS)}[query]
    key_mask = {
        "object": lambda: t.stack([key_mask_for(a_masks["object"][b]) for b in range(N_ANALYSIS)]),
        "random background": lambda: t.stack([key_mask_for(a_masks["random background"][b]) for b in range(N_ANALYSIS)]),
        "image": lambda: key_mask_for(positions=IMAGE_SLICE).expand(N_ANALYSIS, -1),
        "question": lambda: key_mask_for(positions=slice(1 + N_IMAGE_TOKENS, ANSWER_POS)).expand(N_ANALYSIS, -1),
    }[keys]()
    block = t.zeros(N_ANALYSIS, SEQ_LEN, SEQ_LEN, dtype=t.bool, device=device)
    block[:, query_pos, :] = key_mask[:, None, :]
    return block


@t.no_grad()
def blocked_accuracy(layers: list[int], block: Tensor) -> float:
    logits = block_attention_logits(vlm, a_embeds, layers, block, ANSWER_POS)
    return answer_accuracy(logits, colour_ids[a_target_colour], colour_ids)


routes = {
    "answer -> object tokens": ("answer", "object"),
    "answer -> same-size random background": ("answer", "random background"),
    "answer -> all image tokens": ("answer", "image"),
    "question tokens -> all image tokens": ("question", "image"),
    "answer -> question tokens": ("answer", "question"),
}
windows = [(0, 3), (3, 6), (6, 9), (9, 12)]
knockout = {}
for name, (query, keys) in routes.items():
    block = route_block(query, keys)
    knockout[name] = [blocked_accuracy(list(range(lo, hi)), block) for lo, hi in windows] + [blocked_accuracy(list(range(N_LAYERS)), block)]
window_labels = [f"layers {lo}-{hi - 1}" for lo, hi in windows] + ["all layers"]
display(pd.DataFrame(knockout, index=window_labels).T.round(2))

fig, ax = plt.subplots(figsize=(8.5, 4.2))
for i, (name, accs) in enumerate(knockout.items()):
    style = "--s" if "random" in name or name.startswith("question") else "-o"
    ax.plot(range(len(accs)), accs, style, label=name, color=[utils.BLUE, utils.GREY, utils.GREEN, utils.ORANGE, utils.PURPLE][i], markersize=5)
ax.axhline(IMAGE_BLIND["colour"], color="k", ls=":", lw=1.2, label="best image-blind guess")
ax.set_xticks(range(len(window_labels)))
ax.set_xticklabels(window_labels)
ax.set_ylim(0, 1.05)
ax.set_ylabel("colour accuracy (forced choice)")
ax.set_xlabel("blocked in which layers")
ax.set_title("Attention knockout: block a route, measure the answer")
ax.grid(alpha=0.3)
ax.legend(fontsize=8, loc="lower left")
plt.tight_layout()
plt.show()

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/attention_knockout.png" width="700">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                       layers 0-2  layers 3-5  layers 6-8  \
answer -&gt; object tokens                       1.0        1.00        1.00   
answer -&gt; same-size random background         1.0        1.00        1.00   
answer -&gt; all image tokens                    1.0        1.00        1.00   
question tokens -&gt; all image tokens           1.0        1.00        1.00   
answer -&gt; question tokens                     1.0        0.81        0.84   

                                       layers 9-11  all layers  
answer -&gt; object tokens                       0.64        0.63  
answer -&gt; same-size random background         1.00        1.00  
answer -&gt; all image tokens                    0.65        0.65  
question tokens -&gt; all image tokens           1.00        1.00  
answer -&gt; question tokens                     1.00        0.52  </pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

Blocking answer-to-object attention in layers 9–11 reduces colour accuracy from 1.00 to 0.64, while blocking the same-size background set leaves 1.00. Blocking all image tokens
late gives 0.65, the same as blocking the object alone, and across all layers it also reaches 0.65. There is substantial residual performance, so object attention is useful but not the only surviving route.
Blocking answer-to-question attention in layers 3–8 also hurts (0.81 and 0.84; 0.52 across all layers), whereas blocking question-to-image attention alone leaves 1.00. These are separate route ablations with
attention renormalization; the cell does not test a combined blockade of every possible route.
</details>

<details>
<summary>Question - this differs from what Zhang et al. report for LLaVA. What do you expect to explain the difference?</summary>

Architecture, training objective, synthetic scene simplicity, question templates and the selected layer windows could all change the route used. These are hypotheses, not findings
of this comparison. The intervention also renormalizes the surviving attention edges and blocks routes in groups, so matching the paper's exact intervention and task would be
necessary before treating a difference as a model-level mechanism difference. A useful follow-up is a joint route blockade and head-level ablations on held-out tasks; the current
cell tests neither.
</details>

## Putting the four kinds of evidence side by side

We can now fill in the table we started the section with, using numbers from the experiments above: for each kind of evidence, the object's tokens against the matching control.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

late = slice(9, 12)  # the layers where the answer position reads the image
early = slice(0, 9)
summary = pd.DataFrame(
    {
        "object's tokens": {
            "decodable: restricted lens accuracy (mean over layers)": lens_stats["acc_obj"][:, :12].mean().item(),
            "attended: attention mass from answer position, layers 9-11": mass["object"][:, late].mean().item(),
            "sufficient: patching recovery, layers 0-8": float(np.mean([patch_curves["object tokens"][layer] for layer in range(9)])),
            "necessary: accuracy when the answer position can't see them, layers 9-11": knockout["answer -> object tokens"][3],
        },
        "same-size control": {
            "decodable: restricted lens accuracy (mean over layers)": lens_stats["acc_bg"][:, :12].mean().item(),
            "attended: attention mass from answer position, layers 9-11": mass["same-size random background"][:, late].mean().item(),
            "sufficient: patching recovery, layers 0-8": float(np.mean([patch_curves["same-size random background"][layer] for layer in range(9)])),
            "necessary: accuracy when the answer position can't see them, layers 9-11": knockout["answer -> same-size random background"][3],
        },
    }
)
display(summary.round(2))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">                                                    object's tokens  \
decodable: restricted lens accuracy (mean over ...             0.82   
attended: attention mass from answer position, ...             0.34   
sufficient: patching recovery, layers 0-8                      0.94   
necessary: accuracy when the answer position ca...             0.64   

                                                    same-size control  
decodable: restricted lens accuracy (mean over ...               0.28  
attended: attention mass from answer position, ...               0.05  
sufficient: patching recovery, layers 0-8                        0.01  
necessary: accuracy when the answer position ca...               1.00  </pre>

(The control for "decodable" is all background tokens, since the lens doesn't depend on set size.)
</details>

<details>
<summary>Question - which of the four rows would you trust least to tell you where the answer comes from, and why?</summary>

The decodability row is weakest for locating actual use: its 0.82 versus 0.28 readout accuracies show recoverable signal without showing that the model reads it. Attention (0.34
versus 0.05 mass) adds a candidate route; patching (0.94 versus 0.01 recovery) directly tests sufficiency for this contrast. Knockout (0.64 versus 1.00 remaining accuracy) tests
dependence on a route but leaves alternatives. These rows use different units and aggregation windows, so do not compare their magnitudes as one common importance score. The
decodability control uses all background tokens, unlike the matched-size controls in the other rows.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Try it yourself - patch one scene into another

The function below builds two scenes from lists of objects, asks the same colour question on both, and then patches the clean scene's residual stream into the corrupted scene at
the chosen layer and positions. It prints the model's answer on the clean scene, on the corrupted scene and after patching, and the recovery. The choice of positions is `"object"`
(the target's tokens), `"background"` (a same-size random set), `"all"` (all image tokens), `"answer"` (position 59), or an explicit list of `(row, col)` token coordinates. Scenes
should have the same size and number of objects, so that the two runs have the same sequence layout.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

@t.no_grad()
def patch_demo(clean: list[SceneObject], corrupt: list[SceneObject], layer: int, where="object", target: int = 0) -> None:
    clean_scene, corrupt_scene = Scene(clean, noise_seed=0), Scene(corrupt, noise_seed=0)
    prompt, _ = question_and_answer(corrupt_scene, "colour", target)
    tokens, _ = utils.tokenize_prompts(vlm, [prompt])
    feats = featurise([clean_scene, corrupt_scene])
    embeds = build_input_embeddings(vlm, projector(feats), tokens.expand(2, -1))
    _, cache = vlm.run_with_cache(embeds, start_at_layer=0, stop_at_layer=N_LAYERS, names_filter=lambda name: name.endswith("resid_pre"))
    logits = vlm.unembed(vlm.ln_final(vlm(embeds, start_at_layer=0, stop_at_layer=N_LAYERS)[:, ANSWER_POS]))
    obj = np.asarray(utils.object_token_mask(corrupt_scene.objects[target])).flatten()
    mask = t.zeros(1, SEQ_LEN, dtype=t.bool, device=device)
    if where == "object":
        mask[0, IMAGE_SLICE] = t.tensor(obj, device=device)
    elif where == "background":
        occupied = t.tensor(np.any([utils.object_token_mask(o).flatten() for o in corrupt_scene.objects], axis=0), device=device)[None]
        mask[0, IMAGE_SLICE] = random_background_mask(t.tensor(obj, device=device)[None], occupied)[0]
    elif where == "all":
        mask[0, IMAGE_SLICE] = True
    elif where == "answer":
        mask[0, ANSWER_POS] = True
    else:
        for row, col in where:
            mask[0, 1 + TOKEN_GRID * row + col] = True
    patched = patched_answer_logits(vlm, embeds[1:], cache["resid_pre", layer][:1], layer, mask, ANSWER_POS)
    candidates = [" " + c for c in COLOURS]

    def show(name: str, lg: Tensor) -> str:
        probs = lg[colour_ids].softmax(-1)
        best = int(probs.argmax())
        return f"{name:<12} -> {candidates[best].strip():<7} ({probs[best]:.2f})"

    print(prompt[11:-8])
    print(show("clean scene", logits[0]), "|", show("corrupted", logits[1]), "|", show(f"patched L{layer}", patched[0]))
    print(f"patched {int(mask.sum())} positions ({where!r} at layer {layer})")


clean_objects = [SceneObject("red", "circle", (0, 0)), SceneObject("blue", "square", (1, 1))]
corrupt_objects = [SceneObject("green", "circle", (0, 0)), SceneObject("blue", "square", (1, 1))]
for where, layer in [("object", 4), ("background", 4), ("object", 11), ("answer", 11)]:
    patch_demo(clean_objects, corrupt_objects, layer, where)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">What colour is the circle?
clean scene  -&gt; red     (1.00) | corrupted    -&gt; green   (1.00) | patched L4   -&gt; red     (1.00)
patched 4 positions ('object' at layer 4)
What colour is the circle?
clean scene  -&gt; red     (1.00) | corrupted    -&gt; green   (1.00) | patched L4   -&gt; green   (1.00)
patched 4 positions ('background' at layer 4)
What colour is the circle?
clean scene  -&gt; red     (1.00) | corrupted    -&gt; green   (1.00) | patched L11  -&gt; green   (1.00)
patched 4 positions ('object' at layer 11)
What colour is the circle?
clean scene  -&gt; red     (1.00) | corrupted    -&gt; green   (1.00) | patched L11  -&gt; red     (1.00)
patched 1 positions ('answer' at layer 11)</pre>

At layer 4, patching four object positions changes the corrupted green answer to red; four background positions leave it green. At layer 11 the same object patch leaves green at
1.00, while patching the one answer position gives red at 1.00. This concrete scene mirrors the aggregate transfer curve. The probabilities are restricted to colour candidates, and
these four demonstrations are examples rather than an additional evaluation sample.
</details>

Some experiments to try:

- Patch **half** of the object's tokens (pass an explicit list of `(row, col)` coordinates). Does the answer interpolate, or is it all-or-nothing?
- Take the corrupted scene to be one where the circle is in a *different quadrant* from the clean scene's. Patch the clean circle's tokens (at their original positions) into the corrupted scene. Is the answer determined by *which colour is where*, or by *which colour the tokens say*?
- Patch the **other object's** tokens, and ask about the circle: nothing should change, but check.
- Patch at layer 10: the answer position has already started to read the image. Can you find a set of positions that still works there?
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Summary of 4️⃣

- **Decodable** is the weakest kind of evidence. The plain logit lens does not cleanly find the colour at the object's tokens, the restricted lens finds it but also finds some in the background, and neither says anything about use.
- **Attended**: in the last layers, a few heads put most of the answer position's attention on the object's tokens, while the same-size random control gets almost none. In earlier layers the answer position attends to the question.
- **Sufficient**: patching the object's 4-5 tokens from the clean run recovers most of the answer at every layer up to the one where the answer position starts to read them, while patching the same number of background tokens, or other objects' tokens, has very small mean recovery. After that layer, the answer position's own residual stream carries the answer.
- **Necessary**: knocking out the answer position's attention to the object's tokens in the last layers hurts, knocking out the same number of random background tokens does not, and blocking answer-to-question attention in middle layers also hurts. A joint blockade of both routes is not tested here.
- **Attended is not the same as carrying the answer**: only about a quarter (0.23) of the object's tokens are among the tokens that the answer position attends to most, and patching those most-attended tokens recovers about 0.49 against the object's 0.94.
- Evidence of different strength can disagree. Background tokens are somewhat *decodable* but have much smaller average effects under these interventions. Treat each method as answering its own question.

Everything above is for a GPT-2 Small (124M parameters, plus the projector) trained for a few minutes on 5,000 synthetic scenes. Does any of it survive at real scale?
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
# 5️⃣ Is a Real VLM Doing the Same?
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Qwen3-VL-2B on the same scenes

Everything so far was a toy: a 124M-parameter language model, trained for four minutes on a synthetic task, with one question template per kind. The obvious worry is that we have
found a property of *this training run*. So let's repeat the two central experiments (the modality controls, and activation patching of image tokens) on a real VLM that nobody
trained for us on our scenes.

We use **Qwen3-VL-2B-Instruct** (Qwen team, 2025), which is built like our model, with real-world scale in every component:

| | our mini VLM | Qwen3-VL-2B-Instruct |
|---|---|---|
| vision encoder | SigLIP ViT-B/16 (frozen), 12 layers | its own ViT, 24 layers, width 1024, trained jointly with the model |
| merging | 2 × 2 patches → 1 token | 2 × 2 patches → 1 token ("patch merger") |
| image tokens for our scenes | 49 (7 × 7) | **64** (8 × 8: we upsample the 224px scene to 256px) |
| language model | GPT-2 Small, 12 layers, $d = 768$ | Qwen3-1.7B-class, **28 layers**, $d = 2048$ |
| positions | learned absolute | multimodal rotary (M-RoPE) with 2-D image positions |
| other | | **DeepStack**: features from ViT layers 5, 11 and 17 are *added* to the visual tokens' hidden states after LM layers 0, 1 and 2 |

We do not fine-tune Qwen3-VL on our scenes or question templates. We ask the pretrained model in its own chat format: `"What colour is the {shape}? Answer with one word."`, and
read the next-token distribution at the end of the prompt (the position after `<|im_start|>assistant\n`). Same scenes, same objects, same questions. We cannot audit its pretraining
data, so this is a transfer test without task-specific fine-tuning, not a claim about every scene's absence from pretraining.

If you have a small GPU: the model is about 4.4 GB in bfloat16, so the rest of this section needs about 6 GB of GPU memory. We free the GPT-2 VLM's memory first.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

# Free what we no longer need (rerun the earlier cells if you want these back), then load Qwen
for name in ["text_only_lm", "train_features", "a_cache", "a_patterns", "image_resid", "a_embeds", "clean_cache", "clean_patterns", "clean_resids", "clean_embeds", "corrupt_embeds"]:
    globals().pop(name, None)
t.cuda.empty_cache()
qwen, qwen_processor = utils.load_qwen(device)
qwen_tokenizer = qwen_processor.tokenizer
N_QWEN_LAYERS = qwen.config.text_config.num_hidden_layers
print(f"Qwen3-VL-2B-Instruct: {sum(p.numel() for p in qwen.parameters()) / 1e9:.2f}B parameters, {N_QWEN_LAYERS} language-model layers, d_model {qwen.config.text_config.hidden_size}")

example_scene = test_scenes[0]
example_obj = example_scene.objects[0]
example_question = f"What colour is the {example_obj.shape}? Answer with one word."
example_inputs = utils.qwen_inputs(qwen_processor, utils.render_scene(example_scene), example_question, device)
ids = example_inputs["input_ids"][0]
n_image = int((ids == qwen.config.image_token_id).sum())
print(f"prompt: {len(ids)} tokens, of which {n_image} image tokens (an 8 x 8 grid)")
print("first tokens:", [qwen_tokenizer.decode(i) for i in ids[:5]])
print("last tokens: ", [qwen_tokenizer.decode(i) for i in ids[-8:]])
with t.no_grad():
    generated = qwen.generate(**example_inputs, max_new_tokens=4, do_sample=False)
print(f"the question about the {example_obj.colour} {example_obj.shape} -> {qwen_tokenizer.decode(generated[0, len(ids):], skip_special_tokens=True)!r}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">Qwen3-VL-2B-Instruct: 2.13B parameters, 28 language-model layers, d_model 2048
prompt: 85 tokens, of which 64 image tokens (an 8 x 8 grid)
first tokens: ['&lt;|im_start|&gt;', 'user', '\n', '&lt;|vision_start|&gt;', '&lt;|image_pad|&gt;']
last tokens:  [' one', ' word', '.', '&lt;|im_end|&gt;', '\n', '&lt;|im_start|&gt;', 'assistant', '\n']
the question about the orange circle -&gt; ' orange'</pre>

</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - forced choice on a real model

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 5-10 minutes on this exercise.
> ```

We score Qwen the same way as our own model: forced choice among the six colour words. Implement `qwen_forced_choice`. It takes the Qwen model, the processed inputs for **one**
(image, question) pair (a Hugging Face `BatchEncoding`: `model(**inputs)` runs the model) and a tensor of candidate token ids, and returns the logits of the candidates at the
**last position** of the prompt, shape `(n_candidates,)`. (A Hugging Face causal language model returns `.logits` of shape `(batch, seq, vocab)`.)

In Qwen's tokenizer, the colour words that follow the assistant prompt have **no leading space**: the candidate ids are `qwen_tokenizer.encode("red")[0]`, and so on. Check that
each colour is a single token.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.no_grad()
def qwen_forced_choice(model, inputs, candidate_ids: Int[Tensor, " candidate"]) -> Float[Tensor, " candidate"]:
    """Logits of the candidate tokens at the last position of the prompt."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    logits = model(**inputs).logits[0, -1]
    return logits[candidate_ids]
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_qwen_forced_choice(qwen_forced_choice, qwen, qwen_processor)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`model(**inputs).logits[0, -1]` is the vocabulary logits at the last position of the single example; index it with `candidate_ids`.
</details>

Now the modality controls, on Qwen. For 100 test scenes with a random target object each, we ask "What colour is the {shape}?" with four different images: the **true scene**, a
**blank** image, a **noise** image, and a **shuffled** image (the scene of a different question, using your `derangement`). As in Part 3 we add one more: the true scene with the question taken away (an empty question; the six colour words are only used to score the answer). Each Qwen call runs a 2B-parameter model, so this takes
about a minute.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

qwen_colour_ids = t.tensor([qwen_tokenizer.encode(c)[0] for c in COLOURS], device=device)
assert all(len(qwen_tokenizer.encode(c)) == 1 for c in COLOURS), "every colour should be a single Qwen token"

N_QWEN = 100
qwen_rng = np.random.default_rng(11)
q_scenes = test_scenes[:N_QWEN]
q_targets = [int(qwen_rng.integers(len(s.objects))) for s in q_scenes]
q_images = utils.render_scenes(q_scenes)
q_conditions = {
    "true image": q_images,
    "blank image": np.stack([utils.render_scene(Scene([], background=235))] * N_QWEN),
    "noise image": utils.noise_images(N_QWEN, seed=1),
    "shuffled image-question pairs": q_images[derangement(N_QWEN, seed=0).numpy()],
}
qwen_accuracy = {}
qwen_control_traces = []
for name, images in tqdm(q_conditions.items()):
    correct = 0
    for i in range(N_QWEN):
        obj = q_scenes[i].objects[q_targets[i]]
        inputs = utils.qwen_inputs(qwen_processor, images[i], f"What colour is the {obj.shape}? Answer with one word.", device)
        scores = qwen_forced_choice(qwen, inputs, qwen_colour_ids).float().cpu()
        prediction = list(COLOURS)[int(scores.argmax())]
        correct += prediction == obj.colour
        qwen_control_traces.append(dict(condition=name, scene_index=i, kind="colour",
            object_count=len(q_scenes[i].objects), target_index=q_targets[i],
            model_prompt=f"What colour is the {obj.shape}? Answer with one word.",
            answer=obj.colour, prediction=prediction, correct=prediction == obj.colour,
            candidate_words=list(COLOURS), candidate_ids=qwen_colour_ids.cpu().tolist(),
            candidate_logits=scores.tolist(), target_blind_oracle=target_blind_oracle(q_scenes[i], "colour")))
    qwen_accuracy[name] = correct / N_QWEN
# Literal image-only input: true scene, empty question; same targets and six scoring candidates.
correct = 0
for i in range(N_QWEN):
    obj = q_scenes[i].objects[q_targets[i]]
    inputs = utils.qwen_inputs(qwen_processor, q_images[i], "", device)
    scores = qwen_forced_choice(qwen, inputs, qwen_colour_ids).float().cpu()
    prediction = list(COLOURS)[int(scores.argmax())]
    correct += prediction == obj.colour
    qwen_control_traces.append(dict(condition="image-only (no question)", scene_index=i, kind="colour",
        object_count=len(q_scenes[i].objects), target_index=q_targets[i], model_prompt="",
        answer=obj.colour, prediction=prediction, correct=prediction == obj.colour,
        candidate_words=list(COLOURS), candidate_ids=qwen_colour_ids.cpu().tolist(),
        candidate_logits=scores.tolist(), target_blind_oracle=target_blind_oracle(q_scenes[i], "colour")))
qwen_accuracy["image-only (no question)"] = correct / N_QWEN
print({k: f"{v:.0%}" for k, v in qwen_accuracy.items()})
utils.plot_bars(
    qwen_accuracy,
    "Qwen3-VL-2B: colour accuracy (forced choice) with the true image and with modality controls",
    colours=[utils.BLUE] + [utils.GREY] * 3 + [utils.ORANGE],
    hlines={"chance (1/6)": 1 / 6},
)
display(pd.DataFrame(qwen_control_traces).groupby(["condition", "object_count"])[["correct", "target_blind_oracle"]].mean().round(3))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/qwen_controls.png" width="650">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">{'true image': '100%', 'blank image': '13%', 'noise image': '12%', 'shuffled image-question pairs': '13%', 'image-only (no question)': '12%'}
                                            correct  target_blind_oracle
condition                     object_count                              
blank image                   1               0.143                1.000
                              2               0.158                0.500
                              3               0.098                0.333
image-only (no question)      1               0.143                1.000
                              2               0.132                0.500
                              3               0.098                0.333
noise image                   1               0.095                1.000
                              2               0.158                0.500
                              3               0.098                0.333
shuffled image-question pairs 1               0.143                1.000
                              2               0.132                0.500
                              3               0.122                0.333
true image                    1               1.000                1.000
                              2               1.000                0.500
                              3               1.000                0.333</pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

On these 100 synthetic colour questions, Qwen answers every true-image question correctly. Blank, noise and shuffled pairing give 13%, 12% and 13%; correct pairing supplies
information beyond the question text and its defaults. The true-image **empty-question** control scores 12/100. Its grouped scores are 3/21 for single-object scenes, 5/38 for
two-object scenes and 4/41 for three-object scenes. Even single-object accuracy is only 14.3%, despite having no target ambiguity, so ambiguity alone cannot explain the drop.

Removing the question also removes the colour-task instruction and changes the chat readout distribution. Six colour candidates are supplied only to the evaluator. This result does
not imply that Qwen cannot represent or recognize the image's colour. The near-chance score is descriptive, not a significance claim; the ambiguity oracle is an expectation, not an
empirical ceiling. This narrow forced-choice synthetic task is not a natural-image benchmark or evidence about every visual task.
</details>

## Patching image tokens in Qwen

Same experiment as in section 4️⃣, using 20 of its 100 counterfactual pairs (a clean scene, and the same scene with the queried object recoloured), the same question, and patching
from the clean run into the corrupted run, at layer $\ell$ and a set of positions. Qwen is a Hugging Face model without TransformerLens hooks, so we use PyTorch **forward
pre-hooks** on the decoder layers. A pre-hook on `model.model.language_model.layers[ℓ]` sees the layer's input arguments, which include `hidden_states`, the residual stream before
layer $\ell$: Qwen's `resid_pre`. We've written `utils.qwen_cache_layer_inputs(model, inputs)`, which runs the model once and returns `{layer: hidden_states}` for every layer. You
write the patch.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - patch a Hugging Face model with a pre-hook

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 15-20 minutes on this exercise.
> ```

Implement `qwen_patched_logits`. It takes the model, the processed inputs of the **corrupted** run, a layer, a tensor `positions` of sequence positions to patch (a boolean mask of
shape `(seq,)`, or an index tensor) and `source_hidden`, the clean run's residual stream before that layer, shape `(1, seq, d_model)`. Register a forward pre-hook
(`register_forward_pre_hook(hook, with_kwargs=True)`) on `model.model.language_model.layers[layer]` that replaces the layer's `hidden_states` at the given positions with
`source_hidden` at those positions, run the model, and return the last-position logits `(vocab,)`. **Always remove the hook** afterwards (use `try`/`finally`).

In a pre-hook with `with_kwargs=True`, the signature is `hook(module, args, kwargs)` and you return `(args, kwargs)` to modify the call. Depending on the version, `hidden_states`
arrives either as a keyword argument or as `args[0]`; the helper in `utils.qwen_cache_layer_inputs` shows how to handle both. As in the GPT-2 case, **clone before writing**.

The test checks two oracles: patching *every* position with the model's own cache reproduces its logits exactly, and, for layers $\geq 3$, patching every position with *another*
input's cache reproduces that other input's logits. (Why only layers $\geq 3$? It's DeepStack; see below.)
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

@t.no_grad()
def qwen_patched_logits(
    model, inputs, layer: int, positions: Tensor, source_hidden: Float[Tensor, "1 seq d_model"]
) -> Float[Tensor, " vocab"]:
    """Last-position logits of the run on `inputs`, with the residual stream before `layer` overwritten from `source_hidden` at `positions`."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    def patch_hook(module, args, kwargs):
        hidden = (kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]).clone()
        hidden[:, positions] = source_hidden[:, positions]
        if "hidden_states" in kwargs:
            kwargs["hidden_states"] = hidden
            return args, kwargs
        return (hidden,) + tuple(args[1:]), kwargs

    handle = model.model.language_model.layers[layer].register_forward_pre_hook(patch_hook, with_kwargs=True)
    try:
        return model(**inputs).logits[0, -1]
    finally:
        handle.remove()
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_qwen_patched_logits(qwen_patched_logits, qwen, qwen_processor)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

Inside the hook: get `hidden = kwargs["hidden_states"]` (or `args[0]` if the keyword is absent), clone it, assign `hidden[:, positions] = source_hidden[:, positions]`, put it back
into `kwargs` (or rebuild `args`) and return `(args, kwargs)`. Register with `layer_module.register_forward_pre_hook(patch_hook, with_kwargs=True)`; the returned handle has a
`.remove()` method.
</details>

<details>
<summary>Aside - DeepStack, and why layers 0-2 are different</summary>

In most VLMs the image enters the language model once, at the input. Qwen3-VL adds a twist called **DeepStack**: visual features taken from intermediate layers of the ViT (layers
5, 11 and 17) are *added* to the hidden states at the image positions after language-model layers 0, 1 and 2. This gives the early layers of the language model access to the vision
encoder's low- and mid-level features, not only its last layer.

For our purposes this has one consequence. If we patch the clean residual stream at the image positions into layer 0, layers 0, 1 and 2 will then add the *corrupted image's*
DeepStack features on top, because the DeepStack features come from the vision encoder, not from the residual stream. So a patch at layers 0-2 is incomplete, and we expect lower
recovery there than at layer 3. From layer 3 on, the residual stream is all that matters, which is what the second oracle in the test checks.
</details>
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
### Exercise - which layers to patch

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵⚪⚪⚪
>
> You should spend up to 5 minutes on this exercise.
> ```

A 28-layer model has 28 places to patch, and for each place we run the model once per token set and per pair. To keep the sweep affordable we patch **every third layer, and always the last
one**: layers 0, 3, 6, ..., 27 for Qwen's 28 layers. Implement `qwen_layer_schedule(num_layers)`, which returns that list. Each layer must appear **exactly once**: when the last layer is
itself a multiple of three (as 27 is) a naive "every third layer, then add the last" lists it twice, and that layer's runs would be counted double. A model with no layers should raise
a `ValueError`.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

def qwen_layer_schedule(num_layers: int) -> list[int]:
    """Every third layer plus the final layer, each intervention exactly once."""
    # EXERCISE
    # raise NotImplementedError()
    # END EXERCISE
    # SOLUTION
    if num_layers < 1:
        raise ValueError("A model must contain at least one layer")
    return list(dict.fromkeys([*range(0, num_layers, 3), num_layers - 1]))
    # END SOLUTION


# HIDE
if MAIN:
    tests.test_qwen_layer_schedule(qwen_layer_schedule)
# END HIDE

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Hint</summary>

`range(0, num_layers, 3)` gives every third layer starting at 0. Add `num_layers - 1`, and remove repeats while keeping the order: `list(dict.fromkeys(layers))` does that.
</details>

Now the sweep. For 20 of the 100 pairs, at every third layer, we patch six sets of positions: the object's tokens, the same-size random background tokens (a control), the other
objects' tokens, the **highest-attention tokens** (the $k$ image tokens that the last position attends to most, averaged over all layers and heads, with $k$ the number of the object's tokens;
this needs Qwen's attention patterns, which `utils.qwen_attention_patterns` returns), all image tokens, and the last position of the prompt, where Qwen's answer is read. The image tokens
are the 64 positions where `input_ids == image_token_id` (in raster order of the $8 \times 8$ grid), so the same token sets can be built for Qwen's grid. We keep only the pairs where Qwen
answers both the clean and the corrupted scene correctly, and we compute every pair's recovery with your `logit_diff_recovery`, which stops if any kept pair has no contrast. This is about
1,200 forward passes of a 2B model, so allow a minute or two.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

qwen_layers = qwen_layer_schedule(N_QWEN_LAYERS)
qwen_set_names = ["object tokens", "same-size random background", "other objects", "highest-attention tokens", "all image tokens", "answer position"]
qwen_patched = {name: [[] for _ in qwen_layers] for name in qwen_set_names}  # name -> layer -> one (6,) vector of colour logits per kept pair
qwen_clean_logits, qwen_corrupt_logits, qwen_clean_idx, qwen_corrupt_idx = [], [], [], []
for k in tqdm(range(20)):
    clean_scene, corrupt_scene, idx = pairs[k]
    obj = clean_scene.objects[idx]
    question = f"What colour is the {obj.shape}? Answer with one word."
    inputs_clean = utils.qwen_inputs(qwen_processor, utils.render_scene(clean_scene), question, device)
    inputs_corrupt = utils.qwen_inputs(qwen_processor, utils.render_scene(corrupt_scene), question, device)
    clean_ci, corrupt_ci = list(COLOURS).index(obj.colour), list(COLOURS).index(corrupt_scene.objects[idx].colour)
    lc, lk = qwen_forced_choice(qwen, inputs_clean, qwen_colour_ids).float(), qwen_forced_choice(qwen, inputs_corrupt, qwen_colour_ids).float()
    if int(lc.argmax()) != clean_ci or int(lk.argmax()) != corrupt_ci:
        continue  # keep only pairs the model gets right on both scenes
    qwen_clean_logits.append(lc)
    qwen_corrupt_logits.append(lk)
    qwen_clean_idx.append(clean_ci)
    qwen_corrupt_idx.append(corrupt_ci)
    cache = utils.qwen_cache_layer_inputs(qwen, inputs_clean)
    seq = inputs_clean["input_ids"].shape[1]
    image_positions = (inputs_clean["input_ids"][0] == qwen.config.image_token_id).nonzero().flatten()
    image_slice = slice(int(image_positions[0]), int(image_positions[-1]) + 1)
    obj_mask = t.tensor(utils.qwen_object_token_mask(obj).flatten(), device=device)[None]
    occupied = t.tensor(np.any([utils.qwen_object_token_mask(o).flatten() for o in clean_scene.objects], axis=0), device=device)[None]
    token_sets = {
        "object tokens": obj_mask[0],
        "same-size random background": random_background_mask(obj_mask, occupied, seed=k)[0],
        "other objects": (occupied & ~obj_mask)[0],
        "highest-attention tokens": top_attention_mask(utils.qwen_attention_patterns(qwen, inputs_clean), seq - 1, image_slice, int(obj_mask.sum())),
        "all image tokens": t.ones_like(obj_mask[0]),
    }
    for name in qwen_set_names:
        positions = t.zeros(seq, dtype=t.bool, device=device)
        if name == "answer position":
            positions[-1] = True
        else:
            positions[image_positions[token_sets[name]]] = True
        for li, layer in enumerate(qwen_layers):
            qwen_patched[name][li].append(qwen_patched_logits(qwen, inputs_corrupt, layer, positions, cache[layer])[qwen_colour_ids].float())
n_valid = len(qwen_clean_idx)
if n_valid == 0:
    raise ValueError("No correctly answered counterfactual pairs, so recovery is undefined")
# One recovery for every kept pair, from the same function as the mini VLM: shape (layer, pair).
qwen_recovery = {
    name: t.stack([
        logit_diff_recovery(t.stack(rows), t.stack(qwen_clean_logits).cpu(), t.stack(qwen_corrupt_logits).cpu(), t.tensor(qwen_clean_idx), t.tensor(qwen_corrupt_idx))
        for rows in per_layer
    ])
    for name, per_layer in {name: [[row.cpu() for row in rows] for rows in per_layer] for name, per_layer in qwen_patched.items()}.items()
}
qwen_recovery_mean = {name: values.mean(dim=1).tolist() for name, values in qwen_recovery.items()}
print(f"{n_valid} of 20 pairs kept (both scenes answered correctly)")
print(pd.DataFrame(qwen_recovery_mean, index=qwen_layers).T.round(2).to_string())

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">20 of 20 pairs kept (both scenes answered correctly)
                               0     3     6     9     12    15    18    21    24    27
object tokens                0.62  0.73  0.72  0.72  0.69  0.68  0.63  0.55  0.18  0.00
same-size random background  0.06  0.06  0.06  0.06  0.06  0.06  0.05  0.05  0.02  0.00
other objects                0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00 -0.00  0.00
highest-attention tokens     0.69  0.76  0.76  0.77  0.78  0.77  0.71  0.62  0.21  0.00
all image tokens             0.88  1.00  1.00  1.00  1.00  1.00  0.96  0.87  0.29  0.00
answer position              0.00  0.00  0.00  0.00  0.00  0.00  0.03  0.13  0.69  0.99</pre>

</details>

All 20 candidate pairs are kept, since both the clean and the corrupted scene are answered correctly. In Qwen, patching the object's tokens recovers 0.62 of the contrast at layer 0 and 0.68 to 0.73 from layer 3 to layer 15, then
falls: 0.55 at layer 21, 0.18 at layer 24 and 0.00 at layer 27. Matched background recovers 0.05 to 0.06 early, small but not zero, and the other objects 0.00. Patching all image tokens gives 0.88 at layer 0 and 1.00 from layer 3 to layer 15, and the last
position takes over late (0.13 at layer 21, 0.69 at layer 24, 0.99 at layer 27). The most-attended tokens recover 0.69 to 0.78 through layer 15, *more* than the object's tokens, unlike the mini VLM where they recovered about half. The shortfall of the
all-image patch at layer 0 is where Qwen's DeepStack features are added back from the corrupted image; the lower object-only values are probably partly the same effect, but that is a hypothesis we have not tested.

## The signature figure

We now have the same experiment on two very different models. The two have different numbers of layers (12 and 28), so we plot the recovery against **relative depth**, `layer / (n_layers - 1)`:
the first layer is at 0 and the last at 1. The sets and the colours are the same for both models, and the y-axis too.

One more thing before we draw it. Each curve is an average over 100 pairs for the mini VLM and only 20 for Qwen. Would a different set of pairs have given a different curve? We can ask the
data itself, with a **pair bootstrap**: resample the pairs with replacement (as many as you started with), recompute the average, repeat a thousand times, and shade the middle 95% of those
averages. A toy example: three pairs recover 0.9, 0.8 and 0.1. Their mean is 0.6. Drawing three pairs with replacement can give a mean as low as 0.1 (the 0.1 pair three times, one chance in 27) or as
high as 0.9, so with so few pairs the band is wide; with a hundred it is narrow. Resampling whole pairs keeps one pair's numbers at all layers together. The trap: the band only knows about the
pairs we drew. It does not cover scenes we never sampled, other questions, or another training seed. Go deeper: 0.7 Foundations Toolkit, Part 5.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

signature_sets = ["object tokens", "answer position", "other objects", "same-size random background", "highest-attention tokens"]
mini_recovery = {name: patch_recovery["answer position (59)" if name == "answer position" else name] for name in signature_sets}
signature_models = {
    f"mini VLM (GPT-2 Small, {N_LAYERS} layers, {len(pairs)} pairs)": (mini_recovery, list(range(N_LAYERS)), N_LAYERS),
    f"Qwen3-VL-2B ({N_QWEN_LAYERS} layers, {n_valid} pairs)": (qwen_recovery, qwen_layers, N_QWEN_LAYERS),
}
utils.plot_signature_recovery(
    signature_models,
    signature_sets,
    controls=("other objects", "same-size random background", "highest-attention tokens"),
    title="Where is the answer? Recovery from patching each token set, mini VLM and Qwen3-VL (bands: 95% pair bootstrap)",
)
for model_name, (recovery, layers, n_layers) in signature_models.items():
    print(model_name)
    for name in ["object tokens", "highest-attention tokens", "answer position"]:
        mean, low, high = utils.pair_bootstrap_band(recovery[name].T)
        cells = [f"L{layer}: {mean[i]:.2f} [{low[i]:.2f}, {high[i]:.2f}]" for i, layer in enumerate(layers) if layer in (0, layers[len(layers) // 2], layers[-2], layers[-1])]
        print(f"  {name:<26}" + "   ".join(cells))

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/signature_recovery.png" width="900">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Expected output</summary>

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">mini VLM (GPT-2 Small, 12 layers, 100 pairs)
  object tokens             L0: 0.94 [0.93, 0.96]   L6: 0.94 [0.93, 0.96]   L10: 0.61 [0.59, 0.62]   L11: 0.19 [0.18, 0.21]
  highest-attention tokens  L0: 0.49 [0.43, 0.54]   L6: 0.49 [0.43, 0.55]   L10: 0.34 [0.29, 0.38]   L11: 0.14 [0.12, 0.16]
  answer position           L0: 0.00 [0.00, 0.00]   L6: 0.00 [-0.00, 0.00]   L10: 0.35 [0.34, 0.36]   L11: 0.78 [0.77, 0.79]
Qwen3-VL-2B (28 layers, 20 pairs)
  object tokens             L0: 0.62 [0.54, 0.70]   L15: 0.68 [0.59, 0.77]   L24: 0.18 [0.14, 0.21]   L27: 0.00 [-0.00, 0.00]
  highest-attention tokens  L0: 0.69 [0.64, 0.74]   L15: 0.77 [0.71, 0.82]   L24: 0.21 [0.18, 0.24]   L27: 0.00 [-0.00, 0.00]
  answer position           L0: 0.00 [0.00, 0.00]   L15: 0.00 [0.00, 0.00]   L24: 0.69 [0.67, 0.71]   L27: 0.99 [0.99, 0.99]</pre>

</details>

<details>
<summary>Help - how should I interpret this?</summary>

Solid lines are the mini VLM and dashed lines are Qwen3-VL; bands show how much each mean moves when the pairs are resampled. **Object tokens (blue):** the mini VLM recovers 0.94 [0.93, 0.96] at layer 0 and layer 6,
then falls to 0.61 [0.59, 0.62] at layer 10 and 0.19 [0.18, 0.21] at layer 11. Qwen recovers 0.62 [0.54, 0.70] at layer 0 and 0.68 [0.59, 0.77] at layer 15, then falls to 0.18 [0.14, 0.21] at layer 24 and 0.00 at layer 27.
**Answer position (red):** about 0 through the middle layers in both models, then up at the end: 0.35 [0.34, 0.36] at layer 10 and 0.78 [0.77, 0.79] at layer 11 for the mini VLM, 0.69 [0.67, 0.71] at layer 24 and 0.99 at layer 27
for Qwen. The hand-over happens in the last quarter of the depth in both. **Controls:** other objects (green) stay at 0.00 and the same-size random background (grey) at 0.01 in the mini VLM and about 0.06 in Qwen, near zero compared with the object's tokens. **Highest-attention
tokens (orange):** about 0.49 [0.43, 0.54] in the mini VLM, roughly half the object tokens; in Qwen 0.77 [0.71, 0.82] at layer 15, above the object's tokens in the mean, though the two bands overlap.

What the figure supports: both models carry the answer in a small set of image tokens until late and then move it to the last position, with controls near zero. What it doesn't: the Qwen bands are wide (20 pairs), the bands cover the choice of pairs and not the
choice of scenes or a training seed, and the same shape is not the same circuit. We don't know why Qwen's most-attended tokens do so well; it is a question for the Bonus.
</details>

<details>
<summary>Question - what would you need to claim that the two models implement "the same algorithm"? What do these curves show, and what do they not show?</summary>

The curves show a shared coarse pattern on this synthetic colour task: early object-localized information can restore the clean answer, and a late last-position state carries the
answer. They do not identify equal heads, computations or routes, and the patching locations have different spatial resolutions and relative depths. Claiming the same algorithm
would need matched tasks, broader independent examples, finer component/path interventions and tests of competing mechanisms. Treat the resemblance as a
bounded replication of a causal pattern, not identity of algorithms.
</details>

## Summary of 5️⃣

- A pretrained 2B-parameter VLM, without our task-specific fine-tuning, answers these colour questions correctly. Its accuracy is near chance on blank, noise and shuffled images: correct image–question pairing matters.
- Patching the clean residual stream into the corrupted run, **the object's image tokens recover much of the answer contrast** through the middle of the network, while the same number of random background tokens and the other objects' tokens recover little. As in the mini VLM, the answer then moves to the last position in the later layers.
- The two curves have the same shape on a relative-depth axis, but the details differ (where the hand-over happens and how complete the object-only patch is), and the architecture adds complications (DeepStack in the first three layers).
  That is a reasonable basis for saying that the qualitative picture of "object tokens in the middle, last position at the end" is not an accident of our training run, and no basis for saying the circuits are alike.

That's the full loop: build a VLM from parts, check with controls that it uses the image, find which tokens and which layers matter, and check the findings on a real model. Before the quiz, a word about
what it means to be right here: everything above is one training seed on synthetic shapes, with a 20-pair sweep on the real model. The Bonus lists what this lesson doesn't show.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

del qwen
t.cuda.empty_cache()

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
You've built the mini VLM, trained it two ways, and taken it apart with lenses, attention, patching and a real model for comparison. We then ran four more experiments on it that we haven't
shown you. Before you see them, predict them.

This is how Stanford's CS 312 *[Deep Learning Alchemy](https://deep-learning-alchemy.github.io/)* tests understanding: if you understand a system, you can say what it will do in a setting you
haven't tried. A wrong answer here isn't a failure. It's the most useful thing that can happen in this lesson, because it shows you exactly which part of your picture is off.

**How it works.**

1. Each question shows its change as a diff against a call you ran in this lesson.
2. Write your answers, and one sentence of reasoning for each, in the answer cell. The reasoning should name a result from this lesson. Don't run any new experiments first.
3. Run `quiz.score(my_answers)`, then open the **Reveal** dropdowns. Each has the real numbers and what a good mental model would have said.

**Difficulty** says how far a question takes you from what you ran:

- **Easy** means something you built, used on a case it was designed for.
- **Medium** means one setting turned beyond the range you ran.
- **Hard** means a regime you never saw, or a number rather than a direction.

**Scoring.** An interval question gives full points for the right interval and half for a neighbouring one. A ranking gives one point for every pair of runs you put in the right order. Write `=`
when you think two runs land within 0.02 of each other. There are 18 points in total. The score is for you: it isn't saved or sent anywhere. Expect to get around half.

> ```yaml
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 20-25 minutes on this quiz.
> ```

All four questions are about the trained model you made in Part 3, regime (B), unless a question says otherwise.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Q1. Twenty new pairs &nbsp;·&nbsp; Easy &nbsp;·&nbsp; 4 points

**From the lesson.** In Part 4 you patched the object's tokens from the clean run into the corrupted run, at every layer, on 100 recolouring pairs, and read off the recovery of the
clean-versus-corrupted logit difference.

**This question changes.** The pairs. We draw 20 new ones from a different seed (so different scenes, objects and colours), and patch the object's tokens at layer 4 exactly as you did.

```diff
- pairs = utils.make_counterfactual_pairs(100, seed=7)
+ pairs = utils.make_counterfactual_pairs(20, seed=123)
```

The mean recovery over the 20 pairs, from patching the object's tokens at layer 4, is in:

| a | b | c | d | e |
|---|---|---|---|---|
| below 0.5 | 0.5 to 0.8 | 0.8 to 0.9 | 0.9 to 1.0 | above 1.0 |

## Q2. A fourth object &nbsp;·&nbsp; Medium &nbsp;·&nbsp; 5 points

**From the lesson.** Every scene you trained on had one, two or three objects. In Part 3 the model answered `count` questions correctly 99.9% of the time, with the answer one of `one`, `two` or `three`.

**This question changes.** The scenes. We draw 500 scenes with **four** objects (one in each quadrant) and ask the same count question. The model is scored by forced choice among four words now,
because the right answer is `four`.

```diff
- scene = utils.sample_scene(rng)                      # 1, 2 or 3 objects
+ scene = utils.sample_scene(rng, n_objects=4)         # four objects
- candidates = [" one", " two", " three"]
+ candidates = [" one", " two", " three", " four"]
```

The count accuracy on the 500 four-object scenes is in:

| a | b | c | d | e |
|---|---|---|---|---|
| below 2% | 2% to 20% | 20% to 50% | 50% to 80% | above 80% |

## Q3. More steps, same projector &nbsp;·&nbsp; Medium &nbsp;·&nbsp; 3 points

**From the lesson.** In Part 3 the projector-only run (regime A, 600 steps) reached a colour accuracy of 0.594, and the run that also trained GPT-2 (regime B, 1,500 steps) reached 1.000. The two differ in
steps *and* in what was trained.

**This question changes.** We add a third run: the projector only, with GPT-2 frozen, but trained for the same 1,500 steps as regime B.

```diff
- losses_a = train_vlm(lm_a, projector_a, train_features, train_scenes, n_steps=600, train_lm=False)
+ losses_l = train_vlm(lm_l, projector_l, train_features, train_scenes, n_steps=1500, train_lm=False)
```

**Rank the colour accuracy** of **S** (projector only, 600 steps: Part 3's regime A), **L** (projector only, 1,500 steps: the new run) and **J** (projector and GPT-2, 1,500 steps: Part 3's regime B), lowest
first, for example `S < L = J`. Use `=` for runs within 0.02 of each other.

## Q4. Shape instead of colour &nbsp;·&nbsp; Hard &nbsp;·&nbsp; 6 points

**From the lesson.** In Part 4 the object's tokens recovered most of the answer through layers 0 to 8, and then the recovery fell as the answer moved to the last position. Recovery stayed well
above 0.5 until the last two layers.

**This question changes.** The question. We ask about **shape** (`What shape is the red object?`) instead of colour. We draw 40 new pairs of scenes with one or two objects that differ in exactly one
thing, the **shape** of the queried object (a circle becomes a square, say), and patch the union of the object's tokens in the two scenes, layer by layer, as in Part 4.

```diff
- pairs = utils.make_counterfactual_pairs(100, seed=7)              # the queried object changes colour
+ pairs = utils.make_shape_pairs(40, seed=11)                       # the queried object changes shape
- prompt = question_and_answer(scene, "colour", target)[0]
+ prompt = question_and_answer(scene, "shape", target)[0]
```

At which layer does the mean recovery from patching the object's tokens first fall **below 0.5**?

| a | b | c | d | e |
|---|---|---|---|---|
| layer 0 (it is already below 0.5 there) | layer 1 | layers 2 to 5 | layers 6 to 10 | layer 11, or it never falls below 0.5 |

## Your answers

Fill in the cell below, every answer and one sentence of reasoning for each, before you open any Reveal. Then run it: `quiz.score` prints your score.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: []

my_answers = {
    "Q1": "",  # "a" to "e"
    "Q2": "",  # "a" to "e"
    "Q3": "",  # e.g. "S < L = J"
    "Q4": "",  # "a" to "e"
}
my_reasons = {  # one sentence each: which result from this lesson are you relying on?
    "Q1": "",
    "Q2": "",
    "Q3": "",
    "Q4": "",
}

sys.path.append(str(root_dir / "infrastructure" / "extension" / "learner"))
from quiz import Interval, Quiz, Ranking

quiz = Quiz(
    {
        "Q1": Interval("d", points=4),
        "Q2": Interval("a", points=5),
        "Q3": Ranking("S < L = J"),
        "Q4": Interval("b", points=6),
    }
)
quiz.score(my_answers)

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Reveal

<details>
<summary>Reveal - Q1: twenty new pairs</summary>

On the 20 new pairs (seed 123), patching the object's tokens at layer 4 recovers **0.963** of the clean-versus-corrupted logit difference (95% pair-bootstrap band 0.947 to 0.977). Across layers 0 to 8 the mean stays between 0.954 and 0.964. In Part 4, the 100 pairs gave 0.94 at layer 4. **Answer: d.**

**The mental model.** The Part 4 curve is not a property of those 100 pairs: it is flat at about 0.94 through layer 9 because the answer's colour contrast sits in the object's tokens until the answer position starts to read it, and a new
sample of scenes doesn't change that. The band is narrow because this contrast is large and consistent from pair to pair. If you chose b or c, you were treating 20 pairs as too few to trust a result from 100.
If you chose e, you expected more than full recovery; that can happen when a patch overshoots, but nothing in Part 4 suggested it would.
</details>

<details>
<summary>Reveal - Q2: a fourth object</summary>

On 500 scenes with four objects the model says `three` for 487 of them (97.4%) and `two` for the other 13. It never says `one` or `four`. The count accuracy is **0.000** (0 of 500). **Answer: a.**

**The mental model.** The model learned to map images to one of three words, because those were the only count answers it was ever trained on. Part 3's 99.9% on count was a score on scenes with one to three objects and the words `one`, `two`
and `three`. With four objects it behaves like a model that counts up to three and then stops: `three` is the largest count it knows, and `four` is a word it has never had to produce. If you chose c, d or e, you expected counting to extend itself the way it does for people.
It doesn't: a trained answer vocabulary is a fixed set of outputs.
</details>

<details>
<summary>Reveal - Q3: more steps, same projector</summary>

| run | what was trained | steps | colour accuracy |
| --- | --- | --- | --- |
| S | projector only | 600 | 0.594 |
| L | projector only | 1,500 | 0.992 |
| J | projector and GPT-2 | 1,500 | 1.000 |

The new run's final training loss is 0.042 (the joint run's was 0.0315). **Answer: `S < L = J`.**

**The mental model.** In Part 3 regime A and regime B differed in two ways at once, steps and what was trained, so the gap from 0.594 to 1.000 could not be assigned to either. This run separates them: the projector alone, given the same 1,500 steps, reaches 0.992 on colour,
within 0.008 of the joint run, so most of the gap was a shorter schedule, not a frozen language model. If you chose `S = L < J`, you gave the credit to unfreezing GPT-2, which is what the table in Part 3 invites you to believe. The comparison was only a recipe comparison, as its Help dropdown said.
(Colour is the only kind scored here; position questions are the ones where unfreezing may still matter.)
</details>

<details>
<summary>Reveal - Q4: shape instead of colour</summary>

On 40 shape pairs, patching the object's tokens recovers 0.80 at layer 0, and then only **0.15** at layer 1; it stays at 0.13 to 0.15 through layer 9, and is 0.10 at layer 10 and 0.08 at layer 11. The mean first falls below 0.5 at **layer 1**. For reference, patching *all* image tokens recovers
1.00 at layer 0, 0.35 at layers 1 to 5, 0.38 at layers 6 to 9, 0.24 at layer 10 and 0.15 at layer 11. **Answer: b.**

**The mental model.** For colour, the object's tokens carry the answer until layer 9, so you may have expected the same for shape. They don't: by the input to layer 1, most of the shape contrast has already left the image tokens, and even a patch of the whole image recovers only 0.35.
So block 0 has moved most of the shape information somewhere else (we haven't measured where; the knockouts in Part 4 were run for colour questions only, and a natural guess to test is that the question tokens take it). The colour result from Part 4 does not transfer to shape: it describes what this model does for colour questions,
not for every question. If you chose d or e, you assumed the object's tokens keep the answer until late for every attribute. If you chose a, you expected the object to be insufficient already at layer 0, but its tokens still recover 0.80 there.
</details>

## What this quiz tested

- **Results carry over to new samples of the same thing, not to new things** (Q1, Q2, Q4). Twenty new colour pairs gave the same curve as the hundred in Part 4, but a fourth object and a shape question both fell outside it.
- **A model can't emit what it never had to say** (Q2). The count head is a choice among three trained words.
- **Compare like with like** (Q3). Part 3's two regimes differed in steps and in what was trained; the extra run showed that most of the gap was steps.
- **A picture of "where the answer lives" belongs to a question** (Q4). Shape information left the object's tokens after one block, where colour information stayed through layer 9.

## Try it yourself - reproduce one result

Q2 is the cheapest to reproduce: a hundred four-object scenes take a few seconds. Run it with your own functions. Then look at which wrong word the model picks, and ask what in Part 3 would have told you.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

rng4 = np.random.default_rng(1234)
scenes4 = [utils.sample_scene(rng4, n_objects=4) for _ in range(100)]
tokens4, lens4 = utils.tokenize_prompts(vlm, [" Question: How many shapes are there? Answer:"] * len(scenes4))
with t.no_grad():
    logits4 = vlm_answer_logits(vlm, projector, featurise(scenes4), tokens4, lens4)
count_ids = t.tensor([vlm.to_single_token(w) for w in [" one", " two", " three", " four"]], device=device)
says = Counter([" one", " two", " three", " four"][i] for i in logits4[:, count_ids].argmax(-1).tolist())
print(f"{len(scenes4)} scenes with four objects: the model says {dict(says)}; accuracy {answer_accuracy(logits4, count_ids[3].repeat(len(scenes4)), count_ids):.0%}")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## The runs behind this quiz

Every number in the reveals above comes from the next cell, which runs all four experiments: 20 new colour pairs, 500 four-object scenes, one more projector-only training run of 1,500
steps (the slow one, about two minutes), and 40 shape pairs. You don't need to run it. It's here so that you can check any answer, and so that the quiz can never drift from the code: the
cell ends by checking that the answers written in the key above are still the ones these runs produce.
'''

# ! CELL TYPE: code
# ! FILTERS: []
# ! TAGS: [main]

def letter(value: float, edges: list[float]) -> str:
    """Which of five intervals a-e the value falls in, for four ascending edges."""
    return "abcde"[sum(value >= edge for edge in edges)]


def object_recovery_curve(pair_list: list, kind: str) -> dict[str, Tensor]:
    """Recovery of every pair at every layer, shape (layer, pair), from patching the object's tokens (the union of its tokens in the clean and corrupted scenes) and, for reference, all image tokens."""
    n = len(pair_list)
    clean_s, corrupt_s, targets = [p[0] for p in pair_list], [p[1] for p in pair_list], [p[2] for p in pair_list]
    word = (lambda s, i: s.objects[i].colour) if kind == "colour" else (lambda s, i: s.objects[i].shape)
    prompts = [question_and_answer(s, kind, i)[0] for s, i in zip(clean_s, targets)]
    tokens, lens = utils.tokenize_prompts(vlm, prompts)
    assert (lens == lens[0]).all(), "all prompts should have the same length"
    seq_q = 1 + N_IMAGE_TOKENS + int(lens[0])  # BOS, image tokens, question: the answer position is the last one
    ids_clean = t.tensor([vlm.to_single_token(" " + word(s, i)) for s, i in zip(clean_s, targets)], device=device)
    ids_corrupt = t.tensor([vlm.to_single_token(" " + word(s, i)) for s, i in zip(corrupt_s, targets)], device=device)
    with t.no_grad():
        embeds_clean = build_input_embeddings(vlm, projector(featurise(clean_s)), tokens)
        embeds_corrupt = build_input_embeddings(vlm, projector(featurise(corrupt_s)), tokens)
        _, cache_q = vlm.run_with_cache(embeds_clean, start_at_layer=0, stop_at_layer=N_LAYERS, names_filter=lambda name: name.endswith("resid_pre"))
        resids = {layer: cache_q["resid_pre", layer] for layer in range(N_LAYERS)}
        final = lambda x: vlm.unembed(vlm.ln_final(vlm(x, start_at_layer=0, stop_at_layer=N_LAYERS)[:, seq_q - 1]))
        logits_clean, logits_corrupt = final(embeds_clean), final(embeds_corrupt)
    masks_q = t.tensor(np.stack([(utils.object_token_mask(c.objects[i]) | utils.object_token_mask(k.objects[i])).flatten() for c, k, i in zip(clean_s, corrupt_s, targets)]), device=device)
    mask = t.zeros(n, seq_q, dtype=t.bool, device=device)
    mask[:, IMAGE_SLICE] = masks_q
    everything = t.zeros_like(mask)
    everything[:, IMAGE_SLICE] = True
    return patching_sweep(vlm, embeds_corrupt, resids, {"object tokens": mask, "all image tokens": everything}, logits_clean, logits_corrupt, ids_clean, ids_corrupt, seq_q - 1, list(range(N_LAYERS)))


# Q1: 20 new colour pairs
q1_recovery = object_recovery_curve(utils.make_counterfactual_pairs(20, seed=123), "colour")["object tokens"]
q1_mean, q1_low, q1_high = utils.pair_bootstrap_band(q1_recovery.T)
print(f"Q1: object-token recovery at layer 4 on 20 new colour pairs: {q1_mean[4]:.3f} (95% band {q1_low[4]:.3f} to {q1_high[4]:.3f}); layers 0-8 range {q1_mean[:9].min():.3f} to {q1_mean[:9].max():.3f}")

# Q2: four objects
rng4 = np.random.default_rng(777)
scenes4 = [utils.sample_scene(rng4, n_objects=4) for _ in range(500)]
tokens4, lens4 = utils.tokenize_prompts(vlm, [" Question: How many shapes are there? Answer:"] * len(scenes4))
with t.no_grad():
    logits4 = vlm_answer_logits(vlm, projector, featurise(scenes4), tokens4, lens4)
q2_accuracy = answer_accuracy(logits4, count_ids[3].repeat(len(scenes4)), count_ids)
q2_says = Counter([" one", " two", " three", " four"][i] for i in logits4[:, count_ids].argmax(-1).tolist())
print(f"Q2: 500 four-object scenes: the model says {dict(q2_says)}; count accuracy {q2_accuracy:.3f}")

# Q3: projector only, 1,500 steps (the training features were freed before Qwen, so featurise them again)
train_features = featurise(train_scenes)
t.manual_seed(0)
projector_l = Projector().to(device)
lm_l = copy.deepcopy(gpt2)
losses_l = train_vlm(lm_l, projector_l, train_features, train_scenes, n_steps=1500, train_lm=False)
q3_runs = {
    "S": results_a["colour"],
    "L": evaluate_vlm(lm_l, projector_l, test_features, test_questions)["colour"],
    "J": results_b["colour"],
}
del lm_l
t.cuda.empty_cache()
print("Q3: colour accuracy " + ", ".join(f"{name} {value:.3f}" for name, value in q3_runs.items()) + f"; final loss of L {np.mean(losses_l[-50:]):.3f}")

# Q4: shape pairs
q4_curves = object_recovery_curve(utils.make_shape_pairs(40, seed=11), "shape")
q4_mean, q4_low, q4_high = utils.pair_bootstrap_band(q4_curves["object tokens"].T)
q4_all_mean = q4_curves["all image tokens"].mean(dim=1)
below = [layer for layer in range(N_LAYERS) if q4_mean[layer] < 0.5]
q4_first = below[0] if below else None
print("Q4: object-token recovery by layer on 40 shape pairs: " + " ".join(f"{v:.2f}" for v in q4_mean.tolist()))
print("Q4: all image tokens, for reference:                 " + " ".join(f"{v:.2f}" for v in q4_all_mean.tolist()))
print(f"Q4: first layer below 0.5: {q4_first}")

# one figure of all the quiz's runs
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
layers_axis = np.arange(N_LAYERS)
axes[0, 0].plot(layers_axis, patch_curves["object tokens"], "-o", color=utils.GREY, markersize=3.5, label="Part 4: 100 pairs")
axes[0, 0].plot(layers_axis, q1_mean.numpy(), "-o", color=utils.BLUE, markersize=4, label="Q1: 20 new pairs")
axes[0, 0].fill_between(layers_axis, q1_low.numpy(), q1_high.numpy(), color=utils.BLUE, alpha=0.2, lw=0)
axes[0, 0].axvline(4, color="k", ls=":", lw=1)
axes[0, 0].set(title="Q1: object-token recovery, colour pairs", xlabel="layer", ylabel="recovery", ylim=(-0.05, 1.1))
axes[0, 0].legend(fontsize=8)
words4 = [" one", " two", " three", " four"]
axes[0, 1].bar([w.strip() for w in words4], [q2_says.get(w, 0) / len(scenes4) for w in words4], color=[utils.GREY, utils.GREY, utils.ORANGE, utils.BLUE])
axes[0, 1].set(title="Q2: what the model says on 500 four-object scenes", ylabel="fraction of scenes", ylim=(0, 1.05))
axes[1, 0].bar(["S: proj. only\n600 steps", "L: proj. only\n1,500 steps", "J: proj. + GPT-2\n1,500 steps"], [q3_runs["S"], q3_runs["L"], q3_runs["J"]], color=[utils.GREY, utils.ORANGE, utils.BLUE])
axes[1, 0].axhline(IMAGE_BLIND["colour"], color="k", ls=":", lw=1.2, label="best image-blind guess")
axes[1, 0].set(title="Q3: colour accuracy", ylabel="accuracy", ylim=(0, 1.08))
axes[1, 0].legend(fontsize=8)
axes[1, 1].plot(layers_axis, patch_curves["object tokens"], "-o", color=utils.GREY, markersize=3.5, label="colour pairs (Part 4)")
axes[1, 1].plot(layers_axis, q4_mean.numpy(), "-o", color=utils.PURPLE, markersize=4, label="Q4: 40 shape pairs")
axes[1, 1].fill_between(layers_axis, q4_low.numpy(), q4_high.numpy(), color=utils.PURPLE, alpha=0.2, lw=0)
axes[1, 1].plot(layers_axis, q4_all_mean.cpu().numpy(), "--s", color=utils.GREEN, markersize=3.5, label="Q4: all image tokens")
axes[1, 1].axhline(0.5, color="k", ls=":", lw=1.2)
axes[1, 1].set(title="Q4: object-token recovery, shape pairs", xlabel="layer", ylabel="recovery", ylim=(-0.05, 1.1))
axes[1, 1].legend(fontsize=8)
plt.tight_layout()
plt.show()

# the written key must still be what these runs produce
assert letter(float(q1_mean[4]), [0.5, 0.8, 0.9, 1.0]) == quiz.questions["Q1"].key, "Q1 key has drifted"
assert letter(q2_accuracy, [0.02, 0.2, 0.5, 0.8]) == quiz.questions["Q2"].key, "Q2 key has drifted"
assert Ranking.from_values(q3_runs, eps=0.02).key_text == quiz.questions["Q3"].key_text, "Q3 key has drifted"
assert ("e" if q4_first is None else "abcde"[sum(q4_first >= edge for edge in [1, 2, 6, 11])]) == quiz.questions["Q4"].key, "Q4 key has drifted"
print("The written key matches these runs.")

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: [html]

r'''
<img src="media/quiz_runs.png" width="900">
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
<details>
<summary>Help - how should I read this figure?</summary>

The top left shows Q1: the object-token recovery on the 20 new pairs (blue, with its band) lies on top of the 100-pair curve from Part 4 (grey). The top right is Q2: on four-object scenes the model says `three` almost every time and `four` never. The bottom left is Q3: colour accuracy of the three runs, with the best image-blind guess dotted;
the projector-only run with 1,500 steps (orange) is nearly level with the joint run (blue). The bottom right is Q4: the shape-pair recovery (purple, with its band) drops at layer 1 and stays low, while the Part 4 colour curve (grey) stays high until layer 9. The green dashed line is the all-image-tokens patch for the shape pairs, which also falls to 0.35 at layer 1.
</details>
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
## What this lesson doesn't show

Everything in this lesson comes from one training seed of one mini VLM on synthetic shapes, so a different seed could move the numbers (the quiz gives you a taste of this: change the
training and the answer changes). The Qwen3-VL comparison uses 20 counterfactual pairs of the same synthetic scenes, which is why its bands are wide. Several limits are worth
keeping in mind:

- **"Is it looking?" is a behavioural question.** Blank, noise, shuffled and random-projector controls show that the answers depend on the image; they don't say what the model represents.
  The image-only control is the trained model on a prompt it never saw, not a separately trained image-only model.
- **Patching shows sufficiency, and the object mask is geometric.** SigLIP is contextual, so some information about the object sits in tokens outside the object's mask; this is why patching all
  image tokens recovers more than the object's tokens alone.
- **Knockout blocks routes in groups,** with attention renormalised over the survivors, and a joint blockade of every route is not tested.
- **The left/right result is reported, not explained.** The model reaches only 94.5% on `horizontal` after training (top or bottom: 99.7%), and the periodic-position idea in the first Bonus sections is a guess to test.
- **No natural images, no new templates, no held-out colour-shape combinations.** The scenes and questions are synthetic and fixed.
- **Curves that agree are not the same circuit.** The mini VLM and Qwen share a coarse shape on a relative-depth axis; their heads, routes and sizes are different.
'''

# ! CELL TYPE: markdown
# ! FILTERS: []
# ! TAGS: []

r'''
## Two stages or one? How long does left-versus-right take?

LLaVA trains in two stages: first the projector alone, then projector and language model. In this section regime (B) trained everything together from the start, and regime (A)
stopped after the first stage. Try the two-stage recipe: train the projector for 600 steps with the language model frozen (that is exactly regime (A)), then **continue from
there**: call `train_vlm` again with `train_lm=True`, reusing the projector you just trained and the same (so far untouched) language model, for another 1,500 steps. Compare
held-out accuracy against joint training at a matched total compute budget. Nobody has run this continuation or a learning-rate ablation for you, so the result is open. In the joint run,
horizontal accuracy (94.5%) trails vertical (99.7%) despite almost perfect colour, shape and count; use that observation to motivate a measured learning-curve study rather than
assuming a particular "take-off".

- Plot accuracy on each kind against the number of training steps. Is the take-off sudden?
- Which regime reaches high spatial accuracy first, and which setting of the learning rates matters most?
- What does the model do *before* the take-off on left/right questions: does it have a constant guess (50%), or does it use partial position information?

## Why is left-versus-right the hard one?

Image tokens are laid out in raster order and GPT-2 sees only 1-D absolute positions 1-49. *Vertical* position is a monotone function of the position index (`row = (index - 1) //
7`); *horizontal* position is a **periodic** one (`col = (index - 1) % 7`), which a model with learned absolute position embeddings has to discover. Test this idea:

- Check the training-time accuracy curves for `horizontal` against `vertical`. Which one takes off first?
- Replace GPT-2's positional embeddings for image tokens by a *2-D* learned embedding (one embedding per row plus one per column, added to the projector's output), and see whether left/right becomes as easy as top/bottom.
- Qwen2-VL's multimodal RoPE gives image tokens 2-D positions for this reason. Can you give the *question* tokens a 2-D position too, and what does that do to the attention analysis of section 4️⃣?

## LoRA instead of full fine-tuning

Full fine-tuning updates all of GPT-2's weights. Implement a LoRA adapter on the attention projections of GPT-2 (`W + (alpha/r) A B` with $A \in \mathbb{R}^{d \times r}$, $B \in
\mathbb{R}^{r \times d}$, $B$ initialised to zero) and train only the projector and the adapters. How small can the rank $r$ be before accuracy drops? Is the *interpretability*
result (patching curve, attention heads) the same for the LoRA model? (Hugging Face's `peft` library does this for a `GPT2LMHeadModel`; writing the low-rank update yourself is a
good exercise.)

## Is a random projector a control or a different model?

In section 3️⃣ the *trained* VLM with a *random* projector fails, which tells us the trained language model depends on the *trained* projector. A different experiment: keep the
projector at its random initialisation (frozen!), and train **only the language model** on the questions. Does it learn to answer? A random linear map of SigLIP's features
preserves most of their information (a fact about random projections), so GPT-2 might learn to read it. If it does, what has the "projector is trained" control actually shown, and
what would a better control look like?

## Compositional generalisation

All 18 colour-shape pairs appear in training. Hold out three of them (say red triangles, blue circles, green squares), train again, and evaluate on scenes that contain them. A VLM
that has learned colour and shape as separate attributes should generalise; one that memorised "red triangle" as a unit would not. Does the patching curve of section 4️⃣ look the
same on the held-out combinations?

## Language priors against evidence

Real VLMs hallucinate: they say what is *likely* rather than what is *visible*. Create a training distribution with a strong prior (say 90% of circles are red) and train on it.
Then evaluate on scenes with *blue* circles. How often does the model answer "red"? Can you find the layer at which the prior wins over the image, by patching *the answer position*
from a clean run (red circle, expected) into a run on a blue circle? (Section [12.5] studies this arbitration between the image and the text prior in depth.)

## A tuned lens for image tokens

The plain logit lens failed to find colour words in the object's tokens, but the restricted lens found colour information at layer 0 already. Train a tuned lens ([7.1]) on image
tokens, with a KL objective against the model's final distribution over the six colour words at the *answer* position, and ask whether the tuned lens finds colour at the object's
tokens in a way the logit lens didn't. Does the tuned lens also "find" colour at the background tokens?

## Scale it up

- Remove the 2 × 2 merge: 196 image tokens instead of 49. What happens to the attention picture (more tokens per object) and to the patching result (a single token matters even less)?
- Swap GPT-2 for a stronger small language model (SmolLM2-360M or Qwen2.5-0.5B-Instruct), which uses rotary positions. Does left/right get easier?
- Add a second object question ("Is the {colour} {shape} left of the {colour} {shape}?"). Is the answer still read from two sets of tokens at the same layers?

## Further reading

- Liu et al. (2023), [*Visual Instruction Tuning*](https://arxiv.org/abs/2304.08485) (LLaVA). The recipe we shrank. Liu et al. (2024), [*Improved Baselines with Visual Instruction Tuning*](https://arxiv.org/abs/2310.03744) (LLaVA-1.5), introduced the MLP projector.
- Zhai et al. (2023), [*Sigmoid Loss for Language Image Pre-Training*](https://arxiv.org/abs/2303.15343) (SigLIP). The vision encoder we used.
- Neo et al. (2025), [*Towards Interpreting Visual Information Processing in Vision-Language Models*](https://arxiv.org/abs/2410.07149). The logit lens and token ablation on LLaVA's image tokens that sections 4️⃣-5️⃣ echo.
- Zhang et al. (2025), [*Cross-modal Information Flow in Multimodal Large Language Models*](https://arxiv.org/abs/2411.18620). Attention knockout between image, question and last position, window by window.
- Jiang et al. (2025), [*Interpreting and Editing Vision-Language Representations to Mitigate Hallucinations*](https://arxiv.org/abs/2410.02762). Logit-lens-style readouts of image tokens used to find and remove hallucinated objects.
- Darcet et al. (2024), [*Vision Transformers Need Registers*](https://arxiv.org/abs/2309.16588). Why some background patch tokens of a ViT carry global information (relevant to why background tokens were somewhat "decodable" in our lens experiment).
- Wang et al. (2024), [*Qwen2-VL*](https://arxiv.org/abs/2409.12191). The multimodal RoPE and the patch merger that the real model in section 5️⃣ builds on.
- Karamcheti et al. (2024), [*Prismatic VLMs: Investigating the Design Space of Visually-Conditioned Language Models*](https://arxiv.org/abs/2402.07865). A careful study of the choices we made by fiat (frozen vs fine-tuned encoder, one stage vs two, projector design).
- Tong et al. (2024), [*Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs*](https://arxiv.org/abs/2401.06209). Failures of VLMs on simple visual patterns, tracing back to the vision encoder.
'''


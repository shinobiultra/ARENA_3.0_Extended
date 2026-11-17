# ARENA Extended

This fork of [ARENA](https://github.com/ARENA-education/ARENA_materials) keeps the original course
(chapters 0–4) exactly as upstream ships it and adds lessons on modern architectures and
interpretability methods: sections 0.6 and 1.6, and chapters 5–18. Every extension lesson follows
the ARENA pattern: you build the method yourself, test each piece, and finish with a result you can
see next to controls that should fail. Optional companions to Stanford's
[Deep Learning Alchemy (CS 312)](https://deep-learning-alchemy.github.io/) are listed in
[ALCHEMY.md](ALCHEMY.md).

## Start here

- **New to ARENA?** Start at the beginning of the original course, chapter 0.
- **Done ARENA chapters 0–1?** Start at
  [0.6 How to Know When an Interpretability Result Is Fake](chapter0_fundamentals/instructions/pages/06_%5B0.6%5D_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice.md).
- The [study guide](mastery_hub/STUDY-GUIDE.md) shows the order of the extension lessons and how to
  work through one.
- Download a fixed version from the [latest release](https://github.com/shinobiultra/ARENA_3.0_Extended/releases/latest).

## Setup

The course runs on Linux x86_64 (WSL2 on Windows works) with Python managed by
[uv](https://docs.astral.sh/uv/). A GPU needs an NVIDIA driver ≥ 580. macOS isn't supported,
because the CUDA wheel index the course uses has no macOS wheels.

```bash
git clone https://github.com/shinobiultra/ARENA_3.0_Extended.git
cd ARENA_3.0_Extended
uv sync --locked
```

Then open the lessons in the local reader. Each page offers its exercises and solutions notebooks
for download:

```bash
uv run streamlit run mastery_hub/read.py --server.address=127.0.0.1 --server.port=18505
```

Or open a lesson's notebook directly with `uv run jupyter lab`. Each lesson's notebooks sit in its
chapter's `exercises/` folder, and its page names them at the top.

A few lessons need more:

- 5.4 uses the CUDA Mamba kernels: `uv sync --locked --extra cuda-kernels` (this builds them from
  source).
- Lessons on gated models (such as Gemma) say so in their Setup. Accept the model's licence on
  Hugging Face, then run `uv run hf auth login`.
- Models and datasets download on first use, so the first run of a lesson takes longer.

### What your hardware runs

| Hardware | What runs |
| --- | --- |
| No GPU | 62 lessons fully and 16 more slowly. For the 8 CUDA lessons (5.4, 5.8, 11.1, 13.1, 13.2, 14.1, 14.2, 16.7), read the saved outputs in the solutions notebook or rent a GPU. |
| 8 GB GPU | Every lesson except 11.1 |
| 16 GB GPU or more | Every lesson (11.1 peaks at 13.6 GiB) |

## What the extension adds

| Chapter | Topic |
| --- | --- |
| [0.6](chapter0_fundamentals/instructions/pages/06_%5B0.6%5D_How_to_Know_When_an_Interpretability_Result_Is_Fake_Practice.md), [1.6](chapter1_transformer_interp/instructions/pages/40_%5B1.6%5D_Local_Frontier_ML_Infrastructure_Practice.md) | Spotting fake interpretability results; local ML infrastructure |
| [5](chapter5_modern_architectures/instructions/pages) | Modern architectures |
| [6](chapter6_sparse_feature_methods/instructions/pages) | Sparse feature methods |
| [7](chapter7_activation_to_language/instructions/pages) | Activation to language |
| [8](chapter8_automated_circuits/instructions/pages) | Automated circuits |
| [9](chapter9_alignment_interpretability/instructions/pages) | Alignment interpretability |
| [10](chapter10_capstone_research_sprint/instructions/pages) | Capstone research sprint |
| [11](chapter11_representation_geometry/instructions/pages) | Representation geometry |
| [12](chapter12_vlm_interpretability/instructions/pages) | VLM interpretability |
| [13](chapter13_image_generation_interpretability/instructions/pages) | Image-generation interpretability |
| [14](chapter14_jepa_world_models/instructions/pages) | JEPAs and world models |
| [15](chapter15_peft_misalignment/instructions/pages) | PEFT and misalignment |
| [16](chapter16_shapley_attribution_baselines/instructions/pages) | Shapley attribution baselines |
| [17](chapter17_training_dynamics/instructions/pages) | Training dynamics |
| [18](chapter18_intrinsic_interpretability/instructions/pages) | Intrinsic interpretability |

---

# Original ARENA

Everything below is upstream ARENA's README, unchanged. Its install instructions set up the
original course on its own; the Setup above covers both.

### ARENA [slack channel](https://info-arena.github.io/ARENA_img/slack.html)

* Please report any errors/concerns with the material in #errata.

# Install Instructions

1) Clone the repo
```
git clone https://github.com/ARENA-education/ARENA_materials.git
```
2) Run the install script. `install_uv.sh` uses [uv](https://docs.astral.sh/uv/) and creates the environment at `ARENA_materials/.venv`:
```
bash ARENA_materials/install_uv.sh
```
If that fails, fall back to the conda-based script:
```
bash ARENA_materials/install.sh
```

This GitHub repo hosts the exercises and Streamlit pages for the ARENA program.

You can find a summary of each of the chapters below. For more detailed information (including the different ways you can access the exercises), click on the links in the chapter headings.

Scroll to the end to see our instructions for submitting PRs.

# [Chapter 0: Fundamentals](https://learn.arena.education/chapter0_fundamentals/)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/headers/header-ch0.png" width="400">

The material on this page covers the first five days of the curriculum. It can be seen as a grounding in all the fundamentals necessary to complete the more advanced sections of this course (such as RL, transformers, mechanistic interpretability, training at scale, and generative models).

Some highlights from this chapter include:

* Building your own 1D and 2D convolution functions
* Building and loading weights into a Residual Neural Network, and finetuning it on a classification task
* Working with [weights and biases](https://wandb.ai/site) to optimise hyperparameters
* Implementing your own backpropagation mechanism
* Building your own GANs and VAEs, and using them to generate images

# [Chapter 1: Transformer Interpretability](https://learn.arena.education/chapter1_transformer_interp/)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/headers/header-ch1.png" width="400">

The material on this page covers transformers (what they are, how they are trained, how they are used to generate output) as well as mechanistic interpretability (what it is, what are some of the most important results in the field so far, why it might be important for alignment) and other topics related to interpretability (function vectors & model steering).

Some highlights from this chapter include:

* Building your own transformer from scratch, and using it to sample autoregressive output
* Using the [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) library developed by Neel Nanda to locate induction heads in a 2-layer model
* Finding a circuit for [indirect object identification](https://arxiv.org/abs/2211.00593) in GPT-2 small
* Interpreting model trained on toy tasks, e.g. classification of bracket strings, or modular arithmetic
* Replicating Anthropic's results on [superposition](https://transformer-circuits.pub/2022/toy_model/index.html), and training sparse autoencoders to recover features from superposition
* Using [steering vectors](https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector) to induce behavioural changes in GPT2-XL

Unlike the first chapter (where all the material was compulsory), all sections of this chapter are optional extensions other than the first two exercise sets. In these first two sets you will build and train transformers, and gain a basic understanding of mechanistic interpretability of transformer models which includes induction heads & use of TransformerLens. After this, you can pick any of the other six sets of exercises you want - there are no prerequisites!

If you've finished the compulsory material and are choosing between the other six sets of exercises, we weakly recommend choosing one of the first three (IOI, superposition, and function vectors). IOI should appeal to the experimentalists, superposition to the theorists / mathematicians, and function vectors to the engineers, so there's something for everyone!

Additionally, each optional set of exercises includes a lot of suggested bonus material / further exploration once you've finished, including suggested papers to read and replicate.

# [Chapter 2: Reinforcement Learning](https://learn.arena.education/chapter2_rl/)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/headers/header-ch2.png" width="400">

Reinforcement learning is an important field of machine learning. It works by teaching agents to take actions in an environment to maximise their accumulated reward.

In this chapter, you will be learning about some of the fundamentals of RL, and working with OpenAI’s Gym environment to run your own experiments.

Some highlights from this chapter include:

* Building your own agent to play the multi-armed bandit problem, implementing methods from [Sutton & Barto](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf)
* Implementing a Deep Q-Network (DQN) and Proximal Policy Optimization (PPO) to play the CartPole game
* Applying RLHF to autoregressive transformers like the ones you built in the previous chapter

Additionally, the later exercise sets include a lot of suggested bonus material / further exploration once you've finished, including suggested papers to read and replicate.

# [Chapter 3: LLM Evaluations](https://learn.arena.education/chapter3_llm_evals/)

<img src="https://raw.githubusercontent.com/callummcdougall/computational-thread-art/master/example_images/misc/headers/header-ch3.png" width="400">

The material in this chapter covers LLM evaluations (what they are for, how to design and build one). Evals produce empirical evidence on the model's capabilities and behavioral tendencies, which allows developers and regulators to make important decisions about training or deploying the model. In this chapter, you will learn the fundamentals of two types of eval: designing a simple multiple-choice (MC) question evaluation benchmark and building an LLM agent for an agent task to evaluate model capabilities with scaffolding. 

Some highlights from this chapter include:

* Design and generate your own MCQ eval from scratch using LLMs, implementing Anthropic's [model-written eval](https://arxiv.org/abs/2212.09251) method
* Using the [Inspect](https://inspect.ai-safety-institute.org.uk/) library written by the UK AI Safety Institute (AISI) to run evaluation experiments
* Building a LLM agent that plays the Wikipedia Racing game
* Implementing ReAct and inflexion as elicitation methods for LLM agents 

The exercises are written in collaboration with [Apollo Research](https://www.apolloresearch.ai/), and designed to give you the foundational skills for doing safety evaluation research on language models. 

# [Chapter 4: Alignment Science](https://learn.arena.education/chapter4_alignment_science/)

Coming soon!

---

# Submitting PRs

If you want to submit a PR to the repo (e.g. fixing a bug or typo), this would be much appreciated! The easiest way to do this for us is by editing the **master Python file** (not the notebook) in `infrastructure/master_files`, since these are the files that generate all other pages (both Colabs, Streamlit pages, solutions files). For example, if you want to edit material 2.2 (Q-Learning and DQN), you should edit just `infrastructure/master_files/master_2_2.py`. After PRs are merged, we then run code which updates all the other files based on this one (so you don't have to worry about any of those other files, only the master Python file!).

If you find the PR confusing (because you're not sure exactly what to edit in these master files), then please either send a message in the `#errata` Slack channel, or just make a PR on non-master files (e.g. the `solutions.py` file or the markdown files) and we'll be able to merge it & replicate it on the master files.

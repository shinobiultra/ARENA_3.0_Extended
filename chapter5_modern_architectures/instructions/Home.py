import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    "## TLDR (if you're in a hurry!)\n"
    '\n'
    '**Prerequisites:** [1.1] Transformer from Scratch (essential), [1.2] Intro to Mech Interp (helpful).\n'
    '\n'
    "- Start with **5.1 Gemma from Scratch**: it's the modern version of [1.1], and every later section reuses its components (RMSNorm, RoPE, grouped-query attention, gated "
    'MLPs, KV caches).\n'
    '- Then pick a track:\n'
    '    - *Attention variants*: 5.2 Mixture of Experts.\n'
    '    - *Recurrent and state-space models*: 5.3 Mamba → 5.4 Mamba-2 → 5.5 State Tracking, then 5.6 (linear attention, RetNet, RWKV) and 5.7 (Griffin, xLSTM, Hyena, hybrids).\n'
    '    - *Non-autoregressive generation*: 5.8 Discrete Diffusion Language Models.\n'
    '    - *Specialist models*: 5.9 Embedding and Function-Calling Models.\n'
    '- Each set is 1-2 days of work.\n'
    '\n'
    '## Description of material\n'
    '\n'
    "Original ARENA teaches interpretability on GPT-2-style transformers. Today's open models look different: Gemma and Llama use RMSNorm, rotary embeddings, grouped-query and "
    'sliding-window attention; many frontier models are mixtures of experts; and state-space models, linear-attention RNNs and diffusion language models are serious '
    "alternatives to the transformer. In this chapter you'll build each of these from scratch, check your implementation against real released weights, and then ask what each "
    'architecture makes easier or harder to interpret: where state lives in a recurrent model, what a router specialises on, what "the residual stream at layer L" even means '
    'for a diffusion model that generates all tokens at once.'
)

render_chapter_home(__file__, INTRO)

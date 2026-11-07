import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    "## TLDR (if you're in a hurry!)\n"
    '\n'
    '**Prerequisites:** [0.5.1] VAEs and [0.5.2] GANs; 12.1 helps.\n'
    '\n'
    '- **13.1 Diffusion Models and Cross-Attention** first (it builds a small diffusion model, then looks inside Stable Diffusion).\n'
    '- **13.2 Denoising-Time Causal Patching** asks *when* during generation each decision is made.\n'
    '- 13.3 (concept directions and LoRA), 13.4 (diffusion SAEs), 13.5 (diffusion transformers) and 13.6 (autoregressive image tokens) in any order.\n'
    '\n'
    '## Description of material\n'
    '\n'
    'Image generators are a different kind of model to interpret: there\'s no single forward pass, the "answer" emerges over dozens of denoising steps, and the output is pixels '
    'rather than tokens. This chapter builds the main generator families from scratch and studies them with attention attribution, causal patching across timesteps, concept '
    'directions, sparse autoencoders and adapter analysis.'
)

render_chapter_home(__file__, INTRO)

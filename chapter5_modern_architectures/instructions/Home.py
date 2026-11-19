import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    '## Description of material\n'
    '\n'
    'Original ARENA teaches interpretability on GPT-2-style transformers. Today\'s open models look different: Gemma and Llama use RMSNorm, rotary embeddings, grouped-query and sliding-window attention; many frontier models are mixtures of experts; and state-space models, linear-attention RNNs and diffusion language models are serious alternatives to the transformer. In this chapter you\'ll build each of these from scratch, check your implementation against real released weights, and then ask what each architecture makes easier or harder to interpret: where state lives in a recurrent model, what a router specialises on, what "the residual stream at layer L" even means for a diffusion model that generates all tokens at once.'
)

render_chapter_home(__file__, INTRO)

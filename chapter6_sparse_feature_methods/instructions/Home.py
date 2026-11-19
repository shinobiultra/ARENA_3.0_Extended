import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    '## Description of material\n'
    '\n'
    "Sparse dictionary learning is the workhorse of modern mechanistic interpretability. This chapter goes beyond the vanilla ReLU SAE: you'll train TopK, BatchTopK, gated and JumpReLU SAEs, evaluate them on the sparsity-fidelity frontier, use Google's Gemma Scope dictionaries to find and steer features, study the pathologies that make features misleading, and build transcoders and crosscoders, which are the building blocks for circuit tracing and model diffing."
)

render_chapter_home(__file__, INTRO)

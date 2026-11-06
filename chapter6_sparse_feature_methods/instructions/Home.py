import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from infrastructure.extension.home import render_chapter_home

INTRO = (
    "## TLDR (if you're in a hurry!)\n"
    '\n'
    '**Prerequisites:** [1.3.3] Interpretability with SAEs (or at least [1.5.4] Toy Models of Superposition & SAEs).\n'
    '\n'
    "- **6.1 SAE Variants** comes first: you'll train every major SAE architecture yourself and learn how to compare them.\n"
    '- **6.2 Gemma Scope** shows how to use released SAEs on a real model, and **6.3** covers what goes wrong (feature splitting, absorption) and how to build dashboards you '
    'can trust.\n'
    '- **6.4 Transcoders** and **6.5 Crosscoders** extend the dictionary-learning idea to replacing MLPs and comparing models; 6.4 is a prerequisite for the attribution-graph '
    'section in chapter 8.\n'
    '\n'
    '## Description of material\n'
    '\n'
    "Sparse dictionary learning is the workhorse of modern mechanistic interpretability. This chapter goes beyond the vanilla ReLU SAE: you'll train TopK, BatchTopK, gated and "
    "JumpReLU SAEs, evaluate them on the sparsity-fidelity frontier, use Google's Gemma Scope dictionaries to find and steer features, study the pathologies that make features "
    'misleading, and build transcoders and crosscoders, which are the building blocks for circuit tracing and model diffing.'
)

render_chapter_home(__file__, INTRO)

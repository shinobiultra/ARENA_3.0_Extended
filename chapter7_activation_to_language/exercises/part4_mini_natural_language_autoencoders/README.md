# [7.4] Mini Natural Language Autoencoders

This section asks whether a language model contributes order-sensitive meaning
to an activation-to-text-to-activation bottleneck, or whether a lexical decoder
can reproduce the same result without processing language.

Pinned `EleutherAI/pythia-70m-deduped` supplies 96 training and 32 held-out
layer-4 states. The activation verbalizer predicts complete 24-token GPT-NeoX
explanations. Within each lexical context, all four number/time states contain
the same token multiset; only the order of `one`/`many` and
`current`/`earlier` changes. The activation reconstructor passes these ordered
tokens through Pythia before fitting the paper-style normalized affine head.

On CUDA 13.2, all 32 held-out explanations are exact. The contextual path gets
normalized FVE `0.816` and preserves 32/32 original Pythia choices. A matched
bag-of-tokens ridge model sees the same texts and uses the same reconstruction
head, but gets FVE `0.642` and behavior parity `0.0`. Its within-factor FVE is
still `0.841`, which visibly demonstrates that lexical noun/style context can
look strong while losing the order-coded causal factors.

Additional controls include a format-matched fixed-context explanation, direct
factor regression, class means, class-mean-conditioned AR training, lexical
relabeling, and eight independently shuffled activation-verbalizer targets.
Identity patching is exact, a one-row patch changes no other row, and a real
text edit changes the selected candidate logit by `5.61`.

This is a supervised warm-start, not unsupervised NLA discovery, RL training,
free-form explanation generation, or unseen-vocabulary generalization. The
full CUDA path peaks at `1.009 GB` on the RTX 5090 Laptop GPU.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python -m pytest -q \
  chapter7_activation_to_language/exercises/part4_mini_natural_language_autoencoders/tests.py
```

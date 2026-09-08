# 11.5 Contrast-Consistent Search

Can an unlabelled consistency objective identify the feature you intended?
Nine exercise groups expose twelve core functions, with immediate tests, hints,
solutions and fifteen controlled figures. An exact counterexample makes a target
and an independent distractor equally consistent. The native study then compares
customer sentiment with another reader's opinion inside frozen GPT-2.

All three sites, two prompt forms and three text variants remain visible. Each
case retains ten centered CCS restarts, ten raw fits, ten corrupted-pair fits,
paired-difference PCA, a supervised probe and 32 random directions. Training loss
selects restarts; separate calibration labels fix orientation. Final evaluation
uses new review frames and adjectives without refitting. Its 512 contrast pairs
come from 64 base reviews, giving 98,304 reporter scores across the three sites.
They are repeated measurements, not 98,304 independent reviews.

Two **Try It Yourself** cells change the synthetic distractor's strength and
change a review while holding the native reporters fixed. The native gallery
shows actual review text, both opinions, three competing scores and unrestricted
winning tokens. Restricted two-answer classification is distinguished from what
the model predicts over its full vocabulary.

`lesson.md` is canonical. The builder completes each function from its own
adjacent solution dropdown. Execute the whole lesson in a new directory:

```bash
CUDA_VISIBLE_DEVICES='' uv run python chapter11_representation_geometry/exercises/part5_contrast_consistent_search/execute_learner.py --output-directory ~/.cache/arena_extended/ccs/my-own-run
```

Cache the pinned `gpt2` revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`
first; model loading uses local files. To download the required files once:

```bash
uv run python - <<'PYTHON'
from huggingface_hub import snapshot_download
snapshot_download(
    "gpt2", revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
    allow_patterns=["config.json", "generation_config.json", "model.safetensors",
                    "merges.txt", "tokenizer.json", "tokenizer_config.json", "vocab.json"],
)
PYTHON
```

Execution uses CPU float32 and reporter fitting uses float64. Each complete native record uses about 0.9 GB of disk per run.
`ccs_reproduction.py --learner-directory DIRECTORY --output-directory NEW_DIRECTORY`
reloads the model, recaptures all splits, refits every reporter and compares
all actual tensor values, tokens, scores and metrics, including the fixed play.

Publication requires the complete own-answer run, independent reproduction and
rendered figure/math/dropdown review. The normal course report also performs a
fresh CPU study:

```bash
CUDA_VISIBLE_DEVICES='' uv run python scripts/run_extension_verification_reports.py --section 11.5
```

The evidence concerns controlled binary review sentences. Consistency alone does
not establish truth, a model's beliefs, causal necessity or a unique semantic
feature. This is a scoped method study, not the original paper's full benchmark.

For a documentation or publication-only refresh, set
`ARENA_CCS_REPRODUCTION_DIRECTORY` to a completed fresh reproduction directory.
The report then verifies every scientific source, source snapshot, native tensor
artifact and the exact own-answer learner before reusing those results. It
explicitly records `native_evidence_reused` and whether the model was reloaded
in that report invocation. Any scientific change rejects reuse. Omitting the
variable performs a new independent model capture and refit.

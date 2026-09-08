# Word-level Mini VLM candidate experiments

These are independent candidate measurements, not completed-learner or release
validation. Every JSON is losslessly gzip compressed; `archive_manifest.json`
records compressed and raw hashes. Source snapshots preserve the exact candidate
implementation. Model weight files remain outside the course repository.

All twelve development runs use the same 600-update budget: joint, text-only,
image-only and shuffled-pair training at seeds 0, 1 and 2. Each checkpoint is
chosen using only its development exact generation score, taking the earliest
checkpoint on a tie. `frozen_selection.json.gz` binds all choices and weight
hashes before any final evaluation. All seeds and all 36 final evaluations
remain in the record.

The joint models average 89.6% exact generation on new familiar solid images
and 85.8% on paired muted images. On the three unseen color–shape combinations,
color answers average 90.3%, shape answers 8.0%, and complete captions **0%**.
Do not retune models from those final results. The fixed example gallery shows
the failure rather than filtering for successful captions.

The causal study uses the first four instances of every familiar combination
in the fixed final recipe, each with color and shape questions: 72 pairs per
seed. Counterfactuals cycle only the queried attribute. Geometry, background
and sensor noise match exactly. The object region is the union of occupied
clean/donor pixels, mapped to overlapping patch rows. Each stage retains eight
equal-size background and eight unrestricted random-region samples, plus object,
all-visual and full-sequence conditions: 285 generation experiments in total.

Joint-correct endpoint denominators are 50/72, 51/72 and 45/72 for seeds 0–2;
every raw pair remains available. Input object patches reproduce the donor
generation; background input patches are inert. Contextualized background
tokens can affect answers. Visual-only patches after the final decoder block
are structurally inert, while full-sequence patches reproduce donor generation
at every stage. These checks pass for every pair, even when the donor itself
answers incorrectly. The figures show all seeds and all random draws.

The model has 127,392 parameters and a 24-word vocabulary. It is a controlled
RGB-to-language learning experiment, not a pretrained open-domain VLM or a
replication of a large-model paper's headline result.

# Mini VLM from Scratch

Build the visual-token bridge and ask which image regions causally change a
mini VLM's answer. Implement the methods in the
[exercises](12.3_Mini_VLM_from_Scratch_exercises.ipynb), with immediate checks,
help and solution dropdowns, controls and parameter play. Start the
[guided prediction checkpoint](../../../mastery_hub/units/extension_predictions/exercises.ipynb#section-12-3)
before opening the answer-rich reference.

The [solutions](12.3_Mini_VLM_from_Scratch_solutions.ipynb) retain actual outputs
and 16 figures from the reviewed October 16 execution. Image-only, text-only,
shuffled-image and object/background patches expose different failure modes.
The Mini image-only counting control remains strong; Qwen's empty-query failure
also occurs on single-object examples. Patching uses selected counterfactual pairs,
not a complete independent evaluation. Highest-attention comparison and the shape
transfer remain prospective student experiments.

The raw controller's schema failure and separate metadata-only recovery remain
preserved. No trained Mini VLM state dictionaries were saved. See the
[publication record](../../../docs/mastery/migrations/minivlm_publication_2026-10-19.md)
for source/runtime identities, controls, output parity and scientific limits.

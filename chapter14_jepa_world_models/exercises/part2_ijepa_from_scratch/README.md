# [14.2] I-JEPA from Scratch

This section builds a small Image-based Joint-Embedding Predictive Architecture
end to end on an exact generated world. A visible colour key identifies an
object whose every class-coloured pixel is hidden by the target mask. Students
implement the world, patchification, structured masks, online and EMA target
encoders, latent predictor, normalized objective, collapse diagnostics,
transfer probes, key-only causal intervention, compound joint-target control, and the
training loop.

The signature claim is causal and deliberately narrow: on pixel-distinct generated
scenes sharing all twelve object-position templates, correct visible context predicts the masked object's target-latent class,
while swapping only the visible key changes the prediction to the alternate
key's class with zero non-key pixel change. Erasing the key reduces accuracy to chance, zero latent context raises loss, and the
untrained architecture remains at chance.
Random encoder features already support strong linear probes, so probe accuracy
is presented as a negative result rather than evidence for learned semantics.

Files:

- `14.2_I-JEPA_from_Scratch_exercises.ipynb`: learner notebook with twelve
  exercises, immediate tests, expected outputs, help, interpretation, and
  solution dropdowns.
- `14.2_I-JEPA_from_Scratch_solutions.ipynb`: completed solution with all 28 code cells executed in a clean kernel,
  including the three-seed CUDA signature and the intervention play cell.
- `solutions.py`: visible reference implementation and strict CPU/CUDA runners.
- `tests.py`: exact toy oracles for projection, positions, context scattering,
  target masking, prediction metrics, validation, production containment,
  key-only interventions, gradient boundaries, and a real end-to-end result.
- `artifacts.lock.yml`: claim boundary, controls, thresholds, and runtime pins.
- `verification_report.json`: supporting release evidence, never lesson input.
- `expected_outputs/`: pinned reference values and interpretation.

`run_gpu_test` requires CUDA and has no CPU fallback. This controlled result is
not an ImageNet transfer claim and is not parity with Meta's pretrained I-JEPA.

All twelve learner solution dropdowns are generated from top-level source,
dedented, and parsed as Python in both notebooks and the instruction page.

Original ARENA files are not modified by this section.

To rebuild the surfaces, run `uv run python
chapter14_jepa_world_models/exercises/part2_ijepa_from_scratch/build_notebooks.py`.
Rebuilding invalidates stored output, so execute a learner copy completed from
its twelve dropdowns before publishing. The AST parity test checks every answer
against the reference implementation and the paired solution cell.

The target-training ablation changes **both** stop-gradient and EMA; it does
not isolate EMA's causal effect. Our normalized squared-distance loss, learned
positions, and label-located target block are teaching simplifications relative
to the official smooth-L1, fixed-position, sampled multi-block implementation.
CIFAR/STL learning and released-checkpoint comparison remain unfinished.

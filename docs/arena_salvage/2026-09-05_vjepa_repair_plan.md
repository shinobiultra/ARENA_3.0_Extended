# 14.3 V-JEPA learner repair plan

The current candidate already states the correct negative conclusion: its
future-mask predictor does not beat persistence. Preserve that result and all
existing numerical acceptance thresholds. This is a generated-world lab, not
completion of the required released V-JEPA / V-JEPA 2 paths.

Inspected the exercise and solution notebooks, instruction export, readable
builder, reference implementation, semantic tests, reference generator, README,
and artifact lock. The original transformer and CNN/ResNet lessons inspected
earlier remain the learner-style comparators; original ARENA is append-only.

1. Add a student-implemented blank-future control. Preserve the observed frames,
   erase every future pixel, and encode the result in the same teacher space.
   This tests context carryover in teacher features, since target attention sees
   the entire clip. Test nonmutation, future erasure, and score derivation.
2. Enforce the promised indivisible-dimension rejection in TubeletEmbedding,
   including a semantic test that would fail on silent Conv3D truncation.
3. Make each solution notebook cell match a learner cell completed only from
   its own dropdown. Use AST substitution for stubs, including classes; preserve
   provided helpers and immediate tests. Add expected shapes and interpretation
   guidance where the existing exercise jumps directly to a large answer.
4. Show the blank-future control in the signature figure, tables, and play cell.
   Explain that even perfect clip-key retrieval would not isolate future dynamics
   when teacher features retain context. Add visible controls for the loss and
   collapse panels. Keep exact dynamics explicitly privileged by velocity labels.
5. Execute all completed learner cells in a fresh CPU kernel, inspect rendered
   figures, regenerate three-seed references without relaxing thresholds, run
   semantic and preservation checks, and record the candidate's remaining scope.

No original chapter, new method, or new topic directory is changed by this repair.

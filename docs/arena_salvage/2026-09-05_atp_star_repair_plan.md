# 8.6 AtP* scientific and learner repair plan

Current learner failure: the real experiment is a single hidden reference-runner
call; the core Q/K and GradDrop method is not implemented in the notebook. The
playground is only Markdown code. The real gradient hooks detach every Q/K or
attention-pattern tensor, which removes downstream paths through later attention
operations and invalidates the claimed node gradients. The checked-in report says
accepted despite having CPU-only evidence and incomplete paper diagnostics.

Target claim: use exact nonlinear oracles to understand softmax saturation and
cancellation, then compare properly computed node-gradient and Q/K-corrected
estimates against exact final-position Pythia interventions. Do not preserve a
positive ranking claim if the corrected experiment fails it.

The notebook pair, page, source/tests, lock/report, roadmap AtP* requirements
(Extension-Roadmap.md:825-849 and 5269-5288), and original ARENA intervention
comparators have been inspected. Original ARENA is not to be edited.

1. Independently test cross-layer gradient flow before correcting hooks. Retain
   intermediate gradients while preserving the computation graph; only GradDrop's
   explicitly selected branches should detach.
2. Refresh the real numerical comparison and keep the original acceptance
   thresholds unchanged. Measure both exact effects and matched controls.
3. Add a readable source builder and expose real patching, vanilla gradients,
   pattern gradients, Q/K correction, and GradDrop in learner cells. Carry support
   dependencies into the learner and test its actual answers.
4. Convert the play instructions to runnable cells. Generate ranking/recovery and
   joint-faithfulness figures from the completed learner run, with controls.
5. Keep release status unaccepted until strict CUDA evidence and independent
   scientific/editorial review are complete. The paper's subset-sampling diagnostic
   and complete circuit interpretation remain outside the current candidate.

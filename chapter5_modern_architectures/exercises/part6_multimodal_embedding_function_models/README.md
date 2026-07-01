# [5.6] Embedding Retrieval and Function-Calling Controls Verification Assets

Generated support files for the roadmap verification contract.

- `5.6_Multimodal_Embedding_and_Function_Calling_Models_exercises.ipynb`
  is the learner-facing notebook with local stubs for masked mean pooling,
  retrieval metrics, centroid probes, tool masking, FunctionGemma call parsing,
  function-call reports, and schema-token attribution.
- `5.6_Multimodal_Embedding_and_Function_Calling_Models_solutions.ipynb`
  validates the section-local reference implementation against the visible
  tests and checks the committed CUDA report highlights.
- `artifacts.lock.yml` pins the current artifact contract.
- `verification_report.schema.json` defines the required final report.
- `expected_outputs/smoke_test.json` records the smoke-test contract.
- `expected_outputs/reference_metrics.json` records baseline/control slots.

The current graded run includes pinned public BGE and FunctionGemma Mobile
Actions checkpoints, the pinned Mobile Actions dataset revision, authenticated
direct EmbeddingGemma retrieval, authenticated base FunctionGemma CUDA loading,
dtype/device metadata, and measured VRAM. Manual-gated Google artifacts are
claimed only after their local authentication/readiness checks pass.

# [5.6] Embedding Routing for Reliable Function Calls

This section asks one concrete question: when is request-to-schema embedding
similarity strong enough to justify a tool call?

The learner-facing notebooks begin with an exact three-axis semantic encoder and
16 English requests. Students implement masked pooling, cosine retrieval,
confidence-and-margin abstention, availability masking, and separate tool/no-call
metrics. The notebook then generates a labeled retrieval heatmap and compares it
against shuffled-schema and always-call controls.

The real-model escalation runs pinned `BAAI/bge-small-en-v1.5` retrieval on CPU
with the same student-written metrics. The existing EmbeddingGemma and
FunctionGemma CUDA paths remain release verification evidence; they are not the
lesson's signature result.

Files:

- `5.6_Multimodal_Embedding_and_Function_Calling_Models_exercises.ipynb`:
  student implementation and play cells.
- `5.6_Multimodal_Embedding_and_Function_Calling_Models_solutions.ipynb`:
  inline solved implementation, executed CPU outputs, controls, and real BGE
  comparison.
- `tests.py`: immediate semantic tests for each core operation.
- `solutions.py`: section-local reference implementations and the preserved
  pinned release verification path.
- `verification_report.json`: supporting serialized CUDA evidence only.

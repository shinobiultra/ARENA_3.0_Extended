# 9.1 measured learner outputs

The main teaching outputs are the six figures and concrete benign generations produced by the notebook. These files support reproducibility after the learner has implemented and run the method.

`completed_refusal_study.json.gz` contains all actual development score/null measurements, all fifteen candidate generations, the frozen choice and all 85 final conditions, with per-example outcomes, explicit denominators, benign completions/token IDs and sensitive completion hashes/flags. `learner_execution.json` binds this raw file, the paired notebooks, all six figures and every section Python source plus `lesson.md` to a complete own-answer execution. Numerical-only and changed-source candidates cannot publish this evidence.

`direction_candidates/` preserves the independent development and final studies, including all failed candidates and strong nulls, losslessly with original/compressed SHA256 hashes and exact candidate sources. They do not replace the completed learner. `legacy_proxy/` preserves superseded prototype files for diagnosis and supplies no accepted evidence.

The pinned dataset uses class-wise rows 0:32 for training, 32:64 for development and 64:100 for final evaluation. Prompt normalization is unique across all 201 source rows. The extra refusal row is unused. Thirty-two new benign task inputs are evaluated under the same four capability templates. Random and genuine training-permutation nulls use seed 901; all 32 separation draws and the first eight causal draws are retained. Undefined directions are reported, not resampled.

`smoke_test.json` is an actual analytic run showing perfect ranking but 50% accuracy after a test-distribution shift under the fixed training threshold. `reference_metrics.json` declares the protocol checks; it is not measured real-model evidence. `direction_evidence.py` recomputes actual metrics and selection from saved rows. The report entry point additionally regenerates baseline and all 34 unit-strength conditions on CUDA, checks every example hash/marker/length and benign token sequence, and freshly checks the 65 target/null classification directions.

Exact IDs, strings, tokens, marker labels and selection inventory must match. Recomputed float metrics use absolute tolerance 1e-6; fresh FP32 activation/score comparison uses 1e-4. Neither tolerance permits changing a generated outcome. Do not retune on final results, delete a strong null or infer safety from marker absence. Runtime, hardware and peak memory belong to the actual execution/report.

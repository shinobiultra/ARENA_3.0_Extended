# Seven roadmap evidence rows reconciled

The 2026-10-16 root audit found seven registry rows pointing to retired exercise
directories. The repair changes only those rows in
`docs/roadmap_requirement_registry.yml`. It identifies current teaching methods,
corrects their current section numbers and evidence paths, and replaces obsolete
method descriptions. It does not establish scientific or learner acceptance.

The exact pre-edit registry is preserved in
`docs/mastery/evidence/2026-10-16-roadmap-evidence-repair/roadmap_requirement_registry.before.yml`.
Its 129,258 bytes have SHA-256
`90307936c28e2f88832c5ba5bbe853d4ff24d7cd765e5b0be9cdff324a5e2145`.
Historical numerical results and acceptance counts remain in those original
bytes; they are not reassigned to the replacement notebooks.

## Method trace

Each replacement cites its maintained master, generated student notebook, and
local callable tests. The two Data Shapley rows additionally cite the existing
root-test and CPU-execution migration record. No path was accepted merely because
it exists.

| Requirement | Current lesson | Inspected method and retained limitation |
| --- | --- | --- |
| `circuits.attribution_patching_eap_integrated_effects` | 8.2, `part2_attribution_patching_eap` | `integrated_gradients_component`, `build_edge_mask`, `eap_scores`, and `faithfulness_curve` teach exact interventions, the 32,491-edge GPT-2 graph, gradient contractions, and held-out/random-circuit comparisons. Retired author-parity claims are not transferred. |
| `circuits.eap_ig` | 8.2, same directory | `eap_ig_scores` fixes clean-to-corrupt sender differences and averages receiver gradients at `alpha=k/n_steps`. The visible examples use five and twenty steps and compare input-edge completeness and circuit faithfulness. They do not reproduce the retired eight-point task/KL selector study. |
| `circuits.atp_star` | 8.3, `part3_atp_star` | `exact_node_effects`, `node_gradients`, `qk_fix_estimates`, `graddrop_scores`, and `group_effects` implement current GPT-2 exercises. Group tests explicitly lack the paper's statistical false-negative bound; that remains a bonus. The former Pythia description was stale. |
| `circuits.acdc` | 8.4, `part4_acdc_circuit_metrics` | `acdc` performs sequential, reverse-topological pruning against the current circuit, including disconnection cleanup. Induction and GPT-2 greater-than exercises replace the retired study as teaching evidence, without inheriting its 78-edge result or CUDA counts. |
| `circuits.circuit_metrics` | 8.4, same directory | `faithfulness`, `completeness_scores`, `edge_drops`, `roc_point`, and `gpt2_kl` teach normalized faithfulness, sampled completeness, edge minimality, sparsity, known-circuit ROC, and KL. Held-out task prompts do not establish the retired study's OOD wording-shift claim. |
| `data_attribution.exact_mc_data_shapley` | 16.9, `part9_data_shapley` | `train_on_subset`, `exact_data_shapley`, `mc_data_shapley`, and `tmc_data_shapley` define a fixed-denominator logistic retraining game and digit studies with matched random deletion controls. These remain bounded studies and differ from In-Run accounting. |
| `data_attribution.in_run_data_shapley` | 16.9, same directory | `backprop_factors`, `ghost_dot_products`, `val_hvp`, and `run_in_run_shapley` implement multi-step SGD credit and second-order corrections, with exact local-game telescoping and ordinary-SGD parameter parity. Adam/AdamW attribution remains an unimplemented bonus. |

The maintained masters are under
`infrastructure/chapters/chapter8_automated_circuits/master_8_{2,3,4}.py` and
`infrastructure/chapters/chapter16_shapley_attribution_baselines/master_16_9.py`.
Exact paths and SHA-256 values for every cited evidence file are in
`comparison.json` beside the preserved registry. The prior Data Shapley execution
belongs to its separately recorded source hashes; this repair did not execute it.

## Tracking identity and acceptance

`section` identifies the current lesson. The separate `course_sections` field is
validated against the legacy keys in `docs/arena_style_rewrite_status.yml`.
ACDC retains that registry's historical `8.3` key, and exact/MC Data Shapley retains
its historical `16.7` key. Comments and reasons make these tracking identities
explicit. Renaming the entire style registry is outside this repair.

The EAP row now references only legacy/current `8.2`; its former `8.3` mapping was
redundant with the ACDC row and described a different retired lesson. Every
previously tracked section remains represented. Roadmap line anchors and every
status value are unchanged: four repaired rows remain `TAUGHT`, two remain
`TOY_ONLY`, and AtP remains `UNIMPLEMENTED`. Existing TAUGHT labels are retained
registry classifications, not new scientific approvals.

Across all 205 requirements, the counts remain 54 TAUGHT, 20 TOY_ONLY,
35 PREFLIGHT_ONLY, 13 DEFERRED, and 83 UNIMPLEMENTED. The same 83 UNIMPLEMENTED IDs
still block course completion. This repair does not alter report requirements,
scientific metrics, thresholds, original ARENA, notebooks, pages, shared syllabus,
chapter configuration, or checkpoint acceptance.

## Validation

The existing `tests/test_roadmap_final_completeness.py` suite passed all 11 tests
in 2.61 seconds. It includes stale roadmap-hash rejection, missing evidence and
blocker rejection, tracked-section coverage, and propagation of UNIMPLEMENTED
requirements into the completion gate. No tests or audit implementations changed.

A separate read-only comparison verified the original snapshot hash, exactly
seven changed rows, all other rows and top-level metadata unchanged, all statuses
and roadmap anchors unchanged, all replacement evidence files present, zero
registry-structure blockers, and the unchanged set of 83 status blockers.
`registry-tests.txt`, `comparison.json`, and the evidence README preserve the
result and exact test command. This was an offline registry validation with no
model execution, notebook kernels, downloads, GPU use, or cluster work. The
whole-course content, scientific evidence, and human walkthrough gates remain
separate.

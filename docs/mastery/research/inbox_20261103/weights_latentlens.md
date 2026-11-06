# Weight-space programming and LatentLens: source review and proposed ARENA bridges

Draft literature/planning review, 2026-11-03. No experiment, model download, training, inference, external LLM call, cluster job, Zotero mutation or notebook modification was performed. Recommendations below are proposals, not accepted curriculum or replicated findings. Local identity reconciliation uses the coordinating review's read-only API inventory: **97 top-level 00 Inbox records, zero descendant collections**. That inventory is the scope of the absence claim below; it is not a search of the entire Zotero library.

**Exact collection match:** Sparks is **INB027**, with the requested exact title, all four authors, date `2026/11/02`, the authors' blog URL and no DOI. Its stored `journalArticle` type does not establish journal publication; the verified source is a blog. No metadata correction was made. Exact Zotero keys and record versions remain in the private inventory.

**Explicit external supplement:** none of those 97 records has a LatentLens title. **INB056** and **INB061** are untitled `webpage` records at `https://bennokrojer.com/interp.html`, not verified LatentLens paper records. The user explicitly prioritized LatentLens, so the independently verified paper below is reviewed as a supplemental primary source. Do not count it as an exact Inbox paper match, merge those two webpage records, or infer their intended paper identity. The coordinating researcher owns any wider-library search. No private annotations are reproduced here.

## Identity, version and evidence ledger

| Work | Exact identity and current source | What was actually inspected |
| --- | --- | --- |
| Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights | Rikiya Takehi, Jayden Teoh, Alex Zhang, Omar Khattab; authors' blog, 2026-11-02. [Primary post](https://rikiyat.github.io/blog/weight-space-programming/). No arXiv identifier, DOI or peer-reviewed version verified. An alphaXiv route named `2611.weight-space-programming` is not a valid arXiv identifier. | Full post sections, release README, experiment 1 and 8 READMEs, Gemma follow-up README, edited-model card. Released numbers remain author/rebuilder reports; tensor bytes and answer grading were not audited locally. |
| LatentLens: Revealing Highly Interpretable Visual Tokens in LLMs | Benno Krojer, Shravan Nayak, Oscar Mañas, Vaibhav Adlakha, Desmond Elliott, Siva Reddy, Marius Mosbach. [arXiv:2602.00462v5](https://arxiv.org/abs/2602.00462v5), 2026-07-22; initial v1 2026-01-31; metadata says ICML 2026 Camera Ready. [arXiv DOI](https://doi.org/10.48550/arXiv.2602.00462). | v5 full-text methods, main experiments and relevant appendices; v1 inspected only to detect version drift; current repository README/license and author blog inspected. HTML section/figure pointers below, not PDF-page verification. |
| LatentLens software | [McGill-NLP/latentlens](https://github.com/McGill-NLP/latentlens), live `main` as read 2026-11-03. | README, reproduction instructions and license. No install, execution, clone or index/checkpoint download. Not pinned to a verified commit. |

Version traps: v1 described 10 VLMs; v5 describes 15. The current package's default index corpus differs from the paper's reproduction corpus. Neither change may be silently harmonized. The title **LatentLens** here is not **Logit Lens**, **Tuned Lens**, **LatentQA** or **LatentReadouts**. Keep the existing `Logit Lens & Latent Readouts` topic distinct from prediction-attribution explainers; topic placement does not itself establish methodological equivalence.

## What the sources establish

### Weight-space programming: preserve the access conditions

The post demonstrates three different activities: reading serialized parameters, constructing numerical weights, and adaptively editing weights with behavioral feedback. Its strongest weight-only demonstrations are small; the larger-model demonstrations permit external execution. Experiment 7's weight-only sycophancy identification fails before attention observations are introduced. Experiment 8 reports Qwen MATH-500 7.9→48.1%, GSM8K 13.8→67.1%, IFEval 21.0→62.3%, and token-limit hits 93→33%, after a targeted edit. A less-looping Gemma follow-up finds no established math improvement. These are early demonstrations, not evidence of arbitrary network understanding, general capability acquisition or reliable no-evaluation auditing. [Post, experiments 1–8 and additional experiment](https://rikiyat.github.io/blog/weight-space-programming/).

Experiment 1 is particularly clean to teach: 17,253 weights, rounded to three decimals, serialized alongside architecture/channel/action semantics; no code execution; four best-of-three-seed policies; four correct game categories under the authors' scorer, where Frogger counts as Freeway. This is not blind interpretation of an undocumented byte array. [Experiment 1 release README, Setup/Runs](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp1_which_game/README.md).

Experiment 8 allows up to 300 CPU Python runs and 10/25/50 batched forward-pass requests. A request can generate up to 512 practice answers or teacher-force up to 1,024 texts with internals. Practice data include labeled math problems. The 50-request trajectory takes 158 turns, 80 Python runs and 20.3 hours, including a 4.7-hour outage. Reported evaluation is separate from editor feedback. [Experiment 8 release README, Setup/Which run is which](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/README.md).

The release's native Codex tools are disabled, but an external sandbox executes emitted Python and returns outputs in experiments 3/6/7/8. Thus its event-audit term `tool_free` does not mean an experiment lacked computation or feedback. Earlier prompt trials are omitted. Logs/networks are released under CC BY 4.0, author scripts under MIT, and third-party material retains separate licenses. [Release README, How the model was run/Not included/Licenses](https://github.com/RikiyaT/lm_understand-logs/).

The edited checkpoint's card reports a rebuilt, hash-checked edit against base revision `2fc06364715b967f1860aea9cf38778875588b17`. Its rerun reports 51.0% on all 500 MATH-500 problems with four samples each, but only a 25-problem base smoke test. That is useful automated evidence, not a full paired replication or human acceptance. The card reports 43.4M generated tokens, 51 minutes on an H200. No such run occurred here. [Edited-model card, Results/Verifying the weights/Reproduction](https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming).

The Gemma follow-up differs in sampling, output budget and grading: format-agnostic versus strict boxed-answer grading are both recorded, and earlier strict-format runs are excluded from the post. Retain the null result, while avoiding the stronger inference that editing can never add math ability. Model, intervention search, prompts and protocol all differ. [Gemma follow-up README, Setup/Which run is which](https://raw.githubusercontent.com/RikiyaT/lm_understand-logs/main/exp8_math/gemma_followup/README.md).

**Reviewer's interpretation and assumptions to test:**

- “Native” here names the editor's mode of working on numerical parameters. It does not establish a new weight-native input channel: some inputs are ordinary decimal text, and larger arrays are inspected/manipulated through Python.
- No gradients or optimizer loop is a meaningful restriction. It does not imply no labeled practice data, no task priors, no iterative search or zero total optimization cost.
- A sparse edit and behavior change establish an intervention on the evaluated system. They do not identify a unique mechanistic explanation; alternative edits, equivalent parameterizations and decoding changes may also rescue completion.
- “Math improvement” should be decomposed into finished-answer accuracy, truncation rate, answer length and capability retention. Rescuing completion can produce large headline gains without learning a new mathematical procedure.
- Four selected tasks and one editor family cannot estimate broad capability or robustness. Novel task seeds, architecture variants and function-preserving reparameterizations are discriminating follow-ups.
- An output-column edit is coordinate-dependent. Do not compare its parameter fraction with a LoRA rank or infer that a small fraction guarantees small behavioral collateral effects.

### LatentLens: description retrieval is the measured object

LatentLens retrieves contextual text vectors by cosine similarity, including across text/reference layers; it returns the associated descriptions. v5 studies 15 VLMs and reports judge-labeled interpretability of 68%, versus EmbeddingLens 32% and LogitLens 24%. A token qualifies if any top-five candidate is judged related, including abstract/global relations; authors validate the judge on 1,020 annotations, κ=0.68. The paper index uses 2.99M Visual Genome phrases, selected layers and up to 20 contextual vectors per vocabulary token. Its reported per-backbone index cost is about 2 GPU-hours encoding, 13 hours including index building, 26 GB storage; a 1% corpus is about 250 MB. These are source reports. [v5 §§3.2–4.5; Figures 2–5; Appendices A–C](https://arxiv.org/html/2602.00462v5).

The current library uses 117k generated concept sentences and prefix deduplication, explicitly reserving validation for a forthcoming companion paper; `reproduce/` retains Visual Genome/reservoir sampling/float8. The README's Qwen2-VL quickstart requires ≥24 GB VRAM and automatically downloads embeddings. Do not call that script during planning or present its newer corpus as paper parity. [Current repository, Bundled Corpus/Quickstart/Reproducing Paper Results](https://github.com/McGill-NLP/latentlens). The software license is MIT, with Apache-2.0-derived Molmo code noted separately; model and corpus rights need their own review. [License](https://raw.githubusercontent.com/McGill-NLP/latentlens/main/LICENSE).

**Reviewer's interpretation and assumptions to test:**

- A plausible nearest phrase shows recoverable semantic similarity under a corpus and metric. It does not establish that the model used that feature for an answer, that the phrase exhausts the vector's content, or that the neighborhood is unique.
- Nearest neighbors exist for random/background vectors too. A score threshold and semantic evaluation need empirical nulls; cosine magnitude alone cannot establish interpretability.
- Comparing richer contextual reference sets with static vocabulary sets changes the search space. Corpus-size sweeps, unique-concept controls and equal candidate counts help separate representation quality from a larger opportunity to find a plausible word.
- Reusing a corpus across exploration and scoring can bias apparent coverage. Keep source-image overlap, caption near-duplicates, domain coverage and corpus-selection decisions explicit.
- The reference layer need not equal the image layer. Force same-layer retrieval as a comparator before interpreting a layer-alignment heatmap.
- Global/abstract matches should be separate from patch-local concrete matches. A beautiful object caption assigned to background may satisfy the published broad rubric while failing a local-grounding question.
- Phrase retrieval on cached arrays can be a faithful core-method exercise; constructing synthetic arrays with semantic labels already embedded is only a method illustration. Neither reproduces the paper's model-level result.

## Fit to existing ARENA, without new topics

The closest original comparator inspected was [original 1.2 Intro to Mech Interp](../../../../chapter1_transformer_interp/exercises/part2_intro_to_mech_interp/1.2_Intro_to_Mech_Interp_exercises.ipynb): concrete attention visualizations, student-written detectors/composition scores, immediate checks, random-initialization baselines and targeted ablations. Follow that progression from visible pattern to quantitative claim to a discriminating intervention. No original file was changed.

| Existing section | Proposed use | Boundary to preserve |
| --- | --- | --- |
| [7.1 Lenses and Patchscopes](../../../../chapter7_activation_to_language/instructions/pages/01_[7.1]_Logit_Lens_Tuned_Lens_and_Patchscopes.md) | Add LatentLens as a contextual-reference comparator in its future review plan; predict which descriptions differ from final-vocabulary decoding. | Keep the current pilot question coherent; no new chapter or model family. A training-free retrieval lens and trained tuned lens answer different questions. |
| [12.4 Visual Token Flow](../../../../chapter12_vlm_interpretability/instructions/pages/04_[12.4]_Visual_Token_Flow_in_Real_VLMs.md), using 12.3 prerequisites | Best home for future image-patch descriptions plus object/background interventions. | Its current organism is explicitly analytic and real adapters pending. A LatentLens bridge cannot silently qualify native VLM hooks. |
| [11.2 Representation Alignment](../../../../chapter11_representation_geometry/instructions/pages/02_[11.2]_CCA_SVCCA_and_Representation_Alignment.md) | Crosslink cosine-neighbor geometry, layer mismatch, scaling and causal-transport limits. | Alignment/similarity is not causal equivalence. Do not treat a visually attractive projection as validation. |
| [15.4 Adapter proxies](../../../../chapter15_peft_misalignment/instructions/pages/04_[15.4]_LoRA_Phase_Transitions_and_Misalignment_Proxies.md) | Mathematical bridge from output-direction projection to targeted weight edits. | Current task is a planted harmless proxy. Sparse manual edits are not LoRA training or real misalignment qualification. |
| [17.3 Fine-tuning diffs](../../../../chapter17_training_dynamics/instructions/pages/03_[17.3]_Fine-Tuning_Diffs_and_Capability_Emergence.md) | Best home for a weight-reading/editing research case: compare structural differences, then measure changed behavior. | Existing checkpoints are constructed, with native studies pending. No inferred training dynamics from a single before/after pair. |
| 17.2 Induction Head Emergence | Optional connection to the Sparks middle-block head-identification experiment. | Identifying heads in trained networks and measuring their emergence over checkpoints are separate studies. |
| [18.2 Intrinsic structures](../../../../chapter18_intrinsic_interpretability/instructions/pages/02_[18.2]_Sparse_and_Modular_Training_KANs_and_Sparse_Trees.md) | Optional question: does a hand-compiled small network become more legible than a trained equivalent? | Handcrafting dense weights does not establish modularity, intrinsic interpretability, sparse training or a KAN result. |

Recommendation: plan LatentLens as a **7.1→12.4 bridge**, and Sparks as a **17.3 case study with a 15.4 projection bridge**. They should complement pilot repairs, not expand the topic roster.

## Proposed learner scaffold A: “Does the nearest phrase describe this patch?”

Current learner failure: a list of plausible words can be mistaken for a causal explanation, or cached by a hidden utility without exposing the retrieval method.

Target claim: students can retrieve and compare contextual neighbors, detect corpus/layer/background failures and distinguish description from behavioral use. This is a proposal; no result has been generated.

1. **Cold open and forecast.** Show one image, one marked object patch and one background patch, without descriptions. Student forecasts which lens will be patch-local and records confidence. Reuse a pinned existing VLM only after a separate run approval; begin with a small licensed cache if one already exists and its provenance is verified.
2. **Implement the core.** Student writes `normalize_rows`, `contextual_topk(query, bank, metadata, k)`, `gather_context_window`, and `layer_alignment_counts`. Preserve model revision, hook boundary, token offsets, image/processor identity, corpus source and layer in every index entry. Helpers may load arrays, but must not hide similarity, selection or layer aggregation.
3. **Immediate tests and expected output.** Hand-computable vectors establish cosine ordering, invariance to positive query rescaling, zero-vector policy, stable ties, exact chunked/brute-force top-k parity and preservation of sentence/token/layer identities. Include a fixture where the nearest contextual vector comes from another layer. Expected output is named checks plus the actual neighbors, not just “passed.”
4. **Signature result.** Image-patch overlay with nearest words/context snippets, a side-by-side EmbeddingLens/LogitLens/contextual panel, and an image-layer × reference-layer neighbor heatmap. Use the same image patches for every method. Every display retains failed and background examples.
5. **Negative controls.** Norm-matched random queries, shuffled vector-to-caption metadata, unrelated corpus, same-layer-only search, reduced corpus with fixed seed and concept coverage, text-only versus image-conditioned observations, object versus background patches. A failure to separate any control is an interpretable negative result, not a reason to remove it.
6. **Causal bridge.** Only after authentic model execution is approved: replace/ablate the described object tokens and measure a predeclared image-question answer change, alongside background, unrelated donor and text-only controls. Agreement with a phrase does not guarantee intervention success; show mismatches.
7. **Try It Yourself.** Change the patch, layer, corpus domain and top-k; forecast which factor will cause a global label or missing concept before revealing neighbors. Provide help on tokenization and cosine conventions, interpretation and complete solution dropdowns.
8. **Changed-condition transfer.** Freeze retrieval choices on DEV images; predict behavior on a distinct visual domain and a caption corpus with the named concept removed. Public reference examples remain guided practice. Any later scored TEST set must be genuinely untouched and admitted separately.

**Illustration / replication / research extension:** a NumPy cosine fixture is an illustration; applying the paper's retrieval rule to provenance-checked real hidden-state caches reproduces the core operation; reproducing v5 headline rates additionally requires its exact VLMs, data, index and judge protocol. Replacing Visual Genome with the new package corpus, reducing the bank, or adding intervention tests is a declared extension. No numerical success forecast is justified for a reduced exercise.

## Proposed learner scaffold B: “Can a readable weight edit fix behavior?”

Current learner failure: a before/after headline hides architecture semantics, numerical operation, feedback budget and failure decomposition.

Target claim: students can implement and audit a targeted edit, test its behavioral mechanism and recognize when a structure-based prediction is invalid. The initial task should be harmless and exactly specified; it must not stand in for an unrun Qwen replication.

1. **Cold open and forecast.** Present a small network with an explicit formula and two candidate task rules. Ask what can be inferred without running it, and what evidence would distinguish the rules. Hold out task metadata; preserve architecture/input semantics openly.
2. **Student functions.** `read_mlp(weights, x)`, `output_direction_edit(W, u, strength, columns)`, `weight_diff_summary(before, after)`, `matched_random_edit`, `behavior_breakdown`, `permute_hidden_units`. Use column/row conventions explicitly and normalize `u` once. Keep numerical edits visible in notebook cells.
3. **Immediate tests and expected output.** An exact tiny MLP truth table; projection cancellation; sign invariance under `u→-u` as a positive identity control, since `u uᵀ` is unchanged; zero-strength identity; edit support/magnitude; same Frobenius norm for random controls; a hidden-unit permutation accompanied by inverse changes in the next layer preserving outputs. Count nonzero stored-byte changes separately from mathematically targeted parameters.
4. **Signature result.** Before/after weight heatmaps with edited support, task decision-boundary plots and a behavior-by-condition matrix. For a future authenticated looping-model study, add paired rescues/regressions, finished/truncated answer outcomes, generation lengths and accuracy versus edit strength/token budget. A single aggregate accuracy is inadequate.
5. **Controls.** Zero edit, same-support/same-norm random direction, wrong output columns, wrong layer, a signed-update contrast and a decoding-only intervention. The signed-update contrast replaces `ΔW` by `-ΔW` (equivalently reverses the strength coefficient), not `u` by `-u`; it is a counterfactual comparison with no guarantee of opposite behavior. Evaluate preservation of an independent benign task. Match sample counts and feedback budgets when comparing editors/optimizers; include editor inference and evaluation in cost.
6. **Try It Yourself.** Change an input convention, apply an exact function-preserving hidden-unit permutation, and forecast whether behavior, parameter appearance and weight-reading explanation each change. A representation-sensitive explainer can change despite identical network function; make that failure visible.
7. **Changed-condition transfer.** Freeze an edit on DEV; forecast a new task distribution and a different failure mechanism. A no-loop comparator is informative, but use matched grading/token-budget conditions before attributing differences to capability acquisition. Report all selected and rejected edit candidates and failed runs.
8. **Paper connection.** Separate one-shot decimal-text reading, code-generated weight compilation and adaptive feedback editing in the final learner explanation. Ask which mode their exercise actually tested. Include help, expected examples and complete student-visible solutions for every core function.

**Illustration / replication / research extension:** hand-authored networks on a finite domain demonstrate the algebra only. A faithful Sparks reading replication needs the exact released prompts/weights and an authorized editor call; replaying logs tests neither the editor nor generalization. A faithful edit evaluation needs pinned base+diff bytes, exact numerical reconstruction, decoding and graders, plus paired evaluation. Novel seeds, reparameterization controls, matched alternative editors and unseen failure classes are research extensions. No model inference is authorized by this note.

## Resource/access boundary and useful further reading

No runtime or storage budget is silently inferred from a tiny NumPy exercise. GPU-free editing can still require substantial GPU evaluation. Before any later run, freeze the artifact revisions, data roles, grading, query/sequence/token budget, walltime, memory and selection rules. Prefer existing licensed caches and one small approved native configuration. Benchmark any compilation or precision change against the numerical contract; upgrading software during a reproduction is a visible protocol change. Any cluster run must use normal/lowest QOS, smaller/unused hardware where adequate and at most two jobs without new permission.

LatentLens reported index costs and the package's ≥24 GB quickstart are not local measurements or a recommendation to request a large GPU. Model/corpus gating, image redistribution rights, caption overlap and allowed checkpoint caching remain to be verified. Sparks' editor access/pricing, model availability and reproducibility across current model revisions remain unknown here.

Three narrowly useful connections:

- **Tuned Lens**, Nora Belrose et al., [arXiv:2303.08112](https://arxiv.org/abs/2303.08112): per-layer trained affine decoding is an existing 7.1 comparator. Abstract/metadata read only; no new replication claim. Separate vocabulary-prediction quality from visual-token description quality.
- **Representations of Text and Images Align From Layer One**, Evžen Wybitul, Javier Rando, Florian Tramèr, Stanislav Fort, [arXiv:2601.08017v1](https://arxiv.org/abs/2601.08017): synthesis-based alignment is a useful independent operationalization to discuss alongside retrieval. Abstract/metadata read only; synthesis would require new execution authority and is not part of this plan.
- Benno Krojer's [Behind the Scenes: LatentLens](https://bennokrojer.com/behind-the-scenes-latentlens.html), July 2026: a first-person account of failed assumptions and toy-data detours, useful for the learner's “which simpler check should we try first?” reflection. It is a research-process source, not independent validation of the main result.

## Open decision queue for the coordinating researcher

No user answer is fabricated. These are proposed questions for the single coordinating interviewer, not independent permission requests from a worker.

| ID | Status / dependency | Decision and recommendation |
| --- | --- | --- |
| WL-Q01 | Factual reconciliation resolved for the scoped 97-record Inbox; wider-library match unresolved and owned by coordinator | Sparks matches INB027. No exact LatentLens paper title appears in scope; INB056/INB061 are retained as untitled webpages. The verified LatentLens paper is an explicit external supplement. Preserve local metadata and v1/v5 distinctions. |
| WL-Q02 | Ready; independent of a wider-library match | Curriculum placement. Recommendation: 7.1→12.4 contextual-readout bridge, 17.3 weight-edit case with 15.4 math crosslink; no added topics. |
| WL-Q03 | Deferred until a future execution request | Desired evidence layer. Recommendation: a transparent real-cache core-method exercise first; faithful full-paper replication only under an explicit resource/data/protocol allowance. |
| WL-Q04 | Deferred; depends on WL-Q03 and artifact audit | Paper corpus versus current package corpus. Recommendation: preserve a named Visual Genome paper track and label the concepts-corpus variant as an extension. |
| WL-Q05 | Deferred; depends on WL-Q03 | Is the future weight-edit objective completion rescue or new capability acquisition? Recommendation: define completion rescue first and require matched no-loop/decoding controls before stronger claims. |

## Handoff and checks

Changed files: this new planning note and worker-owned receipts only (subsequently relocated byte-for-byte into the ignored research evidence tree; the local custody manifest retains their original paths and hashes). Original ARENA: preserved; original notebook read as comparator, no writes. Student-facing improvements, signatures, controls and play cells above are specified proposals, not implemented outputs. Checks: exact Sparks and untitled-page metadata read from the 97-record API inventory; source identities/versions and cited sections inspected; local chapter targets read; Markdown local links checked; no numerical or learner acceptance test claimed. Limitations: wider-library LatentLens identity search remains with the coordinator; no PDF-page audit, byte-level edit audit, live code execution, run qualification or human learner acceptance.

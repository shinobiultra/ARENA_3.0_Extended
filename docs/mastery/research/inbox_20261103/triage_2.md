# Inbox triage: INB034–INB066

**Draft planning, 3 November 2026.** All 33 records appear once in the matrix and once in the detailed ledger. Source snapshot: lead-supplied version 10797, 97-record Inbox. Public bibliographic identities only; no private library identifiers, annotations, tags or raw exports are reproduced.

Primary sources were checked for each identity. Research records received at least metadata/abstract screening; selected method reads are stated explicitly. The edited proceedings volume received metadata/contents screening, and the two author-page records refer to one essay. This is initial triage, not a full-text evidence review or a replication. All methods, figures, prompts and controls below are proposals. No model/data downloads or computation were performed; no Zotero or original ARENA changes were made.

Target lesson numbers follow the current syllabus: **8.3 is AtP*; 8.4 is ACDC and Circuit Metrics.** These proposals preserve the existing lesson inventory and do not promote readiness. Priorities mean 1 = strongest existing-lesson fit, 2 = optional follow-on, 3 = background/deferred. Replication tiers describe a possible future exercise, not work executed or authorized here.

## Six most useful controlled additions

**INB038 — Do Sparse Autoencoders Capture Concept Manifolds?** ([primary](https://arxiv.org/abs/2604.28119)). The source distinguishes a coherent small group of dictionary atoms from atoms that tile local regions, and warns that reconstruction alone does not establish coherent manifold recovery. Our proposed addition is a circle-plus-line exercise inside 11.6, linked to 6.3: students implement known manifold sampling, nonnegative sparse codes, restricted reconstruction and angle-indexed tuning curves. They predict whether a dictionary with lower error must have better compact capture before revealing the two plots. Hold reconstruction budget and sample count fixed, introduce a correlated nuisance variable, and show that permuting angle labels destroys semantic organization while preserving aggregate reconstruction. The signature result is a population tuning heatmap with equal-error dictionaries that have different restricted capture. Start with CPU illustration; a real SAE study needs separate checkpoint, activation, geometry and evaluation split choices. No claim that the toy reproduces the paper's LLM results. Read: selected definitions and synthetic experiment caption, not a complete proof audit.

**INB040 — The Geometric Wall: Manifold Structure Predicts Layerwise Sparse Autoencoder Scaling Laws** ([primary](https://arxiv.org/abs/2605.09887)). The paper's key methodological detail is that most layers expose only two shared dictionary widths, so their apparent width exponents come from a no-floor fit; richer showcase layers support separate floor calibration. Our proposed lab in 6.2/11.6 asks whether the available grid identifies the claimed asymptote. Students implement log-error surface fitting on a supplied synthetic table with known floor, profile fits over candidate floors, and compare depth-only against geometry-conditioned predictions on held-out layers. Plot the equally plausible curves outside the observed width range and ask which observation would distinguish them. Shuffle geometry, separate geometry samples from reconstruction samples, and report extrapolation uncertainty. The signature is an identifiability plot, not an assertion of a universal geometric wall. A bounded real study of released summary tables could follow an access audit; rebuilding hundreds of SAEs is unnecessary for this lesson. Read: selected fitting and geometry-estimation methods and code-availability statement.

**INB044 — MIB: A Mechanistic Interpretability Benchmark** ([primary](https://arxiv.org/abs/2504.13151)). MIB makes two distinctions immediately useful to the existing circuit lessons: localization of a circuit versus a causal variable, and recovering positive task contribution versus matching the full model including negative contributions. Our proposed 8.4 addendum implements normalized faithfulness and integrated CPR/CMD on an exact graph containing a suppressive component. Before seeing results, students predict whether deleting that component improves the task while harming model fidelity. They then draw both size-indexed curves and compare exact, random and learned rankings at matched circuit size. Full and empty graphs expose normalization denominators; counterfactual examples need a capability check before localization is meaningful. The signature is a circuit that outperforms the full model yet has worse CMD. This CPU illustration teaches estimands; a real MIB run requires its published splits and protocols. Historical leaderboard claims from other papers are not current independently verified rankings. Read: selected circuit metrics, model capability and benchmark split descriptions.

**INB047 — A Geometric Unification of Concept Learning with Concept Cones** ([primary](https://arxiv.org/abs/2512.07355)). The paper compares nonnegative concept dictionaries by conic recovery, while treating CBM concepts as human-plausible reference directions rather than guaranteed causal truth. Our proposed addition to 18.1 uses two tiny dictionaries spanning the same plane but different cones. Students implement projected-gradient nonnegative least squares, plot the feasible cones, and measure recovery of a held-out reference direction as the sparsity budget changes. An unrestricted least-squares baseline reveals why span agreement alone misses sign constraints. Rotate cones, include duplicated atoms and deliberately mislabel a reference concept; then separately intervene in a transparent classifier to see why geometric containment does not certify faithful concepts. Ask students to predict recovery before revealing the residual curve. This CPU exercise creates a clear bridge to 6.3 without replacing either lesson. A paper replication requires checking exact CBM/SAE variants and metrics first. Read: selected cone definition and operationalization, not the complete experimental appendix.

**INB048 — Model metamers reveal divergent invariances between biological and artificial neural networks** ([primary](https://www.nature.com/articles/s41593-023-01442-0)). The publisher describes optimizing input stimuli to match internal activations, then testing recognition rather than assuming activation agreement implies shared human invariance. Our proposed 12.7/11.2 addendum starts with a transparent many-to-one map so students implement the matching loss and gradient update and can see distinct inputs with identical features. A later, separately bounded real-model study could produce a reference/early/late gallery paired with feature residual and independent-model recognition. Students predict whether a perfect feature match forces recognizable content. Noise and near-reference initializations, a pixel-match control, and the identity-map counterexample clarify both optimization artifacts and the test's limits. The signature is low feature error together with changed recognizable content. Learner inspection is an observation, not a replication of the paper's controlled human recognition experiment; any human evaluation needs its own protocol. Publisher code/data pointers were read, but repositories and runtime were not audited. Read: selected methods, results, limitations and availability statements.

**INB058 — Evaluating the Causal Faithfulness of SHAP as a Candidate Attribution Method for SAE Features** ([primary](https://arxiv.org/abs/2610.17898)). The paper evaluates attribution to probe outputs against interventions read through the LM, and explicitly reports reconstruction, token-scope and feature-activity complications. Our proposed 16.8 addition makes the target mismatch a prediction task. Students implement a tiny known decoder plus nonlinear readout, calculate probe SHAP/IG, and plot attribution agreement next to actual signed and magnitude intervention effects. They repeat with reconstruction-only passes, last-token versus all-token edits and active-only candidate sets; matched-activity random features test whether ranking gains merely find nonzero activations. A declared family of tests and multiple-comparison correction accompanies insertion/deletion minimal pairs, with held-out examples reserved before ranking. The signature is high inter-method agreement beside poor causal ranking, followed by an explicit confound decomposition. The CPU mechanism illustration comes first; a bounded GPT-2/SAE replication is contingent on exact artifact and sampling audits. Read: selected methods 3.3-3.6, not a full reanalysis; the paper's findings remain reported evidence.

## All-record matrix

| ID | Public identity and primary source | Disposition / priority | Existing lesson | Concrete learner contribution |
|---|---|---|---|---|
| INB034 | [Arbitration Failure, Not Perceptual Blindness: How Vision-Language Models Resolve Visual-Linguistic Conflicts](https://arxiv.org/abs/2604.09364) | existing lesson addition / P1 | 12.5 Object Hallucination and Modality Arbitration; 7.1 Logit Lens | Implement visual-versus-prior logit trajectories and a patch dispatcher over token groups. |
| INB035 | [The Solvability of Interpretability Evaluation Metrics](https://arxiv.org/abs/2205.08696) | existing lesson addition / P1 | 16.8 SHAP vs Activation Patching; 8.4 ACDC and Circuit Metrics | Implement exhaustive subset scoring and beam search on a tiny transparent classifier. |
| INB036 | [Beyond the Linear Representation Hypothesis: Non-Linear Activation Steering in Text-to-Image Models](https://arxiv.org/abs/2610.06945) | optional replication / P2 | 13.3 Concept Directions and Diffusion LoRA; 11.6 Feature Manifolds | Fit a one-dimensional curve and compare its interpolation to an endpoint chord on known curved data. |
| INB037 | [World Modeling in Transformers](https://arxiv.org/abs/2609.21748) | existing lesson addition / P2 | 14.5 Maze Transformers and World Models; 5.5 State Tracking | Implement localization probes and counterfactual position swaps on a tiny documented maze fixture. |
| INB038 | [Do Sparse Autoencoders Capture Concept Manifolds?](https://arxiv.org/abs/2604.28119) | existing lesson addition / P1 | 11.6 Feature Manifolds; 6.3 Feature Splitting, Absorption and Dashboards | Construct a circle plus an independent line; implement sparse coding, restricted reconstruction and feature tuning curves. |
| INB039 | [A Mechanistic Account of Attention Sinks in GPT-2: One Circuit, Broader Implications for Mitigation](https://arxiv.org/abs/2604.14722) | existing lesson addition / P2 | 8.1 Activation Patching Refresher; 5.1 Gemma from Scratch | Decompose attention logits into query-bias, positional-MLP and key terms; ablate each separately. |
| INB040 | [The Geometric Wall: Manifold Structure Predicts Layerwise Sparse Autoencoder Scaling Laws](https://arxiv.org/abs/2605.09887) | optional replication / P1 | 6.2 Gemma Scope and Feature Steering; 11.6 Feature Manifolds | Fit log-error surfaces, calculate width exponents and test geometry-to-exponent regression on held-out layers. |
| INB041 | [Matryoshka attribution: Learning to attribute language model outputs to representations and weights](https://arxiv.org/abs/2609.25518) | optional replication / P2 | 8.4 ACDC and Circuit Metrics; 8.8 Benchmarking Circuit Discovery | Implement differentiable top-k masks on a planted finite computation graph and sample k during optimization. |
| INB042 | [Few-Step Generation via Data-Space Iteration](https://arxiv.org/abs/2610.12102) | background / P3 | 13.1 Diffusion Models and Cross-Attention Attribution | Trace a supplied toy denoiser under two explicit iteration rules. |
| INB043 | [Transfer between Modalities with MetaQueries](https://arxiv.org/abs/2504.06256) | background / P3 | 12.3 Mini VLM from Scratch; 13.5 Diffusion Transformers and ConceptAttention | Implement a query-to-latent adapter in a tiny supplied interface fixture. |
| INB044 | [MIB: A Mechanistic Interpretability Benchmark](https://arxiv.org/abs/2504.13151) | existing lesson addition / P1 | 8.4 ACDC and Circuit Metrics; 8.8 Benchmarking Circuit Discovery; 11.3 Causal Abstraction | Implement size-indexed circuit curves, normalized faithfulness, integrated CPR and CMD on an exact graph. |
| INB045 | [Exploiting Vision Encoder Vulnerabilities for Universal Adversarial Perturbations on Large Vision-Language Models](https://arxiv.org/abs/2412.08108) | optional replication / P2 | 12.2 CLIP Features, SAEs and Typographic Attacks | Implement a component-selection audit on authorized benign image-category mismatches. |
| INB046 | [Local Support Learning](https://arxiv.org/abs/2610.02126) | optional replication / P2 | 15.2 LoRA vs Full Fine-Tuning | Implement a Gaussian likelihood gate multiplying a rank-one update on two synthetic input regions. |
| INB047 | [A Geometric Unification of Concept Learning with Concept Cones](https://arxiv.org/abs/2512.07355) | existing lesson addition / P1 | 18.1 Concept Bottlenecks, Backpacks and IIT; 6.3 Feature Splitting | Implement nonnegative least-squares reconstruction of reference concept directions and compare unrestricted spans. |
| INB048 | [Model metamers reveal divergent invariances between biological and artificial neural networks](https://www.nature.com/articles/s41593-023-01442-0) | existing lesson addition / P1 | 12.7 Dense Vision Features - DINO and SAM; 11.2 Representation Alignment | Implement feature-matching loss and input-gradient steps first on a transparent non-injective map. |
| INB049 | [Adaptive Probe-based Steering for Robust LLM Jailbreaking](https://arxiv.org/abs/2605.20286) | background / P3 | 9.1 Refusal Directions and Representation Engineering | Derive normalized steering magnitude on a benign sentiment-direction fixture. |
| INB050 | [Vision-Default, Prior-Override: Causal Mechanisms of Perception-Knowledge Conflict in Vision-Language Models](https://arxiv.org/abs/2606.28273) | existing lesson addition / P1 | 12.5 Object Hallucination and Modality Arbitration | Implement head-wise noising and direct answer-logit projection with a held-out conflict prompt set. |
| INB051 | [AI Models Can Provably Hide Arbitrary Capabilities](https://arxiv.org/abs/2610.13928) | background / P3 | 9.5 White-Box Monitors and Red-Teaming with Internals; 10.1 Capstone Research Sprint | Work through a harmless finite trigger table to distinguish absence from inaccessible behavior. |
| INB052 | [AnyAttack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models](https://jiamingzhang94.github.io/anyattack/) | background / P3 | 12.2 CLIP Features, SAEs and Typographic Attacks | Compare targeted transfer estimands and split diagrams against per-image optimization. |
| INB053 | [Corrupted but Correct: Why Vision-Language Models Lie to Themselves Internally](https://arxiv.org/abs/2610.03445) | existing lesson addition / P2 | 12.5 Object Hallucination and Modality Arbitration; 7.1 Logit Lens | Implement teacher-forced log-probability and free-generation scoring with explicit prefix conditions. |
| INB054 | [Mechanisms of Prompt-Induced Hallucination in Vision–Language Models](https://aclanthology.org/2026.acl-long.1941/) | existing lesson addition / P1 | 12.5 Object Hallucination and Modality Arbitration | Implement count-consistent versus overcounted prompt pairs and head ablation scoring. |
| INB055 | [Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations](https://arxiv.org/abs/2609.19101) | optional replication / P2 | 9.5 White-Box Monitors and Red-Teaming with Internals | Implement training-only mean-difference direction, threshold calibration and held-out ROC/precision-recall scoring. |
| INB056 | [Better late than never: Getting into interpretability in 2025](https://bennokrojer.com/interp.html) | background / P3 | 10.1 Capstone Research Sprint | Write one testable claim at behavioral, algorithmic and implementation levels for an existing capstone. |
| INB057 | [Computer Security. ESORICS 2025 International Workshops: ANUBIS 2025, SECAI 2025, SecAssure 2025, STMUS 2025, Toulouse, France, September 22–24, 2025, Revised Selected Papers, Part II](https://link.springer.com/book/10.1007/978-3-032-16092-8) | defer/out of scope / P3 | 10.1 Capstone Research Sprint: bibliographic screening only | No method selected; first identify an exact chapter and its question. |
| INB058 | [Evaluating the Causal Faithfulness of SHAP as a Candidate Attribution Method for SAE Features](https://arxiv.org/abs/2610.17898) | existing lesson addition / P1 | 16.8 SHAP vs Activation Patching; 6.2 Gemma Scope and Feature Steering | Implement signed and magnitude correlations against interventions, active-only stratification and a declared multiple-testing family. |
| INB059 | [DepthBench: Measuring How Residual Connections Enable More Computational Depth](https://arxiv.org/abs/2609.32534) | background / P3 | 17.4 Training Dynamics Across Architectures | Compute parameter counts and effective-depth diagnostics on supplied finite network traces. |
| INB060 | [Same Task, Different Circuits: Disentangling Modality-Specific Mechanisms in VLMs](https://arxiv.org/abs/2506.09047) | existing lesson addition / P1 | 12.4 Visual Token Flow in Real VLMs; 12.5 Modality Arbitration | Implement late-to-early state injection and compare modality circuit overlap on matched tasks. |
| INB061 | [Better late than never: Getting into interpretability in 2025](https://bennokrojer.com/interp.html) | duplicate / P3 | 10.1 Capstone Research Sprint: refer to INB056 | No independent lab; reuse INB056 framing exercise. |
| INB062 | [Vision-Language Models Create Cross-Modal Task Representations](https://arxiv.org/abs/2410.22330) | existing lesson addition / P1 | 12.4 Visual Token Flow in Real VLMs; 11.2 Representation Alignment | Extract task summaries at the stated token/layer and patch across held-out modality-task pairs. |
| INB063 | [When Models Manipulate Manifolds: The Geometry of a Counting Task](https://arxiv.org/abs/2601.04480) | existing lesson addition / P1 | 11.6 Feature Manifolds; 8.7 Circuit Tracing with Attribution Graphs | Implement character-count accumulation and controlled coordinate interventions on a transparent linebreaking fixture. |
| INB064 | [Chameleon: Mixed-Modal Early-Fusion Foundation Models](https://arxiv.org/abs/2405.09818) | background / P3 | 13.6 Autoregressive Image Tokens; 12.3 Mini VLM from Scratch | Implement a mixed token mask and next-token loss accounting on a supplied sequence. |
| INB065 | [Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling](https://arxiv.org/abs/2501.17811) | background / P3 | 13.6 Autoregressive Image Tokens; 12.3 Mini VLM from Scratch | Decompose a claimed gain into a hypothetical factorial ablation design. |
| INB066 | [SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers](https://arxiv.org/abs/2410.10629) | background / P3 | 13.5 Diffusion Transformers and ConceptAttention; 5.6 Linear Attention | Implement token-count and attention-memory accounting across resolutions/compression factors. |

## Detailed ledger

The “source fact” sentence reports the source, while every design field is our proposal. No proposed signature figure below is represented as an existing paper figure. UNKNOWN is retained where access, compute, code or data was not checked.

### INB034 — Arbitration Failure, Not Perceptual Blindness: How Vision-Language Models Resolve Visual-Linguistic Conflicts

- **Identity / version:** verified title and authors on primary source. arXiv v2, 2026-04-13.
- **Primary sources:** [source 1](https://arxiv.org/abs/2604.09364); [source 2](https://arxiv.org/html/2604.09364v2).
- **Read:** abstract plus selected methods: section 6, Figure 3 caption and token-type decomposition.
- **Decision:** existing lesson addition, P1; 12.5 Object Hallucination and Modality Arbitration; 7.1 Logit Lens.
- **Source fact:** Reports preserved visual encoding despite wrong answers and stronger effects of full-sequence than last-token patches.
- **Why / intuition role:** Direct bridge from decoded layer trajectories to causal arbitration, with patch scope as an experimental variable.
- **Student method role:** Implement visual-versus-prior logit trajectories and a patch dispatcher over token groups.
- **Signature figure / visualization role:** Layerwise visual/prior logit curves beside patch flip rates by token group.
- **Prediction role:** If visual information is decodable early, must a last-token patch recover a visually correct answer?
- **Controls and baselines:** Last token, image tokens, text tokens, full sequence; same-image sham; matched image sizes and prompt lengths.
- **Replication role:** bounded real study; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB035 — The Solvability of Interpretability Evaluation Metrics

- **Identity / version:** verified title and authors on primary source. arXiv v2, 2023-02-03; EACL Findings 2023 noted.
- **Primary sources:** [source 1](https://arxiv.org/abs/2205.08696).
- **Read:** metadata and abstract.
- **Decision:** existing lesson addition, P1; 16.8 SHAP vs Activation Patching; 8.4 ACDC and Circuit Metrics.
- **Source fact:** Optimizes comprehensiveness/sufficiency explanations by beam search and questions their use as independent quality metrics.
- **Why / intuition role:** A small counterexample makes metric optimization versus explanation validation visible before circuit benchmarking.
- **Student method role:** Implement exhaustive subset scoring and beam search on a tiny transparent classifier.
- **Signature figure / visualization role:** Deletion/sufficiency curves for metric-optimized, attribution and random subsets.
- **Prediction role:** Will the explanation selected for best deletion score also recover the planted causal variables?
- **Controls and baselines:** Exact subset optimum, random same-size subsets, planted irrelevant correlates; separate selection metric from validation metric.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** Abstract reports solvex package release; installability, data and dependencies unchecked.

### INB036 — Beyond the Linear Representation Hypothesis: Non-Linear Activation Steering in Text-to-Image Models

- **Identity / version:** verified title and authors on primary source. arXiv v1, 2026-10-03.
- **Primary sources:** [source 1](https://arxiv.org/abs/2610.06945).
- **Read:** metadata and abstract.
- **Decision:** optional replication, P2; 13.3 Concept Directions and Diffusion LoRA; 11.6 Feature Manifolds.
- **Source fact:** Proposes KANSteer curves through intermediate concept states for text-to-image steering.
- **Why / intuition role:** Useful later counterexample to straight-line interpolation; a new standalone KAN topic is unnecessary.
- **Student method role:** Fit a one-dimensional curve and compare its interpolation to an endpoint chord on known curved data.
- **Signature figure / visualization role:** Matched endpoint chord and curve, with intermediate attribute labels and off-path error.
- **Prediction role:** Can accurate endpoint steering still miss every meaningful intermediate state?
- **Controls and baselines:** Straight chord, piecewise linear curve, shuffled intermediate order, held-out intermediate states.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB037 — World Modeling in Transformers

- **Identity / version:** verified title and authors on primary source. arXiv v1, 2026-09-18.
- **Primary sources:** [source 1](https://arxiv.org/abs/2609.21748).
- **Read:** metadata and abstract.
- **Decision:** existing lesson addition, P2; 14.5 Maze Transformers and World Models; 5.5 State Tracking.
- **Source fact:** Analyzes TaxiGPT localization failures as interference among superposed intersection features.
- **Why / intuition role:** Distinguishes possessing state information from using it correctly during navigation.
- **Student method role:** Implement localization probes and counterfactual position swaps on a tiny documented maze fixture.
- **Signature figure / visualization role:** Maze with true versus decoded locations and legal-move errors before/after swaps.
- **Prediction role:** Can a failed route coexist with accurate location decoding, and what intervention separates the accounts?
- **Controls and baselines:** Held-out routes, randomized locations, same-affordance versus different-affordance swaps.
- **Replication role:** bounded real study; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB038 — Do Sparse Autoencoders Capture Concept Manifolds?

- **Identity / version:** verified title and authors on primary source. arXiv v1, 2026-04-30.
- **Primary sources:** [source 1](https://arxiv.org/abs/2604.28119); [source 2](https://arxiv.org/html/2604.28119v1).
- **Read:** abstract plus selected methods: sections 3-4.1, Definitions 1-3 and synthetic Figure 4 caption.
- **Decision:** existing lesson addition, P1; 11.6 Feature Manifolds; 6.3 Feature Splitting, Absorption and Dashboards.
- **Source fact:** Distinguishes compact subspace capture from local tiling and dilution; uses restricted reconstruction and coactivation structure.
- **Why / intuition role:** Turns the familiar one-feature-one-concept picture into a falsifiable population-code lesson.
- **Student method role:** Construct a circle plus an independent line; implement sparse coding, restricted reconstruction and feature tuning curves.
- **Signature figure / visualization role:** Manifold angle versus atom activation heatmap, beside reconstruction and restricted-capture curves.
- **Prediction role:** Can two dictionaries have equal reconstruction error while only one uses a compact coherent group for the whole circle?
- **Controls and baselines:** Equal reconstruction budget, rotated circle, permuted angles, correlated nuisance variable; train/held-out manifold positions.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB039 — A Mechanistic Account of Attention Sinks in GPT-2: One Circuit, Broader Implications for Mitigation

- **Identity / version:** verified arXiv identity from DOI; stored Semantic Scholar URL is secondary. arXiv v1, 2026-04-16.
- **Primary sources:** [source 1](https://arxiv.org/abs/2604.14722).
- **Read:** metadata and abstract.
- **Decision:** existing lesson addition, P2; 8.1 Activation Patching Refresher; 5.1 Gemma from Scratch.
- **Source fact:** Attributes GPT-2 sinks to one interacting circuit but notes sinks persist in architectures missing its components.
- **Why / intuition role:** Excellent warning against exporting a model-specific head story into an architectural necessity claim.
- **Student method role:** Decompose attention logits into query-bias, positional-MLP and key terms; ablate each separately.
- **Signature figure / visualization role:** First-position attention and task loss under component interventions.
- **Prediction role:** Does eliminating this circuit in GPT-2 imply all models without its query bias cannot have attention sinks?
- **Controls and baselines:** Random components, BOS/token-position swaps, different architecture, downstream loss alongside attention mass.
- **Replication role:** bounded real study; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB040 — The Geometric Wall: Manifold Structure Predicts Layerwise Sparse Autoencoder Scaling Laws

- **Identity / version:** verified title and authors on primary source. arXiv v1, 2026-05-11.
- **Primary sources:** [source 1](https://arxiv.org/abs/2605.09887); [source 2](https://arxiv.org/html/2605.09887v1).
- **Read:** abstract plus selected methods: sections 2.4, 3.1-3.2 and code-availability statement.
- **Decision:** optional replication, P1; 6.2 Gemma Scope and Feature Steering; 11.6 Feature Manifolds.
- **Source fact:** Studies 844 SAE checkpoints across 68 layers; most-layer fits exclude an asymptotic floor because width coverage is limited.
- **Why / intuition role:** Scaling-law fitting is more valuable here as an identifiability and held-out prediction exercise than as a new universal law.
- **Student method role:** Fit log-error surfaces, calculate width exponents and test geometry-to-exponent regression on held-out layers.
- **Signature figure / visualization role:** Observed versus predicted width exponents with depth-only comparator; floor-fit profiles versus width coverage.
- **Prediction role:** Can two widths distinguish an irreducible error floor from a slowly decaying power law?
- **Controls and baselines:** Depth-only and shuffled-geometry regressions, separate geometry/error samples, richer-width synthetic ground truth.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** Paper links anonymous fitting/validation repository; contents, releases and data access unchecked.

### INB041 — Matryoshka attribution: Learning to attribute language model outputs to representations and weights

- **Identity / version:** verified title and authors on primary source. arXiv v1, 2026-09-22.
- **Primary sources:** [source 1](https://arxiv.org/abs/2609.25518).
- **Read:** metadata and abstract.
- **Decision:** optional replication, P2; 8.4 ACDC and Circuit Metrics; 8.8 Benchmarking Circuit Discovery.
- **Source fact:** Learns nested component subsets with sigmoid top-k masks and randomized sparsity during training.
- **Why / intuition role:** Adds a learned circuit ordering comparator without treating a reported leaderboard rank as durable evidence.
- **Student method role:** Implement differentiable top-k masks on a planted finite computation graph and sample k during optimization.
- **Signature figure / visualization role:** Faithfulness versus size for one learned nesting, independent masks and greedy ranking.
- **Prediction role:** Must the best circuit at size k be a subset of the best circuit at size k+1?
- **Controls and baselines:** Exact optimal subsets, same-size random masks, held-out task variants, matched optimization steps.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB042 — Few-Step Generation via Data-Space Iteration

- **Identity / version:** verified title and authors on primary source. arXiv v1, 2026-10-08.
- **Primary sources:** [source 1](https://arxiv.org/abs/2610.12102).
- **Read:** metadata and abstract.
- **Decision:** background, P3; 13.1 Diffusion Models and Cross-Attention Attribution.
- **Source fact:** Proposes repeated data-space refinement instead of a manually discretized flow trajectory.
- **Why / intuition role:** Useful objective/sampling contrast after core diffusion; not a reason to expand the method curriculum.
- **Student method role:** Trace a supplied toy denoiser under two explicit iteration rules.
- **Signature figure / visualization role:** Trajectory overlays and final error versus equal network-evaluation budget.
- **Prediction role:** Does using fewer named timesteps guarantee less computation at matched quality?
- **Controls and baselines:** Equal evaluations, fixed initialization, schedule-tuned comparator; distinguish illustration from distillation replication.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB043 — Transfer between Modalities with MetaQueries

- **Identity / version:** resolved missing URL/DOI via matching official arXiv title/authors and author repository. arXiv v1, 2025-04-08.
- **Primary sources:** [source 1](https://arxiv.org/abs/2504.06256); [source 2](https://github.com/facebookresearch/metaquery).
- **Read:** metadata and abstract.
- **Decision:** background, P3; 12.3 Mini VLM from Scratch; 13.5 Diffusion Transformers and ConceptAttention.
- **Source fact:** Uses learnable queries between a frozen multimodal LM and a diffusion decoder.
- **Why / intuition role:** Architecture diagram clarifies that unified outputs need not require jointly training every backbone.
- **Student method role:** Implement a query-to-latent adapter in a tiny supplied interface fixture.
- **Signature figure / visualization role:** Gradient-flow diagram and adapter output before/after query permutation.
- **Prediction role:** If the backbone is frozen, can its knowledge still affect a trained decoder interface?
- **Controls and baselines:** Frozen/random queries, query permutation, text-only conditions; exclude full model training.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** Official repository exposes code/configuration and describes data licensing; download completeness and runtime unchecked.

### INB044 — MIB: A Mechanistic Interpretability Benchmark

- **Identity / version:** verified title and authors on primary source. arXiv v2, 2025-06-09; ICML 2025.
- **Primary sources:** [source 1](https://arxiv.org/abs/2504.13151); [source 2](https://arxiv.org/html/2504.13151v2).
- **Read:** abstract plus selected methods: sections 2.3-3.1 and circuit metric equations; not full benchmark audit.
- **Decision:** existing lesson addition, P1; 8.4 ACDC and Circuit Metrics; 8.8 Benchmarking Circuit Discovery; 11.3 Causal Abstraction.
- **Source fact:** Separates circuit localization from causal-variable localization and CPR from CMD.
- **Why / intuition role:** Provides the right evaluation vocabulary for comparing circuit methods without conflating performance and matching model computation.
- **Student method role:** Implement size-indexed circuit curves, normalized faithfulness, integrated CPR and CMD on an exact graph.
- **Signature figure / visualization role:** Paired CPR/CMD curves where removing a suppressive component improves task score but worsens model matching.
- **Prediction role:** Can a circuit score above full-model performance while being a worse explanation of that model?
- **Controls and baselines:** Empty/full circuit, same-size random graphs, negative-contribution nodes, counterfactual task validity; declare zero-denominator behavior.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB045 — Exploiting Vision Encoder Vulnerabilities for Universal Adversarial Perturbations on Large Vision-Language Models

- **Identity / version:** verified title and authors on primary source. arXiv v3, 2026-06-29.
- **Primary sources:** [source 1](https://arxiv.org/abs/2412.08108).
- **Read:** metadata and abstract.
- **Decision:** optional replication, P2; 12.2 CLIP Features, SAEs and Typographic Attacks.
- **Source fact:** Reports universal perturbations targeting middle-layer value components of vision encoders and transfer through shared encoders.
- **Why / intuition role:** Can sharpen localization of vulnerability beyond final embeddings; keep it within controlled robustness study planning.
- **Student method role:** Implement a component-selection audit on authorized benign image-category mismatches.
- **Signature figure / visualization role:** Layer/component vulnerability heatmap beside transfer across encoder-sharing pairs.
- **Prediction role:** Should changing only the language decoder remove a perturbation learned against a shared vision encoder?
- **Controls and baselines:** Same-norm random perturbation, matched optimizer budgets, held-out images, shared/different encoder.
- **Replication role:** bounded real study; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB046 — Local Support Learning

- **Identity / version:** verified title and authors on primary source. arXiv v1, 2026-10-01.
- **Primary sources:** [source 1](https://arxiv.org/abs/2610.02126).
- **Read:** metadata and abstract.
- **Decision:** optional replication, P2; 15.2 LoRA vs Full Fine-Tuning.
- **Source fact:** Pairs an adapter with a GMM likelihood gate to localize updates to its training distribution.
- **Why / intuition role:** Concrete retention-versus-adaptation trade-off using existing PEFT mechanics.
- **Student method role:** Implement a Gaussian likelihood gate multiplying a rank-one update on two synthetic input regions.
- **Signature figure / visualization role:** Adaptation and retained-task error versus region overlap and gate threshold.
- **Prediction role:** When new and old activation distributions overlap, can a closed-world likelihood gate preserve both tasks?
- **Controls and baselines:** Ungated adapter, random gate, oracle domain labels, overlap sweep; evaluate old data only after selection.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB047 — A Geometric Unification of Concept Learning with Concept Cones

- **Identity / version:** verified title and authors on primary source. arXiv v2, 2026-06-07.
- **Primary sources:** [source 1](https://arxiv.org/abs/2512.07355); [source 2](https://arxiv.org/html/2512.07355v2).
- **Read:** abstract plus selected methods: section 2.2 cone definition and section 2.3 operationalization.
- **Decision:** existing lesson addition, P1; 18.1 Concept Bottlenecks, Backpacks and IIT; 6.3 Feature Splitting.
- **Source fact:** Unifies nonnegative concept dictionaries through conic hulls and measures recovery of human-defined CBM directions by SAE cones.
- **Why / intuition role:** Connects supervised and discovered concepts while keeping human plausibility separate from causal faithfulness.
- **Student method role:** Implement nonnegative least-squares reconstruction of reference concept directions and compare unrestricted spans.
- **Signature figure / visualization role:** 2D cone containment diagram plus recovery residual versus sparsity.
- **Prediction role:** Can two dictionaries span the same plane but differ in the human concept directions their nonnegative codes recover?
- **Controls and baselines:** Signed least squares, rotated/random cones, mislabeled reference concepts; test intervention effects separately.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB048 — Model metamers reveal divergent invariances between biological and artificial neural networks

- **Identity / version:** verified title and authors on primary source. Nature Neuroscience 26, 2017-2034 (2023); DOI 10.1038/s41593-023-01442-0.
- **Primary sources:** [source 1](https://www.nature.com/articles/s41593-023-01442-0); [source 2](https://github.com/jenellefeather/model_metamers_pytorch).
- **Read:** abstract and selected publisher methods/results/limitations and code/data availability excerpts.
- **Decision:** existing lesson addition, P1; 12.7 Dense Vision Features - DINO and SAM; 11.2 Representation Alignment.
- **Source fact:** Synthesizes stimuli matched at a model stage; late-stage model matches often fail human recognition, across vision and audition.
- **Why / intuition role:** A direct counterexample to equating representation match, benchmark similarity and shared perceptual invariance.
- **Student method role:** Implement feature-matching loss and input-gradient steps first on a transparent non-injective map.
- **Signature figure / visualization role:** Reference/early/late matched-image gallery beside feature error and independent recognition.
- **Prediction role:** If internal activations match almost perfectly, must a person recognize the same object?
- **Controls and baselines:** Noise and reference initializations, pixel-match control, independent model, anti-aliasing comparator; genuine human recognition requires separately designed evaluation.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** Publisher lists public code, human data, checkpoints and metamer viewer; Word-Speaker-Noise data requires author request; repository contents/runtime unchecked.

### INB049 — Adaptive Probe-based Steering for Robust LLM Jailbreaking

- **Identity / version:** verified arXiv identity from DOI; stored Semantic Scholar URL is secondary. arXiv v1, 2026-05-19; ICML 2026 noted.
- **Primary sources:** [source 1](https://arxiv.org/abs/2605.20286).
- **Read:** metadata and abstract.
- **Decision:** background, P3; 9.1 Refusal Directions and Representation Engineering.
- **Source fact:** Adapts steering strengths from contrastive activation statistics and reports stronger probe-based attacks.
- **Why / intuition role:** Useful caveat about tuning and robustness of refusal direction lessons; no new jailbreak implementation needed.
- **Student method role:** Derive normalized steering magnitude on a benign sentiment-direction fixture.
- **Signature figure / visualization role:** Behavior shift and unrelated-task degradation versus normalized versus fixed strength.
- **Prediction role:** Can improved target shift simply come from larger perturbation norms?
- **Controls and baselines:** Equal perturbation norm, random direction, held-out templates and unrelated-task loss.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** Abstract links attack code; code/data contents and controlled reproducibility unchecked.

### INB050 — Vision-Default, Prior-Override: Causal Mechanisms of Perception-Knowledge Conflict in Vision-Language Models

- **Identity / version:** verified title and authors on primary source. arXiv v1, 2026-06-26.
- **Primary sources:** [source 1](https://arxiv.org/abs/2606.28273).
- **Read:** metadata and abstract.
- **Decision:** existing lesson addition, P1; 12.5 Object Hallucination and Modality Arbitration.
- **Source fact:** Reports sparse late attention heads necessary for prior overrides and distinguishes routing from writing heads.
- **Why / intuition role:** Pairs well with INB034: compare component granularity rather than assume a single generic vision-versus-language story.
- **Student method role:** Implement head-wise noising and direct answer-logit projection with a held-out conflict prompt set.
- **Signature figure / visualization role:** Head-by-layer causal heatmap with visual/prior answer transition matrix.
- **Prediction role:** Would removing prior-override heads help equally on visually and prior-grounded baseline trials?
- **Controls and baselines:** Same-size random heads, sham ablation, congruent images, prompt paraphrases; test attention routing separately from output writing.
- **Replication role:** bounded real study; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB051 — AI Models Can Provably Hide Arbitrary Capabilities

- **Identity / version:** verified title and authors on primary source. arXiv v2, 2026-10-14.
- **Primary sources:** [source 1](https://arxiv.org/abs/2610.13928).
- **Read:** metadata and abstract.
- **Decision:** background, P3; 9.5 White-Box Monitors and Red-Teaming with Internals; 10.1 Capstone Research Sprint.
- **Source fact:** Constructs hidden computations with cryptographic hardness assumptions and tests elicitation methods on a small proxy task.
- **Why / intuition role:** Boundary reading: failure to elicit does not imply absence; worst-case construction is not evidence ordinary models hide such circuits.
- **Student method role:** Work through a harmless finite trigger table to distinguish absence from inaccessible behavior.
- **Signature figure / visualization role:** Coverage diagram of tested inputs and untested trigger conditions.
- **Prediction role:** What does a negative probe result rule out without a bound on the trigger space?
- **Controls and baselines:** Known present/absent behavior; explicit query budget and assumptions; do not claim toy example replicates cryptographic theorem.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB052 — AnyAttack: Towards Large-scale Self-supervised Adversarial Attacks on Vision-language Models

- **Identity / version:** resolved title via author project and CVPR paper; current arXiv title differs; publication is conference paper, not journal. CVPR 2025 published title; arXiv title variant.
- **Primary sources:** [source 1](https://jiamingzhang94.github.io/anyattack/); [source 2](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_Anyattack_Towards_Large-scale_Self-supervised_Adversarial_Attacks_on_Vision-language_Models_CVPR_2025_paper.pdf); [source 3](https://arxiv.org/abs/2410.05346).
- **Read:** private stored abstract plus primary project summary and CVPR title/abstract.
- **Decision:** background, P3; 12.2 CLIP Features, SAEs and Typographic Attacks.
- **Source fact:** Published AnyAttack framework trains a transferable adversarial generator using large-scale self-supervision.
- **Why / intuition role:** Threat-model comparison only: faithful pretraining is far beyond a short learner exercise.
- **Student method role:** Compare targeted transfer estimands and split diagrams against per-image optimization.
- **Signature figure / visualization role:** Targeted-success transfer matrix with explicit image and target hold-outs.
- **Prediction role:** Does success on one encoder establish transfer to a different encoder or merely another decoder?
- **Controls and baselines:** Random matched-norm noise, per-image optimization at matched budget; source/target separation.
- **Replication role:** faithful paper replication; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** Author project links code/models; contents, training data accessibility and weights unchecked.

### INB053 — Corrupted but Correct: Why Vision-Language Models Lie to Themselves Internally

- **Identity / version:** verified title and authors on primary source. arXiv v2, 2026-10-27.
- **Primary sources:** [source 1](https://arxiv.org/abs/2610.03445).
- **Read:** metadata and abstract.
- **Decision:** existing lesson addition, P2; 12.5 Object Hallucination and Modality Arbitration; 7.1 Logit Lens.
- **Source fact:** Reports teacher-forced target-loss versus free-generation dissociation and flags circularity in one probe estimate.
- **Why / intuition role:** Excellent exercise in choosing the behavioral readout before interpreting a successful optimization.
- **Student method role:** Implement teacher-forced log-probability and free-generation scoring with explicit prefix conditions.
- **Signature figure / visualization role:** Teacher-forced loss versus generated target success, colored by first-token outcome.
- **Prediction role:** Can near-zero target-caption training loss coexist with never generating the target caption?
- **Controls and baselines:** Matched prefixes, clean images, unconditioned decoding, independent probe split; audit invariant ranks before attributing a mechanism.
- **Replication role:** bounded real study; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB054 — Mechanisms of Prompt-Induced Hallucination in Vision–Language Models

- **Identity / version:** verified ACL title/authors/DOI; conference paper rather than unqualified journal article. ACL 2026 conference paper; DOI 10.18653/v1/2026.acl-long.1941.
- **Primary sources:** [source 1](https://aclanthology.org/2026.acl-long.1941/).
- **Read:** metadata and abstract.
- **Decision:** existing lesson addition, P1; 12.5 Object Hallucination and Modality Arbitration.
- **Source fact:** Studies overcounting prompts and model-specific heads whose ablation reduces prompt-induced counting hallucinations.
- **Why / intuition role:** A small count-controlled image fixture can make prompt copying versus visual correction visible.
- **Student method role:** Implement count-consistent versus overcounted prompt pairs and head ablation scoring.
- **Signature figure / visualization role:** Actual count by stated count response matrix before/after ablation.
- **Prediction role:** Will a head identified on low object counts also correct overcounting at higher counts?
- **Controls and baselines:** Congruent prompts, understated counts, random heads, object-layout swaps and held-out counts.
- **Replication role:** bounded real study; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB055 — Monitoring and Discovering Reward Hacking with Internal Representations during LLM Evaluations

- **Identity / version:** verified title and authors on primary source. arXiv v2, 2026-10-09.
- **Primary sources:** [source 1](https://arxiv.org/abs/2609.19101).
- **Read:** metadata and abstract.
- **Decision:** optional replication, P2; 9.5 White-Box Monitors and Red-Teaming with Internals.
- **Source fact:** Reports difference-of-means reward-hacking monitors and comparisons at matched false-positive rate.
- **Why / intuition role:** Fits existing monitor curriculum if issue/task identity leakage and definition of hacking are addressed.
- **Student method role:** Implement training-only mean-difference direction, threshold calibration and held-out ROC/precision-recall scoring.
- **Signature figure / visualization role:** Matched-FPR detection curves by held-out environment, with lead-time distribution.
- **Prediction role:** If probe positives are reviewed only after scoring, can discovered cases provide an unbiased precision estimate?
- **Controls and baselines:** Task/rollout-group splits, shuffled labels, behavior-only baseline, negative-control tasks, blinded adjudication.
- **Replication role:** bounded real study; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB056 — Better late than never: Getting into interpretability in 2025

- **Identity / version:** missing title resolved to Better late than never: Getting into interpretability in 2025; same URL as INB061; not a LatentLens paper record. Author webpage read 2026-11-03; essay refers to 2025.
- **Primary sources:** [source 1](https://bennokrojer.com/interp.html).
- **Read:** author essay read.
- **Decision:** background, P3; 10.1 Capstone Research Sprint.
- **Source fact:** Author essay discusses research taste, abstraction levels, multimodal interpretability and actionability.
- **Why / intuition role:** Use a short research-framing reading rather than treating a personal perspective as an empirical method.
- **Student method role:** Write one testable claim at behavioral, algorithmic and implementation levels for an existing capstone.
- **Signature figure / visualization role:** Claim-to-evidence ladder with one concrete counterexample per level.
- **Prediction role:** What observation would change your favored explanation rather than merely add an attractive visualization?
- **Controls and baselines:** Competing explanations and a null outcome; distinguish quoted opinion from testable claim.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** Public essay; no associated replication data/code asserted.

### INB057 — Computer Security. ESORICS 2025 International Workshops: ANUBIS 2025, SECAI 2025, SecAssure 2025, STMUS 2025, Toulouse, France, September 22–24, 2025, Revised Selected Papers, Part II

- **Identity / version:** verified publisher title/DOI; edited conference-proceedings volume. Springer conference proceedings, copyright 2026; DOI 10.1007/978-3-032-16092-8.
- **Primary sources:** [source 1](https://link.springer.com/book/10.1007/978-3-032-16092-8).
- **Read:** publisher metadata and table-of-contents screening only; no chapter read.
- **Decision:** defer/out of scope, P3; 10.1 Capstone Research Sprint: bibliographic screening only.
- **Source fact:** Publisher identifies an edited ESORICS workshop proceedings volume, not one interpretability study.
- **Why / intuition role:** A volume-level entry cannot justify integrating an unspecified chapter. Preserve it until the relevant chapter is identified.
- **Student method role:** No method selected; first identify an exact chapter and its question.
- **Signature figure / visualization role:** Proposed bibliography tree: proceedings to chapter to claim.
- **Prediction role:** Which chapter provides the claim, and would its experiment actually address the learner question?
- **Controls and baselines:** Require chapter identity and primary full text before designing a result.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** Publisher metadata/table of contents accessible; chapter full-text access/code/data UNKNOWN.

### INB058 — Evaluating the Causal Faithfulness of SHAP as a Candidate Attribution Method for SAE Features

- **Identity / version:** verified title and authors on primary source. arXiv v1, 2026-10-13; preprint.
- **Primary sources:** [source 1](https://arxiv.org/abs/2610.17898); [source 2](https://arxiv.org/html/2610.17898v1).
- **Read:** abstract plus selected methods: sections 3.3-3.6, including probe target, reconstruction baseline and intervention scope mismatch.
- **Decision:** existing lesson addition, P1; 16.8 SHAP vs Activation Patching; 6.2 Gemma Scope and Feature Steering.
- **Source fact:** Distinguishes attribution agreement from LM causal faithfulness; reports an active-feature confound and insertion/deletion asymmetry.
- **Why / intuition role:** Directly tests the most tempting unsupported conclusion in the existing attribution-agreement lesson.
- **Student method role:** Implement signed and magnitude correlations against interventions, active-only stratification and a declared multiple-testing family.
- **Signature figure / visualization role:** Agreement matrix beside causal-effect scatterplots and insertion-versus-deletion detection curves.
- **Prediction role:** Can SHAP and IG agree perfectly while both fail to predict the LM effect of deleting a feature?
- **Controls and baselines:** Reconstruction-only SAE pass, matched token scope, active-only and activity-matched random features, shuffled scores, held-out minimal pairs.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB059 — DepthBench: Measuring How Residual Connections Enable More Computational Depth

- **Identity / version:** verified title and authors on primary source. arXiv v1, 2026-09-26.
- **Primary sources:** [source 1](https://arxiv.org/abs/2609.32534).
- **Read:** metadata and abstract.
- **Decision:** background, P3; 17.4 Training Dynamics Across Architectures.
- **Source fact:** Compares depth/width allocations at matched parameter count and pretraining recipe across residual architectures.
- **Why / intuition role:** A fairness checklist for architecture comparisons; recreating ten training families is not needed.
- **Student method role:** Compute parameter counts and effective-depth diagnostics on supplied finite network traces.
- **Signature figure / visualization role:** Loss versus depth at fixed parameter budgets, with per-layer causal removal.
- **Prediction role:** Does doubling layer count imply doubling useful sequential computation?
- **Controls and baselines:** Matched parameters and tokens, report FLOPs/walltime separately, shallow-wide and identity-layer controls.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB060 — Same Task, Different Circuits: Disentangling Modality-Specific Mechanisms in VLMs

- **Identity / version:** verified title and authors on primary source. arXiv v3, 2025-10-03.
- **Primary sources:** [source 1](https://arxiv.org/abs/2506.09047).
- **Read:** metadata and abstract.
- **Decision:** existing lesson addition, P1; 12.4 Visual Token Flow in Real VLMs; 12.5 Modality Arbitration.
- **Source fact:** Reports largely separate modality circuits with late representation alignment and late-to-early visual patching gains.
- **Why / intuition role:** Complements cross-modal task vectors without equating shared high-level content with shared low-level circuits.
- **Student method role:** Implement late-to-early state injection and compare modality circuit overlap on matched tasks.
- **Signature figure / visualization role:** Layer-pair intervention heatmap beside image/text circuit overlap.
- **Prediction role:** Can two modalities align in a late task representation while still using disjoint circuits?
- **Controls and baselines:** Same-layer patch, wrong-task late state, random same-norm state, text-only/image-only baselines.
- **Replication role:** bounded real study; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB061 — Better late than never: Getting into interpretability in 2025

- **Identity / version:** duplicate of INB056 by exact URL; title resolved to Better late than never: Getting into interpretability in 2025; not LatentLens. Author webpage read 2026-11-03.
- **Primary sources:** [source 1](https://bennokrojer.com/interp.html).
- **Read:** same author essay as INB056.
- **Decision:** duplicate, P3; 10.1 Capstone Research Sprint: refer to INB056.
- **Source fact:** Same public URL and resolved essay title as INB056.
- **Why / intuition role:** Duplicate bibliographic record only; retain both Inbox IDs without inventing a second paper.
- **Student method role:** No independent lab; reuse INB056 framing exercise.
- **Signature figure / visualization role:** No independent figure; reuse INB056 claim-to-evidence ladder.
- **Prediction role:** Would counting two records for one URL falsely suggest two independent pieces of evidence?
- **Controls and baselines:** Canonical URL equality; do not deduplicate distinct papers based only on common author.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** Public essay; no independent code/data artifact.

### INB062 — Vision-Language Models Create Cross-Modal Task Representations

- **Identity / version:** resolved missing identifiers with exact PMLR/arXiv authors and title; conference paper, not journal article. arXiv v2, 2025-05-07; ICML 2025 PMLR 267:41157-41176.
- **Primary sources:** [source 1](https://arxiv.org/abs/2410.22330); [source 2](https://proceedings.mlr.press/v267/luo25c.html); [source 3](https://task-vectors-are-cross-modal.github.io/).
- **Read:** abstract plus author-project cross-modal patching explanation; no full-paper audit.
- **Decision:** existing lesson addition, P1; 12.4 Visual Token Flow in Real VLMs; 11.2 Representation Alignment.
- **Source fact:** Finds task-vector transfer across modality and format; paper title previously Task Vectors are Cross-Modal.
- **Why / intuition role:** A causal transfer matrix supplies stronger evidence than a colored embedding projection.
- **Student method role:** Extract task summaries at the stated token/layer and patch across held-out modality-task pairs.
- **Signature figure / visualization role:** Source-modality by target-modality task-transfer accuracy with wrong-task controls.
- **Prediction role:** Does task-wise t-SNE clustering predict successful cross-modal patching on held-out examples?
- **Controls and baselines:** Wrong-task and random vectors, same-modality patch, zero vector, prompting baseline; select layer on separate split.
- **Replication role:** bounded real study; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** Official repository search summary states code and data available; contents/licenses/runtime unchecked.

### INB063 — When Models Manipulate Manifolds: The Geometry of a Counting Task

- **Identity / version:** verified title and authors on primary source. arXiv v1, 2026-01-08.
- **Primary sources:** [source 1](https://arxiv.org/abs/2601.04480).
- **Read:** metadata and abstract.
- **Decision:** existing lesson addition, P1; 11.6 Feature Manifolds; 8.7 Circuit Tracing with Attribution Graphs.
- **Source fact:** Reports curved character-count manifolds and geometric transformations during fixed-width linebreaking in Claude 3.5 Haiku.
- **Why / intuition role:** A concrete task joins geometric coordinates, sparse feature families and causal decision boundaries.
- **Student method role:** Implement character-count accumulation and controlled coordinate interventions on a transparent linebreaking fixture.
- **Signature figure / visualization role:** Count manifold colored by remaining line width, beside break-probability intervention curves.
- **Prediction role:** Can a curved count representation support a linear break/no-break readout after a geometric transformation?
- **Controls and baselines:** Token count versus character count, variable token lengths, shuffled counts, boundary-matched counterexamples.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** Paper studies a proprietary model; full internal access and released artifacts UNKNOWN. CPU fixture would not replicate Claude findings.

### INB064 — Chameleon: Mixed-Modal Early-Fusion Foundation Models

- **Identity / version:** verified title and authors on primary source. arXiv v2, 2025-03-21.
- **Primary sources:** [source 1](https://arxiv.org/abs/2405.09818).
- **Read:** metadata and abstract.
- **Decision:** background, P3; 13.6 Autoregressive Image Tokens; 12.3 Mini VLM from Scratch.
- **Source fact:** Describes early-fusion token models for interleaved text/image understanding and generation.
- **Why / intuition role:** Architecture comparator explains early fusion without adding another full model family implementation.
- **Student method role:** Implement a mixed token mask and next-token loss accounting on a supplied sequence.
- **Signature figure / visualization role:** Interleaved token timeline with per-modality loss contributions.
- **Prediction role:** Would a single averaged token loss fairly compare text and image performance?
- **Controls and baselines:** Separate per-modality denominators, shuffled modality segments and text-only comparator.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** UNKNOWN: code/data/checkpoint access not checked

### INB065 — Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling

- **Identity / version:** resolved missing URL/DOI from exact arXiv title/authors; research preprint rather than demonstrated journal publication. arXiv v1, 2025-01-29.
- **Primary sources:** [source 1](https://arxiv.org/abs/2501.17811).
- **Read:** metadata and abstract.
- **Decision:** background, P3; 13.6 Autoregressive Image Tokens; 12.3 Mini VLM from Scratch.
- **Source fact:** Reports Janus-Pro improvements from training strategy, data expansion and model scaling.
- **Why / intuition role:** Use as confounding example: simultaneous recipe changes do not isolate one architectural cause.
- **Student method role:** Decompose a claimed gain into a hypothetical factorial ablation design.
- **Signature figure / visualization role:** Factorial table separating model size, data and training recipe.
- **Prediction role:** Can the headline before/after result identify which of three simultaneous changes caused the gain?
- **Controls and baselines:** Single-factor and interaction comparisons, matched inference protocol; no new training proposed.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** Primary abstract states code/models publicly available; actual repository, data and deployment constraints unchecked.

### INB066 — SANA: Efficient High-Resolution Image Synthesis with Linear Diffusion Transformers

- **Identity / version:** resolved broken hyphenation and missing URL/DOI with exact arXiv authors; technical preprint, journal status unverified. arXiv v3, 2024-10-20.
- **Primary sources:** [source 1](https://arxiv.org/abs/2410.10629); [source 2](https://github.com/NVlabs/Sana).
- **Read:** metadata and abstract.
- **Decision:** background, P3; 13.5 Diffusion Transformers and ConceptAttention; 5.6 Linear Attention.
- **Source fact:** Describes deep compression and linear-attention diffusion transformers for efficient image synthesis.
- **Why / intuition role:** Useful computational-scaling reading; paper throughput claims require matched hardware and quality before reuse.
- **Student method role:** Implement token-count and attention-memory accounting across resolutions/compression factors.
- **Signature figure / visualization role:** Resolution versus tokens, estimated attention memory and measured-quality placeholders.
- **Prediction role:** Does 32x spatial compression reduce tokens by 4x or 16x relative to 8x compression?
- **Controls and baselines:** Dense versus linear attention, equal resolution/batch/precision, report quality and warmup separately.
- **Replication role:** CPU illustration; planning only. UNKNOWN: paper-scale hardware/runtime not checked; no execution authorized.
- **Code/data/access:** Official NVlabs/Sana repository exists; exact original-paper checkpoints/data/runtime/license audit unchecked.

## Metadata repairs proposed for human review only

INB043 resolves to MetaQueries (arXiv:2504.06256). INB052 matches the published CVPR 2025 AnyAttack title; arXiv:2410.05346 has a title variant. INB062 resolves to ICML 2025/PMLR and arXiv:2410.22330. INB065 resolves to arXiv:2501.17811; INB066 to arXiv:2410.10629 with the broken title hyphenation repaired for display. Conference/preprint identities should not be silently promoted to journal publications. INB057 is an edited proceedings volume; no particular chapter has been selected. No library fields were changed.

**INB056 and INB061 are the same author essay, “Better late than never: Getting into interpretability in 2025,” at the exact same URL.** The page links a subsequently released paper; this does not turn either Inbox record into a LatentLens paper record. LatentLens is handled as the separately requested external paper addition by the lead research task.

## Limits and next review boundary

The strongest additions deepen current lessons rather than establish new model families. Real-model proposals need exact revisions, artifact access, split rules, estimands, selection/failure rules and resource budgets before execution. CPU illustrations must remain labeled illustrations. Abstract-only rows need method/figure reading before lesson implementation. Theory or cryptographic claims are not certified by these notes. No paper was ranked by citation count, popularity or a second-hand summary alone. Original ARENA and existing curriculum remain unchanged.

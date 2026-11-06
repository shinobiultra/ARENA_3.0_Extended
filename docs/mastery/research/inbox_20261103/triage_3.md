# Inbox batch 3: INB067–INB097

Draft research triage, 3 November 2026. All 31 entries from the 97-record snapshot (library version 10797) are retained exactly once. This is a reading and planning artifact; no lesson or readiness promotion, model execution, Zotero mutation or GitHub mutation occurred. Original ARENA materials are untouched. Existing 86-lesson/205-requirement coverage stays intact.

Every entry has a verified primary identity. Most reviews stop at primary metadata and abstract; six include selected method text, not a complete paper read. Reported findings are the authors' claims. Figures, prediction prompts and exercises below are our proposals, not existing results. P1 = full read before near-term integration; P2 = later targeted read; P3 = background/defer. UNKNOWN means unverified, not unavailable.

Lesson numbers follow `infrastructure/extension/syllabus.yaml`: **8.3 = AtP*** and **8.4 = ACDC/Circuit Metrics**. Earlier pilot shorthand called ACDC 8.3. Basics additions belong in the separately owned foundations plan, not edits to protected original notebooks.

## Compact disposition matrix

| ID | Public work / primary source | Read / priority | Disposition → existing lesson | Concrete role |
|---|---|---|---|---|
| INB067 | [Contrastive World Models](https://arxiv.org/abs/2609.22175v1) | abstract/identity; P2 | existing lesson addition → 14.3, 14.7 | Direct contrastive-versus-reconstruction comparison strengthens the existing world-model lesson without adding a model family. |
| INB068 | [Information-Theoretic Probing with Minimum Description Length](https://aclanthology.org/2020.emnlp-main.14/) | selected methods; P1 | core integration → 0.6, 11.1 | A concrete bridge from cross-entropy to honest probing belongs in foundations and probe audits. |
| INB069 | [Gradient Routing Localizes and Removes Unintended Behaviors in RL](https://arxiv.org/abs/2610.17062v1) | selected methods; P1 | existing lesson addition → 9.5, 15.2 | Connects white-box monitoring to intervention and adapter removal, while preserving limitations of imperfect monitors. |
| INB070 | [Recurrent Looped Transformer](https://arxiv.org/abs/2610.07591v2) | abstract/identity; P2 | optional replication → 5.5, 5.7 | Use as optional state-tracking comparison; do not combine early proposal claims with later empirical results silently. |
| INB071 | [Looped Diffusion Transformer](https://arxiv.org/abs/2609.40305v1) | abstract/identity; P3 | background → 13.5 | Useful architecture reading, but training a 260M image generator exceeds a pilot integration. |
| INB072 | [Revealing and Reducing Gender Biases in Vision and Language Assistants (VLAs)](https://arxiv.org/abs/2410.19314v2) | abstract/identity; P2 | existing lesson addition → 12.5, 0.6 | Teaches evaluation design and confounds, without treating binary experimental categories as exhaustive human identities. |
| INB073 | [Story Imprinting: AI Assistants Absorb Traits from Human Characters They Resemble](https://arxiv.org/abs/2609.10883v1) | abstract/identity; P2 | background → 9.4, 15.4 | Relevant to emergent misalignment, but new paid fine-tuning and causal claims need a separately frozen study. |
| INB074 | [Targeted Recovery of Weight-Space Mechanisms From Neural Networks](https://arxiv.org/abs/2607.13047v1) | abstract/identity; P2 | optional replication → 8.6, 18.2 | Offers a weight-space counterpart to activation circuits; large-model recovery remains a separate replication. |
| INB075 | [Late-Stage Generalization Collapse in Grokking: Detecting anti-grokking with Weightwatcher](https://arxiv.org/abs/2602.02859v1) | abstract/identity; P2 | existing lesson addition → 17.1 | An important negative outcome for grokking; reported associations do not make a universal collapse detector. |
| INB076 | [Seeing or Knowing? Visual Context Sensitivity in Multimodal Large Language Models](https://arxiv.org/abs/2607.26326v2) | abstract/identity; P2 | existing lesson addition → 12.4, 12.5 | Directly strengthens existing modality-arbitration lessons with an encoding-versus-use distinction. |
| INB077 | [CircuitTree: Composing Circuits into A Syllable Transcriber](https://arxiv.org/abs/2610.12520v1) | abstract/identity; P2 | optional replication → 8.4, 8.6 | Adds a composition failure mode to circuit metrics; audio training is optional and separately scoped. |
| INB078 | [A Distributional View for Visual Mechanistic Interpretability: KL-Minimal Soft-Constraint Principle](https://arxiv.org/abs/2605.17504v1) | abstract/identity; P2 | background → 12.7, 13.1 | Makes KL direction and naturalness-versus-faithfulness trade-offs teachable before expensive diffusion replication. |
| INB079 | [AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts](https://arxiv.org/abs/2010.15980v2) | abstract/identity; P2 | existing lesson addition → 9.5, 7.1 | Historical prompt optimization explains a method and its limits; not evidence of internal causal explanation. |
| INB080 | [SenseNova-U1.5: Towards Native Unified Visual Intelligence](https://arxiv.org/abs/2609.11929v1) | abstract/identity; P3 | defer out of scope → 12.3, 13.6 | New foundation-model family would violate pilot-first focus; retain as background for later architecture survey. |
| INB081 | [LLMZip: Lossless Text Compression using Large Language Models](https://arxiv.org/abs/2306.04050v2) | abstract/identity; P1 | core integration → 0.6, 11.1 | Turns cross-entropy and bits into an observable foundations result without downloading LLaMA. |
| INB082 | [OneCAT: Decoder-Only Auto-Regressive Model for Unified Understanding and Generation](https://arxiv.org/abs/2509.03498v3) | abstract/identity; P3 | background → 12.3, 13.6 | Useful architecture contrast; no addition of a large new family while pilots are unfinished. |
| INB083 | [A Sovereign, Open-Source Foundation Model for German and English](https://arxiv.org/abs/2607.09424v3) | abstract/identity; P3 | defer out of scope → 5.2, 5.7 | A 30B-total model and sovereign infrastructure are peripheral to current pilot completion. |
| INB084 | [Language Generation with Strictly Proper Scoring Rules](https://proceedings.mlr.press/v235/shao24c.html) | selected methods; P1 | core integration → 0.6, 11.1 | Foundational distinction among a training loss, calibration, decision metrics and token versus sequence scoring. |
| INB085 | [Selectivity and Tolerance (“Invariance”) Both Increase as Visual Information Propagates from Cortical Area V4 to IT](https://www.jneurosci.org/content/30/39/12978) | abstract/identity; P2 | background → 12.7, 11.2 | Clarifies ambiguous invariance language; artificial response simulations cannot reproduce biological evidence. |
| INB086 | [Beyond SAEs: Consistent Concept Discovery with Sparse Coding](https://arxiv.org/abs/2610.23801v1) | selected methods; P1 | existing lesson addition → 6.1, 6.3 | An explicit amortized-inference versus per-sample optimization comparison improves the existing SAE pilot. |
| INB087 | [Proving and Improving the Platonic Representation Hypothesis with JEPAs](https://arxiv.org/abs/2610.39302v1) | abstract/identity; P1 | existing lesson addition → 11.2, 14.4 | Delegate substantive appraisal to the named PRH owner; this row preserves inbox coverage. |
| INB088 | [A Mechanistic Understanding of Pronoun Fidelity in LLMs](https://arxiv.org/abs/2606.16407v1) | abstract/identity; P2 | existing lesson addition → 11.3, 9.5 | A concrete multi-mechanism application of DAS; retain fairness categories and template limits explicitly. |
| INB089 | [Neural Networks Are Already Clean](https://already-clean.goodfire.com/) | selected methods; P2 | background → 6.1, 6.3 | Useful adversarial reading for SAE assumptions, not established evidence that superposition is solved. |
| INB090 | [Make it SING: Analyzing Semantic Invariants in Classifiers](https://arxiv.org/abs/2603.14610v2) | abstract/identity; P2 | existing lesson addition → 11.1, 12.7 | A visible linear-algebra bridge from SVD to invariance; semantic captions are not ground truth by themselves. |
| INB091 | [Self-Play Pretraining with Zero Data](https://arxiv.org/abs/2609.30063v1) | abstract/identity; P3 | optional replication → 17.4, 14.5 | An interesting training-dynamics capstone, but a universal-program self-play campaign is outside pilot work. |
| INB092 | [Rethinking Projector Training in Multimodal LLMs](https://arxiv.org/abs/2610.18598v1) | selected methods; P1 | existing lesson addition → 12.3, 11.1, 11.2 | Strong foundations-to-Mini-VLM bridge, provided moment matching is not conflated with semantic correspondence. |
| INB093 | [VISOR++: Universal Visual Inputs based Steering for Large Vision Language Models](https://arxiv.org/abs/2509.25533v2) | abstract/identity; P2 | optional replication → 9.1, 12.2, 12.5 | A useful intervention-access contrast; substantial VLM optimization and API access remain separately authorized. |
| INB094 | [Steered LLM Activations are Non-Surjective](https://arxiv.org/abs/2604.09839v2) | abstract/identity; P1 | existing lesson addition → 9.1, 11.1 | An essential limit on interpreting steering as prompting; finite enumeration is an illustration, not a proof for arbitrary models. |
| INB095 | [What Do Visual-Head Discovery Scores Actually Find? Attention Source, Necessity, and Random-Control Pitfalls in Vision-Language Models](https://arxiv.org/abs/2610.13111v1) | abstract/identity; P1 | duplicate → 12.4, 12.5, 0.6 | Duplicate-by-DOI receipt only; no Zotero merge or mutation. |
| INB096 | [Correcting Attribution Patching with a Few Exact Measurements](https://arxiv.org/abs/2610.27393v1) | selected methods; P1 | existing lesson addition → 8.2, 8.3 | A direct efficiency and failure-analysis follow-up to attribution patching; full large-model replication is separate. |
| INB097 | [Dissecting the Safety Circuit: Neuronal Intervention for Transferable Adversarial Attacks on VLMs](https://proceedings.mlr.press/v306/xie26f.html) | abstract/identity; P2 | optional replication → 9.5, 12.5 | Useful mechanistic-transfer critique; publisher type correction is concrete, while attack execution is outside this reading task. |

## Per-record integration cards

Each card separates primary-source facts from our proposed teaching design. The structured companion [triage_3.json](triage_3.json) contains the same 31 identities and the method/visualization/intuition/results-prediction/replication roles.

### INB067 — Contrastive World Models

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2609.22175v1. primary metadata and abstract triage; not full paper review. Full-read priority: P2.

**Primary sources:** [source 1](https://arxiv.org/abs/2609.22175v1).

**Source facts:** Replaces pixel reconstruction with contrastive future-patch prediction and reports improved distractor robustness in small-scale world-model settings.

**Our proposal:** Implement patch-future InfoNCE and latent transition on a planted moving-object grid. Signature figure: Task success versus distractor strength, beside latent-state decoding. Prediction before running: “Will a reconstruction objective retain more background than task state as distractors increase?” Controls: Shuffled futures; matched latent dimension and optimization budget; reconstruction and momentum-prediction baselines.

**Placement / disposition:** existing lesson addition; 14.3 Action-Conditioned World Models and Latent Planning, 14.7 Recurrent World Models - Dreamer, TD-MPC2 and IRIS. Direct contrastive-versus-reconstruction comparison strengthens the existing world-model lesson without adding a model family.

**Replication and access:** bounded real study. compute: UNKNOWN for faithful replication; model training/inference resources not profiled; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

### INB068 — Information-Theoretic Probing with Minimum Description Length

**Identity / read depth:** VERIFIED primary title/author/identifier match; publisher/author record. primary metadata/abstract plus selected method sections and availability statements; not full paper review. Full-read priority: P1.

**Primary sources:** [source 1](https://aclanthology.org/2020.emnlp-main.14/), [source 2](https://aclanthology.org/2020.emnlp-main.14.pdf).

**Source facts:** Proposes online and variational description-length probes to account for the effort required to predict labels.

**Our proposal:** Implement prequential label codelength, including initial uniform block and training only on prior blocks. Signature figure: Learning curves and cumulative bits for structured versus control labels. Prediction before running: “Can two probes have the same final accuracy and different coding cost?” Controls: Type-consistent random labels; random representations; fixed block schedule; disjoint evaluation groups.

**Placement / disposition:** core integration; 0.6 How to Know When an Interpretability Result Is Fake (Practice), 11.1 PCA, SVD and Geometry Controls. A concrete bridge from cross-entropy to honest probing belongs in foundations and probe audits.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: Paper links https://github.com/lena-voita/description-length-probing; repository not executed; artifact completeness UNKNOWN; data: UNKNOWN; not checked.

### INB069 — Gradient Routing Localizes and Removes Unintended Behaviors in RL

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2610.17062v1. primary metadata/abstract plus selected method sections and availability statements; not full paper review. Full-read priority: P1.

**Primary sources:** [source 1](https://arxiv.org/abs/2610.17062v1).

**Source facts:** GRAFT routes flagged trajectories into sacrificial adapter parameters; the paper reports retained target performance after ablation.

**Our proposal:** Implement per-example retain/forget gradient masks on a tiny two-behavior task; inspect the paper's anchor updates and unrouted optimizer moments separately. Signature figure: True unwanted-behavior versus useful-task performance before/after adapter ablation across monitor recall. Prediction before running: “Does removing flagged training loss prevent undetected behavior, or move learning into the monitor blind spot?” Controls: Penalty/filter baselines; random routing; oracle monitor; matched adapter capacity; unseen blind-spot contexts.

**Placement / disposition:** existing lesson addition; 9.5 White-Box Monitors and Red-Teaming with Internals, 15.2 LoRA vs Full Fine-Tuning. Connects white-box monitoring to intervention and adapter removal, while preserving limitations of imperfect monitors.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: Paper Appendix A names https://github.com/jnward/GRAFT; repository not executed; artifact completeness UNKNOWN; data: UNKNOWN; not checked.

### INB070 — Recurrent Looped Transformer

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2610.07591v2. primary metadata and abstract triage; not full paper review. Full-read priority: P2.

**Primary sources:** [source 1](https://arxiv.org/abs/2610.07591v2), [source 2](https://www.alphaxiv.org/abs/2609.recurrent-looped-transformer), [source 3](https://yifanzhang-pro.github.io/recurrent-looped-tranformer/).

**Source facts:** An earlier alphaXiv architecture proposal carries recurrent decoder state across prompt and response; the later arXiv paper adds finite algorithmic experiments.

**Our proposal:** Implement one recurrent decoder-state transition and compare identical token histories with different prompt boundaries. Signature figure: State-tracking accuracy by test length and charged sequential steps. Prediction before running: “Does boundary-invariant state carry imply improved language modeling or only consistent policy semantics?” Controls: No feedback; reset-state boundary; equal parameters/tokens plus reported FLOPs and serial latency.

**Placement / disposition:** optional replication; 5.5 State Tracking in SSMs and Transformers, 5.7 Griffin, xLSTM, Hyena and Hybrid Models. Use as optional state-tracking comparison; do not combine early proposal claims with later empirical results silently.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Existing URL is an alphaXiv-exclusive September proposal, not a numeric arXiv ID. Later primary arXiv identity is 2610.07591v2 (11 Oct 2026), with five authors. Preserve proposal and later empirical version separately; do not silently replace webpage with preprint.

### INB071 — Looped Diffusion Transformer

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2609.40305v1. primary metadata and abstract triage; not full paper review. Full-read priority: P3.

**Primary sources:** [source 1](https://arxiv.org/abs/2609.40305v1).

**Source facts:** Looped-DiT combines intermediate-loop supervision and self-modulating attention to stabilize shared-block diffusion computation.

**Our proposal:** Draw shared-block unrolling and propose loop-depth versus denoising-step allocation at fixed compute. Signature figure: Quality versus measured total forward FLOPs, with loop depth separated from sampling steps. Prediction before running: “Does an extra internal loop help more than an extra denoising step at the same cost?” Controls: Matched parameters and FLOPs; naive looping; deep-supervision-only; attention-modulation-only.

**Placement / disposition:** background; 13.5 Diffusion Transformers and ConceptAttention. Useful architecture reading, but training a 260M image generator exceeds a pilot integration.

**Replication and access:** faithful paper replication. compute: UNKNOWN for faithful replication; model training/inference resources not profiled; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Primary v1 submitted 30 Sep 2026 UTC; inbox timestamp 1 Oct 2026 may reflect timezone. Normalize only as a proposed bibliographic correction.

### INB072 — Revealing and Reducing Gender Biases in Vision and Language Assistants (VLAs)

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2410.19314v2. primary metadata and abstract triage; not full paper review. Full-read priority: P2.

**Primary sources:** [source 1](https://arxiv.org/abs/2410.19314v2).

**Source facts:** Studies gender-associated outputs across 22 vision-language assistants and compares debiasing/performance trade-offs.

**Our proposal:** Implement paired counterfactual aggregation on an explicitly constructed occupation/trait response table. Signature figure: Paired group differences with intervals and retained task performance. Prediction before running: “Could an aggregate bias reduction conceal a direction reversal on individual traits?” Controls: Matched image and wording pairs; image-only/text-only; shuffled group assignment; family-wise correction.

**Placement / disposition:** existing lesson addition; 12.5 Object Hallucination and Modality Arbitration, 0.6 How to Know When an Interpretability Result Is Fake (Practice). Teaches evaluation design and confounds, without treating binary experimental categories as exhaustive human identities.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: Primary abstract names https://github.com/ExplainableML/vla-gender-bias; repository not inspected or executed; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Pin verified primary version 2410.19314v2 for future full reading; original inbox record does not reliably pin this exact version.

### INB073 — Story Imprinting: AI Assistants Absorb Traits from Human Characters They Resemble

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2609.10883v1. primary metadata and abstract triage; not full paper review. Full-read priority: P2.

**Primary sources:** [source 1](https://arxiv.org/abs/2609.10883v1).

**Source facts:** Reports cross-format behavioral changes after story fine-tuning, including stronger adoption from similar characters.

**Our proposal:** Design counterbalanced story/assistant evaluation with training format separated from evaluation format. Signature figure: Behavior transfer matrix by character affinity and task context. Prediction before running: “Would name overlap, helpfulness, or institutional prestige explain apparent character affinity?” Controls: Matched story content; swapped attributes; base-model and persona controls; held-out story templates.

**Placement / disposition:** background; 9.4 Emergent Misalignment Detection, 15.4 LoRA Phase Transitions and Misalignment Proxies. Relevant to emergent misalignment, but new paid fine-tuning and causal claims need a separately frozen study.

**Replication and access:** bounded real study. compute: UNKNOWN for faithful replication; model training/inference resources not profiled; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

### INB074 — Targeted Recovery of Weight-Space Mechanisms From Neural Networks

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2607.13047v1. primary metadata and abstract triage; not full paper review. Full-read priority: P2.

**Primary sources:** [source 1](https://arxiv.org/abs/2607.13047v1).

**Source facts:** Targeted parameter decomposition uses a catch-all component to focus weight-space decomposition on specified inputs.

**Our proposal:** Implement planted linear-network decomposition with a target component and residual catch-all. Signature figure: Target-task fidelity versus collateral loss and decomposition cost. Prediction before running: “Can a small target component be faithful while the catch-all hides non-target failures?” Controls: Naive decomposition; random target subsets; full-model output check; unseen non-target inputs.

**Placement / disposition:** optional replication; 8.6 Sparse Feature Circuits and SHIFT, 18.2 Sparse and Modular Training, KANs and Sparse Trees. Offers a weight-space counterpart to activation circuits; large-model recovery remains a separate replication.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

### INB075 — Late-Stage Generalization Collapse in Grokking: Detecting anti-grokking with Weightwatcher

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2602.02859v1. primary metadata and abstract triage; not full paper review. Full-read priority: P2.

**Primary sources:** [source 1](https://arxiv.org/abs/2602.02859v1).

**Source facts:** Reports late generalization collapse in extended grokking runs and proposes shuffled-weight spectral diagnostics.

**Our proposal:** Implement shuffled weight spectra and track them against a predeclared checkpoint trajectory. Signature figure: Train/test accuracy and spectral outlier trajectories on the same time axis. Prediction before running: “Do spectral outliers predict future collapse on unseen seeds before using held-out accuracy?” Controls: Predeclared alarm; non-collapsing seeds; norms and entropy baselines; shuffled weights; no retrospective threshold fitting.

**Placement / disposition:** existing lesson addition; 17.1 Checkpoint Archaeology and Grokking. An important negative outcome for grokking; reported associations do not make a universal collapse detector.

**Replication and access:** bounded real study. compute: UNKNOWN for faithful replication; model training/inference resources not profiled; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

### INB076 — Seeing or Knowing? Visual Context Sensitivity in Multimodal Large Language Models

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2607.26326v2. primary metadata and abstract triage; not full paper review. Full-read priority: P2.

**Primary sources:** [source 1](https://arxiv.org/abs/2607.26326v2).

**Source facts:** Separates visual information availability from its use; WhatIfVis tests conflicts between visual context and language priors.

**Our proposal:** Implement image-versus-prior answer scoring and layer intervention on explicit counterfactual pairs. Signature figure: Layer patching heatmap beside visual-context sensitivity curves. Prediction before running: “If a decoder can reconstruct an attribute, must the model use that attribute when answering?” Controls: Text-only; scrambled images; consistent versus conflicting priors; no-op/random-layer patches; held-out attributes.

**Placement / disposition:** existing lesson addition; 12.4 Visual Token Flow in Real VLMs, 12.5 Object Hallucination and Modality Arbitration. Directly strengthens existing modality-arbitration lessons with an encoding-versus-use distinction.

**Replication and access:** bounded real study. compute: UNKNOWN for faithful replication; model training/inference resources not profiled; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Pin verified primary version 2607.26326v2 for future full reading; original inbox record does not reliably pin this exact version.

### INB077 — CircuitTree: Composing Circuits into A Syllable Transcriber

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2610.12520v1. primary metadata and abstract triage; not full paper review. Full-read priority: P2.

**Primary sources:** [source 1](https://arxiv.org/abs/2610.12520v1).

**Source facts:** CircuitTree composes many pruned HuBERT circuits using a density-scaled tree schedule and reports syllable transcription.

**Our proposal:** Implement union and density-aware composition on a planted modular classifier before considering audio circuits. Signature figure: Composite accuracy versus component sharing and circuit size. Prediction before running: “Will the union of individually faithful circuits remain faithful when tasks are composed?” Controls: Naive union; independent circuits; same-size random circuits; permutation of composition order.

**Placement / disposition:** optional replication; 8.4 ACDC and Circuit Metrics, 8.6 Sparse Feature Circuits and SHIFT. Adds a composition failure mode to circuit metrics; audio training is optional and separately scoped.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

### INB078 — A Distributional View for Visual Mechanistic Interpretability: KL-Minimal Soft-Constraint Principle

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2605.17504v1. primary metadata and abstract triage; not full paper review. Full-read priority: P2.

**Primary sources:** [source 1](https://arxiv.org/abs/2605.17504v1).

**Source facts:** Formulates visual feature interpretation as KL-minimal distribution change and instantiates energy-guided diffusion sampling.

**Our proposal:** Derive exponential tilting on a finite image-feature surrogate; implement normalized reweighting directly. Signature figure: Activation gain versus KL distortion frontier, with example inputs. Prediction before running: “Can maximizing activation alone move far from the original data distribution?” Controls: Unregularized activation maximization; top-k retrieval; random feature; KL weight sweep.

**Placement / disposition:** background; 12.7 Dense Vision Features - DINO and SAM, 13.1 Diffusion Models and Cross-Attention Attribution. Makes KL direction and naturalness-versus-faithfulness trade-offs teachable before expensive diffusion replication.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Wrong journalArticle type for the verified source: primary is an arXiv preprint; any journal version UNRESOLVED. Replace discovery-only Semantic Scholar URL with verified primary arXiv URL in any future bibliographic repair; no write made.

### INB079 — AutoPrompt: Eliciting Knowledge from Language Models with Automatically Generated Prompts

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2010.15980v2. primary metadata and abstract triage; not full paper review. Full-read priority: P2.

**Primary sources:** [source 1](https://arxiv.org/abs/2010.15980v2).

**Source facts:** AutoPrompt uses gradients to search discrete trigger tokens for masked language models.

**Our proposal:** Implement first-order embedding-token candidate scores then exact top-candidate evaluation. Signature figure: Trigger-search trajectory with development and held-out accuracy separated. Prediction before running: “Can a nonsensical trigger outperform a natural prompt yet overfit the search examples?” Controls: Random-token search; manual prompts; label-word swaps; equal search budgets; independent test templates.

**Placement / disposition:** existing lesson addition; 9.5 White-Box Monitors and Red-Teaming with Internals, 7.1 Logit Lens, Tuned Lens and Patchscopes. Historical prompt optimization explains a method and its limits; not evidence of internal causal explanation.

**Replication and access:** bounded real study. compute: UNKNOWN for faithful replication; model training/inference resources not profiled; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Pin verified primary version 2010.15980v2 for future full reading; original inbox record does not reliably pin this exact version.

### INB080 — SenseNova-U1.5: Towards Native Unified Visual Intelligence

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2609.11929v1. primary metadata and abstract triage; not full paper review. Full-read priority: P3.

**Primary sources:** [source 1](https://arxiv.org/abs/2609.11929v1).

**Source facts:** Describes an 8B unified multimodal architecture and promises release of training code.

**Our proposal:** Use a paper diagram to compare shared autoregressive interfaces with separate encoders; no new model implementation. Signature figure: Architecture/token-path diagram with reported resource categories. Prediction before running: “Which interface simplifications reduce components without necessarily reducing training cost?” Controls: Separate understanding/generation baselines; same resolution and task; count active parameters and compute.

**Placement / disposition:** defer out of scope; 12.3 Mini VLM from Scratch, 13.6 Autoregressive Image Tokens. New foundation-model family would violate pilot-first focus; retain as background for later architecture survey.

**Replication and access:** faithful paper replication. compute: UNKNOWN for faithful replication; model training/inference resources not profiled; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: Abstract promises open-source training code; current actual release status UNKNOWN; data: UNKNOWN; not checked.

### INB081 — LLMZip: Lossless Text Compression using Large Language Models

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2306.04050v2. primary metadata and abstract triage; not full paper review. Full-read priority: P1.

**Primary sources:** [source 1](https://arxiv.org/abs/2306.04050v2).

**Source facts:** Uses an LLM next-token predictor with lossless coding to estimate compressibility; preliminary comparison uses LLaMA-7B.

**Our proposal:** Implement a tiny arithmetic/range-coding roundtrip and compare actual length with summed negative log probability. Signature figure: Actual bits versus ideal code length across predictable and shuffled byte strings. Prediction before running: “Does a lower predictive cross-entropy always yield smaller complete files once headers and model costs are counted?” Controls: Uniform and empirical unigram codes; byte shuffle; exact roundtrip; tokenizer and model-side-information accounting.

**Placement / disposition:** core integration; 0.6 How to Know When an Interpretability Result Is Fake (Practice), 11.1 PCA, SVD and Geometry Controls. Turns cross-entropy and bits into an observable foundations result without downloading LLaMA.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Wrong journalArticle type for the verified source: use preprint unless a separately verified publication is found. Missing URL resolved to arXiv:2306.04050v2; arXiv DOI 10.48550/arXiv.2306.04050. Journal/conference publication not established here.

### INB082 — OneCAT: Decoder-Only Auto-Regressive Model for Unified Understanding and Generation

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2509.03498v3. primary metadata and abstract triage; not full paper review. Full-read priority: P3.

**Primary sources:** [source 1](https://arxiv.org/abs/2509.03498v3).

**Source facts:** OneCAT combines multimodal understanding and generation using decoder-only autoregression and modality-specific experts.

**Our proposal:** Trace image-token scale hierarchy and identify which adapter/expert handles each modality. Signature figure: Token-flow diagram and latency decomposition by image scale. Prediction before running: “Does a single autoregressive objective remove modality-specific engineering?” Controls: Matched image resolution; external-tokenizer comparison; active versus total parameter accounting.

**Placement / disposition:** background; 12.3 Mini VLM from Scratch, 13.6 Autoregressive Image Tokens. Useful architecture contrast; no addition of a large new family while pilots are unfinished.

**Replication and access:** faithful paper replication. compute: UNKNOWN for faithful replication; model training/inference resources not profiled; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Pin verified primary version 2509.03498v3 for future full reading; original inbox record does not reliably pin this exact version.

### INB083 — A Sovereign, Open-Source Foundation Model for German and English

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2607.09424v3. primary metadata and abstract triage; not full paper review. Full-read priority: P3.

**Primary sources:** [source 1](https://arxiv.org/abs/2607.09424v3).

**Source facts:** Soofi S is a German/English hybrid Mamba-Transformer MoE; release claims include mixture accounting and partial licensed-source statistics.

**Our proposal:** Design a transparent resource/model-card comparison without loading the model. Signature figure: Cache growth and active-versus-total parameters, labeled as reported or unmeasured. Prediction before running: “Can active parameter count alone predict throughput across different cache architectures?” Controls: Equal context, batch, precision and hardware; dense/hybrid baselines; data-license accounting.

**Placement / disposition:** defer out of scope; 5.2 Mixture of Experts from Scratch, 5.7 Griffin, xLSTM, Hyena and Hybrid Models. A 30B-total model and sovereign infrastructure are peripheral to current pilot completion.

**Replication and access:** faithful paper replication. compute: UNKNOWN for faithful replication; model training/inference resources not profiled; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: Abstract distinguishes permissively releasable data-construction artifacts from commercially licensed source accounting; full data access UNKNOWN.

**Bibliographic proposals only:** Pin verified primary version 2607.09424v3 for future full reading; original inbox record does not reliably pin this exact version.

### INB084 — Language Generation with Strictly Proper Scoring Rules

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2405.18906v1. primary proceedings abstract, selected scoring-rule method text and official code README; not full paper review. Full-read priority: P1.

**Primary sources:** [source 1](https://proceedings.mlr.press/v235/shao24c.html), [source 2](https://arxiv.org/abs/2405.18906v1), [source 3](https://github.com/shaochenze/ScoringRulesLM).

**Source facts:** Adapts proper scores to token-level conditional distributions; official code warns results differ between pretraining and fine-tuning.

**Our proposal:** Implement log, Brier and spherical scores on a finite categorical simplex and derive their expected optimum. Signature figure: Expected loss landscapes and reliability curves with deliberately overconfident predictions. Prediction before running: “Will a score reward honest probabilities even when top-1 accuracy is unchanged?” Controls: Misreported probability vectors; log-score baseline; fixed supports; sign-convention tests; separate scratch versus fine-tuning claims.

**Placement / disposition:** core integration; 0.6 How to Know When an Interpretability Result Is Fake (Practice), 11.1 PCA, SVD and Geometry Controls. Foundational distinction among a training loss, calibration, decision metrics and token versus sequence scoring.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: Official README inspected; fairseq-based scoring_rule_loss.py is named. README says scratch training underperforms CE while CE-pretrained fine-tuning improves; no execution or install.; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Resolve missing URL/date to ICML 2024, PMLR 235:44474–44488; type conferencePaper. Associated arXiv DOI 10.48550/arXiv.2405.18906; no publisher DOI verified.

### INB085 — Selectivity and Tolerance (“Invariance”) Both Increase as Visual Information Propagates from Cortical Area V4 to IT

**Identity / read depth:** VERIFIED primary title/author/identifier match; publisher/author record. primary metadata and abstract triage; not full paper review. Full-read priority: P2.

**Primary sources:** [source 1](https://www.jneurosci.org/content/30/39/12978), [source 2](https://www.jneurosci.org/content/jneuro/30/39/12978.full.pdf), [source 3](https://pmc.ncbi.nlm.nih.gov/articles/PMC2975390/).

**Source facts:** Rust and DiCarlo compare population selectivity and tolerance in V4 and IT; tolerance involves transformation generalization, not merely narrow tuning.

**Our proposal:** Implement identity-across-transform readout on synthetic response populations; interpret biological experiment separately. Signature figure: Identity decoding matrices across position/scale alongside selectivity curves. Prediction before running: “Can a population gain both selectivity and tolerance without every neuron becoming invariant?” Controls: Matched population size/SNR; within-transform versus cross-transform decoding; identity scrambling.

**Placement / disposition:** background; 12.7 Dense Vision Features - DINO and SAM, 11.2 CCA, SVCCA and Representation Alignment. Clarifies ambiguous invariance language; artificial response simulations cannot reproduce biological evidence.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

### INB086 — Beyond SAEs: Consistent Concept Discovery with Sparse Coding

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2610.23801v1. primary metadata/abstract plus selected method sections and availability statements; not full paper review. Full-read priority: P1.

**Primary sources:** [source 1](https://arxiv.org/abs/2610.23801v1).

**Source facts:** ConSDL alternates FISTA sparse inference, support refitting and dictionary updates with a co-active similarity penalty.

**Our proposal:** Implement nonnegative FISTA and support-only refit, then match learned atoms across seeds. Signature figure: Dictionary-match matrices and sparsity/reconstruction/consistency frontiers. Prediction before running: “Can two dictionaries reconstruct equally well yet disagree about recovered concepts?” Controls: Same-L0 SAE; no-refit/no-regularizer ablations; fixed inference budget; planted truth; chance dictionary matching.

**Placement / disposition:** existing lesson addition; 6.1 SAE Variants, 6.3 Feature Splitting, Absorption and Dashboards. An explicit amortized-inference versus per-sample optimization comparison improves the existing SAE pilot.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: Primary paper links official code, model weights and project page; contents/licenses/complete reproduction UNKNOWN; no artifacts downloaded; data: UNKNOWN; not checked.

### INB087 — Proving and Improving the Platonic Representation Hypothesis with JEPAs

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2610.39302v1. primary identity/abstract triage only here; substantive review delegated to prh_jepa.md. Full-read priority: P1.

**Primary sources:** [source 1](https://arxiv.org/abs/2610.39302v1).

**Source facts:** Primary identity confirmed for the PRH-with-JEPAs follow-up; theorem and empirical claims are reviewed in the dedicated companion.

**Our proposal:** See prh_jepa.md; do not duplicate theorem assumptions or merge with the earlier PRH paper. Signature figure: See dedicated view-quality/alignment exercise in prh_jepa.md. Prediction before running: “Does an alignment theorem apply when positive-pair assumptions fail?” Controls: Dedicated companion specifies view, rotation and leakage controls.

**Placement / disposition:** existing lesson addition; 11.2 CCA, SVCCA and Representation Alignment, 14.4 The JEPA Family. Delegate substantive appraisal to the named PRH owner; this row preserves inbox coverage.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

Dedicated follow-up: [PRH/JEPA review](prh_jepa.md). This is the newer follow-up, not a duplicate of the original PRH paper.

### INB088 — A Mechanistic Understanding of Pronoun Fidelity in LLMs

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2606.16407v1. primary metadata and abstract triage; not full paper review. Full-read priority: P2.

**Primary sources:** [source 1](https://arxiv.org/abs/2606.16407v1).

**Source facts:** Uses Boundless DAS to analyze simultaneous group-binding, recency and stereotypical subspaces in pronoun fidelity.

**Our proposal:** Implement a counterbalanced pronoun template grid and interchange-intervention accuracy scoring. Signature figure: Mechanism-by-layer interchange accuracy with competing hypotheses. Prediction before running: “Does a high behavioral fit establish one exclusive mechanism when several subspaces coexist?” Controls: Pronoun/name/occupation swaps; held-out templates; random subspaces; donor interchange; non-target retention.

**Placement / disposition:** existing lesson addition; 11.3 Causal Abstraction and Distributed Alignment Search, 9.5 White-Box Monitors and Red-Teaming with Internals. A concrete multi-mechanism application of DAS; retain fairness categories and template limits explicitly.

**Replication and access:** bounded real study. compute: UNKNOWN for faithful replication; model training/inference resources not profiled; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Wrong book type: verified source is a preprint. Missing DOI resolved to arXiv DOI 10.48550/arXiv.2606.16407; primary preprint v1, 15 Jun 2026. Semantic Scholar link is discovery-only.

### INB089 — Neural Networks Are Already Clean

**Identity / read depth:** VERIFIED primary title/author/identifier match; publisher/author record. author essay introduction, provenance and selected claims; not complete essay review. Full-read priority: P2.

**Primary sources:** [source 1](https://already-clean.goodfire.com/).

**Source facts:** Author labels this a working hypothesis and informal essay about native neurons and circuits, challenging interpretations of polysemanticity.

**Our proposal:** Compare blinded neuron-level and SAE-feature explanations on a fixed small set rather than accepting a paradigm claim. Signature figure: Coverage versus intervention specificity for native neurons and learned features. Prediction before running: “Does an interpretable-looking native neuron have causal specificity or reflect selection bias?” Controls: Random neuron/feature samples; blind labels; held-out examples; equal selection budgets; causal ablations.

**Placement / disposition:** background; 6.1 SAE Variants, 6.3 Feature Splitting, Absorption and Dashboards. Useful adversarial reading for SAE assumptions, not established evidence that superposition is solved.

**Replication and access:** bounded real study. compute: UNKNOWN for faithful replication; model training/inference resources not profiled; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Author is Nick Cammarata; Goodfire is publisher/affiliation, not part of the personal surname. Web essay/working hypothesis, 2026; exact day and DOI UNRESOLVED.

### INB090 — Make it SING: Analyzing Semantic Invariants in Classifiers

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2603.14610v2. primary metadata and abstract triage; not full paper review. Full-read priority: P2.

**Primary sources:** [source 1](https://arxiv.org/abs/2603.14610v2).

**Source facts:** SING studies semantic variation in classifier null-space directions using multimodal feature mappings.

**Our proposal:** Implement an SVD null-space projector and verify logit preservation for a linear classifier before nonlinear image claims. Signature figure: Input/feature variations with logit drift and retained semantic attributes. Prediction before running: “Can identical classifier outputs coexist with a large change in semantic content?” Controls: Row-space perturbation; norm-matched random direction; numerical tolerance; off-manifold check.

**Placement / disposition:** existing lesson addition; 11.1 PCA, SVD and Geometry Controls, 12.7 Dense Vision Features - DINO and SAM. A visible linear-algebra bridge from SVD to invariance; semantic captions are not ground truth by themselves.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Pin verified primary version 2603.14610v2 for future full reading; original inbox record does not reliably pin this exact version.

### INB091 — Self-Play Pretraining with Zero Data

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2609.30063v1. primary metadata and abstract triage; not full paper review. Full-read priority: P3.

**Primary sources:** [source 1](https://arxiv.org/abs/2609.30063v1).

**Source facts:** Uses program-generating self-play to train a learner without natural training data and reports transfer scaling.

**Our proposal:** Specify a bounded synthetic sequence curriculum and loss accounting; retain the universal-machine claim as paper reading. Signature figure: Held-out natural-data loss versus total charged generator-plus-learner compute. Prediction before running: “Does adaptation beat a fixed program distribution after generator cost is included?” Controls: Fixed synthetic curriculum; random programs; equal combined compute; untouched transfer data.

**Placement / disposition:** optional replication; 17.4 Training Dynamics Across Architectures, 14.5 Maze Transformers and World Models. An interesting training-dynamics capstone, but a universal-program self-play campaign is outside pilot work.

**Replication and access:** faithful paper replication. compute: UNKNOWN for faithful replication; model training/inference resources not profiled; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

### INB092 — Rethinking Projector Training in Multimodal LLMs

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2610.18598v1. primary metadata/abstract plus selected method sections and availability statements; not full paper review. Full-read priority: P1.

**Primary sources:** [source 1](https://arxiv.org/abs/2610.18598v1).

**Source facts:** PORTAL initializes a projector using marginal means and PCA spreads; it skips projector alignment training, not the subsequent multimodal training pipeline.

**Our proposal:** Implement centered top-k PCA affine moment matching and audit preactivation versus post-GELU moments. Signature figure: Covariance alignment and held-out paired retrieval side by side. Prediction before running: “Can perfect marginal covariance matching still give incorrect image-text pairing?” Controls: Random orthogonal map; paired Procrustes oracle; shuffled pairs; rank/eigenvalue-gap sweeps; no test-set fitting.

**Placement / disposition:** existing lesson addition; 12.3 Mini VLM from Scratch, 11.1 PCA, SVD and Geometry Controls, 11.2 CCA, SVCCA and Representation Alignment. Strong foundations-to-Mini-VLM bridge, provided moment matching is not conflated with semantic correspondence.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: Primary reproducibility checklist names https://github.com/mlbio-epfl/PORTAL; repository contents and runability UNKNOWN; data: UNKNOWN; not checked.

### INB093 — VISOR++: Universal Visual Inputs based Steering for Large Vision Language Models

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2509.25533v2. primary metadata and abstract triage; not full paper review. Full-read priority: P2.

**Primary sources:** [source 1](https://arxiv.org/abs/2509.25533v2).

**Source facts:** VISOR++ optimizes visual inputs to induce target activation patterns and reports behavioral transfer across VLMs.

**Our proposal:** Compare activation steering with a benign visual task-direction surrogate and distinguish white-box crafting from input-only application. Signature figure: Transfer matrix: source/target models and task behavior versus utility. Prediction before running: “Does input-only deployment imply that the steering image was obtained without internal access?” Controls: Random/norm-matched images; activation steering; unseen model and held-out task; unchanged utility.

**Placement / disposition:** optional replication; 9.1 Refusal Directions and Representation Engineering, 12.2 CLIP Features, SAEs and Typographic Attacks, 12.5 Object Hallucination and Modality Arbitration. A useful intervention-access contrast; substantial VLM optimization and API access remain separately authorized.

**Replication and access:** bounded real study. compute: UNKNOWN for faithful replication; model training/inference resources not profiled; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Pin verified primary version 2509.25533v2 for future full reading; original inbox record does not reliably pin this exact version.

### INB094 — Steered LLM Activations are Non-Surjective

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2604.09839v2. primary metadata and abstract triage; not full paper review. Full-read priority: P1.

**Primary sources:** [source 1](https://arxiv.org/abs/2604.09839v2).

**Source facts:** Argues under stated assumptions that continuously steered activations almost surely lack an exact discrete-prompt preimage.

**Our proposal:** Enumerate all short prompts of a finite toy encoder and compare steering points with the finite reachable set. Signature figure: Reachable activation cloud, steered points and nearest-prompt distances. Prediction before running: “Does absence of an exact prompt preimage prevent the same output or approximate behavioral equivalence?” Controls: Unsteered reachable points; random equal-norm shifts; prompt-length bounds; exact versus approximate matching.

**Placement / disposition:** existing lesson addition; 9.1 Refusal Directions and Representation Engineering, 11.1 PCA, SVD and Geometry Controls. An essential limit on interpreting steering as prompting; finite enumeration is an illustration, not a proof for arbitrary models.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Pin verified primary version 2604.09839v2 for future full reading; original inbox record does not reliably pin this exact version.

### INB095 — What Do Visual-Head Discovery Scores Actually Find? Attention Source, Necessity, and Random-Control Pitfalls in Vision-Language Models

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2610.13111v1. primary metadata and abstract triage; not full paper review. Full-read priority: P1.

**Primary sources:** [source 1](https://arxiv.org/abs/2610.13111v1).

**Source facts:** Same arXiv DOI as INB002; primary paper distinguishes head steerability from necessity using layer-matched random sets.

**Our proposal:** Refer to canonical INB002 review; retain this bibliographic row without duplicate integration work. Signature figure: Refer to INB002 necessity versus steerability figure. Prediction before running: “Could random head sets steer just as well while failing necessity tests?” Controls: Layer-matched random sets; matched cardinality; discovery/evaluation splits; multiple comparisons.

**Placement / disposition:** duplicate; 12.4 Visual Token Flow in Real VLMs, 12.5 Object Hallucination and Modality Arbitration, 0.6 How to Know When an Interpretability Result Is Fake (Practice). Duplicate-by-DOI receipt only; no Zotero merge or mutation.

**Replication and access:** bounded real study. compute: UNKNOWN for faithful replication; model training/inference resources not profiled; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Duplicate of INB002 by identical DOI 10.48550/arXiv.2610.13111, title and authors; keep both inbox records and one canonical review.

### INB096 — Correcting Attribution Patching with a Few Exact Measurements

**Identity / read depth:** VERIFIED primary title/author/identifier match; 2610.27393v1. primary metadata/abstract plus selected method sections and availability statements; not full paper review. Full-read priority: P1.

**Primary sources:** [source 1](https://arxiv.org/abs/2610.27393v1).

**Source facts:** Guided-AP uses exact pilot measurements to estimate shared signed residuals and correct unmeasured attribution estimates.

**Our proposal:** Implement exact finite interventions, first-order estimates and pilot-median residual correction at a fixed charged budget. Signature figure: MAE and top-k recall versus exact-patch budget, with calibration and held-out prompts separate. Prediction before running: “When does a small signed correction beat verifying only the largest attribution estimates?” Controls: Verify-more; random pilot; wrong-sign/shuffled residuals; unbiased-error task; identical total acquisitions.

**Placement / disposition:** existing lesson addition; 8.2 Attribution Patching and EAP, 8.3 AtP* from Scratch. A direct efficiency and failure-analysis follow-up to attribution patching; full large-model replication is separate.

**Replication and access:** CPU illustration. compute: UNKNOWN for faithful replication; CPU-sized pedagogical illustration proposed, not benchmarked; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

### INB097 — Dissecting the Safety Circuit: Neuronal Intervention for Transferable Adversarial Attacks on VLMs

**Identity / read depth:** VERIFIED primary title/author/identifier match; publisher/author record. primary metadata and abstract triage; not full paper review. Full-read priority: P2.

**Primary sources:** [source 1](https://proceedings.mlr.press/v306/xie26f.html).

**Source facts:** SCIA uses safety-related circuit interventions to study transfer of adversarial VLM perturbations; official record is ICML 2026 proceedings.

**Our proposal:** Read methods and design benign policy-preference transfer tests before any attack implementation. Signature figure: Source-target intervention transfer with utility and circuit-localization controls. Prediction before running: “Does a linearly probeable safety feature identify a necessary and transferable circuit?” Controls: Probe-only versus causal ablation; random neurons; norm-matched edits; held-out models; task-utility retention.

**Placement / disposition:** optional replication; 9.5 White-Box Monitors and Red-Teaming with Internals, 12.5 Object Hallucination and Modality Arbitration. Useful mechanistic-transfer critique; publisher type correction is concrete, while attack execution is outside this reading task.

**Replication and access:** bounded real study. compute: UNKNOWN for faithful replication; model training/inference resources not profiled; access: Primary text accessible; model/API/dataset permissions and licenses UNKNOWN unless explicitly stated below; code: UNKNOWN; not checked; data: UNKNOWN; not checked.

**Bibliographic proposals only:** Wrong journalArticle type: official publisher records ICML 2026 conference paper, PMLR 306:139323–139338, 6–11 Jul 2026. Missing URL resolved. DOI UNRESOLVED; none asserted.

## Six controlled exercise designs for a full-read queue

### INB068 — Make the student pay to describe every label

The [online-code construction, Eq. 3](https://aclanthology.org/2020.emnlp-main.14.pdf) evaluates successive unseen blocks after fitting on the transmitted prefix. Our proposed exercise starts with a four-class planted representation and two label assignments with similar eventual probe accuracy. Students implement `uniform_bits`, `block_nll_bits` and `prequential_bits`, with hand-checkable uniform/perfect-prediction tests. They predict which assignment is cheaper before seeing cumulative-bit and accuracy panels. Group identifiers prevent duplicated entities leaking across blocks; type-consistent random labels test memorization separately from a fully shuffled label null. The play cell changes representation noise and probe capacity while preserving the block schedule. An interpretation dropdown explains why cheap decodability does not imply causal use. A bounded real study would later use preexisting, licensed representations and untouched evaluation groups. This CPU illustration neither reproduces the paper's ELMo/Penn Treebank setting nor establishes a language-model mechanism.

### INB084 — Accuracy can stay fixed while the honest forecast wins

The [paper](https://proceedings.mlr.press/v235/shao24c.html) and [official README](https://github.com/shaochenze/ScoringRulesLM) motivate a careful separation between mathematical propriety and empirical fine-tuning benefits. Our exercise enumerates a three-outcome distribution, then asks students to implement log, Brier and spherical losses and compute their *expected* loss under a known truth. A simplex heatmap shows the optimum, and a companion reliability panel compares truthful and overconfident predictions with identical argmax decisions. Unit checks cover sign conventions, zero-probability log loss and score invariance to batch duplication. The prediction asks which forecasts can tie on accuracy while differing on all three losses. A play cell shifts the true distribution and recalculates the optimum; no model training is needed. Discussion separates token-conditional sums from scoring the full sequence distribution. Any later optimization study must control gradient scale, initialization and tuning budget, and separate scratch training from CE-pretrained fine-tuning; the README explicitly cautions that these regimes differ.

### INB086 — Equal reconstruction does not identify a unique concept basis

The [ConSDL method](https://arxiv.org/html/2610.23801v1) makes sparse inference and coefficient refitting inspectable. Our exercise begins with known dictionary atoms and generated nonnegative sparse coefficients. Students implement nonnegative shrinkage, FISTA updates, support-only refitting and atom matching across seeds; a fixed-dictionary test checks objective descent before dictionary learning. They predict whether a learned encoder and per-sample optimization can reconstruct equally well yet recover different planted factors. Signature panels show truth-to-learned matching and reconstruction versus L0 versus cross-seed consistency. Controls use equal sparsity, inference-step budgets, no-refit/no-regularizer ablations, duplicate atoms and a chance-matching null. A play cell increases atom correlation until identifiability becomes ambiguous. The solution explains why consistency can also reflect a shared wrong solution. The CPU organism is a pedagogical subset; a faithful ConSDL replication also needs the exact co-active penalty, optimizer and model-activation protocol, which remain a full-read/replication task.

### INB096 — Spend exact interventions where they teach the most

[Guided-AP's method](https://arxiv.org/html/2610.27393v1) separates choosing what to verify from learning a signed correction. Our proposed organism has a differentiable scalar output with controlled curvature, so every exact finite intervention can be computed and hidden as reference truth. Students implement first-order effects, a magnitude-stratified pilot, median signed-residual correction, and budget accounting. They predict when correction wins over verifying the largest estimates and when mixed-sign residuals make it fail. The figure shows MAE and top-k recovery against **total** exact interventions, including pilot acquisition; calibration prompts and untouched evaluation prompts are separate. Wrong-sign corrections, shuffled pilot residuals, an unbiased-error case, random allocation and verify-more are necessary controls. A play cell varies curvature and shared-error strength. Full large-model claims require exact component populations, timing and selection rules; the toy result cannot establish a universal patching speedup or discovery guarantee.

### INB092 — Match the clouds, then ask whether their meanings match

[PORTAL's construction](https://arxiv.org/html/2610.18598v1) maps modality marginals through PCA directions and spreads. Our exercise implements mean/covariance estimation on a training partition, an affine rank-k map, and a covariance check, then deliberately swaps semantic identities while leaving the marginals unchanged. Students predict whether excellent moment agreement forces correct paired retrieval. The signature figure juxtaposes transformed point clouds and a retrieval matrix; scrambled correspondences can look geometrically excellent while failing semantics. Controls include a paired Procrustes oracle, random rotations, rank/eigenvalue-gap sweeps and an independent evaluation partition. Immediate tests check centering, matrix dimensions and low-rank numerical behavior. The play cell changes semantic permutation and compares preactivation moments with post-GELU outputs. This CPU geometry exercise cannot reproduce multimodal benchmark retention. “Training-free” refers to the projector initialization/alignment stage; the paper's pipeline still includes subsequent multimodal training.

### INB069 — Does the blind spot disappear or move?

[GRAFT's algorithm](https://arxiv.org/html/2610.17062v1) includes retain/forget routing, deployment-configuration anchors and special optimizer-moment handling. Our exercise first uses a two-branch synthetic learner with known useful and unwanted labels, an explicit weak-monitor blind spot and no harmful task content. Students implement per-example gradient masks and verify that flagged examples produce zero retain-branch gradient while still updating the sacrificial branch. They predict both true unwanted behavior and monitor-detected behavior before and after branch removal; the two curves must remain separate. Controls compare ordinary training, filtering, reward-style penalties, random routing and an oracle monitor at matched capacity. A play cell varies blind-spot correlation, then tests unseen contexts. This illustrates gradient localization only. Calling it faithful GRAFT would require the actual adapter architecture, GRPO baseline handling, anchors, unrouted second moments and training distribution. Full-paper reading must precede that separately budgeted real study.

## Boundaries and outstanding reading

No paper is marked fully read. The six exercise designs need full method/appendix review before implementation; theorem-heavy INB094 and the dedicated INB087 review need assumption-level checking. Missing publisher DOIs for INB084/INB097, the exact date/DOI of INB089, and publication versions beyond the verified preprints remain UNRESOLVED. Code/data access is UNKNOWN unless a primary source or inspected README explicitly establishes a narrower fact. A public link or promised release is not a completed reproducibility audit.

This batch proposes student implementations, signature figures, controls and play cells; none was executed. Original ARENA files, outputs and assets were not edited. Static checks cover record coverage, required fields, existing lesson IDs and absence of private item identifiers only. They establish document consistency, not scientific results, notebook quality, native execution or learner acceptance.

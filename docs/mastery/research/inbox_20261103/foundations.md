# Foundations that make representation research understandable

Status: **planning proposal, 2026-11-03; no lesson implementation, experiment, model access, or learner acceptance**. This is an additive bridge into the existing course, not a request for more topics. Root Map/Spec integration owns placement and adoption. Public sources below were retrieved 2026-11-03. Equations labeled exact are mathematical definitions/derivations; predictions of algorithm behavior require empirical checking. Exercise and assessment designs are our proposals, not results reported by the cited authors.

The user wants substantial intuition: why UMAP can be useful, what t-SNE trades away, how PCA works, what losses and metrics measure, and why a statistical correction changes an inference. The intended outcome is a learner who can **forecast, calculate, implement, see a failure, and revise a claim** before approaching PRH/JEPA expertise. “UMAP is universally best” is not a learning objective.

## Reuse the original course, preserve it

Original ARENA 0.0 already introduces linear transformations, expectation/variance, derivatives and information theory; its exercises include row normalization, cosine similarity, sampling, accuracy, stable softmax/log-softmax, cross-entropy and einsum. Link those exercises rather than rewriting them. Original 1.1 connects attention geometry to next-token loss. Current extension 11.1 already includes centered PCA, circular probes, trustworthiness/continuity, nonlinear-view controls and a historical calendar study. These are local source observations, not claims that their learner experience has been accepted.

Read-only anchors: `chapter0_fundamentals/exercises/part0_prereqs/0.0_Prerequisites_exercises.ipynb`, its `solutions.py` functions `normalize_rows`, `cos_sim_matrix`, `batched_cross_entropy_loss`; `chapter1_transformer_interp/exercises/part1_transformer_from_scratch/solutions.py:get_log_probs`; `chapter11_representation_geometry/instructions/pages/01_[11.1]_PCA_SVD_and_Geometry_Controls.md`; `chapter11_representation_geometry/exercises/part1_pca_svd_geometry_controls/geometry_methods.py:neighborhood_scores`.

## Proposed prerequisite ladder

Every row follows: read canonical explanation → explain a tiny example → write forecast and confidence → implement the core → run immediate tests → inspect a visible result against controls → explain the mismatch → forecast an unrevealed changed case. Help, interpretation and full-solution dropdowns remain separate. Revealing an answer marks guided practice; it cannot become unseen assessment later.

| Node | One concrete question | Learner core / signature result | Meaningful test, control and transfer |
|---|---|---|---|
| F0: quantities and probability | Which number changes if I change units, axes, or the independent unit? | Annotate samples×features, center/normalize a tiny matrix, calculate expectation/variance and conditional probability; distance heatmap before/after a unit change. | Assert axis-specific answers, zero-vector policy, exact four-point distances; repeat a subject's row and explain why this adds no independent subject. Transfer to grouped images/texts. |
| F1: objectives and gradients | Can two predictors have the same accuracy and different loss? | Implement stable log-softmax, CE, discrete KL, MSE and cosine; contribution bars and loss-vs-prediction curves. | Hand answers below; normalization/support/shape failures; constant predictor, permuted labels, rescaled vectors. Transfer from token distribution to pair affinities to a latent target. |
| F2: PCA/SVD geometry | Does keeping 80% of variance keep the information I need? | Center → SVD → project → reconstruct; arrows, scree curve and reconstruction-error curve with labels revealed afterward. | Orthogonal projector and residual identities; exact rectangle below; random projection and no-reduction baselines. Transfer low-variance signal / high-variance nuisance with changed feature units. |
| F3: nonlinear maps | What is a local-neighbor picture allowed to claim? | Implement tiny affinities, KL, UMAP fuzzy union and edge loss; linked original-neighbor/pair-affinity heatmaps and PCA/t-SNE/UMAP views. Library fit is plumbing only after the objective is visible. | Symmetry, diagonals, normalization vs non-normalization, small hand affinities; independent noise, shuffled labels, original-space neighbor retrieval, seeds/settings. Transfer disconnected graph, density shift and changed metric. |
| F4: metrics and evidence | Did the view preserve neighbors, labels, probabilities, or a causal effect? | Implement neighbor overlap and rank penalties; trustworthiness/continuity vs k beside original-space kNN and metric/confusion/reliability panels. | Identity/isometry score 1; deliberately inject a false neighbor and delete a real neighbor; row-ID misalignment must fail. Transfer class imbalance, grouped holdout, nonlinearity and calibration shift. |
| F5: inference and selection | Is one small p-value evidence after inspecting many alternatives? | Implement Bonferroni, a family manifest and a simple exact null enumeration; plot individual thresholds vs family error, retaining all attempts. | Boundary p=α/m, valid-p null, perfectly dependent tests, independent tests; a selected-minimum counterexample. Transfer changed family, adaptive stopping and subject-vs-pair replication. |

Dependency graph: **F0 → F1 → F3; F0 → F2 → F3; F0+F1+F2 → F4; F0 → F5; F3+F4+F5 → representation-geometry/PRH claims**. F1+F2 feed JEPA loss/representation exercises; F4+F5 feed their evaluation. F2+F4 precede probes, lenses and kernel alignment. These edges are proposed readiness dependencies, not completion claims or a new chapter numbering scheme.

## F1: definitions the learner can calculate

For normalized discrete distributions P,Q and natural logs:

\[
H(P)=-\sum_i p_i\log p_i,\quad
H(P,Q)=-\sum_i p_i\log q_i,\quad
D_{KL}(P\Vert Q)=\sum_i p_i\log\frac{p_i}{q_i}=H(P,Q)-H(P).
\]

Zero p contributes zero; positive p with zero q gives infinite KL. KL is directional, is nonnegative for probability distributions, and is not a metric. Minimizing cross-entropy equals minimizing this KL **only when P is fixed**. Official formula reference: [SciPy entropy, definition and Notes](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html). Numerical implementation must state its reduction axis and log base; silently clipping q changes the objective.

**Own exact worked example:** P=(.9,.1), Q=(.5,.5): H(P)=.325083 nats, H(P,Q)=.693147, KL(P||Q)=.368064; KL(Q||P)=.510826. Student calculates each category's contribution before code; individual contributions can be negative while their sum is nonnegative. Next change Q to (.99,.01) and forecast which direction is more sensitive to the rare outcome before revealing the answer.

For logits z, \(\log q_i=z_i-[m+\log\sum_j e^{z_j-m}]\), m=max(z). A one-hot target gives CE=−log(q_true). For a vector target t and prediction h, explicitly choose \(\|h-t\|^2/d\) or another declared reduction; compare with \(1-h^Tt/(\|h\|\|t\|)\). **Own derivation:** if h=2t≠0, cosine loss is zero while MSE=\(\|t\|^2/d\); if both representations collapse to one constant, pairwise latent MSE can be zero without useful information. This motivates checking scale, target construction, variance and downstream evidence in JEPA; it does not assert that a particular JEPA implementation collapses or uses this exact loss.

Minimum calculus bridge: implement the gradient of \(\frac12(h-t)^2\), and derive CE's logit gradient q−y for one item. Test translation invariance of log-softmax and finite outputs for large logits; test gradient direction independently using a small hand calculation. Help explains why probabilities, logits, and latent vectors have different contracts. No generic “all losses are interchangeable” exercise.

### Just-in-time loss map, rather than a catalog to memorize

| Existing question | Objective to unpack | One decisive distinction / proposed control |
|---|---|---|
| Token/class probability | CE and KL above | Target support, direction and reduction matter; compare constant probabilities and equally accurate predictors. |
| Predict a continuous latent target | MSE or cosine above, as specified by the actual recipe | Magnitude error vs angular error; rescale predictions, and inspect constant-output collapse. |
| Identify the paired text/image or prediction/target among candidates | Paired InfoNCE below | Positive identity, negative sampling and temperature define the task; shuffle pair IDs and insert a semantically duplicate negative. |
| Prevent trivial representations in an applicable self-supervised recipe | Advanced variance/covariance regularization | Inspect batchwise standard deviations and off-diagonal covariance; do not equate a noncollapsed batch with useful representations. |

For a batch of B aligned pairs, let normalized vectors u_i and v_i have scores \(s_{ij}=u_i^Tv_j\), temperature τ>0, one designated positive j=i, and negatives j≠i. A row-direction paired InfoNCE loss is

\[
\ell_i^{u\to v}=-\log\frac{\exp(s_{ii}/\tau)}{\sum_{j=1}^{B}\exp(s_{ij}/\tau)},\qquad
L_{sym}=\frac1{2B}\sum_i(\ell_i^{u\to v}+\ell_i^{v\to u}).
\]

The denominator includes the positive. Bidirectionality swaps the candidate/query roles, equivalently the score-matrix axes. [CPC, §2.3, equation 4](https://arxiv.org/pdf/1807.03748) introduces InfoNCE; [CLIP, §2.3 and Figure 3](https://proceedings.mlr.press/v139/radford21a/radford21a.pdf) supplies the symmetric normalized image/text recipe. The [VL-JEPA v1 methodology](https://arxiv.org/html/2512.10942v1#S2) explicitly adopts bidirectional InfoNCE; the JEPA name alone does not imply MSE, negatives, EMA targets, or a universal anti-collapse mechanism.

**Own exact proposed mini-example:** B=2 with all four scores equal gives each directional loss log 2. If diagonal scores are 1 and off-diagonals 0, each is \(\log(1+e^{-1/\tau})\). Forecast what decreasing τ does when a wrong pair instead has the larger score; then inspect per-pair contributions. A high-scoring negative may be a valid alternate caption, not a truly dissimilar item: false negatives are a substantive contrastive-learning issue, not a code failure. [Chuang et al. 2020, abstract and §2](https://proceedings.neurips.cc/paper/2020/file/63c3ddcc7b23daa1e42dc41f9a44a873-Paper.pdf). This example is a design, not an executed contrastive experiment.

Advanced recipe-specific connection: VICReg combines an invariance term with batchwise standard-deviation and off-diagonal-covariance regularizers. [Bardes et al., §4.1, equations 1–6](https://arxiv.org/html/2105.04906v3#S4.SS1). Explain which axis is the batch and what a constant representation does to each term; do not transfer the recipe or an “EMA prevents collapse” claim to all JEPAs. MAE/Huber robust-regression choices stay deferred unless an actual numeric world-state target, units and outlier/error model make that comparison relevant.

## F2: exact geometry, then a counterexample to a tempting claim

For X∈R^(n×d), \(X_c=X-\mathbf1\mu^T=U\Sigma V^T\). Select k right singular vectors V_k:

\[
Z=X_cV_k,\quad\widehat X=\mathbf1\mu^T+ZV_k^T,\quad
\lambda_j=\sigma_j^2/(n-1),\quad
\mathrm{EVR}_k=\frac{\sum_{j\le k}\sigma_j^2}{\sum_j\sigma_j^2}.
\]

PCA centers; standardizing features and whitening scores are additional operations. Components maximize retained variance; the covariance convention uses n−1. [Official PCA reference, description and Attributes](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). Exact consequence for truncated SVD: squared Frobenius reconstruction error is \(\sum_{j>k}\sigma_j^2\); this is a reconstruction objective, not a label-prediction objective.

**Own exact worked example:** rows (−2,−1),(−2,1),(2,−1),(2,1); singular values 4,2, EVR_1=.8 and total rank-one squared error=4. Let labels be sign(y). Keeping the first component discards the perfectly predictive coordinate. Make the y-axis ten times larger; forecast the new component and explain the changed units. Test V_k^TV_k=I and projector idempotence; compare projectors/reconstructions rather than component signs. Repeated singular values permit different equally valid bases.

Dimension choice is task-dependent: propose rank-feasible candidates from the spectrum and test reconstruction plus original→intermediate neighborhood retention and held-out task evidence. A “95% variance” rule or fixed “50 PCs” is a candidate, never a universal answer. Centering, feature standardization, row normalization, whitening and block weighting get separate names and forecasts. Fit all learned preprocessing only within the fitting split for generalization claims.

## F3: t-SNE and UMAP intuition with the mathematical boundary visible

**t-SNE, source facts.** Gaussian conditional affinities are symmetrized; perplexity is effective neighbor count, not cluster count. The heavy-tailed Student-t kernel provides room to separate neighbors under crowding. Its objective weights missing high-P affinities strongly; this does not calibrate far-apart map distances. See [van der Maaten & Hinton 2008, pp. 2581–2586, equations 1–5, §3.1](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf).

\[
p_{j|i}=\frac{e^{-\|x_i-x_j\|^2/(2\sigma_i^2)}}{\sum_{k\ne i}e^{-\|x_i-x_k\|^2/(2\sigma_i^2)}},\quad
p_{ij}=\frac{p_{j|i}+p_{i|j}}{2n},\quad
q_{ij}=\frac{(1+\|y_i-y_j\|^2)^{-1}}{\sum_{k\ne l}(1+\|y_k-y_l\|^2)^{-1}},\quad
C=\sum_{i\ne j}p_{ij}\log(p_{ij}/q_{ij}).
\]

Set all diagonal affinities to zero; normalize over ordered non-self pairs. \(\mathrm{Perp}(P_i)=\exp H(P_i)\) using natural logs, equivalently \(2^{H_2(P_i)}\). **Own exact check:** uniform probabilities over four neighbors give perplexity 4; concentrated mass tends toward 1. In the P=(.9,.1) example, derivative w.r.t. unconstrained log q is −P; state the normalization coupling before interpreting forces. Compare a Student-t kernel with a Gaussian at distances 1 and 3 (unnormalized t weights .5,.1); forecast how the tail changes room, then verify normalized affinities. t-SNE is a serious comparator for neighborhood visualization, not a strawman that “always loses global structure.”

**UMAP, source facts and idealized teaching objective.** Construct a local weighted neighbor graph, combine directed memberships by fuzzy union, then lay out that graph. Local-scale adaptation plus sparse graph optimization helps explain practical usefulness; better speed/global preservation remains a dataset/implementation comparison. [McInnes, Healy & Melville, §3.1–3.2, §4.3, §6 and Appendix C](https://arxiv.org/html/1802.03426).

\[
w_{i\to j}=\exp[-\max(0,d(x_i,x_j)-\rho_i)/\sigma_i],\quad
\mu_{ij}=w_{i\to j}+w_{j\to i}-w_{i\to j}w_{j\to i},\quad
\nu_{ij}=(1+a\|y_i-y_j\|^{2b})^{-1},\quad
L=-\sum_{i<j}[\mu_{ij}\log\nu_{ij}+(1-\mu_{ij})\log(1-\nu_{ij})].
\]

Here the first expression describes memberships on retained neighbor edges; absent edges have zero directed membership. Edge strengths are **not a single probability distribution summing to one**. The displayed L is the full edgewise binary cross-entropy, up to fixed target-entropy terms and a consistent ordered/unordered-pair convention. It is not the categorical KL of t-SNE. **Own exact check:** directed .8 and .5 combine to .9, not 1.3. For μ=.9, compare ν=.9 and .1 and attribute attraction/repulsion contributions.

Do not claim ordinary negative-sampled software exactly minimizes that displayed full loss. [Damrich & Hamprecht 2021, abstract and §4](https://proceedings.neurips.cc/paper/2021/file/2de5d16682c3c35007e4e92982f1a2ba-Paper.pdf) derive a different effective objective for the implementation they analyze. This is a short advanced caveat about theory vs implementation, not a new methods chapter or an asserted analysis of every future release.

UMAP can create false tears and does not preserve density reliably: a tighter island need not mean a denser original population. [Official clustering documentation, opening caution](https://umap-learn.readthedocs.io/en/latest/clustering.html). `n_neighbors` changes the neighborhood scale; `min_dist` changes packing rather than supplying a calibrated physical length. [Official parameter guide, corresponding sections](https://umap-learn.readthedocs.io/en/latest/parameters.html). Compare seeds and settings; unaligned coordinates may rotate/reflect. [Official reproducibility guide](https://umap-learn.readthedocs.io/en/latest/reproducibility.html). These are observed algorithmic sensitivities to demonstrate, not guarantees of a specific plot outcome.

**Play cell proposal:** one continuous curved cloud and one heteroskedastic cloud, not only separated blobs. Student toggles units/metric, neighbor scale, min_dist, perplexity, initialization and seed while original-space neighbor IDs remain linked. Include a deliberately torn rendering as a known negative example. Forecast whether islands are graph disconnection, real structure, or layout before inspecting graph components and original distances. Score all declared attempts, not the prettiest seed. Never interpret map area, orientation, gap size, or density as calibrated evidence by itself.

### Where the other visualization techniques belong

This is a decision aid inside F2–F4, not new chapters or a requirement to implement every method. “Preserved quantity” means the method's target or representation, not a guarantee of perfect preservation in two dimensions. Controls below are proposed discriminating exercises; no method wins by plot appearance.

| View | Target question / quantity | Failure to recognize / proposed control |
|---|---|---|
| Raw scatter and pair plots | Which measured coordinates vary together, in their actual units? | Pairwise views can miss multivariate relationships; compare the same IDs across coordinates and a shuffled-coordinate view. |
| Distance / Gram heatmap | Which original-space pairs are close, or have large inner products under the declared normalization/kernel? | Row order can manufacture blocks; permute display order while retaining IDs, compare raw/normalized vectors and pairwise-distance vs similarity definitions. Gram entries are inner products, not automatically distances. |
| PCA | Which linear subspace retains squared variation/reconstruction? | High-variance nuisance can dominate; use the rectangle's low-variance label and a no-reduction baseline. |
| Random projection | Does a learned projection beat a simple geometry baseline? | Distance guarantees require sufficient dimension and stated distortion/probability conditions; two plotting axes need not satisfy them. Compare distance distortion and neighborhood retention across declared seeds/dimensions. [Official guide, §8.6.1](https://scikit-learn.org/stable/modules/random_projection.html). |
| Metric MDS | Can display distances approximate the supplied dissimilarities? Raw stress is \(\sum_{i<j}(\delta_{ij}-\|z_i-z_j\|)^2\). | Stress can hide which distances distort and depends on scale; show a Shepard distance plot/residuals and an exact small Euclidean configuration. Classical MDS uses a different strain objective; do not conflate it with stress minimization. [Official MDS guide, §2.2.8](https://scikit-learn.org/stable/modules/manifold.html#multidimensional-scaling). |
| Isomap | Can graph shortest-path distances approximate manifold geodesics before embedding? | A shortcut edge can collapse the intended long route; disconnects also matter. Add/remove one bridge on a curved strip and compare graph paths with ambient Euclidean distances. [Official Isomap guide and practical cautions](https://scikit-learn.org/stable/modules/manifold.html#isomap). |
| Laplacian / spectral embedding | Which coordinates vary smoothly over a chosen weighted neighborhood graph? | Graph construction/components and near-degenerate eigenvalues can destabilize coordinates; inspect components, eigengaps and neighbor perturbations, comparing subspaces rather than eigenvector signs. An eigengap is a diagnostic, not automatic proof of intrinsic dimension. [Official spectral embedding reference](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.SpectralEmbedding.html). |
| Diffusion maps | Which points have similar t-step transition distributions for a declared Markov operator? Coordinates weight eigenvectors by eigenvalues raised to t; truncation approximates the associated diffusion geometry. | Kernel normalization, bandwidth and diffusion time change the geometry; compare t and graph construction, including a bottleneck control. Diffusion distance is not ordinary Euclidean distance, UMAP's edge loss, or automatically the PRH comparison kernel. [Coifman & Lafon 2006, §2.2–2.5](https://www.math.ucdavis.edu/~strohmer/courses/270/diffusion_maps.pdf). |
| t-SNE / UMAP | Which neighborhood-affinity relations can this map communicate? | Attractive islands can mislead about density or global distance; use F3's original-neighbor views, nulls and parameter/seed controls. See the original papers and official cautions above. |

The graph, Markov operator, pairwise distances and Gram matrix are separate objects. A learner moving toward PRH/kernel alignment must write down which one is being compared and why, before selecting a display. Geodesic/spectral/diffusion implementations stay advanced unless an existing lesson needs that target question.

## F4: audit a visualization and match metrics to the claim

| Term | What it answers | Example / boundary |
|---|---|---|
| Objective / loss | What quantity is optimized over which examples and parameters? | Training CE or paired InfoNCE; its improvement does not by itself certify the desired downstream behavior. |
| Evaluation metric | What declared behavior or geometry is summarized on which cohort? | Heldout recall, original-space neighbor retention or retrieval rank; can coincide algebraically with a loss while serving a different role. |
| Uncertainty measure | How variable or uncertain is that estimate under a stated sampling model/unit? | A grouped-sample interval or a null distribution; across-seed spread measures algorithmic variability, not independent-subject uncertainty. |

Let original ranks r_X(i,j) and display ranks r_Y(i,j) exclude self, start at 1 and use a declared tie policy. Define U_i as display k-neighbors absent from original k-neighbors; V_i is the reverse. For 1≤k<n/2:

\[
A=\frac{2}{nk(2n-3k-1)},\quad
T(k)=1-A\sum_i\sum_{j\in U_i}(r_X(i,j)-k),\quad
C(k)=1-A\sum_i\sum_{j\in V_i}(r_Y(i,j)-k).
\]

Trustworthiness penalizes false display neighbors; continuity penalizes missing original neighbors. Report both and simple neighbor overlap \(\frac1{nk}\sum_i|N_X^k(i)\cap N_Y^k(i)|\); with equal k, unweighted neighbor precision and recall coincide, while the rank penalties differ. [Official trustworthiness formula](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.trustworthiness.html); [Venna et al. 2010, §2 and evaluation measures](https://jmlr.org/papers/volume11/venna10a/venna10a.pdf). These diagnostics depend on the chosen input metric and k; they do not establish semantic groups or causality.

Proposed visible result: aligned panels for original→PCA intermediate and original→final T/C/overlap vs k, original-space labeled kNN under grouped heldout splits, independently shuffled-label null, independent-feature-noise geometry, and seed/settings stability based on neighbor sets. Random labels leave unsupervised geometry metrics unchanged but should remove label-predictive evidence in expectation; a finite null draw can look good by chance. A row-permuted map with unreconciled IDs is an error, not a legitimate negative control. Approximate/subsample diagnostics must disclose sample IDs and coverage.

Training objective, report metric and uncertainty measure answer different questions. Define units/denominator/aggregation/direction for each. Accuracy=correct/n; precision=TP/(TP+FP); recall=TP/(TP+FN), with an explicit zero-denominator policy. Binary log loss is \(-n^{-1}\sum[y\log q+(1-y)\log(1-q)]\); binary Brier is \(n^{-1}\sum(q-y)^2\). These score probabilities, not merely thresholded decisions; Brier alone does not isolate calibration. [Official metric definitions, §3.4.4 and Brier discussion](https://scikit-learn.org/stable/modules/model_evaluation.html). A reliability diagram compares confidence bins with empirical correctness and counts; finite binning and cohort choice matter. [Guo et al. 2017, §2, equations 1–3](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf).

**Own exact counterexamples:** predicting the majority class on 95 negative/5 positive items gives 95% accuracy and zero positive recall. On a cohort with 80 correct/20 incorrect hard predictions, assigning confidence .99 to every chosen class gives the same 80% accuracy as .8; mean binary log losses are .929074 vs .500402, Brier .1961 vs .16. This constructed cohort is an arithmetic illustration, not a general calibration finding. Plot accuracy, loss and reliability side by side; a threshold slider changes the confusion matrix while probabilities stay fixed.

Choose additional metrics only when the question needs them: rank correlation (e.g. Spearman, the correlation of ranks) asks about monotonic ordering, not agreement in distance scale [official definition](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html); ROC AUC summarizes binary ranking across false/true-positive trade-offs, not calibration or a chosen operating threshold [official ROC AUC reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html); a precision–recall curve shows threshold-dependent positive predictive value vs coverage and should be shown with prevalence and counts when positives are rare [official precision–recall reference](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_curve.html). Proposed control: change class prevalence or apply a monotonic score transform and predict which rankings, probabilities and operating-point metrics change. Repeated pairs from the same subjects require an appropriate grouped uncertainty analysis regardless of metric.

## F5: Bonferroni, family error, and the selection trap

For a declared finite family of m hypotheses and valid individual null p-values, reject H_i when \(p_i\le\alpha/m\); adjusted \(p_i^{adj}=\min(1,mp_i)\). Valid means \(P_{H_i}(p_i\le u)\le u\). Exact union-bound argument:

\[
\mathrm{FWER}=P(V\ge1)\le\sum_{i\in I_0}P(p_i\le\alpha/m)\le |I_0|\alpha/m\le\alpha.
\]

**Independence is not required.** Dependence can make the bound conservative. [NIST Bonferroni method, general inequality and interpretation](https://www.itl.nist.gov/div898/handbook/prc/section4/prc473.htm). **Own exact example:** α=.05, m=4, p=(.004,.02,.03,.2): threshold .0125, only first rejected; adjusted values (.016,.08,.12,.8). With independent all-null tests at .05 each, uncorrected FWER=1−.95^m; with four identical valid p-values, uncorrected FWER=.05. The distinction makes dependence tangible.

FWER is P(at least one false rejection). FDR is \(E[V/\max(R,1)]\), where V counts false and R all rejections; they coincide under the complete null but differ otherwise. Original BH controls FDR under its stated independence conditions; do not substitute “5% FDR” for “5% chance of any false claim.” [Benjamini & Hochberg 1995, pp. 289–291, §2–3](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL06/Benjamini%20and%20Y%20FDR.pdf). Teaching BH's full procedure/dependence extensions is advanced; the base gate requires choosing the intended error rate correctly.

Proposed family manifest lists layers, seeds, metrics, prompts, settings, exclusions and outcome-driven retries before inspection. Bonferroni over a fixed, valid family does **not** repair invalid p-values, optional stopping, post-selection tests, or omitted adaptive searches. Own derivation: select the minimum of m independent Uniform(0,1) null p-values but report it as one preselected test; P(min p≤.05)=1−.95^m, not .05. Registering only the winner loses the family. All attempted analysis choices remain visible; a redesigned confirmatory study requires fresh untouched evaluation or an explicitly justified selection-aware method. Do not convert exploratory seed/parameter tuning into protected evaluation.

The basic statistics bridge must also cover: sample vs population, effect size vs p-value, uncertainty vs practical importance, paired contrasts, independence/grouping, interval interpretation, and permutation exchangeability. A bootstrap or permutation count is not an independent sample size. Thousands of pairwise distances among n subjects are not thousands of independent subjects. Power/sample-size design, simultaneous intervals, adaptive inference and dependency-specific FDR procedures belong to advanced research preparation when a concrete study needs them; this proposal does not invent sample-size thresholds or launch simulations.

## Math confidence gate and downstream transfer

The gate is a resumable diagnostic, **not a performance-mastery badge**. Retain each item, forecast, confidence, explanation, help exposure and remediation status; never prefill learner answers. Ask the learner to (1) compute one asymmetric KL and support edge case; (2) reconstruct the PCA rectangle and identify the lost label; (3) distinguish t-SNE normalized affinities from UMAP memberships; (4) identify a false-neighbor vs missing-neighbor error; (5) apply Bonferroni and name its valid-p/family conditions; (6) explain why a good plot/probe/loss need not show causal use. An incorrect item links to its shortest original 0.0 or companion repair and returns to a newly varied item. No global numerical pass percentage is adopted here; reviewer/user acceptance must establish what explanations suffice.

Changed-condition transfer is mandatory: change units or nuisance variance after PCA; change metric/density/connectivity after the map; change target scale after MSE/cosine; change class balance after accuracy; change family/adaptation after p-values. The learner forecasts before viewing outcomes and explains which prior rule transfers or fails.

Downstream: PRH requires declaring representation identity, preprocessing, kernel/geometry, independent unit and chance baseline; JEPA requires identifying prediction/target tensors, masking, reduction/gradient paths, collapse alternatives and heldout task evidence; probes/lenses distinguish accessibility/decodability from causal use and account for label/token alignment; kernel alignment compares declared centered Gram matrices with invariance/null/selection checks rather than substituting attractive 2D views. Advanced derivations and real-model work stay in their existing authorized routes. This foundations increment supplies prerequisites only.

## Integration boundary and unresolved decisions

Root should carry these recommendations into the Map/Spec: one additive foundations route with F0–F5 edges; explicit worked examples and repair links; formulas plus conditions; visible controls; heldout transfer and exposure tracking; no method selected by plot beauty; preserve existing source/learner acceptance boundaries. Do not mark these lessons implemented or accepted from this note.

| ID | Status / dependency | Ready decision and recommendation |
|---|---|---|
| FOUND-01 | Proposed; root owns user interview | Place this as an additive companion/remediation spine linked from 0.0 and 11.1, rather than a new broad chapter. |
| FOUND-02 | Proposed; depends on route placement | Use F0–F5 as minimum support; keep category theory, extended FDR/adaptive inference, and additional manifold algorithms as optional advanced connections. |
| FOUND-03 | Unresolved; depends on accepted gate rubric | Define sufficient learner explanations by human walkthrough of the six diagnostic items; do not infer competence from numeric code tests or confidence. |

No user decision is fabricated here. Root consolidates the queue with the existing project IDs. No Zotero contents, data/model files, weights, inference, training, GPU/cluster jobs, installs, or GitHub mutations were used. Original ARENA files were read only. Validation of this planning artifact checks formulas and links; proposed learner exercises have not been run or accepted.

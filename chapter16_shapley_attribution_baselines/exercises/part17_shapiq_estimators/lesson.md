# [16.17] One Query, Many Interactions

# Introduction

Today you'll build two algorithms that estimate feature interactions without asking a model about every possible feature subset. Your first algorithm will turn **one model response into an update for every pair**. Your second will fit a sequence of weighted regressions to the same responses. You'll be able to see what each query buys you, where sampling noise enters, and when the regression has too little information to identify an answer.

The central question is simple: **how accurately can we estimate every pair interaction when we can query only some coalitions?**

We will use exact small games to understand the algorithms, then repeat the comparison on eight-feature Concrete-strength predictors. Eight features are small enough to enumerate all 256 coalitions for a reference. The sampled estimators will receive only their requested values. Keeping the exact answer outside the estimator lets us measure an error that would normally be unknown.

This lesson continues 16.3's Shapley Interaction Index (SII). Both algorithms target the **same SII**. Shapley-Taylor and Faith-Shap from 16.16 choose different allocations; disagreement between those definitions is a different issue from the estimation error studied here.

## Content & Learning Objectives

### 1️⃣ Share one query across many interactions

You'll rewrite the four-term pair derivative as a weighted sum over a common set of coalitions. This is the step that lets SHAP-IQ reuse each model evaluation for all pairs.

> ##### Learning Objectives
>
> - Distinguish a coalition, a context and an interaction set.
> - Derive the sign and magnitude of a shared-query update.
> - See why one apparently large update can cancel in the complete answer.

### 2️⃣ Sample a game, measure the uncertainty

You'll implement a size-then-subset sampler and the SHAP-IQ mean and standard error. The same random draw can occur more than once: it costs one cached model query but still contributes multiple times to the Monte Carlo average.

> ##### Learning Objectives
>
> - Separate a size probability from a per-coalition probability.
> - Implement importance weighting and an iid standard error.
> - Count draw requests and unique queries correctly.

### 3️⃣ Fit the pairwise KernelSHAP-IQ limit

You'll solve a constrained least-squares problem, then use it to fit main effects and pair residuals. The interesting part is the boundary: infinite weight does not mean every boundary residual can be made zero.

> ##### Learning Objectives
>
> - Separate the feasible particular solution from free directions.
> - Derive the pair design and boundary least-squares constraint.
> - Detect rank deficiency rather than silently regularizing it away.

### 4️⃣ Read the real-model comparison

You'll apply both estimators to every held-out recipe for seven known predictors, compare against exact SII, and inspect two editable experiments.

> ##### Learning Objectives
>
> - Read signed errors alongside query-budget curves and model controls.
> - Compare estimated SE with the exact sampling variance of a finite game.
> - Explain the limits of a precomputed-oracle query benchmark.

The eight exercises should take about 90–120 minutes, followed by 30–45 minutes for the model comparison and plays. If the nullspace derivation is unfamiliar, work through its three-coefficient example and use the solution to continue; return to the proof once the end-to-end result is clear.

## Setup

All methods use CPU float64. An integer mask contains feature `i` exactly when bit `i` is set. Pair columns follow lexicographic feature order: `(0,1), (0,2), ...`. We supply input validation, indexing helpers and plotting. The sampling and estimation algorithms will be your own functions.

<!-- code: setup -->

## Cold open: one coalition says something about six pairs

Return to the controlled score from 16.3. Two weak individual features have a strong joint effect, and a triple contributes additional pairwise credit:

$$v(x)=0.2x_0+0.2x_1+5x_2+4x_3+3x_0x_1+2x_0x_1x_2.$$

Suppose you query only coalition `{0,1}`. Its value is `3.4`. Can that single response tell you the interaction of the entirely absent pair `{2,3}`?

<details>
<summary>Predict before continuing</summary>

It cannot identify that pair by itself. It can contribute one signed term to an estimator of that pair. Other queries must cancel contributions due to unrelated features. The complete SII of `{2,3}` is zero, even though this query contributes positively to its shared-query sum.

</details>

<!-- code: cold_open -->

# 1️⃣ Share one query across many interactions

## Turn masks into model inputs

Our algorithms use a matrix $Z$ with one queried coalition per row and one feature per column. Its entries are zero or one. For four features, mask `6` is binary `0110`, so its row is `[0,1,1,0]`. The printed binary string puts the most significant bit first; the feature columns put feature zero first. Keep these conventions distinct.

### Exercise - implement `coalition_design`

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵⚪⚪
>
> You should spend up to 5 minutes on this exercise.
> ```

Use a vector of feature bits and a broadcast bitwise AND. Preserve repeated input masks and their order.

<!-- exercise: coalition_design -->

<details>
<summary>Expected output</summary>

`[0,1,6,15]` maps to the empty row, feature zero, features one and two, and the full row. The focused test prints `All tests in test_coalition_design passed!`.

</details>

<details>
<summary>Help - which axis should be broadcast?</summary>

Compare `masks[:, None]` with `1 << torch.arange(n_players)`. Convert the nonzero result to float64 after the bitwise operation.

</details>

<!-- solution: coalition_design -->

## Rearrange the pair derivative

For a fixed pair $S=\{i,j\}$ and a disjoint context $C$, the second difference is

$$\Delta_S v(C)=v(C\cup S)-v(C\cup\{i\})-v(C\cup\{j\})+v(C).$$

SII averages this derivative with weight

$$m_s(c)=\frac{1}{(n-s+1)\binom{n-s}{c}}.$$

For pairs, $s=2$. Equivalently, the size of the context is uniform over `0..n-2`, then the context is uniform within that size. The direct definition asks about different contexts for each pair. To share queries, expand every derivative and collect the coefficient of the **same queried coalition** $T$.

If $k=\lvert T\cap S\rvert$, the context behind this occurrence must be $C=T\setminus S$. The derivative contributes sign $(-1)^{s-k}$ and context size $\lvert T\rvert-k$. Therefore

$$I^{\mathrm{SII}}(S)=\sum_{T\subseteq N}v_0(T)\,\gamma_s(T,S),\qquad
\gamma_s(T,S)=\frac{(-1)^{s-\lvert T\cap S\rvert}}{(n-s+1)\binom{n-s}{\lvert T\rvert-\lvert T\cap S\rvert}},$$

where $v_0(T)=v(T)-v(\varnothing)$. This is SHAP-IQ's shared-evaluation representation. The model query is independent of the pair; only its weight changes.

### Exercise - implement `sii_query_weights`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15 minutes on this exercise.
> ```

Return one gamma row for each query and one column for each interaction of the requested order. The supplied `interaction_masks` enumerates the interaction columns, not the power set of model queries. Compute overlap sizes using the two binary designs; use the formula above for sign and denominator.

<!-- exercise: sii_query_weights -->

<details>
<summary>Expected output</summary>

For `n_players=4`, query `{0,1}` and pair order, the gamma row is

`[1/3, -1/6, -1/6, -1/6, -1/6, 1/3]`.

For example, pair `{0,2}` overlaps once, so its sign is negative and its context contains one of the other two players. The complete additive game returns zero in every pair column.

</details>

<details>
<summary>Help - avoid one Python loop per query and pair</summary>

If `Z` contains query designs and `A` contains interaction designs, `Z @ A.T` gives every overlap size. Subtract from each query's population count. Precompute the short denominator vector for context sizes `0..n-order` and index it with that result.

</details>

<!-- solution: sii_query_weights -->

<!-- code: shared_query_figure -->

<details>
<summary>Interpretation - why does the absent pair receive a positive update?</summary>

The update is a term in an algebraic sum, not a claim that the absent pair caused the response. The plot shows the complete additive control canceling after all coalitions are included. At finite sampling budgets the cancellation is imperfect. A large contribution from one query should never be read as a recovered interaction by itself.

</details>

## Try It Yourself: follow one query

Change the coalition mask below. Predict which pair columns become negative before running it. Then remove the interaction terms while retaining the additive weights.

<!-- code: query_play -->

# 2️⃣ Sample a game, measure the uncertainty

## Query the boundary exactly

We will query the empty and full sets, every singleton and every singleton complement. For eight features, these are 18 unique coalitions. Sample the remaining 238 coalitions with replacement.

This is a fixed teaching design, not an outcome-driven choice of strata. The exact boundary is useful for both estimators and lets them share the same requested values. A query cache will eliminate duplicate model calls while preserving duplicate samples in the averages.

## A size is not a coalition

Let $p_k$ be the probability of drawing size $k$. After choosing a uniformly random set of that size, its probability is

$$p(T)=\frac{p_{\lvert T\rvert}}{\binom{n}{\lvert T\rvert}}.$$

The kernel proposal uses a per-coalition weight proportional to $1/\binom{n-2}{k-1}$. Multiplying by the number of coalitions of that size gives size mass proportional to $1/[k(n-k)]$. Uniform sampling over *coalitions* instead gives size mass proportional to $\binom{n}{k}$. Both distributions are restricted to sizes `2..n-2` here.

### Exercise - implement `sample_coalitions`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 12 minutes on this exercise.
> ```

Create a local seeded generator. Draw a size from the appropriate distribution, generate a uniform random feature permutation, and retain its first `size` elements. Return the mask and its **per-coalition probability** for every draw. Do not enumerate all `2**n_players` masks.

<!-- exercise: sample_coalitions -->

<details>
<summary>Expected output</summary>

Repeated calls with the same seed return the same masks and probabilities; the global Torch RNG is unchanged. For eight features, each uniform interior coalition has probability `1/238`. Kernel sampling gives equal probabilities to sets of the same size, with larger per-set mass near the support boundaries.

</details>

<details>
<summary>Help - make a random subset without a power-set table</summary>

Draw a `[draws, n_players]` matrix of independent continuous random scores and argsort each row. Its first `k` feature indices form a uniform size-`k` subset. Sum their bit values to make the integer mask. Divide the selected size probability by `math.comb(n_players, k)`.

</details>

<!-- solution: sample_coalitions -->

<!-- code: sampling_figure -->

## SHAP-IQ is an importance-weighted mean

Let $B$ be the exact boundary. Its contribution is

$$c_B(S)=\sum_{T\in B}v_0(T)\gamma_s(T,S).$$

For an interior draw $T_r\sim p$, define $Y_r(S)=v_0(T_r)\gamma_s(T_r,S)/p(T_r)$. Then

$$\widehat I(S)=c_B(S)+\frac1K\sum_{r=1}^{K}Y_r(S),\qquad
\widehat{\mathrm{SE}}(S)=\sqrt{\frac{1}{K(K-1)}\sum_{r=1}^{K}\big(Y_r(S)-\overline Y(S)\big)^2}.$$

The mean is unbiased for the fixed finite game under iid sampling. The SE estimates sampling variability conditional on that game. It says nothing about uncertainty in the training data, fitted model or choice of donor background. A normal error band can still have poor coverage at a small budget.

### Exercise - implement `shapiq_estimate`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 12 minutes on this exercise.
> ```

The first boundary value is the empty coalition. Center both boundary and sampled values by it. Compute the exact boundary sum and the mean of the weighted sampled contributions. Return the estimate and the sample standard error. Support arbitrary leading game axes.

<!-- exercise: shapiq_estimate -->

<details>
<summary>Expected output</summary>

Summing a complete uniformly enumerated interior recovers exact SII. A repeated sample retains its multiplicity. Adding a constant to every game value changes neither the estimate nor its SE. An additive game's finite-sample pair estimate can be nonzero, while a constant game is exactly zero after centering.

</details>

<details>
<summary>Help - which variance correction do I need?</summary>

Use sample variance with denominator `K-1`, then divide by `K` before taking the square root. The exact boundary contributes no sampling variance. Do not deduplicate the array of contributions and do not divide by the number of unique queried coalitions.

</details>

<!-- solution: shapiq_estimate -->

## Count what you actually asked the model

### Exercise - implement `query_sampled_game`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Concatenate boundary and sampled masks, query each unique mask once, then use an inverse index to reconstruct the original requested order. Return the boundary values, repeated sampled values, observed masks and counts. The callable may return several games at once, with the queried mask on its last axis.

<!-- exercise: query_sampled_game -->

<details>
<summary>Expected output</summary>

For four players, sampled masks `[3,3,5,12]` make four draw requests plus ten exact-boundary requests, but only thirteen unique queries. The two occurrences of `3` reappear twice in `sampled_values`.

</details>

<details>
<summary>Help - one call can serve repeated requests</summary>

`torch.unique(requested, sorted=True, return_inverse=True)` gives both the query set and the restoration index. Evaluate the callable on the unique set, then index its final axis with the inverse. The model sees only this subset.

</details>

<!-- solution: query_sampled_game -->

# 3️⃣ Fit the pairwise KernelSHAP-IQ limit

## A constrained least-squares building block

We want to minimize $\lVert W^{1/2}(X\beta-y)\rVert^2$ subject to $C\beta=d$. Find one feasible vector $\beta_0=C^+d$, and let the columns of $Z$ form an orthonormal basis for the nullspace of $C$. Every feasible answer is

$$\beta=\beta_0+Zu.$$

Now solve an ordinary weighted least-squares problem in $u$:

$$\min_u\left\lVert W^{1/2}XZ u-W^{1/2}(y-X\beta_0)\right\rVert^2.$$

For a small example, fit `[1,2,3]` with an identity design subject to the three coefficients summing to zero. The closest feasible answer is `[-1,0,1]`: subtract the mean. The nullspace implementation performs the corresponding projection for a general design.

### Exercise - implement `constrained_wls`

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 20 minutes on this exercise.
> ```

Use SVD to identify the constraint rank and nullspace. Check that the particular solution actually satisfies the targets. Solve in the free coordinates with an SVD pseudoinverse and report the free rank, singular values, equality residuals and whether all free directions are identified.

Compare small singular values with `rtol` times the spectral norm of the **original weighted design**. A theoretically zero `X @ Z` can contain roundoff; normalizing only by that nearly zero matrix would mistake roundoff for usable information.

<!-- exercise: constrained_wls -->

<details>
<summary>Expected output</summary>

The identity example returns `[-1,0,1]`. Contradictory equalities raise an error. A design that only observes the already constrained sum has zero identified free directions, returns a minimum-norm free solution and preserves the equality. It does not add a ridge penalty.

</details>

<details>
<summary>Help - two different SVDs do two different jobs</summary>

The constraint SVD gives `Z` from the unused right-singular vectors. The weighted free-design SVD solves for `u`. Keep only singular directions above the declared threshold. If none survive, the minimum-norm free coordinate is zero. `beta0` remains a valid feasible solution.

</details>

<!-- solution: constrained_wls -->

## First fit main effects, then fit the residual

The first stage is KernelSHAP: fit the binary feature design using per-coalition weights

$$\mu_1(t)=\binom{n-2}{t-1}^{-1},\qquad 1\leq t\leq n-1,$$

subject to $\sum_i\phi_i=v_0(N)$. Boundary values of sizes one and `n-1` are included exactly; sampled interior rows receive the Monte Carlo adjustment `1/(K*p(T))`. The endpoint equality is exact to the precision of the numerical solve.

For pairs, the appropriate residual design is **not** just the indicator that both features are present. It is

$$X_2(T,\{i,j\})=z_i z_j-\tfrac12(z_i+z_j).$$

This is `-1/2` when exactly one pair member is present and zero otherwise. It follows from building the two-additive reconstruction from SII: adding pair credit also removes half that credit from each incident singleton. The finite interior weights become

$$\mu_2(t)=\binom{n-4}{t-2}^{-1},\qquad 2\leq t\leq n-2.$$

### Exercise - implement `pair_residual_design` and `kernel_weights`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 10 minutes on this exercise.
> ```

Implement the pair formula and the two finite kernels. Return zero weight outside the finite range as a marker; we handle those boundary rows separately. Do not treat this zero marker as a claim that boundary values do not matter.

<!-- exercise: pair_residual_design,kernel_weights -->

<details>
<summary>Expected output</summary>

The pair design is unchanged by complementing the coalition. Empty and full rows are zero. For six players and order two, the finite weights at sizes `2,3,4` are `[1,1/2,1]` **per coalition**.

</details>

<details>
<summary>Help - distinguish design from weighting</summary>

The design tells you what a pair coefficient predicts for one coalition. The kernel tells you how much its squared residual matters. Compute the two independently; multiply by the sampling adjustment only when assembling the regression.

</details>

<!-- solution: pair_residual_design,kernel_weights -->

## The boundary cannot usually be fit exactly

The paper gives singleton and complement residuals equal weights that tend to infinity. For feature $i$, both rows of the pair design are $-C_i/2$, where $C_i$ is one for each incident pair. Let their residuals after the main-effect fit be $r_i$ and $r_{-i}$. Their boundary objective is

$$(r_i+\tfrac12C_i\beta)^2+(r_{-i}+\tfrac12C_i\beta)^2.$$

Its minimum occurs at $C_i\beta=-(r_i+r_{-i})$. Because the main effects sum to $v_0(N)$, this simplifies to

$$C_i\beta=v_0(N)-v_0(\{i\})-v_0(N\setminus\{i\}).$$

This is a constraint on the **boundary least-squares minimizer**. It does not set either original residual to zero. All feature/pair incidence rows have full row rank for `n>=3`, so these boundary-minimizing targets are jointly feasible. The interior objective then chooses among them. This implements the pairwise infinite-weight limit without a huge numerical penalty.

<details>
<summary>Worked example - the pure four-way game</summary>

Only the full coalition has value one. Each main Shapley value is `1/4`. The singleton residual is `-1/4`; its complement residual is `-3/4`. Their best common prediction is `-1/2`, leaving residuals `+1/4` and `-1/4`. The incidence constraint says each feature's three pair values sum to one. Symmetry gives each pair `1/3`, the correct SII. Forcing both boundary residuals to zero would demand incompatible predictions.

</details>

### Exercise - implement `kernelshap_iq`

> ```yaml
> Difficulty: 🔴🔴🔴🔴⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 20 minutes on this exercise.
> ```

Center at the empty value. Fit main effects with the first kernel and endpoint sum. Form sampled interior residuals, then fit the pair design with the second kernel and the incidence targets derived above. Use the same sampled values and probabilities in both stages. Return both coefficient families and their rank diagnostics.

<!-- exercise: kernelshap_iq -->

<details>
<summary>Expected output</summary>

With complete uniform interior enumeration, the result matches ordinary Shapley and pair SII on every tested game. The pure four-way example returns `1/4` for each main effect and `1/3` for each pair. The boundary residual remains nonzero while its minimizing incidence equality holds.

</details>

<details>
<summary>Help - assemble the two targets separately</summary>

The first target is the single full-minus-empty value. The second target has one component per feature: full-minus-empty, minus its singleton-minus-empty and complement-minus-empty. Find those boundary positions by mask, not by assuming feature order matches sorted mask order.

</details>

<!-- solution: kernelshap_iq -->

<!-- code: boundary_figure -->

<details>
<summary>Interpretation - what do the finite-penalty curves establish?</summary>

They check the connection with the paper's limit: increasing the boundary weight approaches the same SII answer. They do not promise that larger numerical penalties are always safer. The direct limit avoids an ill-conditioned penalty scale; the rank diagnostic still matters when there are too few sampled interior directions.

</details>

# 4️⃣ Read the real-model comparison

## What is one query in this experiment?

For one held-out Concrete recipe, the game keeps the requested features and takes every absent feature from one intact training recipe. It averages predictions over 64 fixed donors. One coalition query therefore costs 64 underlying predictions; exact enumeration costs 256 coalition queries or 16,384 donor predictions per model/row.

We refit the same seven known predictors from 16.14/16.15 and require all 21 model-state identities to match. The 758 training rows, 153 explanation rows and 119 audit rows are split by recipe groups. Clean seeds, matched shuffled-label seeds and ridge all remain in the comparison.

For this repeated sampling benchmark, we construct the complete model-game oracle once. Each estimator only receives its queried subset through your `query_sampled_game`. The complete table and exact SII are available to the evaluator afterward. The figures compare **logical query budgets**, not measured speedups over direct model calls. The up-front oracle cost is shown, and it can dominate at eight features.

<details>
<summary>Why keep the exact table outside the estimator?</summary>

The reference must be able to measure all 28 errors even when a method queried only 34 distinct masks. Supplying the complete table directly to the estimator would make that budget meaningless. The query-wrapper tests alter all unqueried values and require both estimates to remain unchanged.

</details>

<!-- code: methods_namespace -->

<!-- code: native_study -->

## Read model quality before explanation size

<!-- code: model_controls -->

The shuffled-label predictors are controls for learning useful predictive structure. They can still have nonzero interactions. Ridge is an additive model and its exact pair SII should be near rounding error. Finite SHAP-IQ may estimate nonzero ridge interactions because its sampled cancellation is imperfect; that is part of the comparison.

## How much error remains at each budget?

<!-- code: convergence -->

Every line uses all rows of its indicated held-out split, weighting recipe groups equally before averaging pair errors. Replicate variation comes from the 32 sampling streams. Those streams are shared across rows, so rows are not counted as independent Monte Carlo replicates. All model seeds and both sampling proposals remain visible. The zero-pair baseline shows the error of doing no interaction estimation at all.

## Signed answers at the fixed recipe

<!-- code: signed_comparison -->

This view fixes the model, source row, replicate and budget before looking at results. Read the signed estimate alongside its error. An accurate mean across many pairs can conceal a wrong sign on one pair, while a single noisy replicate can look much worse than the aggregate curve.

## Does the reported sampling uncertainty mean what we think?

<!-- code: variance_comparison -->

Because this game is finite, we can enumerate the exact one-draw variance under each proposal and divide by `K`. Compare that prediction with average estimated SE squared and observed replicate squared errors. Their agreement is a sampling diagnostic. It does not give a confidence interval for concrete chemistry or for future model fits.

## Draws, unique queries and identified directions

<!-- code: query_costs -->

Drawing a coalition twice increases the Monte Carlo sample count while the cache pays for its model evaluation once. Keep both counts. After enough draws, the logical unique-query count approaches 256; complete enumeration is then a serious alternative. The pair fit has twenty free directions after its eight incidence constraints. A low-budget estimate can be rank deficient even if the solver returns a finite coefficient vector.

## Try It Yourself: change the query budget

Change the recipe, model, proposal, seed or number of draws below. Predict which quantity should change: the target SII, the estimate, the reported uncertainty, or the cost. Changing only the seed leaves the target fixed. Changing the recipe or model changes the game. The result is exploratory and does not overwrite the fixed study.

<!-- code: native_play -->

<details>
<summary>Ideas to try</summary>

1. Keep `clean_0` and source row `0`, but switch proposals at the same draw budget. Inspect both the error and unique-query count.
2. Set the model to `ridge`. Compare its exact near-zero pairs with the finite SHAP-IQ error and the KernelSHAP-IQ rank diagnostic.
3. Reduce the budget until the pair fit loses free rank. A finite answer is not the same as an identified regression.
4. Change only the seed and find a case where one pair's sign changes. Compare the two estimates with exact SII before telling a feature-interaction story.

</details>

# Conclusion

One model response can update every pair once we rearrange the SII sum. SHAP-IQ estimates that shared sum directly; KernelSHAP-IQ fits main effects and pair residuals to the same queries. They can differ at a finite budget while targeting the same exact interaction index.

You have also seen three distinct limits: finite sampling leaves noise, incomplete designs leave unidentified directions, and an exact interaction remains a property of a chosen fitted model and replacement game. The eight-feature experiment makes those limits measurable. It does not establish large-feature scalability or physical causal interactions.

## Further reading

- [SHAP-IQ: Unified Approximation of any-order Shapley Interactions](https://proceedings.neurips.cc/paper_files/paper/2023/file/264f2e10479c9370972847e96107db7f-Paper-Conference.pdf), especially the shared-query representation and iid estimator in Section 4.
- [KernelSHAP-IQ: Weighted Least Square Optimization for Shapley Interactions](https://proceedings.mlr.press/v235/fumagalli24a.html), especially the pairwise theorem and sequential algorithm. The higher-order WLS results include conjectures; this lesson uses the proved pairwise case.
- Section 16.3 for the discrete-derivative SII definition; 16.16 for changing the allocation rule instead of approximating a fixed one.
- Sections 16.14–16.15 for the known Concrete model/data recipe and finite-donor interpretation.

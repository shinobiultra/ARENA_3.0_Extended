# [16.19] Explain the Proxy, Check the Model

# Introduction

Suppose we can predict a model's response using a small collection of regression trees. Each tree asks a sequence of yes/no questions about which input features are present. Can we turn those questions into a signed list of interactions, and how much would that list explain about the original model?

Today you'll build the boosted-tree route used by **ProxySPEX**. You will fit trees to a model's responses, extract the exact Fourier spectrum of those trees, keep a compact set of coefficients and refit those coefficients on the queried responses. The important distinction will become visible: **an exact explanation of a proxy can still be an inaccurate explanation of the model**.

We'll start with a four-feature function whose answer we can inspect completely. Then we'll separate
fitting, extraction and truncation on larger analytic games, including a pure parity function that
the proxy fails to learn. Finally, we'll use saved GPT-2 responses from the common word-mask
experiment in 16.18. Your functions will fit fresh proxies on those responses, extract their
spectra and make predictions before comparing them with the held-out model responses.
Both the analytic experiments and the native refits run the functions you implement.

This lesson continues the Boolean Fourier convention from 16.18. A Fourier coefficient describes a signed parity pattern in a function. It is **not** a Shapley Interaction Index coefficient. The conversion in 16.18 is a separate mathematical operation, and it should be labeled when used.

## Content & Learning Objectives

### 1️⃣ Fit a function one residual tree at a time

> ##### Learning Objectives
>
> - Read a binary retention mask and its Boolean characters.
> - Implement the squared-loss residual update used in gradient boosting.
> - Distinguish a fitted tree primitive from the ensemble you build around it.

### 2️⃣ Turn tree paths into signed interactions

> ##### Learning Objectives
>
> - Expand a path indicator as a product of two-term Fourier expressions.
> - Explain why repeated features require symmetric difference, or XOR.
> - Accumulate leaf and tree contributions without changing their predictions.

### 3️⃣ Keep a compact explanation and refit it carefully

> ##### Learning Objectives
>
> - Select the smallest energy-ranked prefix that reaches a stated fraction.
> - Reproduce the author's strict-boundary failure without confusing it with the corrected selection.
> - Keep duplicate masks together and learn each validation fold's support from its training rows.
> - Solve the explicit ridge objective and distinguish development scores from an untouched audit.

### 4️⃣ Separate the errors and inspect the words

> ##### Learning Objectives
>
> - Compare model-to-proxy, extraction, truncation and refinement errors.
> - Interpret negative controls alongside successful examples.
> - Explore a path and an energy setting, then refit a word-mask explanation at a chosen prompt and query budget.

The ten exercise groups should take about 2–3 hours, plus time to explore the figures. If implementing the orchestration in the final exercise feels mechanical, read its fold loop carefully, use the solution and spend the saved time predicting the failure cases. The central ideas are the residual update, path multiplication and separation of the different errors.

## Setup

Each row of `masks` is a 0/1 vector; **1 retains a feature and 0 masks it**. Feature zero is the first column. A support is a sorted tuple of feature indices, such as `(0,2)`, and the empty tuple `()` denotes the constant term. Calculations use NumPy float64, with a wider accumulator for the energy boundary.

We supply shape validation, plotting and scikit-learn's `DecisionTreeRegressor`. The tree primitive
chooses splits and fits leaf means. You will implement the boosting loop and every spectrum,
selection and refinement operation in the notebook. The visible ensemble is an educational
squared-residual implementation; the author's actual code uses LightGBM. We will keep that
difference explicit when comparing their behavior.

<!-- code: setup -->

## Cold open: a score that requires three features together

Consider four retention bits and the score

$$f(x)=0.3+x_0+2x_0x_1x_2.$$

Feature 3 does nothing. Retaining feature 0 adds one point, but the additional two points require features 0, 1 and 2 together. The full score takes only three values: `0.3`, `1.3` and `3.3`. We can plot all sixteen masks without fitting anything.

<details>
<summary>Predict before continuing</summary>

A depth-one tree can separate `x0=0` from `x0=1`, but it cannot distinguish the two scores within `x0=1`. A depth-three tree has enough levels to represent the whole function. Having enough depth is necessary here; on other functions, a greedy split rule can still choose unhelpful early features.

</details>

<!-- code: cold_open -->

# 1️⃣ Fit a function one residual tree at a time

## A basis that changes sign

For a support $S$, define the Boolean character

$$\chi_S(x)=(-1)^{\sum_{i\in S}x_i}=\prod_{i\in S}(1-2x_i).$$

For two features, the four masks `00`, `01`, `10`, `11` give $\chi_{\{0,1\}}=[1,-1,-1,1]$. The constant character is always one. Characters form an orthonormal basis under the uniform distribution on **all** binary masks:

$$F(S)=2^{-n}\sum_{x\in\{0,1\}^n}f(x)\chi_S(x),\qquad
f(x)=\sum_S F(S)\chi_S(x).$$

We will use full cubes only for small analytic checks. The scalable extraction later will use the paths of fitted trees; it will not ask the original model for all $2^n$ responses.

Notice the sign convention. The positive retention effect $f(x)=x_0$ has spectrum $F(\varnothing)=1/2$ and $F(\{0\})=-1/2$. A negative Fourier coefficient is therefore not, by itself, a negative effect of retaining a feature.

### Exercise - implement `character_matrix` and `predict_spectrum`

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 8 minutes on this exercise.
> ```

Return a matrix with one row per mask and one column per support. Reduce each support's bit sum modulo two, then map even and odd parity to `+1` and `-1`. Reconstruct a function by multiplying this matrix by its signed coefficients. An empty spectrum predicts zero; the empty support predicts a constant.

<!-- exercise: character_matrix,predict_spectrum -->

<details>
<summary>Expected output</summary>

For the two-feature cube and supports `[(), (0,), (0,1)]`, the rows are `[1,1,1]`, `[1,1,-1]`, `[1,-1,-1]`, `[1,-1,1]`. The tests check every character through five features, including orthogonality, and reconstruct a signed polynomial independently.

</details>

<details>
<summary>Help - where does the constant column come from?</summary>

Summing a row over an empty support gives zero. Thus `1 - 2*(sum % 2)` automatically creates the constant column. A spectrum with no columns is different: return a `(rows,0)` matrix so its product with an empty coefficient vector is a zero vector.

</details>

<!-- solution: character_matrix,predict_spectrum -->

## Why the next tree fits a residual

Let $g_0(x)=\bar y$ be the mean training response. At round $t$, our current prediction is $g_{t-1}$. The negative gradient of half the squared loss is the residual

$$r_i^{(t)}=y_i-g_{t-1}(x_i).$$

Fit a regression tree $h_t$ to these residuals, then update

$$g_t(x)=g_{t-1}(x)+\eta h_t(x).$$

The learning rate $\eta$ shrinks each update. A tree with a large positive leaf value is correcting an underprediction in that leaf's region. Its leaf value is not the final model response. Prediction requires the initial mean and **every** shrinkage-weighted tree.

For training responses `[0,0,2,2]` split perfectly by one feature, the initial mean is `1`. Residuals are `[-1,-1,1,1]`. At learning rate `0.1`, one perfect residual tree changes the predictions to `[.9,.9,1.1,1.1]`; a second tree fits the remaining residuals `[-.9,-.9,.9,.9]`. Trees are trained sequentially, not independently on the original responses.

### Exercise - implement `fit_boosted_trees` and `predict_boosted_trees`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15 minutes on this exercise.
> ```

Store the initial mean, tree list, learning rate, feature count and training MSE after each round. Use the disclosed `DecisionTreeRegressor` with squared-error splitting and seed `seed + step`. Fit exactly the requested number of trees; there is no hidden early-stopping rule. The prediction function must reproduce the same accumulation on new masks.

Use the dictionary keys `initial_value`, `trees`, `learning_rate`, `n_features` and `train_mse` for these quantities. Also retain `parameters` (the four supplied tree settings) and `seed`, so later cells can identify what was fitted. The loss array has `n_estimators + 1` entries because it includes the initial constant prediction.

<!-- exercise: fit_boosted_trees,predict_boosted_trees -->

<details>
<summary>Expected output</summary>

The test reconstructs seven sequential residual fits using independent calls to the CART primitive. Your tree predictions and each recorded training loss must agree. The figure below shows decreasing training error, but you should judge prediction quality on the separately generated audit masks.

</details>

<details>
<summary>Help - initialize once, then update the running vector</summary>

Create `prediction = np.full(len(values), values.mean())` before the loop. At each round, fit on `values - prediction`, then update `prediction` in place. Use this updated vector to compute the next residual. When predicting later, start from the stored mean again.

</details>

<!-- solution: fit_boosted_trees,predict_boosted_trees -->

<!-- code: boosting_figure -->

<details>
<summary>Interpretation - why can training error be misleading?</summary>

The shuffled-response model learns patterns in the shuffled labels and can reduce its own training loss. Those patterns do not predict the real score. The audit curve compares all models to the same true responses, where the shuffled control fails. The depth-one model also has a structural limitation: a sum of stumps remains additive, regardless of the number of boosting rounds.

</details>

# 2️⃣ Turn tree paths into signed interactions

## One path is a product of small indicators

The branch `x_i=0` has indicator $(1+\chi_{\{i\}})/2$, while `x_i=1` has indicator $(1-\chi_{\{i\}})/2$. A path is the product of its branch indicators. For example, the path `x0=0`, then `x1=1` gives

$$\mathbf1[x_0=0,x_1=1]
=\frac{1+\chi_{\{0\}}}{2}\frac{1-\chi_{\{1\}}}{2}
=\frac14\left(1+\chi_{\{0\}}-\chi_{\{1\}}-\chi_{\{0,1\}}\right).$$

At mask `01`, the four signed contributions are all `+1/4`, so their sum is one. At `00`, they cancel. The interaction coefficient is part of an exact algebraic expansion of the path; it does not mean that the leaf is active at every mask containing that interaction.

The rule for multiplying characters is

$$\chi_A\chi_B=\chi_{A\triangle B},$$

where $\triangle$ is symmetric difference. A repeated feature cancels because $\chi_i^2=1$. Requiring `x0=0` twice leaves the same indicator. Requiring `x0=0` and `x0=1` produces zero. Set union gives the wrong answer for these cases.

### Exercise - implement `path_indicator_fourier`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 15 minutes on this exercise.
> ```

Start from `{(): 1.0}`. For each `(feature, required_bit)`, send half of every coefficient to its existing support and the signed other half to the support with that feature toggled. Add contributions that reach the same support. Drop exact zeros after each multiplication.

<!-- exercise: path_indicator_fourier -->

<details>
<summary>Expected output</summary>

`[(0,0),(1,1)]` expands to `{(): .25, (0,): .25, (1,): -.25, (0,1): -.25}`. Repeating the first constraint changes nothing. Adding `(0,1)` makes the whole path impossible and returns an empty spectrum. The test enumerates short paths independently, including repeated and contradictory constraints.

</details>

<details>
<summary>Help - multiplication can contribute twice to one key</summary>

Use `tuple(sorted(set(support) ^ {feature}))` for the toggled support and accumulate with `get(key, 0) + contribution`. Assignment without accumulation loses terms when different incoming supports land on the same output support.

</details>

<!-- solution: path_indicator_fourier -->

<!-- code: path_figure -->

## From leaves to a whole tree

A regression tree partitions the input cube into disjoint leaf regions. If leaf $\ell$ predicts $v_\ell$, the tree is

$$h(x)=\sum_{\ell}v_\ell\,\mathbf1[x\text{ follows path }\ell].$$

We already know how to expand each indicator. Traverse the tree, carry the current path to each leaf, multiply its spectrum by the leaf value, then add it to the running tree spectrum. The result is exact on the binary cube, including masks never used to fit the tree.

A CART threshold in `[0,1)` separates binary zero from one. A threshold below zero sends both binary values right; a threshold at least one sends both left. These one-sided cases matter when interpreting a supplied tree on binary inputs. A constant tree has just one leaf and contributes only an intercept.

### Exercise - implement `tree_to_fourier`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 18 minutes on this exercise.
> ```

Use `tree.tree_.children_left`, `children_right`, `feature`, `threshold` and `value`. A leaf has both child indices equal to `-1`. Do not multiply the leaves by a boosting learning rate here: this function extracts one tree, and the ensemble will handle shrinkage.

<!-- exercise: tree_to_fourier -->

<details>
<summary>Expected output</summary>

Every small fitted tree must match an independent full-cube Fourier calculation. A deliberately repeated-feature tree contains an unreachable leaf with value `999`; that value must make no contribution. A known tree implementing `-1.25*chi_(0,1,2)` extracts exactly one negative nonconstant coefficient.

</details>

<details>
<summary>Help - carry the path rather than the training rows</summary>

The recursion receives a node index and the list of branch constraints. At a reachable split, recurse on the left with `(feature,0)` and on the right with `(feature,1)`. At a leaf, call your path expansion and accumulate `leaf_value * coefficient`. No model evaluations or training rows are needed.

</details>

<!-- solution: tree_to_fourier -->

<!-- code: tree_figure -->

## Linearity gives the ensemble spectrum

Your ensemble is $g(x)=\bar y+\eta\sum_t h_t(x)$, so its Fourier coefficients are

$$G(S)=\bar y\,\mathbf1[S=\varnothing]+\eta\sum_t H_t(S).$$

A support can occur in many trees. Its contributions can reinforce or cancel; selecting the largest terms from each tree separately is not equivalent to selecting the largest terms from the summed spectrum. The initial training mean is also not necessarily the final Fourier intercept, because later trees may contribute constant coefficients.

### Exercise - implement `ensemble_to_fourier`

> ```yaml
> Difficulty: 🔴⚪⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 6 minutes on this exercise.
> ```

Initialize the empty-support coefficient with the stored mean. Extract every tree and accumulate its learning-rate-weighted coefficients. Keep the intercept even if it is zero.

<!-- exercise: ensemble_to_fourier -->

<details>
<summary>Expected output</summary>

Reconstructed ensemble predictions agree with the independent sum of tree predictions to numerical precision for constant, additive, parity and polynomial games. This check compares the spectrum to the **fitted ensemble**, not to the original function.

</details>

<details>
<summary>Help - there are two different sums</summary>

First, a tree sums leaf-path contributions. Then, the ensemble sums whole-tree spectra. Neither sum requires truncation or refitting. Keep both operations exact so later errors have a clear origin.

</details>

<!-- solution: ensemble_to_fourier -->

<!-- code: parity_figure -->

<details>
<summary>Interpretation - how can parity fail when its spectrum has only one term?</summary>

For pure third-order parity, conditioning on just one involved feature still leaves a balanced
positive/negative response. A greedy tree sees no immediate impurity improvement that identifies a
useful first split. With many irrelevant features, its early choices can consume its depth before it
reaches the needed combination. The bounded fitted proxy in the left panel therefore misses parity.
The exact known parity tree on the right is a separate construction: once a tree represents the
function, extraction recovers its signed coefficient. Sparsity in the Fourier basis and ease of
greedy tree fitting are different properties.

</details>

# 3️⃣ Keep a compact explanation and refit it carefully

## What does “retain 95% of the energy” mean?

Under the uniform mask distribution, Parseval's identity gives

$$\mathbb E_x[f(x)^2]=\sum_S F(S)^2.$$

If we discard coefficients without changing the others, the mean squared reconstruction error on the full cube is exactly the sum of the discarded squared coefficients. This motivates ranking coefficients by squared magnitude. It does **not** guarantee a particular error under a different mask distribution, nor does it repair a proxy that already predicts the model badly.

We exclude the intercept from the energy budget and always retain it separately. Among nonconstant
terms, sort by descending magnitude, break equal-magnitude ties by support tuple and keep the
**smallest prefix whose cumulative fraction is at least the requested fraction**. For squared
magnitudes `[9,1]`, fraction `.9` keeps the first term; fraction `.95` keeps both. A single nonzero
coefficient must survive any positive requested fraction.

The actual author implementation has a different boundary behavior: it finds the coefficient at the
95% crossing, then retains coefficients **strictly larger** than that boundary. It drops the
crossing term and all ties. Its executed pure-parity example selected no support and subsequently
failed during regression. The author source also lacks an explicit always-kept intercept. We
preserve that behavior as a diagnostic rather than silently calling the correction the original
implementation.

### Exercise - implement `select_energy_support`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 12 minutes on this exercise.
> ```

Return the selected spectrum, supports, requested fraction, retained fraction and counts. Fraction
zero means intercept only; fraction one means every nonzero nonconstant term. If there is no
nonconstant energy, retain the intercept and report that all available energy is retained. Normalize
magnitudes before squaring and use a wider accumulator so extreme finite scales and an exact `.9`
boundary behave sensibly.

The return keys are `spectrum`, `supports`, `requested_energy_fraction`, `retained_energy_fraction`, `selected_nonconstant_count` and `available_nonconstant_count`. `supports` follows the selected dictionary's order, with the intercept first; the two count fields exclude that intercept.

<!-- exercise: select_energy_support -->

<details>
<summary>Expected output</summary>

For `{(): .1, (0,): 3, (1,): -1}`, your `.95` selection keeps all three terms and reports `1.0`. The author's strict rule retains only `(0,)`, losing the intercept and keeping 90% of nonconstant energy. For pure parity, your selected support is the parity term plus an intercept; the author's selection is empty.

</details>

<details>
<summary>Help - find a prefix, not a strict magnitude threshold</summary>

After sorting, calculate the cumulative normalized energy and use `searchsorted(..., side="left") + 1`. Treat fraction zero and zero energy before searching. Build the intercept entry independently. Using `abs(value) > boundary` repeats the known author failure.

</details>

<!-- solution: select_energy_support -->

<!-- code: author_diagnostic -->

<!-- code: energy_figure -->

<details>
<summary>Interpretation - why does the corrected curve still have jumps?</summary>

The support is discrete. A small change in requested energy can add a whole coefficient, so reconstruction error changes in steps. Correcting the boundary ensures the stated retained fraction; it does not make 95% mean 100%. The additive example intentionally drops a small true term at 95%, and no coefficient-only refit can recreate a support that was removed.

</details>

## Keep duplicate masks on one side of a fold

We need to choose tree settings and ridge strength using only the queried training responses. A validation fold holds back some of those rows. If the same mask occurs twice, however, putting one copy in training and one in validation exposes exactly the response being scored. A deterministic mask game has no new information in that duplicate.

We'll group identical masks before splitting. Shuffle the unique group IDs with the supplied seed,
partition them into three folds and expand the groups back to row indices. Every validation fold has
a complementary training set, and every training row is scored once across the folds. This grouping
changes how the selection score should be interpreted compared with the author's row-based reference
CV.

### Exercise - implement `grouped_mask_folds`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 10 minutes on this exercise.
> ```

Return explicit `train_indices` and `validation_indices` for each fold. Validate that there are enough unique masks. Use `np.unique(..., axis=0, return_inverse=True)` to map rows to groups; never deduplicate the response array and lose the original row accounting.

<!-- exercise: grouped_mask_folds -->

<details>
<summary>Expected output</summary>

The same seed reproduces the split. Every row occurs in one validation fold, and no mask appears on both sides of any fold. The test uses many unequal duplicate counts, so equal numbers of unique groups need not produce equal numbers of rows.

</details>

<details>
<summary>Help - split group IDs, then recover row IDs</summary>

`np.unique` returns one group ID per original row. After shuffling and partitioning the unique IDs, `np.isin(group_ids, validation_groups)` identifies validation rows. Its complement identifies the training rows.

</details>

<!-- solution: grouped_mask_folds -->

## Ridge fits coefficients on a chosen support

For retained supports $K$, construct the character design $\Phi_{iS}=\chi_S(x_i)$. A ridge coefficient fit solves

$$\hat\beta_\alpha=\arg\min_\beta\|\Phi\beta-y\|_2^2+\alpha\|\beta\|_2^2.$$

The empty support supplies an explicit all-ones column. In this lesson, that coefficient is regularized too; there is no additional unpenalized intercept. The loss is a **sum**, so the same numeric alpha has a different relative strength when the number of training rows changes.

If $\Phi=U\operatorname{diag}(s)V^\top$, then

$$\hat\beta_\alpha=V\operatorname{diag}\!\left(\frac{s}{s^2+\alpha}\right)U^\top y.$$

One singular-value decomposition therefore gives the entire specified alpha path. Positive alpha also makes the problem well defined when retained character columns are linearly dependent on the sampled rows. Orthogonality on the complete cube does not imply orthogonality on a small sampled design.

### Exercise - implement `ridge_path`

> ```yaml
> Difficulty: 🔴🔴⚪⚪⚪
> Importance: 🔵🔵🔵🔵⚪
>
> You should spend up to 12 minutes on this exercise.
> ```

Return one coefficient row per alpha. Use a reduced SVD and broadcast the shrinkage factor across alphas. Reject empty designs and nonpositive or nonfinite alphas rather than choosing a new regularization rule silently.

<!-- exercise: ridge_path -->

<details>
<summary>Expected output</summary>

Your solutions agree with independently solved normal equations, including a design with duplicate columns. For three all-ones rows, responses all equal to `2` and alpha `1`, the fitted intercept is `6/(3+1)=1.5`.

</details>

<details>
<summary>Help - watch the orientation of the returned singular vectors</summary>

NumPy returns `u, singular, vt`. After multiplying `u.T @ values` by `singular/(singular**2 + alpha)`, multiply the row representation by `vt`. Do not invert small singular values first; the ridge formula handles them directly.

</details>

<!-- solution: ridge_path -->

## Selecting an alpha also requires training-only supports

Suppose we extract a spectrum from all training responses, select its support and then score ridge fits on validation folds. The validation responses have already helped select the columns being scored. Keeping those values out of the final regression fit is not enough.

Instead, each fold needs its own tree fit, extracted spectrum and selected support, learned from
that fold's training rows. For each candidate alpha, fit coefficients on those rows and score the
held-back rows. Pool squared errors across validation rows, then choose the first alpha attaining
the smallest pooled MSE. Finally select the support from the full training fit and refit its
coefficients on all available training rows with that alpha.

The fold supports can differ in size and identity. We are choosing a procedure that learns supports and coefficients from training data, not pretending that every fold solves the same fixed-column regression.

### Exercise - implement `refine_coefficients`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 18 minutes on this exercise.
> ```

Accept the full-training spectrum, explicit fold records and one training-only spectrum per fold. Validate the partitions and duplicate grouping. Select each fold's support, evaluate the fixed alpha path and accumulate **sums** of squared validation errors before dividing by the total number of rows. Return the final spectrum and the alpha-selection details, including each fold's supports.

Use `spectrum` for the final refitted coefficients and `selection` for the full-training energy-selection result. The `cv` record contains `alphas`, `fold_mse`, pooled `validation_mse`, `selected_index`, `selected_alpha` and `fold_supports`. Its fold-loss matrix has one row per fold and one column per alpha.

<!-- exercise: refine_coefficients -->

<details>
<summary>Expected output</summary>

The focused test deliberately supplies three different fold spectra. Your validation losses and selected alpha must match independent normal-equation calculations for those distinct designs. Overlapping partitions, incomplete validation coverage and duplicate masks crossing a fold must raise errors.

</details>

<details>
<summary>Help - unequal folds should not receive equal weight automatically</summary>

For each alpha, add the total squared error over that fold's validation rows. Divide once by the total number of rows after processing all folds. An unweighted average of fold means would overweight small folds. The final full-training fit happens only after alpha selection.

</details>

<!-- solution: refine_coefficients -->

# 4️⃣ Separate the errors and inspect the words

## Put the steps together without giving them the audit answers

Our prospective tree grid is depths `{1,3}` by tree counts `{128,512}`, with learning rate `.1`, minimum leaf rows `2` and three grouped folds. The ridge alpha list is `(1e-6, 1e-3, .1, 1, 10)`. These choices were fixed before the model experiment. The original author used a larger LightGBM grid, five-fold R² scoring and its own RidgeCV behavior; the two fitting procedures are not identical.

Choose the proxy candidate by pooled validation MSE. Retain that candidate's separately fitted fold spectra for the ridge-selection exercise. Fit the chosen tree settings once on all training rows, extract and truncate that spectrum, then perform the final coefficient refinement.

Using the same development folds to select tree settings and ridge alpha is **not nested cross-validation**. The resulting scores have been used to choose the procedure and can be optimistic. They are useful selection records, not an independent estimate of generalization. The audit masks and values must remain outside `fit_proxy`; they enter only the evaluation below.

### Exercise - implement `fit_proxy` and `stage_errors`

> ```yaml
> Difficulty: 🔴🔴🔴⚪⚪
> Importance: 🔵🔵🔵🔵🔵
>
> You should spend up to 20 minutes on this exercise.
> ```

Connect your existing functions. Return the proxy, full/truncated/refined spectra, selected settings, explicit fold indices and fit counts. Count individual tree fits as well as whole ensembles. This function receives query rows and values, so it makes zero oracle calls itself; the caller must separately count how those rows were obtained.

For evaluation, report RMSE for model→proxy, proxy→extracted spectrum, full spectrum→truncated spectrum and truncated→refined spectrum. Also report each resulting predictor's RMSE against the model. These RMSEs do **not** add: different errors can cancel or reinforce. Refinement can make model fidelity worse, and that outcome should remain visible.

<!-- exercise: fit_proxy,stage_errors -->

<details>
<summary>Expected output</summary>

The small orchestration test observes every tree fit and checks that each candidate's validation fold is absent from its training arrays. Your selected fold supports, final fit count and reconstructed proxy must agree. `fit_proxy(..., audit_values=...)` must be rejected: the fitting API has no audit argument.

</details>

<details>
<summary>Help - save fold spectra while evaluating each tree candidate</summary>

Within the candidate loop, fit one model per fold and collect its extracted spectrum. When a candidate improves the pooled loss, retain its fold spectra. After choosing the candidate, fit the full-training ensemble and pass those retained spectra to `refine_coefficients`. The audit values are never needed for any of these operations.

</details>

<!-- solution: fit_proxy,stage_errors -->

<!-- code: student_namespace -->

## Three games, three different outcomes

Now use 456 sampled masks of 28 features and 2,048 independent audit masks. The function values are analytic, not language-model outputs. The masks are disjoint, and the audit responses are computed only for evaluation. This scale is large enough that we are not enumerating the input cube, while the formulas still give us an independent answer at every requested mask.

We compare an additive score, a signed parity and a sparse polynomial. For each game, also fit on
shuffled training responses. A constant predictor uses the training mean; an additive least-squares
predictor uses the same training rows. All predictors are evaluated against the same real audit
responses. The code below runs your full fixed grid again; the plotted values are not loaded from a
report.

<!-- code: analytic_study -->

<!-- code: analytic_signature -->

<details>
<summary>Interpretation - read the stages before ranking the methods</summary>

For the additive game, the tree fit is very accurate, but 95% energy selection intentionally omits a
small true coefficient. The refit cannot add that missing column. For the parity game, the fitted
proxy itself is poor, while extraction still reproduces it accurately. For the polynomial, the
selected support can permit a much better coefficient fit than the tree's original coefficients.
Look at the actual executed figures for the magnitudes and the controls; none of these outcomes is a
guarantee for an arbitrary model.

The extraction check is shown separately as a prediction-against-prediction plot, with its near-floating-point RMSE stated explicitly. A nearly zero extraction error says the algebra worked. The model-prediction panels answer the different question of whether the learned explanation predicts the target responses.

</details>

<!-- code: cv_figure -->

<details>
<summary>Interpretation - what was fitted, and what was queried?</summary>

Four candidates across three folds plus one full-training fit give 13 fitted ensembles per run.
Individual tree counts also include every tree in those ensembles. Five ridge alphas across three
folds plus a final selected-alpha solution give 16 coefficient fits, computed with four SVDs.
Reusing training arrays for these fits makes no additional calls to the queried function. In a
language-model study, requested masks, unique model evaluations, cache hits and inference time must
be counted by the query caller, separately from these fitting counts.

</details>

## Try It Yourself: change a path and its energy budget

Edit `PLAY_PATH` and `PLAY_ENERGY` below. Start with a repeated constraint, then make it contradictory. Predict the full spectrum before executing. Next lower the retained fraction and find masks where the truncated reconstruction differs most from the original indicator. This play calls your path expansion and support selection directly; it makes no model queries.

<!-- code: analytic_play -->

## Apply the same check to word masks

In the common experiment with 16.18, each of the 28 features is one word position in a GPT-2
prompt. Retaining a word keeps all its tokens. Masking it replaces those tokens by the EOS token
and removes their attention keys, while keeping every absolute position fixed. The final ` to`
suffix remains visible. Repeated occurrences of the same word are separate features.

The response is the next-token logit of the first name minus that of the second name. A positive
unmasked response favors the intended recipient in the original prompt. A subject-only
counterfactual changes the intended recipient, so its expected sign reverses. These controls ask
whether GPT-2 exhibits the behavior we want to explain. Approximation error asks whether the
explanation predicts its responses under masking. Neither question answers the other.

The comparison must show SPEX, ProxySPEX, constant/additive baselines and shuffled-response controls on the same independent audit masks. The word-labeled spectrum should sit beside its prediction errors. A large interaction over names or other words is interesting only with its sign convention, model behavior and fidelity evidence in view.

The shared data file contains actual queried responses, the fixed mask plan and the recorded
comparison across 32 prompts, three seeds and three query budgets. The loader verifies its data
contract and hashes. It does not load a fitted tree model. The function below requires all thirteen
algorithm functions to belong to your completed notebook namespace, then fits real and shuffled
responses using those functions. It attaches audit responses only after constructing predictions.

The first plot describes the complete recorded comparison. Its shaded bands show the middle half
of the prompt-seed results, not a confidence interval: templates and name swaps are related.
The second plot is your own fresh refit at prompt 0, seed 0 and 1,824 requested training rows,
chosen before inspecting outcomes. All 512 audit masks remain outside training and selection.
The fixed SPEX result is labeled as a saved comparator; you can reconstruct SPEX yourself in 16.18.

<!-- code: native_bridge -->

<!-- code: native_study -->

<details>
<summary>Expected output and interpretation - separate fidelity, algebra and behavior</summary>

The native cell renders the complete comparison and your selected refit's predictions, stage
errors and signed word terms. It prints the settings selected by training-only cross-validation
and the actual ensemble and tree fit counts. The extraction predictions should lie on the
proxy identity line, within floating-point error. There is no corresponding guarantee that
the proxy or refined spectrum beats the additive baseline or that shuffled labels always do worse
on every individual prompt.

Start with the unmasked and counterfactual signs. Then compare the real-response fit with
query-matched baselines and shuffled-response fits. Finally, locate any loss of fidelity between
fitting, truncation and refinement. A negative Fourier coefficient means the score is lower when
its character is positive; it does not directly mean that each named word has a negative causal
effect. A parity term changes sign when any one of its retained bits changes.

</details>

## Compare the visible ensemble with the actual implementations

Our residual loop uses scikit-learn CART trees and the small disclosed grid. The author's
ProxySPEX and the maintained implementation use LightGBM with their own actual grids and
selection code. The shared experiment also ran those implementations at prompt 0, seed 0 and
each budget, supplying the fixed query responses to their fitting routines. The table below
shows their recorded stage errors. `full` is the extracted spectrum, `selected` is its truncated
spectrum, and `final` is the implementation's returned result. Missing stages remain missing;
the accompanying status and source exception explain why.

These are fixed-query algorithm comparisons. They do not reproduce the packages' public random
sampling interfaces, and they do not establish equivalence between our boosting loop and LightGBM.
Compare a reference's tree error with its full-spectrum error before attributing a difference to
support selection. The exact-query records also preserve the author's SPEX early-return behavior.

<!-- code: native_references -->

## Try It Yourself: inspect a prompt and budget

Change `PROMPT_INDEX` from 0 to 31, `QUERY_BUDGET` among 456, 912 and 1824, and `SEED` among 0, 1
and 2. Predict whether a larger query budget will change the support, the error or both. Executing
the cell fits a new proxy and shuffled-response control with your functions on the saved training
responses. It makes zero additional model queries. The budget curves describe the fixed recorded
study; the word plot shows the proxy you have just refitted beside the saved SPEX comparator.

Inspecting another prompt after seeing its audit is exploration. It does not supply a new
independent test. Keep the complete comparison in view when discussing a striking interaction,
and remember that equal requested budgets can contain different numbers of unique masks.
Requesting a new prompt or new model responses would be a separate experiment with additional costs.

<!-- code: native_play -->

## What we have learned, and where it stops

A tree path can be expanded exactly because its branch indicators have simple Fourier
representations. XOR multiplication handles repeated features, and linearity extends the result to a
genuine boosted ensemble. Those are algebraic guarantees about the fitted proxy. They do not
guarantee that greedy trees discover a sparse parity, that an energy cutoff preserves every
meaningful term or that a refit improves an unseen model response.

Our visible implementation uses a bounded squared-residual tree grid. The author's actual LightGBM
execution and the maintained-package execution are separate reference results, with their own fitted
models and selection behavior. The strict author boundary failure remains a useful counterexample,
not a result that the corrected implementation should inherit or conceal. The complete input cube
was used only for small analytic checks; the 28-feature examples use sampled masks with
independently computed audit values.

The word-mask experiment measures approximation to a defined intervention on GPT-2. Its prompts
share four templates and a small name inventory, and masking creates unusual inputs. The results
do not establish the causes of natural-language behavior, transfer to arbitrary prompts or
performance on a much larger model. Replaying cached responses verifies your implementation's
use of those data; it does not reproduce the model's original inference or create a fresh audit.

## Further research: investigate a failure you can explain

Choose one direction and keep the prediction target fixed:

- Increase irrelevant features in a pure parity game while holding depth and tree count fixed. Compare tree error with extraction error to locate the failure.
- Change retained energy around a boundary tie. Track the actual retained fraction and the signed supports, not only their count.
- Compare the ridge choice with the untouched audit after the fixed run. If you start selecting alpha from that audit, label the result exploratory and obtain a new audit before claiming generalization.
- Construct two equally accurate proxies with different spectra. Which interactions are stable, and which reflect underdetermination from the queried masks?

## Reading

- [ProxySPEX paper](https://papers.neurips.cc/paper_files/paper/2025/hash/68dced1638d71d4d21598f79ae91262e-Abstract-Conference.html), especially Appendix A.2 for tree-spectrum extraction and the discussion of hierarchical structure.
- [Pinned author source](https://github.com/basics-lab/spectral-explain/tree/61400b5388b8ecbf18fe18c4d821ffe3e8a5f9c8), particularly `utils.py` for the actual LightGBM grid, extraction and strict truncation rule.
- [SPEX paper](https://proceedings.mlr.press/v267/kang25a.html) and 16.18 for sparse recovery in the Fourier basis and the separate attribution conversion.
- Repository reference evidence: `docs/arena_salvage/2026-09-08_spex_reference_execution.md`. This records untouched analytic author runs, including the pure-parity regression failure. It supports the source comparison; it is not the source of the learner's figures.

# A contribution accountant for one training run

The eight-example experiment gave us two rankings with a high correlation. But
what, exactly, did the gradient-based ranking allocate? It did not retrain every
subset. Nor did its values necessarily add up to the improvement of the trained
model. We'll now give it a precise target and measure its approximation error.

Imagine pausing training just before an SGD update. Four images are about to
change the weights. We can ask how much each image contributes to the change in
validation loss at **this checkpoint**. Then we can repeat the question at every
checkpoint actually visited by the full training run.

## Content & Learning Objectives

### 1️⃣ Define the local game

> ##### Learning Objectives
>
> - Distinguish retraining a subset from removing its update at a fixed checkpoint.
> - Keep the original batch denominator when evaluating local coalitions.
> - Explain why exact step credits telescope but Taylor credits need not.

### 2️⃣ Compute gradient similarities without storing every gradient

> ##### Learning Objectives
>
> - Derive the affine-layer ghost dot product, including biases.
> - Check it against independently computed parameter gradients.
> - Identify why parameter sharing or batch-dependent layers need another derivation.

### 3️⃣ Debug a nonlinear model's training data

> ##### Learning Objectives
>
> - Attribute a real minibatch-SGD trajectory on handwritten digits.
> - Compare a frozen trajectory explanation with actual deletion and retraining.
> - Test shuffled targets and matched random deletions, and measure computation cost.

We'll use plain SGD so that the update is exactly the one in our equations.
Momentum, Adam, clipping and weight decay change the update rule; the formulas
below are not automatic implementations of those variants.

<!-- code: setup -->

## Exercise 7 — expose the neural network's gradients

```yaml
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵🔵⚪
Time: 25 minutes
```

Our classifier has two affine layers and a tanh hidden layer. The parameters are
stored as `(W1, b1, W2, b2)`, with weights shaped `(input, output)`. For each
example, return the input activation and the loss derivative with respect to
each affine layer's output. Do **not** average the derivatives across the batch.

The output derivative for binary cross entropy is $\sigma(s)-y$. For the hidden
layer, propagate that derivative through `W2` and multiply by $1-h^2$.
`per_example_gradients` then expands each layer into its weight and bias
gradients. This intentionally expensive version will be our correctness oracle.

<details><summary>Help — dimensions and the bias</summary>

For a single example, the weight gradient is the outer product of the layer's
input and output derivative. Its bias gradient is just the output derivative.
Flatten the weight gradient in `(input, output)` order, then append the bias
gradient. Concatenate layers in forward order.

</details>

<!-- exercise: mlp_forward,mlp_factors,per_example_gradients -->
<!-- solution: mlp_forward,mlp_factors,per_example_gradients -->

**Expected:** every coordinate agrees with a separate `nn.Sequential` model's
autograd gradient, including the bias coordinates.

## Exercise 8 — ghost dot products

```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵🔵🔵
Time: 20 minutes
```

Let $a_i$ be an affine layer's input and $d_i$ its output derivative for example
$i$. Its weight gradient is $a_i d_i^T$. Expanding the Frobenius inner product gives

$$\langle a_i d_i^T,a_j d_j^T\rangle
   = (a_i^T a_j)(d_i^T d_j).$$

The bias contributes another $d_i^T d_j$. Therefore the full contribution from
this layer is $(a_i^T a_j+1)(d_i^T d_j)$. Sum that expression over the layers.

Implement the entire training-by-validation similarity matrix using matrix
multiplication and elementwise multiplication. No array should acquire a
`(batch, number_of_parameters)` dimension in this implementation.

<details><summary>Help — two Gram matrices, one elementwise product</summary>

For each layer compute `a @ other_a.T` and `d @ other_d.T`. Add one to the first
matrix for the bias. Multiply the matrices elementwise, then add the layer
contributions. Different layers own different parameters, so there are no
cross-layer terms in the parameter-space dot product.

</details>

<!-- exercise: ghost_dot_products -->
<!-- solution: ghost_dot_products -->

**Expected:** agreement with the explicit-gradient oracle to floating-point
roundoff. Zero inputs must still produce nonzero similarities when their bias
gradients are nonzero. Examples 0 and 1 below are exact duplicates.

<!-- code: ghost_picture -->

<details><summary>Interpretation — what has disappeared?</summary>

The gradient information has not been approximated. We changed the order of
operations so that the outer-product parameter gradients never need to be
stored. This identity assumes independent examples and separate affine
parameters. Applying it unchanged to tied weights, token-shared projections or
batch normalization can omit cross terms. Faster code is useful only after the
identity matches the model being analyzed.

</details>

## Exercise 9 — define the game before allocating it

```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵🔵🔵
Time: 25 minutes
```

Write $g_{t,i}=\nabla_\theta\ell_i(\theta_t)$ and let the original minibatch have
$B$ examples. The actual SGD update is

$$\theta_{t+1}=\theta_t-\frac{\eta}{B}\sum_{i\in B_t}g_{t,i}.$$

For a coalition $S\subseteq B_t$, hold the starting checkpoint and denominator
fixed, and use only its gradient contributions:

$$v_t(S)=L_{val}(\theta_t)-L_{val}\left(\theta_t-
          \frac{\eta}{B}\sum_{i\in S}g_{t,i}\right).$$

Positive utility means the update helps. This sign is the opposite of the
loss-change convention in the In-Run Data Shapley paper. We also show the batch
normalization explicitly rather than absorbing it into the learning rate.

Implement `sgd_step` and `local_coalition_values`. Enumerate coalitions in binary
mask order, so entry 0 is empty and entry `2**B - 1` contains the whole batch.
Every candidate must start from the same parameters; do not mutate the checkpoint.

<details><summary>Help — why not divide by the coalition size?</summary>

We are removing contributions from a realized update, not redefining the
optimizer for each coalition. Dividing by `len(S)` would amplify the remaining
examples when another is removed, changing the game. Keep `B` even for singleton
coalitions. The empty coalition has no update and zero utility.

</details>

<!-- exercise: sgd_step,local_coalition_values -->
<!-- solution: sgd_step,local_coalition_values -->

**Expected:** all sixteen utilities for a four-example batch match independently
constructed flat-parameter updates. The original checkpoint remains unchanged.

## Exercise 10 — allocate each local update exactly

```yaml
Difficulty: 🔴🔴⚪⚪⚪
Importance: 🔵🔵🔵🔵⚪
Time: 15 minutes
```

Adapt your earlier exact Shapley implementation to the dense bit-mask table.
For player $i$, average its marginal contribution over coalitions not containing
$i$, with weight $1/[B\binom{B-1}{|S|}]$.

<details><summary>Help — finding an adjacent coalition</summary>

`mask & (1 << i)` tells you whether player `i` is present. If absent,
`mask | (1 << i)` is the coalition with it added. `mask.bit_count()` gives the
coalition size. The weights already account for all permutation multiplicities.

</details>

<!-- exercise: batch_shapley -->
<!-- solution: batch_shapley -->

**Expected:** agreement with all 24 explicit permutations, and
$\sum_i\phi_{t,i}=v_t(B_t)$ because $v_t(\varnothing)=0$.

## Exercise 11 — first- and second-order credit

```yaml
Difficulty: 🔴🔴🔴🔴⚪
Importance: 🔵🔵🔵🔵⚪
Time: 25 minutes
```

Let $g_{val}$ be the gradient of **mean** validation loss and $\alpha=\eta/B$.
The first-order local game is additive, giving

$$\phi^{(1)}_{t,i}=\alpha\,g_{t,i}^T g_{val}.$$

For the quadratic approximation with validation Hessian $H_{val}$, the self
term belongs wholly to example $i$, while each pair interaction is split equally.
The resulting allocation is

$$\phi^{(2)}_{t,i}=\alpha\,g_{t,i}^Tg_{val}
-\frac{\alpha^2}{2}\,g_{t,i}^TH_{val}\sum_{j\in B_t}g_{t,j}.$$

Implement both expressions. `first_order_scores` receives similarities against
individual validation examples, so average its columns. The second-order
exercise uses explicit tiny gradients and a Hessian to make the derivation
inspectable; it is not a scalable ghost Hessian implementation.

<details><summary>Help — the factor of one half</summary>

Expand the quadratic loss term for a coalition. The diagonal contribution is
$-\alpha^2 g_i^T H g_i/2$. An unordered pair contributes
$-\alpha^2 g_i^T H g_j$, half to each member. Combining them yields the formula
above, including the self term in the sum over `j`.

</details>

<!-- exercise: first_order_scores,second_order_scores -->
<!-- solution: first_order_scores,second_order_scores -->

**Expected:** the quadratic formula exactly allocates a known quadratic game.
On a nonlinear network it remains an approximation. The next experiment measures
that error against exact local Shapley, without requiring either approximation
to win at every step size.

## Try It Yourself — when does the approximation stop balancing?

Change `TRY_STEP_SIZES`. Predict how the maximum attribution error and the error
in total allocated improvement should change. Then compare the first-order,
second-order and exact results on the same checkpoint and examples.

<!-- code: local_play -->

<details><summary>Interpretation — efficiency is about a particular game</summary>

Both Taylor formulas allocate their approximated games exactly. Their sum need
not equal the true nonlinear loss reduction. A nonzero balance error does not
mean the Shapley formula is wrong; it measures the cost of replacing the original
game by its Taylor approximation. Large updates can invalidate that replacement.

</details>

## Exercise 12 — accumulate a complete training trajectory

```yaml
Difficulty: 🔴🔴🔴⚪⚪
Importance: 🔵🔵🔵🔵🔵
Time: 30 minutes
```

Implement the accountant. Before each SGD update, compute training and validation
factors at the **same current checkpoint**, allocate that step, and add its
credits to the corresponding training rows. Use `index_add_` so repeated visits
to an example accumulate. Then perform the actual training update.

When `exact=True`, also enumerate the small batch's local game. Summing exact
step credits over training should telescope:

$$\sum_i\sum_t\phi_{t,i}=L_{val}(\theta_0)-L_{val}(\theta_T).$$

Only this full-coalition sum follows the real training trajectory. For a smaller
coalition, every local game still starts from the checkpoint produced by **all**
training data. Its cumulative utility is not the loss of a model independently
retrained on that coalition.

<details><summary>Help — deletion is a new trajectory</summary>

For `removed`, zero the corresponding examples' gradient contributions but keep
the original schedule and denominator. Later checkpoints will change, so later
gradients must be recomputed. This is our fixed-cost retraining control. It is
different from subtracting an example's previously recorded scores.

</details>

<!-- exercise: run_accountant -->
<!-- solution: run_accountant -->

**Expected:** exact step allocations equal every measured loss difference and
their total equals the initial-to-final improvement. Removing every row leaves
the initialization unchanged. Validation labels affect the scores, not SGD.

## Which handwritten images helped this classifier learn?

We now use the pinned UCI handwritten-digit data from 16.9: 128 training images,
48 separate ranking images and 64 images from the official test file. The labels
encode digit 1 as zero and digit 7 as one; twelve training labels have predeclared flips. A 64→16→1 tanh
network trains for four epochs in batches of four. We keep all three declared
seeds, including the first seed's exact local allocations at every step.

The ranking split defines the attribution target. We choose deletion sets before
evaluating any final test score. Each retraining uses the same initialization and
original batch schedule, with removed rows contributing zero and denominator four.
We compare low credit, high credit, shuffled-target credit, known corruption and
eight random sets matched to the low-credit set's observed-label counts.

This is a controlled classifier experiment. Its class pair, small network and
planted label noise do not establish performance on natural annotation errors or
foundation-model pretraining. The fixed splits also reuse the earlier chapter's
data; they are not a new independent benchmark.

<!-- code: digits -->

## Follow the loss, then inspect individual credits

The exact cumulative curve should follow the measured ranking-loss improvement.
The first-order curve can deviate. In the scatter plot, every training image
appears, with known corruptions marked rather than filtered out.

<!-- code: trajectory_plot -->

For the declared first seed, the actual loss improvement and exact total credit
are both about **0.5969**, while the first-order credits sum to about **1.2847**.
The shortcut allocates more than twice the observed improvement. Its rankings
can still contain useful information, but its values are poorly calibrated in
this run. Keep that failure visible when interpreting the next experiment.

<details><summary>Interpretation — the same image can matter differently later</summary>

A row receives credit only when it appears in the batch. Its sign depends on the
current network and validation target. The shuffled-target comparison keeps the
training trajectory exactly fixed: only the question asked of that trajectory
changes. Do not read a single cumulative rank as a permanent property of an image.

</details>

## Does low credit justify deleting an image?

Look at the digit gallery before the aggregate deletion plot. A red border marks
a known flipped training label. Low credit may also identify a correctly labeled
but difficult or underrepresented example. Inspect both possibilities.

In the final plot, positive values mean deletion lowered untouched test loss.
Every seed and every random control is shown. Compare the spread as well as the
mean: exact accounting for one trajectory does not guarantee useful data pruning.

<!-- code: deletion_plot -->

The fixed comparison includes a clear counterexample: deleting the twelve
lowest-credit examples improves test loss for seeds 16701 and 16703, but **worsens
it for seed 16702**. Removing the known label flips helps all three. The low-credit
rule is therefore neither an oracle for the corrupted labels nor a guaranteed
pruning strategy, even though it improves the mean in this small comparison.

## Measure the cost instead of assuming it

Our explicit-gradient and ghost implementations compute the same similarities.
The timing experiment reports repeated measurements on your actual CPU. Small
matrices, threading, allocator behavior and concurrent work can dominate the
timings. This does not reproduce the paper's foundation-model throughput study.
The second panel measures complete training with and without accounting, checks
that both routes reach identical parameters, and includes validation-gradient and
loss-recording overhead. A fast dot-product kernel need not make full accounting free.

<!-- code: timing -->

## Try It Yourself — remove one image and follow the new path

Change `TRY_REMOVED_ROW` to a negative-credit row, then to a positive-credit row.
Compare the original and retrained ranking-loss trajectories. Does the sign of
the predicted credit agree with the final difference? If it disagrees, find
where the trajectories began to diverge. This is exploratory reuse of the ranking
split; it must not replace the predeclared final test comparison above.

<!-- code: native_play -->

## Further investigation

- Compare early and late credit for the same image, using the saved step matrix.
- Construct two identical rows and change only their positions in the SGD schedule.
  Which symmetry applies to a local game, and which fails across different visits?
- Compare a smaller learning rate at a fixed number of steps with a longer run.
  Explain why these are different experiments before comparing their scores.
- Derive which assumptions break when the optimizer adds momentum or clipping.

The first-order calculation connects to dense-checkpoint TracIn, while the exact
local game identifies what is being approximated. It is not exact retraining
Data Shapley. Ghost products remove one computational bottleneck; they do not
remove this distinction or establish that a negative-credit example is harmful
under every possible retraining intervention.

For the original definitions and scalable implementation, read Sections 3–4 of
[Data Shapley in One Training Run](https://proceedings.iclr.cc/paper_files/paper/2025/file/20fdaf67581e6d7157376d1ed584040a-Paper-Conference.pdf)
and the authors' [GhostSuite repository](https://github.com/Jiachen-T-Wang/GhostSuite).
This lesson implements the independent-example affine identity explicitly; it
does not claim GhostSuite package parity or support for arbitrary neural layers.

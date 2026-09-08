# [16.18] Find the Interactions Without Trying Every Subset

Can a few carefully chosen groups of queries identify a sparse, signed interaction spectrum?

> **Unpublished analytic draft.** The exercises and analytic figures execute. The shared GPT-2 study and its native play are **UNEXECUTED** here. No native result is implied by the analytic examples.

# Introduction

Suppose a score depends on 31 switches. A switch could represent whether a word is retained in a prompt. You can query the score for any setting, but checking every setting would take more than two billion queries. You suspect only a few groups of switches matter together. How could you find those groups without listing them first?

SPEX turns this question into a sparse Fourier recovery problem. Its sampling patterns make many unknown interactions share a small number of **bins**. Different shifts give each interaction a distinctive sign pattern. Error-correcting codes recover a candidate from that pattern; signed subtraction then makes previously mixed bins easier to interpret.

You will implement that chain yourself. The substantial mathematical step is BCH decoding, so we supply finite-field arithmetic and work through the encoder, syndromes, error locator and root search separately. Each exercise has immediate tests and an adjacent solution. The time estimates are guides: inspecting a solution and then explaining its key step is a useful way to keep moving.

The signature result is a visible collision becoming a signed interaction. We will then ask whether the same method still works when the spectrum is dense, the degree budget is wrong, or the training responses are shuffled. Those controls are part of the question, not optional decoration.

## Content & Learning Objectives

### 1️⃣ Read a score in a different basis

> ##### Learning Objectives
>
> - Evaluate Boolean characters with a consistent bit convention.
> - Implement the normalized fast Walsh–Hadamard transform.
> - Distinguish a signed Fourier coefficient from a feature attribution.

### 2️⃣ Put unknown interactions into observable bins

> ##### Learning Objectives
>
> - Construct shifted GF(2) subcube queries.
> - Explain the identity connecting one transformed bin to its hidden supports.
> - Count requested masks and distinct masks separately.

### 3️⃣ Decode a low-degree support

> ##### Learning Objectives
>
> - Build the same systematic shortened BCH design as the author implementation.
> - Turn parity observations into syndromes, an error locator and feature positions.
> - Use uncertain parity signs in a bounded Chase search.

### 4️⃣ Subtract collisions and check the explanation

> ##### Learning Objectives
>
> - Estimate signed amplitudes and reject inconsistent candidates.
> - Implement iterative factor messages and collision subtraction.
> - Test sparse recovery against exact truth and meaningful failures.

### 5️⃣ Connect the spectrum to attribution and a word-mask game

> ##### Learning Objectives
>
> - Convert Fourier terms to Shapley values and the Shapley interaction index.
> - Define whole-word masking without moving absolute token positions.
> - Separate native approximation fidelity from correctness of a language-model answer.

## Setup

Use the repository's existing environment. This lesson does not install `galois`, `sparse-transform` or a model. The supplied field tables perform arithmetic; the functions that implement SPEX appear below.

<!-- code: setup -->

# 1️⃣ Read a score in a different basis

## Three switches can hide a joint effect

Our first score depends only on features 1, 3 and 7. The other 28 features do nothing. An odd number of retained active features changes the sign of the score. Compare this with an additive score, where each switch has the same effect regardless of the other settings.

<!-- code: cold_open -->

Look at `000 → 001` and `010 → 011`. The last switch changes the parity score in opposite directions. A single per-feature weight cannot reproduce that behavior. Our goal is to recover the group `{1,3,7}` and its **negative** coefficient.

## Boolean characters

For a set of feature positions $S$, define

$$
\chi_S(x)=(-1)^{\sum_{i\in S}x_i}, \qquad
f(x)=\sum_{S\subseteq[n]}F(S)\chi_S(x).
$$

Here `x` is a binary retain mask, not a vector of word embeddings. A support is a tuple of feature positions, such as `(1, 3, 7)`; `()` means the constant. Feature 0 is the first column. For integer IDs it is also the most significant bit: `[1,0,0]` is ID 4.

### Exercise - Implement signed Boolean characters

> **Difficulty:** 🔴⚪⚪⚪⚪ | **Importance:** 🔵🔵🔵🔵🔵 | **Time:** 10 minutes

Implement `boolean_character` and `predict_spectrum`. Empty supports should produce ones. Accumulate the actual signed coefficients; taking absolute values would change the function you are explaining.

<details><summary>Help - Where do the signs come from?</summary>

Sum just the selected columns, reduce modulo two, and map parity 0 to +1 and parity 1 to -1. A prediction is a weighted sum of these character vectors. You do not need a matrix containing every possible support.

</details>

<!-- exercise: characters -->

**Expected:** every character through five features agrees with its direct parity definition, including the constant and negative coefficients.

<!-- solution: characters -->

## The Walsh transform

On a small complete cube, orthogonality gives

$$F(S)=2^{-n}\sum_x f(x)\chi_S(x).$$

The fast transform uses the same butterfly repeatedly: replace a pair of blocks $a,b$ by $a+b,a-b$. Divide by the transform length once, at the end. This gives the **normalized forward** transform. Applying that same function twice returns the original input divided by its length.

### Exercise - Implement the normalized Walsh transform

> **Difficulty:** 🔴🔴⚪⚪⚪ | **Importance:** 🔵🔵🔵🔵🔵 | **Time:** 15 minutes

Implement `normalized_fwht`, preserving arbitrary leading batch dimensions and the requested axis. Copy both halves of a butterfly before overwriting either one.

<details><summary>Help - Why copy the two halves?</summary>

If you overwrite `left` before computing `left - right`, the second assignment reads the updated values. Work on a copy, reshape into blocks of length `2 * width`, and advance `width` by factors of two.

</details>

<!-- exercise: walsh -->

**Expected:** all 62 basis functions for $n=1,\ldots,5$ transform to the correct one-hot coefficient vector. Forward followed by the scaled inverse reconstructs every input exactly in these tests.

<!-- solution: walsh -->

<!-- code: walsh_figure -->

The three nonzero bars recover `.3 + .5 χ_{0} - .8 χ_{1,2}`. This small exhaustive example checks the convention. It is not the scalable algorithm: the next section transforms only a small **subcube** of a much larger feature space.

# 2️⃣ Put unknown interactions into observable bins

## Hash supports, then shift the queries

Draw a binary matrix $M\in\mathbb F_2^{n\times b}$. For a small mask $\ell\in\mathbb F_2^b$ and a shift $d\in\mathbb F_2^n$, query

$$u_d(\ell)=f(M\ell+d\pmod 2).$$

The $b$-bit Walsh transform satisfies

$$U_d(j)=\sum_{S:M^\top 1_S=j}\!F(S)(-1)^{d^\top1_S}.$$

This identity is the central mechanism. A support hashes to $j=M^\top1_S$. A bin can be empty, contain one support, or contain a collision. The zero-shift bin sums the coefficients. Other shifts change signs in a support-dependent way.

### Exercise - Implement GF(2) queries and bin IDs

> **Difficulty:** 🔴🔴⚪⚪⚪ | **Importance:** 🔵🔵🔵🔵🔵 | **Time:** 15 minutes

Implement `shifted_subsample_queries` and `bin_index`. Return queries in `(shift, subcube mask, feature)` order. All matrix operations here are over GF(2): reduce modulo two.

<details><summary>Help - Which dimension is transformed?</summary>

Enumerate the $2^b$ small masks, multiply by `matrix.T`, then add each shift. The feature dimension remains n. `bin_index` computes a b-bit vector from the support and converts that vector to a big-endian integer.

</details>

<!-- exercise: shifts -->

**Expected:** each query agrees with a direct scalar GF(2) calculation, through a 31-feature example. Empty supports hash to bin zero.

<!-- solution: shifts -->

<!-- code: shift_example -->

Both hidden terms reach bin `10`. Its four observed values are `[.3, -1.3, 1.3, 1.3]`. They are not a single coefficient times a ±1 signature, so simply reading their signs is insufficient. After one coefficient is learned elsewhere, subtracting its signature can reveal the other.

# 3️⃣ Decode a low-degree support

## A systematic code gives us the shifts

We need to recognize a support without trying all $\binom{n}{0}+\cdots+\binom{n}{t}$ possibilities. SPEX builds a binary BCH code with a message dimension at least n and an error-correction budget t. Its systematic column-generator has the form

$$G=\begin{bmatrix}I\\P\end{bmatrix},\qquad
c=Gk=\begin{bmatrix}k\\Pk\end{bmatrix}.$$

The parity matrix $P$ becomes the shift matrix, with one zero row prepended. If a bin contains $F(S)\chi_S$, dividing its shifted values by its zero-shift value reveals $(-1)^{Pk}$. The coefficient's sign cancels in that ratio.

The source uses a full row-generator `[I | P_full.T]`. When its message dimension exceeds n, it keeps the **last n message rows**. We match that convention exactly; replacing it with an arbitrary parity-check matrix would create a different sampler and decoder.

### Exercise - Construct and encode the shortened BCH code

> **Difficulty:** 🔴🔴🔴🔴⚪ | **Importance:** 🔵🔵🔵🔵⚪ | **Time:** 30 minutes

Implement `bch_generator_polynomial`, `make_bch_code` and `bch_encode`. The supplied `binary_field(m)` performs finite-field multiplication and division. Field addition is bitwise XOR. The generator polynomial contains roots $\alpha,\ldots,\alpha^{2t}$, closed under the binary conjugacy map $r\mapsto2r$.

<details><summary>Help - A route through the construction</summary>

1. For each desired root power, repeatedly double it modulo $2^m-1$ until its conjugacy class closes.
2. Multiply the factors $x+\alpha^r$ in the extension field. Conjugacy closure makes the final coefficients binary.
3. Search increasing m until `nc - degree(generator) >= n`.
4. Build the visible systematic recurrence, keep the last n message rows, and transpose their parity part.

The actual BCH library uses particular primitive polynomials. Its defaults differ from generic `galois.GF` defaults at some field sizes; the supplied table matches the executed BCH reference.

</details>

<!-- exercise: construction -->

**Expected:** full generator and delay matrices match the pinned author fixtures at n=5, 28 and 511. Every five-feature message encodes into a polynomial divisible by the generator.

<!-- solution: construction -->

<!-- code: code_example -->

### Exercise - Build the complete query design

> **Difficulty:** 🔴🔴⚪⚪⚪ | **Importance:** 🔵🔵🔵🔵🔵 | **Time:** 15 minutes

Implement `make_query_design` and `transform_query_values`. Keep duplicate requests in the design, and count distinct masks separately. Use a local `RandomState(seed)` so the chosen design is reproducible without changing anyone else's RNG state.

<details><summary>Help - Flattening is part of the contract</summary>

The oracle receives `(Q,n)` rows in group, delay, small-mask order. Reshape its returned values to `(C,p+1,2**b)` and transform the last axis. Query count is $C(p+1)2^b$, whether or not some rows repeat.

</details>

<!-- exercise: query_design -->

**Expected:** n=28,t=3,seed=0 produces requested/distinct counts 456/418, 912/874 and 1824/1786 at b=3,4,5. The transformed bins match the independent signed-sum identity above.

<!-- solution: query_design -->

<!-- code: design_receipt -->

## Treat the unknown support as an error pattern

We do not observe the message $k$. We observe parity estimates and form the received word

$$r=\begin{bmatrix}0\\Pk+e\end{bmatrix}.$$

Its difference from the valid codeword is ([k;e]). Therefore hard decoding is guaranteed when

$$|k|+|e|\le t.$$

The support itself consumes error-correction capacity. A degree-three interaction at t=3 leaves no guaranteed allowance for a corrupted parity sign.

## Syndromes and an error locator

Treat the received bits as polynomial coefficients, highest power first. Evaluate that polynomial at $\alpha^j$, for $j=1,\ldots,2t$, to get the syndromes. A valid codeword has zero syndromes; a corrupted codeword does not.

The Berlekamp–Massey algorithm finds the shortest recurrence explaining the syndrome sequence. At step r it computes a discrepancy

$$\delta_r=S_r+\sum_{i=1}^{L}C_iS_{r-i}$$

in the finite field. A nonzero discrepancy updates the recurrence with a shifted, rescaled previous recurrence. Its final coefficients describe the **error-locator polynomial**, in ascending powers.

### Exercise - Compute syndromes and the locator recurrence

> **Difficulty:** 🔴🔴🔴🔴⚪ | **Importance:** 🔵🔵🔵🔵🔵 | **Time:** 30 minutes

Implement `bch_syndromes` and `berlekamp_massey`. Keep the polynomial-order conventions explicit. Removed leading message positions are zero, so the shortened received word still has a direct polynomial interpretation.

<details><summary>Help - Track four pieces of state</summary>

Keep the current locator `C`, a previous locator `B`, the current degree, and the number of steps since the last degree update. If discrepancy is zero, increase that gap. Otherwise add a shifted copy of B scaled by the ratio of discrepancies. When `2 * degree <= r`, update B and the degree as well. In characteristic two, subtraction is XOR too.

</details>

<!-- exercise: syndromes -->

**Expected:** syndromes match independent polynomial evaluation. For known error positions, the locator equals the independently multiplied factors $\prod_i(1+\alpha^{q_i}z)$.

<!-- solution: syndromes -->

### Exercise - Find roots and recover feature positions

> **Difficulty:** 🔴🔴🔴⚪⚪ | **Importance:** 🔵🔵🔵🔵🔵 | **Time:** 25 minutes

Implement `chien_search` and `decode_bch_parity`. Search the nc code positions, not all candidate supports. Reject an incorrect number of roots, roots in removed positions, or a corrected word outside the systematic code.

<details><summary>Help - Which power should you try?</summary>

An error at a coordinate whose polynomial power is q contributes a locator root at $\alpha^{-q}$. Search all nc coordinates and convert successful roots to shortened-word positions. Correct those bits, verify the codeword, then read its first n bits as the feature support.

</details>

<!-- exercise: hard_decode -->

**Expected:** all 92 n=5,t=2 cases with support weight plus parity errors at most t decode correctly. Additional tests cross field boundaries through n=511,t=5.

<!-- solution: hard_decode -->

<!-- code: decode_example -->

## Soft information: which parity signs should we distrust?

A shift ratio near zero gives weak evidence for either sign. Chase decoding tries a small collection of flips among the least confident parity signs, then hard-decodes each resulting word.

The default here matches the author's bounded search: with depth d it considers at most $d(d+1)/2$ low-cost patterns. The wrapper uses d=2t. It returns a successful hard decode immediately; otherwise it chooses a successful trial with the fewest corrected bits. Hash-bin validation comes afterward.

### Exercise - Implement the bounded Chase search

> **Difficulty:** 🔴🔴🔴⚪⚪ | **Importance:** 🔵🔵🔵🔵⚪ | **Time:** 25 minutes

Implement `chase_patterns` and `soft_decode_parity`. Keep the default bounded ordering and the explicitly labelled `expanded` alternative distinct. Neither routine enumerates feature supports.

<details><summary>Help - Why keep trial order?</summary>

Two perturbations can have the same cost, and two decoded messages can have the same correction count. Stable ordering makes that ambiguity reproducible. The default heap keeps at most $d(d+1)/2$ partial patterns; the expanded play variant lists all $2^d$ patterns of the selected parity positions.

</details>

<!-- exercise: chase -->

**Expected:** hard decoding fails for a degree-three support with one corrupted parity sign. A correctly low confidence on that sign lets the bounded soft decoder recover `(1,3,7)`.

<!-- solution: chase -->

## Try It Yourself - Change the confidence

Predict what happens when the bad sign is assigned confidence 10 instead of .01. You can also change its position or increase the support degree. Gray rows in the result mean explicit decoding failure; a wrong decoded support is a different outcome.

<!-- code: confidence_play -->

# 4️⃣ Subtract collisions and check the explanation

## A decoded support is only a candidate

For a candidate support $S$, construct its full shift signature $s_S$, including the zero shift. Estimate its signed amplitude by

$$\widehat F(S)=\frac{s_S^\top U(j)}{p+1}.$$

Then check both $M^\top1_S=j$ and the fraction of residual energy

$$\frac{\|U(j)-\widehat F(S)s_S\|_2^2}{\|U(j)\|_2^2}<\gamma.$$

The author default is $\gamma=.9$. This permissive threshold can accept a dominant term in a collision; it is not a proof that the bin contained exactly one term. A false decoded support can still pass, which is why we measure false positives against analytic truth.

### Exercise - Check a signed bin candidate

> **Difficulty:** 🔴🔴⚪⚪⚪ | **Importance:** 🔵🔵🔵🔵🔵 | **Time:** 20 minutes

Implement `signed_bin_check` and `decode_bin`. Keep negative amplitudes negative. Reject an explicit BCH failure, a mismatched bin, and a near-zero zero-shift denominator.

<details><summary>Help - Why not use absolute correlation?</summary>

The fitted amplitude is signed. The residual checks how well that signed component explains the observations. Taking its magnitude and subtracting a positive component would double the error for a negative coefficient. The shift ratio removes the sign for decoding; the amplitude estimate restores it for reconstruction.

</details>

<!-- exercise: signed_bin -->

**Expected:** ±1.25 singletons decode with the correct sign. The same support is rejected in the wrong bin; near-zero reference-shift cancellation is rejected explicitly.

<!-- solution: signed_bin -->

## Iterative messages

One sweep reads all active bins in the current residual state. Bins send accepted signed amplitudes to candidate supports. A support averages the messages it receives, adds that amount to its coefficient estimate, and subtracts the same signed signature from every bin it hashes to. Changed bins become active again.

This is how information from one hash group can resolve a collision in another. Revisit a support if a later sweep finds a correction to its coefficient. A stop reason describes why computation ended; it does not certify exact recovery.

### Exercise - Implement iterative collision subtraction

> **Difficulty:** 🔴🔴🔴🔴⚪ | **Importance:** 🔵🔵🔵🔵🔵 | **Time:** 35 minutes

Implement `recover_spectrum`. Record each sweep's accepted events, signed updates, changed bins and energies. Keep factor detection separate from subtraction so every detection in a sweep sees the same residual state.

<details><summary>Help - Why average repeated detections?</summary>

If three bins all estimate the same coefficient as -1.25, their combined estimate is -1.25, not -3.75. Average their messages, update the spectrum once, then subtract from all connected bins. This matches the author's `peel_average=True` behavior.

</details>

<!-- exercise: peeling -->

**Expected:** every small signed basis term is recovered. The polynomial example needs several sweeps and recovers all 13 supports with audit RMSE below $10^{-4}$. A one-sweep cap leaves a residual; dense and wrong-degree controls fail.

<!-- solution: peeling -->

<!-- code: methods_namespace -->

## Signature result - Watch a collision disappear

The analytic oracle is

$$f(x)=.3+x_0-.7x_3+1.2x_1x_7-2.4x_2x_4x_9.$$

Expanding $x_i=(1-\chi_{\{i\}})/2$ gives 13 signed Fourier terms. The displayed occupancy is computed from that independent truth, not inferred from the decoder. Follow the selected bin across sweeps, then compare every recovered sign with truth. The horizontal no-subtraction control shows what the residual would do if we decoded candidates but never sent messages.

<!-- code: signature -->

## Try It Yourself - Change the hidden interaction

Add a negative term, move its support, change the hash seed, or reduce `PLAY_T` below the true interaction degree. Before running, predict whether the issue is a collision, a coding limit, or insufficient sparsity. The play reports every requested analytic mask; it makes no model calls.

<!-- code: support_play -->

## Controls and baselines - More queries do not fix every misspecification

We now keep seeds 0,1,2 and budgets b=3,4,5 fixed. Each design is independently drawn for its b; these are not nested query prefixes. Predictions are checked on 4,096 independent masks. The audit does not choose decoder settings.

- **Sparse polynomial:** 13 exact signed terms; the intended success case.
- **Additive:** a constant plus six signed singleton terms; a simpler success control.
- **Dense:** 256 signed degree-three terms with total squared coefficient energy one; the available bins are insufficient.
- **Wrong degree:** a degree-three parity with t=1; its coding assumption is false.
- **Shuffled:** the sparse game's labels are permuted across distinct training masks, with duplicates kept consistent. The original game remains the audit target.

The constant and additive predictors use exactly the same training queries. Normalized RMSE divides by the true audit-score standard deviation, so differently scaled games can be compared. Support recall alone is insufficient: the next panel counts false interactions too.

<!-- code: analytic_comparison -->

<details><summary>Interpret the curves before continuing</summary>

Find one setting with low error, high support recall and no false positives. Then find a dense or shuffled setting where a nonempty spectrum is still a poor explanation. Finally compare the raw and unique query counts. Cached repeated masks can reduce physical inference, but neither count is a model-forward timing measurement.

</details>

# 5️⃣ Connect the spectrum to attribution and a word-mask game

## Fourier coefficients are not Shapley interactions

For a nonempty target set $T$, the Shapley interaction index is

$$I_{\mathrm{SII}}(T)=(-2)^{|T|}\sum_{\substack{S\supseteq T\\|S|-|T|\;\mathrm{even}}}\frac{F(S)}{|S|-|T|+1}.$$

Singleton SII gives the Shapley value. In particular,

$$\phi_i=-2\sum_{\substack{S\ni i\\|S|\;\mathrm{odd}}}\frac{F(S)}{|S|}.$$

This convention uses binary 1 to retain a feature. Do not replace $F(S)$ by its magnitude or label it SII before conversion. These equations define SII, not Shapley–Taylor interactions.

### Exercise - Convert the signed spectrum

> **Difficulty:** 🔴🔴⚪⚪⚪ | **Importance:** 🔵🔵🔵🔵🔵 | **Time:** 15 minutes

Implement `fourier_to_sii` and `fourier_to_shapley`, evaluating only requested target sets. You need not enumerate the power set of a large support universe.

<details><summary>Help - A quick sign check</summary>

For $F(\{1,3,7\})=-1.25$, each of those three features has Shapley value $+2.5/3$, the pair SII is zero, and the triple SII is +10. The Shapley values sum to $f(1)-f(0)=2.5$.

</details>

<!-- exercise: attribution -->

**Expected:** every nonempty interaction of every basis function through five features agrees with the independent factorial-weighted finite-difference definition. Shapley efficiency holds.

<!-- solution: attribution -->

<!-- code: attribution_figure -->

The two games have the same Shapley values and different interactions. Per-feature attribution cannot, by itself, tell you how those features work together.

## Define the native intervention before asking the model

The shared prospective study uses 28 word features in GPT-2 prompts. A word can span multiple subtokens. A dropped word must hide **all** its subtokens. The suffix remains available at a fixed absolute position. We replace hidden token IDs and set their attention-mask entries to zero while leaving position IDs unchanged.

This defines a particular fixed-position intervention. It is not natural deletion, and the resulting masked text need not resemble normal language. Native approximation quality will therefore describe this specified mask game.

### Exercise - Mask whole words while keeping positions fixed

> **Difficulty:** 🔴🔴⚪⚪⚪ | **Importance:** 🔵🔵🔵🔵🔵 | **Time:** 15 minutes

Implement `masked_word_inputs`. A token-to-feature map of `-1` denotes the always-retained suffix. Return input IDs, the attention mask and explicit absolute position IDs for every coalition.

<details><summary>Help - Do not mask just the first subtoken</summary>

Index the feature mask by the token-to-feature map for every variable token. If a word maps to two tokens, both receive the same retain bit. Build positions from the original token sequence length and broadcast them unchanged to every row.

</details>

<!-- exercise: word_masking -->

**Expected:** the two tokens belonging to the first toy word are hidden together, the suffix stays visible, and every mask keeps positions `[0,1,2,3,4]`.

<!-- solution: word_masking -->

<!-- code: word_masking_example -->

## Native comparison - Pending shared execution

The code below passes your completed method namespace to the shared study. That study also needs the completed ProxySPEX namespace from 16.19, so the two estimators use the same declared prompts, budgets, audit masks and controls. It must bind the model revision and all query settings before native outcomes.

This draft leaves `RUN_NATIVE_STUDY=False`. The analytic figures above do not establish GPT-2 fidelity. When
the shared run is executed, inspect actual model scores beside SPEX predictions, word-labelled signed terms,
query-matched constant/additive baselines and shuffled-response failures. Keep incorrect unmasked answers
visible: approximating an incorrect answer accurately is still approximation, not semantic correctness.

<!-- code: native_study -->

## Try It Yourself - Inspect a completed native prompt

After a completed shared study, change the prompt, seed or an existing budget below. This play reuses the recorded predictions and makes zero additional model evaluations. Looking at its audit outcomes is exploratory; it does not change the fixed study's selected settings.

<!-- code: native_play -->

## Limitations and source checks

SPEX needs a spectrum that is sufficiently sparse for its bins and sufficiently low-degree for its code. Dense, shuffled and wrong-degree controls show distinct ways those assumptions can fail. A successful BCH decode is still a candidate; even the hash and residual checks do not prove that a returned interaction is real.

The course uses the actual systematic code and bounded Chase mechanism. Seventy full matrix fixtures, 132
guaranteed-radius decode cases and three ordered 28-feature designs agree with the executed author reference.
Of 26 complete analytic reference runs, 23 match within $10^{-8}$. Two polynomial cases differ by at most
$3.94\times10^{-5}$, because of near-zero denominator guards and floating-point Chase ties. A dense case
differs because the author wrapper accepts a candidate after the BCH decoder reports failure; this lesson
rejects that explicit failure. The traces remain available rather than being described as exact output parity.

Two paper/code details matter for interpretation: the displayed Shapley expression in Table 1 omits the -2
factor present in Appendix C equation (46), and the paper's message formula shows a sum where the prose and
executed code use an average. The author wrapper requests Lasso but returns before its regression branch under
its default setting; neither these exercises nor that executed wrapper include final Lasso refinement.

## Further research - Find an assumption that breaks

Try constructing two supports that collide in every hash group. Can extra shifts distinguish them if the hash identity still puts them together? Try an equal-and-opposite collision with zero reference shift. Then compare the bounded and expanded Chase variants on a dense game: does trying more candidates reduce error, or merely create more false positives?

Carry those questions into 16.19. A tree proxy offers a different route to a spectrum, but its prediction error, spectrum extraction and coefficient selection must be checked separately.

## Reading links

- [SPEX: Scaling Feature Interaction Explanations for LLMs — ICML 2025](https://proceedings.mlr.press/v267/kang25a.html), especially Appendix A.3–A.4 and Appendix C.
- [Pinned author implementation](https://github.com/basics-lab/spectral-explain/tree/61400b5388b8ecbf18fe18c4d821ffe3e8a5f9c8).
- [ProxySPEX — NeurIPS 2025](https://papers.neurips.cc/paper_files/paper/2025/hash/68dced1638d71d4d21598f79ae91262e-Abstract-Conference.html).
- Repository reference evidence: `docs/arena_salvage/2026-09-08_spex_reference_execution.md`. Analytic evidence supports these exercises; it does not close the shared native requirement.

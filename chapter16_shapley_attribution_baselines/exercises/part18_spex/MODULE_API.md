# SPEX core: learner API and analytic boundary

This is development for the already-selected SPEX/ProxySPEX requirement. These
files do not publish a lesson, run a native model, or mark the requirement taught.
The root plan is
[`2026-09-08_spex_proxyspex_learner_plan.md`](../../../docs/arena_salvage/2026-09-08_spex_proxyspex_learner_plan.md).
Original ARENA files and existing section 16.17 are outside this change.

The existing learner failure is that no visible sparse Fourier decoder is taught.
The bounded claim here is that low-degree sparse signed spectra can be recovered
from shifted Boolean subtransforms. The intended signature is a collision becoming
a decoded signed interaction, followed by a visible recovery trace. Tests and
trace dictionaries support that result; the eventual notebook must draw it.

Student implementations belong in the notebook namespace. `spex_methods.py`
provides inspectable reference bodies for those cells; importing and calling the
entire module is not an acceptable replacement for implementing the method. Only
the finite-field lookup arithmetic in `spex_coding.py` is supplied as plumbing.
The notebook builder should expose these groups with immediate tests:

1. `boolean_character`, `normalized_fwht`, `predict_spectrum`.
2. `shifted_subsample_queries`, `transform_query_values`, query accounting.
3. `bch_generator_polynomial`, `make_bch_code`, `bch_encode`, `bch_syndromes`.
4. `berlekamp_massey`, `chien_search`, `decode_bch_parity`.
5. `chase_patterns`, `soft_decode_parity`, `signed_bin_check`, `decode_bin`.
6. `recover_spectrum`: decode factors, average messages, subtract collisions.
7. `fourier_to_sii`, `fourier_to_shapley`.

The public test functions accept replacement student callables for these groups.
Dataclasses, NumPy, `heapq`, `lru_cache`, typing imports, the binary-array input
validator and `BinaryField`/`binary_field` can be supplied in setup. All dependency
method functions must still be in the same visible notebook namespace. The
expanded Chase variant, editable support signs/collisions, and degree/budget
changes are prospective play cells; no play notebook has been authored here.

## Mask and spectrum convention

Every mask array has shape `(Q, n)` and values zero or one. Feature 0 is the first
column. Integer mask IDs use big-endian bits, so `[1,0,0]` is 4. `mask_ids` uses
Python integers and works beyond 64 features. Supports are sorted
`tuple[int, ...]` values containing feature positions, not binary indicator
tuples. The empty tuple is the constant.

The basis is `chi_S(x) = (-1)**sum(x[i] for i in S)`. A spectrum is
`dict[tuple[int, ...], float]`, retaining coefficient signs. A value of one has
the caller's retain-feature meaning; the core does not reinterpret masks.
`predict_spectrum(masks, spectrum)` returns one real value per mask.

`normalized_fwht` divides by the axis length. To invert its output, apply it
again and multiply by that length. Only the small b-dimensional query subcube
is enumerated. `binary_masks` refuses dimensions above 16; the sampler limits
`b <= 12`. Nothing constructs a `2**n` model-value table.

## Query and recovery API

```python
design = make_query_design(n=28, t=3, b=3, C=3, seed=0)
values = oracle(design.masks)  # caller controls physical evaluations and caching
result = recover_spectrum(design, values)
prediction = predict_spectrum(audit_masks, result.spectrum)
```

`QueryDesign` contains `n, t, b, C, seed, code, matrices, delays, masks,
requested_count, unique_count`. Matrices have shape `(C,n,b)` and delays have
shape `(p+1,n)`. Mask rows are ordered by group, then delay, then big-endian
subcube mask. Every requested row, including duplicates, remains in `masks`.
Supply exactly one finite scalar per requested row. Precomputed values are
fully supported; recovery itself never calls an oracle.

Each b design resets a local `np.random.RandomState(seed)` and draws the three
matrices consecutively. Different b budgets are independently drawn designs,
not nested prefix samples. The local RNG avoids altering the caller's global
random state while reproducing the author design for an independently reset b.

For n=28,t=3, the full code is BCH(63,45), shortened by 17 message positions;
there are 18 parity rows plus the zero shift. At seed 0, b=3/4/5 gives
requested/unique counts 456/418, 912/874, and 1824/1786. These match the actual
author masks, including order. A physical cache can reduce inference work;
neither requested nor unique counts imply a model-forward count. The optional
`sample_and_recover(design, oracle, **kwargs)` makes one batch request for all
rows and keeps the same accounting.

`RecoveryResult` contains `spectrum`, `trace`, requested/unique counts,
`initial_bins`, `residual_bins`, total `residual_energy`, and `stop_reason`.
Each trace sweep records energies, accepted factor events, rejected/zero bin
counts, signed coefficient updates, all touched bins, and the spectrum after
the sweep. These snapshots make the collision plot and intermediate predictions
reconstructible. Trace spectrum dictionaries have tuple keys; convert them to
records before writing JSON.

The current numerical defaults, selected from the analytic reference before
native outcomes, are:

| Parameter | Default | Origin |
| --- | --- | --- |
| `max_iterations` | 100 | author transform cap |
| `chase_depth` | `None`, resolved to `2*t` | author wrapper; six at t=3 |
| `chase_variant` | `"author"` | bounded source selection described below |
| `residual_fraction` | 0.9 | author's residual-energy acceptance threshold |
| `bin_energy_floor` | 1e-9 | author's noise_sd=0 mean-bin-energy cutoff |
| `stagnation_window` | 15 | explicit support-stagnation cap, same scale as source |
| `min_amplitude` | 1e-8 | numerical guard on reference-shift division and final coefficients |

The final guard is an explicit course choice. The reference comparisons already
counted support at 1e-8. An iteration cap or undecodable residual is a stop reason,
not evidence that the true spectrum has been recovered. No final Lasso or Ridge
fit is performed.

## BCH construction and decoding

`make_bch_code(n,t)` supports every integer n from 1 through 511 and t from 1
through 5, without combinatorial support search. It closes roots alpha^1 through
alpha^(2t) under binary Frobenius conjugacy and multiplies their minimal factors.
It constructs the systematic row generator `[I | P_full.T]`, then takes the
**last n message rows** and transposes their parity part. The column-vector
paper convention is `[I; P]`.

The field polynomials are the actual `galois.BCH==0.4.11` defaults. They differ
from plain `galois.GF(2**m)` defaults at m=6,7,10. The supplied binary polynomial
integers for m=2,...,10 are 7,11,19,37,67,137,285,529,1033; primitive element is x.

Hard decoding forms `[zero message, parity observations]`, evaluates 2t
syndromes, constructs the error locator with Berlekamp–Massey, and runs Chien
root search over the nc code positions. It rejects missing roots, roots in
removed leading message positions, or a corrected word outside the systematic
code. `BCHDecode` reports success, support, corrected positions (relative to the
shortened word), syndromes, locator and a reason. The guarantee is
`support_weight + parity_error_weight <= t`. Beyond it a decoder can fail or
return a different codeword. This is why hash-bin and residual checks remain
necessary. Some BCH parameter searches yield a code with a stronger actual
distance than requested; the course syndrome decoder explicitly uses the
requested t bound.

The default soft path returns a successful hard decode immediately. Otherwise
it considers at most `d*(d+1)//2` low-cost perturbations of the d least confident
parity signs, using the author's bounded partial-sum heap and ordering. Unique
successful messages are ranked by corrected-bit count, with stable trial-order
ties. Bin and residual validation happen **after** that selection.

`chase_variant="expanded"` is a labelled alternative: it enumerates at most
`2**d` parity perturbations and takes the first confidence-ordered candidate
passing the decoder and bin checks. It is not the author default. Historical
analytic exploratory runs with d=6, including false positives on the dense and
wrong-degree controls, are retained under
`/home/bitwise/.cache/arena_extended/sparse_interaction_discovery/analytic_core_v1/`.

## Paper, source and course distinctions

The primary sources are the [SPEX paper](https://proceedings.mlr.press/v267/kang25a.html)
and [pinned author code](https://github.com/basics-lab/spectral-explain/tree/61400b5388b8ecbf18fe18c4d821ffe3e8a5f9c8).
The [executed reference report](../../../docs/arena_salvage/2026-09-08_spex_reference_execution.md)
records source identities and unmodified failures.

- Table 1's Shapley expression omits -2. The course conversion follows Appendix C
  equation (46), verified against independent factorial-weighted finite differences.
- The paper's code dimension/shortening prose is ambiguous. The implementation
  uses designed distance 2t+1 and the last n systematic message rows, matching
  the actual generator matrices.
- The signed amplitude is `signature @ observation / len(signature)`, including
  the zero-shift row. Acceptance uses a hash-bin check and residual-energy
  fraction. The paper's displayed positive-correlation condition alone would
  mishandle negative amplitudes.
- The paper's Algorithm 4 shows summation; its prose and the author
  `peel_average=True` use mean messages. The course uses their mean.
- The paper describes Chase trials generally. The author's actual wrapper uses
  `parity_decode_2chase_t2_max_likelihood`, with the bounded partial-sum list and
  correction-count choice described above; it does not search all 64 patterns.
- The author's wrapper asks for Lasso, but the transform returns before its
  regression branch when `report=False`. The executed wrapper makes zero
  regression calls. The course likewise does not claim final refinement.
- The course keeps Boolean characters exactly real, guards division when
  `abs(U0)<=1e-8`, and rejects explicit BCH failure. These are disclosed
  differences, not claims of exact source-equivalent output. In the saved
  n=31 sparse-polynomial seed=1,b=3 run, the author decodes three initial bins
  with U0 magnitudes between 5.55e-17 and 1.53e-16; the course guard skips them.
  The course takes seven sweeps versus six and its largest terminal coefficient
  difference is 3.94e-5. At b=4 the largest difference is 7.29e-6.
- In the dense-spectrum seed=1,b=3 run, the author accepts group 2/bin 0's
  constant coefficient +0.1228906058 even though BCH reports `errors=-1`.
  The course rejects that failure. The other four recovered supports agree.
  Instrumented, unmodified author runs reproduce these saved results exactly;
  their accepted observations and ratios are retained in
  `reference_execution_v1/artifacts/author_spex_discrepancy_traces.json`.

## Analytic checks and remaining work

Run from the repository's existing pinned environment:

```bash
.venv/bin/python -m pytest chapter16_shapley_attribution_baselines/exercises/part18_spex/spex_methods_tests.py -q
```

The eleven focused tests cover all 62 basis functions through five features,
all 92 n=5,t=2 bounded error cases, additional field-boundary cases through
n=511,t=5, exact author G/D hashes, query identities/counts, negative amplitudes,
Chase rescue, every small-basis SII finite difference, collision traces, finite
iteration budgets, and failing wrong-degree/dense controls. No native model or
isolated-reference dependency is imported.

Independent reference receipts additionally compare 70 complete G/D fixtures,
132 guaranteed-radius decoder cases and three exact ordered n=28 query designs.
All pass. The receipts and source snapshots are in
`reference_execution_v1/artifacts/course_bch_independent_parity.json` beneath
the sparse-interaction cache. Across 26 saved author recovery runs, 23 agree
within 1e-8. The two polynomial numerical-guard differences and one dense
failed-decode difference above remain visible in
`reference_execution_v1/artifacts/course_spex_independent_parity.json`. These
are separate estimator receipts, not part of the exact BCH parity statement.

Remaining root-owned work includes the visible learner cells/pages, native
protocol and actual native outcomes, source-bound reproduction, rendered plots,
play-cell execution, and an independent learner review. No original ARENA
preservation audit or course-release claim is made by this scoped core change.

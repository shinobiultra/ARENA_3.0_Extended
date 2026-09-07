# Reference threshold finding after the first native run

Recorded 2026-09-06, after the frozen experiment's first fits. The original
protocol, data manifest and data loader remain byte-unchanged. This addendum
does not change the trained models, data, backgrounds, selected rows, missing
feature games or exact-oracle tolerance.

The original expectation that unmodified SHAP 0.49.1 interventional output would
meet the stated tolerance was falsified. Its C++ `dense_independent` conversion
stores a double-precision tree threshold in the single-precision `Node.thres`.
Scikit-learn instead compares a float32-rounded input to the original float64
threshold. Rounding a threshold upward can therefore change the represented
model at an exactly representable input. The library's default additivity
check is too permissive to establish the stricter contract used here.

An observed example is the first clean forest's component tree 31, node 32,
feature 4: threshold 10.099999904632568 becomes10.100000381469727, equal to the
float32 representation of input 10.1. The original split takes the right branch;
the rounded split takes the left. Explanation rows 7/9 (source rows 139/162) show
about 0.4902 MPa error in base plus summed contributions. The unmodified
path-dependent implementation agrees with the original prediction.

Keep the unmodified reference arrays, exact errors and failed parity flag.
Do not increase the tolerance or describe this reference as passing. A second,
explicitly corrected compiled reference replaces every threshold in a separate
explainer with the greatest float32 value less than or equal to the original
threshold. This is a numerical compatibility correction, not an adjustment
of individual examples or attributions.

For any finite original threshold t and any float32-representable query q,
q <= t if and only if q <= floor_float32(t). Thus this transformation preserves
the sklearn partition on its actual input domain. The fitted sklearn model
and the student implementation keep their original thresholds. Record every
changed threshold and verify its lower/upper float32 neighbors. The corrected
compiled reference must meet the original strict tolerance and must also agree
with exhaustive subset values. Remaining float32 value arithmetic is still
covered by the originally stated compiled-reference tolerance.

The learner sees both results. Exactness refers to the declared game and
numerical prediction semantics; agreement with a popular library is supporting
evidence, and a failed library comparison is a scientific result to investigate.

## Full-study diagnosis

The complete independent check retains 4,896 model-row combinations across all
18 models and both held-out splits, with 39,168 local coordinates per game.
Of 5,480 internal split thresholds, 941 round upward. Over all 1,030 original
data rows, the raw rounded thresholds create 217 split-routing disagreements;
the lattice-floor correction creates zero. These are node/row comparisons,
not 217 distinct mispredicted examples.

Raw vector parity fails for 469 model-row combinations; 18 model-row prediction
sums exceed the stated tolerance. The maximum raw local-coordinate discrepancy
is approximately 0.376032 MPa; the largest base-plus-contributions discrepancy
is approximately 0.616826 MPa. The two-row example above is one identified case,
not the total affected count.

The corrected compiled reference's maximum local-coordinate discrepancy is
approximately 2.825e-6 MPa and its maximum prediction-sum discrepancy 3.142e-6 MPa,
within the original compiled tolerance. Independently enumerating the game of
the raw rounded-threshold trees on all 816 primary-family model-row combinations
matches the raw compiled vectors within 1.456e-6 MPa. That diagnoses the full
allocation, including background effects, rather than checking sums alone.

The student's float64 dynamic program separately agrees with exhaustive values
of the actual sklearn game to about 2.49e-14 MPa on the audit split. Compiled
reference precision and float64 oracle precision are distinct comparisons.

# Priority pilot interpretation repairs — 2026-10-16

Current learner failure: 12.1 attributes differences between pretrained checkpoints
to loss alone and reads an uncalibrated sigmoid score as an empirical match chance.
9.1 explains its uncentered SVD as if random row signs changed its recovered axis.

Target claims: the controlled tiny experiments and the pretrained comparison answer
different questions; pairwise model scores require calibration before frequency
claims. For the actual uncentered reading-vector implementation, row signs leave
the second-moment matrix unchanged, unlike mean-centering a finite sample.

Signature results, controls, baselines and play cells stay as published. All student
methods and scientific code cells remain unchanged. The worked algebra and score
interpretation are corrected in prose, with existing expected outputs preserved.

Files: 12.1 and 9.1 canonical masters and generated notebook/pages; exact preceding
versions in `../evidence/pilot_prose_repairs_20261016/published_before_prose.tar.gz`.
Validation: master generation without execution, complete code-cell/output/count
hash equivalence, original-ARENA preservation, a tiny numerical Gram/sign check and
independent interpretation review. No model rerun, downloads or new scientific
acceptance are claimed. The original historical artifacts remain recoverable.

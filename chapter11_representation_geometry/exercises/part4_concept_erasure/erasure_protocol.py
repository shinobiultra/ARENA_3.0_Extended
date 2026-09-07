"""Frozen evaluation recipe, after the retained development-only study.

All prompt splits were written before the first native run. The two sites and
both covariance settings are retained; none is selected on final behavior.
"""
SITES = ((6, "subject"), (9, "end"))
SHRINKAGE = 0.01
PROBE_L2 = 0.001
PROBE_STEPS = 150
SVD_ATOL = 1e-10
RANDOM_DRAWS = 16
RANDOM_SEED_STARTS = (1000, 2000)
FIXED_EXAMPLES_PER_TEMPLATE = 1
FINAL_ROLE_ROWS = 288
FINAL_OBJECT_ROWS = 384
CONDITIONS_PER_SITE = 5 + 3 * RANDOM_DRAWS


def condition_names():
    base = ["No intervention", "Probe coefficient", "Orthogonal covariance", "LEACE empirical", "LEACE shrinkage .01"]
    return base + [f"{family}/draw{draw:02d}" for family in ("Random rank one", "Matched empirical", "Matched shrinkage")
                   for draw in range(RANDOM_DRAWS)]

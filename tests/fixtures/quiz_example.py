"""A made-up lesson's quiz key, used to test the shared scorer end to end.

This is what a real lesson declares: its questions and answers, nothing else.
"""

from quiz import Interval, Quiz, Ranking, Word  # ty: ignore[import-not-found]

# Image-to-text R@1 of four runs; runs within 0.01 of each other tie.
RUNS = {"A": 0.96, "B": 0.97, "C": 0.58, "D": 0.01}

quiz = Quiz(
    {
        "Q1": Word("falls", points=1, choices=("falls", "rises", "unchanged")),
        "Q2": Interval("c", points=2),
        "Q3": Ranking.from_values(RUNS, eps=0.02),
    }
)

# One learner's answers: Q1 right, Q2 one interval off, Q3 gets one of six pairs wrong.
EXAMPLE_ANSWERS = {"Q1": "falls", "Q2": "d", "Q3": "D < C < A < B"}

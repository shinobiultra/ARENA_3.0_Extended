"""Score Held-out Quiz answers in the learner's own notebook.

Every extension lesson and Alchemy companion ends its core with a quiz. The lesson declares only its
key; this file does the scoring. Nothing is saved or sent anywhere: ``score`` just prints a table.

    from quiz import Interval, Quiz, Ranking, Word

    quiz = Quiz({
        "Q1": Word("falls", points=1),
        "Q2": Interval("c", points=2),                                # five intervals, a to e
        "Q3": Ranking.from_values({"A": 0.96, "B": 0.97, "C": 0.58}, eps=0.02),
    })
    my_answers = {"Q1": "falls", "Q2": "d", "Q3": "C < A = B"}
    quiz.score(my_answers)

Scoring: the right interval earns full points and a neighbouring one half. A ranking earns one
point per pair of runs whose order you got right, where ``=`` means "within the stated eps".
A word earns full points or nothing. An answer that can't be read scores zero, and the table says why.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, pairwise

OPTIONS = "abcde"


@dataclass(frozen=True)
class Result:
    """What one question earned."""

    question: str
    answer: str
    key: str
    points: float
    out_of: float
    note: str = ""  # why an answer could not be scored


def _text(answer: object) -> str | None:
    return answer.strip() if isinstance(answer, str) else None


@dataclass(frozen=True)
class Interval:
    """One of five exhaustive intervals (a to e). A neighbouring interval earns half."""

    key: str
    points: float

    def __post_init__(self):
        if self.key not in OPTIONS:
            raise ValueError(f"key must be one of {', '.join(OPTIONS)}, not {self.key!r}")

    @property
    def out_of(self) -> float:
        return self.points

    @property
    def key_text(self) -> str:
        return self.key

    def grade(self, answer: str) -> tuple[float, str]:
        answer = answer.lower()
        if answer not in OPTIONS:
            return 0, "answer a, b, c, d or e"
        gap = abs(OPTIONS.index(answer) - OPTIONS.index(self.key))
        return self.points * {0: 1, 1: 0.5}.get(gap, 0), ""


@dataclass(frozen=True)
class Word:
    """A one-word answer. ``choices``, when given, are the words a learner may pick from."""

    key: str
    points: float
    choices: tuple[str, ...] | None = None

    @property
    def out_of(self) -> float:
        return self.points

    @property
    def key_text(self) -> str:
        return self.key

    def grade(self, answer: str) -> tuple[float, str]:
        answer = answer.lower()
        if self.choices and answer not in (c.lower() for c in self.choices):
            return 0, "answer one of: " + ", ".join(self.choices)
        return (self.points if answer == self.key.lower() else 0), ""


class Ranking:
    """An ordering of runs written ``D < C < A = B``. One point per correctly ordered pair.

    ``Ranking("D < C < A = B")`` fixes the key by hand. ``Ranking.from_values(runs, eps)`` builds it from
    the key runs' numbers: two runs within ``eps`` of each other count as tied (``=``).
    """

    def __init__(self, key: str):
        ranks = _parse_ranking(key)
        if isinstance(ranks, str) or len(ranks) < 2:
            raise ValueError(f"a ranking needs at least two runs: {key!r}")
        self._labels = set(ranks)
        self._order = {pair: _compare(ranks[pair[0]], ranks[pair[1]]) for pair in _pairs(ranks)}
        self.key_text = " ".join(key.upper().split())

    @classmethod
    def from_values(cls, values: dict[str, float], eps: float) -> Ranking:
        """Order runs by value, ascending. Runs closer than or equal to ``eps`` tie."""
        if len(values) < 2:
            raise ValueError("a ranking needs at least two runs")
        runs = sorted(values.items(), key=lambda item: item[1])
        text = runs[0][0]
        for (_, previous), (label, value) in pairwise(runs):
            text += (" = " if value - previous <= eps else " < ") + label
        ranking = cls(text)
        # pairs are tied or ordered by their own values, not by neighbours, so a chain a~b~c can't tie a with c
        ranking._order = {(x, y): _compare(values[x], values[y], eps) for x, y in _pairs(values)}
        return ranking

    @property
    def out_of(self) -> float:
        return len(self._order)

    def grade(self, answer: str) -> tuple[float, str]:
        ranks = _parse_ranking(answer)
        if isinstance(ranks, str):
            return 0, ranks
        if set(ranks) != self._labels:
            return 0, "rank exactly these runs: " + ", ".join(sorted(self._labels))
        right = sum(_compare(ranks[x], ranks[y]) == truth for (x, y), truth in self._order.items())
        return right, ""


def _pairs(runs):
    return combinations(sorted(runs), 2)


def _compare(a: float, b: float, eps: float = 0) -> int:
    """-1 if a is below b, 0 if they tie (within eps), +1 if a is above b."""
    return 0 if abs(a - b) <= eps else (a > b) - (a < b)


def _parse_ranking(text: str) -> dict[str, int] | str:
    """'D < C < A = B' -> {'D': 0, 'C': 1, 'A': 2, 'B': 2}; a message when it can't be read."""
    ranks: dict[str, int] = {}
    for position, group in enumerate(text.upper().replace(" ", "").split("<")):
        for label in group.split("="):
            if not label.isalnum() or label in ranks:
                return "write runs from lowest to highest with < and =, each run once, e.g. D < C < A = B"
            ranks[label] = position
    return ranks


class Quiz:
    """A lesson's key: question id -> ``Interval``, ``Ranking`` or ``Word``."""

    def __init__(self, questions: dict[str, Interval | Ranking | Word]):
        if not questions:
            raise ValueError("a quiz needs at least one question")
        self.questions = dict(questions)

    def grade(self, my_answers: dict[str, str]) -> list[Result]:
        """Score every question; an answer that can't be read earns 0 with a note."""
        results = []
        for name, question in self.questions.items():
            raw = my_answers.get(name, "")
            answer = _text(raw)
            if answer is None:
                points, note = 0, "answer with text, e.g. 'c'"
                answer = str(raw)
            elif not answer:
                points, note = 0, "no answer"
            else:
                points, note = question.grade(answer)
            results.append(Result(name, answer, question.key_text, points, question.out_of, note))
        return results

    def score(self, my_answers: dict[str, str]) -> None:
        """Print your score table. Nothing is sent anywhere or saved."""
        if not isinstance(my_answers, dict):
            raise TypeError(
                f"my_answers must be a dict like {{'Q1': 'c'}}, not {type(my_answers).__name__}"
            )
        unknown = sorted(set(my_answers) - set(self.questions))
        if unknown:
            print(
                f"Not scored: {', '.join(map(str, unknown))} is not a question here (questions: {', '.join(self.questions)})\n"
            )
        results = self.grade(my_answers)
        print(f"{'question':<10}{'your answer':<22}{'answer':<22}points")
        for r in results:
            print(f"{r.question:<10}{r.answer or '-':<22}{r.key:<22}{r.points:g} / {r.out_of:g}")
            if r.note:
                print(f"{'':<10}not scored: {r.note}")
        print(
            f"{'total':<54}{sum(r.points for r in results):g} / {sum(r.out_of for r in results):g}"
        )

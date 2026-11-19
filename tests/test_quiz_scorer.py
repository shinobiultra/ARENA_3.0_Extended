"""Unit tests of the shared Held-out Quiz scorer (`quiz.score`), CPU only."""

import importlib
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "infrastructure/extension/learner"))
sys.path.insert(0, str(ROOT / "tests/fixtures"))

quiz_module = importlib.import_module("quiz")
Interval, Quiz, Ranking, Word = (
    quiz_module.Interval,
    quiz_module.Quiz,
    quiz_module.Ranking,
    quiz_module.Word,
)

pytestmark = pytest.mark.cpu


def points(quiz, answers):
    return {r.question: r.points for r in quiz.grade(answers)}


def one(question, answer):
    """Points earned on a single question."""
    return points(Quiz({"Q": question}), {"Q": answer})["Q"]


# ----------------------------------------------------------------- intervals


@pytest.mark.parametrize("answer, expected", [("c", 4), ("b", 2), ("d", 2), ("a", 0), ("e", 0)])
def test_interval_full_half_zero(answer, expected):
    assert one(Interval("c", points=4), answer) == expected


def test_interval_ignores_case_and_spaces():
    assert one(Interval("c", points=4), "  C ") == 4


def test_interval_edge_has_only_one_neighbour():
    assert one(Interval("a", points=4), "b") == 2
    assert one(Interval("e", points=4), "d") == 2


def test_interval_key_must_be_one_of_the_options():
    with pytest.raises(ValueError, match="key"):
        Interval("f", points=1)


# ------------------------------------------------------------------- rankings


def test_ranking_exact_order_gets_one_point_per_pair():
    # four runs -> six pairs -> six points
    assert one(Ranking("D < C < A = B"), "D < C < A = B") == 6


def test_ranking_counts_each_correctly_ordered_pair():
    # swapping the neighbours C and D breaks exactly one of the six pairs
    assert one(Ranking("D < C < A = B"), "C < D < A = B") == 5
    # fully reversed: every strict pair is wrong, and the A = B tie is called a strict order
    assert one(Ranking("D < C < A = B"), "A < B < C < D") == 0


def test_ranking_tie_in_key_needs_a_tie_in_the_answer():
    # A = B is one pair: calling it A < B loses that pair only
    assert one(Ranking("D < C < A = B"), "D < C < A < B") == 5


def test_ranking_from_values_ties_runs_within_eps():
    ranking = Ranking.from_values({"A": 0.96, "B": 0.97, "C": 0.58, "D": 0.01}, eps=0.02)
    assert ranking.key_text == "D < C < A = B"
    assert one(ranking, "D < C < A = B") == 6
    # without a tie inside eps the same runs are strictly ordered
    strict = Ranking.from_values({"A": 0.96, "B": 0.97}, eps=0.001)
    assert strict.key_text == "A < B"


def test_ranking_eps_boundary_counts_as_a_tie():
    assert Ranking.from_values({"A": 1.0, "B": 1.5}, eps=0.5).key_text == "A = B"


def test_ranking_answer_order_of_written_ties_does_not_matter():
    assert one(Ranking("D < C < A = B"), "d < c < b = a") == 6


@pytest.mark.parametrize(
    "bad",
    [
        "D < C < A",  # missing run B
        "D < C < A = B = E",  # unknown run
        "D < C < A < A = B",  # repeated run
        "D > C > A = B",  # only < and = are understood
        "",
    ],
)
def test_ranking_malformed_answers_score_zero_with_a_reason(bad):
    result = Quiz({"Q": Ranking("D < C < A = B")}).grade({"Q": bad})[0]
    assert result.points == 0
    assert result.note  # says what was wrong


def test_ranking_needs_at_least_two_runs():
    with pytest.raises(ValueError, match="two"):
        Ranking("A")


# ----------------------------------------------------------------- one word


def test_word_matches_ignoring_case_and_spaces():
    assert one(Word("softmax", points=1), "  Softmax ") == 1


def test_word_wrong_word_scores_zero():
    assert one(Word("softmax", points=1), "sigmoid") == 0


def test_word_outside_declared_choices_is_malformed_not_just_wrong():
    result = Quiz({"Q": Word("falls", points=1, choices=("falls", "rises"))}).grade({"Q": "fals"})[
        0
    ]
    assert result.points == 0
    assert "falls" in result.note and "rises" in result.note


# --------------------------------------------------------- malformed input


def test_missing_answer_scores_zero_and_says_so():
    result = Quiz({"Q": Interval("c", points=2)}).grade({})[0]
    assert (result.points, result.answer) == (0, "")
    assert "no answer" in result.note


def test_non_string_answer_scores_zero_without_crashing():
    for bad in (3, None, ["c"]):
        result = Quiz({"Q": Interval("c", points=2)}).grade({"Q": bad})[0]
        assert result.points == 0
        assert "text" in result.note


def test_interval_outside_the_options_is_malformed():
    result = Quiz({"Q": Interval("c", points=2)}).grade({"Q": "z"})[0]
    assert result.points == 0
    assert "a, b, c, d or e" in result.note


def test_unknown_question_id_is_reported_not_ignored(capsys):
    Quiz({"Q1": Interval("c", points=2)}).score({"Q1": "c", "Q7": "a"})
    out = capsys.readouterr().out
    assert "Q7" in out and "Q1" in out  # names the typo and the real ids


def test_answers_must_be_a_dict():
    with pytest.raises(TypeError, match="my_answers"):
        Quiz({"Q": Interval("c", points=2)}).score(["c"])


def test_quiz_needs_at_least_one_question():
    with pytest.raises(ValueError, match="question"):
        Quiz({})


# ------------------------------------------------------- the score table


def test_score_prints_a_table_with_total(capsys):
    quiz = Quiz({"Q1": Word("falls", points=1), "Q2": Interval("c", points=2)})
    assert quiz.score({"Q1": "falls", "Q2": "d"}) is None  # prints, returns nothing to echo
    out = capsys.readouterr().out
    assert "question" in out and "your answer" in out and "points" in out
    assert "1 / 1" in out and "1 / 2" in out
    assert "total" in out and "2 / 3" in out


def test_score_never_writes_or_connects(tmp_path, monkeypatch, capsys):
    def no_network(*args, **kwargs):
        raise AssertionError("the scorer must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.chdir(tmp_path)
    Quiz({"Q": Interval("c", points=2)}).score({"Q": "c"})
    assert list(tmp_path.iterdir()) == []
    capsys.readouterr()


# ------------------------------------------------- end to end on a fixture


def test_example_lesson_end_to_end(capsys):
    example = importlib.import_module("quiz_example")

    example.quiz.score(example.EXAMPLE_ANSWERS)
    lines = capsys.readouterr().out.splitlines()
    rows = {line.split()[0]: line for line in lines if line.startswith(("Q1", "Q2", "Q3", "total"))}
    assert rows["Q1"].endswith("1 / 1")
    assert rows["Q2"].endswith("1 / 2")
    assert rows["Q3"].endswith("5 / 6")  # the A = B tie was called A < B
    assert rows["total"].endswith("7 / 9")

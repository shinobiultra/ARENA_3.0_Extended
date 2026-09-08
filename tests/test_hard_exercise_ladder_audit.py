import json

import pytest

from scripts.audit_hard_exercise_ladders import (
    fixture_readme_missing_terms,
    ladder_blockers,
    visible_test_summary,
)


def test_fixture_readme_missing_terms_flags_incomplete_provenance(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text("# Fixture\n\n## Random seed\n0\n")

    missing = fixture_readme_missing_terms(readme)

    assert "How this fixture was produced" in missing
    assert "Allowed tolerances" in missing
    assert "rtol=" in missing


def test_visible_test_summary_counts_ladder_signals(tmp_path):
    tests = tmp_path / "tests.py"
    tests.write_text(
        "\n".join(
            [
                "def test_shape_contract(fn):",
                "    result = fn()",
                "    assert result['shape'] == (2, 3)",
                "    assert result['device'] == 'cuda'",
                "    print('All tests in `test_shape_contract` passed!')",
                "",
                "def test_notebook_contract(run_smoke_test):",
                "    result = run_smoke_test(cpu=True)",
                "    assert result['accepted']",
                "    print('All tests in `test_notebook_contract` passed!')",
            ]
        )
    )

    summary = visible_test_summary(tests)

    assert summary.test_function_count == 2
    assert summary.assert_count == 3
    assert summary.success_print_count == 2
    assert summary.placeholder_tests == ()


@pytest.mark.parametrize("message", [
    'f"All tests in test_complete_basis_operators passed! ({n} players; all {2**n} basis games)"',
    'f"All tests in test_invalid_inputs passed! ({function.__name__} rejected {kwargs})"',
])
def test_visible_feedback_accepts_explanatory_fstrings_after_checks(tmp_path, message):
    tests = tmp_path / "tests.py"
    tests.write_text(f"def test_case():\n    assert actual == expected\n    print({message})\n")
    assert visible_test_summary(tests).success_print_count == 1


@pytest.mark.parametrize("message", ['""', 'f""', 'f" {result} "', 'f"Report: {report}"'])
def test_visible_feedback_rejects_empty_dynamic_only_and_report_dump_messages(tmp_path, message):
    tests = tmp_path / "tests.py"
    tests.write_text(f"def test_case():\n    assert actual == expected\n    print({message})\n")
    assert visible_test_summary(tests).success_print_count == 0


def test_informative_fstring_cannot_replace_a_check(tmp_path):
    tests = tmp_path / "tests.py"
    tests.write_text('def test_case():\n    print(f"All tests passed! ({n} cases)")\n')
    assert visible_test_summary(tests).success_print_count == 0


def test_ladder_blockers_flag_non_cuda_report_and_thin_tests(tmp_path, monkeypatch):
    root = tmp_path
    fixture = root / "expected_outputs/README.md"
    fixture.parent.mkdir(parents=True)
    fixture.write_text(
        "\n".join(
            [
                "How this fixture was produced",
                "Trusted implementation",
                "Random seed",
                "Allowed tolerances rtol=1e-5 atol=1e-6",
                "When to regenerate",
            ]
        )
    )
    tests = root / "tests.py"
    tests.write_text("def test_only_one(fn):\n    assert fn()\n")
    report = root / "verification_report.json"
    report.write_text(
        json.dumps(
            {
                "accepted": True,
                "tests_passed": True,
                "metrics": {"gpu_evidence": {"category": "placeholder_only", "uses_cuda": False}},
            }
        )
    )

    import scripts.audit_hard_exercise_ladders as audit

    monkeypatch.setattr(audit, "ROOT", root)
    blockers = ladder_blockers(
        [
            {
                "notebook_id": "thin_notebook",
                "difficulty": "3",
                "release_status": "accepted_with_placeholder_only",
                "fixture_provenance_source": "expected_outputs/README.md",
                "visible_tests_source": "tests.py",
                "report_source": "verification_report.json",
            }
        ]
    )

    assert "thin_notebook: release_status=accepted_with_placeholder_only" in blockers
    assert any("only 1 visible tests" in blocker for blocker in blockers)
    assert any("print ARENA-style success messages" in blocker for blocker in blockers)
    assert "thin_notebook: report lacks accepted CUDA section metric" in blockers


def test_current_hard_exercise_ladder_audit_passes():
    assert ladder_blockers() == []


def test_feedback_accepts_concrete_messages_after_checks(tmp_path):
    path = tmp_path / 'tests.py'
    path.write_text('def test_projection(fn):\n    assert fn() == 0\n    print("Projection and gradients agree!")\n')
    assert visible_test_summary(path).success_print_count == 1


def test_feedback_rejects_print_only_and_premature_success(tmp_path):
    path = tmp_path / 'tests.py'
    path.write_text('def test_empty():\n    print("All tests in `empty` passed!")\n\ndef test_premature(fn):\n    print("All tests in `premature` passed!")\n    assert fn()\n')
    assert visible_test_summary(path).success_print_count == 0


def test_ladder_counts_real_numpy_and_torch_assertions(tmp_path):
    path = tmp_path / 'tests.py'
    path.write_text('''def test_numbers(fn):
    t.testing.assert_close(fn(), expected)
    np.testing.assert_allclose(fn(), expected)
    np.testing.assert_array_equal(ids, expected_ids)
    logger.assert_close("this is not a numerical test")
    print("Numerical references agree.")
''')
    result = visible_test_summary(path)
    assert result.assert_count == 3
    assert result.success_print_count == 1

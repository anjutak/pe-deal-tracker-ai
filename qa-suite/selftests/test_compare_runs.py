"""
Tests for compare_runs.py: the tool that decides "did anything break?" must
itself be right, or a regression could slip through. Runs offline.
"""
import json

import pytest

import compare_runs as cr

P, F = True, False


@pytest.mark.parametrize("before, after, expected", [
    ([P, P, P], [P, F, P], "broken"),         # stable pass -> any fail
    ([P], [F], "broken"),                     # single-run pytest results
    ([P, F, P], [F, F, P], "worse"),          # flaky -> fails more often
    ([F, F, F], [P, P, P], "fixed"),
    ([P, F, F], [P, P, P], "fixed"),
    ([F, F, F], [P, F, F], "better"),
    ([F, F, F], [F, F, F], "still_failing"),
    ([P, P, P], [P, P, P], "same"),
    ([P, F, P], [F, P, P], "same"),           # same rate, different order
    ([], [P], None),                          # no data on one side
])
def test_change(before, after, expected):
    assert cr.change(before, after) == expected


def test_classify_uses_the_more_serious_check():
    # Keyword check got fixed but the judge now fails: that's a regression, not a fix.
    before = {"keyword": [F, F, F], "judge": [P, P, P]}
    after = {"keyword": [P, P, P], "judge": [P, F, P]}
    assert cr.classify(before, after) == "broken"


def test_compare_groups_new_and_missing_cases():
    before = {"a": {"keyword": [P], "judge": [P]}, "gone": {"keyword": [P], "judge": [P]}}
    after = {"a": {"keyword": [F], "judge": [P]}, "added": {"keyword": [P], "judge": [P]}}
    groups = cr.compare(before, after)
    assert groups["broken"] == ["a"] and groups["new"] == ["added"] and groups["missing"] == ["gone"]


def test_keyword_pass_matches_the_suite_rule():
    # Same must_contain / must_not_contain logic as test_golden_case.
    assert cr.keyword_pass("lookup_01", "The deal team member is Patel.")
    assert not cr.keyword_pass("lookup_01", "The deal team member is Kim.")
    assert not cr.keyword_pass("debt_05", "4.30% + 425% = 429.30%")  # forbidden wrong-unit answer


def test_loads_both_result_formats(tmp_path):
    repeats = tmp_path / "flakiness.json"
    repeats.write_text(json.dumps({"details": [{"id": "lookup_01", "runs": [
        {"answer": "Patel", "keyword_pass": True, "judge_pass": True},
        {"error": "timeout"},  # infrastructure error: ignored, not counted as a fail
        {"answer": "Kim", "keyword_pass": False, "judge_pass": False},
    ]}]}))
    assert cr.load(repeats) == {"lookup_01": {"keyword": [P, F], "judge": [P, F]}}

    pytest_run = tmp_path / "judge_results.json"
    pytest_run.write_text(json.dumps({"results": [{"id": "lookup_01", "answer": "It is Patel.", "passed": False}]}))
    assert cr.load(pytest_run) == {"lookup_01": {"keyword": [P], "judge": [F]}}

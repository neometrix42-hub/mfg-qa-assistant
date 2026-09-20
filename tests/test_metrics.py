"""Tests for eval metrics. A bug here silently invalidates the README table."""

from decimal import Decimal

from evals.metrics import (
    contains_rate,
    mean,
    mrr,
    percentile,
    recall_at_k,
    result_match,
)


# ---------------------------------------------------------------- recall@k

def test_recall_counts_only_the_top_k():
    retrieved = ["a", "b", "c", "d"]
    assert recall_at_k(retrieved, ["a"], 1) == 1.0
    assert recall_at_k(retrieved, ["d"], 3) == 0.0  # present, but below the cut
    assert recall_at_k(retrieved, ["d"], 4) == 1.0


def test_recall_with_several_expected():
    assert recall_at_k(["a", "b", "c"], ["a", "c"], 3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["a", "z"], 3) == 0.5


def test_recall_of_nothing_expected_is_one():
    """Nothing to find means nothing missed - otherwise pure-SQL cases would
    drag the document metric down for no reason."""
    assert recall_at_k([], [], 5) == 1.0


# --------------------------------------------------------------------- MRR

def test_mrr_uses_first_correct_rank():
    assert mrr(["a", "b"], ["a"]) == 1.0
    assert mrr(["x", "a"], ["a"]) == 0.5
    assert mrr(["x", "y", "a"], ["a"]) == 1 / 3


def test_mrr_is_zero_when_absent():
    assert mrr(["x", "y"], ["a"]) == 0.0


def test_recall_and_mrr_disagree_on_deep_hits():
    """The case that justifies tracking both: perfect recall, poor rank."""
    retrieved = ["x", "y", "z", "w", "a"]
    assert recall_at_k(retrieved, ["a"], 5) == 1.0
    assert mrr(retrieved, ["a"]) == 0.2


# ------------------------------------------------------------ result_match

def test_result_match_ignores_row_order():
    assert result_match([(1,), (2,)], [(2,), (1,)])


def test_result_match_normalises_numeric_types():
    """Decimal from Postgres must compare equal to a plain int."""
    assert result_match([(Decimal("3"),)], [(3,)])
    assert result_match([(Decimal("3.0"),)], [(3.0,)])


def test_result_match_detects_real_differences():
    assert not result_match([(1,)], [(2,)])
    assert not result_match([(1,)], [(1,), (2,)])


def test_result_match_handles_none():
    assert not result_match(None, [(1,)])
    assert result_match([(None,)], [(None,)])


# ----------------------------------------------------------- contains_rate

def test_contains_rate_is_case_insensitive():
    assert contains_rate("The interval is 12 MONTHS.", ["12 months"]) == 1.0


def test_contains_rate_is_partial():
    assert contains_rate("quarantine only", ["quarantine", "NCR tag"]) == 0.5


def test_contains_rate_of_no_requirements_is_one():
    assert contains_rate("anything", []) == 1.0


# -------------------------------------------------------------- percentile

def test_percentile_interpolates():
    assert percentile([0, 10], 0.5) == 5.0
    assert percentile([1, 2, 3, 4], 0.0) == 1.0
    assert percentile([1, 2, 3, 4], 1.0) == 4.0


def test_percentile_edge_cases():
    assert percentile([], 0.95) == 0.0
    assert percentile([7], 0.95) == 7.0


def test_mean_of_empty_is_zero():
    assert mean([]) == 0.0

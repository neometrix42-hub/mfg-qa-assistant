"""Structural checks on the golden set.

These need no database and no API. They catch the mistakes that would make the
results table meaningless: duplicate ids, cases citing chunks the chunker never
produces, or data cases with no reference to compare against.

`python -m evals.run --verify` is the companion check - it executes every
reference_sql against the real database. Run that too, once Postgres is up.
"""

import pytest

from evals.run import load_golden_set
from src.ingest.chunk import chunk_all

VALID_TYPES = {"doc", "data", "hybrid"}


def count(text: str) -> int:
    return max(1, int(len(text.split()) * 1.3))


@pytest.fixture(scope="module")
def cases():
    return load_golden_set()


@pytest.fixture(scope="module")
def citations():
    return {c.citation for c in chunk_all(256, 32, count)}


def test_golden_set_is_not_empty(cases):
    assert cases


def test_ids_are_unique(cases):
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))


def test_every_case_is_well_formed(cases):
    for case in cases:
        assert case.get("id"), case
        assert case.get("question", "").strip(), case["id"]
        assert case.get("type") in VALID_TYPES, case["id"]


def test_expected_chunks_actually_exist(cases, citations):
    """A case citing a chunk the chunker never produces scores 0 recall
    forever, and it looks like a retrieval failure rather than a typo."""
    missing = [
        (case["id"], citation)
        for case in cases
        for citation in (case.get("expected_chunks") or [])
        if citation not in citations
    ]
    assert not missing, f"cases cite non-existent chunks: {missing}"


def test_data_cases_have_a_reference(cases):
    """Without reference_sql there is nothing to compare the model's SQL to,
    so the case silently contributes nothing to sql_match."""
    for case in cases:
        if case["type"] == "data":
            assert case.get("reference_sql"), f"{case['id']} has no reference_sql"


def test_reference_sql_is_read_only(cases):
    """The harness runs these on the read-only connection; a write would fail
    at eval time rather than here, mid-sweep, after spending money."""
    from src.tools.query_measurements import validate

    for case in cases:
        if sql := case.get("reference_sql"):
            validate(sql)  # raises UnsafeSQL if not a single read-only SELECT


def test_unanswerable_cases_exist(cases):
    """Without them the set cannot distinguish 'grounded' from
    'confidently wrong'."""
    assert any(c["id"].startswith("unanswerable") for c in cases)


def test_there_is_a_hybrid_case(cases):
    assert any(c["type"] == "hybrid" for c in cases)

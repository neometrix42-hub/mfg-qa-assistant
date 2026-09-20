"""Tests for the chunker. Run: pytest

A stub token counter is used so these run without loading the embedding model
(which pulls in torch). The packing logic is what is under test, not the
tokenizer - swap in the real counter only when you want to check real budgets.
"""

import pytest

from src.ingest.chunk import _hard_split, _pack, chunk_all, parse_document, DOCS_DIR


def count(text: str) -> int:
    """Roughly 1.3 tokens per whitespace word. Good enough to exercise packing."""
    return max(1, int(len(text.split()) * 1.3))


# --------------------------------------------------------------- parsing

def test_every_document_parses():
    docs = sorted(DOCS_DIR.glob("*.md"))
    assert docs, "no documents found"
    for path in docs:
        meta, sections = parse_document(path)
        assert meta["doc_id"] and meta["doc_title"]
        assert sections, f"{path.name} produced no sections"


def test_numbered_sections_are_extracted():
    meta, sections = parse_document(DOCS_DIR / "sop_calibration.md")
    numbers = {s.number for s in sections if s.number}
    assert {"3.1", "3.2", "3.3"} <= numbers


def test_golden_set_citations_resolve():
    """The golden set cites these. If chunking stops producing them, recall@k
    silently measures nothing."""
    citations = {c.citation for c in chunk_all(256, 32, count)}
    for required in ("sop_calibration#3.2", "sop_nonconformance#2.1"):
        assert required in citations


def test_context_header_is_prepended():
    chunks = chunk_all(256, 32, count)
    target = next(c for c in chunks if c.citation == "sop_calibration#3.2")
    assert target.content.startswith(target.doc_title)
    assert "3.2" in target.content.split("\n")[0]


def test_preamble_does_not_repeat_the_title():
    for chunk in chunk_all(256, 32, count):
        first_line = chunk.content.split("\n")[0]
        assert first_line != f"{chunk.doc_title} - {chunk.doc_title}"


# --------------------------------------------------------------- packing

@pytest.mark.parametrize("budget", [60, 128, 256])
def test_no_chunk_exceeds_budget(budget):
    """The one that matters. An oversized chunk is silently truncated at
    embedding time - invisible in output, shows up only as bad recall."""
    for chunk in chunk_all(budget, 15, count):
        assert count(chunk.content) <= budget


def test_overlap_carries_content_between_adjacent_chunks():
    body = "\n\n".join(f"Sentence block number {i} with several filler words." for i in range(20))
    pieces = _pack(body, budget=40, overlap=15, count=count)
    assert len(pieces) > 1
    shared = set(pieces[0].split()) & set(pieces[1].split())
    assert any(word.isdigit() for word in shared), "no real overlap between pieces"


def test_overlap_tail_cannot_push_a_chunk_over_budget():
    """Regression: the flush check runs against the OLD cur_tokens, so without
    capping the tail allowance, tail + next unit could exceed the budget."""
    body = "\n\n".join(f"Block {i} " + "filler " * 20 for i in range(15))
    for piece in _pack(body, budget=40, overlap=30, count=count):
        assert count(piece) <= 40


def test_hard_split_bounds_and_preserves_words():
    giant = " ".join(["word"] * 500)
    pieces = _hard_split(giant, budget=40, count=count)
    assert len(pieces) > 1
    assert all(count(p) <= 40 for p in pieces)
    assert sum(len(p.split()) for p in pieces) == 500


def test_pack_handles_a_unit_larger_than_the_whole_budget():
    giant = " ".join(["word"] * 500)
    assert all(count(p) <= 40 for p in _pack(giant, budget=40, overlap=10, count=count))

"""Tests for the pure parts of hybrid search.

Query building and rank fusion need no database, so they are cheap to test and
worth testing: a bug in either silently changes every retrieval number.
"""

from src.tools.search_documents import rrf_fuse, to_or_query


# ------------------------------------------------------------- query building

def test_words_are_ored_not_anded():
    """AND semantics would require every term in one chunk and return nothing
    for a full-sentence question."""
    # "be" is dropped: two characters or fewer is noise.
    assert to_or_query("how often must the CMM be calibrated") == \
        "how or often or must or the or CMM or calibrated"


def test_short_words_are_dropped():
    q = to_or_query("is it a CMM")
    assert "CMM" in q
    assert " is " not in f" {q} "


def test_part_numbers_survive():
    """The whole reason keyword search was added: 'P-4417' has no useful
    embedding, but it is an exact token."""
    assert "P-4417" in to_or_query("Part P-4417 failed a flatness check")


def test_punctuation_does_not_break_the_query():
    q = to_or_query("What happens if it fails?? (urgent!) -- 50% out")
    assert "??" not in q and "(" not in q
    assert "fails" in q and "urgent" in q


def test_empty_query_is_empty():
    assert to_or_query("!! ?? ..") == ""


def test_query_is_length_capped():
    assert len(to_or_query(" ".join(f"word{i}" for i in range(100))).split(" or ")) == 40


# ---------------------------------------------------------------- rank fusion

def test_rank_one_in_both_lists_wins():
    scores = rrf_fuse([[10, 20], [10, 30]], k=60)
    assert scores[10] > scores[20]
    assert scores[10] > scores[30]


def test_agreement_beats_a_single_high_rank():
    """RRF deliberately rewards documents both retrievers found. This is also
    why a chunk only one retriever finds can drop out - the behaviour that
    cost us hybrid_003."""
    scores = rrf_fuse([[1, 99], [2, 99]], k=60)
    assert scores[99] > scores[1]


def test_scores_decrease_with_rank():
    scores = rrf_fuse([[1, 2, 3]], k=60)
    assert scores[1] > scores[2] > scores[3]


def test_missing_from_a_list_is_not_an_error():
    scores = rrf_fuse([[1], []], k=60)
    assert set(scores) == {1}


def test_k_controls_how_much_top_ranks_dominate():
    small = rrf_fuse([[1, 2]], k=1)
    large = rrf_fuse([[1, 2]], k=1000)
    assert small[1] / small[2] > large[1] / large[2]

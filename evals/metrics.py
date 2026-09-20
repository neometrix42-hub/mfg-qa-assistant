"""Eval metrics.

YOU WRITE THESE. They are small, pure functions - easy to unit test, and worth
testing, because a bug here silently invalidates every number in your README.
"""


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Fraction of expected citations present in the top k retrieved.

    TODO: intersect expected with retrieved[:k], divide by len(expected).
    Return 1.0 when expected is empty (nothing to find = nothing missed).
    """
    raise NotImplementedError


def mrr(retrieved: list[str], expected: list[str]) -> float:
    """Mean reciprocal rank of the FIRST correct citation.

    TODO: find the lowest index i where retrieved[i] is in expected,
    return 1/(i+1). Return 0.0 if none match.

    Why both this and recall@k: recall says "did we find it", MRR says "how
    far down the list". A system with good recall but bad MRR is stuffing the
    right answer in at position 9, where the model may ignore it.
    """
    raise NotImplementedError


def result_match(actual_rows: list[tuple], expected_rows: list[tuple]) -> bool:
    """True if two SQL result sets are equivalent.

    TODO: compare as sorted lists of tuples - row ORDER should not matter
    unless the question asked for ordering. Normalise numeric types first
    (Decimal('3') and 3 must compare equal, or you will chase phantom failures).
    """
    raise NotImplementedError


def contains_rate(answer: str, must_contain: list[str]) -> float:
    """Fraction of required strings present in the answer (case-insensitive).

    TODO: straightforward. Return 1.0 when must_contain is empty.
    """
    raise NotImplementedError


def faithfulness(answer: str, context: str, client, model: str) -> tuple[bool, str]:
    """LLM-as-judge: is `answer` supported by `context`?

    TODO: one Claude call. Ask for a verdict plus one sentence of reasoning,
    and LOG THE REASONING - you will be asked about judge reliability.

    Known limitation, and the right thing to say in an interview: an LLM judge
    is weak and correlated with the model being judged. That is exactly why it
    sits alongside exact-match checks (contains_rate, result_match) rather than
    replacing them.
    """
    raise NotImplementedError


def percentile(values: list[float], p: float) -> float:
    """p50 / p95 latency. TODO: sort, index at p * (n-1), interpolate."""
    raise NotImplementedError

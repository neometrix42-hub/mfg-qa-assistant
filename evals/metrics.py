"""Eval metrics.

Small pure functions on purpose - easy to unit test, and worth testing, because
a bug here silently invalidates every number in your README.
"""

from decimal import Decimal

JUDGE_PROMPT = """You are grading whether an answer is supported by the context it was given.

CONTEXT:
{context}

ANSWER:
{answer}

Is every factual claim in the ANSWER supported by the CONTEXT?
Reply with exactly one line:
SUPPORTED: <one sentence why>
or
UNSUPPORTED: <one sentence naming the unsupported claim>"""


def recall_at_k(retrieved: list[str], expected: list[str], k: int) -> float:
    """Fraction of expected citations present in the top k retrieved."""
    if not expected:
        return 1.0  # nothing to find means nothing missed
    top = set(retrieved[:k])
    return sum(1 for e in expected if e in top) / len(expected)


def mrr(retrieved: list[str], expected: list[str]) -> float:
    """Reciprocal rank of the FIRST correct citation, 0.0 if none.

    Recall says "did we find it"; MRR says "how far down". A system with good
    recall but poor MRR is burying the right chunk at position 9, where the
    model may well ignore it.
    """
    if not expected:
        return 1.0
    wanted = set(expected)
    for index, citation in enumerate(retrieved):
        if citation in wanted:
            return 1.0 / (index + 1)
    return 0.0


def _normalise(value):
    """Make Decimal('3'), 3 and 3.0 compare equal."""
    if isinstance(value, (Decimal, float, int)) and not isinstance(value, bool):
        return round(float(value), 6)
    if value is None:
        return None
    return str(value)


def result_match(actual_rows, expected_rows) -> bool:
    """True if two SQL result sets are equivalent.

    Row order is ignored - unless the question asked for ordering, two correct
    queries may legitimately return rows in different orders.
    """
    if actual_rows is None or expected_rows is None:
        return False
    actual = sorted(tuple(_normalise(v) for v in row) for row in actual_rows)
    expected = sorted(tuple(_normalise(v) for v in row) for row in expected_rows)
    return actual == expected


def contains_rate(answer: str, must_contain: list[str]) -> float:
    """Fraction of required strings present in the answer, case-insensitive."""
    if not must_contain:
        return 1.0
    low = answer.lower()
    return sum(1 for s in must_contain if s.lower() in low) / len(must_contain)


def faithfulness(answer: str, context: str, client, model: str) -> tuple[bool, str]:
    """LLM-as-judge: is `answer` supported by `context`?

    Returns (verdict, reasoning). The reasoning is logged deliberately - you
    will be asked about judge reliability, and being able to show what the
    judge actually said is the difference between a real answer and a shrug.

    Known limitation, and the right thing to say out loud: an LLM judge is weak
    and correlated with the model being judged. That is exactly why it sits
    alongside exact-match checks (contains_rate, result_match) rather than
    replacing them.
    """
    if not context.strip():
        return False, "No context was retrieved, so nothing supports the answer."

    response = client.messages.create(
        model=model,
        max_tokens=256,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(context=context[:20000], answer=answer),
        }],
    )
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text.upper().startswith("SUPPORTED"), text


def percentile(values: list[float], p: float) -> float:
    """Linear-interpolated percentile. p is a fraction, e.g. 0.95."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = p * (len(ordered) - 1)
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0

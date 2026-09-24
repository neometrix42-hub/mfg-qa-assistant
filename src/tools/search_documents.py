"""Retrieval over doc_chunks: vector, keyword, or hybrid.

WHY HYBRID
----------
Measured on the golden set, pure vector search missed in three specific ways:

  * hybrid_001 - "Part P-4417 failed a flatness check" never retrieved the
    non-conformance procedure at all. Part numbers have no meaningful
    embedding; they are exact tokens.
  * hybrid_003 - asked about non-conformance Records (7 years), returned
    calibration Records (3 years). Near-identical sections across documents
    collapse into near-identical vectors.
  * doc_004 - asked about failing calibration acceptance criteria, returned the
    First Article Inspection "Acceptance" section.

Keyword search is good at exactly what embeddings are bad at, and vice versa.
Hybrid runs both and fuses the rankings.

FUSION: RECIPROCAL RANK FUSION
------------------------------
score(d) = sum over lists of 1 / (k + rank(d))

RRF uses only the RANK a document got in each list, never the raw score. That
matters here because cosine distance (0-2, lower better) and ts_rank_cd
(unbounded, higher better) are not comparable, and any attempt to normalise
them needs tuning that would itself have to be measured. RRF needs none.
k=60 is the value from the original paper and is not sensitive.
"""

import re
from collections import defaultdict
from dataclasses import dataclass

from src import runtime
from src.config import cfg
from src.db import connect
from src.ingest.embed import embed_text

# pgvector distance operators:
#   <=>  cosine distance (used here; 0 = identical)
#   <->  L2 / Euclidean
#   <#>  negative inner product
# With normalised embeddings, cosine and inner product rank identically.
VECTOR_SQL = """
    SELECT chunk_id, doc_id, doc_title, section, content,
           1 - (embedding <=> %(vec)s) AS score
    FROM doc_chunks
    ORDER BY embedding <=> %(vec)s
    LIMIT %(n)s
"""

# Normalisation flag 1 divides the rank by 1 + log(document length). Without it
# ts_rank_cd ignores length entirely, so a long chunk that mentions a term in
# passing outranks a short chunk that is entirely about it. That length bias
# measurably cost us the "non-conformance records" case.
KEYWORD_SQL = """
    SELECT chunk_id, doc_id, doc_title, section, content,
           ts_rank_cd(content_tsv, websearch_to_tsquery('english', %(q)s), 1) AS score
    FROM doc_chunks
    WHERE content_tsv @@ websearch_to_tsquery('english', %(q)s)
    ORDER BY score DESC
    LIMIT %(n)s
"""

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    doc_title: str
    section: str | None
    content: str
    score: float  # higher is better; only comparable WITHIN a search mode

    @property
    def citation(self) -> str:
        return f"{self.doc_id}#{self.section}" if self.section else self.doc_id


def to_or_query(text: str) -> str:
    """Turn a natural question into an OR tsquery.

    AND semantics (what plainto_tsquery gives) are far too strict for a full
    sentence - "how often must the CMM be calibrated" would require every term
    to appear in one chunk and usually returns nothing. OR plus ts_rank_cd
    ranks by how many query terms a chunk matches, which is what we want.

    Words of 1-2 characters are dropped as noise. Hyphenated tokens are kept
    whole so 'P-4417' survives; Postgres indexes it as 'p-4417', 'p' and '4417'.
    """
    words = [w for w in _WORD.findall(text) if len(w) > 2]
    return " or ".join(words[:40])


def rrf_fuse(rankings: list[list[int]], k: int | None = None) -> dict[int, float]:
    """Reciprocal Rank Fusion over several ranked lists of chunk_ids."""
    k = k or cfg.rrf_k
    scores: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1.0 / (k + rank)
    return dict(scores)


def _rows_to_hits(rows) -> list[Hit]:
    return [Hit(*row) for row in rows]


def search_docs(
    query: str,
    top_k: int | None = None,
    mode: str | None = None,
) -> list[Hit]:
    """Return the top_k most relevant chunks.

    mode: 'vector' | 'keyword' | 'hybrid'. Defaults to the configured mode so
    the eval sweep can vary it without touching call sites.

    Retrieval depth is an engineering decision measured by the sweep, not
    something the model picks per call - which is why top_k is not on the tool
    schema in agent.py.
    """
    k = top_k or runtime.top_k()
    search_mode = mode or runtime.search_mode()
    pool = max(k, cfg.candidate_pool)

    with connect() as conn:
        if search_mode in ("vector", "hybrid"):
            vec = embed_text(query)
            vector_hits = _rows_to_hits(
                conn.execute(VECTOR_SQL, {"vec": vec, "n": pool}).fetchall()
            )
        else:
            vector_hits = []

        if search_mode in ("keyword", "hybrid"):
            tsquery = to_or_query(query)
            keyword_hits = (
                _rows_to_hits(conn.execute(KEYWORD_SQL, {"q": tsquery, "n": pool}).fetchall())
                if tsquery
                else []
            )
        else:
            keyword_hits = []

    if search_mode == "vector":
        hits = vector_hits[:k]
    elif search_mode == "keyword":
        hits = keyword_hits[:k]
    elif search_mode == "hybrid":
        by_id = {h.chunk_id: h for h in vector_hits + keyword_hits}
        scores = rrf_fuse(
            [[h.chunk_id for h in vector_hits], [h.chunk_id for h in keyword_hits]]
        )
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]
        hits = [
            Hit(
                chunk_id=by_id[cid].chunk_id,
                doc_id=by_id[cid].doc_id,
                doc_title=by_id[cid].doc_title,
                section=by_id[cid].section,
                content=by_id[cid].content,
                score=score,
            )
            for cid, score in ordered
        ]
    else:
        raise ValueError(f"Unknown search mode: {search_mode!r}")

    # The eval harness needs rank order to compute recall@k and MRR.
    runtime.record_retrieval([h.citation for h in hits], [h.content for h in hits])
    return hits


if __name__ == "__main__":
    questions = [
        "how often must the CMM be calibrated",
        "Part P-4417 failed a flatness check, what does our procedure require next",
        "how long must we keep non-conformance records",
    ]
    for mode in ("vector", "keyword", "hybrid"):
        print(f"\n{'=' * 70}\nMODE: {mode}")
        for question in questions:
            print(f"\n  {question}")
            for hit in search_docs(question, top_k=3, mode=mode):
                print(f"    [{hit.score:.4f}] {hit.citation}")

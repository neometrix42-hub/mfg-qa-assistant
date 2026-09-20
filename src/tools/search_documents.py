"""Semantic search over doc_chunks using pgvector.

Week 3 goal: this works and returns sane chunks with NO LLM involved. Run the
smoke test at the bottom and read the output with your own eyes before wiring
up Claude. Retrieval is where RAG projects die.
"""

from dataclasses import dataclass

from src.config import cfg
from src.db import connect
from src.ingest.embed import embed_text

# pgvector distance operators:
#   <=>  cosine distance    (what we use; 0 = identical, 2 = opposite)
#   <->  L2 / Euclidean
#   <#>  negative inner product
# With normalised embeddings, cosine and inner product rank identically.
SEARCH_SQL = """
    SELECT chunk_id, doc_id, doc_title, section, content,
           embedding <=> %(vec)s AS distance
    FROM doc_chunks
    ORDER BY embedding <=> %(vec)s
    LIMIT %(k)s
"""


@dataclass
class Hit:
    chunk_id: int
    doc_id: str
    doc_title: str
    section: str | None
    content: str
    distance: float

    @property
    def citation(self) -> str:
        return f"{self.doc_id}#{self.section}" if self.section else self.doc_id


def search_docs(query: str, top_k: int | None = None) -> list[Hit]:
    """Return the top_k chunks most similar to `query`."""
    k = top_k or cfg.top_k
    vec = embed_text(query)

    with connect() as conn:
        rows = conn.execute(SEARCH_SQL, {"vec": vec, "k": k}).fetchall()

    return [Hit(*row) for row in rows]


if __name__ == "__main__":
    # Week-3 smoke test. READ these chunks. If one could not answer a question
    # on its own, the chunking is wrong and no downstream tuning will fix it.
    for question in [
        "how often must the CMM be calibrated",
        "what do I do with a part that failed inspection",
        "when is a first article inspection required",
    ]:
        print(f"\n=== {question}")
        for hit in search_docs(question, top_k=3):
            print(f"  [{hit.distance:.4f}] {hit.citation}")
            print(f"    {hit.content[:110].replace(chr(10), ' ')}...")

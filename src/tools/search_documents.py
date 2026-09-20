"""Semantic search over doc_chunks using pgvector.

YOU WRITE THIS ONE. It is short, and retrieval quality is the thing your whole
eval harness measures - you need to understand it line by line.

Week 3 goal: this module works and returns sane chunks, with NO LLM involved.
Test it from a script and read the output with your own eyes before you ever
wire up Claude. Retrieval is where RAG projects die.
"""

from dataclasses import dataclass

from src.config import cfg
from src.db import connect
from src.ingest.embed import embed_text


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
    """Return the top_k most similar chunks to `query`.

    TODO - implement:
      1. vec = embed_text(query)
      2. SELECT chunk_id, doc_id, doc_title, section, content,
                embedding <=> %s AS distance
         FROM doc_chunks
         ORDER BY embedding <=> %s
         LIMIT %s
      3. Map rows to Hit objects and return them.

    Notes:
      - `<=>` is pgvector's cosine distance. Lower is better (0 = identical).
        The other operators are `<->` (L2) and `<#>` (inner product) - know the
        difference, it is a likely interview question.
      - Pass the vector as a parameter. Do NOT f-string it into the SQL.
      - Use `top_k or cfg.top_k` so the eval sweep can override it.
    """
    raise NotImplementedError("Week 3: implement vector search")


if __name__ == "__main__":
    # Your week-3 smoke test. Run it and READ the chunks that come back.
    for hit in search_docs("how often must the CMM be calibrated"):
        print(f"[{hit.distance:.4f}] {hit.citation}: {hit.content[:120]}...")

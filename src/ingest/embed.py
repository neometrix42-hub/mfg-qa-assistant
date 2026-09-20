"""Embedding via sentence-transformers. Runs locally - no API cost.

YOU WRITE THIS ONE.

all-MiniLM-L6-v2 produces 384-dimensional vectors, which is why doc_chunks.embedding
is vector(384). If you swap the model, change the schema to match or inserts fail.
"""

from functools import lru_cache

from src.config import cfg


@lru_cache(maxsize=1)
def _model():
    """Load once and cache. Loading per call is slow enough to ruin your evals."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(cfg.embedding_model)


def embed_text(text: str) -> list[float]:
    """Embed a single string (a query, or one chunk).

    TODO: return _model().encode(text, normalize_embeddings=True).tolist()

    normalize_embeddings=True matters: with normalised vectors, cosine distance
    and inner product agree, and pgvector's `<=>` behaves predictably.
    """
    raise NotImplementedError("Week 3: implement embedding")


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Embed many strings at once.

    TODO: use _model().encode(texts, ...) in ONE call, not a loop over embed_text.
    Batching is roughly an order of magnitude faster and you will re-embed the
    whole corpus once per config in the sweep.
    """
    raise NotImplementedError("Week 3: implement batch embedding")

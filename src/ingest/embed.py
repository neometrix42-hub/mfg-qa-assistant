"""Embedding via sentence-transformers. Runs locally - no API cost.

all-MiniLM-L6-v2 produces 384-dimensional vectors, which is why
doc_chunks.embedding is vector(384). Swap the model and the schema must change
to match, or inserts fail.

It also truncates input at 256 tokens. `max_input_tokens()` exposes that so the
ingest pipeline can refuse to build an index it would silently corrupt.
"""

from functools import lru_cache

import numpy as np

from src.config import cfg


@lru_cache(maxsize=1)
def _model():
    """Load once and cache. Loading per call is slow enough to ruin your evals."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(cfg.embedding_model)


def max_input_tokens() -> int:
    """Tokens beyond this are silently discarded before embedding."""
    return int(_model().max_seq_length)


def count_tokens(text: str) -> int:
    """Token count using the embedding model's own tokenizer."""
    return len(_model().tokenizer.encode(text, add_special_tokens=False))


def embed_text(text: str) -> np.ndarray:
    """Embed a single string (a query, or one chunk).

    normalize_embeddings=True matters: with normalised vectors, cosine distance
    and inner product agree, and pgvector's `<=>` behaves predictably.
    """
    return _model().encode(text, normalize_embeddings=True)


def embed_batch(texts: list[str]) -> list[np.ndarray]:
    """Embed many strings in one call.

    Batching is roughly an order of magnitude faster than looping embed_text,
    and the config sweep re-embeds the whole corpus once per configuration.
    """
    vectors = _model().encode(
        texts,
        normalize_embeddings=True,
        batch_size=64,
        show_progress_bar=len(texts) > 100,
    )
    return list(vectors)

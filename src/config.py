"""Central config. Everything tunable lives here so evals/run.py can sweep it."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    database_url: str = os.getenv("DATABASE_URL", "")
    readonly_database_url: str = os.getenv("READONLY_DATABASE_URL", "")

    claude_model: str = os.getenv("CLAUDE_MODEL", "claude-opus-5")
    # The faithfulness judge runs once per eval case, so a full sweep calls it
    # hundreds of times. Setting JUDGE_MODEL=claude-haiku-4-5 cuts that cost
    # substantially; it is your call whether the cheaper judge is good enough,
    # and the eval harness records the judge model in every results file.
    judge_model: str = os.getenv("JUDGE_MODEL", "claude-opus-5")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # USD per million tokens, for the cost column in eval results.
    input_price_per_mtok: float = float(os.getenv("INPUT_PRICE_PER_MTOK", "5.0"))
    output_price_per_mtok: float = float(os.getenv("OUTPUT_PRICE_PER_MTOK", "25.0"))

    # Retrieval knobs - these are what the config sweep varies.
    # chunk_size is in TOKENS, measured with the embedding model's own tokenizer.
    # Hard ceiling: all-MiniLM-L6-v2 truncates at 256 tokens. Going above that
    # does not embed more text, it silently discards the tail.
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "256"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "32"))
    top_k: int = int(os.getenv("TOP_K", "5"))

    # Safety limits for LLM-generated SQL.
    max_sql_rows: int = 500
    max_result_chars: int = 8000


cfg = Config()


def require_keys() -> None:
    """Fail loudly at startup rather than mysteriously at the first API call."""
    missing = [
        name
        for name, value in [
            ("ANTHROPIC_API_KEY", cfg.anthropic_api_key),
            ("DATABASE_URL", cfg.database_url),
            ("READONLY_DATABASE_URL", cfg.readonly_database_url),
        ]
        if not value
    ]
    if missing:
        raise RuntimeError(f"Missing env vars: {', '.join(missing)}. Copy .env.example to .env")

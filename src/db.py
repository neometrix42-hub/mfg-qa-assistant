"""Database connections.

Two separate connection functions on purpose:

  connect()          full access - ingest, doc_chunks, request_log
  connect_readonly() the qa_reader role - the ONLY connection LLM SQL runs on

Never let the LLM path touch connect(). That separation is the whole point.
"""

from contextlib import contextmanager

import psycopg
from pgvector.psycopg import register_vector

from src.config import cfg


@contextmanager
def connect():
    """Full-access connection."""
    with psycopg.connect(cfg.database_url) as conn:
        register_vector(conn)
        yield conn


@contextmanager
def connect_readonly():
    """Read-only connection for LLM-generated SQL (qa_reader role)."""
    with psycopg.connect(cfg.readonly_database_url) as conn:
        conn.read_only = True
        yield conn


def healthcheck() -> bool:
    """True if both connections work. Run this before anything else."""
    try:
        with connect() as conn:
            conn.execute("SELECT 1")
        with connect_readonly() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as exc:  # noqa: BLE001 - startup diagnostics
        print(f"Database healthcheck failed: {exc}")
        return False


if __name__ == "__main__":
    print("OK" if healthcheck() else "FAILED")

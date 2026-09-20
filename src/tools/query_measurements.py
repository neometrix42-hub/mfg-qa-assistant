"""Layer 2 of the SQL defence: validate LLM-generated SQL before executing it.

This file is written for you because it is security-critical and easy to get
subtly wrong. Read it until you can explain every check - you WILL be asked
"what if Claude writes bad SQL?" in an interview.

Three independent layers:
  1. sql/002_readonly_role.sql  - qa_reader physically cannot write
  2. this module                - reject anything that is not a single SELECT
  3. row + character truncation - results cannot blow up the context window
"""

import re

from src.config import cfg
from src.db import connect_readonly

# Statement must start with one of these (after comment stripping).
_ALLOWED_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)

# Any of these as a standalone word is an immediate reject.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|"
    r"vacuum|reindex|call|do|merge|set|reset|listen|notify)\b",
    re.IGNORECASE,
)

_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_HAS_LIMIT = re.compile(r"\blimit\s+\d+", re.IGNORECASE)


class UnsafeSQL(ValueError):
    """Raised when generated SQL fails validation."""


def _strip_comments(sql: str) -> str:
    """Comments can hide forbidden keywords from a naive scan."""
    return _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql))


def validate(sql: str) -> str:
    """Return a safe, LIMIT-bounded statement, or raise UnsafeSQL."""
    cleaned = _strip_comments(sql).strip()

    if not cleaned:
        raise UnsafeSQL("Empty statement.")

    # Reject stacked statements: allow at most one trailing semicolon.
    body = cleaned.rstrip().rstrip(";").rstrip()
    if ";" in body:
        raise UnsafeSQL("Multiple statements are not allowed.")

    if not _ALLOWED_START.match(body):
        raise UnsafeSQL("Only SELECT and WITH statements are allowed.")

    if (found := _FORBIDDEN.search(body)) is not None:
        raise UnsafeSQL(f"Forbidden keyword: {found.group(0).upper()}")

    if not _HAS_LIMIT.search(body):
        body = f"{body} LIMIT {cfg.max_sql_rows}"

    return body


def run_readonly_sql(sql: str) -> str:
    """Validate, EXPLAIN, execute, and format results for the model.

    Errors are RETURNED as text rather than raised. That is deliberate: the tool
    result goes back to Claude, which then corrects its own SQL. That
    self-correction loop is worth demonstrating in a demo.
    """
    try:
        safe_sql = validate(sql)
    except UnsafeSQL as exc:
        return f"SQL rejected: {exc}\nRewrite it as a single read-only SELECT."

    try:
        with connect_readonly() as conn:
            # EXPLAIN first: catches syntax and column errors without running
            # the query, and gives Claude a precise message to correct against.
            try:
                conn.execute(f"EXPLAIN {safe_sql}")
            except Exception as exc:  # noqa: BLE001 - surfaced to the model
                return f"SQL did not validate: {exc}\nCheck table and column names, then retry."

            cur = conn.execute(safe_sql)
            columns = [d.name for d in cur.description or []]
            rows = cur.fetchall()
    except Exception as exc:  # noqa: BLE001 - surfaced to the model
        return f"Query failed: {exc}"

    if not rows:
        return f"Query ran successfully but returned no rows.\n\nSQL:\n{safe_sql}"

    header = " | ".join(columns)
    divider = "-" * len(header)
    body = "\n".join(" | ".join("NULL" if v is None else str(v) for v in row) for row in rows)
    out = f"SQL:\n{safe_sql}\n\n{header}\n{divider}\n{body}\n\n({len(rows)} rows)"

    # Layer 3: never let a wide result set eat the context window.
    if len(out) > cfg.max_result_chars:
        out = out[: cfg.max_result_chars] + "\n... (truncated)"
    return out

"""Per-request tracing and per-run config overrides.

Two problems this solves:

1. The eval harness needs to know WHICH chunks were retrieved and WHAT SQL was
   generated - not just the final answer. Without that, recall@k and MRR cannot
   be computed at all. A Trace collects it as the tools run.

2. The config sweep needs to vary top_k without mutating the frozen global
   Config or re-importing modules. `override()` scopes that to one run.

Both use contextvars, so nothing leaks between concurrent requests.
"""

import contextvars
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from src.config import cfg


# ------------------------------------------------------------------ tracing

@dataclass
class Trace:
    """What the tools did during one request."""

    retrieved: list[str] = field(default_factory=list)  # citations, in rank order
    contexts: list[str] = field(default_factory=list)  # retrieved chunk text
    sql: list[str] = field(default_factory=list)  # every SQL the model generated
    tools_called: list[str] = field(default_factory=list)


_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar("trace", default=None)


def current_trace() -> Trace | None:
    return _trace.get()


@contextmanager
def tracing() -> Iterator[Trace]:
    """Collect tool activity for the duration of the block."""
    trace = Trace()
    token = _trace.set(trace)
    try:
        yield trace
    finally:
        _trace.reset(token)


def record_retrieval(citations: list[str], contexts: list[str]) -> None:
    if (trace := _trace.get()) is not None:
        trace.retrieved.extend(citations)
        trace.contexts.extend(contexts)
        trace.tools_called.append("search_documents")


def record_sql(sql: str) -> None:
    if (trace := _trace.get()) is not None:
        trace.sql.append(sql)
        trace.tools_called.append("query_measurements")


# ----------------------------------------------------------------- overrides

_overrides: contextvars.ContextVar[dict] = contextvars.ContextVar("overrides", default={})


@contextmanager
def override(**values) -> Iterator[None]:
    """Temporarily override config values for one eval run."""
    token = _overrides.set({**_overrides.get(), **values})
    try:
        yield
    finally:
        _overrides.reset(token)


def top_k() -> int:
    return int(_overrides.get().get("top_k", cfg.top_k))

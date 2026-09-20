"""The agent: Claude with two tools, deciding which to call.

Why tool calling instead of a hand-written if/else router:
  1. Hybrid questions ("P-4417 failed flatness, what does the SOP say?") work
     without enumerating every combination.
  2. Adding a third tool later costs one function, not a rewritten classifier.
"""

import time

import anthropic
from anthropic import beta_tool

from src import runtime
from src.config import cfg
from src.tools.query_measurements import run_readonly_sql
from src.tools.search_documents import search_docs

client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

SYSTEM = """You answer questions about a manufacturing quality system.

You have two tools:
- search_documents: procedures, standards, calibration intervals, workflows
- query_measurements: actual inspection numbers, pass/fail counts, trends

Rules:
- Always ground answers in tool results. Never invent a tolerance or a count.
- Cite the document title and section for anything from search_documents.
- Show the SQL you ran for anything from query_measurements.
- Some questions need both tools. Use both.
- If the tools do not contain the answer, say so plainly. Do not guess.
"""


@beta_tool
def search_documents(query: str) -> str:
    """Search quality procedures, SOPs and standards by meaning.

    Use this for questions about how something should be done, required
    intervals, tolerances specified in procedures, or workflow steps.

    Args:
        query: What to look for, in natural language.
    """
    hits = search_docs(query)
    if not hits:
        return "No relevant documents found."
    return "\n\n".join(f"[{h.doc_title} section {h.section}]\n{h.content}" for h in hits)


@beta_tool
def query_measurements(sql: str) -> str:
    """Run a read-only SQL SELECT against the inspection database.

    Use this for questions about actual measured values, counts of
    pass/fail results, trends over time, or specific parts and batches.

    Schema:
      parts(part_id, part_number, description, material, drawing_rev)
      inspection_runs(run_id, part_id, machine_id, operator, batch_id, inspected_at)
      measurements(measurement_id, run_id, feature_name, characteristic,
                   nominal, lower_tol, upper_tol, actual, deviation, in_spec)

    Notes:
      - in_spec is already computed. Use it instead of recomputing tolerances.
      - deviation = actual - nominal, already computed.
      - Form characteristics (flatness, position) are unilateral: nominal 0,
        lower_tol 0, upper_tol positive.

    Args:
        sql: A single SELECT statement. No INSERT/UPDATE/DELETE/DDL.
    """
    return run_readonly_sql(sql)


def ask(question: str, log: bool = True) -> dict:
    """Ask a question. Returns the answer plus everything the evals need."""
    started = time.monotonic()

    with runtime.tracing() as trace:
        runner = client.beta.messages.tool_runner(
            model=cfg.claude_model,
            max_tokens=16000,
            system=SYSTEM,
            tools=[search_documents, query_measurements],
            messages=[{"role": "user", "content": question}],
        )

        last = None
        input_tokens = output_tokens = 0
        for message in runner:
            last = message
            # Usage is per-request; the loop may make several.
            input_tokens += message.usage.input_tokens
            output_tokens += message.usage.output_tokens

    if last is None:
        raise RuntimeError("Tool runner produced no messages")

    result = {
        "question": question,
        "answer": "".join(b.text for b in last.content if b.type == "text"),
        "tools_called": trace.tools_called,
        "retrieved": trace.retrieved,
        "contexts": trace.contexts,
        "sql": trace.sql,
        "latency_ms": int((time.monotonic() - started) * 1000),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }

    if log:
        _log(result)
    return result


def _log(result: dict) -> None:
    """Append to request_log. Never let logging break a request."""
    from src.db import connect

    try:
        with connect() as conn:
            conn.execute(
                "INSERT INTO request_log (question, tools_called, generated_sql, answer,"
                " input_tokens, output_tokens, latency_ms) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                (
                    result["question"],
                    result["tools_called"],
                    "\n---\n".join(result["sql"]) or None,
                    result["answer"],
                    result["input_tokens"],
                    result["output_tokens"],
                    result["latency_ms"],
                ),
            )
            conn.commit()
    except Exception as exc:  # noqa: BLE001 - observability must not break the app
        print(f"[warn] request_log write failed: {exc}")


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How often must the CMM be calibrated?"
    out = ask(q)
    print(out["answer"])
    print(f"\n[tools: {out['tools_called']} | {out['latency_ms']}ms "
          f"| {out['input_tokens']}in/{out['output_tokens']}out]")

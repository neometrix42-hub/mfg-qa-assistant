"""The agent: Claude with two tools, deciding which to call.

This is the core of the project. The tool-calling loop is handled by the SDK's
tool runner, so all you write are the tool functions themselves.

Why tool calling instead of a hand-written if/else router? Two reasons, and you
should be able to say both out loud:
  1. Hybrid questions ("part P-4417 failed flatness, what does the SOP say?")
     work without you enumerating every combination.
  2. Adding a third tool later costs one function, not a rewritten classifier.
"""

import time

import anthropic
from anthropic import beta_tool

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
def search_documents(query: str, top_k: int = 5) -> str:
    """Search quality procedures, SOPs and standards by meaning.

    Use this for questions about how something should be done, required
    intervals, tolerances specified in procedures, or workflow steps.

    Args:
        query: What to look for, in natural language.
        top_k: How many chunks to return.
    """
    hits = search_docs(query, top_k)
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

    Args:
        sql: A single SELECT statement. No INSERT/UPDATE/DELETE/DDL.
    """
    return run_readonly_sql(sql)


def ask(question: str, log: bool = True) -> dict:
    """Ask a question. Returns the answer plus metadata for the eval harness.

    TODO (week 4):
      1. Build the runner (skeleton below).
      2. Iterate it, keeping the LAST message.
      3. Collect which tools were called and any generated SQL - the eval
         harness and request_log both need this, so capture it here rather
         than trying to reconstruct it later.
      4. If log=True, INSERT a row into request_log.
    """
    started = time.monotonic()

    runner = client.beta.messages.tool_runner(
        model=cfg.claude_model,
        max_tokens=16000,
        system=SYSTEM,
        tools=[search_documents, query_measurements],
        messages=[{"role": "user", "content": question}],
    )

    last = None
    tools_called: list[str] = []
    for message in runner:
        last = message
        for block in message.content:
            if block.type == "tool_use":
                tools_called.append(block.name)

    if last is None:
        raise RuntimeError("Tool runner produced no messages")

    answer = "".join(b.text for b in last.content if b.type == "text")

    return {
        "question": question,
        "answer": answer,
        "tools_called": tools_called,
        "latency_ms": int((time.monotonic() - started) * 1000),
        "input_tokens": last.usage.input_tokens,
        "output_tokens": last.usage.output_tokens,
    }


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "How often must the CMM be calibrated?"
    result = ask(q)
    print(result["answer"])
    print(f"\n[tools: {result['tools_called']} | {result['latency_ms']}ms]")

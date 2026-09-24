"""Eval runner - the thing that gets you hired.

    python -m evals.run                    one run at the current config
    python -m evals.run --sweep            the config grid
    python -m evals.run --limit 10         first 10 cases (cheap smoke test)
    python -m evals.run --sweep --dry-run  cost estimate, no API calls

Output: evals/results/<timestamp>.json plus a markdown table on stdout.
That table goes at the TOP of your README.
"""

import argparse
import itertools
import json
from datetime import datetime
from pathlib import Path

import yaml

from evals import metrics
from src.config import cfg

RESULTS_DIR = Path(__file__).resolve().parent / "results"
GOLDEN_SET = Path(__file__).resolve().parent / "golden_set.yaml"

# chunk_size is in TOKENS and caps at 256 because all-MiniLM-L6-v2 truncates
# there. Sweeping 512/1024 with this model measures nothing useful: the chunk
# text still reaches Claude, but only its first 256 tokens influenced the
# embedding that retrieved it.
SWEEP = {
    "chunk_size": [128, 192, 256],
    "chunk_overlap": [0, 32],
    "top_k": [3, 5, 10],
}

# Trimmed grid - the full product is 18 configs, which is more API spend than
# this project justifies. These six vary each axis while holding the others.
SWEEP_CONFIGS = [
    {"chunk_size": 128, "chunk_overlap": 0, "top_k": 5},
    {"chunk_size": 192, "chunk_overlap": 32, "top_k": 5},
    {"chunk_size": 256, "chunk_overlap": 32, "top_k": 3},
    {"chunk_size": 256, "chunk_overlap": 32, "top_k": 5},
    {"chunk_size": 256, "chunk_overlap": 32, "top_k": 10},
    {"chunk_size": 256, "chunk_overlap": 0, "top_k": 5},
]


def load_golden_set(limit: int | None = None) -> list[dict]:
    cases = yaml.safe_load(GOLDEN_SET.read_text(encoding="utf-8")) or []
    return cases[:limit] if limit else cases


def _run_reference_sql(sql: str):
    from src.db import connect_readonly

    with connect_readonly() as conn:
        return conn.execute(sql).fetchall()


def evaluate_case(case: dict, config: dict, client) -> dict:
    """Run one golden-set case and score it."""
    from src.agent import ask

    result = ask(case["question"], log=False)
    expected = case.get("expected_chunks") or []
    k = config["top_k"]

    row = {
        "id": case["id"],
        "type": case["type"],
        "recall": metrics.recall_at_k(result["retrieved"], expected, k),
        "mrr": metrics.mrr(result["retrieved"], expected),
        "contains": metrics.contains_rate(result["answer"], case.get("must_contain") or []),
        "latency_ms": result["latency_ms"],
        "input_tokens": result["input_tokens"],
        "output_tokens": result["output_tokens"],
        "sql_executed": len(result["sql"]),
        "sql_match": None,
        "faithful": None,
        "judge": "",
        "error": None,
    }

    # SQL correctness: pass if ANY query the model ran matches the reference.
    if reference := case.get("reference_sql"):
        try:
            expected_rows = _run_reference_sql(reference)
            row["sql_match"] = any(
                metrics.result_match(_run_reference_sql(s), expected_rows)
                for s in result["sql"]
            )
        except Exception as exc:  # noqa: BLE001 - a bad reference_sql is YOUR bug
            row["error"] = f"reference_sql failed: {exc}"
            row["sql_match"] = False

    # Faithfulness only applies where context was actually retrieved.
    if result["contexts"]:
        verdict, reasoning = metrics.faithfulness(
            result["answer"], "\n\n".join(result["contexts"]), client, cfg.judge_model
        )
        row["faithful"], row["judge"] = verdict, reasoning

    return row


def run_once(config: dict, cases: list[dict], client, rebuild: bool = True) -> dict:
    """Run the full golden set under one config and aggregate."""
    from src import runtime
    from src.ingest.chunk import build_index

    if rebuild:
        # Chunking changes the chunks, so the embeddings MUST be rebuilt.
        # Skipping this is the classic way to produce a sweep that measures
        # nothing: every row would score against whatever index happened to
        # be loaded last.
        print(f"  rebuilding index at chunk_size={config['chunk_size']} "
              f"overlap={config['chunk_overlap']}...")
        build_index(chunk_size=config["chunk_size"], overlap=config["chunk_overlap"])

    rows = []
    with runtime.override(top_k=config["top_k"]):
        for i, case in enumerate(cases, 1):
            print(f"  [{i}/{len(cases)}] {case['id']}", flush=True)
            rows.append(evaluate_case(case, config, client))

    doc_rows = [r for r in rows if r["type"] in ("doc", "hybrid")]
    sql_rows = [r for r in rows if r["sql_match"] is not None]
    judged = [r for r in rows if r["faithful"] is not None]
    latencies = [r["latency_ms"] for r in rows]

    total_in = sum(r["input_tokens"] for r in rows)
    total_out = sum(r["output_tokens"] for r in rows)
    cost = (total_in / 1e6) * cfg.input_price_per_mtok + \
           (total_out / 1e6) * cfg.output_price_per_mtok

    return {
        "config": config,
        "model": cfg.claude_model,
        "judge_model": cfg.judge_model,
        "n_cases": len(rows),
        f"recall@{config['top_k']}": metrics.mean([r["recall"] for r in doc_rows]),
        "mrr": metrics.mean([r["mrr"] for r in doc_rows]),
        "sql_match": metrics.mean([float(r["sql_match"]) for r in sql_rows]),
        "contains": metrics.mean([r["contains"] for r in rows]),
        "faithful": metrics.mean([float(r["faithful"]) for r in judged]),
        "p50_ms": metrics.percentile(latencies, 0.50),
        "p95_ms": metrics.percentile(latencies, 0.95),
        "cost_per_query": cost / max(1, len(rows)),
        "total_cost": cost,
        "cases": rows,
    }


def to_markdown(results: list[dict]) -> str:
    """Render results as the markdown table that goes in the README."""
    header = ("| config | recall | MRR | sql_match | contains | faithful | "
              "p50 ms | p95 ms | $/query |")
    divider = "|---|---|---|---|---|---|---|---|---|"

    def score(r: dict) -> float:
        recall = next(v for k, v in r.items() if k.startswith("recall@"))
        return recall + r["mrr"] + r["sql_match"] + r["contains"]

    best = max(results, key=score) if results else None
    lines = [header, divider]
    for r in results:
        c = r["config"]
        label = f"cs={c['chunk_size']} ov={c['chunk_overlap']} k={c['top_k']}"
        recall = next(v for k, v in r.items() if k.startswith("recall@"))
        cells = [
            label, f"{recall:.2f}", f"{r['mrr']:.2f}", f"{r['sql_match']:.2f}",
            f"{r['contains']:.2f}", f"{r['faithful']:.2f}",
            f"{r['p50_ms']:.0f}", f"{r['p95_ms']:.0f}", f"${r['cost_per_query']:.4f}",
        ]
        if r is best and len(results) > 1:
            cells = [f"**{v}**" for v in cells]
        lines.append("| " + " | ".join(cells) + " |")

    if best and len(results) > 1:
        c = best["config"]
        lines.append("")
        lines.append(f"Best: chunk_size={c['chunk_size']}, overlap={c['chunk_overlap']}, "
                     f"top_k={c['top_k']} — chosen on measured recall and answer accuracy, "
                     f"not preference.")
    return "\n".join(lines)


RETRIEVAL_MODES = ("vector", "keyword", "hybrid")


def run_retrieval_only(cases: list[dict], top_ks=(3, 5, 10)) -> int:
    """Compare retrieval strategies. No API calls, so this costs nothing.

    Retrieval is the half of a RAG system you can tune for free. Do it before
    spending anything on the agent: a fix that does not improve recall will not
    improve answers either.
    """
    from src import runtime
    from src.tools.search_documents import search_docs

    scored = [c for c in cases if c.get("expected_chunks")]
    if not scored:
        print("No cases have expected_chunks; nothing to measure.")
        return 1

    print(f"{len(scored)} cases with expected chunks, no API calls\n")
    results = []
    for mode in RETRIEVAL_MODES:
        largest = max(top_ks)
        retrieved = []
        with runtime.override(search_mode=mode):
            for case in scored:
                hits = search_docs(case["question"], top_k=largest)
                retrieved.append(([h.citation for h in hits], case["expected_chunks"]))
        row = {"mode": mode, "mrr": metrics.mean([metrics.mrr(g, e) for g, e in retrieved])}
        for k in top_ks:
            row[f"recall@{k}"] = metrics.mean(
                [metrics.recall_at_k(g, e, k) for g, e in retrieved]
            )
        row["_detail"] = retrieved
        results.append(row)

    header = "| mode | " + " | ".join(f"recall@{k}" for k in top_ks) + " | MRR |"
    print(header)
    print("|---" * (len(top_ks) + 2) + "|")
    best = max(results, key=lambda r: (r[f"recall@{top_ks[0]}"], r["mrr"]))
    for r in results:
        cells = [r["mode"]] + [f"{r[f'recall@{k}']:.2f}" for k in top_ks] + [f"{r['mrr']:.3f}"]
        if r is best:
            cells = [f"**{c}**" for c in cells]
        print("| " + " | ".join(cells) + " |")

    # Per-case ranks, so a regression is attributable rather than just visible.
    print(f"\n{'case':<18}" + "".join(f"{m:>10}" for m in RETRIEVAL_MODES))
    for i, case in enumerate(scored):
        ranks = []
        for r in results:
            got, exp = r["_detail"][i]
            rank = next((j + 1 for j, g in enumerate(got) if g in exp), None)
            ranks.append(str(rank) if rank else "-")
        print(f"{case['id']:<18}" + "".join(f"{x:>10}" for x in ranks))

    print(f"\nBest: {best['mode']}")
    return 0


def verify_golden_set(cases: list[dict]) -> int:
    """Execute every reference_sql and report. No API calls, so this is free.

    Run it whenever you add cases or change the data generator. A reference_sql
    that silently returns the wrong rows makes sql_match measure nothing, and
    you would never notice from the results table.
    """
    ids = [c.get("id") for c in cases]
    problems = [f"duplicate id: {i}" for i in {x for x in ids if ids.count(x) > 1}]
    problems += [f"missing id or question: {c}" for c in cases
                 if not c.get("id") or not c.get("question")]
    problems += [f"{c['id']}: unknown type {c.get('type')!r}" for c in cases
                 if c.get("type") not in ("doc", "data", "hybrid")]

    print(f"{len(cases)} cases: "
          f"{sum(1 for c in cases if c['type'] == 'doc')} doc, "
          f"{sum(1 for c in cases if c['type'] == 'data')} data, "
          f"{sum(1 for c in cases if c['type'] == 'hybrid')} hybrid\n")

    for case in cases:
        if not (sql := case.get("reference_sql")):
            continue
        try:
            rows = _run_reference_sql(sql)
            preview = str(rows[0]) if rows else "(no rows)"
            flag = "  <- returns nothing; is that really the expected answer?" if not rows else ""
            print(f"OK    {case['id']:<20} {len(rows):>4} row(s)  {preview[:60]}{flag}")
            if not rows:
                problems.append(f"{case['id']}: reference_sql returned no rows")
        except Exception as exc:  # noqa: BLE001 - reporting, not handling
            print(f"FAIL  {case['id']:<20} {exc}")
            problems.append(f"{case['id']}: {exc}")

    if problems:
        print("\nproblems:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\nall reference SQL verified.")
    return 0


def estimate_cost(n_cases: int, n_configs: int) -> float:
    """Rough estimate before spending anything. Assumes ~6k in / ~700 out per
    case including tool round-trips, plus one judge call."""
    per_case = (6000 / 1e6) * cfg.input_price_per_mtok + (700 / 1e6) * cfg.output_price_per_mtok
    return per_case * n_cases * n_configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="run the config grid")
    parser.add_argument("--limit", type=int, help="only the first N cases")
    parser.add_argument("--dry-run", action="store_true", help="estimate cost, call nothing")
    parser.add_argument("--verify", action="store_true",
                        help="check every reference_sql executes; no API calls, no cost")
    parser.add_argument("--retrieval-only", action="store_true",
                        help="compare vector/keyword/hybrid retrieval; no API calls, no cost")
    parser.add_argument("--yes", action="store_true", help="skip the cost confirmation")
    args = parser.parse_args()

    cases = load_golden_set(args.limit)
    if not cases:
        raise SystemExit("golden_set.yaml is empty - write the questions first.")

    if args.verify:
        raise SystemExit(verify_golden_set(cases))

    if args.retrieval_only:
        raise SystemExit(run_retrieval_only(cases))

    configs = SWEEP_CONFIGS if args.sweep else [{
        "chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap, "top_k": cfg.top_k,
    }]

    estimate = estimate_cost(len(cases), len(configs))
    print(f"{len(cases)} cases x {len(configs)} config(s) = "
          f"{len(cases) * len(configs)} runs, est. ${estimate:.2f} "
          f"({cfg.claude_model}, judge {cfg.judge_model})")

    if args.dry_run:
        return
    if not args.yes and input("proceed? [y/N] ").strip().lower() != "y":
        return

    import anthropic

    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)

    results = []
    for i, config in enumerate(configs, 1):
        print(f"\nconfig {i}/{len(configs)}: {config}")
        results.append(run_once(config, cases, client))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")

    print("\n" + to_markdown(results))
    print(f"\nactual spend: ${sum(r['total_cost'] for r in results):.2f}")
    print(f"written to {out}")


def _full_grid() -> list[dict]:
    """The untrimmed product, if you ever want all 18."""
    keys = list(SWEEP)
    return [dict(zip(keys, combo, strict=True))
            for combo in itertools.product(*(SWEEP[k] for k in keys))]


if __name__ == "__main__":
    main()

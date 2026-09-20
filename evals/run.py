"""Eval runner - the thing that gets you hired.

Run:   python -m evals.run              single run, current config
       python -m evals.run --sweep      the full grid

Output: evals/results/<timestamp>.json plus a markdown table on stdout.
The markdown table goes at the TOP of your README.

YOU WRITE THIS. It is the most valuable file in the repo - do not outsource it,
because every interview question worth winning comes from here.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

RESULTS_DIR = Path(__file__).resolve().parent / "results"
GOLDEN_SET = Path(__file__).resolve().parent / "golden_set.yaml"

# The grid. 3 chunk sizes x 2 overlaps x 3 top_k = 18 configs; trim to ~6 for cost.
#
# chunk_size is in TOKENS and caps at 256 because all-MiniLM-L6-v2 truncates
# there. Sweeping 512/1024 with this model measures nothing useful: the chunk
# text still gets returned to Claude, but only its first 256 tokens influenced
# the embedding that retrieved it. Either stay at/below 256, or switch to a
# longer-context embedding model and update vector(384) in the schema to match.
SWEEP = {
    "chunk_size": [128, 192, 256],
    "chunk_overlap": [0, 32],
    "top_k": [3, 5, 10],
}


def load_golden_set() -> list[dict]:
    return yaml.safe_load(GOLDEN_SET.read_text(encoding="utf-8"))


def run_once(config: dict) -> dict:
    """Run the full golden set under one config, return aggregate metrics.

    TODO:
      1. Re-ingest documents at this chunk_size/overlap (chunking changes
         the chunks, so the embeddings must be rebuilt - do not skip this).
      2. For each golden-set case:
           - call agent.ask(question)
           - doc/hybrid: recall@k and MRR against expected_chunks
           - data: execute reference_sql, compare with result_match
           - all: contains_rate, faithfulness
           - record latency and tokens
      3. Aggregate into a dict of metric -> value.

    Cost note: 50 questions x 6 configs = 300 Claude calls. Run ONE config
    end to end first and check the cost before launching the full sweep.
    """
    raise NotImplementedError("Week 5: implement the eval loop")


def to_markdown(results: list[dict]) -> str:
    """Render results as a markdown table for the README.

    Suggested columns:
      config | recall@5 | MRR | sql_match | contains | faithful | p95 ms | $/query

    TODO: implement. Bold the winning row.
    """
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sweep", action="store_true", help="run the full config grid")
    args = parser.parse_args()

    configs = _expand_sweep() if args.sweep else [_current_config()]
    results = [run_once(c) for c in configs]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RESULTS_DIR / f"{stamp}.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(to_markdown(results))
    print(f"\nWritten to {out}")


def _current_config() -> dict:
    from src.config import cfg

    return {"chunk_size": cfg.chunk_size, "chunk_overlap": cfg.chunk_overlap, "top_k": cfg.top_k}


def _expand_sweep() -> list[dict]:
    """TODO: itertools.product over SWEEP. Trim to ~6 configs to control cost."""
    raise NotImplementedError


if __name__ == "__main__":
    main()

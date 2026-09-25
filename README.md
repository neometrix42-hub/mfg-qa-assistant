# Manufacturing QA Assistant

A question-answering system over manufacturing inspection data and quality documents.
Claude chooses between hybrid search across quality procedures and generated SQL over a
measurements database, then answers with citations. Retrieval is measured against a
golden set: hybrid search reaches **recall@10 of 1.00**, against 0.93 for vector search
alone.

> **Status:** ingestion, hybrid retrieval and the eval harness run end to end, with 64
> tests passing. Agent-level evaluation needs an `ANTHROPIC_API_KEY`; deployment is
> pending. See [PROJECT_SPEC.md](PROJECT_SPEC.md) for the full design.

---

## Results

### Retrieval: vector vs keyword vs hybrid

Measured over 15 golden-set cases with known correct chunks.
Reproduce with `python -m evals.run --retrieval-only` (no API calls, no cost).

| mode | recall@3 | recall@5 | recall@10 | MRR |
|---|---|---|---|---|
| vector only | 0.80 | 0.87 | 0.93 | **0.758** |
| keyword only | 0.87 | 0.93 | 0.93 | 0.691 |
| **hybrid (RRF)** | **0.87** | **0.93** | **1.00** | 0.746 |

**Hybrid is the default.** It wins recall at every depth and reaches perfect
recall@10, at the cost of 0.012 MRR — it occasionally demotes an easy rank-1
result to rank 2. With `top_k=5` the model sees the chunk either way, so recall
is the metric that decides the answer and MRR is the one to trade.

**What hybrid fixed, and what it cost:**

| case | vector | hybrid | |
|---|---|---|---|
| `hybrid_001` "Part P-4417 failed a flatness check…" | not in top 10 | **4** | ✅ Part numbers are exact tokens with no useful embedding |
| `doc_004` "CMM that fails its acceptance criteria" | 4 | **3** | ✅ Was matching the FAI "Acceptance" section instead |
| `hybrid_003` non-conformance records retention | 8 | 9 | ➖ Near-miss either way |
| `doc_003` calibration certificate retention | **1** | 2 | ⚠️ Demoted by fusion |

One tuning step mattered: `ts_rank_cd` ignores document length by default, so a
long chunk mentioning a term in passing outranked a short chunk entirely about
it. Adding length normalisation moved recall@10 from 0.93 to 1.00.

### Agent quality

Pending — needs an `ANTHROPIC_API_KEY`. Run `python -m evals.run --sweep`.

| config | sql_match | contains | faithful | p95 ms | $/query |
|---|---|---|---|---|---|
| _pending_ | – | – | – | – | – |

Golden set: 24 cases so far (14 document, 7 data, 3 hybrid), including two
deliberately unanswerable ones. See [`evals/golden_set.yaml`](evals/golden_set.yaml).

---

## Architecture

```
                 +----------------------+
  user question  |   Claude (tool use)  |
  -------------> |   claude-opus-5      |
                 +----------+-----------+
                            | picks one or both
             +--------------+--------------+
             v                             v
    search_documents(query)      query_measurements(sql)
             |                             |
             v                             v
     pgvector similarity          read-only SQL, validated
     over doc_chunks              over parts/runs/measurements
             |                             |
             +--------------+--------------+
                            v
                grounded answer + citations
                            |
                            v
                     request_log (observability)
```

Both tools hit one Postgres database. The vector store and the business data live
together, so relational queries and semantic search share a single source of truth.

**Two retrieval modes, chosen by the model.** Rather than a hand-written router,
Claude sees two tools and decides. Hybrid questions ("part P-4417 failed flatness,
what does the SOP require?") work without enumerating every combination.

---

## SQL safety

LLM-generated SQL runs behind three independent layers:

1. **Read-only role** — `qa_reader` holds `SELECT` on three tables only, with a 5s
   `statement_timeout`. It physically cannot write. ([`sql/002_readonly_role.sql`](sql/002_readonly_role.sql))
2. **Statement validation** — comments stripped, single statement enforced, `SELECT`/`WITH`
   only, forbidden keywords rejected, `LIMIT` injected. ([`src/tools/query_measurements.py`](src/tools/query_measurements.py))
3. **Result truncation** — row and character caps so a wide result cannot exhaust the
   context window.

Validation failures are returned to Claude as tool results rather than raised, so the
model corrects its own SQL.

---

## Setup

Requires Python 3.11+ and Docker.

```bash
pip install -e ".[dev]"
cp .env.example .env          # add your ANTHROPIC_API_KEY for the agent
docker compose up -d          # Postgres + pgvector on :5433, schema applied automatically
python -m src.db              # should print OK
```

Load the data and build the search index:

```bash
python -m src.ingest.load_measurements   # generates 8 parts / ~1,200 runs / ~3,650 measurements
python -m src.ingest.chunk               # SOPs -> chunks -> embeddings -> doc_chunks
```

Both are idempotent — rerun either after changing the generator or the documents.

### Try it without an API key

Retrieval is the half of a RAG system you can develop for free. These commands
touch no paid API:

```bash
python -m evals.run --retrieval-only   # the results table above, reproduced
python -m src.tools.search_documents   # same three questions, vector vs keyword vs hybrid
python -m evals.run --verify           # checks every golden-set reference_sql executes
python -m pytest tests/ -q             # 64 tests
```

### With an API key

```bash
python -m src.agent "How often must the CMM be calibrated?"
python -m src.agent "Which machine has the worst failure rate, and what does the SOP say?"
streamlit run src/ui.py
python -m evals.run --sweep            # full agent eval; prints a cost estimate first
uvicorn src.api:app --reload           # HTTP API on :8000
```

`--sweep` shows the estimated spend and asks before calling anything. Setting
`JUDGE_MODEL=claude-haiku-4-5` in `.env` cuts the cost of the faithfulness judge.

### Day to day

The database runs in Docker and does not start itself:

```bash
docker compose up -d     # after a reboot
docker compose down      # stop it; data persists in the named volume
docker compose down -v   # stop AND wipe the data, forcing a clean reingest
```

### Configuration

Everything tunable lives in `.env` and is read by [`src/config.py`](src/config.py):

| Variable | Default | Effect |
|---|---|---|
| `SEARCH_MODE` | `hybrid` | `vector`, `keyword` or `hybrid` |
| `TOP_K` | `5` | Chunks passed to the model |
| `CHUNK_SIZE` | `256` | Tokens per chunk. **Do not exceed 256** — the embedding model truncates there |
| `CHUNK_OVERLAP` | `32` | Token overlap between chunks |
| `RRF_K` | `60` | Reciprocal Rank Fusion constant |
| `CLAUDE_MODEL` | `claude-opus-5` | Model for the agent |

Changing `CHUNK_SIZE` or `CHUNK_OVERLAP` requires rerunning `python -m src.ingest.chunk`,
since the embeddings are rebuilt from the chunks.

---

## Data

**Structured** — a synthetic dimensional inspection dataset (8 parts, ~1,200
inspection runs, ~3,650 measurements over Jan–Sep 2026), generated by
[`load_measurements.py`](src/ingest/load_measurements.py).

The obvious public candidates (NASA C-MAPSS, UCI SECOM, Bosch Production Line)
are anonymised process-sensor streams, not dimensional metrology — relabelling
`sensor_147` as "bore diameter" would be dressing data up as something it isn't.
Modelling the process instead is both more honest and a better demonstration:
the generator includes tool-wear drift within a batch, per-machine measurement
bias, unilateral form tolerances, and occasional process excursions, at a target
capability around Cpk 1.33.

Generation is seeded and deterministic, so the eval set's reference answers stay
stable.

One pattern is deliberately planted: **ARM-01 (articulated arm) shows a 23.5%
out-of-tolerance rate against 1.6–5.3% for the bridge CMMs** — a measurement
system problem, not a manufacturing one. Finding it is a good test of whether
the assistant is actually useful.

**Documents** — the SOP corpus in `data/documents/` is **synthetic**, authored by me
and modelled on standard industry practice from my work in metrology and dimensional
inspection. It contains no proprietary or customer data.

---

## What I learned / what I'd do differently

<!-- Three honest bullets, written at the end. Name a real limitation.
     "What does it get wrong?" is the interview question juniors fail. -->

- _pending_
- _pending_
- _pending_

# Manufacturing QA Assistant — Project Spec

**One-line pitch (use this verbatim on your resume and in interviews):**

> A question-answering system over manufacturing inspection data and quality documents. Claude picks between semantic search over SOPs and generated SQL over a measurements database, then answers with citations. Evaluated against a 50-question golden set across 6 retrieval configurations.

That sentence is the whole point of this project. Everything below exists to make it true.

---

## 0. Why this project and not a normal RAG chatbot

A RAG chatbot over PDFs is the single most common portfolio project in the world right now. It is no longer a signal.

This project differs in three ways a hiring manager will notice in under thirty seconds:

1. **Two retrieval modes, chosen by the model.** Documents *and* structured data, with tool calling deciding which to use. A real architecture, not a tutorial.
2. **A domain almost nobody in the applicant pool has.** You can write a realistic inspection SOP. A CS grad cannot. The document corpus itself is evidence of your background.
3. **Measured, not claimed.** An eval harness with a results table. This is the rarest thing in a junior portfolio.

Keep all three. If you have to cut scope, cut features — never cut the evals.

---

## 1. What you are building

A FastAPI service plus a small UI that answers questions like:

| Question | Path taken |
|---|---|
| "What's the calibration interval for the CMM?" | Document search |
| "Which parts failed flatness tolerance in March?" | SQL over measurements |
| "Part 4417 failed flatness — what does our SOP say to do next?" | Both |

Claude sees two tools and decides which to call. That decision *is* the agent.

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

Both tools hit **one Postgres database**. That is deliberate — the vector store and the business data live together, and you get to show SQL rather than hide it behind a toy vector library.

---

## 2. Stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.11+ | You already have it |
| API | FastAPI | 15-minute learning curve, industry standard |
| Database | Postgres 16 + `pgvector` | One DB for vectors *and* relational data |
| Embeddings | `sentence-transformers`, `all-MiniLM-L6-v2` (384-dim) | Runs locally, free, no API cost |
| LLM | Anthropic API, `claude-opus-5` | Tool calling + generation |
| UI | Streamlit | One file. Do not build a React frontend. |
| Container | Docker + docker-compose | Postgres + app in one command |
| Evals | pytest + a custom runner | Your differentiator |

**On cost:** local embeddings are free. Your only spend is Claude API calls — roughly a few dollars total for development plus eval runs. If you want to cut that further, `claude-sonnet-5` or `claude-haiku-4-5` are drop-in model-string swaps; that's your call to make, not a default I'll pick for you.

---

## 3. Data — public only

> **Do not use Neometrix client data.** Real customer scan or inspection data is not yours to publish, and a hiring manager who spots it will read it as a judgment failure — far worse than a thinner project. This is non-negotiable.

**Structured (pick one):**
- NASA C-MAPSS turbofan degradation — sensor time series, clean, well documented
- UCI SECOM — semiconductor manufacturing, 591 sensors with pass/fail labels
- Bosch Production Line Performance (Kaggle) — large, genuinely industrial

Reshape whichever you pick into the schema below. The reshaping is itself ETL work worth describing.

**Documents — write them yourself.**

This is the part only you can do. Author 15–25 short documents (1–3 pages each):
- Inspection procedures (CMM setup, first-article inspection, in-process checks)
- Calibration schedules and gauge R&R procedure
- A GD&T quick-reference in your own words
- Non-conformance handling / disposition workflow
- Equipment operating notes and known failure modes

Write them the way your industry actually writes them, with section numbers and revision headers. **State clearly in the README that this corpus is synthetic, authored by you, and modelled on standard industry practice.** That is honest, sidesteps copyright entirely, and demonstrates domain knowledge more convincingly than any scraped PDF.

---

## 4. Database schema

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE parts (
    part_id      TEXT PRIMARY KEY,
    part_number  TEXT NOT NULL,
    description  TEXT,
    material     TEXT,
    drawing_rev  TEXT
);

CREATE TABLE inspection_runs (
    run_id        BIGSERIAL PRIMARY KEY,
    part_id       TEXT REFERENCES parts(part_id),
    machine_id    TEXT NOT NULL,
    operator      TEXT,
    batch_id      TEXT,
    inspected_at  TIMESTAMPTZ NOT NULL
);

CREATE TABLE measurements (
    measurement_id BIGSERIAL PRIMARY KEY,
    run_id         BIGINT REFERENCES inspection_runs(run_id),
    feature_name   TEXT NOT NULL,        -- 'bore_dia_A'
    characteristic TEXT NOT NULL,        -- 'diameter' | 'flatness' | 'position'
    nominal        NUMERIC NOT NULL,
    lower_tol      NUMERIC NOT NULL,     -- signed, e.g. -0.05
    upper_tol      NUMERIC NOT NULL,     -- signed, e.g. +0.05
    actual         NUMERIC NOT NULL,
    deviation      NUMERIC GENERATED ALWAYS AS (actual - nominal) STORED,
    in_spec        BOOLEAN GENERATED ALWAYS AS
                   (actual >= nominal + lower_tol AND actual <= nominal + upper_tol) STORED
);

CREATE INDEX ON measurements (run_id);
CREATE INDEX ON measurements (characteristic, in_spec);

CREATE TABLE doc_chunks (
    chunk_id    BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,
    doc_title   TEXT NOT NULL,
    section     TEXT,
    content     TEXT NOT NULL,
    token_count INT,
    embedding   vector(384)
);

CREATE INDEX ON doc_chunks USING hnsw (embedding vector_cosine_ops);

CREATE TABLE request_log (
    id             BIGSERIAL PRIMARY KEY,
    asked_at       TIMESTAMPTZ DEFAULT now(),
    question       TEXT NOT NULL,
    tools_called   TEXT[],
    generated_sql  TEXT,
    retrieved_ids  BIGINT[],
    answer         TEXT,
    input_tokens   INT,
    output_tokens  INT,
    latency_ms     INT,
    error          TEXT
);
```

The two generated columns are a small thing that reads as competence: the spec logic lives in the database, not scattered through Python.

---

## 5. Repo structure

```
mfg-qa-assistant/
├── README.md                 <- the actual deliverable (see section 9)
├── docker-compose.yml
├── Dockerfile
├── pyproject.toml
├── .env.example              <- never commit .env
├── data/
│   ├── documents/            <- your authored SOPs (markdown)
│   └── raw/                  <- downloaded public dataset
├── sql/
│   ├── 001_schema.sql
│   └── 002_readonly_role.sql
├── src/
│   ├── config.py
│   ├── db.py
│   ├── ingest/
│   │   ├── load_measurements.py
│   │   ├── chunk.py
│   │   └── embed.py
│   ├── tools/
│   │   ├── search_documents.py
│   │   └── query_measurements.py
│   ├── agent.py              <- the tool-calling loop
│   ├── api.py                <- FastAPI
│   └── ui.py                 <- Streamlit
├── evals/
│   ├── golden_set.yaml
│   ├── run.py
│   ├── metrics.py
│   └── results/
└── tests/
```

---

## 6. The agent — tool calling

Use the Anthropic Python SDK's tool runner. It handles the loop so you write only the tool functions.

```python
# src/agent.py
import anthropic
from anthropic import beta_tool

from src.tools.search_documents import search_docs
from src.tools.query_measurements import run_readonly_sql

client = anthropic.Anthropic()

SYSTEM = """You answer questions about a manufacturing quality system.

You have two tools:
- search_documents: for procedures, standards, calibration intervals, workflows
- query_measurements: for actual inspection numbers, pass/fail counts, trends

Rules:
- Always ground answers in tool results. Never invent a tolerance or a count.
- Cite the document title and section for anything from search_documents.
- Show the SQL you ran for anything from query_measurements.
- If the tools do not contain the answer, say so plainly.
"""


@beta_tool
def search_documents(query: str, top_k: int = 5) -> str:
    """Search quality procedures, SOPs and standards by meaning.

    Args:
        query: What to look for, in natural language.
        top_k: How many chunks to return.
    """
    hits = search_docs(query, top_k)
    return "\n\n".join(
        f"[{h.doc_title} section {h.section}]\n{h.content}" for h in hits
    )


@beta_tool
def query_measurements(sql: str) -> str:
    """Run a read-only SQL SELECT against the inspection database.

    Schema:
      parts(part_id, part_number, description, material, drawing_rev)
      inspection_runs(run_id, part_id, machine_id, operator, batch_id, inspected_at)
      measurements(measurement_id, run_id, feature_name, characteristic,
                   nominal, lower_tol, upper_tol, actual, deviation, in_spec)

    Args:
        sql: A single SELECT statement. No INSERT/UPDATE/DELETE/DDL.
    """
    return run_readonly_sql(sql)


def ask(question: str) -> str:
    runner = client.beta.messages.tool_runner(
        model="claude-opus-5",
        max_tokens=16000,
        system=SYSTEM,
        tools=[search_documents, query_measurements],
        messages=[{"role": "user", "content": question}],
    )
    last = None
    for message in runner:
        last = message
    return "".join(b.text for b in last.content if b.type == "text")
```

**Interview note:** be ready to explain why tool calling beats a hand-written if/else router. Answer: the model handles hybrid questions without you enumerating them, and adding a third tool later costs one function instead of a rewritten classifier.

---

## 7. SQL safety — do not skip this

Letting an LLM write SQL against your database is the part interviewers will probe. Have a real answer.

**Layer 1 — a read-only database role:**

```sql
CREATE ROLE qa_reader LOGIN PASSWORD 'changeme';
GRANT CONNECT ON DATABASE mfgqa TO qa_reader;
GRANT USAGE ON SCHEMA public TO qa_reader;
GRANT SELECT ON parts, inspection_runs, measurements TO qa_reader;
ALTER ROLE qa_reader SET statement_timeout = '5s';
```

**Layer 2 — validate before executing:**
- Strip comments, normalise whitespace
- Must begin with `SELECT` or `WITH`
- Reject any `;` other than a single trailing one (blocks stacked statements)
- Reject `INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|GRANT|COPY` as keywords
- Append `LIMIT 500` if no `LIMIT` is present
- Run `EXPLAIN <sql>` first — if it fails, return the error to Claude as the tool result rather than raising. Claude will correct its own SQL, and that self-correction loop is worth demonstrating.

**Layer 3 — truncate results** to a sane number of rows before they enter the context window.

Defence in depth, three layers, each independently sufficient. Say exactly that in the interview.

---

## 8. The eval harness — your differentiator

This is the section that gets you hired. Budget real time for it.

### Golden set — `evals/golden_set.yaml`

Write 50 questions. Split roughly 20 document / 20 data / 10 hybrid.

```yaml
- id: doc_007
  question: "How often must a CMM be calibrated for production inspection?"
  type: doc
  expected_chunks: ["sop_calibration#3.2"]
  must_contain: ["12 months"]

- id: data_004
  question: "How many parts failed flatness tolerance in March 2026?"
  type: data
  reference_sql: >
    SELECT COUNT(*) FROM measurements m
    JOIN inspection_runs r USING (run_id)
    WHERE m.characteristic = 'flatness'
      AND m.in_spec = false
      AND r.inspected_at >= '2026-03-01'
      AND r.inspected_at <  '2026-04-01';

- id: hybrid_002
  question: "Part P-4417 failed flatness last week. What does our procedure require next?"
  type: hybrid
  expected_chunks: ["sop_nonconformance#2.1"]
  must_contain: ["non-conformance", "quarantine"]
```

### Metrics

**Retrieval (doc questions)**
- `recall@k` — fraction of questions where an expected chunk appears in the top k
- `MRR` — mean reciprocal rank of the first correct chunk

**SQL (data questions)**
- `execution_rate` — fraction where generated SQL runs without error
- `result_match` — fraction where the result set equals running `reference_sql`

**Answer quality (all)**
- `contains_rate` — fraction where every `must_contain` string appears
- `faithfulness` — a second Claude call judging whether the answer is supported by the retrieved context. Log the judge's reasoning; you will be asked about LLM-as-judge limitations, and *"I know it's a weak judge, so I pair it with exact-match checks"* is the right answer.

**Operations**
- p50 / p95 latency, mean tokens, cost per query

### The config sweep — the actual money shot

Run the full golden set across a grid:

| chunk_size | overlap | top_k |
|---|---|---|
| 256 / 512 / 1024 | 0 / 15% | 3 / 5 / 10 |

Write results to `evals/results/<timestamp>.json` and emit a markdown table. **Put that table at the top of your README.**

This is what turns "I built a RAG app" into "I measured six configurations across fifty questions and chose one on evidence." Nobody at your level does this. Do it.

---

## 9. The README — the real deliverable

Most people will read your README and nothing else. Structure:

1. **The one-line pitch** from the top of this spec
2. **Results table first** — before architecture, before setup. Numbers above the fold.
3. **One screenshot or GIF** of a hybrid question being answered
4. **Architecture diagram** (the ASCII one above is fine)
5. **What I learned / what I'd do differently** — 3 honest bullets. Name a real limitation. This reads as senior.
6. **Setup** — must be `docker compose up` plus one ingest command. If a reviewer can't run it in two minutes, they won't.
7. **A clear note** that the document corpus is synthetic and authored by you

---

## 10. Build milestones

| Week | Deliverable | Done when |
|---|---|---|
| 2 | Repo, docker-compose, schema, structured data loaded | You can answer a real question in `psql` |
| 3 | Documents written, chunked, embedded, vector search working | Semantic search returns sane chunks — **no LLM yet** |
| 4 | Tool-calling agent end to end | Claude correctly picks the tool for all three question types |
| 5 | Eval harness + golden set + first results | `python -m evals.run` prints a table |
| 6 | Config sweep, tuning, README with numbers | Best config chosen *on evidence* |
| 7 | Streamlit UI, Docker, deployed | A stranger can open a URL and use it |
| 8 | Observability, polish, 2-min demo video | Link on your resume |

Week 3 having no LLM in it is intentional. Retrieval quality is where RAG projects live or die, and you cannot debug it through a chat interface.

---

## 11. Do NOT build these

Every one of these has killed a portfolio project by eating the time the evals needed:

- Authentication or multi-user support
- A React/Next.js frontend
- Kubernetes
- Fine-tuning anything
- Conversation memory or multi-turn chat history
- More than two tools
- Comparing three vector databases
- A second dataset

Scope discipline is itself a hireable signal. If asked why there's no auth: *"It's a single-user demo. Auth would have cost me the eval harness, and the eval harness is the interesting part."*

---

## 12. Interview questions you will be asked

Prepare a 60–90 second answer to each. Rehearse out loud.

1. **"Walk me through what happens when someone asks a question."**
2. **"How do you know it works?"** → the golden set, the metrics, the sweep. This is the one you win on.
3. **"What if Claude writes bad SQL?"** → three-layer defence, plus EXPLAIN errors fed back for self-correction.
4. **"Why chunk size 512?"** → because the sweep said so. Give the numbers.
5. **"What does it get wrong?"** → name a real failure class (multi-hop questions, ambiguous date ranges). Never say "nothing."
6. **"Why should a mechanical engineer build this?"** → *"Because I've spent a year in metrology. I know what question a quality engineer actually asks, and I can tell when the answer is wrong. Most people building these can't."*

Question 5 is the one juniors fail. Knowing your system's limits is the clearest signal of engineering maturity you can send.

---

## 13. Start tomorrow

```bash
mkdir mfg-qa-assistant && cd mfg-qa-assistant
git init
mkdir -p data/documents data/raw sql src/ingest src/tools evals/results tests
```

Then, in order:

1. `docker-compose.yml` with `pgvector/pgvector:pg16`
2. `sql/001_schema.sql` — paste section 4, run it, confirm the tables exist
3. Write **three** SOPs by hand. Not thirty. Three.
4. Chunk and embed them, then query for similarity in raw SQL

You will have working semantic search before you write a single line of LLM code. That is the correct order.

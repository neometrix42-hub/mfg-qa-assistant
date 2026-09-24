-- Manufacturing QA Assistant - schema
-- Runs automatically on first `docker compose up`.

CREATE EXTENSION IF NOT EXISTS vector;

-- ---------------------------------------------------------------- structured

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
    feature_name   TEXT NOT NULL,        -- e.g. 'bore_dia_A'
    characteristic TEXT NOT NULL,        -- 'diameter' | 'flatness' | 'position' | ...
    nominal        NUMERIC NOT NULL,
    lower_tol      NUMERIC NOT NULL,     -- signed, e.g. -0.05
    upper_tol      NUMERIC NOT NULL,     -- signed, e.g. +0.05
    actual         NUMERIC NOT NULL,
    -- Spec logic lives in the database, not scattered through Python.
    deviation      NUMERIC GENERATED ALWAYS AS (actual - nominal) STORED,
    in_spec        BOOLEAN GENERATED ALWAYS AS
                   (actual >= nominal + lower_tol AND actual <= nominal + upper_tol) STORED
);

CREATE INDEX ON inspection_runs (inspected_at);
CREATE INDEX ON measurements (run_id);
CREATE INDEX ON measurements (characteristic, in_spec);

-- ----------------------------------------------------------------- documents

CREATE TABLE doc_chunks (
    chunk_id    BIGSERIAL PRIMARY KEY,
    doc_id      TEXT NOT NULL,          -- 'sop_calibration'
    doc_title   TEXT NOT NULL,          -- 'SOP-014 CMM Calibration'
    section     TEXT,                   -- '3.2'
    content     TEXT NOT NULL,
    token_count INT,
    embedding   vector(384),            -- all-MiniLM-L6-v2
    -- Keyword half of hybrid search. See sql/003_fulltext.sql for why.
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED
);

-- HNSW index for cosine similarity. Build it AFTER bulk-loading for speed.
CREATE INDEX ON doc_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX ON doc_chunks (doc_id, section);

-- ------------------------------------------------------------- observability

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

CREATE INDEX ON request_log (asked_at DESC);

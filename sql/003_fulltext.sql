-- Full-text search column for hybrid retrieval.
--
-- Vector search alone fails on exact tokens: a part number like 'P-4417' has no
-- useful embedding, and near-identical sections across documents ("Records"
-- appears in all three SOPs with different retention periods) collapse into
-- nearly the same vector. Keyword search handles both cases well and fails on
-- the paraphrases that vectors handle well. Hybrid search runs both.
--
-- Written with IF NOT EXISTS so it is safe to apply to a database that already
-- has data, not just a fresh container.

ALTER TABLE doc_chunks
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('english', content)) STORED;

CREATE INDEX IF NOT EXISTS doc_chunks_content_tsv_idx
    ON doc_chunks USING GIN (content_tsv);

-- Layer 1 of the SQL-injection defence: the LLM's connection can only read.
--
-- Even if every check in src/tools/query_measurements.py is bypassed, this role
-- physically cannot write. Be able to explain this in an interview.

CREATE ROLE qa_reader LOGIN PASSWORD 'qa_reader_dev_password';

GRANT CONNECT ON DATABASE mfgqa TO qa_reader;
GRANT USAGE ON SCHEMA public TO qa_reader;

-- SELECT only, and only on the three business tables.
-- Deliberately NOT granted on doc_chunks or request_log.
GRANT SELECT ON parts, inspection_runs, measurements TO qa_reader;

-- A runaway query cannot hold a connection open.
ALTER ROLE qa_reader SET statement_timeout = '5s';

-- Future tables are not automatically readable.
ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE ALL ON TABLES FROM qa_reader;

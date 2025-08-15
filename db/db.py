# db.py
import sqlite3
import logging
from pathlib import Path

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

_DB_CONN = None

_SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

-- ===== CORE: Project & Corpus =====
CREATE TABLE IF NOT EXISTS project (
  id            TEXT PRIMARY KEY,
  name          TEXT NOT NULL,
  description   TEXT,
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS corpus (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS corpus_item (
  id            TEXT PRIMARY KEY,
  corpus_id     TEXT NOT NULL REFERENCES corpus(id) ON DELETE CASCADE,
  uri           TEXT NOT NULL,
  content_hash  TEXT NOT NULL,
  metadata_json TEXT,
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP

  file_path TEXT,                               -- e.g., /data/docs/a.pdf
  original_filename TEXT,
  file_size_bytes INTEGER,
  mime_type TEXT,
  fs_modified_time TEXT,
  storage_backend TEXT,                         -- 'local'|'s3'...
  storage_key TEXT,                             -- blob pointer if copied
  ingested_at TEXT,

  url TEXT,
  url_normalized TEXT,
  http_status_last INTEGER,
  http_etag TEXT,
  http_last_modified TEXT,
  fetched_at TEXT,
  fetch_error TEXT
  
);
CREATE INDEX IF NOT EXISTS idx_corpus_item_corpus ON corpus_item(corpus_id);
CREATE INDEX IF NOT EXISTS idx_corpus_item_hash ON corpus_item(content_hash);

CREATE TABLE IF NOT EXISTS corpus_snapshot (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS corpus_snapshot_item (
  id               TEXT PRIMARY KEY,
  snapshot_id      TEXT NOT NULL REFERENCES corpus_snapshot(id) ON DELETE CASCADE,
  corpus_item_id   TEXT NOT NULL REFERENCES corpus_item(id) ON DELETE RESTRICT,
  content_hash     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_snapshot_item_snapshot ON corpus_snapshot_item(snapshot_id);

-- ===== INDEX BUILD & CONTENT =====
CREATE TABLE IF NOT EXISTS index_build (
  id                          TEXT PRIMARY KEY,
  project_id                  TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
  corpus_snapshot_id          TEXT NOT NULL REFERENCES corpus_snapshot(id) ON DELETE RESTRICT,
  chunking_cfg_version_id     TEXT NOT NULL,
  embedding_cfg_version_id    TEXT NOT NULL,
  vector_params_json          TEXT,
  input_hash                  TEXT NOT NULL,
  created_at                  TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_index_build_input ON index_build(input_hash);

CREATE TABLE IF NOT EXISTS document (
  id              TEXT PRIMARY KEY,
  index_build_id  TEXT NOT NULL REFERENCES index_build(id) ON DELETE CASCADE,
  corpus_item_id  TEXT REFERENCES corpus_item(id) ON DELETE SET NULL,
  title           TEXT,
  meta_json       TEXT
);
CREATE INDEX IF NOT EXISTS idx_document_index_build ON document(index_build_id);

CREATE TABLE IF NOT EXISTS chunk (
  id              TEXT PRIMARY KEY,
  index_build_id  TEXT NOT NULL REFERENCES index_build(id) ON DELETE CASCADE,
  document_id     TEXT NOT NULL REFERENCES document(id) ON DELETE CASCADE,
  span_start      INTEGER NOT NULL,
  span_end        INTEGER NOT NULL,
  text_hash       TEXT NOT NULL,
  text            TEXT NOT NULL,
  embedding_hash  TEXT,
  metadata_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_chunk_index_build ON chunk(index_build_id);
CREATE INDEX IF NOT EXISTS idx_chunk_document ON chunk(document_id);

-- ===== VERSIONED CONFIGS =====
CREATE TABLE IF NOT EXISTS embedding_config_version (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
  parent_id     TEXT REFERENCES embedding_config_version(id) ON DELETE SET NULL,
  version_label TEXT,
  model_name    TEXT NOT NULL,
  dims          INTEGER,
  normalize     INTEGER DEFAULT 1,
  params_json   TEXT,
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS chunking_config_version (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
  parent_id     TEXT REFERENCES chunking_config_version(id) ON DELETE SET NULL,
  version_label TEXT,
  strategy      TEXT,
  size          INTEGER,
  overlap       INTEGER,
  rules_json    TEXT,
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS retriever_config_version (
  id               TEXT PRIMARY KEY,
  project_id       TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
  parent_id        TEXT REFERENCES retriever_config_version(id) ON DELETE SET NULL,
  version_label    TEXT,
  top_k            INTEGER NOT NULL,
  mmr_lambda       REAL,
  filters_json     TEXT,
  hybrid_weights_json TEXT,
  created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS generation_config_version (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
  parent_id     TEXT REFERENCES generation_config_version(id) ON DELETE SET NULL,
  version_label TEXT,
  model_name    TEXT NOT NULL,
  temperature   REAL,
  max_tokens    INTEGER,
  stop_json     TEXT,
  system_prompt TEXT,
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS judge_config_version (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
  parent_id     TEXT REFERENCES judge_config_version(id) ON DELETE SET NULL,
  version_label TEXT,
  judge_model   TEXT NOT NULL,
  rubric_json   TEXT,
  thresholds_json TEXT,
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS prompt_template_version (
  id              TEXT PRIMARY KEY,
  project_id      TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
  parent_id       TEXT REFERENCES prompt_template_version(id) ON DELETE SET NULL,
  version_label   TEXT,
  template_text   TEXT NOT NULL,
  variables_json  TEXT,
  content_hash    TEXT NOT NULL,
  created_at      TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_prompt_tpl_hash ON prompt_template_version(content_hash);

CREATE TABLE IF NOT EXISTS prompt_instance (
  id                    TEXT PRIMARY KEY,
  test_run_id           TEXT NOT NULL,
  run_question_id       TEXT NOT NULL,
  template_version_id   TEXT NOT NULL REFERENCES prompt_template_version(id) ON DELETE RESTRICT,
  rendered_text         TEXT NOT NULL,
  variables_json        TEXT,
  content_hash          TEXT NOT NULL,
  created_at            TEXT DEFAULT CURRENT_TIMESTAMP
);

-- ===== TESTS & QUESTIONS =====
CREATE TABLE IF NOT EXISTS test (
  id            TEXT PRIMARY KEY,
  project_id    TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
  name          TEXT NOT NULL,
  description   TEXT,
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS question (
  id            TEXT PRIMARY KEY,
  test_id       TEXT NOT NULL REFERENCES test(id) ON DELETE CASCADE,
  text          TEXT NOT NULL,
  metadata_json TEXT,
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_question_test ON question(test_id);

CREATE TABLE IF NOT EXISTS golden_answer (
  id          TEXT PRIMARY KEY,
  question_id TEXT NOT NULL REFERENCES question(id) ON DELETE CASCADE,
  text        TEXT NOT NULL
);

-- ===== EXECUTION =====
CREATE TABLE IF NOT EXISTS test_run (
  id                  TEXT PRIMARY KEY,
  project_id          TEXT NOT NULL REFERENCES project(id) ON DELETE CASCADE,
  test_id             TEXT NOT NULL REFERENCES test(id) ON DELETE RESTRICT,
  corpus_snapshot_id  TEXT REFERENCES corpus_snapshot(id) ON DELETE RESTRICT,
  index_build_id      TEXT REFERENCES index_build(id) ON DELETE RESTRICT,
  retriever_cfg_id    TEXT NOT NULL REFERENCES retriever_config_version(id) ON DELETE RESTRICT,
  generation_cfg_id   TEXT NOT NULL REFERENCES generation_config_version(id) ON DELETE RESTRICT,
  judge_cfg_id        TEXT NOT NULL REFERENCES judge_config_version(id) ON DELETE RESTRICT,
  prompt_tpl_ver_id   TEXT NOT NULL REFERENCES prompt_template_version(id) ON DELETE RESTRICT,
  tags_json           TEXT,
  seed                INTEGER,
  git_commit          TEXT,
  env_json            TEXT,
  started_at          TEXT DEFAULT CURRENT_TIMESTAMP,
  ended_at            TEXT
);
CREATE INDEX IF NOT EXISTS idx_test_run_test ON test_run(test_id);

CREATE TABLE IF NOT EXISTS run_question (
  id               TEXT PRIMARY KEY,
  test_run_id      TEXT NOT NULL REFERENCES test_run(id) ON DELETE CASCADE,
  question_id      TEXT NOT NULL REFERENCES question(id) ON DELETE RESTRICT,
  question_text    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_run_question_run ON run_question(test_run_id);

CREATE TABLE IF NOT EXISTS retrieval_log (
  id                     TEXT PRIMARY KEY,
  run_question_id        TEXT NOT NULL REFERENCES run_question(id) ON DELETE CASCADE,
  retriever_cfg_id       TEXT NOT NULL REFERENCES retriever_config_version(id) ON DELETE RESTRICT,
  params_json            TEXT,
  created_at             TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS retrieved_chunk (
  id               TEXT PRIMARY KEY,
  retrieval_log_id TEXT NOT NULL REFERENCES retrieval_log(id) ON DELETE CASCADE,
  rank             INTEGER NOT NULL,
  score            REAL,
  document_id      TEXT REFERENCES document(id) ON DELETE SET NULL,
  chunk_id         TEXT REFERENCES chunk(id) ON DELETE SET NULL,
  span_start       INTEGER,
  span_end         INTEGER,
  filters_json     TEXT
);
CREATE INDEX IF NOT EXISTS idx_retrieved_bylog ON retrieved_chunk(retrieval_log_id, rank);

CREATE TABLE IF NOT EXISTS llm_call (
  id                 TEXT PRIMARY KEY,
  run_question_id    TEXT NOT NULL REFERENCES run_question(id) ON DELETE CASCADE,
  model              TEXT NOT NULL,
  role_inputs_json   TEXT,
  prompt_tokens      INTEGER,
  completion_tokens  INTEGER,
  latency_ms         INTEGER,
  output_text        TEXT,
  error_json         TEXT,
  created_at         TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS answer (
  id               TEXT PRIMARY KEY,
  run_question_id  TEXT NOT NULL REFERENCES run_question(id) ON DELETE CASCADE,
  text             TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evaluation (
  id                 TEXT PRIMARY KEY,
  run_question_id    TEXT NOT NULL REFERENCES run_question(id) ON DELETE CASCADE,
  judge_cfg_id       TEXT NOT NULL REFERENCES judge_config_version(id) ON DELETE RESTRICT,
  metric_name        TEXT NOT NULL,
  score              REAL NOT NULL,
  rationale_text     TEXT,
  raw_judge_output_json TEXT,
  created_at         TEXT DEFAULT CURRENT_TIMESTAMP,
  superseded_by_id   TEXT REFERENCES evaluation(id) ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS idx_eval_runq_metric ON evaluation(run_question_id, metric_name);

CREATE TABLE IF NOT EXISTS human_review (
  id               TEXT PRIMARY KEY,
  run_question_id  TEXT NOT NULL REFERENCES run_question(id) ON DELETE CASCADE,
  decision         TEXT,
  new_score        REAL,
  notes            TEXT,
  reviewer         TEXT,
  created_at       TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS run_kpi (
  id            TEXT PRIMARY KEY,
  test_run_id   TEXT NOT NULL REFERENCES test_run(id) ON DELETE CASCADE,
  metrics_json  TEXT,
  cost_estimate REAL,
  time_seconds  REAL,
  created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_run_kpi_run ON run_kpi(test_run_id);

"""

def open(db_path: str):
    """Open a single DB connection and initialize schema if not exists."""
    global _DB_CONN
    if _DB_CONN is None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.executescript(_SCHEMA_SQL)
        _DB_CONN = conn
        logger.info("Database connection established successfully")
        return _DB_CONN
    
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Opening database connection to: {db_file.absolute()}")
    try:
        _DB_CONN = sqlite3.connect(str(db_file))
        logger.info("Database connection established successfully")
        
        logger.info("Initializing database schema")
        _DB_CONN.executescript(_SCHEMA_SQL)
        _DB_CONN.row_factory = sqlite3.Row
        logger.info("Database schema initialized successfully")
        
        return _DB_CONN
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

def get_conn():
    """Get the already-open connection."""
    if _DB_CONN is None:
        error_msg = "Database connection not initialized. Call open() first."
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    logger.debug("Returning active database connection")
    return _DB_CONN

def close():
    """Close the connection."""
    global _DB_CONN
    if _DB_CONN is not None:
        try:
            logger.info("Closing database connection")
            _DB_CONN.close()
            logger.info("Database connection closed successfully")
        except Exception as e:
            logger.error(f"Error closing database connection: {str(e)}")
            raise
        finally:
            _DB_CONN = None
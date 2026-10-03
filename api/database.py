import os
from contextlib import contextmanager
from typing import Generator

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from dotenv import load_dotenv

load_dotenv()

_pool: ThreadedConnectionPool | None = None


def _get_pool() -> ThreadedConnectionPool:
    """Initialize the connection pool on first call; reuse thereafter."""
    global _pool
    if _pool is None:
        database_url = os.environ["DATABASE_URL"]
        _pool = ThreadedConnectionPool(minconn=1, maxconn=10, dsn=database_url)
    return _pool


@contextmanager
def get_connection() -> Generator:
    """Check out a connection from the pool; return it on exit.

    Rolls back on any exception so the connection is clean when returned.
    """
    pool = _get_pool()
    conn = pool.getconn()
    try:
        yield conn
    except Exception:
        conn.rollback()
        raise
    finally:
        pool.putconn(conn)


def init_db() -> None:
    """Create the traces table and indexes — safe to call on every startup."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    project_name  TEXT NOT NULL DEFAULT 'default',
                    run_name      TEXT,
                    prompt        TEXT NOT NULL,
                    response      TEXT NOT NULL,
                    model         TEXT NOT NULL,
                    latency_ms    INTEGER NOT NULL,
                    input_tokens  INTEGER,
                    output_tokens INTEGER,
                    cost_usd      DECIMAL(10,6),
                    contexts      TEXT[],
                    faithfulness  FLOAT,
                    relevancy     FLOAT,
                    eval_status   TEXT NOT NULL DEFAULT 'pending',
                    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );

                CREATE INDEX IF NOT EXISTS idx_traces_project_created
                    ON traces(project_name, created_at DESC);

                CREATE INDEX IF NOT EXISTS idx_traces_eval_pending
                    ON traces(eval_status)
                    WHERE eval_status = 'pending';
            """)
            conn.commit()

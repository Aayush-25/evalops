import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

# Mock ragas and datasets so evaluator.py can be imported in tests without
# the real libraries installed. Actual evaluate() calls are patched per-test.
for _mod in ("ragas", "ragas.metrics", "datasets"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_cursor() -> MagicMock:
    """Reusable psycopg2 cursor mock; configure return values per-test."""
    cursor = MagicMock()
    cursor.fetchone.return_value = (1,)
    cursor.fetchall.return_value = []
    return cursor


@pytest.fixture
def client(mock_cursor: MagicMock):
    """FastAPI TestClient with fully mocked database layer.

    Yields (TestClient, mock_cursor) so individual tests can set
    cursor.fetchone / cursor.fetchall before making requests.
    """
    mock_conn = MagicMock()
    # psycopg2 cursors are context managers: `with conn.cursor() as cur:`
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    @contextmanager
    def _mock_get_connection():
        yield mock_conn

    # Patch the bare 'database' module that main.py imports (not api.database —
    # they're separate sys.modules entries because uvicorn runs from api/).
    with patch("database.init_db"), \
         patch("database.get_connection", _mock_get_connection):
        from api.main import app
        with TestClient(app) as c:
            yield c, mock_cursor

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

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

    # Patch before importing app so startup event uses mock
    with patch("api.database.init_db"), \
         patch("api.database.get_connection", _mock_get_connection):
        from api.main import app
        with TestClient(app) as c:
            yield c, mock_cursor

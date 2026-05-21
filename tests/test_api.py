import uuid
from datetime import datetime, timezone
from decimal import Decimal

import pytest
from pydantic import ValidationError

from api.models import (
    EvaluateRequest,
    HealthResponse,
    TraceCreate,
    TraceResponse,
)

FAKE_UUID = uuid.uuid4()
FAKE_NOW = datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# TraceCreate validation
# ---------------------------------------------------------------------------

class TestTraceCreate:
    def test_accepts_required_fields_only(self):
        t = TraceCreate(
            prompt="What is 2+2?",
            response="4",
            model="gpt-4o",
            latency_ms=150,
        )
        assert t.project_name == "default"
        assert t.run_name is None
        assert t.input_tokens is None

    def test_rejects_missing_prompt(self):
        with pytest.raises(ValidationError):
            TraceCreate(response="4", model="gpt-4o", latency_ms=150)

    def test_rejects_missing_response(self):
        with pytest.raises(ValidationError):
            TraceCreate(prompt="hi", model="gpt-4o", latency_ms=150)

    def test_rejects_missing_model(self):
        with pytest.raises(ValidationError):
            TraceCreate(prompt="hi", response="bye", latency_ms=150)

    def test_rejects_missing_latency_ms(self):
        with pytest.raises(ValidationError):
            TraceCreate(prompt="hi", response="bye", model="gpt-4o")

    def test_accepts_all_optional_fields(self):
        t = TraceCreate(
            prompt="p",
            response="r",
            model="gpt-4o",
            latency_ms=100,
            input_tokens=10,
            output_tokens=5,
            cost_usd=Decimal("0.000030"),
            contexts=["ctx1", "ctx2"],
            project_name="my-project",
            run_name="run-1",
        )
        assert t.project_name == "my-project"
        assert t.contexts == ["ctx1", "ctx2"]


# ---------------------------------------------------------------------------
# TraceResponse validation
# ---------------------------------------------------------------------------

class TestTraceResponse:
    def test_constructs_from_all_fields(self):
        t = TraceResponse(
            id=FAKE_UUID,
            project_name="default",
            run_name=None,
            prompt="p",
            response="r",
            model="gpt-4o",
            latency_ms=100,
            input_tokens=10,
            output_tokens=5,
            cost_usd=Decimal("0.000030"),
            contexts=None,
            faithfulness=None,
            relevancy=None,
            eval_status="pending",
            created_at=FAKE_NOW,
        )
        assert str(t.id) == str(FAKE_UUID)
        assert t.eval_status == "pending"


# ---------------------------------------------------------------------------
# EvaluateRequest
# ---------------------------------------------------------------------------

class TestEvaluateRequest:
    def test_defaults_to_default_project(self):
        r = EvaluateRequest()
        assert r.project_name == "default"

    def test_accepts_custom_project(self):
        r = EvaluateRequest(project_name="prod")
        assert r.project_name == "prod"


# ---------------------------------------------------------------------------
# HealthResponse
# ---------------------------------------------------------------------------

class TestHealthResponse:
    def test_constructs(self):
        h = HealthResponse(status="ok", db_connected=True)
        assert h.status == "ok"
        assert h.db_connected is True


# ---------------------------------------------------------------------------
# database module — unit tests with mocked pool
# ---------------------------------------------------------------------------

class TestDatabase:
    def test_init_db_executes_create_table(self):
        """init_db must issue CREATE TABLE and commit, even if called twice."""
        from unittest.mock import MagicMock, patch
        from contextlib import contextmanager

        mock_cursor = MagicMock()
        mock_conn = MagicMock()
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

        @contextmanager
        def _mock_get_conn():
            yield mock_conn

        with patch("api.database.get_connection", _mock_get_conn):
            from api.database import init_db
            init_db()

        mock_cursor.execute.assert_called_once()
        sql = mock_cursor.execute.call_args[0][0]
        assert "CREATE TABLE IF NOT EXISTS traces" in sql
        assert "CREATE INDEX IF NOT EXISTS idx_traces_project_created" in sql
        mock_conn.commit.assert_called_once()


# ---------------------------------------------------------------------------
# Shared fixture data for route tests
# ---------------------------------------------------------------------------

import uuid as _uuid
from datetime import datetime as _datetime, timezone as _tz
from decimal import Decimal as _Decimal

_FAKE_UUID = str(_uuid.uuid4())
_FAKE_NOW = _datetime.now(_tz.utc)

# Matches the 15-column SELECT order in main.py's _row_to_trace()
_FAKE_ROW = (
    _FAKE_UUID,           # id
    "default",            # project_name
    None,                 # run_name
    "What is 2+2?",       # prompt
    "4",                  # response
    "gpt-4o",             # model
    150,                  # latency_ms
    10,                   # input_tokens
    5,                    # output_tokens
    _Decimal("0.000030"), # cost_usd
    None,                 # contexts
    None,                 # faithfulness
    None,                 # relevancy
    "pending",            # eval_status
    _FAKE_NOW,            # created_at
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_ok_when_db_responds(self, client):
        c, cursor = client
        cursor.fetchone.return_value = (1,)
        r = c.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok", "db_connected": True}


# ---------------------------------------------------------------------------
# POST /traces
# ---------------------------------------------------------------------------

class TestCreateTrace:
    def test_creates_trace_with_required_fields(self, client):
        c, cursor = client
        cursor.fetchone.return_value = _FAKE_ROW
        r = c.post("/traces", json={
            "prompt": "What is 2+2?",
            "response": "4",
            "model": "gpt-4o",
            "latency_ms": 150,
        })
        assert r.status_code == 201
        data = r.json()
        assert data["prompt"] == "What is 2+2?"
        assert data["model"] == "gpt-4o"
        assert data["eval_status"] == "pending"
        assert data["project_name"] == "default"

    def test_rejects_missing_required_field(self, client):
        c, _ = client
        r = c.post("/traces", json={
            "prompt": "hi",
            # missing response, model, latency_ms
        })
        assert r.status_code == 422

    def test_cursor_receives_parameterized_query(self, client):
        c, cursor = client
        cursor.fetchone.return_value = _FAKE_ROW
        c.post("/traces", json={
            "prompt": "test prompt",
            "response": "test response",
            "model": "gpt-4o",
            "latency_ms": 200,
        })
        cursor.execute.assert_called_once()
        # Verify parameterized — no string formatting
        call_args = cursor.execute.call_args[0]
        assert "%s" in call_args[0]
        assert "test prompt" in call_args[1]


# ---------------------------------------------------------------------------
# GET /traces
# ---------------------------------------------------------------------------

class TestListTraces:
    def test_returns_empty_list_when_no_rows(self, client):
        c, cursor = client
        cursor.fetchall.return_value = []
        r = c.get("/traces")
        assert r.status_code == 200
        assert r.json() == []

    def test_returns_list_when_rows_exist(self, client):
        c, cursor = client
        cursor.fetchall.return_value = [_FAKE_ROW]
        r = c.get("/traces")
        assert r.status_code == 200
        assert len(r.json()) == 1
        assert r.json()[0]["model"] == "gpt-4o"

    def test_accepts_project_name_filter(self, client):
        c, cursor = client
        cursor.fetchall.return_value = [_FAKE_ROW]
        r = c.get("/traces?project_name=default")
        assert r.status_code == 200
        # project_name param is passed to the query
        call_args = cursor.execute.call_args[0]
        assert "default" in call_args[1]

    def test_accepts_pagination_params(self, client):
        c, cursor = client
        cursor.fetchall.return_value = []
        r = c.get("/traces?limit=10&offset=5&days_back=30")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# POST /evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_evaluation_started_with_count(self, client):
        c, cursor = client
        cursor.fetchall.return_value = [(_FAKE_UUID,), (_FAKE_UUID,)]
        r = c.post("/evaluate", json={"project_name": "default"})
        assert r.status_code == 200
        data = r.json()
        assert data["message"] == "evaluation started"
        assert data["trace_count"] == 2

    def test_uses_default_project_name_when_omitted(self, client):
        c, cursor = client
        cursor.fetchall.return_value = []
        r = c.post("/evaluate", json={})
        assert r.status_code == 200
        assert r.json()["trace_count"] == 0

    def test_pending_filter_in_query(self, client):
        c, cursor = client
        cursor.fetchall.return_value = []
        c.post("/evaluate", json={"project_name": "prod"})
        call_args = cursor.execute.call_args[0]
        assert "pending" in call_args[0]
        assert "prod" in call_args[1]

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

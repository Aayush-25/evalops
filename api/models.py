from datetime import datetime
from decimal import Decimal
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class TraceCreate(BaseModel):
    prompt: str
    response: str
    model: str
    latency_ms: int
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    cost_usd: Optional[Decimal] = None
    contexts: Optional[list[str]] = None
    project_name: str = "default"
    run_name: Optional[str] = None


class TraceResponse(BaseModel):
    id: UUID
    project_name: str
    run_name: Optional[str]
    prompt: str
    response: str
    model: str
    latency_ms: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_usd: Optional[Decimal]
    contexts: Optional[list[str]]
    faithfulness: Optional[float]
    relevancy: Optional[float]
    eval_status: str
    created_at: datetime


class EvaluateRequest(BaseModel):
    project_name: str = "default"


class HealthResponse(BaseModel):
    status: str
    db_connected: bool

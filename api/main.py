import os
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api import database
from api.models import EvaluateRequest, HealthResponse, TraceCreate, TraceResponse

load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    database.init_db()
    yield


app = FastAPI(title="EvalOps API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    try:
        with database.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return HealthResponse(status="ok", db_connected=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/traces", response_model=TraceResponse, status_code=201)
def create_trace(trace: TraceCreate) -> TraceResponse:
    try:
        with database.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO traces (
                        project_name, run_name, prompt, response, model,
                        latency_ms, input_tokens, output_tokens, cost_usd, contexts
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id, project_name, run_name, prompt, response,
                              model, latency_ms, input_tokens, output_tokens,
                              cost_usd, contexts, faithfulness, relevancy,
                              eval_status, created_at
                    """,
                    (
                        trace.project_name, trace.run_name, trace.prompt,
                        trace.response, trace.model, trace.latency_ms,
                        trace.input_tokens, trace.output_tokens,
                        trace.cost_usd, trace.contexts,
                    ),
                )
                row = cur.fetchone()
                conn.commit()
        return _row_to_trace(row)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/traces", response_model=list[TraceResponse])
def list_traces(
    project_name: Optional[str] = None,
    days_back: int = 7,
    limit: int = 50,
    offset: int = 0,
) -> list[TraceResponse]:
    try:
        with database.get_connection() as conn:
            with conn.cursor() as cur:
                if project_name:
                    cur.execute(
                        """
                        SELECT id, project_name, run_name, prompt, response,
                               model, latency_ms, input_tokens, output_tokens,
                               cost_usd, contexts, faithfulness, relevancy,
                               eval_status, created_at
                        FROM traces
                        WHERE project_name = %s
                          AND created_at >= NOW() - (INTERVAL '1 day' * %s)
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        (project_name, days_back, limit, offset),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, project_name, run_name, prompt, response,
                               model, latency_ms, input_tokens, output_tokens,
                               cost_usd, contexts, faithfulness, relevancy,
                               eval_status, created_at
                        FROM traces
                        WHERE created_at >= NOW() - (INTERVAL '1 day' * %s)
                        ORDER BY created_at DESC
                        LIMIT %s OFFSET %s
                        """,
                        (days_back, limit, offset),
                    )
                rows = cur.fetchall()
        return [_row_to_trace(r) for r in rows]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate")
def start_evaluation(
    req: EvaluateRequest, background_tasks: BackgroundTasks
) -> dict:
    try:
        with database.get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id FROM traces WHERE eval_status = 'pending' AND project_name = %s",
                    (req.project_name,),
                )
                trace_ids = [row[0] for row in cur.fetchall()]

        background_tasks.add_task(_run_evaluation, trace_ids, req.project_name)
        return {"message": "evaluation started", "trace_count": len(trace_ids)}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _run_evaluation(trace_ids: list, project_name: str) -> None:
    """Placeholder — RAGAS evaluation to be implemented."""
    pass


def _row_to_trace(row: tuple) -> TraceResponse:
    return TraceResponse(
        id=row[0],
        project_name=row[1],
        run_name=row[2],
        prompt=row[3],
        response=row[4],
        model=row[5],
        latency_ms=row[6],
        input_tokens=row[7],
        output_tokens=row[8],
        cost_usd=row[9],
        contexts=row[10],
        faithfulness=row[11],
        relevancy=row[12],
        eval_status=row[13],
        created_at=row[14],
    )

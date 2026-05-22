# EvalOps

EvalOps is a lightweight LLM evaluation and observability platform that lets you instrument any Python LLM application with a two-line decorator, store traces in PostgreSQL, and automatically score them for faithfulness and answer relevancy using RAGAS. It ships as three independent pieces: a FastAPI backend, a Python SDK, and a Streamlit dashboard.

## Live Demo

| Service | URL |
|---------|-----|
| API docs (Swagger) | https://evalops-production.up.railway.app/docs |
| Dashboard | https://evalops-p5mh2czqnbwgkpzyvmaey7.streamlit.app |

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Your LLM Application               │
│                                                     │
│   from evalops import EvalOpsTracer                 │
│                                                     │
│   tracer = EvalOpsTracer("http://localhost:8000")   │
│                                                     │
│   @tracer.trace(model="gpt-4o")                     │
│   def ask(prompt: str) -> str:                      │
│       return openai_client.chat(...)                │
└────────────────────┬────────────────────────────────┘
                     │ POST /traces  (httpx)
                     ▼
┌─────────────────────────────────────────────────────┐
│                 FastAPI  (api/)                      │
│                                                     │
│   POST /traces    →  persist trace                  │
│   GET  /traces    →  list + filter traces           │
│   POST /evaluate  →  enqueue RAGAS scoring          │
│   GET  /health    →  liveness + DB check            │
└──────────┬──────────────────────────┬───────────────┘
           │ psycopg2                 │ BackgroundTask
           │                         ▼
           │              ┌──────────────────────────┐
           │              │      RAGAS Engine         │
           │              │   faithfulness            │
           │              │   answer_relevancy        │
           │              └────────────┬─────────────┘
           │                           │ UPDATE scores
           ▼                           ▼
┌─────────────────────────────────────────────────────┐
│                   PostgreSQL 15                     │
│             (traces — 15-column table)              │
└───────────────────────────┬─────────────────────────┘
                            │ direct SQL
                            ▼
┌─────────────────────────────────────────────────────┐
│            Streamlit Dashboard (dashboard/)          │
│                                                     │
│   Overview · Trace Explorer · Single Trace          │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.11+
- Docker (for PostgreSQL)
- OpenAI API key (for RAGAS scoring)

### 1. Environment

```bash
git clone <repo>
cd evalops
cp .env.example .env
# Edit .env — set DATABASE_URL, POSTGRES_PASSWORD, OPENAI_API_KEY
```

### 2. Database

```bash
docker compose up -d
# Wait ~10 seconds for the health check to pass
docker compose ps   # evalops_db should show "healthy"
```

### 3. API

```bash
pip install -r api/requirements.txt
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Verify:

```bash
curl http://localhost:8000/health
# {"status":"ok","db_connected":true}
```

### 4. SDK

```bash
pip install -e sdk/
```

Instrument your code:

```python
from evalops import EvalOpsTracer

tracer = EvalOpsTracer("http://localhost:8000", project_name="my-project")

@tracer.trace(model="gpt-4o")
def ask(prompt: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
```

Or with token tracking:

```python
with tracer.span(prompt=prompt, model="gpt-4o") as span:
    resp = openai_client.chat.completions.create(...)
    span.set_response(resp.choices[0].message.content)
    span.set_tokens(
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
    )
```

### 5. Trigger Evaluation

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"project_name": "my-project"}'
```

### 6. Dashboard

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

---

## Tests

```bash
pytest tests/ sdk/evalops/tests/ -v
# 52 tests, ~0.3s
```

---

## Design Decisions

**Raw SQL over ORM.**
psycopg2 with a `ThreadedConnectionPool` handles the connection lifecycle without an ORM abstraction layer. The schema is a single `traces` table with 15 columns — simple enough that raw SQL is easier to read and debug than generated queries.

**Non-blocking evaluation.**
`POST /evaluate` returns immediately with a count of pending traces and hands off to FastAPI's `BackgroundTasks`. The evaluator marks rows `running` before touching RAGAS (preventing double-evaluation on concurrent calls) and resets them to `failed` if RAGAS throws, so no trace gets silently lost.

**Span/Tracer separation in the SDK.**
`EvalOpsTracer` owns the HTTP client and `log()`. `Span` is a pure timing primitive — it holds mutable state during a timed block, then calls `log()` on clean exit. The `@tracer.trace()` decorator is a thin wrapper around `span()`. This keeps the two concerns independently testable: Span tests mock `tracer.log`; Tracer tests mock the HTTP layer with `respx`.

**Bare imports in `api/`.**
`api/main.py` and `api/evaluator.py` use `import database` / `from models import ...` rather than `from api import database`. This is intentional: uvicorn runs from inside the `api/` directory where these modules are top-level. The `pytest.ini` pythonpath includes both `.` and `api` so tests resolve the same bare imports without the server's working directory.

**Dashboard queries are direct SQL.**
The Streamlit dashboard connects directly to PostgreSQL rather than going through the API. This avoids serialising large result sets over HTTP and lets the dashboard use SQL aggregations (GROUP BY day, AVG FILTER) that would be awkward to expose as REST endpoints. The API is only called for write operations (triggering evaluation).

**`@st.cache_data(ttl=30)` on every query.**
Each page re-fetches at most once per 30 seconds. "Trigger Evaluation" calls `st.cache_data.clear()` so the next render picks up updated scores immediately.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI 0.111 + Uvicorn |
| Database driver | psycopg2-binary 2.9 (ThreadedConnectionPool) |
| Schema validation | Pydantic v2 |
| Evaluation | RAGAS 0.1.21 (faithfulness, answer_relevancy) |
| Evaluation dataset | HuggingFace `datasets` |
| SDK HTTP client | httpx |
| Dashboard | Streamlit |
| Database | PostgreSQL 15 (Docker) |
| Testing | pytest + respx (httpx mock) |
| Python | 3.11+ |

---

## Project Structure

```
evalops/
├── api/
│   ├── main.py          # FastAPI app — 4 routes
│   ├── evaluator.py     # RAGAS evaluation engine
│   ├── database.py      # Connection pool + schema init
│   ├── models.py        # Pydantic v2 request/response models
│   └── requirements.txt
├── sdk/
│   ├── pyproject.toml
│   └── evalops/
│       ├── tracer.py    # EvalOpsTracer — HTTP client + decorator
│       ├── span.py      # Span — timing context manager
│       └── pricing.py   # compute_cost() — token cost estimator
├── dashboard/
│   ├── app.py           # Streamlit dashboard (3 pages)
│   └── requirements.txt
├── data/
│   └── golden_dataset.json   # 10 HR Q&A pairs for smoke testing
├── tests/               # API + evaluator tests (pytest)
├── docker-compose.yml
└── .env.example
```

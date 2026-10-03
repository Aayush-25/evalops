# EvalOps — LLM Evaluation & Observability Platform

**Open-source quality monitoring for LLM applications**

[![Dashboard Live](https://img.shields.io/badge/Dashboard-Live%20on%20Streamlit-ff4b4b)](https://evalops-p5mh2czqnbwgkpzyvmaey7.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-Aayush--25%2Fevalops-181717?logo=github)](https://github.com/Aayush-25/evalops)
[![Tests](https://img.shields.io/badge/tests-52%20passing-brightgreen)](https://github.com/Aayush-25/evalops)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)

---

## What It Is

Most LLM applications ship without any systematic way to measure whether their answers are accurate or relevant — a bug that only surfaces when users complain. EvalOps gives you a two-line decorator that captures every LLM call, stores it in PostgreSQL, and automatically scores each response for hallucination and relevancy using RAGAS, with results visible in a live dashboard.

---

## Live Demo

| Service | URL |
|---------|-----|
| Streamlit Dashboard | https://evalops-p5mh2czqnbwgkpzyvmaey7.streamlit.app |
| Source Code | https://github.com/Aayush-25/evalops |

---

## Architecture

```
┌──────────────────────────────────────────┐
│       SDK  (pip install evalops)         │
│                                          │
│   @tracer.trace(model="gpt-4o")          │
│   def ask_llm(prompt: str) -> str: ...   │
└────────────────┬─────────────────────────┘
                 │ HTTP POST /traces
                 ▼
┌──────────────────────────────────────────┐
│           FastAPI Backend                │
│                                          │
│  POST /traces   GET /traces              │
│  POST /evaluate GET /health              │
└────────────────┬─────────────────────────┘
                 │ psycopg2 raw SQL
                 ▼
┌──────────────────────────────────────────┐
│         PostgreSQL Database              │
│                                          │
│  traces table — id, prompt, response,    │
│  model, faithfulness, relevancy, ...     │
└────────────────┬─────────────────────────┘
                 │ RAGAS evaluation (BackgroundTask)
                 ▼
┌──────────────────────────────────────────┐
│   Streamlit Dashboard  (Streamlit Cloud) │
│                                          │
│  Overview · Trace Explorer · Single Trace│
└──────────────────────────────────────────┘
```

---

## What It Measures

EvalOps scores every LLM response on two dimensions using [RAGAS](https://docs.ragas.io):

**Faithfulness** — *Is the answer actually supported by the source documents?*
A score of `1.0` means every claim in the response can be traced back to the provided context. A low score is a hallucination signal: the model stated something it cannot justify from the retrieved text.

**Answer Relevancy** — *Does the answer actually address what was asked?*
A score of `1.0` means the response directly and completely answers the question. A low score means the model went off-topic or gave a generic non-answer.

Both scores appear in the dashboard and are stored per-trace, so you can track quality over time and catch regressions across model versions or prompt changes.

---

## SDK Usage

```python
from evalops import EvalOpsTracer

tracer = EvalOpsTracer(
    api_url="http://localhost:8000",
    project_name="my-project",
)

@tracer.trace(model="gpt-4o")
def ask_llm(prompt: str) -> str:
    return openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    ).choices[0].message.content
```

The decorator automatically measures latency and ships the trace. For token counts and cost tracking, use the context manager form:

```python
with tracer.span(prompt=prompt, model="gpt-4o") as span:
    resp = openai_client.chat.completions.create(...)
    span.set_response(resp.choices[0].message.content)
    span.set_tokens(
        input_tokens=resp.usage.prompt_tokens,
        output_tokens=resp.usage.completion_tokens,
    )
```

---

## Quick Start

```bash
# 1. Clone and configure environment
git clone https://github.com/Aayush-25/evalops && cd evalops
cp .env.example .env   # set DATABASE_URL, POSTGRES_PASSWORD, OPENAI_API_KEY

# 2. Start PostgreSQL
docker compose up -d

# 3. Run the API  (must run from inside api/ — see Design Decisions)
cd api && pip install -r requirements.txt && uvicorn main:app --reload

# 4. Run the dashboard  (new terminal, from project root)
pip install -r dashboard/requirements.txt && streamlit run dashboard/app.py

# 5. Send a test trace
curl -X POST http://localhost:8000/traces \
  -H "Content-Type: application/json" \
  -d '{"prompt":"What is the capital of France?","response":"Paris","model":"gpt-4o","latency_ms":230}'
```

---

## Design Decisions

**BackgroundTasks over Celery.**
RAGAS scoring takes 2–10 seconds per batch. FastAPI's built-in `BackgroundTasks` handles this without a broker, a worker process, or Redis infrastructure. The evaluator marks rows `running` before calling RAGAS and resets them to `failed` on any exception, giving the same at-least-once delivery guarantee Celery would provide — without the operational overhead that's unjustified at this scale.

**PostgreSQL over a key-value store.**
Traces are structured, relational data that the dashboard queries with aggregations (`AVG FILTER`, `GROUP BY day`, pagination with `LIMIT/OFFSET`). PostgreSQL handles all of this natively; a key-value store would push that complexity into application code.

**Streamlit over React.**
The dashboard is a read-heavy analytics tool, not a product UI. Streamlit renders pandas DataFrames and line charts in a few lines of Python, deploys to Streamlit Cloud in one click, and requires no build pipeline, no npm, and no state management library. The trade-off — Streamlit reruns the full script on interaction — is irrelevant for a low-traffic internal dashboard with 30-second query caching.

**Raw SQL over SQLAlchemy.**
EvalOps has one table. The queries are straightforward enough that an ORM adds indirection without simplifying anything. Raw psycopg2 with a `ThreadedConnectionPool` makes every query visible, easy to profile, and impossible to accidentally make N+1. All user values go through `%s` parameterization — there is no string formatting of user input anywhere in the codebase.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| API framework | FastAPI 0.111 + Uvicorn |
| Database | PostgreSQL 15 |
| Database driver | psycopg2-binary (ThreadedConnectionPool, raw SQL) |
| Data validation | Pydantic v2 |
| Evaluation engine | RAGAS 0.1.21 — faithfulness, answer_relevancy |
| Evaluation dataset | HuggingFace `datasets` |
| SDK HTTP client | httpx |
| Dashboard | Streamlit |
| Deployment | Docker (API + PostgreSQL), Streamlit Cloud |
| Testing | pytest + respx — 52 tests, ~0.3 s |
| Python | 3.11+ |

---

## Project Structure

```
evalops/
├── api/
│   ├── main.py           # FastAPI app — 4 routes
│   ├── evaluator.py      # RAGAS evaluation engine
│   ├── database.py       # ThreadedConnectionPool + schema init
│   ├── models.py         # Pydantic v2 request/response models
│   ├── CLAUDE.md         # Import conventions (bare imports for uvicorn)
│   └── requirements.txt
├── sdk/
│   ├── pyproject.toml    # pip install -e sdk/
│   └── evalops/
│       ├── tracer.py     # EvalOpsTracer — HTTP client, decorator, span factory
│       ├── span.py       # Span — timing context manager, auto cost computation
│       └── pricing.py    # compute_cost() — per-model token pricing table
├── dashboard/
│   ├── app.py            # Streamlit dashboard — Overview, Trace Explorer, Single Trace
│   └── requirements.txt
├── tests/
│   ├── conftest.py       # Fixtures — mocked DB + ragas/datasets stubs
│   ├── test_api.py       # API route + model tests (22 tests)
│   └── test_evaluator.py # Evaluator unit tests (6 tests)
├── data/
│   └── golden_dataset.json  # 10 HR Q&A pairs — 7 grounded, 3 hallucinated
├── docker-compose.yml
├── pytest.ini
└── .env.example
```

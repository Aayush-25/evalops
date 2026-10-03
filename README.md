# EvalOps

**Trace LLM calls, store them in PostgreSQL, and inspect evaluation results.**

EvalOps is a backend learning project with a Python tracing SDK, FastAPI API, RAGAS evaluator, and Streamlit dashboard. It explores instrumentation, relational storage, background evaluation, and testable API design.

[Quick start](#quick-start) · [SDK usage](#sdk-usage) · [Tests](#tests) · [Limitations](#current-limitations)

## Architecture

```text
Your Python application
        │ tracing SDK / HTTP
        ▼
    FastAPI API ───────────────► PostgreSQL
        │                           ▲
        │ POST /evaluate            │ reads traces and scores
        ▼                           │
RAGAS background evaluation    Streamlit dashboard
        │                           │
        └── writes scores to DB     └── POST /evaluate to API
```

The SDK records traces; evaluation starts when `/evaluate` is called, including through the dashboard. It is not automatically triggered by every traced call.

## What is implemented

- Trace capture: prompt, response, model, latency, optional token counts, contexts, and estimated cost.
- PostgreSQL storage with parameterized queries, connection pooling, and indexes.
- RAGAS faithfulness and answer-relevancy scoring in batches of up to 20 pending traces.
- A dashboard for aggregate metrics, recent traces, and individual trace inspection.
- Unit tests covering the API, evaluator error handling, SDK tracing, and cost calculations.

## Quick start

Requires Python 3.11+, Docker with Compose, and an OpenAI API key for real RAGAS evaluation. Commands below use a macOS/Linux shell.

```bash
git clone https://github.com/Aayush-25/evalops.git
cd evalops
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r api/requirements.txt -r dashboard/requirements.txt
python -m pip install -e 'sdk[dev]'
cp .env.example .env
```

Edit `.env`: set the same database password in `POSTGRES_PASSWORD` and `DATABASE_URL`, and replace the placeholder `OPENAI_API_KEY`. The dashboard uses both `DATABASE_URL` for reads and `API_URL` for evaluation requests.

```bash
docker compose up -d db
# Start from api/ because its modules use local imports.
cd api
python -m uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

In another terminal, from the repository root:

```bash
source .venv/bin/activate
python -m streamlit run dashboard/app.py
```

Open the API docs at `http://127.0.0.1:8000/docs` and the dashboard at `http://localhost:8501`. Docker Compose starts **PostgreSQL only**; the API and dashboard run separately.

### Try the API

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/traces \
  -H 'Content-Type: application/json' \
  -d '{"project_name":"demo","prompt":"What is the capital of France?","response":"Paris","model":"example","latency_ms":230,"contexts":["Paris is the capital of France."]}'

# Makes calls to the configured evaluation provider.
curl -X POST http://127.0.0.1:8000/evaluate \
  -H 'Content-Type: application/json' \
  -d '{"project_name":"demo"}'

curl 'http://127.0.0.1:8000/traces?project_name=demo'
```

## SDK usage

Install the SDK from this repository with `python -m pip install -e sdk/`.

```python
from evalops import EvalOpsTracer

with EvalOpsTracer(api_url="http://127.0.0.1:8000", project_name="demo") as tracer:
    with tracer.span(
        prompt="What is the capital of France?",
        model="example",
        contexts=["Paris is the capital of France."],
    ) as span:
        span.set_response("Paris")
```

The SDK also offers `@tracer.trace(model="...")`. Token counts must be supplied with `span.set_tokens(...)`; the decorator does not extract usage from a provider response. Costs come from the SDK's pricing table and supplied counts, not a billing integration.

## Tests

From the repository root, after installing the dependencies above:

```bash
python -m pytest -q
```

The existing suite contains 52 tests. Database access and RAGAS calls are mocked, and SDK HTTP calls use `respx`. Passing this suite does not establish compatibility with a live database, the current evaluation provider, or a deployed service.

## Current limitations

- This is a development project, not a hardened public service: authentication, rate limiting, and tenant isolation are not implemented.
- FastAPI background tasks are in-process. There is no durable queue, automatic crash recovery, or delivery guarantee.
- Selecting pending traces and marking them running are separate operations; concurrent evaluations can select the same rows.
- Meaningful faithfulness scores require real retrieved contexts. A score is an evaluation signal, not proof of correctness.
- Provider versions and model availability need an integration check before deployment. Dependency versions are defined in `api/requirements.txt`.
- `data/golden_dataset.json` is sample data; the evaluator reads pending database traces, not this file.

## Repository map

| Path | Purpose |
| --- | --- |
| `api/` | Routes, request models, database access, evaluator |
| `sdk/evalops/` | Tracer, spans, and estimated pricing |
| `dashboard/app.py` | Streamlit UI with direct database reads |
| `tests/` | API and evaluator unit tests |
| `sdk/evalops/tests/` | SDK unit tests |
| `docker-compose.yml` | Local PostgreSQL service |

The local setup above is the reproducible entry point. A hosted deployment must be checked independently before being presented as a working demo.

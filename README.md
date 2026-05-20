# EvalOps

Lightweight LLM Evaluation and Observability Platform MVP.

## Quick Start

```bash
# 1. Copy and fill in environment variables
cp .env.example .env

# 2. Start PostgreSQL
docker compose up -d

# 3. Install API dependencies
pip install -r api/requirements.txt

# 4. Start the API
uvicorn api.main:app --reload

# 5. Verify
curl http://localhost:8000/health
```

## Running Tests

```bash
pytest tests/ -v
```

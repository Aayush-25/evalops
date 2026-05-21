# EvalOps Python SDK — Design Spec

**Date:** 2026-05-21
**Status:** Approved
**Context:** Phase 1 API is complete (4 REST routes, PostgreSQL, Pydantic v2, full test suite). This spec covers the Phase 2 Python SDK that wraps that API.

---

## Goal

Replace the Phase 1 stub (`EvalOpsTracer.log()` only) with a full SDK that supports auto-timing via decorators and context managers, proper packaging, and a complete test suite.

---

## File Structure

```
sdk/
├── pyproject.toml              # pip install -e sdk/
└── evalops/
    ├── __init__.py             # exports: EvalOpsTracer
    ├── tracer.py               # EvalOpsTracer — HTTP client + log() + decorator + span factory
    ├── span.py                 # Span — timing context manager
    ├── pricing.py              # compute_cost() — unchanged from Phase 1
    └── tests/
        ├── __init__.py
        ├── test_tracer.py      # EvalOpsTracer tests (respx mock)
        └── test_span.py        # Span + decorator tests (no HTTP)
```

---

## Packaging

`pyproject.toml` declares the package so it is installable via `pip install -e sdk/`.

**Runtime dependencies:** `httpx>=0.27`
**Dev/test dependencies:** `pytest`, `respx`
**No tiktoken** — token counting is the caller's responsibility.

---

## API Surface

### Manual log (existing interface, unchanged)

```python
tracer.log(
    prompt="What is 2+2?",
    response="4",
    model="gpt-4o",
    latency_ms=230,
    input_tokens=10,       # optional
    output_tokens=5,       # optional
    contexts=["doc1"],     # optional
    run_name="run-1",      # optional
)
```

### Context manager

```python
with tracer.span(prompt=prompt, model="gpt-4o") as span:
    result = openai_client.chat(...)
    span.set_response(result.choices[0].message.content)
    span.set_tokens(
        input_tokens=result.usage.prompt_tokens,
        output_tokens=result.usage.completion_tokens,
    )
# On __exit__ (no exception): computes latency_ms, cost_usd, calls tracer.log()
# On __exit__ (exception): re-raises, does NOT send the trace
```

### Decorator

```python
@tracer.trace(model="gpt-4o", project_name="prod")
def ask_llm(prompt: str) -> str:
    return openai_client.chat(...)
```

The decorator:
- Extracts `prompt` from the first positional argument (cast to `str`) or the `prompt` keyword argument.
- Wraps the function body in a `Span`.
- Calls `span.set_response(str(result))` with the return value.
- Preserves the original function signature via `functools.wraps`.
- Re-raises any exception without sending the trace.

### Lifecycle

```python
# Option A — explicit close
tracer = EvalOpsTracer(api_url="http://localhost:8000", project_name="default")
tracer.close()

# Option B — context manager
with EvalOpsTracer(api_url="http://localhost:8000") as tracer:
    ...
```

---

## Span Internals

`Span` is the timing primitive. It is created by `tracer.span()` and internally by `tracer.trace()`.

### State

| Field | Type | Set by |
|---|---|---|
| `_tracer` | `EvalOpsTracer` | constructor |
| `_prompt` | `str` | constructor |
| `_model` | `str` | constructor |
| `_run_name` | `Optional[str]` | constructor kwarg |
| `_contexts` | `Optional[list[str]]` | constructor kwarg or `set_contexts()` |
| `_response` | `str` | `set_response()`, default `""` |
| `_input_tokens` | `Optional[int]` | `set_tokens()` |
| `_output_tokens` | `Optional[int]` | `set_tokens()` |
| `_start` | `float` | `__enter__` (`time.monotonic()`) |

### Methods

- `set_response(text: str) -> None`
- `set_tokens(*, input_tokens: int, output_tokens: int) -> None`
- `set_contexts(contexts: list[str]) -> None`
- `__enter__() -> Span` — starts timer, returns `self`
- `__exit__(exc_type, exc_val, exc_tb) -> bool` — on no exception: computes `latency_ms` and `cost_usd`, calls `tracer.log()`; on exception: returns `False` (re-raises, no trace sent)

### Cost computation

On `__exit__`, if both `_input_tokens` and `_output_tokens` are set, `cost_usd = compute_cost(model, input_tokens, output_tokens)`. Otherwise `cost_usd=None`.

---

## EvalOpsTracer

```python
class EvalOpsTracer:
    def __init__(self, api_url: str, project_name: str = "default") -> None
    def log(self, prompt, response, model, latency_ms, **kwargs) -> dict
    def span(self, *, prompt: str, model: str, **kwargs) -> Span
    def trace(self, *, model: str, **kwargs)   # decorator factory
    def close(self) -> None
    def __enter__(self) -> "EvalOpsTracer"
    def __exit__(self, *args) -> None
```

`tracer.span()` is a factory — it constructs and returns a `Span` but does NOT enter it. The `with` statement enters it.

`tracer.trace()` is a decorator factory. It returns a decorator that wraps the target function. The wrapper calls `tracer.span()` and uses it as a context manager internally.

---

## Error Handling

- HTTP errors from `tracer.log()` propagate as `httpx.HTTPStatusError` — no swallowing, no retries.
- Exceptions inside a `span()` block or `@trace`-decorated function are re-raised. The trace is not sent.
- If `set_response()` is never called, `response=""` is sent. This is intentional — the caller can see the empty field rather than silently losing the trace.

---

## Testing Strategy

### `test_span.py` — no HTTP

- `Span.__enter__` starts a timer.
- `Span.__exit__` computes positive `latency_ms` and calls `tracer.log()` with correct args.
- `set_tokens()` causes `cost_usd` to be computed and passed to `log()`.
- Exception in span body: `log()` is never called.
- `set_response()` not called: `log()` receives `response=""`.

### `test_tracer.py` — respx mock

- `log()` POSTs to `/traces` with correct JSON body.
- `log()` raises `httpx.HTTPStatusError` on non-2xx response.
- `@trace` decorator sends a trace with correct `prompt` and `response`.
- `EvalOpsTracer` as context manager calls `close()` on exit.

---

## Out of Scope

- Async support (`AsyncEvalOpsTracer`) — Phase 3 if needed.
- Auto token counting via tiktoken — caller provides tokens from API response.
- Provider-specific integrations (OpenAI, Anthropic adapters).
- Retry logic / fire-and-forget background sending.
- Batch log submission.

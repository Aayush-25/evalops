import functools
from typing import Optional

import httpx

from .span import Span


class EvalOpsTracer:
    def __init__(self, api_url: str, project_name: str = "default") -> None:
        self.api_url = api_url.rstrip("/")
        self.project_name = project_name
        self._client = httpx.Client()

    def log(
        self,
        prompt: str,
        response: str,
        model: str,
        latency_ms: int,
        *,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        cost_usd: Optional[float] = None,
        contexts: Optional[list[str]] = None,
        run_name: Optional[str] = None,
    ) -> dict:
        payload: dict = {
            "prompt": prompt,
            "response": response,
            "model": model,
            "latency_ms": latency_ms,
            "project_name": self.project_name,
        }
        if input_tokens is not None:
            payload["input_tokens"] = input_tokens
        if output_tokens is not None:
            payload["output_tokens"] = output_tokens
        if cost_usd is not None:
            payload["cost_usd"] = cost_usd
        if contexts is not None:
            payload["contexts"] = contexts
        if run_name is not None:
            payload["run_name"] = run_name
        r = self._client.post(f"{self.api_url}/traces", json=payload)
        r.raise_for_status()
        return r.json()

    def span(
        self,
        *,
        prompt: str,
        model: str,
        run_name: Optional[str] = None,
        contexts: Optional[list[str]] = None,
    ) -> Span:
        return Span(
            tracer=self,
            prompt=prompt,
            model=model,
            run_name=run_name,
            contexts=contexts,
        )

    def trace(
        self,
        *,
        model: str,
        run_name: Optional[str] = None,
        contexts: Optional[list[str]] = None,
    ):
        def decorator(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                prompt = args[0] if args else kwargs.get("prompt", "")
                with self.span(
                    prompt=str(prompt),
                    model=model,
                    run_name=run_name,
                    contexts=contexts,
                ) as span:
                    result = fn(*args, **kwargs)
                    span.set_response(str(result))
                return result
            return wrapper
        return decorator

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EvalOpsTracer":
        return self

    def __exit__(self, *args) -> None:
        self.close()

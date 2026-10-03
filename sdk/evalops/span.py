import time
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .tracer import EvalOpsTracer

from .pricing import compute_cost


class Span:
    def __init__(
        self,
        tracer: "EvalOpsTracer",
        prompt: str,
        model: str,
        run_name: Optional[str] = None,
        contexts: Optional[list[str]] = None,
    ) -> None:
        self._tracer = tracer
        self._prompt = prompt
        self._model = model
        self._run_name = run_name
        self._contexts = contexts
        self._response: str = ""
        self._input_tokens: Optional[int] = None
        self._output_tokens: Optional[int] = None
        self._start: float = 0.0

    def set_response(self, text: str) -> None:
        self._response = text

    def set_tokens(self, *, input_tokens: int, output_tokens: int) -> None:
        self._input_tokens = input_tokens
        self._output_tokens = output_tokens

    def set_contexts(self, contexts: list[str]) -> None:
        self._contexts = contexts

    def __enter__(self) -> "Span":
        self._start = time.monotonic()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if exc_type is not None:
            return False
        latency_ms = int((time.monotonic() - self._start) * 1000)
        cost_usd: Optional[float] = None
        if self._input_tokens is not None and self._output_tokens is not None:
            raw = compute_cost(self._model, self._input_tokens, self._output_tokens)
            cost_usd = raw if raw > 0 else None
        self._tracer.log(
            prompt=self._prompt,
            response=self._response,
            model=self._model,
            latency_ms=latency_ms,
            input_tokens=self._input_tokens,
            output_tokens=self._output_tokens,
            cost_usd=cost_usd,
            contexts=self._contexts,
            run_name=self._run_name,
        )
        return False

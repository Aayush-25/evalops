import httpx


class EvalOpsTracer:
    """Thin client for logging LLM traces to the EvalOps API."""

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
        **kwargs,
    ) -> dict:
        payload = {
            "prompt": prompt,
            "response": response,
            "model": model,
            "latency_ms": latency_ms,
            "project_name": self.project_name,
            **kwargs,
        }
        r = self._client.post(f"{self.api_url}/traces", json=payload)
        r.raise_for_status()
        return r.json()

    def close(self) -> None:
        self._client.close()

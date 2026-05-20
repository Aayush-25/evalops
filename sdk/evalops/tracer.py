import httpx


class EvalOpsTracer:
    """Thin client for logging LLM traces to the EvalOps API."""

    def __init__(self, api_url: str, project_name: str = "default") -> None:
        self.api_url = api_url.rstrip("/")
        self.project_name = project_name

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
        with httpx.Client() as client:
            r = client.post(f"{self.api_url}/traces", json=payload)
            r.raise_for_status()
            return r.json()

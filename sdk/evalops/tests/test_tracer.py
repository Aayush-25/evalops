import json

import httpx
import pytest
import respx

from evalops import EvalOpsTracer

BASE_URL = "http://fake-api"

FAKE_TRACE = {
    "id": "00000000-0000-0000-0000-000000000001",
    "project_name": "default",
    "run_name": None,
    "prompt": "What is 2+2?",
    "response": "4",
    "model": "gpt-4o",
    "latency_ms": 100,
    "input_tokens": None,
    "output_tokens": None,
    "cost_usd": None,
    "contexts": None,
    "faithfulness": None,
    "relevancy": None,
    "eval_status": "pending",
    "created_at": "2026-05-21T00:00:00+00:00",
}


class TestLog:
    @respx.mock
    def test_posts_to_traces_endpoint(self):
        route = respx.post(f"{BASE_URL}/traces").mock(
            return_value=httpx.Response(201, json=FAKE_TRACE)
        )
        with EvalOpsTracer(api_url=BASE_URL) as tracer:
            result = tracer.log(prompt="What is 2+2?", response="4", model="gpt-4o", latency_ms=100)
        assert route.called
        assert result["eval_status"] == "pending"

    @respx.mock
    def test_raises_on_non_2xx(self):
        respx.post(f"{BASE_URL}/traces").mock(return_value=httpx.Response(500))
        with EvalOpsTracer(api_url=BASE_URL) as tracer:
            with pytest.raises(httpx.HTTPStatusError):
                tracer.log(prompt="p", response="r", model="gpt-4o", latency_ms=10)

    @respx.mock
    def test_sends_correct_json_body(self):
        route = respx.post(f"{BASE_URL}/traces").mock(
            return_value=httpx.Response(201, json=FAKE_TRACE)
        )
        with EvalOpsTracer(api_url=BASE_URL, project_name="my-proj") as tracer:
            tracer.log(
                prompt="hello",
                response="world",
                model="gpt-4o",
                latency_ms=50,
                input_tokens=10,
                output_tokens=5,
            )
        body = json.loads(route.calls[0].request.content)
        assert body["prompt"] == "hello"
        assert body["project_name"] == "my-proj"
        assert body["input_tokens"] == 10

    @respx.mock
    def test_omits_none_fields_from_body(self):
        route = respx.post(f"{BASE_URL}/traces").mock(
            return_value=httpx.Response(201, json=FAKE_TRACE)
        )
        with EvalOpsTracer(api_url=BASE_URL) as tracer:
            tracer.log(prompt="p", response="r", model="gpt-4o", latency_ms=10)
        body = json.loads(route.calls[0].request.content)
        assert "input_tokens" not in body
        assert "cost_usd" not in body
        assert "run_name" not in body


class TestContextManager:
    @respx.mock
    def test_close_called_on_exit(self):
        respx.post(f"{BASE_URL}/traces").mock(
            return_value=httpx.Response(201, json=FAKE_TRACE)
        )
        with EvalOpsTracer(api_url=BASE_URL) as tracer:
            pass
        with pytest.raises(RuntimeError):
            tracer._client.get(BASE_URL)


class TestSpanIntegration:
    @respx.mock
    def test_span_sends_trace_on_exit(self):
        route = respx.post(f"{BASE_URL}/traces").mock(
            return_value=httpx.Response(201, json=FAKE_TRACE)
        )
        with EvalOpsTracer(api_url=BASE_URL) as tracer:
            with tracer.span(prompt="What is 2+2?", model="gpt-4o") as span:
                span.set_response("4")
        body = json.loads(route.calls[0].request.content)
        assert body["prompt"] == "What is 2+2?"
        assert body["response"] == "4"
        assert body["model"] == "gpt-4o"

    @respx.mock
    def test_span_does_not_send_on_exception(self):
        route = respx.post(f"{BASE_URL}/traces").mock(
            return_value=httpx.Response(201, json=FAKE_TRACE)
        )
        with EvalOpsTracer(api_url=BASE_URL) as tracer:
            with pytest.raises(RuntimeError):
                with tracer.span(prompt="p", model="gpt-4o"):
                    raise RuntimeError("fail")
        assert not route.called


class TestDecorator:
    @respx.mock
    def test_sends_prompt_and_response(self):
        route = respx.post(f"{BASE_URL}/traces").mock(
            return_value=httpx.Response(201, json=FAKE_TRACE)
        )
        with EvalOpsTracer(api_url=BASE_URL) as tracer:
            @tracer.trace(model="gpt-4o")
            def ask(prompt: str) -> str:
                return "the answer"

            result = ask("a question")

        assert result == "the answer"
        body = json.loads(route.calls[0].request.content)
        assert body["prompt"] == "a question"
        assert body["response"] == "the answer"

    @respx.mock
    def test_preserves_return_value(self):
        respx.post(f"{BASE_URL}/traces").mock(
            return_value=httpx.Response(201, json=FAKE_TRACE)
        )
        with EvalOpsTracer(api_url=BASE_URL) as tracer:
            @tracer.trace(model="gpt-4o")
            def ask(prompt: str) -> str:
                return "42"

            assert ask("what?") == "42"

    @respx.mock
    def test_does_not_send_on_exception(self):
        route = respx.post(f"{BASE_URL}/traces").mock(
            return_value=httpx.Response(201, json=FAKE_TRACE)
        )
        with EvalOpsTracer(api_url=BASE_URL) as tracer:
            @tracer.trace(model="gpt-4o")
            def ask(prompt: str) -> str:
                raise RuntimeError("API failure")

            with pytest.raises(RuntimeError):
                ask("question")

        assert not route.called

    @respx.mock
    def test_preserves_function_name(self):
        respx.post(f"{BASE_URL}/traces").mock(
            return_value=httpx.Response(201, json=FAKE_TRACE)
        )
        with EvalOpsTracer(api_url=BASE_URL) as tracer:
            @tracer.trace(model="gpt-4o")
            def my_llm_call(prompt: str) -> str:
                return "ok"

            assert my_llm_call.__name__ == "my_llm_call"

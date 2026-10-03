import time
from unittest.mock import MagicMock

import pytest

from evalops.span import Span


@pytest.fixture
def mock_tracer():
    tracer = MagicMock()
    tracer.log = MagicMock(return_value={"id": "fake-id"})
    return tracer


class TestSpanBasic:
    def test_enter_returns_self(self, mock_tracer):
        span = Span(tracer=mock_tracer, prompt="hi", model="gpt-4o")
        assert span.__enter__() is span

    def test_exit_calls_log_with_correct_fields(self, mock_tracer):
        span = Span(tracer=mock_tracer, prompt="hello", model="gpt-4o")
        span.set_response("world")
        with span:
            pass
        mock_tracer.log.assert_called_once()
        kw = mock_tracer.log.call_args.kwargs
        assert kw["prompt"] == "hello"
        assert kw["response"] == "world"
        assert kw["model"] == "gpt-4o"
        assert kw["latency_ms"] >= 0

    def test_latency_ms_is_positive(self, mock_tracer):
        with Span(tracer=mock_tracer, prompt="p", model="gpt-4o"):
            time.sleep(0.01)
        kw = mock_tracer.log.call_args.kwargs
        assert kw["latency_ms"] >= 10

    def test_default_response_is_empty_string(self, mock_tracer):
        with Span(tracer=mock_tracer, prompt="p", model="gpt-4o"):
            pass
        assert mock_tracer.log.call_args.kwargs["response"] == ""


class TestSpanTokensAndCost:
    def test_set_tokens_passed_to_log(self, mock_tracer):
        with Span(tracer=mock_tracer, prompt="p", model="gpt-4o") as span:
            span.set_tokens(input_tokens=10, output_tokens=5)
        kw = mock_tracer.log.call_args.kwargs
        assert kw["input_tokens"] == 10
        assert kw["output_tokens"] == 5

    def test_cost_computed_for_known_model(self, mock_tracer):
        # gpt-4o: $5/M input → 1M tokens = $5.00
        with Span(tracer=mock_tracer, prompt="p", model="gpt-4o") as span:
            span.set_tokens(input_tokens=1_000_000, output_tokens=0)
        kw = mock_tracer.log.call_args.kwargs
        assert kw["cost_usd"] == pytest.approx(5.0)

    def test_cost_is_none_for_unknown_model(self, mock_tracer):
        with Span(tracer=mock_tracer, prompt="p", model="unknown-xyz") as span:
            span.set_tokens(input_tokens=100, output_tokens=50)
        assert mock_tracer.log.call_args.kwargs["cost_usd"] is None

    def test_cost_is_none_when_no_tokens_provided(self, mock_tracer):
        with Span(tracer=mock_tracer, prompt="p", model="gpt-4o"):
            pass
        assert mock_tracer.log.call_args.kwargs["cost_usd"] is None


class TestSpanContexts:
    def test_set_contexts_passed_to_log(self, mock_tracer):
        with Span(tracer=mock_tracer, prompt="p", model="gpt-4o") as span:
            span.set_contexts(["doc1", "doc2"])
        assert mock_tracer.log.call_args.kwargs["contexts"] == ["doc1", "doc2"]

    def test_contexts_via_constructor(self, mock_tracer):
        with Span(tracer=mock_tracer, prompt="p", model="gpt-4o", contexts=["ctx"]):
            pass
        assert mock_tracer.log.call_args.kwargs["contexts"] == ["ctx"]

    def test_run_name_via_constructor(self, mock_tracer):
        with Span(tracer=mock_tracer, prompt="p", model="gpt-4o", run_name="run-1"):
            pass
        assert mock_tracer.log.call_args.kwargs["run_name"] == "run-1"


class TestSpanOnException:
    def test_log_not_called_on_exception(self, mock_tracer):
        with pytest.raises(ValueError):
            with Span(tracer=mock_tracer, prompt="p", model="gpt-4o"):
                raise ValueError("oops")
        mock_tracer.log.assert_not_called()

    def test_exception_is_reraised(self, mock_tracer):
        with pytest.raises(RuntimeError, match="boom"):
            with Span(tracer=mock_tracer, prompt="p", model="gpt-4o"):
                raise RuntimeError("boom")

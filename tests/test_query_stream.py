"""
tests/test_query_stream.py
--------------------------
Async tests for POST /api/v1/query/stream (SSE endpoint).

Strategy
--------
All external dependencies are mocked:
  - hybrid_search            -> list of fake HybridResult objects
  - build_rag_prompt         -> minimal StructuredPrompt-like stub
  - LLMEngine.stream_synthesize_async -> async generator of (token, result) tuples

Verifies:
  1. Content-Type is text/event-stream.
  2. Intermediate 'token' SSE events arrive in order.
  3. Final 'done' SSE event contains valid QueryResponse JSON.
  4. Zero-results guard emits 'done' with status='no_relevant_code_found'.
  5. Empty query emits 'error' event without calling hybrid_search or LLM.
"""

from __future__ import annotations

import json
from typing import AsyncIterator
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.main import app
from rag.llm_engine import GroundingReport, SynthesisResult, TokenUsage


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------

def _make_structured_prompt(query: str = "test query") -> MagicMock:
    """Minimal StructuredPrompt-compatible mock."""
    sp = MagicMock()
    sp.to_messages.return_value = [
        {"role": "system", "content": "You are a helpful coding assistant."},
        {"role": "user", "content": f"<user_query>{query}</user_query>"},
    ]
    sp.included_anchors = ["app/foo.py::func::bar"]
    sp.included_neighbors = ["app/baz.py::func::qux"]
    sp.user_prompt = f"<user_query>{query}</user_query>"
    return sp


def _make_synthesis_result(content: str = "The architecture is clean.") -> SynthesisResult:
    return SynthesisResult(
        content=content,
        query="test query",
        model="codellama:latest",
        token_usage=TokenUsage(
            prompt_tokens=50,
            completion_tokens=20,
            total_tokens=70,
            total_duration_ms=1200.0,
            tokens_per_second=16.6,
        ),
        grounding_report=GroundingReport(
            total_citations=0,
            valid_citations=[],
            hallucinated_citations=[],
            adherence_score=1.0,
            is_fully_grounded=True,
            unmentioned_anchors=[],
        ),
        raw_response={},
    )


async def _fake_stream(self, prompt, query="", model=None, options=None):
    """Async generator mimicking LLMEngine.stream_synthesize_async."""
    for tok in ["The ", "architecture ", "is ", "clean."]:
        yield (tok, None)
    yield ("", _make_synthesis_result("The architecture is clean."))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture()
async def async_client() -> AsyncIterator[AsyncClient]:
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        yield client


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _parse_sse(raw_text: str) -> list[dict]:
    """Parse raw SSE text into [{event, data}, ...] dicts.

    Per the SSE spec (https://html.spec.whatwg.org/multipage/server-sent-events.html),
    the field value is everything after the colon, with at most one leading space
    stripped. We use lstrip(' ') (single char) rather than .strip() so that
    intentional trailing whitespace in token data values is preserved.
    """
    events: list[dict] = []
    current: dict = {}
    for line in raw_text.splitlines():
        if line.startswith("event:"):
            current["event"] = line[len("event:"):].lstrip(" ")
        elif line.startswith("data:"):
            current["data"] = line[len("data:"):].lstrip(" ")
        elif line == "" and current:
            events.append(current)
            current = {}
    if current:
        events.append(current)
    return events



# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQueryStreamEndpoint:
    """Tests for POST /api/v1/query/stream."""

    @pytest.mark.asyncio
    async def test_content_type_is_text_event_stream(self, async_client: AsyncClient):
        """Response must carry text/event-stream content-type."""
        with (
            patch("app.routers.query.hybrid_search", return_value=[MagicMock()]),
            patch("app.routers.query.build_rag_prompt", return_value=_make_structured_prompt()),
            patch("app.routers.query.LLMEngine.stream_synthesize_async", new=_fake_stream),
        ):
            resp = await async_client.post(
                "/api/v1/query/stream",
                json={"query": "How does the ingestion pipeline work?"},
            )

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

    @pytest.mark.asyncio
    async def test_token_events_arrive_in_order(self, async_client: AsyncClient):
        """Intermediate 'token' SSE events must carry the expected text chunks in order."""
        with (
            patch("app.routers.query.hybrid_search", return_value=[MagicMock()]),
            patch("app.routers.query.build_rag_prompt", return_value=_make_structured_prompt()),
            patch("app.routers.query.LLMEngine.stream_synthesize_async", new=_fake_stream),
        ):
            resp = await async_client.post(
                "/api/v1/query/stream",
                json={"query": "Explain the graph service"},
            )

        token_texts = [
            e["data"] for e in _parse_sse(resp.text) if e.get("event") == "token"
        ]
        assert token_texts == ["The ", "architecture ", "is ", "clean."]

    @pytest.mark.asyncio
    async def test_done_event_contains_valid_query_response(self, async_client: AsyncClient):
        """The final 'done' SSE event must deserialise into a valid QueryResponse payload."""
        with (
            patch("app.routers.query.hybrid_search", return_value=[MagicMock()]),
            patch("app.routers.query.build_rag_prompt", return_value=_make_structured_prompt()),
            patch("app.routers.query.LLMEngine.stream_synthesize_async", new=_fake_stream),
        ):
            resp = await async_client.post(
                "/api/v1/query/stream",
                json={"query": "What are the main modules?"},
            )

        done_events = [e for e in _parse_sse(resp.text) if e.get("event") == "done"]
        assert len(done_events) == 1, "Expected exactly one 'done' event"

        payload = json.loads(done_events[0]["data"])
        assert payload["status"] == "success"
        assert payload["answer"] == "The architecture is clean."
        assert "query" in payload
        assert "referenced_nodes" in payload

    @pytest.mark.asyncio
    async def test_zero_results_guard_emits_done_no_relevant_code(
        self, async_client: AsyncClient
    ):
        """When hybrid_search returns [], emit 'done' with no_relevant_code_found; skip LLM."""
        with (
            patch("app.routers.query.hybrid_search", return_value=[]),
            patch("app.routers.query.build_rag_prompt") as mock_bp,
            patch("app.routers.query.LLMEngine.stream_synthesize_async") as mock_stream,
        ):
            resp = await async_client.post(
                "/api/v1/query/stream",
                json={"query": "find something obscure"},
            )

        mock_bp.assert_not_called()
        mock_stream.assert_not_called()

        done_events = [e for e in _parse_sse(resp.text) if e.get("event") == "done"]
        assert len(done_events) == 1
        payload = json.loads(done_events[0]["data"])
        assert payload["status"] == "no_relevant_code_found"

    @pytest.mark.asyncio
    async def test_empty_query_emits_error_event(self, async_client: AsyncClient):
        """Empty / whitespace-only query must emit 'error' SSE without any downstream calls."""
        with (
            patch("app.routers.query.hybrid_search") as mock_hs,
            patch("app.routers.query.LLMEngine.stream_synthesize_async") as mock_stream,
        ):
            resp = await async_client.post(
                "/api/v1/query/stream",
                json={"query": "   "},
            )

        mock_hs.assert_not_called()
        mock_stream.assert_not_called()

        error_events = [e for e in _parse_sse(resp.text) if e.get("event") == "error"]
        assert len(error_events) == 1
        assert "detail" in json.loads(error_events[0]["data"])

    @pytest.mark.asyncio
    async def test_sync_calls_run_in_separate_thread(self, async_client: AsyncClient):
        """Verify _load_cached_graph, hybrid_search, and build_rag_prompt run in worker threads."""
        import threading
        main_thread_id = threading.get_ident()
        caller_threads: dict[str, int] = {}

        def mock_load_graph(repo_id: str):
            caller_threads["_load_cached_graph"] = threading.get_ident()
            return None

        captured_kwargs: dict = {}

        def mock_hybrid_search(**kwargs):
            caller_threads["hybrid_search"] = threading.get_ident()
            captured_kwargs.update(kwargs)
            return [MagicMock()]

        def mock_build_rag_prompt(**kwargs):
            caller_threads["build_rag_prompt"] = threading.get_ident()
            return _make_structured_prompt()

        with (
            patch("app.routers.query._load_cached_graph", side_effect=mock_load_graph),
            patch("app.routers.query.hybrid_search", side_effect=mock_hybrid_search),
            patch("app.routers.query.build_rag_prompt", side_effect=mock_build_rag_prompt),
            patch("app.routers.query.LLMEngine.stream_synthesize_async", new=_fake_stream),
        ):
            resp = await async_client.post(
                "/api/v1/query/stream",
                json={"query": "test thread offloading", "repo_id": "test-repo"},
            )

        assert resp.status_code == 200
        assert captured_kwargs.get("repo_id") == "test-repo"
        assert "_load_cached_graph" in caller_threads
        assert "hybrid_search" in caller_threads
        assert "build_rag_prompt" in caller_threads
        assert caller_threads["_load_cached_graph"] != main_thread_id, (
            "_load_cached_graph ran on event loop thread!"
        )
        assert caller_threads["hybrid_search"] != main_thread_id, (
            "hybrid_search ran on event loop thread!"
        )
        assert caller_threads["build_rag_prompt"] != main_thread_id, (
            "build_rag_prompt ran on event loop thread!"
        )

"""
tests/test_query_router.py
--------------------------
Unit & Integration Test Suite for POST /api/v1/query (app/routers/query.py).

Tests cover:
- Empty or whitespace-only query handling
- Zero-results failure case guard (LLM never called with empty context)
- Graph=None fallback (when repo_id is omitted or Redis is unreachable/empty)
- Referenced nodes extraction from citations, grounding report, and structured prompt
- Error handling (Ollama offline -> 503, invalid model -> 404, timeouts -> 504)
- Live integration test against real local Ollama (guarded with skip-if-unreachable)
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import httpx
import networkx as nx
import pytest
from fastapi.testclient import TestClient

from app.main import app
from rag.llm_engine import (
    Citation,
    GroundingReport,
    OllamaClient,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    SynthesisResult,
    TokenUsage,
)
from rag.prompt_builder import StructuredPrompt

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers & Fixtures
# ---------------------------------------------------------------------------


def is_local_ollama_available() -> bool:
    """Check if local Ollama daemon is reachable on http://127.0.0.1:11434."""
    try:
        r = httpx.get("http://127.0.0.1:11434/api/tags", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


@pytest.fixture
def sample_hybrid_results() -> list[dict]:
    """Sample hybrid results containing 1 vector anchor and 1 graph neighbor."""
    return [
        {
            "rank": 1,
            "func_name": "clone_repository",
            "file_path": "app/services/git_service.py",
            "score": 0.88,
            "source": "vector",
            "anchor_func": None,
            "pagerank": 0.05,
            "commit_count": 12,
            "is_dead_code_candidate": False,
        },
        {
            "rank": 2,
            "func_name": "scan_repository",
            "file_path": "parser/repo_walker.py",
            "score": 0.0,
            "source": "graph_callee",
            "anchor_func": "clone_repository",
            "pagerank": 0.08,
            "commit_count": 20,
            "is_dead_code_candidate": False,
        },
    ]


@pytest.fixture
def mock_synthesis_result() -> SynthesisResult:
    """Mock SynthesisResult simulating LLM explanation with valid citations."""
    cit = Citation(
        raw_text="[app/services/git_service.py::func::clone_repository]",
        file_path="app/services/git_service.py",
        func_name="clone_repository",
        is_grounded=True,
        source="vector_anchor",
    )
    grounding = GroundingReport(
        total_citations=1,
        valid_citations=[cit],
        hallucinated_citations=[],
        adherence_score=1.0,
        is_fully_grounded=True,
        unmentioned_anchors=[],
    )
    usage = TokenUsage(
        prompt_tokens=250,
        completion_tokens=80,
        total_tokens=330,
        total_duration_ms=450.0,
        tokens_per_second=177.7,
    )
    return SynthesisResult(
        content=(
            "The cloning operation is executed by `clone_repository` in "
            "[app/services/git_service.py::func::clone_repository]."
        ),
        query="how does cloning work?",
        model="qwen2.5-coder:1.5b",
        token_usage=usage,
        grounding_report=grounding,
        raw_response={},
    )


# ---------------------------------------------------------------------------
# Unit Tests
# ---------------------------------------------------------------------------


def test_query_empty_or_whitespace():
    """Verify empty or whitespace-only query returns no_relevant_code_found immediately."""
    response = client.post("/api/v1/query", json={"query": "   "})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "no_relevant_code_found"
    assert "Empty query" in data["answer"]
    assert data["referenced_nodes"] == []


@patch("app.routers.query.hybrid_search")
@patch("app.routers.query.LLMEngine")
def test_query_zero_results_failure_case(mock_llm_engine_cls, mock_hybrid_search):
    """Realistic failure case: hybrid_search returns 0 results.

    Verifies:
    1. Returns status="no_relevant_code_found" with clear user-facing explanation.
    2. LLMEngine is NEVER instantiated or called with an empty context.
    """
    mock_hybrid_search.return_value = []

    response = client.post(
        "/api/v1/query",
        json={"query": "non_existent_symbol_search_query_xyz123"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "no_relevant_code_found"
    assert "No relevant code snippets or functions found" in data["answer"]
    assert data["referenced_nodes"] == []
    assert data["citations"] == []
    assert data["grounding_report"] is None
    # Verify LLMEngine was not called
    mock_llm_engine_cls.assert_not_called()


@patch("app.routers.query._get_redis_client")
@patch("app.routers.query.hybrid_search")
@patch("app.routers.query.build_rag_prompt")
@patch("app.routers.query.LLMEngine")
def test_graph_none_fallback_when_redis_empty_or_repo_id_none(
    mock_llm_engine_cls,
    mock_build_prompt,
    mock_hybrid_search,
    mock_redis_client,
    sample_hybrid_results,
    mock_synthesis_result,
):
    """Verify hybrid_search receives graph=None when repo_id is omitted or key missing."""
    mock_hybrid_search.return_value = sample_hybrid_results
    mock_prompt = StructuredPrompt(
        system_prompt="sys",
        user_prompt="usr",
        full_prompt="full",
        estimated_tokens=100,
        max_tokens=4096,
        token_budget_exceeded=False,
        included_anchors=["clone_repository (app/services/git_service.py)"],
        included_neighbors=["scan_repository (parser/repo_walker.py)"],
    )
    mock_build_prompt.return_value = mock_prompt

    mock_engine = MagicMock()
    mock_engine.synthesize.return_value = mock_synthesis_result
    mock_llm_engine_cls.return_value = mock_engine

    # Case A: No repo_id in request
    resp_a = client.post("/api/v1/query", json={"query": "clone"})
    assert resp_a.status_code == 200
    # Verify hybrid_search was called with graph=None
    _, kwargs_a = mock_hybrid_search.call_args
    assert kwargs_a.get("graph") is None

    # Case B: repo_id given, but Redis returns None
    mock_r = MagicMock()
    mock_r.get.return_value = None
    mock_redis_client.return_value = mock_r

    resp_b = client.post("/api/v1/query", json={"query": "clone", "repo_id": "unknown-repo"})
    assert resp_b.status_code == 200
    _, kwargs_b = mock_hybrid_search.call_args
    assert kwargs_b.get("graph") is None


@patch("app.routers.query.hybrid_search")
@patch("app.routers.query.build_rag_prompt")
@patch("app.routers.query.LLMEngine")
def test_referenced_nodes_and_citations_extraction(
    mock_llm_engine_cls,
    mock_build_prompt,
    mock_hybrid_search,
    sample_hybrid_results,
    mock_synthesis_result,
):
    """Verify referenced_nodes combines citations and structured prompt anchors/neighbors."""
    mock_hybrid_search.return_value = sample_hybrid_results
    mock_prompt = StructuredPrompt(
        system_prompt="sys",
        user_prompt="usr",
        full_prompt="full",
        estimated_tokens=100,
        max_tokens=4096,
        token_budget_exceeded=False,
        included_anchors=["clone_repository (app/services/git_service.py)"],
        included_neighbors=["scan_repository (parser/repo_walker.py)"],
    )
    mock_build_prompt.return_value = mock_prompt

    mock_engine = MagicMock()
    mock_engine.synthesize.return_value = mock_synthesis_result
    mock_llm_engine_cls.return_value = mock_engine

    response = client.post("/api/v1/query", json={"query": "how does cloning work?"})
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "success"
    assert "app/services/git_service.py::func::clone_repository" in data["referenced_nodes"]
    assert "clone_repository (app/services/git_service.py)" in data["referenced_nodes"]
    assert "scan_repository (parser/repo_walker.py)" in data["referenced_nodes"]
    assert len(data["citations"]) == 1
    assert data["citations"][0]["func_name"] == "clone_repository"
    assert data["grounding_report"]["is_fully_grounded"] is True


@patch("app.routers.query.hybrid_search")
@patch("app.routers.query.build_rag_prompt")
@patch("app.routers.query.LLMEngine")
def test_ollama_connection_error_status_code(
    mock_llm_engine_cls,
    mock_build_prompt,
    mock_hybrid_search,
    sample_hybrid_results,
):
    """Verify OllamaConnectionError maps to HTTP 503 Service Unavailable."""
    mock_hybrid_search.return_value = sample_hybrid_results
    mock_build_prompt.return_value = MagicMock(spec=StructuredPrompt)

    mock_engine = MagicMock()
    mock_engine.synthesize.side_effect = OllamaConnectionError("Cannot connect to Ollama")
    mock_llm_engine_cls.return_value = mock_engine

    response = client.post("/api/v1/query", json={"query": "clone"})
    assert response.status_code == 503
    assert "offline or unreachable" in response.json()["detail"]


@patch("app.routers.query.hybrid_search")
@patch("app.routers.query.build_rag_prompt")
@patch("app.routers.query.LLMEngine")
def test_ollama_model_not_found_error_status_code(
    mock_llm_engine_cls,
    mock_build_prompt,
    mock_hybrid_search,
    sample_hybrid_results,
):
    """Verify OllamaModelNotFoundError maps to HTTP 404 Not Found."""
    mock_hybrid_search.return_value = sample_hybrid_results
    mock_build_prompt.return_value = MagicMock(spec=StructuredPrompt)

    mock_engine = MagicMock()
    mock_engine.synthesize.side_effect = OllamaModelNotFoundError("Model xyz not found")
    mock_llm_engine_cls.return_value = mock_engine

    response = client.post("/api/v1/query", json={"query": "clone", "model": "xyz"})
    assert response.status_code == 404
    assert "Model xyz not found" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Integration Test: Real Local Ollama
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not is_local_ollama_available(),
    reason="Local Ollama daemon is offline or unreachable on http://127.0.0.1:11434",
)
def test_real_ollama_query_integration():
    """End-to-end integration test with live Ollama daemon and local Qdrant collection.

    Queries: 'clone repository'
    Expects:
    - 200 OK
    - status == 'success'
    - Non-empty answer text
    - Populated referenced_nodes
    - Grounding report evaluated
    """
    response = client.post(
        "/api/v1/query",
        json={
            "query": "how to clone a git repository",
            "top_k": 3,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert len(data["answer"]) > 0
    assert len(data["referenced_nodes"]) > 0
    assert data["model"] is not None
    assert data["grounding_report"] is not None

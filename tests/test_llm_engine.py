"""
tests/test_llm_engine.py
-------------------------
Unit & Integration Test Suite for rag/llm_engine.py.

Tests cover:
- OllamaConfig initialization and options payload formatting
- TokenUsage telemetry extraction and rate calculation
- Citation regex parsing and path normalization
- Grounding adherence evaluation (valid, hallucinated, unmentioned anchors)
- OllamaClient health checking, model discovery, and error handling
- Sync & async chat execution via mock HTTP transport
- Real-time token streaming generators (sync & async)
- LLMEngine orchestration with StructuredPrompt integration
"""

from __future__ import annotations

import asyncio
import json
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from rag.llm_engine import (
    CITATION_REGEX,
    Citation,
    GroundingReport,
    LLMEngine,
    LLMEngineError,
    OllamaClient,
    OllamaConfig,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    SynthesisResult,
    TokenUsage,
    evaluate_grounding,
    extract_citations,
    generate_mock_explanation,
    synthesize_architecture_explanation,
)
from rag.prompt_builder import StructuredPrompt, build_rag_prompt


# ---------------------------------------------------------------------------
# Test Fixtures & Mock Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_ollama_chat_response() -> dict:
    """Simulated standard response payload from Ollama /api/chat."""
    return {
        "model": "codellama",
        "created_at": "2026-09-11T12:00:00.000000Z",
        "message": {
            "role": "assistant",
            "content": (
                "The function `clone_repository` [app/services/git_service.py::func::clone_repository] "
                "clones the target repo. If an error occurs, it delegates cleanup to "
                "[app/services/git_service.py::func::cleanup_repo_directory]."
            ),
        },
        "done": True,
        "total_duration": 500_000_000,       # 500 ms in ns
        "load_duration": 10_000_000,         # 10 ms in ns
        "prompt_eval_count": 450,
        "prompt_eval_duration": 100_000_000, # 100 ms in ns
        "eval_count": 65,
        "eval_duration": 390_000_000,        # 390 ms in ns
    }


@pytest.fixture
def sample_structured_prompt() -> StructuredPrompt:
    """Sample StructuredPrompt for engine input tests."""
    return build_rag_prompt(
        query="How does cleanup happen on clone error?",
        hybrid_results=[
            {
                "rank": 1,
                "func_name": "clone_repository",
                "file_path": "app/services/git_service.py",
                "score": 0.88,
                "similarity_label": "HIGH CONFIDENCE",
                "pagerank": 0.05,
                "commit_count": 10,
                "is_dead_code_candidate": False,
                "source": "vector",
                "anchor_func": None,
            },
            {
                "rank": 2,
                "func_name": "cleanup_repo_directory",
                "file_path": "app/services/git_service.py",
                "score": 0.0,
                "similarity_label": "VERY WEAK / NOISE",
                "pagerank": 0.01,
                "commit_count": 2,
                "is_dead_code_candidate": False,
                "source": "graph_callee",
                "anchor_func": "clone_repository",
            },
        ],
        code_cache={
            "clone_repository": "def clone_repository(): cleanup_repo_directory()"
        },
        max_tokens=2000,
    )


# ---------------------------------------------------------------------------
# Unit Tests: Configuration & Telemetry
# ---------------------------------------------------------------------------

def test_ollama_config_defaults_and_overrides() -> None:
    """Verify OllamaConfig defaults and endpoint URL construction."""
    cfg = OllamaConfig()
    assert cfg.base_url == "http://localhost:11434"
    assert cfg.model == "codellama"
    assert cfg.chat_endpoint == "http://localhost:11434/api/chat"
    assert cfg.generate_endpoint == "http://localhost:11434/api/generate"
    assert cfg.tags_endpoint == "http://localhost:11434/api/tags"

    opts = cfg.to_options_dict()
    assert opts["temperature"] == 0.2
    assert opts["num_predict"] == 2048

    custom_cfg = OllamaConfig(
        base_url="http://remote-ollama:11434/",
        model="llama3",
        temperature=0.7,
        num_predict=1024,
    )
    assert custom_cfg.chat_endpoint == "http://remote-ollama:11434/api/chat"
    assert custom_cfg.to_options_dict()["temperature"] == 0.7


def test_token_usage_parsing(sample_ollama_chat_response: dict) -> None:
    """Verify conversion of Ollama nanosecond telemetry to millisecond metrics and tokens/sec."""
    usage = TokenUsage.from_ollama_response(sample_ollama_chat_response)
    assert usage.prompt_tokens == 450
    assert usage.completion_tokens == 65
    assert usage.total_tokens == 515
    assert usage.eval_duration_ms == 390.0
    assert usage.prompt_eval_duration_ms == 100.0
    assert usage.total_duration_ms == 500.0
    assert usage.tokens_per_second > 0
    # 65 tokens / 0.390s = ~166.67 tokens/sec
    assert 160.0 < usage.tokens_per_second < 170.0


def test_token_usage_fallback_duration() -> None:
    """Verify fallback wall time when Ollama total_duration is missing."""
    usage = TokenUsage.from_ollama_response({}, fallback_wall_time_ms=250.0)
    assert usage.prompt_tokens == 0
    assert usage.total_duration_ms == 250.0


# ---------------------------------------------------------------------------
# Unit Tests: Citation Extraction & Grounding Adherence
# ---------------------------------------------------------------------------

def test_extract_citations_valid() -> None:
    """Verify extraction of standard [file_path::func::<name>] citations."""
    text = (
        "The [app/services/git.py::func::clone] calls "
        "[worker/tasks.py::func::process_task] and returns."
    )
    citations = extract_citations(text)
    assert len(citations) == 2
    raw1, path1, func1 = citations[0]
    assert raw1 == "[app/services/git.py::func::clone]"
    assert path1 == "app/services/git.py"
    assert func1 == "clone"

    raw2, path2, func2 = citations[1]
    assert raw2 == "[worker/tasks.py::func::process_task]"
    assert path2 == "worker/tasks.py"
    assert func2 == "process_task"


def test_extract_citations_windows_path_normalization() -> None:
    """Verify backslashes in Windows filepaths are normalized to forward slashes."""
    text = "Implemented in [app\\services\\ast_engine.py::func::extract_chunks]."
    citations = extract_citations(text)
    assert len(citations) == 1
    _, path, func = citations[0]
    assert path == "app/services/ast_engine.py"
    assert func == "extract_chunks"


def test_evaluate_grounding_perfect_adherence() -> None:
    """Verify 100% adherence score when all cited functions exist in context."""
    anchors = ["app/services/git_service.py::func::clone_repository"]
    neighbors = ["app/services/git_service.py::func::cleanup_repo_directory (graph_callee)"]
    text = (
        "Execution starts at [app/services/git_service.py::func::clone_repository] "
        "and delegates to [app/services/git_service.py::func::cleanup_repo_directory]."
    )
    report = evaluate_grounding(text, anchors, neighbors)
    assert report.total_citations == 2
    assert len(report.valid_citations) == 2
    assert len(report.hallucinated_citations) == 0
    assert report.adherence_score == 1.0
    assert report.is_fully_grounded is True
    assert len(report.unmentioned_anchors) == 0


def test_evaluate_grounding_hallucination_detected() -> None:
    """Verify hallucinated citations reduce the adherence score and are flagged."""
    anchors = ["app/services/git_service.py::func::clone_repository"]
    neighbors = ["app/services/git_service.py::func::cleanup_repo_directory"]
    text = (
        "Call [app/services/git_service.py::func::clone_repository], then call "
        "[imaginary/module.py::func::non_existent_function] which was fabricated."
    )
    report = evaluate_grounding(text, anchors, neighbors)
    assert report.total_citations == 2
    assert len(report.valid_citations) == 1
    assert len(report.hallucinated_citations) == 1
    assert report.adherence_score == 0.5
    assert report.is_fully_grounded is False
    assert report.hallucinated_citations[0].func_name == "non_existent_function"


def test_evaluate_grounding_zero_citations() -> None:
    """When no citations are present, consider grounded (e.g. admitting ignorance)."""
    report = evaluate_grounding("I cannot answer from the given context.", [], [])
    assert report.total_citations == 0
    assert report.adherence_score == 1.0
    assert report.is_fully_grounded is True


# ---------------------------------------------------------------------------
# Unit Tests: Ollama Client Sync & Async via Mock Transport
# ---------------------------------------------------------------------------

def test_ollama_client_health_check_online() -> None:
    """Test health check returns True when Ollama returns HTTP 200."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"models": [{"name": "codellama:latest"}]})

    transport = httpx.MockTransport(handler)
    mock_client = httpx.Client(transport=transport)
    client = OllamaClient(sync_client=mock_client)

    assert client.health_check() is True
    assert client.is_model_available("codellama") is True
    assert client.is_model_available("mistral") is False


def test_ollama_client_health_check_offline() -> None:
    """Test health check returns False when connection is refused."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(handler)
    mock_client = httpx.Client(transport=transport)
    client = OllamaClient(sync_client=mock_client)

    assert client.health_check() is False


def test_ollama_client_chat_success(sample_ollama_chat_response: dict) -> None:
    """Test standard synchronous /api/chat execution."""
    captured_payload: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal captured_payload
        captured_payload = json.loads(request.content.decode("utf-8"))
        return httpx.Response(200, json=sample_ollama_chat_response)

    transport = httpx.MockTransport(handler)
    mock_client = httpx.Client(transport=transport)
    client = OllamaClient(sync_client=mock_client)

    messages = [{"role": "user", "content": "Explain error handling"}]
    res = client.chat(messages=messages)

    assert res["model"] == "codellama"
    assert "clone_repository" in res["message"]["content"]
    assert captured_payload["stream"] is False
    assert captured_payload["messages"] == messages


def test_ollama_client_model_not_found() -> None:
    """Verify HTTP 404 maps to OllamaModelNotFoundError."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'nonexistent' not found"})

    transport = httpx.MockTransport(handler)
    mock_client = httpx.Client(transport=transport)
    client = OllamaClient(sync_client=mock_client)

    with pytest.raises(OllamaModelNotFoundError) as exc_info:
        client.chat(messages=[{"role": "user", "content": "hello"}], model="nonexistent")
    assert "Pull it via: `ollama pull nonexistent`" in str(exc_info.value)


def test_ollama_client_chat_stream() -> None:
    """Verify token streaming reads chunk lines sequentially."""
    stream_lines = [
        json.dumps({"message": {"content": "Hello "}, "done": False}) + "\n",
        json.dumps({"message": {"content": "world!"}, "done": False}) + "\n",
        json.dumps({"message": {"content": ""}, "done": True, "eval_count": 2}) + "\n",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content="".join(stream_lines).encode("utf-8"))

    transport = httpx.MockTransport(handler)
    mock_client = httpx.Client(transport=transport)
    client = OllamaClient(sync_client=mock_client)

    chunks = list(client.chat_stream(messages=[{"role": "user", "content": "hi"}]))
    assert len(chunks) == 3
    words = [c.get("message", {}).get("content", "") for c in chunks]
    assert "".join(words) == "Hello world!"
    assert chunks[-1]["done"] is True


# ---------------------------------------------------------------------------
# Unit Tests: LLMEngine High-Level Orchestration
# ---------------------------------------------------------------------------

def test_llm_engine_synthesize_with_structured_prompt(
    sample_structured_prompt: StructuredPrompt,
    sample_ollama_chat_response: dict,
) -> None:
    """Verify LLMEngine consuming StructuredPrompt and returning SynthesisResult."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=sample_ollama_chat_response)

    transport = httpx.MockTransport(handler)
    mock_client = httpx.Client(transport=transport)
    client = OllamaClient(sync_client=mock_client)
    engine = LLMEngine(client=client)

    result = engine.synthesize(sample_structured_prompt)

    assert isinstance(result, SynthesisResult)
    assert result.model == "codellama"
    assert result.query == "How does cleanup happen on clone error?"
    assert result.token_usage.prompt_tokens == 450
    assert result.token_usage.completion_tokens == 65
    assert len(result.citations) == 2
    assert result.is_grounded is True
    assert result.grounding_report.adherence_score == 1.0

    dict_repr = result.to_dict()
    assert dict_repr["grounding"]["is_fully_grounded"] is True
    assert len(dict_repr["grounding"]["valid_citations"]) == 2


def test_llm_engine_stream_synthesize(
    sample_structured_prompt: StructuredPrompt,
) -> None:
    """Verify stream_synthesize yields tokens and returns final SynthesisResult."""
    stream_lines = [
        json.dumps({
            "message": {"content": "Cloning is managed by "},
            "done": False,
        }) + "\n",
        json.dumps({
            "message": {"content": "[app/services/git_service.py::func::clone_repository]."},
            "done": False,
        }) + "\n",
        json.dumps({
            "message": {"content": ""},
            "done": True,
            "prompt_eval_count": 300,
            "eval_count": 15,
            "total_duration": 200_000_000,
        }) + "\n",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content="".join(stream_lines).encode("utf-8"))

    transport = httpx.MockTransport(handler)
    mock_client = httpx.Client(transport=transport)
    client = OllamaClient(sync_client=mock_client)
    engine = LLMEngine(client=client)

    tokens: list[str] = []
    final_res: SynthesisResult | None = None

    for token, res in engine.stream_synthesize(sample_structured_prompt):
        if token:
            tokens.append(token)
        if res is not None:
            final_res = res

    assert "".join(tokens) == "Cloning is managed by [app/services/git_service.py::func::clone_repository]."
    assert final_res is not None
    assert final_res.token_usage.prompt_tokens == 300
    assert final_res.token_usage.completion_tokens == 15
    assert len(final_res.citations) == 1
    assert final_res.is_grounded is True


def test_generate_mock_explanation() -> None:
    """Verify mock explanation generator produces well-grounded citations."""
    anchors = ["app/services/git_service.py::func::clone_repository"]
    neighbors = ["app/services/git_service.py::func::cleanup_repo_directory (graph_callee)"]
    res = generate_mock_explanation("How does it work?", anchors, neighbors)

    assert res.is_grounded is True
    assert res.grounding_report.adherence_score == 1.0
    assert len(res.grounding_report.valid_citations) >= 2
    assert "clone_repository" in res.content


def test_ollama_client_retry_and_connection_error() -> None:
    """Verify client attempts configured retries upon network connection errors."""
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        raise httpx.ConnectError("Connection refused")

    transport = httpx.MockTransport(handler)
    mock_client = httpx.Client(transport=transport)
    cfg = OllamaConfig(max_retries=2, backoff_factor=0.01)
    client = OllamaClient(config=cfg, sync_client=mock_client)

    with pytest.raises(OllamaConnectionError) as exc_info:
        client.chat(messages=[{"role": "user", "content": "ping"}])

    # 1 initial attempt + 2 retries = 3 attempts total
    assert attempts == 3
    assert "Failed to connect to Ollama" in str(exc_info.value)


def test_ollama_client_timeout_error() -> None:
    """Verify timeout exception triggers OllamaTimeoutError."""
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Request timed out")

    transport = httpx.MockTransport(handler)
    mock_client = httpx.Client(transport=transport)
    cfg = OllamaConfig(max_retries=0)
    client = OllamaClient(config=cfg, sync_client=mock_client)

    with pytest.raises(OllamaTimeoutError):
        client.chat(messages=[{"role": "user", "content": "ping"}])


@pytest.mark.anyio
async def test_ollama_client_chat_async(sample_ollama_chat_response: dict) -> None:
    """Verify asynchronous chat execution."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=sample_ollama_chat_response)

    transport = httpx.MockTransport(handler)
    mock_async_client = httpx.AsyncClient(transport=transport)
    client = OllamaClient(async_client=mock_async_client)

    res = await client.chat_async(messages=[{"role": "user", "content": "hello"}])
    assert res["model"] == "codellama"
    assert "clone_repository" in res["message"]["content"]


@pytest.mark.anyio
async def test_llm_engine_stream_synthesize_async(
    sample_structured_prompt: StructuredPrompt,
) -> None:
    """Verify asynchronous streaming generator on LLMEngine."""
    stream_lines = [
        json.dumps({"message": {"content": "Architecture summary: "}, "done": False}) + "\n",
        json.dumps({
            "message": {"content": "[app/services/git_service.py::func::clone_repository]"},
            "done": False,
        }) + "\n",
        json.dumps({
            "message": {"content": ""},
            "done": True,
            "prompt_eval_count": 200,
            "eval_count": 25,
            "total_duration": 150_000_000,
        }) + "\n",
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content="".join(stream_lines).encode("utf-8"))

    transport = httpx.MockTransport(handler)
    mock_async_client = httpx.AsyncClient(transport=transport)
    client = OllamaClient(async_client=mock_async_client)
    engine = LLMEngine(client=client)

    collected: list[str] = []
    final_res: SynthesisResult | None = None

    async for token, res in engine.stream_synthesize_async(sample_structured_prompt):
        if token:
            collected.append(token)
        if res is not None:
            final_res = res

    assert "".join(collected) == "Architecture summary: [app/services/git_service.py::func::clone_repository]"
    assert final_res is not None
    assert final_res.token_usage.prompt_tokens == 200
    assert final_res.is_grounded is True


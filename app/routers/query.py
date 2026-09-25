"""
routers/query.py
================
Exposes two complementary endpoints for Graph RAG architectural code queries:

  • POST /api/v1/query        – synchronous, full JSON response (blocking, thread-pool dispatched)
  • POST /api/v1/query/stream – async SSE stream; tokens arrive in real-time via text/event-stream

Both endpoints share the same pipeline up to the synthesis step:
1. Accept QueryRequest with natural language question and optional repo_id.
2. If repo_id is supplied, look up the serialized NetworkX graph in Redis under key "graph:{repo_id}".
   Falls back gracefully to graph=None (pure vector search) if not found or Redis is unreachable.
3. Call hybrid_search(query, graph=graph, ...) to retrieve vector anchors and 1-hop graph neighbors.
4. Zero-results Guard: If hybrid_search returns 0 matches, immediately return a clear
   "no_relevant_code_found" response without invoking the LLM with empty context.
5. If matches found: Call build_rag_prompt(...) to format grounded, token-budgeted XML context.

Synthesis diverges:
  • /query        → LLMEngine().synthesize(...)             (non-streaming, full SynthesisResult)
  • /query/stream → LLMEngine().stream_synthesize_async(...) (async generator → SSE)

SSE Event Types emitted by /query/stream:
  • event: token  – each text chunk from Ollama as it streams
  • event: done   – final JSON payload (same fields as QueryResponse) once streaming completes
  • event: error  – JSON error detail if the pipeline fails mid-stream
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import networkx as nx
import redis
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from rag.hybrid_retriever import HybridResult, hybrid_search
from rag.llm_engine import (
    Citation,
    GroundingReport,
    LLMEngine,
    LLMEngineError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    SynthesisResult,
)
from rag.prompt_builder import StructuredPrompt, build_rag_prompt

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["query"])

# ---------------------------------------------------------------------------
# Redis configuration (synchronous client for thread-pool execution)
# ---------------------------------------------------------------------------
REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))


def _get_redis_client() -> redis.Redis:
    """Return a synchronous Redis client instance."""
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


# ---------------------------------------------------------------------------
# Request and Response Pydantic Models
# ---------------------------------------------------------------------------


class QueryRequest(BaseModel):
    """Payload for natural language code architecture querying."""

    query: str = Field(..., description="Developer question or inquiry about the codebase.")
    repo_id: Optional[str] = Field(
        default=None,
        description="Optional repository identifier used to fetch cached NetworkX graph from Redis.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Number of vector anchors to retrieve from Qdrant.",
    )
    score_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional minimum cosine similarity cutoff for vector retrieval.",
    )
    include_graph_neighbors: bool = Field(
        default=True,
        description="Whether to perform 1-hop caller/callee graph expansion on vector anchors.",
    )
    model: Optional[str] = Field(
        default=None,
        description="Optional Ollama model override (e.g. 'qwen2.5-coder:1.5b').",
    )


class CitationModel(BaseModel):
    """Structured representation of a code reference cited in the synthesis."""

    raw_text: str
    file_path: str
    func_name: str
    is_grounded: bool
    source: str
    identifier: str

    @classmethod
    def from_citation(cls, c: Citation) -> CitationModel:
        return cls(
            raw_text=c.raw_text,
            file_path=c.file_path,
            func_name=c.func_name,
            is_grounded=c.is_grounded,
            source=c.source,
            identifier=c.identifier,
        )


class GroundingSummary(BaseModel):
    """Telemetry detailing model adherence to retrieved code context."""

    adherence_score: float
    is_fully_grounded: bool
    total_citations: int
    valid_count: int
    hallucinated_count: int
    unmentioned_anchors: list[str] = []

    @classmethod
    def from_report(cls, report: GroundingReport) -> GroundingSummary:
        return cls(
            adherence_score=report.adherence_score,
            is_fully_grounded=report.is_fully_grounded,
            total_citations=report.total_citations,
            valid_count=len(report.valid_citations),
            hallucinated_count=len(report.hallucinated_citations),
            unmentioned_anchors=report.unmentioned_anchors,
        )


class TokenUsageSummary(BaseModel):
    """Inference token metrics and timing information."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    tokens_per_second: float = 0.0


class QueryResponse(BaseModel):
    """Complete response returned by POST /api/v1/query."""

    query: str
    answer: str
    referenced_nodes: list[str] = Field(
        default_factory=list,
        description="All function and file nodes referenced in the answer, citations, or prompt context.",
    )
    citations: list[CitationModel] = Field(
        default_factory=list,
        description="Extracted inline citations [file::func::<name>].",
    )
    included_anchors: list[str] = Field(
        default_factory=list,
        description="Primary vector anchor functions included in the context.",
    )
    included_neighbors: list[str] = Field(
        default_factory=list,
        description="1-hop structural graph neighbor functions included in the context.",
    )
    grounding_report: Optional[GroundingSummary] = None
    token_usage: Optional[TokenUsageSummary] = None
    model: Optional[str] = None
    status: str = Field(
        default="success",
        description="Status code: 'success' or 'no_relevant_code_found'.",
    )
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def _load_cached_graph(repo_id: str) -> nx.DiGraph | None:
    """Attempt to load a cached NetworkX graph from Redis under rawgraph:{repo_id} or graph:{repo_id}.

    Returns None if the key does not exist or Redis is unreachable.
    """
    try:
        r = _get_redis_client()
        # Prefer rawgraph because it preserves all function nodes with source code before isolate pruning
        raw = r.get(f"rawgraph:{repo_id}") or r.get(f"graph:{repo_id}")
        if not raw:
            logger.info("No cached graph found in Redis for repo_id=%s", repo_id)
            return None
        data = json.loads(raw)
        graph: nx.DiGraph = nx.node_link_graph(data)
        logger.info("Loaded graph for repo_id=%s with %d nodes", repo_id, graph.number_of_nodes())
        return graph
    except Exception as exc:
        logger.warning(
            "Failed to load cached graph from Redis for repo_id=%s (%s). Falling back to graph=None",
            repo_id,
            exc,
        )
        return None


def _build_code_cache(graph: nx.DiGraph | None) -> dict[str, str]:
    """Extract code snippets from graph nodes for prompt construction."""
    cache: dict[str, str] = {}
    if not graph:
        return cache
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("kind") == "function" and attrs.get("code"):
            code_text = attrs["code"]
            cache[node_id] = code_text
            file_p = attrs.get("file") or attrs.get("path") or ""
            func_n = attrs.get("name") or attrs.get("label") or ""
            if file_p and func_n:
                cache[f"{file_p}::{func_n}"] = code_text
            if func_n and func_n not in cache:
                cache[func_n] = code_text
    return cache


def _build_code_cache_from_results(hybrid_results: list) -> dict[str, str]:
    """Build code cache from Qdrant payload code fields in hybrid_results.

    This is the primary code source when the Redis graph is unavailable or
    lacks source code. The Qdrant payload now stores the full function body
    in the 'code' field (added in ingestion.py fix).
    """
    cache: dict[str, str] = {}
    for result in hybrid_results:
        code = result.get("code") or ""
        if not code:
            continue
        func_n = result.get("func_name", "")
        file_p = result.get("file_path", "")
        if file_p and func_n:
            cache[f"{file_p}::{func_n}"] = code
        if func_n:
            cache.setdefault(func_n, code)
    return cache


def _extract_referenced_nodes(
    result: SynthesisResult,
    structured_prompt: StructuredPrompt,
) -> list[str]:
    """Extract all relevant nodes cited in the LLM explanation or provided in the prompt context.

    Pulls identifiers from:
    1. Valid citations in result.grounding_report / result.citations
    2. Any additional citations parsed from text
    3. structured_prompt.included_anchors and structured_prompt.included_neighbors
    """
    seen: set[str] = set()
    referenced_nodes: list[str] = []

    # Priority 1: Grounded valid citations cited in LLM output
    for cit in result.grounding_report.valid_citations:
        ident = cit.identifier
        if ident and ident not in seen:
            seen.add(ident)
            referenced_nodes.append(ident)

    # Priority 2: Any other citations present in result
    for cit in result.citations:
        ident = cit.identifier
        if ident and ident not in seen:
            seen.add(ident)
            referenced_nodes.append(ident)

    # Priority 3: Included vector anchors from StructuredPrompt
    for anchor in structured_prompt.included_anchors:
        if anchor and anchor not in seen:
            seen.add(anchor)
            referenced_nodes.append(anchor)

    # Priority 4: Included graph neighbors from StructuredPrompt
    for neighbor in structured_prompt.included_neighbors:
        if neighbor and neighbor not in seen:
            seen.add(neighbor)
            referenced_nodes.append(neighbor)

    return referenced_nodes


# ---------------------------------------------------------------------------
# Endpoint: POST /api/v1/query
# ---------------------------------------------------------------------------


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query codebase architecture via Graph RAG",
    description=(
        "Executes a multi-stage Graph RAG query pipeline: performs hybrid dense vector retrieval "
        "and 1-hop graph expansion, builds structured prompt context, and synthesizes a grounded "
        "architectural explanation via local Ollama inference."
    ),
)
def query_codebase(body: QueryRequest) -> QueryResponse:
    """Execute Graph RAG retrieval, prompt construction, and LLM explanation synthesis."""
    clean_query = body.query.strip()
    if not clean_query:
        return QueryResponse(
            query=body.query,
            answer="Empty query provided. Please submit a question about the codebase.",
            status="no_relevant_code_found",
            message="Query was empty or contained only whitespace.",
        )

    # 1. Resolve NetworkX graph from Redis if repo_id provided
    graph: nx.DiGraph | None = None
    if body.repo_id:
        graph = _load_cached_graph(body.repo_id)

    # 2. Stage 1 & 2: Hybrid Retrieval (Dense Vector + Graph Neighbor Expansion)
    try:
        hybrid_results: list[HybridResult] = hybrid_search(
            query=clean_query,
            graph=graph,
            top_k=body.top_k,
            score_threshold=body.score_threshold,
            include_graph_neighbors=body.include_graph_neighbors,
            repo_id=body.repo_id,
        )
    except Exception as exc:
        logger.exception("hybrid_search failed for query=%r: %s", clean_query, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search retrieval failed: {exc}",
        ) from exc

    # 3. Guard against realistic failure case: zero results from retrieval
    if not hybrid_results:
        logger.info("hybrid_search returned 0 results for query=%r", clean_query)
        return QueryResponse(
            query=clean_query,
            answer=(
                "No relevant code snippets or functions found in the codebase matching your query. "
                "Please rephrase your inquiry or ensure that repository functions have been indexed."
            ),
            referenced_nodes=[],
            citations=[],
            included_anchors=[],
            included_neighbors=[],
            grounding_report=None,
            token_usage=None,
            model=None,
            status="no_relevant_code_found",
            message="Hybrid search returned 0 results; LLM synthesis bypassed to prevent hallucinations.",
        )

    # 4. Stage 3: Structured Prompt Construction
    try:
        # Build code cache: merge Qdrant payload codes (primary) with graph codes (secondary)
        qdrant_code_cache = _build_code_cache_from_results(hybrid_results)
        graph_code_cache = _build_code_cache(graph)
        merged_code_cache = {**graph_code_cache, **qdrant_code_cache}  # Qdrant wins on conflict

        structured_prompt: StructuredPrompt = build_rag_prompt(
            query=clean_query,
            hybrid_results=hybrid_results,
            code_cache=merged_code_cache,
            repo_root=Path("/app"),
        )
    except Exception as exc:
        logger.exception("build_rag_prompt failed for query=%r: %s", clean_query, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prompt construction failed: {exc}",
        ) from exc

    # 5. Stage 4: LLM Explanation Synthesis via Ollama
    try:
        engine = LLMEngine()
        result: SynthesisResult = engine.synthesize(
            prompt=structured_prompt,
            query=clean_query,
            model=body.model,
        )
    except OllamaConnectionError as exc:
        logger.error("Cannot reach Ollama service: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Ollama inference service is offline or unreachable: {exc}",
        ) from exc
    except OllamaModelNotFoundError as exc:
        logger.error("Requested Ollama model not found: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except OllamaTimeoutError as exc:
        logger.error("Ollama synthesis timed out: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=f"Ollama inference timed out: {exc}",
        ) from exc
    except (OllamaResponseError, LLMEngineError) as exc:
        logger.error("Ollama synthesis error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Error during LLM synthesis: {exc}",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error during LLM synthesis: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during synthesis: {exc}",
        ) from exc

    # 6. Stage 5: Referenced Nodes & Citations Extraction
    referenced_nodes = _extract_referenced_nodes(result, structured_prompt)
    citation_models = [CitationModel.from_citation(c) for c in result.citations]
    grounding_summary = GroundingSummary.from_report(result.grounding_report)
    token_usage_summary = TokenUsageSummary(
        prompt_tokens=result.token_usage.prompt_tokens,
        completion_tokens=result.token_usage.completion_tokens,
        total_tokens=result.token_usage.total_tokens,
        total_duration_ms=result.token_usage.total_duration_ms,
        tokens_per_second=result.token_usage.tokens_per_second,
    )

    return QueryResponse(
        query=clean_query,
        answer=result.content,
        referenced_nodes=referenced_nodes,
        citations=citation_models,
        included_anchors=structured_prompt.included_anchors,
        included_neighbors=structured_prompt.included_neighbors,
        grounding_report=grounding_summary,
        token_usage=token_usage_summary,
        model=result.model,
        status="success",
        message="Explanation successfully synthesized.",
    )


# ---------------------------------------------------------------------------
# Endpoint: POST /api/v1/query/stream  (SSE – Server-Sent Events)
# ---------------------------------------------------------------------------


@router.post(
    "/query/stream",
    response_class=EventSourceResponse,
    status_code=status.HTTP_200_OK,
    summary="Stream codebase architecture query via Graph RAG (SSE)",
    description=(
        "Executes the same Graph RAG pipeline as POST /api/v1/query but streams the "
        "synthesized answer token-by-token as Server-Sent Events (text/event-stream). "
        "Emits three event types: 'token' (each text chunk), 'done' (final JSON "
        "matching QueryResponse), and 'error' (JSON error detail on failure)."
    ),
)
async def query_codebase_stream(body: QueryRequest) -> EventSourceResponse:
    """Async SSE endpoint: streams Ollama tokens in real-time via text/event-stream.

    The pipeline is identical to POST /api/v1/query up to and including
    build_rag_prompt().  The synthesis step uses stream_synthesize_async()
    so the asyncio event loop is never blocked.
    """

    async def _event_generator() -> AsyncIterator[dict]:
        clean_query = body.query.strip()
        if not clean_query:
            yield {
                "event": "error",
                "data": json.dumps(
                    {"detail": "Query was empty or contained only whitespace."}
                ),
            }
            return

        # --- Stage 1: Resolve NetworkX graph from Redis ---
        graph: nx.DiGraph | None = None
        if body.repo_id:
            graph = await asyncio.to_thread(_load_cached_graph, body.repo_id)

        # --- Stage 2: Hybrid Retrieval ---
        try:
            hybrid_results: list[HybridResult] = await asyncio.to_thread(
                hybrid_search,
                query=clean_query,
                graph=graph,
                top_k=body.top_k,
                score_threshold=body.score_threshold,
                include_graph_neighbors=body.include_graph_neighbors,
                repo_id=body.repo_id,
            )
        except Exception as exc:
            logger.exception("hybrid_search failed in /stream for query=%r: %s", clean_query, exc)
            yield {
                "event": "error",
                "data": json.dumps({"detail": f"Hybrid search retrieval failed: {exc}"}),
            }
            return

        # --- Stage 3: Zero-results guard ---
        if not hybrid_results:
            logger.info("hybrid_search returned 0 results in /stream for query=%r", clean_query)
            payload = QueryResponse(
                query=clean_query,
                answer=(
                    "No relevant code snippets or functions found in the codebase matching your query. "
                    "Please rephrase your inquiry or ensure that repository functions have been indexed."
                ),
                status="no_relevant_code_found",
                message="Hybrid search returned 0 results; LLM synthesis bypassed to prevent hallucinations.",
            )
            yield {"event": "done", "data": payload.model_dump_json()}
            return

        # --- Stage 4: Structured Prompt Construction ---
        try:
            code_cache_dict = _build_code_cache(graph)
            structured_prompt: StructuredPrompt = await asyncio.to_thread(
                build_rag_prompt,
                query=clean_query,
                hybrid_results=hybrid_results,
                code_cache=code_cache_dict,
                repo_root=Path("/app"),
            )
        except Exception as exc:
            logger.exception("build_rag_prompt failed in /stream for query=%r: %s", clean_query, exc)
            yield {
                "event": "error",
                "data": json.dumps({"detail": f"Prompt construction failed: {exc}"}),
            }
            return

        # --- Stage 5: Async token streaming via stream_synthesize_async ---
        try:
            engine = LLMEngine()
            final_result: SynthesisResult | None = None

            async for token_chunk, maybe_result in engine.stream_synthesize_async(
                prompt=structured_prompt,
                query=clean_query,
                model=body.model,
            ):
                if token_chunk:
                    # Intermediate token: emit as SSE "token" event
                    yield {"event": "token", "data": token_chunk}
                if maybe_result is not None:
                    final_result = maybe_result

        except OllamaConnectionError as exc:
            logger.error("Cannot reach Ollama in /stream: %s", exc)
            yield {
                "event": "error",
                "data": json.dumps({"detail": f"Ollama inference service is offline or unreachable: {exc}"}),
            }
            return
        except OllamaModelNotFoundError as exc:
            logger.error("Ollama model not found in /stream: %s", exc)
            yield {"event": "error", "data": json.dumps({"detail": str(exc)})}
            return
        except OllamaTimeoutError as exc:
            logger.error("Ollama timed out in /stream: %s", exc)
            yield {
                "event": "error",
                "data": json.dumps({"detail": f"Ollama inference timed out: {exc}"}),
            }
            return
        except (OllamaResponseError, LLMEngineError) as exc:
            logger.error("Ollama synthesis error in /stream: %s", exc)
            yield {
                "event": "error",
                "data": json.dumps({"detail": f"Error during LLM synthesis: {exc}"}),
            }
            return
        except Exception as exc:
            logger.exception("Unexpected error in /stream during synthesis: %s", exc)
            yield {
                "event": "error",
                "data": json.dumps({"detail": f"Unexpected error during synthesis: {exc}"}),
            }
            return

        # --- Stage 6: Assemble final "done" event ---
        if final_result is None:
            # Streaming finished without a final result (empty model response)
            yield {
                "event": "error",
                "data": json.dumps({"detail": "Streaming completed but no content was produced."}),
            }
            return

        referenced_nodes = _extract_referenced_nodes(final_result, structured_prompt)
        citation_models = [CitationModel.from_citation(c) for c in final_result.citations]
        grounding_summary = GroundingSummary.from_report(final_result.grounding_report)
        token_usage_summary = TokenUsageSummary(
            prompt_tokens=final_result.token_usage.prompt_tokens,
            completion_tokens=final_result.token_usage.completion_tokens,
            total_tokens=final_result.token_usage.total_tokens,
            total_duration_ms=final_result.token_usage.total_duration_ms,
            tokens_per_second=final_result.token_usage.tokens_per_second,
        )

        done_payload = QueryResponse(
            query=clean_query,
            answer=final_result.content,
            referenced_nodes=referenced_nodes,
            citations=citation_models,
            included_anchors=structured_prompt.included_anchors,
            included_neighbors=structured_prompt.included_neighbors,
            grounding_report=grounding_summary,
            token_usage=token_usage_summary,
            model=final_result.model,
            status="success",
            message="Streaming synthesis complete.",
        )
        yield {"event": "done", "data": done_payload.model_dump_json()}

    return EventSourceResponse(_event_generator())

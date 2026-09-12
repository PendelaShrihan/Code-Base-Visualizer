"""
routers/query.py
================
Exposes the POST /api/v1/query endpoint for Graph RAG architectural code query.

Pipeline:
1. Accept QueryRequest with natural language question and optional repo_id.
2. If repo_id is supplied, look up the serialized NetworkX graph in Redis under key "graph:{repo_id}".
   Falls back gracefully to graph=None (pure vector search) if not found or Redis is unreachable.
3. Call hybrid_search(query, graph=graph, ...) to retrieve vector anchors and 1-hop graph neighbors.
4. Zero-results Guard: If hybrid_search returns 0 matches, immediately return a clear
   "no_relevant_code_found" response without invoking the LLM with empty context.
5. If matches found: Call build_rag_prompt(...) to format grounded, token-budgeted XML context.
6. Call LLMEngine().synthesize(...) to generate architectural explanation via Ollama.
7. Extract referenced_nodes combining result.citations, result.grounding_report, and
   StructuredPrompt's included_anchors + included_neighbors.
8. Return structured QueryResponse.

Execution Model:
Defined as a standard synchronous `def` route so FastAPI automatically dispatches
execution to the worker thread pool (starlette anyio.to_thread.run_sync), avoiding
blocking the main asyncio event loop during synchronous Qdrant vector retrieval,
Redis network queries, and Ollama HTTP inference.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

import networkx as nx
import redis
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

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
REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
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
    """Attempt to load a cached NetworkX graph from Redis under graph:{repo_id}.

    Returns None if the key does not exist or Redis is unreachable.
    """
    cache_key = f"graph:{repo_id}"
    try:
        r = _get_redis_client()
        raw = r.get(cache_key)
        if not raw:
            logger.info("No cached graph found in Redis for key=%s", cache_key)
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
        structured_prompt: StructuredPrompt = build_rag_prompt(
            query=clean_query,
            hybrid_results=hybrid_results,
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

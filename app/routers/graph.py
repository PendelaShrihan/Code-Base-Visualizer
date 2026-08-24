"""
routers/graph.py
================
Exposes the POST /api/v1/graph/parse endpoint.

Flow
----
1. Accept ``repo_url`` + ``repo_id`` in the request body.
2. Clone (or locate) the repository on disk via ``clone_repository``.
3. Run ``scan_repository`` (CPU-bound) in a thread-pool executor so the
   async event loop is never blocked.
4. Serialize the resulting ``nx.DiGraph`` with ``nx.node_link_data()``.
5. Store the JSON string in Redis under the key ``graph:{repo_id}``.
6. Return a lightweight summary (node + edge counts) to the caller.
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import networkx as nx
import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, HttpUrl

from app.services.git_service import clone_repository
from parser.repo_walker import attach_churn, scan_repository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared thread-pool for CPU-bound work (tree-sitter parsing)
# ---------------------------------------------------------------------------

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="repo-walker")

# ---------------------------------------------------------------------------
# Redis client (same connection settings as main.py)
# ---------------------------------------------------------------------------

_redis: aioredis.Redis = aioredis.Redis(
    host="redis", port=6379, decode_responses=True
)

# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/v1", tags=["graph"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class ParseRequest(BaseModel):
    repo_url: HttpUrl
    repo_id: str


class GraphMetrics(BaseModel):
    node_count: int
    edge_count: int
    file_count: int
    class_count: int
    import_count: int
    function_count: int


class ParseResponse(BaseModel):
    repo_id: str
    cache_key: str
    message: str
    metrics: GraphMetrics


# ---------------------------------------------------------------------------
# Helper — synchronous work delegated to the thread pool
# ---------------------------------------------------------------------------


def _clone_and_scan(url_str: str) -> tuple[Path, nx.DiGraph]:
    """Clone *url_str* and immediately scan the resulting checkout.

    This function is intentionally synchronous: it is invoked inside
    ``asyncio.get_running_loop().run_in_executor`` so it never blocks the
    event loop.

    Returns:
        A ``(clone_path, graph)`` tuple with `pagerank` and `commit_count`
        node attributes populated.

    Raises:
        ValueError:  Propagated from ``clone_repository`` on invalid URLs.
        RuntimeError: Propagated from ``clone_repository`` on git failures.
        NotADirectoryError: Propagated from ``scan_repository`` if the clone
                            path is somehow not a directory.
    """
    clone_path: Path = clone_repository(url_str)
    graph: nx.DiGraph = scan_repository(clone_path)
    attach_churn(graph, clone_path)
    return clone_path, graph


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.post(
    "/graph/parse",
    response_model=ParseResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Parse a repository and cache its dependency graph",
    description=(
        "Clones the given public GitHub repository, recursively scans every "
        "Python source file with the AST engine, builds a NetworkX DiGraph, "
        "serialises it with ``nx.node_link_data()``, and stores the result in "
        "Redis under the key ``graph:{repo_id}``.  Returns graph metrics so "
        "the caller can verify the caching worked correctly."
    ),
)
async def parse_and_cache_graph(body: ParseRequest) -> ParseResponse:
    """
    POST /api/v1/graph/parse

    Request body::

        {
            "repo_url": "https://github.com/owner/repo",
            "repo_id":  "owner-repo-v1"
        }

    Response (201)::

        {
            "repo_id":   "owner-repo-v1",
            "cache_key": "graph:owner-repo-v1",
            "message":   "Graph parsed and cached successfully.",
            "metrics": {
                "node_count":     <int>,
                "edge_count":     <int>,
                "file_count":     <int>,
                "class_count":    <int>,
                "import_count":   <int>,
                "function_count": <int>
            }
        }
    """
    url_str = str(body.repo_url)
    repo_id = body.repo_id
    cache_key = f"graph:{repo_id}"

    # ------------------------------------------------------------------
    # 1 + 2.  Clone the repo and run the recursive scanner (off-thread)
    # ------------------------------------------------------------------
    loop = asyncio.get_running_loop()
    try:
        _clone_path, graph = await loop.run_in_executor(
            _EXECUTOR, _clone_and_scan, url_str
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid repository URL: {exc}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to clone repository: {exc}",
        ) from exc
    except NotADirectoryError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Clone path is not a directory: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected error during repo scan for %s", url_str)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during graph parsing: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # 3.  Serialize the graph with nx.node_link_data()
    # ------------------------------------------------------------------
    node_link: dict[str, Any] = nx.node_link_data(graph)
    graph_json: str = json.dumps(node_link)

    # ------------------------------------------------------------------
    # 4.  Store the JSON in Redis under key  graph:{repo_id}
    # ------------------------------------------------------------------
    try:
        await _redis.set(cache_key, graph_json)
        logger.info(
            "Stored graph for repo_id=%s under key=%s (%d bytes)",
            repo_id,
            cache_key,
            len(graph_json),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Redis write failed for key=%s", cache_key)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to cache graph in Redis: {exc}",
        ) from exc

    # ------------------------------------------------------------------
    # 5.  Compute per-kind node counts for the response metrics
    # ------------------------------------------------------------------
    node_data: list[tuple[str, dict]] = list(graph.nodes(data=True))

    def _count_kind(kind: str) -> int:
        return sum(1 for _, attrs in node_data if attrs.get("kind") == kind)

    metrics = GraphMetrics(
        node_count=graph.number_of_nodes(),
        edge_count=graph.number_of_edges(),
        file_count=_count_kind("file"),
        class_count=_count_kind("class"),
        import_count=_count_kind("import"),
        function_count=_count_kind("function"),
    )

    # ------------------------------------------------------------------
    # 6.  Return the summary
    # ------------------------------------------------------------------
    return ParseResponse(
        repo_id=repo_id,
        cache_key=cache_key,
        message="Graph parsed and cached successfully.",
        metrics=metrics,
    )

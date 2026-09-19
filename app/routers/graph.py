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
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import networkx as nx
import redis.asyncio as aioredis
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, HttpUrl

from app.services.git_service import cleanup_repo_directory, clone_repository
from parser.repo_walker import attach_churn, filter_graph, scan_repository

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared thread-pool for CPU-bound work (tree-sitter parsing)
# ---------------------------------------------------------------------------

_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="repo-walker")

# ---------------------------------------------------------------------------
# Redis client — host is read from the REDIS_HOST env-var so that the same
# image works both inside Docker Compose (host="redis") and from the local
# venv during tests / development (host="127.0.0.1" or "localhost").
# ---------------------------------------------------------------------------

_REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")

_redis: aioredis.Redis = aioredis.Redis(
    host=_REDIS_HOST, port=6379, decode_responses=True
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


class GraphData(BaseModel):
    """Returned by GET /api/v1/graph/{repo_id} — the raw node-link JSON."""

    repo_id: str
    cache_key: str
    # The full node-link payload (as returned by nx.node_link_data) is passed
    # through without re-shaping so the frontend can transform it as needed.
    graph: dict[str, Any]


class TraceData(BaseModel):
    """Returned by GET /api/v1/graph/{repo_id}/trace/{node_id}."""

    repo_id: str
    node_id: str
    # ego-graph subgraph as node-link JSON (includes Level 4 call_target nodes)
    graph: dict[str, Any]


# ---------------------------------------------------------------------------
# Helper — synchronous work delegated to the thread pool
# ---------------------------------------------------------------------------


def _clone_and_scan(url_str: str, repo_id: str) -> tuple[Path, nx.DiGraph, dict[str, Any]]:
    """Clone *url_str*, scan it, and return both the raw and filtered graphs.

    This function is intentionally synchronous: it is invoked inside
    ``asyncio.get_running_loop().run_in_executor`` so it never blocks the
    event loop. Cloned files are cleaned up in the finally block.

    Returns:
        A ``(clone_path, filtered_graph, raw_node_link)`` triple where
        *raw_node_link* is the serialised **pre-filter** graph (still contains
        Level 4 ``call_target`` nodes) needed by the trace endpoint.

    Raises:
        ValueError:  Propagated from ``clone_repository`` on invalid URLs.
        RuntimeError: Propagated from ``clone_repository`` on git failures.
        NotADirectoryError: Propagated from ``scan_repository`` if the clone
                            path is somehow not a directory.
    """
    clone_path: Path = clone_repository(url_str)
    try:
        graph: nx.DiGraph = scan_repository(clone_path)
        attach_churn(graph, clone_path)
        # Serialise the raw (unfiltered) graph BEFORE pruning so the trace
        # endpoint can later fetch Level 4 call_target nodes.
        raw_node_link: dict[str, Any] = nx.node_link_data(graph)
        filter_graph(graph)
        return clone_path, graph, raw_node_link
    finally:
        cleanup_repo_directory(clone_path)


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
        _clone_path, graph, raw_node_link = await loop.run_in_executor(
            _EXECUTOR, _clone_and_scan, url_str, repo_id
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
    # 3.  Serialize the filtered graph with nx.node_link_data()
    # ------------------------------------------------------------------
    node_link: dict[str, Any] = nx.node_link_data(graph)
    graph_json: str = json.dumps(node_link)
    raw_graph_json: str = json.dumps(raw_node_link)
    raw_cache_key = f"rawgraph:{repo_id}"

    # ------------------------------------------------------------------
    # 4.  Store both graphs in Redis
    #     graph:{repo_id}     — filtered (no imports / call_targets)
    #     rawgraph:{repo_id}  — unfiltered (has Level 4 call_target nodes)
    # ------------------------------------------------------------------
    try:
        await _redis.set(cache_key, graph_json)
        await _redis.set(raw_cache_key, raw_graph_json)
        logger.info(
            "Stored graph for repo_id=%s under keys=%s, %s (%d + %d bytes)",
            repo_id,
            cache_key,
            raw_cache_key,
            len(graph_json),
            len(raw_graph_json),
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


# ---------------------------------------------------------------------------
# GET /api/v1/graph/{repo_id}  — retrieve a cached graph from Redis
# ---------------------------------------------------------------------------


@router.get(
    "/graph/{repo_id}",
    response_model=GraphData,
    status_code=status.HTTP_200_OK,
    summary="Retrieve a cached dependency graph",
    description=(
        "Reads the serialised NetworkX node-link graph stored in Redis under the "
        "key ``graph:{repo_id}`` and returns it as JSON.  "
        "Returns **404** if the key has not been populated yet (run "
        "``POST /api/v1/graph/parse`` first).  "
        "Returns **503** if Redis is unreachable."
    ),
)
async def get_graph(repo_id: str) -> GraphData:
    """
    GET /api/v1/graph/{repo_id}

    Response (200)::

        {
            "repo_id":   "<repo_id>",
            "cache_key": "graph:<repo_id>",
            "graph": {
                "directed":    true,
                "multigraph":  false,
                "graph":       {},
                "nodes":       [ { "id": "...", "kind": "...", ... }, ... ],
                "edges":       [ { "source": "...", "target": "...", ... }, ... ]
            }
        }

    Errors:
        404  — key ``graph:{repo_id}`` does not exist in Redis.
        503  — Redis connection error.
    """
    cache_key = f"graph:{repo_id}"

    # ------------------------------------------------------------------
    # Read from Redis
    # ------------------------------------------------------------------
    try:
        raw: str | None = await _redis.get(cache_key)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Redis read failed for key=%s", cache_key)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to read graph from Redis: {exc}",
        ) from exc

    if raw is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"Graph not found for repository '{repo_id}'. "
                "Run POST /api/v1/graph/parse to generate and cache it first."
            ),
        )

    # ------------------------------------------------------------------
    # Deserialize and return
    # ------------------------------------------------------------------
    try:
        graph_payload: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("Corrupt JSON in Redis key=%s: %s", cache_key, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cached graph data is corrupt for repo '{repo_id}'.",
        ) from exc

    # Auto-clean legacy/unfiltered graphs so canvas immediately benefits
    nodes_list = graph_payload.get("nodes", [])
    if any((n.get("name") or n.get("label") or "") in ("os", "sys", "typing", "print", "len", "str", "dict", "list", "int") for n in nodes_list):
        try:
            temp_g = nx.node_link_graph(graph_payload)
            filter_graph(temp_g)
            graph_payload = nx.node_link_data(temp_g)
            await _redis.set(cache_key, json.dumps(graph_payload))
            logger.info("Auto-cleaned legacy graph in Redis for repo_id=%s", repo_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Auto-clean failed for repo_id=%s: %s", repo_id, exc)

    logger.info(
        "Served graph for repo_id=%s from cache (key=%s, nodes=%s, edges=%s)",
        repo_id,
        cache_key,
        len(graph_payload.get("nodes", [])),
        len(graph_payload.get("edges", graph_payload.get("links", []))),
    )

    return GraphData(
        repo_id=repo_id,
        cache_key=cache_key,
        graph=graph_payload,
    )


# ---------------------------------------------------------------------------
# GET /api/v1/graph/{repo_id}/trace/{node_id}  — Level 4 ego-graph trace
# ---------------------------------------------------------------------------


@router.get(
    "/graph/{repo_id}/trace/{node_id:path}",
    response_model=TraceData,
    status_code=status.HTTP_200_OK,
    summary="Extract a Level 4 ego-graph trace around a node",
    description=(
        "Reads the raw (unfiltered) graph stored in Redis under "
        "``rawgraph:{repo_id}`` (which still contains Level 4 call_target nodes). "
        "Builds an ego-graph of radius 2 around ``node_id`` using "
        "``nx.ego_graph`` and returns the subgraph as node-link JSON. "
        "Returns **404** if neither the raw nor the filtered graph key exists."
    ),
)
async def get_trace(repo_id: str, node_id: str) -> TraceData:
    """
    GET /api/v1/graph/{repo_id}/trace/{node_id}

    Returns a focused Level 4 subgraph (ego-graph, radius=2) suitable for
    the frontend 'blackout isolation' mode.  The payload always includes
    call_target nodes because it is built from the unfiltered raw graph.

    Errors:
        404  — no cached graph found for this repo.
        422  — node_id does not exist in the raw graph.
        503  — Redis connection error.
    """
    raw_cache_key = f"rawgraph:{repo_id}"
    fallback_cache_key = f"graph:{repo_id}"

    # Read raw graph from Redis (fall back to filtered graph if raw not yet
    # populated — happens for repos parsed before this feature was added).
    try:
        raw: str | None = await _redis.get(raw_cache_key)
        if raw is None:
            logger.warning(
                "rawgraph key missing for repo_id=%s; falling back to graph key",
                repo_id,
            )
            raw = await _redis.get(fallback_cache_key)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Redis read failed for key=%s", raw_cache_key)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to read raw graph from Redis: {exc}",
        ) from exc

    if raw is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                f"No graph found for repository '{repo_id}'. "
                "Run POST /api/v1/graph/parse first."
            ),
        )

    try:
        raw_payload: dict[str, Any] = json.loads(raw)
        raw_G: nx.DiGraph = nx.node_link_graph(raw_payload)
    except (json.JSONDecodeError, Exception) as exc:
        logger.error("Failed to deserialize raw graph for repo_id=%s: %s", repo_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Corrupt raw graph data for repo '{repo_id}'.",
        ) from exc

    if node_id not in raw_G:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Node '{node_id}' not found in the graph for repo '{repo_id}'.",
        )

    # Build ego-graph: 2-hop subgraph centred on node_id (directed)
    try:
        ego: nx.DiGraph = nx.ego_graph(
            raw_G, node_id, radius=2, center=True, undirected=False
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("ego_graph failed for node_id=%s: %s", node_id, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute ego-graph for node '{node_id}'.",
        ) from exc

    logger.info(
        "Trace for repo_id=%s node_id=%s: %d nodes, %d edges",
        repo_id,
        node_id,
        ego.number_of_nodes(),
        ego.number_of_edges(),
    )

    return TraceData(
        repo_id=repo_id,
        node_id=node_id,
        graph=nx.node_link_data(ego),
    )

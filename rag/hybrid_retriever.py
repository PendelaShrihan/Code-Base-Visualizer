"""
rag/hybrid_retriever.py
-----------------------
Graph-Vector Hybrid Retriever for CodeBase Visualizer.

Architecture Overview
---------------------
This module implements **Graph RAG** (Retrieval-Augmented Generation with
graph-context augmentation) — a two-stage retrieval strategy that combines
the semantic precision of dense vector search with the structural awareness
of a code call-graph:

    Query (natural language)
         │
         ▼  Stage 1 – Dense Vector Retrieval
    Qdrant.query_points()         →  top-k ScoredPoint anchors
         │
         ▼  Stage 2 – Graph Context Expansion
    NetworkX DiGraph.neighbors()  →  1-hop callers + callees per anchor
         │
         ▼  Dedup + Rank
    HybridResult list             →  anchor + graph-neighbour context

Why graph augmentation?
-----------------------
A pure vector search ranks functions by *semantic similarity to the query*,
but it is blind to *structural relationships*.  A function that is closely
related by calling or being called by an anchor may carry equally important
context for code understanding and LLM generation tasks.

Graph-neighbour augmentation adds this structural signal without a second
embedding lookup:

* **Callee context** — functions *called by* the anchor tell you what
  helpers/utilities the anchor depends on.
* **Caller context** — functions that *call* the anchor tell you where and
  how it is used across the codebase.

Design Decisions
----------------
* **1-hop only** — deeper expansions (2+ hops) risk introducing noise and
  context-window bloat.  Users wanting wider coverage can call the helper
  ``_expand_neighbors`` recursively in future iterations.

* **Deduplication by func_name + file_path** — anchors returned by Qdrant
  may overlap with neighbours expanded from other anchors.  We deduplicate
  before returning so the caller never sees duplicates.

* **Graceful degradation** — if the NetworkX graph is not supplied (e.g.
  queried from a REST endpoint that hasn't loaded it yet), the retriever
  falls back to pure-vector results silently.

* **node_id convention** — the graph node IDs follow the repo_walker.py
  scoping convention: ``<rel_file_path>::func::<func_name>``.  The retriever
  searches the graph by this pattern so it integrates with the existing
  ingestion pipeline without changes.

Data Structures
---------------
``HybridResult`` (TypedDict-like dict)
    Returned per result item; superset of ``search_functions()`` output::

        {
            "rank":                   int,        # 1-based position in final list
            "func_name":              str,
            "file_path":              str,
            "score":                  float,      # Qdrant cosine score; 0.0 for graph-only
            "similarity_label":       str,        # "HIGH CONFIDENCE" | "MODERATE RELEVANCE" | ...
            "pagerank":               float,
            "commit_count":           int,
            "is_dead_code_candidate": bool,
            "source":                 str,        # "vector" | "graph_callee" | "graph_caller"
            "anchor_func":            str | None, # originating anchor func_name (None for anchors)
        }
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import networkx as nx
from qdrant_client import QdrantClient

from rag.qdrant_client import COLLECTION_NAME, get_qdrant_client
from rag.test_search import classify_similarity, search_functions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public type alias
# ---------------------------------------------------------------------------

HybridResult = dict[str, Any]
"""
A single item in the list returned by :func:`hybrid_search`.

Keys
----
rank                   : 1-based integer position after dedup + ranking.
func_name              : Bare function name (e.g. ``"clone_repository"``).
file_path              : Relative file path within the repository.
score                  : Cosine similarity score from Qdrant (0.0 for
                         graph-only neighbours — they were not retrieved
                         via vector search).
similarity_label       : Qualitative similarity tier from
                         :func:`rag.test_search.classify_similarity`.
pagerank               : PageRank centrality score stored in Qdrant payload.
commit_count           : Git commit-churn count from Qdrant payload.
is_dead_code_candidate : Dead-code flag from Qdrant payload.
source                 : One of ``"vector"``, ``"graph_callee"``, or
                         ``"graph_caller"`` — how this result was obtained.
anchor_func            : ``func_name`` of the vector-retrieved anchor that
                         triggered graph expansion, or ``None`` when this
                         result *is* the anchor itself.
"""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_graph_node(
    graph: nx.DiGraph,
    func_name: str,
    file_path: str,
) -> str | None:
    """Locate the NetworkX node ID for a given function.

    Node IDs in the repo_walker graph follow the convention::

        <rel_file_path>::func::<func_name>

    We first try an exact match using the known *file_path*, then fall back
    to a linear scan of all ``kind="function"`` nodes that share the same
    *func_name* (handles cases where the Qdrant payload ``file_path``
    differs slightly from the graph-node prefix, e.g. OS separator or
    trailing slash differences).

    Args:
        graph:     The NetworkX DiGraph produced by
                   :func:`parser.repo_walker.scan_repository`.
        func_name: Bare function name to locate.
        file_path: Relative file path from the Qdrant payload.

    Returns:
        The matching node ID string, or ``None`` if not found.
    """
    # --- Attempt 1: deterministic exact-key lookup (O(1)) --------------------
    candidate = f"{file_path}::func::{func_name}"
    if candidate in graph:
        return candidate

    # --- Attempt 2: scan all function nodes matching func_name (O(n)) --------
    for node_id, attrs in graph.nodes(data=True):
        if attrs.get("kind") == "function" and attrs.get("name") == func_name:
            return node_id

    return None


def _payload_from_graph_node(
    graph: nx.DiGraph,
    node_id: str,
) -> dict[str, Any]:
    """Extract metadata from a graph node's attribute dict.

    Falls back to sensible defaults for any missing attributes so the caller
    always receives a well-formed dict.

    Args:
        graph:   The NetworkX DiGraph.
        node_id: A node ID that exists in *graph*.

    Returns:
        Dict with keys: ``func_name``, ``file_path``, ``pagerank``,
        ``commit_count``, ``is_dead_code_candidate``.
    """
    attrs = graph.nodes[node_id]
    return {
        "func_name": attrs.get("name") or attrs.get("label") or node_id,
        "file_path": attrs.get("file") or attrs.get("path") or "",
        "pagerank": float(attrs.get("pagerank", 0.0)),
        "commit_count": int(attrs.get("commit_count", 0)),
        "is_dead_code_candidate": bool(attrs.get("is_dead_code_candidate", False)),
    }


def _expand_neighbors(
    graph: nx.DiGraph,
    anchor_node_id: str,
    anchor_func_name: str,
) -> list[HybridResult]:
    """Return 1-hop caller and callee neighbours of *anchor_node_id*.

    Only nodes with ``kind="function"`` are included; file, import, class,
    and call-target nodes are skipped because they carry no meaningful
    function context for RAG.

    Args:
        graph:            The NetworkX DiGraph.
        anchor_node_id:   Graph node ID of the vector-retrieved anchor.
        anchor_func_name: Human-readable name of the anchor (for
                          ``anchor_func`` field on neighbour results).

    Returns:
        List of :data:`HybridResult` dicts — one per qualifying neighbour —
        with ``score=0.0`` and appropriate ``source`` tag.  Duplicate
        neighbours from both directions (i.e. a node that is both a caller
        and a callee of the anchor) appear twice with different ``source``
        values; deduplication is handled by the caller.
    """
    neighbours: list[HybridResult] = []

    # -- Callees: edges FROM anchor → successor (anchor *calls* successor) ----
    for successor_id in graph.successors(anchor_node_id):
        attrs = graph.nodes[successor_id]
        if attrs.get("kind") != "function":
            continue
        payload = _payload_from_graph_node(graph, successor_id)
        neighbours.append(
            {
                **payload,
                "score": 0.0,
                "similarity_label": classify_similarity(0.0),
                "source": "graph_callee",
                "anchor_func": anchor_func_name,
            }
        )
        logger.debug(
            "_expand_neighbors: callee %r ← anchor %r",
            payload["func_name"],
            anchor_func_name,
        )

    # -- Callers: edges FROM predecessor → anchor (predecessor *calls* anchor) -
    for predecessor_id in graph.predecessors(anchor_node_id):
        attrs = graph.nodes[predecessor_id]
        if attrs.get("kind") != "function":
            continue
        payload = _payload_from_graph_node(graph, predecessor_id)
        neighbours.append(
            {
                **payload,
                "score": 0.0,
                "similarity_label": classify_similarity(0.0),
                "source": "graph_caller",
                "anchor_func": anchor_func_name,
            }
        )
        logger.debug(
            "_expand_neighbors: caller %r → anchor %r",
            payload["func_name"],
            anchor_func_name,
        )

    return neighbours


def _assign_ranks(results: list[HybridResult]) -> None:
    """Assign 1-based ``rank`` values in-place.

    Anchors are already sorted by Qdrant score (descending).  Graph
    neighbours appended after anchors receive subsequent ranks.  This mutates
    *results* directly.

    Args:
        results: Mutable list of :data:`HybridResult` dicts.
    """
    for i, result in enumerate(results, start=1):
        result["rank"] = i


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def hybrid_search(
    query: str,
    graph: nx.DiGraph | None = None,
    top_k: int = 5,
    client: Optional[QdrantClient] = None,
    collection_name: str = COLLECTION_NAME,
    score_threshold: Optional[float] = None,
    include_graph_neighbors: bool = True,
) -> list[HybridResult]:
    """Graph-augmented semantic code search (Graph RAG retrieval).

    **Stage 1 — Vector Retrieval**
        Embeds *query* with the same ``all-MiniLM-L6-v2`` model used during
        ingestion and retrieves ``top_k`` matching functions from Qdrant using
        cosine similarity.

    **Stage 2 — Graph Context Expansion**
        For each vector-retrieved *anchor*, looks up the corresponding node
        in the *graph* (a :class:`networkx.DiGraph` built by
        :func:`parser.repo_walker.scan_repository`) and fetches all 1-hop
        ``func_call`` neighbours — both callees (successors) and callers
        (predecessors).

    **Deduplication & Ranking**
        Results are deduplicated by ``(func_name, file_path)``.  Vector
        anchors retain their Qdrant cosine score; graph-only neighbours
        receive ``score=0.0`` and are appended after anchors, preserving
        anchor rank order.

    Args:
        query:                   Plain English natural language question.
        graph:                   NetworkX DiGraph from
                                 ``parser.repo_walker.scan_repository()``.
                                 Pass ``None`` to fall back to pure vector
                                 search.
        top_k:                   Number of Qdrant results to retrieve
                                 (default: 5).
        client:                  Qdrant client (uses module default if
                                 ``None``).
        collection_name:         Qdrant collection name (default:
                                 ``"code_chunks"``).
        score_threshold:         Optional minimum cosine similarity cutoff
                                 passed through to Qdrant.
        include_graph_neighbors: Set ``False`` to disable graph expansion and
                                 return pure vector results only (useful for
                                 A/B comparisons).

    Returns:
        List of :data:`HybridResult` dicts, 1-indexed by ``rank``.  Anchors
        appear first (ordered by cosine score), followed by unique graph-only
        neighbours ordered by discovery order.  Returns ``[]`` for a blank
        query.

    Example::

        from parser.repo_walker import scan_repository
        from rag.hybrid_retriever import hybrid_search

        graph = scan_repository("/path/to/repo")
        results = hybrid_search("how do I clone a git repository?", graph=graph)
        for r in results:
            print(r["rank"], r["func_name"], r["source"], r["score"])
    """
    if not query.strip():
        logger.debug("hybrid_search: empty query — returning []")
        return []

    client = client or get_qdrant_client()

    # -------------------------------------------------------------------------
    # Stage 1: Dense vector retrieval from Qdrant
    # -------------------------------------------------------------------------
    vector_results = search_functions(
        query=query,
        top_k=top_k,
        client=client,
        collection_name=collection_name,
        score_threshold=score_threshold,
    )

    logger.info(
        "hybrid_search: vector stage returned %d anchor(s) for query %r",
        len(vector_results),
        query,
    )

    # Annotate anchors with hybrid-specific fields
    anchors: list[HybridResult] = [
        {
            **res,
            "source": "vector",
            "anchor_func": None,
        }
        for res in vector_results
    ]

    if not include_graph_neighbors or graph is None:
        if graph is None and include_graph_neighbors:
            logger.debug(
                "hybrid_search: no graph supplied — returning pure vector results"
            )
        _assign_ranks(anchors)
        return anchors

    # -------------------------------------------------------------------------
    # Stage 2: Graph context expansion — 1-hop neighbours per anchor
    # -------------------------------------------------------------------------

    # Pre-populate seen set with anchor keys to avoid re-adding them as
    # graph neighbours if they happen to be adjacent to each other.
    seen: set[tuple[str, str]] = {
        (r["func_name"], r["file_path"]) for r in anchors
    }

    graph_neighbours: list[HybridResult] = []

    for anchor in anchors:
        func_name: str = anchor["func_name"]
        file_path: str = anchor["file_path"]

        anchor_node_id = _find_graph_node(graph, func_name, file_path)

        if anchor_node_id is None:
            logger.debug(
                "hybrid_search: anchor %r (%r) not found in graph — skipping",
                func_name,
                file_path,
            )
            continue

        logger.debug(
            "hybrid_search: expanding 1-hop neighbours for %r (node=%r)",
            func_name,
            anchor_node_id,
        )

        neighbours = _expand_neighbors(graph, anchor_node_id, func_name)

        for neighbour in neighbours:
            key = (neighbour["func_name"], neighbour["file_path"])
            if key not in seen:
                seen.add(key)
                graph_neighbours.append(neighbour)

    logger.info(
        "hybrid_search: graph expansion added %d unique neighbour(s)",
        len(graph_neighbours),
    )

    # -------------------------------------------------------------------------
    # Merge, re-rank, and return
    # -------------------------------------------------------------------------
    combined: list[HybridResult] = anchors + graph_neighbours
    _assign_ranks(combined)

    return combined


# ---------------------------------------------------------------------------
# Display helper
# ---------------------------------------------------------------------------

def format_hybrid_results(query: str, results: list[HybridResult]) -> str:
    """Render a human-readable report for a hybrid search result set.

    Extends the spirit of :func:`rag.test_search.format_results` with an
    additional **source** column that distinguishes vector anchors from
    graph-derived neighbours.

    Args:
        query:   The original search query string.
        results: Output of :func:`hybrid_search`.

    Returns:
        Multi-line formatted string suitable for printing or logging.
    """
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f'HYBRID QUERY: "{query}"')
    lines.append("=" * 80)

    if not results:
        lines.append("No results found.")
        lines.append("=" * 80)
        return "\n".join(lines)

    vector_anchors = [r for r in results if r["source"] == "vector"]
    callees = [r for r in results if r["source"] == "graph_callee"]
    callers = [r for r in results if r["source"] == "graph_caller"]

    lines.append(
        f"  Vector anchors : {len(vector_anchors)}"
        f"  │  Graph callees : {len(callees)}"
        f"  │  Graph callers : {len(callers)}"
    )
    lines.append("-" * 80)

    source_icon = {
        "vector": "⚡",
        "graph_callee": "↓",
        "graph_caller": "↑",
    }

    for res in results:
        icon = source_icon.get(res["source"], "?")
        rank = res["rank"]
        func = res["func_name"]
        fpath = res["file_path"]
        score = res["score"]
        label = res["similarity_label"]
        anchor = res.get("anchor_func")
        src = res["source"]

        lines.append(f"  [{rank}] {icon} {func}()  [{src}]")
        lines.append(f"      File       : {fpath}")
        if src == "vector":
            lines.append(f"      Similarity : {score:.4f} → [{label}]")
        else:
            anchor_label = f"← anchor: {anchor}" if anchor else ""
            lines.append(f"      Similarity : graph-only  {anchor_label}")
        lines.append(
            f"      PageRank   : {res['pagerank']:.6f}"
            f"  │  Churn: {res['commit_count']} commits"
            f"  │  DeadCode: {'YES' if res['is_dead_code_candidate'] else 'No'}"
        )
        lines.append("")

    lines.append("=" * 80)
    return "\n".join(lines)

"""
tests/test_hybrid_retriever.py
-------------------------------
Unit tests for rag/hybrid_retriever.py — Graph-Vector Hybrid Retriever.

Test Strategy
-------------
All tests use:
* An **in-memory QdrantClient** (``QdrantClient(":memory:")``) so no running
  Qdrant service is required.
* A **manually-constructed NetworkX DiGraph** that mirrors the node-ID
  convention used by ``parser.repo_walker`` (``<file>::func::<name>``), so
  the graph lookup logic is exercised realistically.
* ``unittest.mock.patch`` to replace ``rag.test_search.model.encode`` with a
  deterministic mock vector, isolating the embedding model from tests.

Coverage checklist
------------------
[x] Empty / blank query returns []
[x] Pure-vector fallback when graph=None
[x] include_graph_neighbors=False flag skips graph expansion
[x] Graph callee expansion (anchor calls downstream function)
[x] Graph caller expansion (upstream function calls anchor)
[x] Deduplication — neighbour not added twice even if adjacent to two anchors
[x] _find_graph_node exact-key lookup
[x] _find_graph_node fallback name-scan lookup
[x] _find_graph_node returns None for unknown function
[x] Anchor with no graph neighbours (isolated node) — no crash
[x] All result dicts contain required keys
[x] Ranks are 1-based and contiguous
[x] format_hybrid_results — basic smoke test
[x] format_hybrid_results — empty results path
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import networkx as nx
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.hybrid_retriever import (
    HybridResult,
    _assign_ranks,
    _expand_neighbors,
    _find_graph_node,
    _payload_from_graph_node,
    format_hybrid_results,
    hybrid_search,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

COLLECTION = "test_hybrid"
VECTOR_SIZE = 4  # small synthetic dimension for speed


def _make_client() -> QdrantClient:
    """Return a fresh in-memory Qdrant client with the test collection."""
    client = QdrantClient(":memory:")
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    return client


def _seed_points(client: QdrantClient, points: list[PointStruct]) -> None:
    client.upsert(collection_name=COLLECTION, points=points, wait=True)


def _mock_encode(vector: list[float]):
    """Return a patch-ready mock that makes model.encode() return *vector*."""
    mock_vec = MagicMock()
    mock_vec.tolist.return_value = vector
    return mock_vec


# ---------------------------------------------------------------------------
# Shared test graph fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_graph() -> nx.DiGraph:
    """
    Synthetic call graph with 4 function nodes and 2 call edges:

        clone_repository  (app/services/git_service.py::func::clone_repository)
              │ calls
              ▼
        cleanup_repo_directory  (app/services/git_service.py::func::cleanup_repo_directory)
              ▲ called by
        scan_repository  (parser/repo_walker.py::func::scan_repository)

    Also includes one non-function node to verify those are skipped.
    """
    g = nx.DiGraph()

    # Function nodes
    g.add_node(
        "app/services/git_service.py::func::clone_repository",
        kind="function",
        name="clone_repository",
        file="app/services/git_service.py",
        label="clone_repository",
        pagerank=0.0825,
        commit_count=14,
        is_dead_code_candidate=False,
    )
    g.add_node(
        "app/services/git_service.py::func::cleanup_repo_directory",
        kind="function",
        name="cleanup_repo_directory",
        file="app/services/git_service.py",
        label="cleanup_repo_directory",
        pagerank=0.0412,
        commit_count=8,
        is_dead_code_candidate=False,
    )
    g.add_node(
        "parser/repo_walker.py::func::scan_repository",
        kind="function",
        name="scan_repository",
        file="parser/repo_walker.py",
        label="scan_repository",
        pagerank=0.1250,
        commit_count=22,
        is_dead_code_candidate=False,
    )
    g.add_node(
        "rag/embeddings.py::func::embed_chunks",
        kind="function",
        name="embed_chunks",
        file="rag/embeddings.py",
        label="embed_chunks",
        pagerank=0.0610,
        commit_count=9,
        is_dead_code_candidate=False,
    )

    # Non-function node (should be filtered out by _expand_neighbors)
    g.add_node(
        "app/services/git_service.py::file",
        kind="file",
        path="app/services/git_service.py",
        label="app/services/git_service.py",
    )

    # Edges:
    # clone_repository → cleanup_repo_directory  (callee)
    g.add_edge(
        "app/services/git_service.py::func::clone_repository",
        "app/services/git_service.py::func::cleanup_repo_directory",
        rel="func_call",
        edge_type="EXTRACTED",
    )
    # scan_repository → clone_repository  (caller of clone)
    g.add_edge(
        "parser/repo_walker.py::func::scan_repository",
        "app/services/git_service.py::func::clone_repository",
        rel="func_call",
        edge_type="EXTRACTED",
    )
    # file → clone_repository (non-function edge, should be skipped)
    g.add_edge(
        "app/services/git_service.py::file",
        "app/services/git_service.py::func::clone_repository",
        rel="contains",
        edge_type="EXTRACTED",
    )

    return g


# ---------------------------------------------------------------------------
# Tests: _find_graph_node
# ---------------------------------------------------------------------------

class TestFindGraphNode:
    def test_exact_key_lookup(self, sample_graph):
        node_id = _find_graph_node(
            sample_graph,
            func_name="clone_repository",
            file_path="app/services/git_service.py",
        )
        assert node_id == "app/services/git_service.py::func::clone_repository"

    def test_fallback_name_scan(self, sample_graph):
        # Provide a slightly wrong file_path to force the name-scan fallback
        node_id = _find_graph_node(
            sample_graph,
            func_name="embed_chunks",
            file_path="WRONG/path.py",  # exact key won't match
        )
        assert node_id == "rag/embeddings.py::func::embed_chunks"

    def test_returns_none_for_unknown_function(self, sample_graph):
        node_id = _find_graph_node(
            sample_graph,
            func_name="nonexistent_func",
            file_path="nowhere.py",
        )
        assert node_id is None


# ---------------------------------------------------------------------------
# Tests: _payload_from_graph_node
# ---------------------------------------------------------------------------

class TestPayloadFromGraphNode:
    def test_extracts_all_fields(self, sample_graph):
        node_id = "app/services/git_service.py::func::clone_repository"
        payload = _payload_from_graph_node(sample_graph, node_id)
        assert payload["func_name"] == "clone_repository"
        assert payload["file_path"] == "app/services/git_service.py"
        assert payload["pagerank"] == pytest.approx(0.0825)
        assert payload["commit_count"] == 14
        assert payload["is_dead_code_candidate"] is False

    def test_missing_attrs_fall_back_to_defaults(self):
        g = nx.DiGraph()
        g.add_node("bare_node", kind="function")
        payload = _payload_from_graph_node(g, "bare_node")
        assert payload["func_name"] == "bare_node"  # falls back to node_id
        assert payload["file_path"] == ""
        assert payload["pagerank"] == 0.0
        assert payload["commit_count"] == 0
        assert payload["is_dead_code_candidate"] is False


# ---------------------------------------------------------------------------
# Tests: _expand_neighbors
# ---------------------------------------------------------------------------

class TestExpandNeighbors:
    def test_callees_returned(self, sample_graph):
        anchor_id = "app/services/git_service.py::func::clone_repository"
        results = _expand_neighbors(sample_graph, anchor_id, "clone_repository")
        callee_names = [r["func_name"] for r in results if r["source"] == "graph_callee"]
        assert "cleanup_repo_directory" in callee_names

    def test_callers_returned(self, sample_graph):
        anchor_id = "app/services/git_service.py::func::clone_repository"
        results = _expand_neighbors(sample_graph, anchor_id, "clone_repository")
        caller_names = [r["func_name"] for r in results if r["source"] == "graph_caller"]
        assert "scan_repository" in caller_names

    def test_non_function_nodes_filtered(self, sample_graph):
        """The file node adjacent to clone_repository must not appear in output."""
        anchor_id = "app/services/git_service.py::func::clone_repository"
        results = _expand_neighbors(sample_graph, anchor_id, "clone_repository")
        func_names = {r["func_name"] for r in results}
        # The file node has kind="file" so it should be excluded
        assert "app/services/git_service.py" not in func_names

    def test_score_is_zero_for_all_neighbours(self, sample_graph):
        anchor_id = "app/services/git_service.py::func::clone_repository"
        results = _expand_neighbors(sample_graph, anchor_id, "clone_repository")
        for r in results:
            assert r["score"] == 0.0

    def test_anchor_func_set_correctly(self, sample_graph):
        anchor_id = "app/services/git_service.py::func::clone_repository"
        results = _expand_neighbors(sample_graph, anchor_id, "clone_repository")
        for r in results:
            assert r["anchor_func"] == "clone_repository"

    def test_isolated_node_returns_empty(self):
        """A node with no edges must return an empty list without error."""
        g = nx.DiGraph()
        g.add_node("solo.py::func::lonely", kind="function", name="lonely", file="solo.py")
        results = _expand_neighbors(g, "solo.py::func::lonely", "lonely")
        assert results == []


# ---------------------------------------------------------------------------
# Tests: _assign_ranks
# ---------------------------------------------------------------------------

class TestAssignRanks:
    def test_ranks_are_contiguous_and_1_indexed(self):
        items: list[HybridResult] = [{"func_name": f"f{i}"} for i in range(5)]  # type: ignore[assignment]
        _assign_ranks(items)
        assert [r["rank"] for r in items] == [1, 2, 3, 4, 5]

    def test_empty_list_no_error(self):
        _assign_ranks([])  # should not raise


# ---------------------------------------------------------------------------
# Tests: hybrid_search
# ---------------------------------------------------------------------------

class TestHybridSearch:
    """Integration-level tests using in-memory Qdrant + synthetic graph."""

    def _setup_client_with_clone(self) -> QdrantClient:
        """Seed a single 'clone_repository' point that will be our anchor."""
        client = _make_client()
        _seed_points(
            client,
            [
                PointStruct(
                    id=1,
                    vector=[1.0, 0.0, 0.0, 0.0],
                    payload={
                        "func_name": "clone_repository",
                        "file_path": "app/services/git_service.py",
                        "pagerank": 0.0825,
                        "commit_count": 14,
                        "is_dead_code_candidate": False,
                    },
                )
            ],
        )
        return client

    # -- Guard --

    def test_empty_query_returns_empty(self, sample_graph):
        results = hybrid_search("", graph=sample_graph)
        assert results == []

    def test_blank_whitespace_query_returns_empty(self, sample_graph):
        results = hybrid_search("   ", graph=sample_graph)
        assert results == []

    # -- Pure-vector fallback --

    def test_no_graph_returns_vector_only(self):
        """When graph=None, results have source='vector' only."""
        client = self._setup_client_with_clone()
        with patch("rag.test_search.model.encode") as mock_enc:
            mock_enc.return_value = _mock_encode([1.0, 0.0, 0.0, 0.0])
            results = hybrid_search(
                "clone git repo",
                graph=None,
                top_k=3,
                client=client,
                collection_name=COLLECTION,
            )
        assert len(results) >= 1
        for r in results:
            assert r["source"] == "vector"

    def test_include_graph_neighbors_false_skips_expansion(self, sample_graph):
        client = self._setup_client_with_clone()
        with patch("rag.test_search.model.encode") as mock_enc:
            mock_enc.return_value = _mock_encode([1.0, 0.0, 0.0, 0.0])
            results = hybrid_search(
                "clone git repo",
                graph=sample_graph,
                top_k=3,
                client=client,
                collection_name=COLLECTION,
                include_graph_neighbors=False,
            )
        assert all(r["source"] == "vector" for r in results)

    # -- Graph expansion --

    def test_graph_callees_included(self, sample_graph):
        """cleanup_repo_directory (callee of clone_repository) must appear."""
        client = self._setup_client_with_clone()
        with patch("rag.test_search.model.encode") as mock_enc:
            mock_enc.return_value = _mock_encode([1.0, 0.0, 0.0, 0.0])
            results = hybrid_search(
                "clone git repo",
                graph=sample_graph,
                top_k=1,
                client=client,
                collection_name=COLLECTION,
            )
        func_names = {r["func_name"] for r in results}
        assert "cleanup_repo_directory" in func_names

    def test_graph_callers_included(self, sample_graph):
        """scan_repository (caller of clone_repository) must appear."""
        client = self._setup_client_with_clone()
        with patch("rag.test_search.model.encode") as mock_enc:
            mock_enc.return_value = _mock_encode([1.0, 0.0, 0.0, 0.0])
            results = hybrid_search(
                "clone git repo",
                graph=sample_graph,
                top_k=1,
                client=client,
                collection_name=COLLECTION,
            )
        func_names = {r["func_name"] for r in results}
        assert "scan_repository" in func_names

    def test_source_labels_correct(self, sample_graph):
        client = self._setup_client_with_clone()
        with patch("rag.test_search.model.encode") as mock_enc:
            mock_enc.return_value = _mock_encode([1.0, 0.0, 0.0, 0.0])
            results = hybrid_search(
                "clone git repo",
                graph=sample_graph,
                top_k=1,
                client=client,
                collection_name=COLLECTION,
            )
        sources = {r["source"] for r in results}
        # We expect at least vector + one graph source
        assert "vector" in sources
        assert sources & {"graph_callee", "graph_caller"}

    def test_ranks_are_1_based_and_contiguous(self, sample_graph):
        client = self._setup_client_with_clone()
        with patch("rag.test_search.model.encode") as mock_enc:
            mock_enc.return_value = _mock_encode([1.0, 0.0, 0.0, 0.0])
            results = hybrid_search(
                "clone git repo",
                graph=sample_graph,
                top_k=1,
                client=client,
                collection_name=COLLECTION,
            )
        ranks = [r["rank"] for r in results]
        assert ranks == list(range(1, len(results) + 1))

    def test_required_keys_present(self, sample_graph):
        expected_keys = {
            "rank", "func_name", "file_path", "score", "similarity_label",
            "pagerank", "commit_count", "is_dead_code_candidate",
            "source", "anchor_func",
        }
        client = self._setup_client_with_clone()
        with patch("rag.test_search.model.encode") as mock_enc:
            mock_enc.return_value = _mock_encode([1.0, 0.0, 0.0, 0.0])
            results = hybrid_search(
                "clone git repo",
                graph=sample_graph,
                top_k=1,
                client=client,
                collection_name=COLLECTION,
            )
        for r in results:
            assert expected_keys.issubset(r.keys()), (
                f"Missing keys in result: {expected_keys - r.keys()}"
            )

    # -- Deduplication --

    def test_no_duplicate_func_name_file_path_pairs(self, sample_graph):
        """Even when two anchors share a neighbour, it must appear only once."""
        client = _make_client()
        # Seed two anchors that both expand to the same neighbour
        _seed_points(
            client,
            [
                PointStruct(
                    id=1,
                    vector=[1.0, 0.0, 0.0, 0.0],
                    payload={
                        "func_name": "clone_repository",
                        "file_path": "app/services/git_service.py",
                        "pagerank": 0.0825,
                        "commit_count": 14,
                        "is_dead_code_candidate": False,
                    },
                ),
                PointStruct(
                    id=2,
                    vector=[0.9, 0.1, 0.0, 0.0],
                    payload={
                        "func_name": "scan_repository",
                        "file_path": "parser/repo_walker.py",
                        "pagerank": 0.1250,
                        "commit_count": 22,
                        "is_dead_code_candidate": False,
                    },
                ),
            ],
        )
        with patch("rag.test_search.model.encode") as mock_enc:
            mock_enc.return_value = _mock_encode([1.0, 0.0, 0.0, 0.0])
            results = hybrid_search(
                "repository scanning",
                graph=sample_graph,
                top_k=2,
                client=client,
                collection_name=COLLECTION,
            )

        # Build (func_name, file_path) tuples and assert uniqueness
        pairs = [(r["func_name"], r["file_path"]) for r in results]
        assert len(pairs) == len(set(pairs)), "Duplicate (func_name, file_path) found!"

    def test_anchor_not_added_again_as_graph_neighbour(self, sample_graph):
        """If anchor A calls anchor B, B must not appear twice in results."""
        client = _make_client()
        # Both clone_repository and cleanup_repo_directory are Qdrant results,
        # and clone calls cleanup — cleanup must not be added a second time
        # as a graph_callee of clone.
        _seed_points(
            client,
            [
                PointStruct(
                    id=1,
                    vector=[1.0, 0.0, 0.0, 0.0],
                    payload={
                        "func_name": "clone_repository",
                        "file_path": "app/services/git_service.py",
                        "pagerank": 0.0825,
                        "commit_count": 14,
                        "is_dead_code_candidate": False,
                    },
                ),
                PointStruct(
                    id=2,
                    vector=[0.9, 0.1, 0.0, 0.0],
                    payload={
                        "func_name": "cleanup_repo_directory",
                        "file_path": "app/services/git_service.py",
                        "pagerank": 0.0412,
                        "commit_count": 8,
                        "is_dead_code_candidate": False,
                    },
                ),
            ],
        )
        with patch("rag.test_search.model.encode") as mock_enc:
            mock_enc.return_value = _mock_encode([1.0, 0.0, 0.0, 0.0])
            results = hybrid_search(
                "cleanup after clone",
                graph=sample_graph,
                top_k=2,
                client=client,
                collection_name=COLLECTION,
            )

        cleanup_hits = [
            r for r in results if r["func_name"] == "cleanup_repo_directory"
        ]
        assert len(cleanup_hits) == 1, (
            "cleanup_repo_directory should appear exactly once, not as both "
            "a vector anchor and a graph_callee"
        )

    def test_anchor_with_no_graph_match_does_not_crash(self):
        """Anchor whose func_name isn't in the graph should be skipped silently."""
        g = nx.DiGraph()  # empty graph — no matching node
        client = _make_client()
        _seed_points(
            client,
            [
                PointStruct(
                    id=1,
                    vector=[1.0, 0.0, 0.0, 0.0],
                    payload={
                        "func_name": "some_unknown_func",
                        "file_path": "nowhere.py",
                        "pagerank": 0.0,
                        "commit_count": 0,
                        "is_dead_code_candidate": False,
                    },
                )
            ],
        )
        with patch("rag.test_search.model.encode") as mock_enc:
            mock_enc.return_value = _mock_encode([1.0, 0.0, 0.0, 0.0])
            results = hybrid_search(
                "some query",
                graph=g,
                top_k=1,
                client=client,
                collection_name=COLLECTION,
            )
        # Should still return the vector anchor; no crash
        assert len(results) == 1
        assert results[0]["source"] == "vector"


# ---------------------------------------------------------------------------
# Tests: format_hybrid_results
# ---------------------------------------------------------------------------

class TestFormatHybridResults:
    def _make_sample_results(self) -> list[HybridResult]:
        return [
            {
                "rank": 1,
                "func_name": "clone_repository",
                "file_path": "app/services/git_service.py",
                "score": 0.92,
                "similarity_label": "HIGH CONFIDENCE",
                "pagerank": 0.0825,
                "commit_count": 14,
                "is_dead_code_candidate": False,
                "source": "vector",
                "anchor_func": None,
            },
            {
                "rank": 2,
                "func_name": "cleanup_repo_directory",
                "file_path": "app/services/git_service.py",
                "score": 0.0,
                "similarity_label": "WEAK / NOISE",
                "pagerank": 0.0412,
                "commit_count": 8,
                "is_dead_code_candidate": False,
                "source": "graph_callee",
                "anchor_func": "clone_repository",
            },
            {
                "rank": 3,
                "func_name": "scan_repository",
                "file_path": "parser/repo_walker.py",
                "score": 0.0,
                "similarity_label": "WEAK / NOISE",
                "pagerank": 0.1250,
                "commit_count": 22,
                "is_dead_code_candidate": False,
                "source": "graph_caller",
                "anchor_func": "clone_repository",
            },
        ]

    def test_contains_query_header(self):
        results = self._make_sample_results()
        output = format_hybrid_results("clone a repo", results)
        assert "HYBRID QUERY" in output
        assert "clone a repo" in output

    def test_contains_all_func_names(self):
        results = self._make_sample_results()
        output = format_hybrid_results("clone a repo", results)
        assert "clone_repository" in output
        assert "cleanup_repo_directory" in output
        assert "scan_repository" in output

    def test_contains_source_labels(self):
        results = self._make_sample_results()
        output = format_hybrid_results("clone a repo", results)
        assert "vector" in output
        assert "graph_callee" in output
        assert "graph_caller" in output

    def test_contains_anchor_reference(self):
        results = self._make_sample_results()
        output = format_hybrid_results("clone a repo", results)
        assert "anchor: clone_repository" in output

    def test_empty_results_handled(self):
        output = format_hybrid_results("empty query", [])
        assert "No results found" in output

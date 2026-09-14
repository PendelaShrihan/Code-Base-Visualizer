"""
tests/test_search.py
--------------------
Unit and integration tests for rag/test_search.py semantic retrieval
and vector similarity evaluation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.test_search import (
    classify_similarity,
    format_results,
    search_functions,
    seed_sample_functions,
)


def test_classify_similarity():
    """Verify threshold classification tiers."""
    assert classify_similarity(0.85) == "HIGH CONFIDENCE"
    assert classify_similarity(0.70) == "HIGH CONFIDENCE"
    assert classify_similarity(0.69) == "MODERATE RELEVANCE"
    assert classify_similarity(0.45) == "MODERATE RELEVANCE"
    assert classify_similarity(0.44) == "WEAK / NOISE"
    assert classify_similarity(0.10) == "WEAK / NOISE"


def test_search_functions_empty_query():
    """Empty query should immediately return empty results without hitting Qdrant."""
    assert search_functions("") == []
    assert search_functions("   ") == []


def test_search_functions_in_memory():
    """Test search_functions with an in-memory Qdrant instance and mocked embeddings."""
    client = QdrantClient(":memory:")
    collection_name = "test_code_search"

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )

    # Seed 3 dummy points
    points = [
        PointStruct(
            id=1,
            vector=[1.0, 0.0, 0.0, 0.0],
            payload={
                "func_name": "target_func",
                "file_path": "app/target.py",
                "pagerank": 0.05,
                "commit_count": 10,
                "is_dead_code_candidate": False,
            },
        ),
        PointStruct(
            id=2,
            vector=[0.5, 0.5, 0.0, 0.0],
            payload={
                "func_name": "related_func",
                "file_path": "app/related.py",
                "pagerank": 0.02,
                "commit_count": 3,
                "is_dead_code_candidate": False,
            },
        ),
        PointStruct(
            id=3,
            vector=[0.0, 0.0, 1.0, 0.0],
            payload={
                "func_name": "unrelated_func",
                "file_path": "app/unrelated.py",
                "pagerank": 0.01,
                "commit_count": 1,
                "is_dead_code_candidate": True,
            },
        ),
    ]
    client.upsert(collection_name=collection_name, points=points)

    # Mock model.encode to return vector [1.0, 0.0, 0.0, 0.0] matching point 1 exactly
    with patch("rag.test_search.model.encode") as mock_encode:
        mock_vec = MagicMock()
        mock_vec.tolist.return_value = [1.0, 0.0, 0.0, 0.0]
        mock_encode.return_value = mock_vec

        results = search_functions(
            query="find target function",
            top_k=3,
            client=client,
            collection_name=collection_name,
        )

        assert len(results) == 3
        # Top result should be target_func with highest cosine score
        assert results[0]["rank"] == 1
        assert results[0]["func_name"] == "target_func"
        assert results[0]["score"] == pytest.approx(1.0, abs=1e-3)
        assert results[0]["similarity_label"] == "HIGH CONFIDENCE"

        # Second result should be related_func
        assert results[1]["rank"] == 2
        assert results[1]["func_name"] == "related_func"
        assert results[1]["score"] > results[2]["score"]

        # Formatted string verification
        formatted = format_results("find target function", results)
        assert "target_func()" in formatted
        assert "HIGH CONFIDENCE" in formatted
        assert "Confidence Gap" in formatted


def test_seed_sample_functions_in_memory():
    """Verify seed_sample_functions populates vectors into in-memory collection."""
    client = QdrantClient(":memory:")
    count = seed_sample_functions(client=client, collection_name="seeded_test")
    assert count > 0

    info = client.get_collection("seeded_test")
    assert info.points_count == count


def test_repo_id_isolation_search_functions():
    """
    Test multi-tenant repository isolation:
    Seed two synthetic repositories ('repo-alpha' and 'repo-beta') with
    identical function names into the same Qdrant collection.
    Query scoped to one repo_id and confirm ZERO results leak in from the other.
    """
    client = QdrantClient(":memory:")
    collection_name = "test_multi_tenant_isolation"

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=4, distance=Distance.COSINE),
    )

    # Seed overlapping function names across two distinct repos
    points = [
        # repo-alpha points
        PointStruct(
            id=101,
            vector=[1.0, 0.0, 0.0, 0.0],
            payload={
                "func_name": "clone_repository",
                "file_path": "alpha/git_service.py",
                "repo_id": "repo-alpha",
                "pagerank": 0.05,
                "commit_count": 10,
                "is_dead_code_candidate": False,
            },
        ),
        PointStruct(
            id=102,
            vector=[0.8, 0.2, 0.0, 0.0],
            payload={
                "func_name": "parse_ast",
                "file_path": "alpha/ast_parser.py",
                "repo_id": "repo-alpha",
                "pagerank": 0.03,
                "commit_count": 5,
                "is_dead_code_candidate": False,
            },
        ),
        PointStruct(
            id=103,
            vector=[0.6, 0.4, 0.0, 0.0],
            payload={
                "func_name": "compute_metrics",
                "file_path": "alpha/metrics.py",
                "repo_id": "repo-alpha",
                "pagerank": 0.02,
                "commit_count": 2,
                "is_dead_code_candidate": False,
            },
        ),
        # repo-beta points with identical/overlapping function names
        PointStruct(
            id=201,
            vector=[1.0, 0.0, 0.0, 0.0],  # identical vector to id 101
            payload={
                "func_name": "clone_repository",
                "file_path": "beta/vcs/git.py",
                "repo_id": "repo-beta",
                "pagerank": 0.08,
                "commit_count": 20,
                "is_dead_code_candidate": False,
            },
        ),
        PointStruct(
            id=202,
            vector=[0.8, 0.2, 0.0, 0.0],  # identical vector to id 102
            payload={
                "func_name": "parse_ast",
                "file_path": "beta/parsing/syntax.py",
                "repo_id": "repo-beta",
                "pagerank": 0.04,
                "commit_count": 8,
                "is_dead_code_candidate": False,
            },
        ),
        PointStruct(
            id=203,
            vector=[0.6, 0.4, 0.0, 0.0],  # identical vector to id 103
            payload={
                "func_name": "compute_metrics",
                "file_path": "beta/analytics/stats.py",
                "repo_id": "repo-beta",
                "pagerank": 0.01,
                "commit_count": 1,
                "is_dead_code_candidate": False,
            },
        ),
    ]
    client.upsert(collection_name=collection_name, points=points, wait=True)

    with patch("rag.test_search.model.encode") as mock_encode:
        mock_vec = MagicMock()
        mock_vec.tolist.return_value = [1.0, 0.0, 0.0, 0.0]
        mock_encode.return_value = mock_vec

        # 1. Query scoped to repo-alpha: MUST return ONLY repo-alpha results
        results_alpha = search_functions(
            query="clone repository",
            top_k=10,
            client=client,
            collection_name=collection_name,
            repo_id="repo-alpha",
        )
        assert len(results_alpha) == 3
        # Assert strictly zero leakage from repo-beta
        for res in results_alpha:
            assert res["repo_id"] == "repo-alpha", f"Leaked result from wrong repo: {res}"
            assert res["file_path"].startswith("alpha/"), f"File path leaked from wrong repo: {res['file_path']}"
        assert not any(res["repo_id"] == "repo-beta" for res in results_alpha)

        # 2. Query scoped to repo-beta: MUST return ONLY repo-beta results
        results_beta = search_functions(
            query="clone repository",
            top_k=10,
            client=client,
            collection_name=collection_name,
            repo_id="repo-beta",
        )
        assert len(results_beta) == 3
        # Assert strictly zero leakage from repo-alpha
        for res in results_beta:
            assert res["repo_id"] == "repo-beta", f"Leaked result from wrong repo: {res}"
            assert res["file_path"].startswith("beta/"), f"File path leaked from wrong repo: {res['file_path']}"
        assert not any(res["repo_id"] == "repo-alpha" for res in results_beta)

        # 3. Query without repo_id: returns results from both repositories (unfiltered)
        results_unscoped = search_functions(
            query="clone repository",
            top_k=10,
            client=client,
            collection_name=collection_name,
            repo_id=None,
        )
        assert len(results_unscoped) == 6
        repo_ids = {res["repo_id"] for res in results_unscoped}
        assert repo_ids == {"repo-alpha", "repo-beta"}

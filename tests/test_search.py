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

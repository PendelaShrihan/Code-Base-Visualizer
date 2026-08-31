"""
tests/test_qdrant.py
--------------------
Unit tests for Qdrant client connection, code_chunks schema creation,
and collection configuration verification.
"""

from __future__ import annotations

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, PayloadSchemaType, VectorParams

from rag.qdrant_client import (
    COLLECTION_NAME,
    PAYLOAD_INDEXES,
    VECTOR_DISTANCE,
    VECTOR_SIZE,
    create_code_chunks_collection,
    get_qdrant_client,
    verify_collection_config,
)


def test_qdrant_client_factory():
    """Verify get_qdrant_client returns a QdrantClient instance."""
    client = get_qdrant_client(location=":memory:")
    assert isinstance(client, QdrantClient)


def test_create_and_verify_collection_in_memory():
    """
    Test creating the code_chunks collection in an in-memory Qdrant client
    and verifying vector dimensions, distance metric, and configuration.
    """
    client = QdrantClient(":memory:")
    info = create_code_chunks_collection(
        client=client,
        collection_name="test_code_chunks",
        recreate=True,
    )

    assert info is not None
    # Verify vector size is 384 (all-MiniLM-L6-v2)
    assert info.config.params.vectors.size == 384
    # Verify distance metric is Cosine
    assert info.config.params.vectors.distance == Distance.COSINE
    # Verify points count is 0 (schema-only, no vectors yet)
    assert info.points_count == 0


def test_verify_collection_config_validation():
    """Test that verify_collection_config raises errors on nonexistent collection or mismatched settings."""
    client = QdrantClient(":memory:")

    # Non-existent collection
    with pytest.raises(ValueError, match="does not exist"):
        verify_collection_config(client=client, collection_name="non_existent")

    # Mismatched vector size
    client.create_collection(
        collection_name="wrong_size_col",
        vectors_config=VectorParams(size=512, distance=Distance.COSINE),
    )
    with pytest.raises(ValueError, match="vector size mismatch"):
        verify_collection_config(
            client=client,
            collection_name="wrong_size_col",
            expected_size=384,
        )

    # Mismatched distance metric
    client.create_collection(
        collection_name="wrong_dist_col",
        vectors_config=VectorParams(size=384, distance=Distance.DOT),
    )
    with pytest.raises(ValueError, match="distance mismatch"):
        verify_collection_config(
            client=client,
            collection_name="wrong_dist_col",
            expected_distance=Distance.COSINE,
        )


def test_payload_indexes_definition():
    """Verify that payload indexes for file_path and func_name are defined as KEYWORD type."""
    indexed_dict = dict(PAYLOAD_INDEXES)
    assert "file_path" in indexed_dict
    assert indexed_dict["file_path"] == PayloadSchemaType.KEYWORD
    assert "func_name" in indexed_dict
    assert indexed_dict["func_name"] == PayloadSchemaType.KEYWORD


def test_live_qdrant_server_collection():
    """
    Integration test against local/docker Qdrant server if running.
    Checks schema, vector size=384, distance=Cosine, and payload schema.
    """
    try:
        client = get_qdrant_client()
        # Ping
        client.get_collections()
    except Exception as exc:
        pytest.skip(f"Live Qdrant server not reachable: {exc}")

    info = create_code_chunks_collection(client=client, collection_name=COLLECTION_NAME)
    assert info.config.params.vectors.size == VECTOR_SIZE
    assert info.config.params.vectors.distance == VECTOR_DISTANCE
    assert "file_path" in info.payload_schema
    assert "func_name" in info.payload_schema
    assert info.payload_schema["file_path"].data_type == PayloadSchemaType.KEYWORD
    assert info.payload_schema["func_name"].data_type == PayloadSchemaType.KEYWORD

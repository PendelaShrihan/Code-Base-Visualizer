"""
tests/test_ingestion.py
-----------------------
Unit tests for the rag.ingestion batch vector ingestion pipeline.

Strategy
--------
* All Qdrant network calls are intercepted with ``unittest.mock`` so tests run
  without a live Qdrant server.
* SentenceTransformer encoding is patched to return deterministic zero-vectors,
  which keeps tests fast (no model load) and output-deterministic.
* Tests are structured as: pure-logic → side-effects → integration entry-point.
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import MagicMock, call, patch

import pytest

from rag.ingestion import (
    DEFAULT_BATCH_SIZE,
    _UUID_NAMESPACE,
    batch_upsert_chunks,
    extract_function_chunks,
    ingest_graph,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_graph_dict(nodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a minimal graph_dict with the supplied nodes list."""
    return {
        "meta": {"node_count": len(nodes), "edge_count": 0},
        "nodes": nodes,
        "edges": [],
    }


def _func_node(
    node_id: str = "app/foo.py::func::my_func",
    name: str = "my_func",
    file: str = "app/foo.py",
    **extra: Any,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "kind": "function",
        "name": name,
        "file": file,
        "pagerank": 0.001,
        "commit_count": 2,
        "is_dead_code_candidate": False,
        **extra,
    }


# ---------------------------------------------------------------------------
# extract_function_chunks — pure logic tests (no I/O)
# ---------------------------------------------------------------------------

class TestExtractFunctionChunks:
    """Tests for the chunk extraction step."""

    def test_returns_empty_for_empty_graph(self):
        chunks = extract_function_chunks(_make_graph_dict([]))
        assert chunks == []

    def test_filters_non_function_nodes(self):
        """file, class, import, and call_target nodes must be excluded."""
        graph = _make_graph_dict(
            [
                {"id": "foo.py::file", "kind": "file", "path": "foo.py"},
                {"id": "foo.py::class::Bar", "kind": "class", "name": "Bar", "file": "foo.py"},
                {"id": "foo.py::import::os", "kind": "import", "name": "os", "file": "foo.py"},
                {"id": "foo.py::call::bar.baz", "kind": "call_target", "name": "bar.baz", "file": "foo.py"},
            ]
        )
        chunks = extract_function_chunks(graph)
        assert chunks == []

    def test_extracts_function_nodes(self):
        """One function node → one chunk dict with required keys."""
        node = _func_node()
        chunks = extract_function_chunks(_make_graph_dict([node]))

        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk["func_name"] == "my_func"
        assert chunk["file_path"] == "app/foo.py"
        assert chunk["text"] == "func: my_func\nfile: app/foo.py"
        assert chunk["pagerank"] == pytest.approx(0.001)
        assert chunk["commit_count"] == 2
        assert chunk["is_dead_code_candidate"] is False

    def test_mixed_nodes_only_extracts_functions(self):
        nodes = [
            {"id": "a.py::file", "kind": "file", "path": "a.py"},
            _func_node("a.py::func::alpha", "alpha", "a.py"),
            {"id": "a.py::class::MyClass", "kind": "class", "name": "MyClass", "file": "a.py"},
            _func_node("a.py::func::beta", "beta", "a.py"),
        ]
        chunks = extract_function_chunks(_make_graph_dict(nodes))
        assert len(chunks) == 2
        names = {c["func_name"] for c in chunks}
        assert names == {"alpha", "beta"}

    def test_deterministic_uuid_v5_chunk_ids(self):
        """Same node_id must always produce the same chunk_id (UUID v5)."""
        node_id = "app/services/git_service.py::func::clone_repository"
        expected_uuid = str(uuid.uuid5(_UUID_NAMESPACE, node_id))

        node = _func_node(node_id=node_id, name="clone_repository", file="app/services/git_service.py")
        chunks = extract_function_chunks(_make_graph_dict([node]))

        assert chunks[0]["chunk_id"] == expected_uuid

    def test_different_nodes_get_different_chunk_ids(self):
        nodes = [
            _func_node("a.py::func::foo", "foo", "a.py"),
            _func_node("a.py::func::bar", "bar", "a.py"),
        ]
        chunks = extract_function_chunks(_make_graph_dict(nodes))
        ids = [c["chunk_id"] for c in chunks]
        assert ids[0] != ids[1]

    def test_chunk_id_is_valid_uuid_string(self):
        node = _func_node()
        chunks = extract_function_chunks(_make_graph_dict([node]))
        # Must not raise ValueError — i.e., it is a valid UUID
        parsed = uuid.UUID(chunks[0]["chunk_id"])
        assert parsed.version == 5

    def test_missing_optional_fields_default_safely(self):
        """Nodes without pagerank / commit_count / is_dead_code_candidate default to 0 / False."""
        node = {"id": "bare.py::func::bare", "kind": "function", "name": "bare", "file": "bare.py"}
        chunks = extract_function_chunks(_make_graph_dict([node]))
        assert chunks[0]["pagerank"] == 0.0
        assert chunks[0]["commit_count"] == 0
        assert chunks[0]["is_dead_code_candidate"] is False

    def test_uses_label_when_name_missing(self):
        """Falls back to 'label' attribute when 'name' is absent."""
        node = {"id": "x.py::func::helper", "kind": "function", "label": "helper", "file": "x.py"}
        chunks = extract_function_chunks(_make_graph_dict([node]))
        assert chunks[0]["func_name"] == "helper"

    def test_uses_path_when_file_missing(self):
        """Falls back to 'path' attribute when 'file' is absent."""
        node = {"id": "x.py::file", "kind": "function", "name": "foo", "path": "x.py"}
        chunks = extract_function_chunks(_make_graph_dict([node]))
        assert chunks[0]["file_path"] == "x.py"


# ---------------------------------------------------------------------------
# batch_upsert_chunks — mocked Qdrant
# ---------------------------------------------------------------------------

# Zero-vector factory matching all-MiniLM-L6-v2 output shape (384-d)
_ZERO_VECTOR = [0.0] * 384


def _make_chunks(n: int) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": str(uuid.uuid5(_UUID_NAMESPACE, f"node_{i}")),
            "text": f"func: fn_{i}\nfile: f.py",
            "func_name": f"fn_{i}",
            "file_path": "f.py",
            "pagerank": 0.0,
            "commit_count": 0,
            "is_dead_code_candidate": False,
        }
        for i in range(n)
    ]


class TestBatchUpsertChunks:
    """Tests for the embed + upsert step, with Qdrant and encoder mocked."""

    @pytest.fixture(autouse=True)
    def _patch_embed(self):
        """Replace embed_chunks so it populates 'embedding' without loading the model."""
        def _fake_embed(chunks, batch_size=32, show_progress_bar=False):
            for chunk in chunks:
                chunk["embedding"] = _ZERO_VECTOR
            return chunks

        with patch("rag.ingestion.embed_chunks", side_effect=_fake_embed):
            yield

    def test_returns_zero_zero_for_empty_input(self):
        mock_client = MagicMock()
        total, batches = batch_upsert_chunks([], client=mock_client)
        assert total == 0
        assert batches == 0
        mock_client.upsert.assert_not_called()

    def test_single_batch_when_chunks_lt_batch_size(self):
        mock_client = MagicMock()
        chunks = _make_chunks(10)
        total, batches = batch_upsert_chunks(chunks, client=mock_client, batch_size=64)

        assert total == 10
        assert batches == 1
        mock_client.upsert.assert_called_once()

    def test_multiple_batches_for_large_input(self):
        mock_client = MagicMock()
        chunks = _make_chunks(130)
        total, batches = batch_upsert_chunks(chunks, client=mock_client, batch_size=64)

        # 130 chunks / 64 per batch = 3 batches (64 + 64 + 2)
        assert total == 130
        assert batches == 3
        assert mock_client.upsert.call_count == 3

    def test_upsert_called_with_wait_true(self):
        """Qdrant upsert must use wait=True for write consistency."""
        mock_client = MagicMock()
        chunks = _make_chunks(5)
        batch_upsert_chunks(chunks, client=mock_client, batch_size=64)

        _, kwargs = mock_client.upsert.call_args
        assert kwargs.get("wait") is True

    def test_point_ids_match_chunk_ids(self):
        """Each PointStruct's id must match the chunk's chunk_id."""
        mock_client = MagicMock()
        chunks = _make_chunks(3)
        expected_ids = [c["chunk_id"] for c in chunks]

        batch_upsert_chunks(chunks, client=mock_client, batch_size=64)

        _, kwargs = mock_client.upsert.call_args
        actual_ids = [p.id for p in kwargs["points"]]
        assert actual_ids == expected_ids

    def test_batch_size_of_one_calls_upsert_per_chunk(self):
        """batch_size=1 means one upsert call per chunk."""
        mock_client = MagicMock()
        n = 5
        chunks = _make_chunks(n)
        total, batches = batch_upsert_chunks(chunks, client=mock_client, batch_size=1)

        assert total == n
        assert batches == n
        assert mock_client.upsert.call_count == n

    def test_exact_batch_boundary(self):
        """When len(chunks) is exactly a multiple of batch_size, no partial batch."""
        mock_client = MagicMock()
        chunks = _make_chunks(64)
        total, batches = batch_upsert_chunks(chunks, client=mock_client, batch_size=64)

        assert total == 64
        assert batches == 1

    def test_payload_contains_metadata_fields(self):
        """Each PointStruct payload must contain the expected metadata keys."""
        mock_client = MagicMock()
        chunks = _make_chunks(2)
        # Give them distinct values
        chunks[0]["func_name"] = "hello"
        chunks[0]["file_path"] = "greet.py"
        chunks[0]["pagerank"] = 0.99
        chunks[0]["commit_count"] = 7
        chunks[0]["is_dead_code_candidate"] = True

        batch_upsert_chunks(chunks, client=mock_client, batch_size=64)

        _, kwargs = mock_client.upsert.call_args
        payload = kwargs["points"][0].payload
        assert payload["func_name"] == "hello"
        assert payload["file_path"] == "greet.py"
        assert payload["pagerank"] == pytest.approx(0.99)
        assert payload["commit_count"] == 7
        assert payload["is_dead_code_candidate"] is True


# ---------------------------------------------------------------------------
# ingest_graph — entry-point integration tests (all I/O mocked)
# ---------------------------------------------------------------------------

class TestIngestGraph:
    """Tests for the public ingest_graph() entry-point."""

    @pytest.fixture(autouse=True)
    def _patch_deps(self):
        """Mock all external I/O: Qdrant client, collection creation, embed_chunks."""
        self.mock_client = MagicMock()

        def _fake_embed(chunks, batch_size=32, show_progress_bar=False):
            for chunk in chunks:
                chunk["embedding"] = _ZERO_VECTOR
            return chunks

        with (
            patch("rag.ingestion.get_qdrant_client", return_value=self.mock_client),
            patch("rag.ingestion.create_code_chunks_collection") as mock_create,
            patch("rag.ingestion.embed_chunks", side_effect=_fake_embed),
        ):
            self.mock_create_collection = mock_create
            yield

    def test_returns_empty_status_for_graph_with_no_functions(self):
        graph = _make_graph_dict(
            [{"id": "a.py::file", "kind": "file", "path": "a.py"}]
        )
        summary = ingest_graph(graph, client=self.mock_client)

        assert summary["status"] == "empty"
        assert summary["chunks_extracted"] == 0
        assert summary["chunks_upserted"] == 0
        assert summary["batches"] == 0
        self.mock_client.upsert.assert_not_called()

    def test_returns_ok_status_with_function_nodes(self):
        nodes = [_func_node(f"f.py::func::fn{i}", f"fn{i}", "f.py") for i in range(5)]
        graph = _make_graph_dict(nodes)
        summary = ingest_graph(graph, client=self.mock_client)

        assert summary["status"] == "ok"
        assert summary["chunks_extracted"] == 5
        assert summary["chunks_upserted"] == 5
        assert summary["collection"] == "code_chunks"

    def test_collection_created_before_upsert(self):
        """create_code_chunks_collection must be called (lazy init)."""
        graph = _make_graph_dict([_func_node()])
        ingest_graph(graph, client=self.mock_client)

        self.mock_create_collection.assert_called_once_with(
            client=self.mock_client, collection_name="code_chunks"
        )

    def test_custom_collection_name_propagates(self):
        graph = _make_graph_dict([_func_node()])
        summary = ingest_graph(graph, client=self.mock_client, collection_name="my_col")

        assert summary["collection"] == "my_col"
        self.mock_create_collection.assert_called_once_with(
            client=self.mock_client, collection_name="my_col"
        )
        _, kwargs = self.mock_client.upsert.call_args
        assert kwargs["collection_name"] == "my_col"

    def test_summary_keys_present(self):
        graph = _make_graph_dict([_func_node()])
        summary = ingest_graph(graph, client=self.mock_client)

        required_keys = {"chunks_extracted", "chunks_upserted", "batches", "collection", "status"}
        assert required_keys <= set(summary.keys())

    def test_batch_size_forwarded(self):
        """batch_size parameter must be respected — 3 chunks with batch_size=1 → 3 upserts."""
        nodes = [_func_node(f"f.py::func::fn{i}", f"fn{i}", "f.py") for i in range(3)]
        graph = _make_graph_dict(nodes)
        summary = ingest_graph(graph, client=self.mock_client, batch_size=1)

        assert self.mock_client.upsert.call_count == 3
        assert summary["batches"] == 3

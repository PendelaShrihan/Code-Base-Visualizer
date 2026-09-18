"""
tests/test_graph_router.py
--------------------------
Unit tests for the new GET /api/v1/graph/{repo_id} endpoint.

All Redis calls are mocked so tests run without a live Redis instance.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client() -> TestClient:
    return TestClient(app)


def _make_node_link(nodes: int = 3, edges: int = 2) -> dict:
    """Return a minimal nx.node_link_data()-style payload."""
    node_list = [
        {"id": f"file_{i}.py::file", "kind": "file", "label": f"file_{i}.py",
         "path": f"file_{i}.py", "pagerank": 0.01, "commit_count": 1,
         "is_dead_code_candidate": False}
        for i in range(nodes)
    ]
    edge_list = [
        {"source": f"file_{i}.py::file", "target": f"file_{i+1}.py::file",
         "rel": "imports", "edge_type": "EXTRACTED"}
        for i in range(edges)
    ]
    return {
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": node_list,
        "edges": edge_list,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetGraph:
    """Tests for GET /api/v1/graph/{repo_id}."""

    def test_returns_200_when_graph_exists(self, client: TestClient) -> None:
        payload = _make_node_link(nodes=3, edges=2)
        raw_json = json.dumps(payload)

        with patch(
            "app.routers.graph._redis.get",
            new=AsyncMock(return_value=raw_json),
        ):
            response = client.get("/api/v1/graph/test-repo")

        assert response.status_code == 200
        body = response.json()
        assert body["repo_id"] == "test-repo"
        assert body["cache_key"] == "graph:test-repo"
        assert len(body["graph"]["nodes"]) == 3
        assert len(body["graph"]["edges"]) == 2

    def test_graph_response_contains_required_fields(self, client: TestClient) -> None:
        payload = _make_node_link(nodes=5, edges=3)

        with patch(
            "app.routers.graph._redis.get",
            new=AsyncMock(return_value=json.dumps(payload)),
        ):
            response = client.get("/api/v1/graph/my-project")

        assert response.status_code == 200
        body = response.json()
        assert "repo_id" in body
        assert "cache_key" in body
        assert "graph" in body
        graph = body["graph"]
        assert "nodes" in graph
        assert "edges" in graph

    def test_node_attributes_are_preserved(self, client: TestClient) -> None:
        payload = _make_node_link(nodes=1, edges=0)
        first_node = payload["nodes"][0]

        with patch(
            "app.routers.graph._redis.get",
            new=AsyncMock(return_value=json.dumps(payload)),
        ):
            response = client.get("/api/v1/graph/attr-test")

        assert response.status_code == 200
        returned_node = response.json()["graph"]["nodes"][0]
        assert returned_node["kind"] == first_node["kind"]
        assert returned_node["pagerank"] == pytest.approx(first_node["pagerank"])
        assert returned_node["is_dead_code_candidate"] == first_node["is_dead_code_candidate"]

    def test_returns_404_when_graph_not_found(self, client: TestClient) -> None:
        with patch(
            "app.routers.graph._redis.get",
            new=AsyncMock(return_value=None),
        ):
            response = client.get("/api/v1/graph/nonexistent-repo-xyz")

        assert response.status_code == 404
        detail = response.json()["detail"]
        assert "nonexistent-repo-xyz" in detail
        assert "POST /api/v1/graph/parse" in detail

    def test_404_detail_mentions_repo_id(self, client: TestClient) -> None:
        with patch(
            "app.routers.graph._redis.get",
            new=AsyncMock(return_value=None),
        ):
            response = client.get("/api/v1/graph/special-id-123")

        assert "special-id-123" in response.json()["detail"]

    def test_returns_503_when_redis_is_down(self, client: TestClient) -> None:
        from redis.exceptions import ConnectionError as RedisConnectionError

        async def _fail(*_a, **_kw):
            raise RedisConnectionError("Connection refused")

        with patch("app.routers.graph._redis.get", new=_fail):
            response = client.get("/api/v1/graph/any-repo")

        assert response.status_code == 503

    def test_returns_500_on_corrupt_json(self, client: TestClient) -> None:
        with patch(
            "app.routers.graph._redis.get",
            new=AsyncMock(return_value="this is not { valid json"),
        ):
            response = client.get("/api/v1/graph/corrupt-repo")

        assert response.status_code == 500
        assert "corrupt" in response.json()["detail"].lower()

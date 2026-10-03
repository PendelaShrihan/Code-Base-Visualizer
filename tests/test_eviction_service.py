"""
tests/test_eviction_service.py
==============================
Tests for eviction of Qdrant collections/points and cached graphs older than 24 hours.
"""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from app.services.eviction_service import (
    GRAPH_TIMESTAMPS_KEY,
    QDRANT_COLLECTIONS_TIMESTAMPS_KEY,
    evict_all_stale_resources,
    evict_stale_cached_graphs,
    evict_stale_qdrant_collections,
    evict_stale_qdrant_points,
    record_graph_cache_timestamp,
    record_qdrant_collection_timestamp,
)
from worker.celery_app import celery_app
from worker.tasks import evict_stale_resources_task


class TestGraphCacheEviction:
    """Tests for Redis-cached graph eviction."""

    def test_record_graph_cache_timestamp(self):
        mock_redis = MagicMock()
        record_graph_cache_timestamp("repo-abc", timestamp=1000.0, redis_client=mock_redis)

        mock_redis.zadd.assert_called_once_with(GRAPH_TIMESTAMPS_KEY, {"repo-abc": 1000.0})
        mock_redis.set.assert_called_once()
        args, kwargs = mock_redis.set.call_args
        assert args[0] == "graph_meta:repo-abc"
        assert json.loads(args[1])["repo_id"] == "repo-abc"

    def test_evict_stale_cached_graphs(self):
        mock_redis = MagicMock()
        # Stale repo returned by zrangebyscore
        mock_redis.zrangebyscore.return_value = ["stale-repo-1", "stale-repo-2"]
        mock_redis.scan.return_value = (0, [])

        evicted = evict_stale_cached_graphs(max_age_seconds=86400, redis_client=mock_redis)

        assert evicted == 2
        # Verify deletion calls
        assert mock_redis.delete.call_count == 2
        mock_redis.delete.assert_any_call("graph:stale-repo-1", "rawgraph:stale-repo-1", "graph_meta:stale-repo-1")
        mock_redis.delete.assert_any_call("graph:stale-repo-2", "rawgraph:stale-repo-2", "graph_meta:stale-repo-2")
        mock_redis.zrem.assert_any_call(GRAPH_TIMESTAMPS_KEY, "stale-repo-1")
        mock_redis.zrem.assert_any_call(GRAPH_TIMESTAMPS_KEY, "stale-repo-2")


class TestQdrantEviction:
    """Tests for Qdrant collection and vector points eviction."""

    def test_record_qdrant_collection_timestamp(self):
        mock_redis = MagicMock()
        record_qdrant_collection_timestamp("temp_repo_col", timestamp=500.0, redis_client=mock_redis)
        mock_redis.zadd.assert_called_once_with(QDRANT_COLLECTIONS_TIMESTAMPS_KEY, {"temp_repo_col": 500.0})

    def test_evict_stale_qdrant_collections(self):
        mock_client = MagicMock()
        mock_redis = MagicMock()

        # Collections in Qdrant: code_chunks (protected), stale_temp_col, fresh_temp_col
        mock_col1 = MagicMock(name="code_chunks")
        mock_col1.name = "code_chunks"
        mock_col2 = MagicMock(name="temp_old_repo")
        mock_col2.name = "temp_old_repo"
        mock_col3 = MagicMock(name="temp_fresh_repo")
        mock_col3.name = "temp_fresh_repo"

        mock_client.get_collections.return_value = MagicMock(collections=[mock_col1, mock_col2, mock_col3])
        # Redis reports temp_old_repo as older than cutoff
        mock_redis.zrangebyscore.return_value = ["temp_old_repo"]
        mock_redis.zscore.return_value = time.time()  # fresh for others

        deleted = evict_stale_qdrant_collections(
            max_age_seconds=86400,
            qdrant_client=mock_client,
            redis_client=mock_redis,
            protected_collections=("code_chunks",),
        )

        assert deleted == ["temp_old_repo"]
        mock_client.delete_collection.assert_called_once_with(collection_name="temp_old_repo")
        mock_redis.zrem.assert_called_once_with(QDRANT_COLLECTIONS_TIMESTAMPS_KEY, "temp_old_repo")

    def test_evict_stale_qdrant_points(self):
        mock_client = MagicMock()
        mock_client.collection_exists.return_value = True

        result = evict_stale_qdrant_points(
            max_age_seconds=86400,
            qdrant_client=mock_client,
            collection_name="code_chunks",
        )

        assert result == 1
        mock_client.delete.assert_called_once()
        kwargs = mock_client.delete.call_args[1]
        assert kwargs["collection_name"] == "code_chunks"
        assert kwargs["wait"] is True


class TestMasterEvictionAndCeleryTask:
    """Tests for master resource eviction coordination and Celery task execution."""

    @patch("app.services.eviction_service.garbage_collect_temp_repos")
    @patch("app.services.eviction_service.evict_stale_qdrant_points")
    @patch("app.services.eviction_service.evict_stale_qdrant_collections")
    @patch("app.services.eviction_service.evict_stale_cached_graphs")
    def test_evict_all_stale_resources(self, mock_graphs, mock_cols, mock_pts, mock_gc):
        mock_graphs.return_value = 5
        mock_cols.return_value = ["test_col_1"]
        mock_pts.return_value = 1
        mock_gc.return_value = 2

        summary = evict_all_stale_resources(max_age_seconds=86400)

        assert summary["status"] == "success"
        assert summary["max_age_seconds"] == 86400
        assert summary["cached_graphs_evicted"] == 5
        assert summary["qdrant_collections_evicted"] == ["test_col_1"]
        assert summary["qdrant_points_evicted"] == 1
        assert summary["temp_repos_purged"] == 2

    @patch("app.services.eviction_service.evict_all_stale_resources")
    def test_evict_stale_resources_celery_task(self, mock_master):
        mock_master.return_value = {
            "status": "success",
            "cached_graphs_evicted": 3,
            "qdrant_collections_evicted": [],
            "qdrant_points_evicted": 1,
            "temp_repos_purged": 0,
        }

        res = evict_stale_resources_task(max_age_seconds=86400)
        assert res["status"] == "success"
        assert res["cached_graphs_evicted"] == 3
        mock_master.assert_called_once_with(max_age_seconds=86400)

    def test_celery_beat_schedule_registered(self):
        """Verify beat_schedule includes periodic eviction and GC."""
        schedule = celery_app.conf.beat_schedule
        assert "evict_stale_resources_periodic" in schedule
        assert schedule["evict_stale_resources_periodic"]["task"] == "worker.tasks.evict_stale_resources_task"
        assert "garbage_collect_temp_repos_periodic" in schedule

    @patch("app.services.eviction_service.evict_all_stale_resources")
    def test_maintenance_evict_api_endpoint(self, mock_master):
        from fastapi.testclient import TestClient
        from app.main import app

        mock_master.return_value = {
            "status": "success",
            "max_age_seconds": 86400,
            "cached_graphs_evicted": 2,
            "qdrant_collections_evicted": ["temp_1"],
            "qdrant_points_evicted": 1,
            "temp_repos_purged": 1,
            "timestamp": 123456789.0,
        }
        client = TestClient(app)
        resp = client.post("/api/v1/maintenance/evict?max_age_seconds=86400")
        assert resp.status_code == 200
        data = resp.json()
        assert data["cached_graphs_evicted"] == 2
        assert data["qdrant_collections_evicted"] == ["temp_1"]


"""
tests/test_bug_bash_resilience.py
---------------------------------
Comprehensive unit and integration test suite verifying the Weeks 1-6 Bug Bash fixes:
  1. Docker networking & service health resilience (Redis configurable host & graceful /health degradation).
  2. Celery task stability & progress tracking (PROGRESS/RETRY states, memory recycling, transient error retries).
  3. Embedding-pipeline hardening (null-byte stripping, missing text fallback, NaN/Inf sanitization, batch retry).
"""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from rag.embeddings import embed_chunks
from rag.ingestion import batch_upsert_chunks, extract_function_chunks
from worker.celery_app import celery_app
from worker.tasks import NON_RETRYABLE_EXCEPTIONS, _safe_update_state, process_repository_task


# ==============================================================================
# 1. Docker Networking & Service Health Resilience Tests
# ==============================================================================

class TestDockerNetworkingAndHealth:
    """Verifies service health behavior under normal and degraded conditions."""

    def test_health_check_healthy(self):
        """When Redis is operational, /health returns 200 with status='ok'."""
        with patch("app.main.redis_client.incr", new_callable=AsyncMock) as mock_incr:
            mock_incr.return_value = 42
            client = TestClient(app)
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["service"] == "codebase-visualizer"
            assert data["request_count"] == 42

    def test_health_check_degraded_when_redis_offline(self):
        """When Redis network connection fails, /health degrades gracefully instead of crashing with 500."""
        with patch("app.main.redis_client.incr", new_callable=AsyncMock) as mock_incr:
            mock_incr.side_effect = ConnectionError("Could not resolve host 'redis'")
            client = TestClient(app)
            response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "degraded"
            assert data["service"] == "codebase-visualizer"
            assert data["request_count"] == -1


# ==============================================================================
# 2. Celery Task Resilience & Status Resolution Tests
# ==============================================================================

class TestCeleryWorkerResilience:
    """Verifies Celery configuration, progress reporting, and error handling."""

    def test_celery_worker_configuration_resilience(self):
        """Worker configuration must have process recycling and time limits enabled."""
        conf = celery_app.conf
        assert conf.worker_max_tasks_per_child == 50, "Worker processes must recycle after 50 tasks"
        assert conf.worker_max_memory_per_child == 512000, "Worker memory cap must be 512MB"
        assert conf.broker_connection_retry_on_startup is True, "Broker must retry connection on startup"
        assert conf.task_time_limit == 600, "Hard task timeout must be 10 minutes"
        assert conf.task_soft_time_limit == 540, "Soft task timeout must be 9 minutes"

    def test_safe_update_state_standalone_execution(self):
        """Calling _safe_update_state outside a worker request must not raise ValueError."""
        # Standalone invocation: request has no ID
        mock_self = MagicMock()
        mock_self.request.id = None
        _safe_update_state(mock_self, "PROGRESS", {"stage": "cloning"})
        mock_self.update_state.assert_not_called()

        # None self must not raise
        _safe_update_state(None, "PROGRESS", {"stage": "cloning"})

    def test_safe_update_state_active_worker_execution(self):
        """Calling _safe_update_state with active task_id invokes update_state."""
        mock_self = MagicMock()
        mock_self.request.id = "task-uuid-123"
        _safe_update_state(mock_self, "PROGRESS", {"stage": "cloning", "percent": 10})
        mock_self.update_state.assert_called_once_with(
            state="PROGRESS",
            meta={"stage": "cloning", "percent": 10},
        )

    def test_status_endpoint_handles_progress_state(self):
        """GET /api/v1/status/{task_id} returns PROGRESS state and progress metadata."""
        with patch("app.routers.status.AsyncResult") as mock_async_result:
            instance = MagicMock()
            instance.state = "PROGRESS"
            instance.info = {"stage": "scanning", "percent": 30, "repo_url": "https://github.com/foo/bar"}
            mock_async_result.return_value = instance

            client = TestClient(app)
            response = client.get("/api/v1/status/fake-task-id")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "PROGRESS"
            assert data["result"]["stage"] == "scanning"
            assert data["result"]["percent"] == 30

    def test_status_endpoint_handles_retry_state(self):
        """GET /api/v1/status/{task_id} returns RETRY state with clear explanation."""
        with patch("app.routers.status.AsyncResult") as mock_async_result:
            instance = MagicMock()
            instance.state = "RETRY"
            instance.info = "Transient Git network error; retrying in 10s"
            mock_async_result.return_value = instance

            client = TestClient(app)
            response = client.get("/api/v1/status/retrying-task-id")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "RETRY"
            assert "Transient Git network error" in data["error"]

    def test_status_endpoint_handles_redis_connection_error(self):
        """If Celery backend is unreachable, status endpoint returns UNKNOWN instead of 500."""
        with patch("app.routers.status.AsyncResult") as mock_async_result:
            mock_async_result.side_effect = ConnectionError("Redis connection refused")

            client = TestClient(app)
            response = client.get("/api/v1/status/broken-task-id")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "UNKNOWN"
            assert "Task backend query failed" in data["error"]


# ==============================================================================
# 3. Embedding Pipeline Hardening Tests
# ==============================================================================

class TestEmbeddingPipelineHardening:
    """Verifies edge-case hardening across the embedding and vector ingestion pipeline."""

    def test_embed_chunks_missing_text_fallback(self):
        """Chunks with missing, empty, or None 'text' are assigned fallback strings and embedded."""
        chunks = [
            {"func_name": "clean_up", "file_path": "services/clean.py", "text": None},
            {"func_name": "process_all", "file_path": "worker/tasks.py", "text": ""},
            {"text": "def compute(): return 42"},
        ]
        embedded = embed_chunks(chunks)
        assert len(embedded) == 3
        for c in embedded:
            assert "embedding" in c
            assert len(c["embedding"]) == 384

    def test_embed_chunks_strips_null_bytes(self):
        """Null bytes (\\x00) are removed prior to SentenceTransformer encoding."""
        chunks = [
            {"text": "def test_func():\x00 return 'ok\x00'"}
        ]
        embedded = embed_chunks(chunks)
        assert len(embedded) == 1
        assert "embedding" in embedded[0]
        assert len(embedded[0]["embedding"]) == 384

    def test_extract_function_chunks_sanitizes_nan_and_inf(self):
        """PageRank or created_at with NaN or Inf are sanitized to 0.0 or current time."""
        graph_dict = {
            "nodes": [
                {
                    "id": "file.py::func::bad_metrics",
                    "kind": "function",
                    "name": "bad_metrics",
                    "file": "file.py",
                    "code": "def bad_metrics(): pass",
                    "pagerank": float("nan"),
                    "created_at": float("inf"),
                    "commit_count": "invalid",
                }
            ],
            "edges": [],
        }
        chunks = extract_function_chunks(graph_dict)
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk["pagerank"] == 0.0
        assert not math.isnan(chunk["pagerank"])
        assert not math.isinf(chunk["created_at"])
        assert chunk["commit_count"] == 0

    def test_extract_function_chunks_empty_node_id_collision_guard(self):
        """Nodes with missing 'id' generate non-colliding UUIDs."""
        graph_dict = {
            "nodes": [
                {
                    "kind": "function",
                    "name": "func_a",
                    "file": "a.py",
                    "code": "def func_a(): pass",
                },
                {
                    "kind": "function",
                    "name": "func_b",
                    "file": "b.py",
                    "code": "def func_b(): pass",
                },
            ],
            "edges": [],
        }
        chunks = extract_function_chunks(graph_dict)
        assert len(chunks) == 2
        assert chunks[0]["chunk_id"] != chunks[1]["chunk_id"]

    def test_batch_upsert_chunks_retry_on_transient_failure(self):
        """batch_upsert_chunks retries when Qdrant client raises a transient exception."""
        mock_client = MagicMock()
        # Fail once, then succeed on retry
        mock_client.upsert.side_effect = [
            ConnectionResetError("Connection reset by peer"),
            None,
        ]

        chunks = [
            {
                "chunk_id": "test-uuid-1",
                "text": "def hello(): pass",
                "func_name": "hello",
                "file_path": "hello.py",
                "code": "def hello(): pass",
                "pagerank": 0.1,
                "commit_count": 1,
                "is_dead_code_candidate": False,
                "created_at": 1000.0,
            }
        ]

        with patch("time.sleep") as mock_sleep:
            upserted, batches = batch_upsert_chunks(chunks, client=mock_client, batch_size=64)

        assert upserted == 1
        assert batches == 1
        assert mock_client.upsert.call_count == 2
        mock_sleep.assert_called_once()

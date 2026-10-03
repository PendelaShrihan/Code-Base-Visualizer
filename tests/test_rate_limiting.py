"""
tests/test_rate_limiting.py
===========================
Tests for per-IP sliding window rate limiting on POST /api/v1/analyze-repo.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.rate_limiter import SlidingWindowRateLimiter, get_client_ip, rate_limiter


@pytest.fixture(autouse=True)
def _reset_limiter():
    """Reset rate limiter state before each test."""
    rate_limiter._memory_store.clear()
    rate_limiter.max_requests = 5
    rate_limiter.window_seconds = 3600
    yield
    rate_limiter._memory_store.clear()


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


class TestClientIpExtraction:
    """Test resolution of client IP from headers."""

    def test_x_forwarded_for_first_ip(self):
        mock_req = MagicMock()
        mock_req.headers = {"X-Forwarded-For": "203.0.113.195, 70.41.3.18, 150.172.238.178"}
        assert get_client_ip(mock_req) == "203.0.113.195"

    def test_x_real_ip(self):
        mock_req = MagicMock()
        mock_req.headers = {"X-Real-IP": "198.51.100.24"}
        assert get_client_ip(mock_req) == "198.51.100.24"

    def test_client_host_fallback(self):
        mock_req = MagicMock()
        mock_req.headers = {}
        mock_req.client.host = "192.168.1.10"
        assert get_client_ip(mock_req) == "192.168.1.10"

    def test_unknown_fallback(self):
        mock_req = MagicMock()
        mock_req.headers = {}
        mock_req.client = None
        assert get_client_ip(mock_req) == "127.0.0.1"


class TestAnalyzeRepoRateLimiting:
    """Test rate limiting enforcement on POST /api/v1/analyze-repo."""

    @patch("app.routers.analyze.process_repository_task.delay")
    def test_allows_requests_within_limit(self, mock_celery, client: TestClient):
        mock_celery.return_value = MagicMock(id="test-task-uuid")

        rate_limiter.max_requests = 3
        rate_limiter.window_seconds = 60

        headers = {"X-Forwarded-For": "10.0.0.1"}
        payload = {"repo_url": "https://github.com/fastapi/fastapi"}

        # First 3 requests should be accepted (202)
        for i in range(3):
            resp = client.post("/api/v1/analyze-repo", json=payload, headers=headers)
            assert resp.status_code == 202, f"Request {i+1} failed: {resp.text}"
            assert resp.headers.get("X-RateLimit-Limit") == "3"
            assert resp.headers.get("X-RateLimit-Remaining") == str(3 - i - 1)
            assert "X-RateLimit-Reset" in resp.headers

        assert mock_celery.call_count == 3

    @patch("app.routers.analyze.process_repository_task.delay")
    def test_blocks_and_returns_429_when_exceeded(self, mock_celery, client: TestClient):
        mock_celery.return_value = MagicMock(id="test-task-uuid")

        rate_limiter.max_requests = 2
        rate_limiter.window_seconds = 300

        headers = {"X-Forwarded-For": "10.0.0.2"}
        payload = {"repo_url": "https://github.com/fastapi/fastapi"}

        # Request 1 & 2 succeed
        resp1 = client.post("/api/v1/analyze-repo", json=payload, headers=headers)
        assert resp1.status_code == 202

        resp2 = client.post("/api/v1/analyze-repo", json=payload, headers=headers)
        assert resp2.status_code == 202

        # Request 3 must be blocked with 429
        resp3 = client.post("/api/v1/analyze-repo", json=payload, headers=headers)
        assert resp3.status_code == 429
        assert "Rate limit exceeded" in resp3.json()["detail"]
        assert resp3.headers.get("Retry-After") is not None
        assert int(resp3.headers.get("Retry-After")) > 0
        assert resp3.headers.get("X-RateLimit-Remaining") == "0"

        # Ensure Celery delay was NOT called for the blocked request
        assert mock_celery.call_count == 2

    @patch("app.routers.analyze.process_repository_task.delay")
    def test_isolates_rate_limits_per_ip(self, mock_celery, client: TestClient):
        mock_celery.return_value = MagicMock(id="test-task-uuid")

        rate_limiter.max_requests = 1
        rate_limiter.window_seconds = 60

        payload = {"repo_url": "https://github.com/fastapi/fastapi"}

        # Client A makes 1 request
        resp_a1 = client.post(
            "/api/v1/analyze-repo", json=payload, headers={"X-Forwarded-For": "192.168.1.50"}
        )
        assert resp_a1.status_code == 202

        # Client A second request gets 429
        resp_a2 = client.post(
            "/api/v1/analyze-repo", json=payload, headers={"X-Forwarded-For": "192.168.1.50"}
        )
        assert resp_a2.status_code == 429

        # Client B makes 1 request and should SUCCEED
        resp_b1 = client.post(
            "/api/v1/analyze-repo", json=payload, headers={"X-Forwarded-For": "192.168.1.99"}
        )
        assert resp_b1.status_code == 202

    @patch("app.routers.analyze.process_repository_task.delay")
    def test_sliding_window_resets_after_window_elapses(self, mock_celery, client: TestClient):
        mock_celery.return_value = MagicMock(id="test-task-uuid")

        rate_limiter.max_requests = 1
        rate_limiter.window_seconds = 10

        headers = {"X-Forwarded-For": "10.0.0.5"}
        payload = {"repo_url": "https://github.com/fastapi/fastapi"}

        # Request at t=0
        resp1 = client.post("/api/v1/analyze-repo", json=payload, headers=headers)
        assert resp1.status_code == 202

        # Immediate follow-up gets 429
        resp2 = client.post("/api/v1/analyze-repo", json=payload, headers=headers)
        assert resp2.status_code == 429

        # Simulate time passage beyond window by backdating timestamps in memory
        key = "ratelimit:analyze_repo:10.0.0.5"
        rate_limiter._memory_store[key] = [time.time() - 20]

        # Next request should succeed
        resp3 = client.post("/api/v1/analyze-repo", json=payload, headers=headers)
        assert resp3.status_code == 202

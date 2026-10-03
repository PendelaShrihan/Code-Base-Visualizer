"""
app/services/rate_limiter.py
============================
Per-IP sliding window rate limiting for public-facing API protection.

Features:
- Accurate sliding window counter algorithm (Redis Sorted Set).
- Client IP resolution with reverse-proxy support (X-Forwarded-For, X-Real-IP).
- Standard rate limit response headers (Retry-After, X-RateLimit-Limit,
  X-RateLimit-Remaining, X-RateLimit-Reset).
- Automatic in-memory fallback if Redis is unavailable or unconfigured,
  ensuring resilient tests and standalone execution without downtime.
- Configurable thresholds via environment variables:
  * RATE_LIMIT_ANALYZE_REPO_MAX_REQUESTS (default: 5)
  * RATE_LIMIT_ANALYZE_REPO_WINDOW_SECONDS (default: 3600)
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
import uuid
from typing import Any, Optional

import redis.asyncio as aioredis
from fastapi import HTTPException, Request, Response, status

from app.exceptions import RateLimitExceededError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
DEFAULT_MAX_REQUESTS: int = int(os.getenv("RATE_LIMIT_ANALYZE_REPO_MAX_REQUESTS", "5"))
DEFAULT_WINDOW_SECONDS: int = int(os.getenv("RATE_LIMIT_ANALYZE_REPO_WINDOW_SECONDS", "3600"))
_REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")


def get_client_ip(request: Request) -> str:
    """
    Extract the real client IP address from request headers or socket.

    Evaluates:
      1. X-Forwarded-For (first entry in comma-separated chain)
      2. X-Real-IP
      3. Client host socket
      4. Default '127.0.0.1' fallback
    """
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        # First IP is the original client
        client_ip = forwarded_for.split(",")[0].strip()
        if client_ip:
            return client_ip

    real_ip = request.headers.get("X-Real-IP")
    if real_ip and real_ip.strip():
        return real_ip.strip()

    if request.client and request.client.host:
        return request.client.host

    return "127.0.0.1"


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter backed by Redis Sorted Sets with in-memory fallback.
    """

    def __init__(
        self,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        window_seconds: int = DEFAULT_WINDOW_SECONDS,
        redis_client: Optional[aioredis.Redis] = None,
        key_prefix: str = "analyze_repo",
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix
        self._redis = redis_client
        self._redis_disabled_until: float = 0.0
        # In-memory fallback: {key: list of timestamp floats}
        self._memory_store: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def get_redis_client(self) -> aioredis.Redis:
        """Lazily initialize or return the redis client."""
        if self._redis is None:
            self._redis = aioredis.Redis(
                host=_REDIS_HOST,
                port=6379,
                decode_responses=True,
                socket_connect_timeout=0.25,
                socket_timeout=0.25,
            )
        return self._redis

    async def check_rate_limit(
        self,
        request: Request,
        response: Optional[Response] = None,
        max_requests: Optional[int] = None,
        window_seconds: Optional[int] = None,
    ) -> None:
        """
        Enforce rate limit for the client IP.

        Raises:
            HTTPException with status code 429 when rate limit is exceeded.
        """
        limit = max_requests if max_requests is not None else self.max_requests
        window = window_seconds if window_seconds is not None else self.window_seconds
        client_ip = get_client_ip(request)
        redis_key = f"ratelimit:{self.key_prefix}:{client_ip}"
        now = time.time()
        cutoff = now - window

        # Attempt Redis sliding window check
        used_redis = False
        count = 0
        earliest_ts: Optional[float] = None

        if now >= self._redis_disabled_until:
            try:
                r = self.get_redis_client()
                pipe = r.pipeline(transaction=True)
                # Remove timestamps outside current window
                pipe.zremrangebyscore(redis_key, 0, cutoff)
                # Get current count
                pipe.zcard(redis_key)
                # Get oldest timestamp in current window
                pipe.zrange(redis_key, 0, 0, withscores=True)
                results = await asyncio.wait_for(pipe.execute(), timeout=0.25)

                count = int(results[1])
                earliest_items = results[2]
                if earliest_items:
                    earliest_ts = float(earliest_items[0][1])

                used_redis = True
            except Exception as exc:
                self._redis_disabled_until = time.time() + 60.0
                logger.debug(
                    "Redis rate-limit check failed or unavailable (%s). Tripping circuit breaker for 60s, using in-memory fallback.",
                    exc,
                )
                used_redis = False

        if not used_redis:
            # Thread-safe in-memory sliding window fallback
            with self._lock:
                timestamps = self._memory_store.get(redis_key, [])
                timestamps = [ts for ts in timestamps if ts > cutoff]
                count = len(timestamps)
                earliest_ts = timestamps[0] if timestamps else None
                self._memory_store[redis_key] = timestamps

        # Evaluate limit
        if count >= limit:
            reset_ts = (earliest_ts + window) if earliest_ts else (now + window)
            retry_after = max(1, math.ceil(reset_ts - now))
            headers = {
                "Retry-After": str(retry_after),
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "X-RateLimit-Reset": str(int(reset_ts)),
            }
            logger.warning(
                "Rate limit exceeded for IP=%s on %s: %d/%d reqs. Retry after %ds.",
                client_ip,
                self.key_prefix,
                count,
                limit,
                retry_after,
            )
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=(
                    f"Rate limit exceeded: maximum {limit} requests per "
                    f"{window} seconds. Please try again in {retry_after} seconds."
                ),
                headers=headers,
            )

        # Within limit: record request timestamp
        member_id = f"{now}:{uuid.uuid4().hex[:6]}"
        if used_redis:
            try:
                r = self.get_redis_client()
                pipe = r.pipeline(transaction=True)
                pipe.zadd(redis_key, {member_id: now})
                pipe.expire(redis_key, window)
                await asyncio.wait_for(pipe.execute(), timeout=0.25)
            except Exception as exc:
                self._redis_disabled_until = time.time() + 60.0
                logger.debug("Failed to record rate limit hit in Redis: %s", exc)
                used_redis = False

        if not used_redis:
            with self._lock:
                self._memory_store.setdefault(redis_key, []).append(now)


        remaining = max(0, limit - count - 1)
        reset_time = int(now + window)

        if response is not None:
            response.headers["X-RateLimit-Limit"] = str(limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(reset_time)

    async def reset(self, client_ip: Optional[str] = None) -> None:
        """Reset rate limit data for a specific IP or all tracked IPs."""
        with self._lock:
            if client_ip:
                key = f"ratelimit:{self.key_prefix}:{client_ip}"
                self._memory_store.pop(key, None)
            else:
                self._memory_store.clear()

        try:
            r = self.get_redis_client()
            if client_ip:
                await r.delete(f"ratelimit:{self.key_prefix}:{client_ip}")
            else:
                cursor = 0
                while True:
                    cursor, keys = await r.scan(
                        cursor=cursor, match=f"ratelimit:{self.key_prefix}:*", count=100
                    )
                    if keys:
                        await r.delete(*keys)
                    if cursor == 0:
                        break
        except Exception:
            pass


# Global singleton instance for analyze-repo endpoint
rate_limiter = SlidingWindowRateLimiter(
    max_requests=DEFAULT_MAX_REQUESTS,
    window_seconds=DEFAULT_WINDOW_SECONDS,
    key_prefix="analyze_repo",
)


async def rate_limit_analyze_repo(
    request: Request,
    response: Response,
) -> None:
    """FastAPI dependency for per-IP rate limiting on POST /api/v1/analyze-repo."""
    await rate_limiter.check_rate_limit(request=request, response=response)

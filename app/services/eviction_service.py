"""
app/services/eviction_service.py
================================
Abuse prevention and automated resource eviction service.

Periodically evicts:
  1. Redis-cached dependency graphs (graph:{repo_id}, rawgraph:{repo_id}) older than 24h.
  2. Qdrant vector collections and stale vector points older than 24h.
  3. Ephemeral cloned git directories in /tmp/repos/ older than threshold.

Protects the public demo against resource exhaustion (disk saturation,
Redis memory limits, and vector DB bloat).
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Optional, Sequence

from app.services.git_service import garbage_collect_temp_repos

logger = logging.getLogger(__name__)

# Default age limit: 24 hours (86,400 seconds)
DEFAULT_MAX_AGE_SECONDS: int = int(os.getenv("RESOURCE_MAX_AGE_SECONDS", "86400"))
_REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")

# Redis sorted set tracking keys
GRAPH_TIMESTAMPS_KEY: str = "cached_graphs:timestamps"
QDRANT_COLLECTIONS_TIMESTAMPS_KEY: str = "qdrant_collections:timestamps"


# ---------------------------------------------------------------------------
# Redis Client Helpers (Sync & Async)
# ---------------------------------------------------------------------------


_redis_disabled_until: float = 0.0


def get_sync_redis_client(redis_client: Optional[Any] = None) -> Any:
    """Return a synchronous redis client with fast circuit breaker on failure."""
    global _redis_disabled_until
    if redis_client is not None:
        return redis_client
    if time.time() < _redis_disabled_until:
        return None
    try:
        import socket
        import redis

        try:
            sock = socket.create_connection((_REDIS_HOST, 6379), timeout=0.1)
            sock.close()
        except Exception:
            _redis_disabled_until = time.time() + 300.0
            return None

        r = redis.Redis(
            host=_REDIS_HOST,
            port=6379,
            decode_responses=True,
            socket_connect_timeout=0.25,
            socket_timeout=0.25,
        )
        return r
    except Exception as exc:
        _redis_disabled_until = time.time() + 300.0
        logger.debug("Sync Redis unreachable (%s). Disabling sync Redis for 300s.", exc)
        return None




# ---------------------------------------------------------------------------
# Metadata Tracking Helpers
# ---------------------------------------------------------------------------


def record_graph_cache_timestamp(
    repo_id: str,
    timestamp: Optional[float] = None,
    redis_client: Optional[Any] = None,
) -> None:
    """
    Record when a graph was cached in Redis for future eviction tracking.
    """
    ts = timestamp if timestamp is not None else time.time()
    r = get_sync_redis_client(redis_client)
    if r is not None:
        try:
            r.zadd(GRAPH_TIMESTAMPS_KEY, {repo_id: ts})
            # Also store explicit metadata with 48h safety TTL
            meta_key = f"graph_meta:{repo_id}"
            r.set(
                meta_key,
                json.dumps({"repo_id": repo_id, "cached_at": ts}),
                ex=int(DEFAULT_MAX_AGE_SECONDS * 2),
            )
            logger.debug("Recorded cache timestamp for repo_id=%s: %s", repo_id, ts)
        except Exception as exc:
            logger.debug("Failed to record graph timestamp in Redis: %s", exc)


def record_qdrant_collection_timestamp(
    collection_name: str,
    timestamp: Optional[float] = None,
    redis_client: Optional[Any] = None,
) -> None:
    """
    Record when a Qdrant collection was created.
    """
    ts = timestamp if timestamp is not None else time.time()
    r = get_sync_redis_client(redis_client)
    if r is not None:
        try:
            r.zadd(QDRANT_COLLECTIONS_TIMESTAMPS_KEY, {collection_name: ts})
            logger.debug(
                "Recorded collection timestamp for %s: %s", collection_name, ts
            )
        except Exception as exc:
            logger.debug("Failed to record collection timestamp in Redis: %s", exc)


# ---------------------------------------------------------------------------
# Eviction Core Logic
# ---------------------------------------------------------------------------


def evict_stale_cached_graphs(
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    redis_client: Optional[Any] = None,
) -> int:
    """
    Evict Redis-cached graphs older than max_age_seconds.

    Removes keys:
      - graph:{repo_id}
      - rawgraph:{repo_id}
      - graph_meta:{repo_id}

    Returns:
        Number of evicted graph repositories.
    """
    r = get_sync_redis_client(redis_client)
    if r is None:
        logger.warning("Redis client unavailable; skipping graph cache eviction.")
        return 0

    now = time.time()
    cutoff = now - max_age_seconds
    evicted_count = 0

    try:
        # 1. Fetch repo_ids older than cutoff from the sorted set
        stale_repo_ids = r.zrangebyscore(GRAPH_TIMESTAMPS_KEY, 0, cutoff)
        for repo_id in stale_repo_ids:
            keys_to_del = [
                f"graph:{repo_id}",
                f"rawgraph:{repo_id}",
                f"graph_meta:{repo_id}",
            ]
            r.delete(*keys_to_del)
            r.zrem(GRAPH_TIMESTAMPS_KEY, repo_id)
            evicted_count += 1
            logger.info("Evicted stale cached graph for repo_id=%s", repo_id)

        # 2. Check for orphan keys without metadata or with expired TTL
        cursor = 0
        while True:
            cursor, keys = r.scan(cursor=cursor, match="graph:*", count=100)
            for k in keys:
                # Skip sub-namespaces
                if k.startswith("graph_meta:"):
                    continue
                ttl = r.ttl(k)
                if ttl == -1:
                    # Key exists without TTL; assign default TTL or check metadata
                    r.expire(k, max_age_seconds)
            if cursor == 0:
                break

    except Exception as exc:
        logger.error("Error during stale cached graph eviction: %s", exc)

    logger.info("evict_stale_cached_graphs finished: %d graph(s) evicted", evicted_count)
    return evicted_count


def evict_stale_qdrant_collections(
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    qdrant_client: Optional[Any] = None,
    redis_client: Optional[Any] = None,
    protected_collections: Sequence[str] = ("code_chunks",),
) -> list[str]:
    """
    Evict temporary or ephemeral Qdrant collections older than max_age_seconds.

    Protected collections (like the shared 'code_chunks' collection) are not deleted
    as a whole; their stale points are pruned via evict_stale_qdrant_points().

    Returns:
        List of deleted collection names.
    """
    from rag.qdrant_client import get_qdrant_client

    client = qdrant_client or get_qdrant_client()
    r = get_sync_redis_client(redis_client)
    now = time.time()
    cutoff = now - max_age_seconds
    deleted_collections: list[str] = []

    try:
        collections_resp = client.get_collections()
        existing_names = [c.name for c in collections_resp.collections]

        # Check recorded collection creation timestamps
        tracked_stale: list[str] = []
        if r is not None:
            tracked_stale = r.zrangebyscore(
                QDRANT_COLLECTIONS_TIMESTAMPS_KEY, 0, cutoff
            )

        for col_name in existing_names:
            if col_name in protected_collections:
                continue

            # Criteria to delete collection:
            # 1. Specifically tracked as older than cutoff, OR
            # 2. Ephemeral name prefix (test_, temp_, repo_) without recent tracking
            is_stale = col_name in tracked_stale
            if not is_stale and any(col_name.startswith(p) for p in ("test_", "temp_", "repo_")):
                # Check if it has any recorded timestamp
                score = r.zscore(QDRANT_COLLECTIONS_TIMESTAMPS_KEY, col_name) if r else None
                if score is not None and score <= cutoff:
                    is_stale = True

            if is_stale:
                logger.info("Evicting stale Qdrant collection '%s'", col_name)
                try:
                    client.delete_collection(collection_name=col_name)
                    deleted_collections.append(col_name)
                    if r is not None:
                        r.zrem(QDRANT_COLLECTIONS_TIMESTAMPS_KEY, col_name)
                except Exception as del_err:
                    logger.warning("Failed to delete collection '%s': %s", col_name, del_err)

    except Exception as exc:
        logger.error("Error during Qdrant collection eviction: %s", exc)

    logger.info(
        "evict_stale_qdrant_collections finished: %d collection(s) evicted (%s)",
        len(deleted_collections),
        deleted_collections,
    )
    return deleted_collections


def evict_stale_qdrant_points(
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    qdrant_client: Optional[Any] = None,
    collection_name: str = "code_chunks",
) -> int:
    """
    Evict stale vector points older than max_age_seconds from a collection.

    Uses Qdrant's FilterSelector with range filter on the 'created_at' payload field.

    Returns:
        Number of points evicted (or 1 if deletion filter submitted successfully).
    """
    from rag.qdrant_client import get_qdrant_client

    client = qdrant_client or get_qdrant_client()
    now = time.time()
    cutoff = now - max_age_seconds

    try:
        if not client.collection_exists(collection_name):
            logger.debug("Collection '%s' does not exist; skipping point eviction.", collection_name)
            return 0

        from qdrant_client import models

        # Query points older than cutoff using payload filter
        stale_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="created_at",
                    range=models.Range(lt=cutoff),
                )
            ]
        )

        # Delete points matching the filter
        res = client.delete(
            collection_name=collection_name,
            points_selector=models.FilterSelector(filter=stale_filter),
            wait=True,
        )
        logger.info(
            "Submitted stale points eviction for collection '%s' (cutoff=%s): result=%s",
            collection_name,
            cutoff,
            res,
        )
        return 1
    except Exception as exc:
        logger.warning(
            "Point eviction on collection '%s' failed or not supported in current mode: %s",
            collection_name,
            exc,
        )
        return 0


def evict_all_stale_resources(
    max_age_seconds: int = DEFAULT_MAX_AGE_SECONDS,
    redis_client: Optional[Any] = None,
    qdrant_client: Optional[Any] = None,
) -> dict[str, Any]:
    """
    Master eviction procedure: purges stale cached graphs, Qdrant collections/points,
    and temporary git clone directories.

    Returns:
        Comprehensive execution summary.
    """
    logger.info("Starting master resource eviction sweep (max_age_seconds=%d)", max_age_seconds)

    # 1. Evict Redis cached graphs
    graphs_evicted = evict_stale_cached_graphs(
        max_age_seconds=max_age_seconds,
        redis_client=redis_client,
    )

    # 2. Evict ephemeral Qdrant collections
    collections_evicted = evict_stale_qdrant_collections(
        max_age_seconds=max_age_seconds,
        qdrant_client=qdrant_client,
        redis_client=redis_client,
    )

    # 3. Evict stale vector points from the shared collection
    points_evicted = evict_stale_qdrant_points(
        max_age_seconds=max_age_seconds,
        qdrant_client=qdrant_client,
    )

    # 4. Garbage collect stale clone directories in /tmp/repos/
    temp_repos_purged = 0
    try:
        temp_repos_purged = garbage_collect_temp_repos(max_age_seconds=max_age_seconds)
    except Exception as gc_err:
        logger.warning("Temp repo GC encountered error: %s", gc_err)

    summary = {
        "status": "success",
        "max_age_seconds": max_age_seconds,
        "cached_graphs_evicted": graphs_evicted,
        "qdrant_collections_evicted": collections_evicted,
        "qdrant_points_evicted": points_evicted,
        "temp_repos_purged": temp_repos_purged,
        "timestamp": time.time(),
    }
    logger.info("Master resource eviction completed: %s", summary)
    return summary

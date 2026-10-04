"""
worker/tasks.py
---------------
Celery task definitions for background repository processing in CodeBase Visualizer.
"""

from __future__ import annotations

import logging
import logging
import os
from pathlib import Path
from typing import Any

from celery.exceptions import SoftTimeLimitExceeded
from git.exc import GitCommandError
import redis.exceptions

from app.exceptions import (
    CloneTimeoutError,
    FileParseTimeoutError,
    IngestionGuardrailError,
    MaxFileCountExceededError,
    RepoSizeLimitExceededError,
)
from app.services.git_service import (
    clone_repository,
    cleanup_repo_directory,
    garbage_collect_temp_repos,
)
from parser.repo_walker import (
    attach_churn,
    detect_dead_code,
    graph_to_json,
    scan_repository,
)
from rag.ingestion import ingest_graph
from worker.celery_app import celery_app

logger = logging.getLogger(__name__)

# Non-retryable guardrail and client exceptions (permanent failures)
NON_RETRYABLE_EXCEPTIONS = (
    IngestionGuardrailError,
    CloneTimeoutError,
    RepoSizeLimitExceededError,
    MaxFileCountExceededError,
    FileParseTimeoutError,
    ValueError,
)


def _safe_update_state(task_self: Any, state: str, meta: dict[str, Any]) -> None:
    """Safely update task state only when running in a live Celery worker request with task_id."""
    if (
        task_self is not None
        and getattr(task_self, "request", None) is not None
        and getattr(task_self.request, "id", None)
    ):
        try:
            task_self.update_state(state=state, meta=meta)
        except Exception as exc:
            logger.debug("Failed updating Celery task state: %s", exc)


@celery_app.task(
    name="worker.tasks.process_repository_task",
    bind=True,
    max_retries=3,
    default_retry_delay=5,
)
def process_repository_task(self, repo_url: str) -> dict[str, Any]:
    """
    Celery task that orchestrates the end-to-end repository parsing pipeline:
      1. Clones the remote git repository to a temporary directory (full clone).
      2. Recursively scans Python files and extracts the AST / dependency graph.
      3. Computes PageRank centrality and Git commit churn metrics on nodes.
      4. Detects candidate dead code (in-degree 0 in call graph, excluding entry points).
      5. Serializes the NetworkX graph into a JSON-serializable dict.
      6. Batch-embeds all function nodes and upserts vectors to Qdrant.
      7. Cleans up the temporary clone directory and triggers garbage collection.
      8. Returns the graph dictionary with ingestion summary.

    Args:
        repo_url: Public Git / GitHub repository URL (e.g. 'https://github.com/owner/repo').

    Returns:
        Dictionary containing 'meta' (with dead_code_candidates), 'nodes', 'edges',
        and 'ingestion' (with chunk counts and batch stats).
    """
    attempt_num = (getattr(self.request, "retries", 0) + 1) if (getattr(self, "request", None) and getattr(self.request, "retries", None) is not None) else 1
    logger.info("Received process_repository_task (attempt %d) for repo_url: %s", attempt_num, repo_url)

    clone_path: Path | None = None
    try:
        # Step 1: Clone repository to temp directory
        _safe_update_state(self, state="PROGRESS", meta={"stage": "cloning", "repo_url": repo_url, "percent": 10})
        logger.info("Cloning repository: %s", repo_url)
        clone_path = clone_repository(repo_url)
        logger.info("Repository cloned to: %s", clone_path)

        # Step 2: Scan repository and build graph (includes PageRank)
        _safe_update_state(self, state="PROGRESS", meta={"stage": "scanning", "repo_url": repo_url, "percent": 30})
        logger.info("Scanning repository structure at: %s", clone_path)
        graph = scan_repository(clone_path)

        # Step 3: Git Churn Hotspots — count commits touching each file
        _safe_update_state(self, state="PROGRESS", meta={"stage": "metrics", "repo_url": repo_url, "percent": 50})
        attach_churn(graph, clone_path)

        # Step 4: Dead Code Detection on call graph
        dead_candidates = detect_dead_code(graph)
        logger.info(
            "Dead code analysis identified %d candidate(s) in %s",
            len(dead_candidates),
            repo_url,
        )

        logger.info(
            "Scan completed successfully: %d nodes, %d edges, %d dead code candidate(s).",
            graph.number_of_nodes(),
            graph.number_of_edges(),
            len(dead_candidates),
        )

        # Step 5: Serialize graph to JSON-friendly dictionary
        graph_dict = graph_to_json(graph)
        logger.info("Serialized graph dictionary for %s", repo_url)

        # Derive clean repo_id from the Git repository URL (e.g. "psf/requests" -> "psf-requests")
        derived_repo_id: str | None = None
        try:
            parts = [p for p in repo_url.rstrip("/").removesuffix(".git").split("/") if p]
            if len(parts) >= 2:
                derived_repo_id = f"{parts[-2]}-{parts[-1]}"
        except Exception:
            derived_repo_id = None

        if derived_repo_id:
            graph_dict["repo_id"] = derived_repo_id
            if "meta" in graph_dict and isinstance(graph_dict["meta"], dict):
                graph_dict["meta"]["repo_id"] = derived_repo_id

        # Cache in Redis so graph and RAG queries can immediately resolve it
        if derived_repo_id:
            try:
                import json
                import os
                from app.services.eviction_service import get_sync_redis_client, record_graph_cache_timestamp

                _safe_update_state(self, state="PROGRESS", meta={"stage": "caching", "repo_url": repo_url, "percent": 70})
                r = get_sync_redis_client()
                if r is not None:
                    ttl_seconds = int(os.getenv("GRAPH_CACHE_TTL_SECONDS", "86400"))
                    graph_json_str = json.dumps(graph_dict)
                    r.set(f"graph:{derived_repo_id}", graph_json_str, ex=ttl_seconds)
                    r.set(f"rawgraph:{derived_repo_id}", graph_json_str, ex=ttl_seconds)
                    record_graph_cache_timestamp(derived_repo_id, redis_client=r)
                    logger.info("Cached graph in Redis under keys graph:%s and rawgraph:%s (ttl=%ds)", derived_repo_id, derived_repo_id, ttl_seconds)
            except Exception as redis_exc:
                logger.warning("Failed to cache graph in Redis for %s: %s", derived_repo_id, redis_exc)


        # Step 6: Batch Vector Ingestion — embed all function nodes and upsert to Qdrant
        # ingest_graph() is idempotent: re-running on the same repo updates existing
        # Qdrant points (UUID v5 IDs are deterministic from the node ID string).
        try:
            _safe_update_state(self, state="PROGRESS", meta={"stage": "vector_ingestion", "repo_url": repo_url, "percent": 85})
            ingestion_summary = ingest_graph(graph_dict, repo_id=derived_repo_id)
            logger.info(
                "Vector ingestion complete for '%s' (repo_id=%s): %s",
                repo_url,
                derived_repo_id,
                ingestion_summary,
            )
            graph_dict["ingestion"] = ingestion_summary
        except Exception as ingest_exc:
            # Ingestion failure is non-fatal: the graph is still returned so the
            # caller can render the visualisation even if search indexing fails.
            logger.error(
                "Vector ingestion failed for '%s' (non-fatal): %s",
                repo_url,
                ingest_exc,
            )
            graph_dict["ingestion"] = {
                "status": "error",
                "error": str(ingest_exc),
                "chunks_upserted": 0,
            }

        _safe_update_state(self, state="PROGRESS", meta={"stage": "completed", "repo_url": repo_url, "percent": 100})
        return graph_dict

    except NON_RETRYABLE_EXCEPTIONS as guard_exc:
        logger.error("Non-retryable / guardrail violation for repository '%s': %s", repo_url, guard_exc)
        raise
    except SoftTimeLimitExceeded as time_exc:
        logger.error("Celery task execution exceeded soft time limit for repository '%s': %s", repo_url, time_exc)
        raise
    except Exception as exc:
        has_id = (
            self is not None
            and getattr(self, "request", None) is not None
            and getattr(self.request, "id", None) is not None
        )
        current_retries = getattr(self.request, "retries", 0) if has_id else 0
        if has_id and current_retries < self.max_retries:
            backoff_sec = min(60, 2 ** current_retries * 5)
            logger.warning(
                "Transient error processing '%s' (attempt %d/%d). Retrying in %ds... Error: %s",
                repo_url,
                current_retries + 1,
                self.max_retries,
                backoff_sec,
                exc,
            )
            raise self.retry(exc=exc, countdown=backoff_sec)
        logger.exception("Failed to process repository '%s': %s", repo_url, exc)
        raise

    finally:
        # Base Task: Automated Resource Cleanup — delete temporary cloned repo
        if clone_path is not None:
            cleanup_success = cleanup_repo_directory(clone_path)
            if cleanup_success:
                logger.info("Successfully removed temporary repository clone at %s", clone_path)
            else:
                logger.warning("Failed to cleanly remove directory %s", clone_path)

        # Garbage collect any orphaned/stale repositories in /tmp/repos/
        try:
            purged = garbage_collect_temp_repos(max_age_seconds=1800)
            if purged > 0:
                logger.info("Worker GC purged %d stale repository directories", purged)
        except Exception as gc_err:
            logger.warning("Garbage collection sweep encountered an error: %s", gc_err)


@celery_app.task(name="worker.tasks.garbage_collect_task")
def garbage_collect_task(max_age_seconds: int = 1800) -> int:
    """
    Celery task to run standalone or periodic garbage collection on /tmp/repos/.

    Args:
        max_age_seconds: Maximum directory age threshold before purging.

    Returns:
        Number of purged directories.
    """
    logger.info("Running garbage_collect_task (max_age_seconds=%d)", max_age_seconds)
    purged = garbage_collect_temp_repos(max_age_seconds=max_age_seconds)
    logger.info("garbage_collect_task finished: %d directories purged", purged)
    return purged


@celery_app.task(name="worker.tasks.evict_stale_resources_task")
def evict_stale_resources_task(max_age_seconds: int = 86400) -> dict[str, Any]:
    """
    Celery task to evict Qdrant collections/points and cached graphs older than 24 hours.

    Args:
        max_age_seconds: Threshold age in seconds (default: 86,400 = 24 hours).

    Returns:
        Summary dict of evicted resources.
    """
    from app.services.eviction_service import evict_all_stale_resources

    logger.info("Executing scheduled evict_stale_resources_task (max_age_seconds=%d)", max_age_seconds)
    summary = evict_all_stale_resources(max_age_seconds=max_age_seconds)
    logger.info("evict_stale_resources_task completed successfully: %s", summary)
    return summary


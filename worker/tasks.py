"""
worker/tasks.py
---------------
Celery task definitions for background repository processing in CodeBase Visualizer.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

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
from worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="worker.tasks.process_repository_task")
def process_repository_task(repo_url: str) -> dict[str, Any]:
    """
    Celery task that orchestrates the end-to-end repository parsing pipeline:
      1. Clones the remote git repository to a temporary directory (full clone).
      2. Recursively scans Python files and extracts the AST / dependency graph.
      3. Computes PageRank centrality and Git commit churn metrics on nodes.
      4. Detects candidate dead code (in-degree 0 in call graph, excluding entry points).
      5. Serializes the NetworkX graph into a JSON-serializable dict.
      6. Cleans up the temporary clone directory and triggers garbage collection.
      7. Returns the graph dictionary.

    Args:
        repo_url: Public Git / GitHub repository URL (e.g. 'https://github.com/owner/repo').

    Returns:
        Dictionary containing 'meta' (with dead_code_candidates), 'nodes', and 'edges'.
    """
    logger.info("Received process_repository_task for repo_url: %s", repo_url)

    clone_path: Path | None = None
    try:
        # Step 1: Clone repository to temp directory
        logger.info("Cloning repository: %s", repo_url)
        clone_path = clone_repository(repo_url)
        logger.info("Repository cloned to: %s", clone_path)

        # Step 2: Scan repository and build graph (includes PageRank)
        logger.info("Scanning repository structure at: %s", clone_path)
        graph = scan_repository(clone_path)

        # Step 3: Git Churn Hotspots — count commits touching each file
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
        return graph_dict

    except Exception as exc:
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

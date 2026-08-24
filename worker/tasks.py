"""
worker/tasks.py
---------------
Celery task definitions for background repository processing in CodeBase Visualizer.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from app.services.git_service import clone_repository
from parser.repo_walker import attach_churn, graph_to_json, scan_repository
from worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="worker.tasks.process_repository_task")
def process_repository_task(repo_url: str) -> dict[str, Any]:
    """
    Celery task that orchestrates the end-to-end repository parsing pipeline:
      1. Clones the remote git repository to a temporary directory (full clone).
      2. Recursively scans Python files and extracts the AST / dependency graph.
      3. Computes PageRank centrality and Git commit churn metrics on nodes.
      4. Serializes the NetworkX graph into a JSON-serializable dict.
      5. Cleans up the temporary clone directory.
      6. Returns the graph dictionary.

    Args:
        repo_url: Public Git / GitHub repository URL (e.g. 'https://github.com/owner/repo').

    Returns:
        Dictionary containing 'meta', 'nodes', and 'edges' representing the codebase graph.
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

        logger.info(
            "Scan completed successfully: %d nodes, %d edges.",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

        # Step 4: Serialize graph to JSON-friendly dictionary
        graph_dict = graph_to_json(graph)
        logger.info("Serialized graph dictionary for %s", repo_url)
        return graph_dict

    except Exception as exc:
        logger.exception("Failed to process repository '%s': %s", repo_url, exc)
        raise
    finally:
        # Clean up temporary cloned files to prevent disk accumulation
        if clone_path and clone_path.exists():
            try:
                shutil.rmtree(clone_path, ignore_errors=True)
                logger.info("Cleaned up temporary repository clone at %s", clone_path)
            except Exception as clean_err:
                logger.warning("Failed to clean up directory %s: %s", clone_path, clean_err)

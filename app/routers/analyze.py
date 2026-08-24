"""
routers/analyze.py
==================
Exposes the POST /api/v1/analyze-repo endpoint.

Dispatches an asynchronous Celery task (process_repository_task) to clone,
scan, and build the dependency graph of the given repository.
Returns the Celery task UUID immediately without blocking on cloning or scanning.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, status
from pydantic import BaseModel, HttpUrl

from worker.tasks import process_repository_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analyze"])


class AnalyzeRepoRequest(BaseModel):
    repo_url: HttpUrl


class AnalyzeRepoResponse(BaseModel):
    task_id: str
    status: str = "PENDING"
    message: str = "Repository analysis task queued."


@router.post(
    "/analyze-repo",
    response_model=AnalyzeRepoResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Queue asynchronous repository analysis",
    description=(
        "Accepts a public Git repository URL and dispatches a background Celery task "
        "to clone, scan, and parse the repository into an AST dependency graph. "
        "Returns the task_id immediately for asynchronous status polling."
    ),
)
def analyze_repo(body: AnalyzeRepoRequest) -> AnalyzeRepoResponse:
    """
    POST /api/v1/analyze-repo

    Request body:
        { "repo_url": "https://github.com/owner/repo" }

    Response (202):
        {
            "task_id": "<celery-task-uuid>",
            "status": "PENDING",
            "message": "Repository analysis task queued."
        }
    """
    url_str = str(body.repo_url)
    logger.info("Dispatching process_repository_task for repo_url=%s", url_str)

    task = process_repository_task.delay(url_str)

    logger.info("Enqueued task_id=%s for repo_url=%s", task.id, url_str)
    return AnalyzeRepoResponse(
        task_id=task.id,
        status="PENDING",
        message="Repository analysis task queued.",
    )

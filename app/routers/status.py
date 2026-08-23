"""
routers/status.py
=================
Exposes the GET /api/v1/status/{task_id} endpoint.

Queries Celery's result backend for asynchronous task execution state
and maps Celery's internal states (PENDING, STARTED, SUCCESS, FAILURE)
into a structured JSON response.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from celery.result import AsyncResult
from fastapi import APIRouter, status
from pydantic import BaseModel

from worker.celery_app import celery_app

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["status"])


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


@router.get(
    "/status/{task_id}",
    response_model=TaskStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get status of an asynchronous background task",
    description=(
        "Queries the Celery result backend using the given task UUID path parameter. "
        "Returns the current status (PENDING, STARTED, SUCCESS, FAILURE), "
        "including graph analysis output upon success or error details on failure."
    ),
)
def get_task_status(task_id: str) -> TaskStatusResponse:
    """
    GET /api/v1/status/{task_id}

    Path Parameter:
        task_id: Celery task UUID string.

    Response (200):
        PENDING:
            { "task_id": "<uuid>", "status": "PENDING", "result": null, "error": null }
        STARTED:
            { "task_id": "<uuid>", "status": "STARTED", "result": null, "error": null }
        SUCCESS:
            { "task_id": "<uuid>", "status": "SUCCESS", "result": { "meta": {...}, "nodes": [...], "edges": [...] }, "error": null }
        FAILURE:
            { "task_id": "<uuid>", "status": "FAILURE", "result": null, "error": "<error message>" }
    """
    async_result = AsyncResult(task_id, app=celery_app)
    state = async_result.state

    logger.info("Checked status for task_id=%s: state=%s", task_id, state)

    if state == "PENDING":
        return TaskStatusResponse(
            task_id=task_id,
            status="PENDING",
            result=None,
            error=None,
        )

    if state in ("STARTED", "PROCESSING"):
        return TaskStatusResponse(
            task_id=task_id,
            status=state,
            result=None,
            error=None,
        )

    if state == "SUCCESS":
        return TaskStatusResponse(
            task_id=task_id,
            status="SUCCESS",
            result=async_result.result,
            error=None,
        )

    if state == "FAILURE":
        error_msg = str(async_result.info) if async_result.info is not None else "Task execution failed."
        return TaskStatusResponse(
            task_id=task_id,
            status="FAILURE",
            result=None,
            error=error_msg,
        )

    # Catch-all for other Celery states (e.g. RETRY, REVOKED)
    return TaskStatusResponse(
        task_id=task_id,
        status=state,
        result=None,
        error=str(async_result.info) if async_result.failed() else None,
    )

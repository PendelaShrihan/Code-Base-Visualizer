"""
routers/git.py
Exposes the /api/v1/clone endpoint.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, HttpUrl

from app.services.git_service import clone_repository

router = APIRouter(prefix="/api/v1", tags=["git"])


class CloneRequest(BaseModel):
    repo_url: HttpUrl


class CloneResponse(BaseModel):
    repo_url: str
    clone_path: str
    message: str


@router.post(
    "/clone",
    response_model=CloneResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Clone a public GitHub repository",
    description=(
        "Clones the given public GitHub repository to a unique directory on "
        "disk under /tmp/repos/ and returns the local path."
    ),
)
def clone_repo(body: CloneRequest) -> CloneResponse:
    """
    POST /api/v1/clone

    Request body:
        { "repo_url": "https://github.com/owner/repo" }

    Response (201):
        {
            "repo_url":   "https://github.com/owner/repo",
            "clone_path": "/tmp/repos/<uuid>",
            "message":    "Repository cloned successfully."
        }
    """
    url_str = str(body.repo_url)

    try:
        dest = clone_repository(url_str)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    return CloneResponse(
        repo_url=url_str,
        clone_path=str(dest),
        message="Repository cloned successfully.",
    )

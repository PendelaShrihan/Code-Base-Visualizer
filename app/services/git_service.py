"""
git_service.py
Handles cloning of public GitHub repositories to a local temp directory.
"""

import shutil
import uuid
import tempfile
from pathlib import Path

import git
from git.exc import GitCommandError


# Base directory under which every cloned repo will live.
REPOS_BASE_DIR = Path(tempfile.gettempdir()) / "repos"


def clone_repository(repo_url: str) -> Path:
    """
    Clone a public GitHub repository into a unique subdirectory.

    Args:
        repo_url: Public GitHub HTTPS URL, e.g.
                  "https://github.com/owner/repo"

    Returns:
        Path object pointing at the directory that contains the cloned repo.

    Raises:
        ValueError: If the URL is empty or obviously malformed.
        RuntimeError: If the clone operation fails (private repo, bad URL, etc.).
    """
    if not repo_url or not repo_url.strip():
        raise ValueError("repo_url must not be empty.")

    repo_url = repo_url.strip()

    # Create a unique destination directory so concurrent requests never
    # collide, even when cloning the same repo twice.
    clone_id = uuid.uuid4().hex
    dest: Path = REPOS_BASE_DIR / clone_id
    dest.mkdir(parents=True, exist_ok=True)

    try:
        git.Repo.clone_from(repo_url, str(dest))
    except GitCommandError as exc:
        # Remove the destination entirely — clone_from can leave partial
        # files behind on network drops or disk-full errors, so rmdir()
        # (empty-dir only) is not sufficient here.
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(
            f"Failed to clone repository '{repo_url}': {exc}"
        ) from exc

    return dest

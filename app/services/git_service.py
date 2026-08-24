"""
git_service.py
Handles cloning of public GitHub repositories to a local temp directory.
"""

import logging
import shutil
import tempfile
import uuid
from pathlib import Path

import git
from git.exc import GitCommandError

logger = logging.getLogger(__name__)


# Base directory under which every cloned repo will live.
REPOS_BASE_DIR = Path(tempfile.gettempdir()) / "repos"


def clone_repository(repo_url: str) -> Path:
    """
    Clone a public GitHub repository into a unique subdirectory.

    Note:
        Performs a full (non-shallow) clone so that commit history is complete
        and available for git churn analysis (via repo.iter_commits()).

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
        # Full clone (no depth=1 limit) ensures all commits are accessible
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


def count_file_commits(repo_path: Path | str) -> dict[str, int]:
    """
    Count the number of commits modifying each file across the git repository history.

    Uses GitPython's `repo.iter_commits()` to traverse the commit log and aggregates
    modifications per normalized POSIX file path.

    Args:
        repo_path: Path to the local git repository root.

    Returns:
        Dictionary mapping relative POSIX file paths to total commit count.
    """
    path = Path(repo_path)
    churn_counts: dict[str, int] = {}
    try:
        repo = git.Repo(str(path))
        for commit in repo.iter_commits():
            for filepath in commit.stats.files:
                posix_path = Path(filepath).as_posix()
                churn_counts[posix_path] = churn_counts.get(posix_path, 0) + 1
    except Exception as exc:
        logger.warning("Failed to compute commit churn for %s: %s", path, exc)

    return churn_counts

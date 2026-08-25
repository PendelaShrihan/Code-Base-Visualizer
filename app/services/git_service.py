"""
git_service.py
Handles cloning of public GitHub repositories to a local temp directory.
"""

import logging
import os
import shutil
import stat
import tempfile
import time
import uuid
from pathlib import Path

import git
from git.exc import GitCommandError

logger = logging.getLogger(__name__)


# Base directory under which every cloned repo will live.
REPOS_BASE_DIR = Path(tempfile.gettempdir()) / "repos"


def cleanup_repo_directory(repo_path: Path | str) -> bool:
    """
    Safely and thoroughly delete a cloned repository directory.

    Handles read-only file permissions (common in .git directory objects on
    both Windows and Linux filesystems) by clearing the read-only flag upon removal failure.

    Args:
        repo_path: Path to the repository directory to delete.

    Returns:
        True if successfully removed or non-existent, False otherwise.
    """
    path = Path(repo_path)
    if not path.exists():
        return True

    def _remove_readonly(func, fpath, exc_info):
        try:
            os.chmod(fpath, stat.S_IWRITE)
            func(fpath)
        except Exception as err:
            logger.warning("Failed clearing read-only flag for %s: %s", fpath, err)

    try:
        shutil.rmtree(path, onerror=_remove_readonly)
        logger.info("Successfully cleaned up repo directory: %s", path)
        return True
    except Exception as exc:
        logger.warning("Error during cleanup of repo directory %s: %s", path, exc)
        return False


def garbage_collect_temp_repos(
    base_dir: Path | str | None = None,
    max_age_seconds: int = 1800,
) -> int:
    """
    Garbage collect stale temporary repository directories in REPOS_BASE_DIR.

    Scans the temporary repository root for subdirectories older than `max_age_seconds`
    (default 30 minutes) and deletes them to prevent disk-leak vulnerabilities from
    interrupted or orphaned worker jobs.

    Args:
        base_dir: Root directory containing temporary clone folders (defaults to REPOS_BASE_DIR).
        max_age_seconds: Maximum allowed age in seconds before a temp directory is purged.

    Returns:
        Number of stale repo directories purged.
    """
    target_dir = Path(base_dir) if base_dir is not None else REPOS_BASE_DIR
    if not target_dir.exists() or not target_dir.is_dir():
        return 0

    now = time.time()
    purged_count = 0

    try:
        for entry in target_dir.iterdir():
            if entry.is_dir():
                try:
                    stat_info = entry.stat()
                    dir_age = now - stat_info.st_mtime
                    if dir_age > max_age_seconds:
                        logger.info(
                            "Garbage collecting stale temp repo %s (age: %.1fs > %ds)",
                            entry.name,
                            dir_age,
                            max_age_seconds,
                        )
                        if cleanup_repo_directory(entry):
                            purged_count += 1
                except Exception as entry_err:
                    logger.warning("Failed checking entry %s during GC: %s", entry, entry_err)
    except Exception as exc:
        logger.warning("Error during garbage collection in %s: %s", target_dir, exc)

    if purged_count > 0:
        logger.info("Garbage collection complete: purged %d stale directories in %s", purged_count, target_dir)

    return purged_count


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

"""
git_service.py
Handles cloning of public GitHub repositories to a local temp directory.
"""

import concurrent.futures
import logging
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import git
from git.exc import GitCommandError

from app.exceptions import (
    CloneTimeoutError,
    IngestionGuardrailError,
    RepoSizeLimitExceededError,
)

logger = logging.getLogger(__name__)


# Guardrail defaults (configurable via environment variables)
DEFAULT_MAX_REPO_SIZE_MB: int = 200
DEFAULT_MAX_REPO_SIZE_BYTES: int = int(
    os.getenv(
        "MAX_REPO_SIZE_BYTES",
        str(int(os.getenv("MAX_REPO_SIZE_MB", str(DEFAULT_MAX_REPO_SIZE_MB))) * 1024 * 1024),
    )
)
DEFAULT_CLONE_TIMEOUT_SECONDS: float = float(os.getenv("CLONE_TIMEOUT_SECONDS", "120.0"))

# Base directory under which every cloned repo will live.
REPOS_BASE_DIR = Path(tempfile.gettempdir()) / "repos"


def get_directory_size_bytes(directory: Path | str) -> int:
    """
    Calculate the total size in bytes of all regular files in a directory tree.

    Symlinks are skipped to guard against symlink loops or traversal attacks.

    Args:
        directory: Path to root folder to inspect.

    Returns:
        Total size in bytes.
    """
    total = 0
    path = Path(directory)
    if not path.exists():
        return 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                if not os.path.islink(fp):
                    total += os.path.getsize(fp)
            except (OSError, PermissionError):
                continue
    return total


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


def clone_repository(
    repo_url: str,
    timeout_seconds: float | None = None,
    max_size_bytes: int | None = None,
) -> Path:
    """
    Clone a public GitHub repository into a unique subdirectory with defensive guardrails.

    Enforces:
      1. Hard clone timeout to prevent hanging connections or slowloris denial-of-service.
      2. Repository size cap (rejecting clones over roughly 200MB) to prevent disk exhaustion.

    Args:
        repo_url: Public GitHub HTTPS URL, e.g. "https://github.com/owner/repo"
        timeout_seconds: Maximum allowed seconds for the clone operation. Defaults to
                         DEFAULT_CLONE_TIMEOUT_SECONDS (120s or CLONE_TIMEOUT_SECONDS env).
        max_size_bytes: Maximum allowed repository size on disk in bytes. Defaults to
                        DEFAULT_MAX_REPO_SIZE_BYTES (200MB or MAX_REPO_SIZE_MB env).

    Returns:
        Path object pointing at the directory that contains the cloned repo.

    Raises:
        ValueError: If the URL is empty or obviously malformed.
        CloneTimeoutError: If the clone operation exceeds timeout_seconds.
        RepoSizeLimitExceededError: If the cloned repository size exceeds max_size_bytes.
        RuntimeError: If the clone operation fails (private repo, bad URL, etc.).
    """
    if not repo_url or not repo_url.strip():
        raise ValueError("repo_url must not be empty.")

    repo_url = repo_url.strip()
    effective_timeout = (
        timeout_seconds if timeout_seconds is not None else DEFAULT_CLONE_TIMEOUT_SECONDS
    )
    effective_max_size = (
        max_size_bytes if max_size_bytes is not None else DEFAULT_MAX_REPO_SIZE_BYTES
    )

    # Create a unique destination directory so concurrent requests never
    # collide, even when cloning the same repo twice.
    clone_id = uuid.uuid4().hex
    dest: Path = REPOS_BASE_DIR / clone_id
    dest.mkdir(parents=True, exist_ok=True)

    git_env = {
        "GIT_CONFIG_COUNT": "2",
        "GIT_CONFIG_KEY_0": "http.version",
        "GIT_CONFIG_VALUE_0": "HTTP/1.1",
        "GIT_CONFIG_KEY_1": "http.postBuffer",
        "GIT_CONFIG_VALUE_1": "524288000",  # 500 MiB — avoids buffer overflow on large packs
    }

    active_procs: list[Any] = []
    orig_safer_popen = git.cmd.safer_popen

    def _tracking_safer_popen(*args: Any, **kwargs: Any) -> Any:
        proc = orig_safer_popen(*args, **kwargs)
        active_procs.append(proc)
        return proc

    def _kill_active_processes() -> None:
        for proc in active_procs:
            try:
                proc.kill()
            except Exception:
                pass
            if sys.platform == "win32":
                try:
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        capture_output=True,
                        check=False,
                    )
                except Exception:
                    pass

    try:
        git.cmd.safer_popen = _tracking_safer_popen
        if effective_timeout > 0:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(
                git.Repo.clone_from,
                repo_url,
                str(dest),
                depth=1,
                env=git_env,
            )
            try:
                future.result(timeout=effective_timeout)
            except concurrent.futures.TimeoutError as exc:
                _kill_active_processes()
                cleanup_repo_directory(dest)
                raise CloneTimeoutError(
                    f"Cloning repository '{repo_url}' timed out after {effective_timeout:.1f} seconds."
                ) from exc
            finally:
                executor.shutdown(wait=False, cancel_futures=True)
        else:
            git.Repo.clone_from(
                repo_url,
                str(dest),
                depth=1,
                env=git_env,
            )
    except GitCommandError as exc:
        cleanup_repo_directory(dest)
        raise RuntimeError(
            f"Failed to clone repository '{repo_url}': {exc}"
        ) from exc
    except IngestionGuardrailError:
        raise
    except Exception as exc:
        cleanup_repo_directory(dest)
        raise RuntimeError(
            f"Failed to clone repository '{repo_url}': {exc}"
        ) from exc
    finally:
        git.cmd.safer_popen = orig_safer_popen

    # Guardrail: Repo size cap (reject clones over roughly 200MB)
    repo_size = get_directory_size_bytes(dest)
    if repo_size > effective_max_size:
        cleanup_repo_directory(dest)
        size_mb = repo_size / (1024 * 1024)
        limit_mb = effective_max_size / (1024 * 1024)
        raise RepoSizeLimitExceededError(
            f"Cloned repository size ({size_mb:.2f} MB) exceeds maximum allowed limit of {limit_mb:.1f} MB."
        )

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

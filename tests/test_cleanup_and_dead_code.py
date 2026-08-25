"""
tests/test_cleanup_and_dead_code.py
-----------------------------------
Tests for:
1. Automated Resource Cleanup (cleanup_repo_directory, garbage_collect_temp_repos).
2. Dead Code Detection & Entry Point Filtering (detect_dead_code, is_known_entry_point).
3. Celery worker tasks execution and GC orchestration.
"""

from __future__ import annotations

import os
import stat
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest

from app.services.git_service import (
    cleanup_repo_directory,
    garbage_collect_temp_repos,
)
from parser.repo_walker import (
    detect_dead_code,
    graph_to_json,
    is_known_entry_point,
    scan_repository,
)
from worker.tasks import garbage_collect_task, process_repository_task


# ===========================================================================
# 1. Automated Resource Cleanup & Garbage Collection Tests
# ===========================================================================


def test_cleanup_repo_directory_normal(tmp_path: Path):
    """Test cleaning up a regular temporary directory."""
    repo_dir = tmp_path / "test_repo"
    repo_dir.mkdir()
    (repo_dir / "file.py").write_text("print('hello')", encoding="utf-8")
    nested = repo_dir / "nested"
    nested.mkdir()
    (nested / "nested_file.py").write_text("x = 1", encoding="utf-8")

    assert repo_dir.exists()
    success = cleanup_repo_directory(repo_dir)

    assert success is True
    assert not repo_dir.exists()


def test_cleanup_repo_directory_readonly_git_files(tmp_path: Path):
    """Test cleaning up a directory with read-only files (simulating .git pack files)."""
    repo_dir = tmp_path / "git_repo"
    repo_dir.mkdir()
    git_dir = repo_dir / ".git" / "objects" / "pack"
    git_dir.mkdir(parents=True)

    pack_file = git_dir / "sample.pack"
    pack_file.write_bytes(b"GIT PACK DATA")

    # Set file and directory to read-only
    os.chmod(str(pack_file), stat.S_IREAD)

    assert repo_dir.exists()
    success = cleanup_repo_directory(repo_dir)

    assert success is True
    assert not repo_dir.exists()


def test_cleanup_repo_directory_nonexistent(tmp_path: Path):
    """Test cleaning up a path that does not exist returns True without error."""
    nonexistent = tmp_path / "does_not_exist_123"
    assert cleanup_repo_directory(nonexistent) is True


def test_garbage_collect_temp_repos(tmp_path: Path):
    """Test garbage collection purges stale repositories and preserves fresh ones."""
    base_dir = tmp_path / "repos"
    base_dir.mkdir()

    # Create stale repo 1 (age: 3600 seconds)
    stale_1 = base_dir / "stale_1"
    stale_1.mkdir()
    (stale_1 / "file1.py").write_text("pass", encoding="utf-8")
    old_time = time.time() - 3600
    os.utime(str(stale_1), (old_time, old_time))

    # Create stale repo 2 (age: 2000 seconds)
    stale_2 = base_dir / "stale_2"
    stale_2.mkdir()
    (stale_2 / "file2.py").write_text("pass", encoding="utf-8")
    os.utime(str(stale_2), (old_time, old_time))

    # Create fresh repo (age: current)
    fresh = base_dir / "fresh_repo"
    fresh.mkdir()
    (fresh / "file3.py").write_text("pass", encoding="utf-8")

    # Run GC with max_age_seconds = 1800 (30 mins)
    purged_count = garbage_collect_temp_repos(base_dir=base_dir, max_age_seconds=1800)

    assert purged_count == 2
    assert not stale_1.exists()
    assert not stale_2.exists()
    assert fresh.exists()


def test_garbage_collect_temp_repos_empty_or_missing(tmp_path: Path):
    """Test GC on a missing or empty directory."""
    missing = tmp_path / "nonexistent_dir"
    assert garbage_collect_temp_repos(base_dir=missing) == 0


# ===========================================================================
# 2. Dead Code Detection & Entry Point Filtering Tests
# ===========================================================================


def test_is_known_entry_point():
    """Verify entry point detection rules."""
    # Dunders
    assert is_known_entry_point("__init__") is True
    assert is_known_entry_point("__call__") is True
    assert is_known_entry_point("__enter__") is True

    # Standard entry point names
    assert is_known_entry_point("main") is True
    assert is_known_entry_point("__main__") is True
    assert is_known_entry_point("cli") is True
    assert is_known_entry_point("run") is True
    assert is_known_entry_point("app") is True
    assert is_known_entry_point("handler") is True
    assert is_known_entry_point("health_check") is True

    # Route handler prefixes
    assert is_known_entry_point("get_user") is True
    assert is_known_entry_point("post_item") is True
    assert is_known_entry_point("delete_record") is True

    # Tasks
    assert is_known_entry_point("process_repository_task") is True
    assert is_known_entry_point("task_sync") is True

    # Tests
    assert is_known_entry_point("test_feature") is True
    assert is_known_entry_point("feature_test") is True

    # Path heuristics
    assert is_known_entry_point("custom_handler", "app/routers/users.py") is True
    assert is_known_entry_point("sample_fn", "tests/test_basic.py") is True

    # Non-entry point regular helper
    assert is_known_entry_point("compute_distance", "utils/geometry.py") is False
    assert is_known_entry_point("format_string", "app/services/helpers.py") is False


def test_detect_dead_code_on_graph():
    """Test detect_dead_code correctly flags in-degree 0 functions while preserving entry points."""
    g = nx.DiGraph()

    # Function 1: caller calling callee
    g.add_node("mod.py::func::active_caller", kind="function", name="active_caller", file="mod.py")
    g.add_node("mod.py::func::active_callee", kind="function", name="active_callee", file="mod.py")
    g.add_edge("mod.py::func::active_caller", "mod.py::func::active_callee", rel="func_call", edge_type="EXTRACTED")

    # Function 2: Unused function (dead code candidate)
    g.add_node("mod.py::func::orphan_dead_fn", kind="function", name="orphan_dead_fn", file="mod.py")

    # Function 3: Entry points (in-degree 0 but filtered out)
    g.add_node("main.py::func::main", kind="function", name="main", file="main.py")
    g.add_node("mod.py::func::__init__", kind="function", name="__init__", file="mod.py")
    g.add_node("routers/auth.py::func::login", kind="function", name="login", file="routers/auth.py")

    candidates = detect_dead_code(g)
    candidate_names = [c["name"] for c in candidates]

    # orphan_dead_fn and active_caller both have in-degree 0 in call graph,
    # but active_caller is not called by anything in this synthetic graph.
    assert "orphan_dead_fn" in candidate_names
    assert "active_caller" in candidate_names
    assert "active_callee" not in candidate_names  # Called by active_caller

    # Entry points must be excluded
    assert "main" not in candidate_names
    assert "__init__" not in candidate_names
    assert "login" not in candidate_names

    # Check node attributes
    assert g.nodes["mod.py::func::orphan_dead_fn"]["is_dead_code_candidate"] is True
    assert g.nodes["mod.py::func::active_callee"]["is_dead_code_candidate"] is False
    assert g.nodes["main.py::func::main"]["is_dead_code_candidate"] is False


def test_scan_repository_and_dead_code_end_to_end(tmp_path: Path):
    """End-to-end test of scanning a multi-file Python repository and checking dead code output."""
    repo_dir = tmp_path / "sample_codebase"
    repo_dir.mkdir()

    # File 1: utils.py with active and dead functions
    (repo_dir / "utils.py").write_text(
        """
def helper_used():
    return 42

def caller_func():
    return helper_used()

def dead_abandoned_function():
    return "never called"
""",
        encoding="utf-8",
    )

    # File 2: entry.py with main and a dead helper
    (repo_dir / "entry.py").write_text(
        """
def main():
    print("starting")

def another_dead_function():
    pass
""",
        encoding="utf-8",
    )

    graph = scan_repository(repo_dir)
    json_data = graph_to_json(graph)

    dead_candidates = json_data["meta"]["dead_code_candidates"]
    dead_names = {c["name"] for c in dead_candidates}

    assert "dead_abandoned_function" in dead_names
    assert "another_dead_function" in dead_names
    assert "helper_used" not in dead_names  # called by caller_func
    assert "main" not in dead_names         # entry point excluded

    assert json_data["meta"]["dead_code_count"] >= 2


# ===========================================================================
# 3. Celery Worker Task Pipeline Tests
# ===========================================================================


@patch("worker.tasks.garbage_collect_temp_repos")
@patch("worker.tasks.cleanup_repo_directory")
@patch("worker.tasks.attach_churn")
@patch("worker.tasks.scan_repository")
@patch("worker.tasks.clone_repository")
def test_process_repository_task_success(
    mock_clone,
    mock_scan,
    mock_attach_churn,
    mock_cleanup,
    mock_gc,
    tmp_path: Path,
):
    """Test process_repository_task pipeline executes analysis and always runs cleanup and GC."""
    mock_clone_path = tmp_path / "clone_123"
    mock_clone_path.mkdir()
    mock_clone.return_value = mock_clone_path

    # Synthetic graph
    g = nx.DiGraph()
    g.add_node("sample.py::func::unused", kind="function", name="unused", file="sample.py")
    mock_scan.return_value = g
    mock_cleanup.return_value = True
    mock_gc.return_value = 0

    result = process_repository_task("https://github.com/example/repo")

    # Assert graph results
    assert "meta" in result
    assert "nodes" in result
    assert "edges" in result
    assert result["meta"]["dead_code_count"] == 1
    assert result["meta"]["dead_code_candidates"][0]["name"] == "unused"

    # Assert cleanup and GC were invoked
    mock_cleanup.assert_called_once_with(mock_clone_path)
    mock_gc.assert_called_once()


@patch("worker.tasks.garbage_collect_temp_repos")
@patch("worker.tasks.cleanup_repo_directory")
@patch("worker.tasks.scan_repository")
@patch("worker.tasks.clone_repository")
def test_process_repository_task_cleanup_on_error(
    mock_clone,
    mock_scan,
    mock_cleanup,
    mock_gc,
    tmp_path: Path,
):
    """Test process_repository_task cleans up cloned directory even if scanning throws an exception."""
    mock_clone_path = tmp_path / "clone_error"
    mock_clone_path.mkdir()
    mock_clone.return_value = mock_clone_path
    mock_scan.side_effect = RuntimeError("AST parse crash")

    with pytest.raises(RuntimeError, match="AST parse crash"):
        process_repository_task("https://github.com/example/broken_repo")

    # Cleanup and GC must still be triggered in finally
    mock_cleanup.assert_called_once_with(mock_clone_path)
    mock_gc.assert_called_once()


@patch("worker.tasks.garbage_collect_temp_repos")
def test_garbage_collect_task(mock_gc):
    """Test standalone celery GC task."""
    mock_gc.return_value = 3
    purged = garbage_collect_task(max_age_seconds=1200)
    assert purged == 3
    mock_gc.assert_called_once_with(max_age_seconds=1200)

"""
tests/test_ingestion_guardrails.py
----------------------------------
Unit and integration tests for Day 25 Ingestion Guardrails:
1. Repository size cap (rejecting clones over roughly 200MB / configurable cap).
2. Hard clone timeout (aborting and killing hung git operations).
3. Max-file-count limit before AST parsing begins.
4. Defensive cleanup & resource isolation for untrusted external data.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest
from fastapi.testclient import TestClient

from app.exceptions import (
    CloneTimeoutError,
    FileParseTimeoutError,
    IngestionGuardrailError,
    MaxFileCountExceededError,
    RepoSizeLimitExceededError,
)
from app.main import app
from app.services.ast_engine import parse_with_timeout
from app.services.git_service import (
    DEFAULT_CLONE_TIMEOUT_SECONDS,
    DEFAULT_MAX_REPO_SIZE_BYTES,
    clone_repository,
    get_directory_size_bytes,
)
from parser.repo_walker import (
    _build_file_subgraph_with_timeout,
    _iter_python_files,
    is_binary_file,
    is_generated_file,
    is_minified_file,
    scan_repository,
    should_skip_file,
)
from worker.tasks import process_repository_task

client = TestClient(app)


# ===========================================================================
# 1. Repository Size Cap Guardrail Tests
# ===========================================================================


def test_get_directory_size_bytes(tmp_path: Path):
    """Verify get_directory_size_bytes accurately accumulates file bytes."""
    test_dir = tmp_path / "sized_repo"
    test_dir.mkdir()

    (test_dir / "file1.txt").write_bytes(b"A" * 1024)  # 1 KB
    sub_dir = test_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "file2.txt").write_bytes(b"B" * 2048)  # 2 KB

    total_size = get_directory_size_bytes(test_dir)
    assert total_size == 3072

    # Nonexistent path returns 0
    assert get_directory_size_bytes(tmp_path / "nonexistent") == 0


def test_clone_repository_size_cap_exceeded(tmp_path: Path):
    """Test that a cloned repository exceeding max_size_bytes is immediately rejected and purged."""
    limit_bytes = 5000  # 5 KB cap for test

    def fake_clone(url, to_path, **kwargs):
        dest_p = Path(to_path)
        # Write 10 KB file (exceeding 5 KB limit)
        (dest_p / "large_binary.bin").write_bytes(b"X" * 10000)

    with patch("git.Repo.clone_from", side_effect=fake_clone):
        with pytest.raises(RepoSizeLimitExceededError) as exc_info:
            clone_repository(
                "https://github.com/untrusted/huge-repo",
                max_size_bytes=limit_bytes,
            )

        err_msg = str(exc_info.value)
        assert "exceeds maximum allowed limit" in err_msg


def test_clone_repository_size_cap_within_limit(tmp_path: Path):
    """Test that a cloned repository within size limits is successfully retained."""
    limit_bytes = 100000  # 100 KB limit

    def fake_clone(url, to_path, **kwargs):
        dest_p = Path(to_path)
        (dest_p / "module.py").write_text("def hello(): pass", encoding="utf-8")

    with patch("git.Repo.clone_from", side_effect=fake_clone):
        dest = clone_repository(
            "https://github.com/safe/small-repo",
            max_size_bytes=limit_bytes,
        )
        try:
            assert dest.exists()
            assert (dest / "module.py").exists()
        finally:
            from app.services.git_service import cleanup_repo_directory
            cleanup_repo_directory(dest)


# ===========================================================================
# 2. Hard Clone Timeout Guardrail Tests
# ===========================================================================


def test_clone_repository_timeout_exceeded():
    """Test that a hanging git clone triggers CloneTimeoutError and terminates."""
    def hanging_clone(url, to_path, **kwargs):
        time.sleep(1.5)  # Exceeds the 0.2s test timeout

    with patch("git.Repo.clone_from", side_effect=hanging_clone):
        with pytest.raises(CloneTimeoutError) as exc_info:
            clone_repository(
                "https://github.com/untrusted/hanging-repo",
                timeout_seconds=0.2,
            )

        assert "timed out after 0.2 seconds" in str(exc_info.value)


def test_clone_repository_invalid_url():
    """Test that empty or whitespace repository URLs are defensively rejected."""
    with pytest.raises(ValueError, match="repo_url must not be empty"):
        clone_repository("")

    with pytest.raises(ValueError, match="repo_url must not be empty"):
        clone_repository("   ")


# ===========================================================================
# 3. Max-File-Count Limit Guardrail Tests (Before Parsing Begins)
# ===========================================================================


def test_max_file_count_guardrail_exceeded(tmp_path: Path):
    """Test that repositories with file counts exceeding max_files are rejected before parsing."""
    repo_dir = tmp_path / "huge_file_tree"
    repo_dir.mkdir()

    # Create 5 Python files
    for i in range(5):
        (repo_dir / f"script_{i}.py").write_text(f"x = {i}", encoding="utf-8")

    # Set max_files to 3 -> should raise MaxFileCountExceededError
    with pytest.raises(MaxFileCountExceededError) as exc_info:
        scan_repository(repo_dir, max_files=3)

    assert "exceeding the maximum allowed limit of 3" in str(exc_info.value)


def test_max_file_count_guardrail_within_limit(tmp_path: Path):
    """Test that repositories within max_files limit parse normally."""
    repo_dir = tmp_path / "normal_repo"
    repo_dir.mkdir()

    for i in range(3):
        (repo_dir / f"worker_{i}.py").write_text(f"def job_{i}(): pass", encoding="utf-8")

    graph = scan_repository(repo_dir, max_files=10)
    assert isinstance(graph, nx.DiGraph)
    assert graph.number_of_nodes() > 0


def test_iter_python_files_early_abort(tmp_path: Path):
    """Test that _iter_python_files aborts early when max_files is reached."""
    repo_dir = tmp_path / "deep_tree"
    repo_dir.mkdir()

    for i in range(6):
        (repo_dir / f"test_{i}.py").write_text("pass", encoding="utf-8")

    with pytest.raises(MaxFileCountExceededError):
        _iter_python_files(repo_dir, max_files=2)


# ===========================================================================
# 4. Celery Worker Task Guardrail Integration Tests
# ===========================================================================


@patch("worker.tasks.cleanup_repo_directory")
@patch("worker.tasks.clone_repository")
def test_worker_task_guardrail_timeout(mock_clone, mock_cleanup):
    """Test Celery worker cleans up and propagates CloneTimeoutError."""
    mock_clone.side_effect = CloneTimeoutError("Clone timed out after 120s")

    with pytest.raises(CloneTimeoutError):
        process_repository_task("https://github.com/slow/repo")


@patch("worker.tasks.cleanup_repo_directory")
@patch("worker.tasks.clone_repository")
def test_worker_task_guardrail_size_exceeded(mock_clone, mock_cleanup):
    """Test Celery worker cleans up and propagates RepoSizeLimitExceededError."""
    mock_clone.side_effect = RepoSizeLimitExceededError("Repo size 300MB exceeds 200MB")

    with pytest.raises(RepoSizeLimitExceededError):
        process_repository_task("https://github.com/huge/repo")


@patch("worker.tasks.cleanup_repo_directory")
@patch("worker.tasks.scan_repository")
@patch("worker.tasks.clone_repository")
def test_worker_task_guardrail_file_count_exceeded(mock_clone, mock_scan, mock_cleanup, tmp_path: Path):
    """Test Celery worker cleans up and propagates MaxFileCountExceededError."""
    mock_dest = tmp_path / "clone_files"
    mock_dest.mkdir()
    mock_clone.return_value = mock_dest
    mock_scan.side_effect = MaxFileCountExceededError("File count 5000 exceeds 2000")

    with pytest.raises(MaxFileCountExceededError):
        process_repository_task("https://github.com/bomb/repo")

    mock_cleanup.assert_called_once_with(mock_dest)


# ===========================================================================
# 5. FastAPI Endpoints HTTP Guardrail Response Tests
# ===========================================================================


@patch("app.routers.git.clone_repository")
def test_api_clone_timeout_504(mock_clone):
    """POST /api/v1/clone returns 504 Gateway Timeout when clone times out."""
    mock_clone.side_effect = CloneTimeoutError("Cloning timed out after 120.0 seconds.")

    response = client.post("/api/v1/clone", json={"repo_url": "https://github.com/slow/repo"})
    assert response.status_code == 504
    assert "timed out" in response.json()["detail"]


@patch("app.routers.git.clone_repository")
def test_api_clone_size_limit_413(mock_clone):
    """POST /api/v1/clone returns 413 Payload Too Large when repo exceeds size limit."""
    mock_clone.side_effect = RepoSizeLimitExceededError("Cloned repository size (250 MB) exceeds 200 MB.")

    response = client.post("/api/v1/clone", json={"repo_url": "https://github.com/huge/repo"})
    assert response.status_code == 413
    assert "exceeds" in response.json()["detail"]


# ===========================================================================
# 6. Resource-Limited Parsing & Noise File Skipping Guardrail Tests
# ===========================================================================


def test_is_binary_file():
    """Verify null bytes in content trigger is_binary_file detection."""
    assert is_binary_file(b"some\x00binary\x01data") is True
    assert is_binary_file(b"def clean_python_code():\n    return 42\n") is False


def test_is_minified_file():
    """Verify long lines or .min. suffixes trigger is_minified_file."""
    normal_code = b"def add(a, b):\n    return a + b\n"
    assert is_minified_file("calc.py", normal_code) is False

    # Name-based
    assert is_minified_file("bundle.min.py", normal_code) is True

    # Line-length based (>1000 chars)
    long_line = b"x = 1; " * 300 + b"\n"
    assert is_minified_file("packed.py", long_line) is True


def test_is_generated_file():
    """Verify generated suffixes, path parts, and headers trigger is_generated_file."""
    clean_code = b"def user_logic(): pass\n"
    assert is_generated_file("service.py", clean_code) is False

    # Protobuf naming
    assert is_generated_file("auth_pb2.py", clean_code) is True
    assert is_generated_file("auth_pb2_grpc.py", clean_code) is True

    # Path part
    assert is_generated_file("vendor/some_lib/core.py", clean_code) is True
    assert is_generated_file("dist/output.py", clean_code) is True
    assert is_generated_file("node_modules/pkg/index.py", clean_code) is True

    # Header comments
    assert is_generated_file("out.py", b"# @generated by build tool\ndef f(): pass") is True
    assert is_generated_file("out.py", b"# DO NOT EDIT - Code generated automatically\n") is True


def test_should_skip_file():
    """Verify should_skip_file returns (True, reason) for pathological/noise files."""
    skip, reason = should_skip_file("vendor/lib.py", b"def f(): pass")
    assert skip is True
    assert "vendor" in reason

    skip, reason = should_skip_file("binary.py", b"\x00\x00\x03\x04")
    assert skip is True
    assert "binary" in reason

    skip, reason = should_skip_file("huge.py", b"x = 1; " * 300)
    assert skip is True
    assert "minified" in reason

    skip, reason = should_skip_file("api_pb2.py", b"def f(): pass")
    assert skip is True
    assert "generated" in reason

    skip, reason = should_skip_file("clean.py", b"def f(): pass")
    assert skip is False
    assert reason is None


def test_scan_repository_skips_vendor_and_dist_dirs(tmp_path: Path):
    """Verify scan_repository prunes vendor, dist, and node_modules directories."""
    repo = tmp_path / "repo_with_vendor"
    repo.mkdir()

    # Legitimate code
    (repo / "main.py").write_text("def run(): pass\n", encoding="utf-8")

    # Noise directories
    (repo / "vendor").mkdir()
    (repo / "vendor" / "dep.py").write_text("def vendored(): pass\n", encoding="utf-8")

    (repo / "dist").mkdir()
    (repo / "dist" / "bundle.py").write_text("def built(): pass\n", encoding="utf-8")

    (repo / "node_modules").mkdir()
    (repo / "node_modules" / "sub.py").write_text("def nm(): pass\n", encoding="utf-8")

    graph = scan_repository(repo)
    node_files = [attrs.get("file", "") for _, attrs in graph.nodes(data=True)]

    assert any("main.py" in f for f in node_files)
    assert not any("vendor" in f for f in node_files)
    assert not any("dist" in f for f in node_files)
    assert not any("node_modules" in f for f in node_files)


def test_scan_repository_skips_binary_minified_generated(tmp_path: Path):
    """Verify scan_repository filters out binary, minified, and generated files."""
    repo = tmp_path / "mixed_repo"
    repo.mkdir()

    # Valid file
    (repo / "service.py").write_text("def process(): pass\n", encoding="utf-8")

    # Binary file
    (repo / "bogus.py").write_bytes(b"\x00\x01\x02\x03def bogus(): pass")

    # Minified file
    (repo / "packed.min.py").write_text("def a(): pass\n", encoding="utf-8")
    (repo / "wide.py").write_text("x = 1; " * 300, encoding="utf-8")

    # Generated files
    (repo / "schema_pb2.py").write_text("def schema(): pass\n", encoding="utf-8")
    (repo / "stub.py").write_text("# @generated by grpc\ndef stub(): pass\n", encoding="utf-8")

    graph = scan_repository(repo)
    node_files = [attrs.get("file", "") for _, attrs in graph.nodes(data=True)]

    assert any("service.py" in f for f in node_files)
    assert not any("bogus.py" in f for f in node_files)
    assert not any("packed.min.py" in f for f in node_files)
    assert not any("wide.py" in f for f in node_files)
    assert not any("schema_pb2.py" in f for f in node_files)
    assert not any("stub.py" in f for f in node_files)


def test_parse_with_timeout_exceeded():
    """Verify parse_with_timeout raises FileParseTimeoutError on hanging parse."""
    mock_parser = MagicMock()
    mock_parser.parse.side_effect = lambda s: time.sleep(1.0)

    with patch("app.services.ast_engine._parser", mock_parser):
        with pytest.raises(FileParseTimeoutError) as exc_info:
            parse_with_timeout(b"def slow(): pass", timeout=0.1)

        assert "timed out after 0.1 seconds" in str(exc_info.value)


def test_build_file_subgraph_with_timeout_exceeded(tmp_path: Path):
    """Verify _build_file_subgraph_with_timeout raises FileParseTimeoutError."""
    test_file = tmp_path / "slow.py"
    test_file.write_text("def slow(): pass", encoding="utf-8")

    def hanging_subgraph(*args, **kwargs):
        time.sleep(1.0)
        return nx.DiGraph()

    with patch("parser.repo_walker._build_file_subgraph", side_effect=hanging_subgraph):
        with pytest.raises(FileParseTimeoutError) as exc_info:
            _build_file_subgraph_with_timeout(test_file, tmp_path, b"def slow(): pass", timeout=0.1)

        assert "timed out after 0.1 seconds" in str(exc_info.value)


def test_scan_repository_resilience_to_pathological_file_timeout(tmp_path: Path):
    """Verify that one timing-out pathological file is skipped without stalling or failing scan_repository."""
    repo = tmp_path / "timeout_test_repo"
    repo.mkdir()

    (repo / "good.py").write_text("def good_func(): pass\n", encoding="utf-8")
    (repo / "pathological.py").write_text("def pathological_func(): pass\n", encoding="utf-8")

    orig_build = _build_file_subgraph_with_timeout

    def selective_hang(file_path, repo_root, source_code, timeout=5.0):
        if Path(file_path).name == "pathological.py":
            raise FileParseTimeoutError(f"Parsing {Path(file_path).name} timed out after {timeout:.1f}s.")
        return orig_build(file_path, repo_root, source_code, timeout=timeout)

    with patch("parser.repo_walker._build_file_subgraph_with_timeout", side_effect=selective_hang):
        graph = scan_repository(repo, file_timeout_seconds=0.5)

    node_files = [attrs.get("file", "") for _, attrs in graph.nodes(data=True)]
    # 'good.py' was successfully parsed into the graph
    assert any("good.py" in f for f in node_files)
    # 'pathological.py' was safely skipped
    assert not any("pathological.py" in f for f in node_files)

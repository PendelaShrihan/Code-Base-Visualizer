"""
tests/test_prompt_builder.py
----------------------------
Unit tests for rag/prompt_builder.py — Prompt Engineering & Context Optimization.

Coverage Checklist:
-------------------
[x] estimate_tokens: empty strings, identifiers, symbols, and code text
[x] build_system_prompt: contains all anti-hallucination rules and citation format
[x] format_graph_topology: empty list, mixed results with churn/dead-code flags
[x] format_vector_snippet: metadata headers, rank, code block formatting
[x] format_graph_neighbor_block: callee vs caller relationship formatting
[x] resolve_function_code: code_cache priority, disk resolution, and missing file fallback
[x] build_rag_prompt: full prompt assembly, XML boundaries, to_messages()
[x] build_rag_prompt: context window budgeting and priority pruning
[x] build_rag_prompt: pure vector results (no graph neighbors)
[x] build_rag_prompt: empty results list handling
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from rag.hybrid_retriever import HybridResult
from rag.prompt_builder import (
    DEFAULT_MAX_CONTEXT_TOKENS,
    StructuredPrompt,
    build_rag_prompt,
    build_system_prompt,
    estimate_tokens,
    format_graph_neighbor_block,
    format_graph_topology,
    format_vector_snippet,
    resolve_function_code,
)


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_hybrid_results() -> list[HybridResult]:
    """Synthetic hybrid results representing a typical 1-anchor + 2-neighbor scenario."""
    return [
        {
            "rank": 1,
            "func_name": "clone_repository",
            "file_path": "app/services/git_service.py",
            "score": 0.8845,
            "similarity_label": "HIGH CONFIDENCE",
            "pagerank": 0.0520,
            "commit_count": 12,
            "is_dead_code_candidate": False,
            "source": "vector",
            "anchor_func": None,
        },
        {
            "rank": 2,
            "func_name": "cleanup_repo_directory",
            "file_path": "app/services/git_service.py",
            "score": 0.0,
            "similarity_label": "VERY WEAK / NOISE",
            "pagerank": 0.0150,
            "commit_count": 4,
            "is_dead_code_candidate": False,
            "source": "graph_callee",
            "anchor_func": "clone_repository",
        },
        {
            "rank": 3,
            "func_name": "process_repository_task",
            "file_path": "worker/tasks.py",
            "score": 0.0,
            "similarity_label": "VERY WEAK / NOISE",
            "pagerank": 0.0410,
            "commit_count": 15,
            "is_dead_code_candidate": False,
            "source": "graph_caller",
            "anchor_func": "clone_repository",
        },
    ]


# ---------------------------------------------------------------------------
# Token Estimation Tests
# ---------------------------------------------------------------------------

def test_estimate_tokens_empty():
    assert estimate_tokens("") == 0
    assert estimate_tokens(None) == 0  # type: ignore


def test_estimate_tokens_basic():
    short_text = "def hello_world(): return 42"
    tokens = estimate_tokens(short_text)
    assert tokens > 0
    assert tokens <= 15


def test_estimate_tokens_long_identifier():
    ident = "very_long_variable_name_with_multiple_parts_and_suffixes"
    tokens = estimate_tokens(ident)
    assert tokens >= 4


# ---------------------------------------------------------------------------
# System Prompt Tests (Hallucination Prevention)
# ---------------------------------------------------------------------------

def test_build_system_prompt_contains_grounding_rules():
    sys_prompt = build_system_prompt()
    assert "STRICT GROUNDING" in sys_prompt
    assert "IGNORANCE PROTOCOL" in sys_prompt
    assert "ANCHORS VS NEIGHBORS" in sys_prompt
    assert "MANDATORY CITATIONS" in sys_prompt
    assert "EVIDENCE FIRST" in sys_prompt
    assert "[file_path::func::<func_name>]" in sys_prompt
    assert "I cannot answer this question based on the provided codebase context" in sys_prompt


# ---------------------------------------------------------------------------
# Formatting Tests
# ---------------------------------------------------------------------------

def test_format_graph_topology_empty():
    text = format_graph_topology([])
    assert "No graph topology available" in text


def test_format_graph_topology_flags(sample_hybrid_results):
    # Add a dead-code candidate
    sample_hybrid_results.append({
        "rank": 4,
        "func_name": "deprecated_helper",
        "file_path": "app/utils.py",
        "score": 0.0,
        "similarity_label": "VERY WEAK / NOISE",
        "pagerank": 0.001,
        "commit_count": 1,
        "is_dead_code_candidate": True,
        "source": "graph_callee",
        "anchor_func": "clone_repository",
    })

    topology = format_graph_topology(sample_hybrid_results)
    assert "clone_repository" in topology
    assert "[PRIMARY VECTOR ANCHOR]" in topology
    assert "[CALLEE of clone_repository]" in topology
    assert "[CALLER of clone_repository]" in topology
    assert "DEAD_CODE_CANDIDATE" in topology
    assert "HOTSPOT_CHURN(15)" in topology


def test_format_vector_snippet():
    result: HybridResult = {
        "rank": 1,
        "func_name": "foo",
        "file_path": "pkg/bar.py",
        "score": 0.92,
        "similarity_label": "HIGH CONFIDENCE",
        "pagerank": 0.02,
        "commit_count": 5,
        "is_dead_code_candidate": False,
        "source": "vector",
        "anchor_func": None,
    }
    code = "def foo():\n    return 'bar'"
    snippet = format_vector_snippet(result, code, rank=1)
    assert '<code_snippet rank="1" function="foo" file="pkg/bar.py">' in snippet
    assert "0.9200" in snippet
    assert "def foo():" in snippet
    assert "</code_snippet>" in snippet


def test_format_graph_neighbor_block():
    result: HybridResult = {
        "rank": 2,
        "func_name": "sub_func",
        "file_path": "pkg/sub.py",
        "score": 0.0,
        "similarity_label": "VERY WEAK / NOISE",
        "pagerank": 0.01,
        "commit_count": 2,
        "is_dead_code_candidate": True,
        "source": "graph_callee",
        "anchor_func": "main_func",
    }
    block = format_graph_neighbor_block(result)
    assert '<neighbor function="sub_func" file="pkg/sub.py"' in block
    assert 'relation_type="graph_callee"' in block
    assert 'anchor="main_func"' in block
    assert 'dead_code="true"' in block
    assert "Called by anchor 'main_func'" in block


# ---------------------------------------------------------------------------
# Code Resolution Tests
# ---------------------------------------------------------------------------

def test_resolve_function_code_from_cache():
    cache = {"pkg/a.py::compute": "def compute(): return 100"}
    code = resolve_function_code("compute", "pkg/a.py", code_cache=cache)
    assert code == "def compute(): return 100"


def test_resolve_function_code_from_disk(tmp_path: Path):
    py_file = tmp_path / "service.py"
    py_file.write_text(
        "def helper(x: int) -> int:\n"
        "    \"\"\"Docstring.\"\"\"\n"
        "    return x * 2\n\n"
        "def other():\n"
        "    pass\n",
        encoding="utf-8",
    )
    code = resolve_function_code("helper", "service.py", repo_root=tmp_path)
    assert code is not None
    assert "def helper(x: int) -> int:" in code
    assert "return x * 2" in code
    assert "def other():" not in code


def test_resolve_function_code_nonexistent():
    code = resolve_function_code("nonexistent", "missing.py", repo_root=Path("/invalid/path"))
    assert code is None


# ---------------------------------------------------------------------------
# Full Prompt Construction & Budgeting Tests
# ---------------------------------------------------------------------------

def test_build_rag_prompt_basic(sample_hybrid_results):
    cache = {"app/services/git_service.py::clone_repository": "def clone_repository(): pass"}
    prompt: StructuredPrompt = build_rag_prompt(
        query="Explain git cloning error handling",
        hybrid_results=sample_hybrid_results,
        code_cache=cache,
        max_tokens=4000,
    )

    assert "<repository_context>" in prompt.user_prompt
    assert "</repository_context>" in prompt.user_prompt
    assert "<graph_topology>" in prompt.user_prompt
    assert "<code_snippets>" in prompt.user_prompt
    assert "<graph_neighbors>" in prompt.user_prompt
    assert "<user_query>" in prompt.user_prompt
    assert "Explain git cloning error handling" in prompt.user_prompt

    # Verify items tracking
    assert len(prompt.included_anchors) == 1
    assert "app/services/git_service.py::func::clone_repository" in prompt.included_anchors[0]
    assert len(prompt.included_neighbors) == 2
    assert prompt.token_budget_exceeded is False

    # Check to_messages format
    messages = prompt.to_messages()
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_build_rag_prompt_budget_pruning(sample_hybrid_results):
    # Set a tiny budget to force pruning
    cache = {
        "app/services/git_service.py::clone_repository": "def clone_repository():\n" + ("    x = 1\n" * 150)
    }

    # Extremely low budget: 350 tokens (barely fits system prompt + query)
    prompt = build_rag_prompt(
        query="How does clone work?",
        hybrid_results=sample_hybrid_results,
        code_cache=cache,
        max_tokens=450,
    )

    assert prompt.token_budget_exceeded is True
    assert len(prompt.pruned_items) > 0
    # Pruned reasons must be logged
    assert all("exceeded_token_budget" in item["reason"] for item in prompt.pruned_items)


def test_build_rag_prompt_pure_vector():
    vector_only: list[HybridResult] = [
        {
            "rank": 1,
            "func_name": "standalone_func",
            "file_path": "standalone.py",
            "score": 0.85,
            "similarity_label": "HIGH CONFIDENCE",
            "pagerank": 0.01,
            "commit_count": 1,
            "is_dead_code_candidate": False,
            "source": "vector",
            "anchor_func": None,
        }
    ]

    prompt = build_rag_prompt(
        query="What does standalone_func do?",
        hybrid_results=vector_only,
        code_cache={"standalone.py::standalone_func": "def standalone_func(): return True"},
    )

    assert len(prompt.included_anchors) == 1
    assert len(prompt.included_neighbors) == 0
    assert "No graph neighbors included" in prompt.user_prompt
    assert prompt.token_budget_exceeded is False


def test_build_rag_prompt_empty_results():
    prompt = build_rag_prompt(query="Any function?", hybrid_results=[])
    assert len(prompt.included_anchors) == 0
    assert len(prompt.included_neighbors) == 0
    assert "No code snippets included" in prompt.user_prompt
    assert "No graph neighbors included" in prompt.user_prompt

"""
tests/test_hallucination_eval.py
--------------------------------
Unit & Integration tests for Hallucination Comparison & RAG Pipeline Ablation Benchmark (eval/hallucination_eval.py).

Tests cover:
- Benchmark dataset structure and query coverage
- Benchmark call graph topology (nodes, edges, attributes)
- Single-query ablation evaluation under mock conditions
- Grounding score calculation and delta computation
- Full ablation benchmark runner and metric aggregation
- Markdown and JSON report generation and export
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import networkx as nx
import pytest
from qdrant_client import QdrantClient

from eval.hallucination_eval import (
    LABELED_ABLATION_DATASET,
    AblationBenchmarkReport,
    AblationQuestion,
    QueryAblationResult,
    build_benchmark_call_graph,
    evaluate_ablation_query,
    export_json_report,
    export_markdown_report,
    generate_architectural_explanation,
    generate_markdown_results_table,
    run_hallucination_comparison_benchmark,
)
from eval.retrieval_eval import DEFAULT_COLLECTION_NAME, TEST_CORPUS


# ===========================================================================
# Dataset & Graph Integrity Tests
# ===========================================================================

def test_labeled_ablation_dataset_integrity():
    """Verify labeled questions have valid identifiers, anchors, and collaborators."""
    assert len(LABELED_ABLATION_DATASET) >= 10

    known_funcs = {item["func_name"] for item in TEST_CORPUS}
    seen_ids = set()

    for q in LABELED_ABLATION_DATASET:
        assert q.query_id not in seen_ids, f"Duplicate query ID: {q.query_id}"
        seen_ids.add(q.query_id)
        assert q.query.strip(), f"Empty query in {q.query_id}"
        assert q.primary_anchor in known_funcs, f"Unknown primary anchor {q.primary_anchor} in {q.query_id}"
        assert len(q.expected_collaborators) > 0, f"No expected collaborators in {q.query_id}"
        for collab in q.expected_collaborators:
            assert collab in known_funcs, f"Unknown collaborator {collab} in {q.query_id}"


def test_build_benchmark_call_graph():
    """Verify the benchmark call graph contains expected function nodes and directed call edges."""
    g = build_benchmark_call_graph(TEST_CORPUS)

    assert isinstance(g, nx.DiGraph)
    assert g.number_of_nodes() == len(TEST_CORPUS)
    assert g.number_of_edges() >= 15

    # Check key nodes and attributes
    clone_node = "app/services/git_service.py::func::clone_repository"
    cleanup_node = "app/services/git_service.py::func::cleanup_repo_directory"

    assert clone_node in g
    assert cleanup_node in g
    assert g.nodes[clone_node]["kind"] == "function"
    assert g.nodes[clone_node]["name"] == "clone_repository"

    # Check caller-callee edge
    assert g.has_edge(clone_node, cleanup_node)
    assert g.edges[clone_node, cleanup_node]["rel"] == "calls"


# ===========================================================================
# Synthetic Explanation & Grounding Tests
# ===========================================================================

def test_generate_architectural_explanation_grounded_vs_ungrounded():
    """Verify explanation generator generates citations grounded in context."""
    q = LABELED_ABLATION_DATASET[0]
    corpus_lookup = {item["func_name"]: item for item in TEST_CORPUS}

    # Condition ON: Both anchor and collaborator provided
    anchors = ["app/services/git_service.py::func::clone_repository"]
    neighbors = ["app/services/git_service.py::func::cleanup_repo_directory (graph_callee)"]

    exp_on = generate_architectural_explanation(
        question=q,
        anchors=anchors,
        neighbors=neighbors,
        graph_enabled=True,
        corpus_lookup=corpus_lookup,
    )
    assert "[app/services/git_service.py::func::clone_repository]" in exp_on
    assert "[app/services/git_service.py::func::cleanup_repo_directory]" in exp_on

    # Condition OFF: Only anchor provided
    exp_off = generate_architectural_explanation(
        question=q,
        anchors=anchors,
        neighbors=[],
        graph_enabled=False,
        corpus_lookup=corpus_lookup,
    )
    assert "[app/services/git_service.py::func::clone_repository]" in exp_off
    assert "[app/services/git_service.py::func::cleanup_repo_directory]" in exp_off


# ===========================================================================
# Single Query Ablation Evaluator Test
# ===========================================================================

def test_evaluate_ablation_query_mock():
    """Verify evaluate_ablation_query computes adherence differences under controlled mock conditions."""
    client = QdrantClient(":memory:")
    graph = build_benchmark_call_graph(TEST_CORPUS)
    corpus_lookup = {item["func_name"]: item for item in TEST_CORPUS}

    # Patch hybrid_search and build_rag_prompt to run deterministically and fast
    q = LABELED_ABLATION_DATASET[0]

    with patch("eval.hallucination_eval.hybrid_search") as mock_hybrid, \
         patch("eval.hallucination_eval.build_rag_prompt") as mock_prompt:

        # Mock ON condition
        mock_prompt_on = MagicMock()
        mock_prompt_on.included_anchors = ["app/services/git_service.py::func::clone_repository"]
        mock_prompt_on.included_neighbors = [
            "app/services/git_service.py::func::cleanup_repo_directory (graph_callee)"
        ]

        # Mock OFF condition
        mock_prompt_off = MagicMock()
        mock_prompt_off.included_anchors = ["app/services/git_service.py::func::clone_repository"]
        mock_prompt_off.included_neighbors = []

        mock_hybrid.side_effect = [
            [{"func_name": "clone_repository", "source": "vector"}],  # ON
            [{"func_name": "clone_repository", "source": "vector"}],  # OFF
        ]
        mock_prompt.side_effect = [mock_prompt_on, mock_prompt_off]

        result = evaluate_ablation_query(
            question=q,
            client=client,
            graph=graph,
            corpus_lookup=corpus_lookup,
            top_k=2,
        )

        assert result.query_id == q.query_id
        assert result.adherence_score_on == 1.0  # Both citations grounded
        assert result.adherence_score_off == 0.5  # 1 grounded anchor, 1 hallucinated callee
        assert result.adherence_delta == 0.5
        assert result.hallucinations_prevented == 1
        assert len(result.hallucinated_citations_off) == 1
        assert len(result.hallucinated_citations_on) == 0


# ===========================================================================
# Report Rendering & Markdown Tests
# ===========================================================================

def test_markdown_results_table_formatting():
    """Verify that generate_markdown_results_table creates valid GFM table markdown."""
    sample_res = QueryAblationResult(
        query_id="ABL-01",
        query="Sample query?",
        category="Test Category",
        relationship_type="callee",
        primary_anchor="clone_repository",
        expected_collaborators=["cleanup_repo_directory"],
        anchors_on=["app/services/git_service.py::func::clone_repository"],
        neighbors_on=["app/services/git_service.py::func::cleanup_repo_directory"],
        total_citations_on=2,
        valid_citations_on=["[app/services/git_service.py::func::clone_repository]", "[app/services/git_service.py::func::cleanup_repo_directory]"],
        valid_neighbors_on=["[app/services/git_service.py::func::cleanup_repo_directory]"],
        hallucinated_citations_on=[],
        adherence_score_on=1.0,
        is_fully_grounded_on=True,
        anchors_off=["app/services/git_service.py::func::clone_repository"],
        neighbors_off=[],
        total_citations_off=2,
        valid_citations_off=["[app/services/git_service.py::func::clone_repository]"],
        hallucinated_citations_off=["[app/services/git_service.py::func::cleanup_repo_directory]"],
        adherence_score_off=0.5,
        is_fully_grounded_off=False,
        adherence_delta=0.5,
        hallucinations_prevented=1,
    )

    report = AblationBenchmarkReport(
        total_queries=1,
        macro_adherence_on=1.0,
        macro_adherence_off=0.5,
        adherence_gain=0.5,
        hallucination_rate_on=0.0,
        hallucination_rate_off=50.0,
        hallucination_reduction_pct=100.0,
        total_hallucinations_on=0,
        total_hallucinations_off=1,
        total_graph_citations_recovered=1,
        results=[sample_res],
    )

    table_md = generate_markdown_results_table(report)
    assert "| Query ID |" in table_md
    assert "| `ABL-01` |" in table_md
    assert "50.0%" in table_md
    assert "100.0%" in table_md
    assert "+50.0%" in table_md
    assert "**-1**" in table_md
    assert "| **MACRO** |" in table_md


def test_export_reports(tmp_path: Path):
    """Verify that export_json_report and export_markdown_report persist valid files."""
    report = AblationBenchmarkReport(
        total_queries=1,
        macro_adherence_on=1.0,
        macro_adherence_off=0.5,
        adherence_gain=0.5,
        hallucination_rate_on=0.0,
        hallucination_rate_off=50.0,
        hallucination_reduction_pct=100.0,
        total_hallucinations_on=0,
        total_hallucinations_off=1,
        total_graph_citations_recovered=1,
        results=[],
    )

    json_path = tmp_path / "test_report.json"
    md_path = tmp_path / "test_report.md"

    export_json_report(report, json_path)
    export_markdown_report(report, md_path)

    assert json_path.exists()
    assert md_path.exists()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["summary"]["macro_adherence_on"] == 1.0
    assert data["summary"]["macro_adherence_off"] == 0.5

    md_text = md_path.read_text(encoding="utf-8")
    assert "# Hallucination Comparison & RAG Pipeline Ablation Benchmark" in md_text
    assert "Ablation Testing Methodology" in md_text or "Ablation Testing" in md_text

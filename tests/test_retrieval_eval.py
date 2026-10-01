"""
tests/test_retrieval_eval.py
----------------------------
Unit tests for the Quantitative Retrieval Evaluation Harness (eval/retrieval_eval.py).
Validates IR metrics (Precision@k, Recall@k, F1@k, MRR), dataset integrity,
mocked query evaluation, and report serialization.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from eval.retrieval_eval import (
    GROUND_TRUTH_DATASET,
    TEST_CORPUS,
    EvaluationItem,
    QueryEvalResult,
    RetrievalEvaluationReport,
    RetrievedItem,
    calculate_f1_at_k,
    calculate_precision_at_k,
    calculate_recall_at_k,
    calculate_reciprocal_rank,
    evaluate_single_query,
    export_json_report,
    export_markdown_report,
    render_terminal_report,
    run_retrieval_evaluation,
)


# ===========================================================================
# Metric Calculation Tests
# ===========================================================================

def test_precision_at_k():
    """Verify Precision@k calculations for varying intersections and k cutoffs."""
    # Single target found in top-3 -> 1/3
    assert calculate_precision_at_k(["funcA", "funcB", "funcC"], ["funcA"], k=3) == pytest.approx(1 / 3)

    # Two targets found in top-3 -> 2/3
    assert calculate_precision_at_k(["funcA", "funcB", "funcC"], ["funcA", "funcB"], k=3) == pytest.approx(2 / 3)

    # Three targets found in top-3 -> 3/3 = 1.0
    assert calculate_precision_at_k(["funcA", "funcB", "funcC"], ["funcA", "funcB", "funcC"], k=3) == 1.0

    # No targets found -> 0.0
    assert calculate_precision_at_k(["funcX", "funcY", "funcZ"], ["funcA"], k=3) == 0.0

    # Edge cases
    assert calculate_precision_at_k([], ["funcA"], k=3) == 0.0
    assert calculate_precision_at_k(["funcA"], ["funcA"], k=0) == 0.0


def test_recall_at_k():
    """Verify Recall@k calculations for single and multi-target queries."""
    # Single target in top-3 -> 1/1 = 1.0
    assert calculate_recall_at_k(["funcA", "funcB", "funcC"], ["funcA"], k=3) == 1.0

    # Target at rank 3 -> still 1.0
    assert calculate_recall_at_k(["funcB", "funcC", "funcA"], ["funcA"], k=3) == 1.0

    # Target outside top-k cutoff
    assert calculate_recall_at_k(["funcB", "funcC", "funcD", "funcA"], ["funcA"], k=3) == 0.0

    # Multi-target: 1 of 2 retrieved in top-3 -> 0.5
    assert calculate_recall_at_k(["funcA", "funcX", "funcY"], ["funcA", "funcB"], k=3) == 0.5

    # Multi-target: 2 of 2 retrieved in top-3 -> 1.0
    assert calculate_recall_at_k(["funcA", "funcB", "funcY"], ["funcA", "funcB"], k=3) == 1.0

    # Empty ground truth handled safely
    assert calculate_recall_at_k(["funcA"], [], k=3) == 1.0


def test_f1_at_k():
    """Verify F1@k harmonic mean calculation."""
    # Both 1.0 -> 1.0
    assert calculate_f1_at_k(1.0, 1.0) == 1.0

    # Precision 1/3, Recall 1.0 -> 2 * (1/3 * 1) / (1/3 + 1) = (2/3) / (4/3) = 0.5
    p = 1 / 3
    r = 1.0
    assert calculate_f1_at_k(p, r) == pytest.approx(0.5)

    # Zeros
    assert calculate_f1_at_k(0.0, 0.0) == 0.0
    assert calculate_f1_at_k(0.0, 1.0) == 0.0


def test_reciprocal_rank():
    """Verify Reciprocal Rank computation by position of first relevant hit."""
    assert calculate_reciprocal_rank(["funcA", "funcB", "funcC"], ["funcA"]) == 1.0
    assert calculate_reciprocal_rank(["funcB", "funcA", "funcC"], ["funcA"]) == 0.5
    assert calculate_reciprocal_rank(["funcB", "funcC", "funcA"], ["funcA"]) == pytest.approx(1 / 3)
    assert calculate_reciprocal_rank(["funcX", "funcY", "funcZ"], ["funcA"]) == 0.0


# ===========================================================================
# Benchmark Dataset & Corpus Integrity Tests
# ===========================================================================

def test_ground_truth_dataset_integrity():
    """Ensure benchmark dataset contains 20 well-formed items and corpus covers all expected targets."""
    assert len(GROUND_TRUTH_DATASET) == 20

    corpus_funcs = {c["func_name"] for c in TEST_CORPUS}
    query_ids = set()

    for item in GROUND_TRUTH_DATASET:
        assert item.query_id not in query_ids, f"Duplicate query ID: {item.query_id}"
        query_ids.add(item.query_id)
        assert len(item.query.strip()) > 10
        assert len(item.expected_functions) >= 1
        assert len(item.expected_files) >= 1
        assert item.category != ""
        assert item.difficulty in ("direct", "synonym", "conceptual", "cross-cutting")

        # Every expected function must exist in our test corpus!
        for func in item.expected_functions:
            assert func in corpus_funcs, f"Expected function '{func}' from {item.query_id} not in test corpus!"


# ===========================================================================
# Single Query Evaluation Tests (Mocked)
# ===========================================================================

def test_evaluate_single_query_mocked():
    """Test evaluate_single_query metrics and diagnostics with mocked retrieval."""
    item = EvaluationItem(
        query_id="QTEST",
        query="Mock test question",
        expected_functions=["funcA"],
        expected_files=["app/test.py"],
        category="Testing",
        difficulty="direct",
        rationale="Testing rationale",
    )

    mock_raw = [
        {"rank": 1, "func_name": "funcA", "file_path": "app/test.py", "score": 0.85},
        {"rank": 2, "func_name": "funcB", "file_path": "app/other.py", "score": 0.65},
        {"rank": 3, "func_name": "funcC", "file_path": "app/another.py", "score": 0.40},
    ]

    with patch("eval.retrieval_eval.execute_vector_retrieval", return_value=mock_raw):
        mock_client = MagicMock()
        res = evaluate_single_query(item, client=mock_client, collection_name="test_col", k=3)

        assert res.query_id == "QTEST"
        assert res.precision_at_k == pytest.approx(1 / 3, abs=1e-4)
        assert res.recall_at_k == 1.0
        assert res.f1_at_k == pytest.approx(0.5, abs=1e-4)
        assert res.hit_at_k is True
        assert res.top1_match is True
        assert res.reciprocal_rank == 1.0
        assert res.confidence_gap == pytest.approx(0.20, abs=1e-4)
        assert res.false_positives == ["funcB", "funcC"]
        assert res.false_negatives == []


# ===========================================================================
# Reporting and Serialization Tests
# ===========================================================================

def test_render_terminal_and_export_reports(tmp_path: Path):
    """Test terminal formatting, Markdown export, and JSON serialization."""
    item_res = QueryEvalResult(
        query_id="Q01",
        query="Test query",
        category="Repo",
        difficulty="direct",
        expected_functions=["funcA"],
        retrieved_functions=["funcA", "funcB", "funcC"],
        retrieved_items=[
            RetrievedItem(1, "funcA", "fileA.py", 0.82, True),
            RetrievedItem(2, "funcB", "fileB.py", 0.61, False),
            RetrievedItem(3, "funcC", "fileC.py", 0.42, False),
        ],
        precision_at_k=1 / 3,
        recall_at_k=1.0,
        f1_at_k=0.5,
        hit_at_k=True,
        reciprocal_rank=1.0,
        top1_match=True,
        false_positives=["funcB", "funcC"],
        false_negatives=[],
        confidence_gap=0.21,
    )

    report = RetrievalEvaluationReport(
        k=3,
        total_queries=1,
        macro_precision_at_k=1 / 3,
        macro_recall_at_k=1.0,
        macro_f1_at_k=0.5,
        hit_rate_at_k=1.0,
        mean_reciprocal_rank=1.0,
        top1_accuracy=1.0,
        category_breakdown={"Repo": {"count": 1.0, "precision_at_k": 1 / 3, "recall_at_k": 1.0, "hit_rate": 1.0, "mrr": 1.0}},
        difficulty_breakdown={"direct": {"count": 1.0, "precision_at_k": 1 / 3, "recall_at_k": 1.0, "hit_rate": 1.0, "mrr": 1.0}},
        query_results=[item_res],
    )

    term_output = render_terminal_report(report)
    assert "QUANTITATIVE RETRIEVAL EVALUATION HARNESS" in term_output
    assert "Macro Precision@3" in term_output
    assert "Macro Recall@3" in term_output

    # Test Markdown export
    md_file = tmp_path / "test_report.md"
    export_markdown_report(report, md_file)
    assert md_file.exists()
    content = md_file.read_text(encoding="utf-8")
    assert "Quantitative Retrieval Evaluation Report" in content
    assert "The Precision@k vs Recall@k Dilemma" in content

    # Test JSON export
    json_file = tmp_path / "test_report.json"
    export_json_report(report, json_file)
    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert data["k"] == 3
    assert data["total_queries"] == 1
    assert data["macro_recall_at_k"] == 1.0

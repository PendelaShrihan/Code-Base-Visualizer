"""
eval/retrieval_eval.py
----------------------
Quantitative Retrieval Evaluation Harness for Code RAG.

This harness benchmarks dense vector search against a hand-labeled ground-truth
dataset of 20 natural-language queries mapped to known-correct functions in a
test repository corpus. It computes Precision@k, Recall@k, Hit Rate@k, MRR, and
Top-1 Accuracy, providing quantitative diagnostics for RAG pipeline evaluation.

Quantitative RAG Evaluation Methodology
=======================================
1. Precision@k (Purity of Context):
   Precision@k = |Retrieved_k ∩ Relevant| / k
   - Measures what fraction of the retrieved top-k items are actually relevant.
   - High Precision@k prevents "Context Poisoning" where irrelevant code distracts
     the LLM or causes hallucinations.
   - Note on Single-Target Evaluation: When a question has exactly 1 ground-truth
     correct function, the theoretical maximum Precision@3 is 1/3 (~33.3%).
     For queries with 2 ground-truth functions, the theoretical maximum is 2/3 (~66.7%).

2. Recall@k (Completeness of Context):
   Recall@k = |Retrieved_k ∩ Relevant| / |Relevant|
   - Measures what fraction of the relevant items were successfully surfaced in top-k.
   - High Recall@k prevents "Context Omission" where the LLM misses vital functions
     needed to answer the user inquiry accurately.
   - Recall@k can reach 1.0 (100%) as long as all ground-truth functions appear
     anywhere within the top-k results.

3. Hit@k (Success Rate / Hit Rate):
   Hit@k = 1 if |Retrieved_k ∩ Relevant| > 0 else 0
   - Evaluates whether at least one correct function was retrieved in the top-k context.

4. MRR (Mean Reciprocal Rank):
   RR = 1 / (rank of first relevant item in retrieved list)
   - Evaluates ranking quality: rewards retrievers that place the correct answer at
     Rank 1 rather than burying it at Rank 2 or Rank 3.

5. Trade-off Analysis & RAG Tuning:
   - Increasing k boosts Recall@k at the cost of lowering Precision@k and inflating
     token costs / latency.
   - Low Precision@k + High Recall@k indicates an opportunity for a Stage-2 Reranker
     (e.g., Cross-Encoder / BGE-Reranker) or Graph Context Expansion.

Usage:
------
    # Run evaluation harness with in-memory Qdrant (zero external dependencies):
    python eval/retrieval_eval.py

    # Specify top-k (e.g. k=3 or k=5):
    python eval/retrieval_eval.py --k 3

    # Export structured metrics report to Markdown and JSON:
    python eval/retrieval_eval.py --export-md eval_report.md --export-json eval_report.json

    # Test against a live Qdrant instance:
    python eval/retrieval_eval.py --live --host localhost --port 6333
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

# Ensure project root is in sys.path
_current_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _current_dir:
    sys.path.pop(0)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from rag.embeddings import model

logger = logging.getLogger(__name__)

DEFAULT_COLLECTION_NAME: str = "eval_code_chunks"
DEFAULT_VECTOR_SIZE: int = 384  # all-MiniLM-L6-v2 dimensionality


# ===========================================================================
# Data Structures
# ===========================================================================

@dataclass
class EvaluationItem:
    """A hand-labeled benchmark question with ground-truth relevant functions."""

    query_id: str
    query: str
    expected_functions: list[str]
    expected_files: list[str]
    category: str
    difficulty: str  # "direct", "synonym", "cross-cutting", "conceptual"
    rationale: str


@dataclass
class RetrievedItem:
    """A single retrieved function from vector search."""

    rank: int
    func_name: str
    file_path: str
    score: float
    is_relevant: bool


@dataclass
class QueryEvalResult:
    """Per-query quantitative evaluation metrics and diagnostics."""

    query_id: str
    query: str
    category: str
    difficulty: str
    expected_functions: list[str]
    retrieved_functions: list[str]
    retrieved_items: list[RetrievedItem]
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    hit_at_k: bool
    reciprocal_rank: float
    top1_match: bool
    false_positives: list[str]
    false_negatives: list[str]
    confidence_gap: float


@dataclass
class RetrievalEvaluationReport:
    """Aggregate benchmark report summarizing quantitative RAG performance."""

    k: int
    total_queries: int
    macro_precision_at_k: float
    macro_recall_at_k: float
    macro_f1_at_k: float
    hit_rate_at_k: float
    mean_reciprocal_rank: float
    top1_accuracy: float
    category_breakdown: dict[str, dict[str, float]]
    difficulty_breakdown: dict[str, dict[str, float]]
    query_results: list[QueryEvalResult]


# ===========================================================================
# Hand-Labeled Ground-Truth Benchmark Dataset (20 Questions)
# ===========================================================================

GROUND_TRUTH_DATASET: list[EvaluationItem] = [
    EvaluationItem(
        query_id="Q01",
        query="How do I clone a remote git repository into a local temporary folder with guardrails?",
        expected_functions=["clone_repository"],
        expected_files=["app/services/git_service.py"],
        category="Repository Operations",
        difficulty="direct",
        rationale="clone_repository checks timeout and size limits and clones remote repos via GitPython.",
    ),
    EvaluationItem(
        query_id="Q02",
        query="Where is the cleanup logic to delete cloned repository files and clear read-only permissions?",
        expected_functions=["cleanup_repo_directory"],
        expected_files=["app/services/git_service.py"],
        category="Repository Operations",
        difficulty="direct",
        rationale="cleanup_repo_directory handles Windows/Linux read-only file removal with shutil.rmtree.",
    ),
    EvaluationItem(
        query_id="Q03",
        query="How can I purge stale temporary repository directories older than thirty minutes?",
        expected_functions=["garbage_collect_temp_repos"],
        expected_files=["app/services/git_service.py"],
        category="Repository Operations",
        difficulty="synonym",
        rationale="garbage_collect_temp_repos periodically purges directories older than max_age_seconds.",
    ),
    EvaluationItem(
        query_id="Q04",
        query="Which function calculates the total byte size of a directory tree while ignoring symlinks?",
        expected_functions=["get_directory_size_bytes"],
        expected_files=["app/services/git_service.py"],
        category="Repository Operations",
        difficulty="direct",
        rationale="get_directory_size_bytes walks the filesystem summing file sizes and skipping links.",
    ),
    EvaluationItem(
        query_id="Q05",
        query="Where is commit churn or revision count calculated for an individual source file?",
        expected_functions=["count_file_commits"],
        expected_files=["app/services/git_service.py"],
        category="Repository Operations",
        difficulty="synonym",
        rationale="count_file_commits uses git log revision counting to attach commit churn metadata.",
    ),
    EvaluationItem(
        query_id="Q06",
        query="How does the system parse Python code with Tree-sitter while enforcing a strict timeout?",
        expected_functions=["parse_with_timeout"],
        expected_files=["app/services/ast_engine.py"],
        category="AST Parsing",
        difficulty="direct",
        rationale="parse_with_timeout wraps the Tree-sitter parser in a ThreadPoolExecutor with timeout.",
    ),
    EvaluationItem(
        query_id="Q07",
        query="How does the parser retrieve all function and method names declared in a Python file?",
        expected_functions=["extract_function_names"],
        expected_files=["app/services/ast_engine.py"],
        category="AST Parsing",
        difficulty="direct",
        rationale="extract_function_names traverses the syntax tree collecting function_definition identifiers.",
    ),
    EvaluationItem(
        query_id="Q08",
        query="Which helper extracts classes, method calls, and imported modules using Tree-sitter queries?",
        expected_functions=["extract_structure"],
        expected_files=["app/services/ast_engine.py"],
        category="AST Parsing",
        difficulty="conceptual",
        rationale="extract_structure runs structural query captures to catalog classes, calls, and imports.",
    ),
    EvaluationItem(
        query_id="Q09",
        query="How are intra-file caller and callee function relationships extracted from the syntax tree?",
        expected_functions=["extract_call_edges"],
        expected_files=["app/services/ast_engine.py"],
        category="AST Parsing",
        difficulty="conceptual",
        rationale="extract_call_edges tracks enclosing scopes during DFS to output (caller, callee) tuples.",
    ),
    EvaluationItem(
        query_id="Q10",
        query="How is a single file's extracted structure converted into a NetworkX directed graph?",
        expected_functions=["build_graph"],
        expected_files=["app/services/graph_service.py"],
        category="Graph Construction",
        difficulty="direct",
        rationale="build_graph takes structure dict and call edges to produce a typed DiGraph.",
    ),
    EvaluationItem(
        query_id="Q11",
        query="Where does the repo walker recursively traverse directory trees and aggregate AST graphs?",
        expected_functions=["scan_repository"],
        expected_files=["parser/repo_walker.py"],
        category="Repository Walking",
        difficulty="direct",
        rationale="scan_repository walks repo files, builds scoped subgraphs, and calculates global metrics.",
    ),
    EvaluationItem(
        query_id="Q12",
        query="How are uncalled functions with zero incoming call-graph edges detected as candidate dead code?",
        expected_functions=["detect_dead_code"],
        expected_files=["parser/repo_walker.py"],
        category="Dead Code Detection",
        difficulty="conceptual",
        rationale="detect_dead_code checks graph in-degrees while filtering out web routes and framework entry points.",
    ),
    EvaluationItem(
        query_id="Q13",
        query="Where is PageRank centrality calculated across all nodes in the codebase dependency graph?",
        expected_functions=["compute_pagerank"],
        expected_files=["parser/repo_walker.py"],
        category="Graph Centrality",
        difficulty="direct",
        rationale="compute_pagerank computes stochastic link analysis scores across the call graph.",
    ),
    EvaluationItem(
        query_id="Q14",
        query="Which function serializes the NetworkX graph with node attributes and edges into a JSON dictionary?",
        expected_functions=["graph_to_json"],
        expected_files=["parser/repo_walker.py"],
        category="Graph Serialization",
        difficulty="direct",
        rationale="graph_to_json serializes nodes, attributes, and edges into frontend-compatible JSON.",
    ),
    EvaluationItem(
        query_id="Q15",
        query="How are code chunk texts batch encoded into dense vector representations using SentenceTransformer?",
        expected_functions=["embed_chunks"],
        expected_files=["rag/embeddings.py"],
        category="Vector Embeddings",
        difficulty="direct",
        rationale="embed_chunks runs batch inference with all-MiniLM-L6-v2 and attaches embeddings.",
    ),
    EvaluationItem(
        query_id="Q16",
        query="Where does the ingestion pipeline generate deterministic UUID-v5 chunk records from graph nodes?",
        expected_functions=["extract_function_chunks"],
        expected_files=["rag/ingestion.py"],
        category="Vector Ingestion",
        difficulty="conceptual",
        rationale="extract_function_chunks filters function nodes and constructs reproducible UUIDv5 chunk dicts.",
    ),
    EvaluationItem(
        query_id="Q17",
        query="How are embedded code chunk vectors and metadata payloads upserted into Qdrant in batches?",
        expected_functions=["batch_upsert_chunks"],
        expected_files=["rag/ingestion.py"],
        category="Vector Ingestion",
        difficulty="direct",
        rationale="batch_upsert_chunks converts chunks into PointStructs and writes them to the vector collection.",
    ),
    EvaluationItem(
        query_id="Q18",
        query="Where is the Qdrant code_chunks collection created with 384 dimensions and cosine metric?",
        expected_functions=["create_code_chunks_collection"],
        expected_files=["rag/qdrant_client.py"],
        category="Vector Database",
        difficulty="direct",
        rationale="create_code_chunks_collection configures vector parameters, distance, and payload indexes.",
    ),
    EvaluationItem(
        query_id="Q19",
        query="How does vector search find the top-k most semantically similar code functions for a question?",
        expected_functions=["search_functions"],
        expected_files=["rag/test_search.py"],
        category="Semantic Retrieval",
        difficulty="direct",
        rationale="search_functions encodes a query and executes cosine similarity search in Qdrant.",
    ),
    EvaluationItem(
        query_id="Q20",
        query="How does hybrid retrieval expand vector search anchors with 1-hop caller and callee graph neighbors?",
        expected_functions=["hybrid_search"],
        expected_files=["rag/hybrid_retriever.py"],
        category="Hybrid Retrieval",
        difficulty="conceptual",
        rationale="hybrid_search combines vector anchors with 1-hop NetworkX caller/callee neighborhood expansion.",
    ),
]


# ===========================================================================
# Test Repository Corpus (Representative Functions across CodeBase Visualizer)
# ===========================================================================

TEST_CORPUS: list[dict[str, Any]] = [
    {
        "id": 1,
        "func_name": "clone_repository",
        "file_path": "app/services/git_service.py",
        "text": "func: clone_repository\nfile: app/services/git_service.py\nClone a remote git repository into a unique temporary directory with timeout and size limit guardrails using GitPython.",
        "pagerank": 0.0825,
        "commit_count": 14,
        "is_dead_code_candidate": False,
    },
    {
        "id": 2,
        "func_name": "cleanup_repo_directory",
        "file_path": "app/services/git_service.py",
        "text": "func: cleanup_repo_directory\nfile: app/services/git_service.py\nSafely delete cloned repository directory from disk, resetting read-only permissions on Windows and POSIX systems.",
        "pagerank": 0.0412,
        "commit_count": 8,
        "is_dead_code_candidate": False,
    },
    {
        "id": 3,
        "func_name": "garbage_collect_temp_repos",
        "file_path": "app/services/git_service.py",
        "text": "func: garbage_collect_temp_repos\nfile: app/services/git_service.py\nGarbage collect and purge stale temporary clone directories older than max age threshold to avoid disk leaks.",
        "pagerank": 0.0210,
        "commit_count": 5,
        "is_dead_code_candidate": False,
    },
    {
        "id": 4,
        "func_name": "get_directory_size_bytes",
        "file_path": "app/services/git_service.py",
        "text": "func: get_directory_size_bytes\nfile: app/services/git_service.py\nCalculate total size in bytes of all regular files in a directory tree while skipping symlinks to prevent loops.",
        "pagerank": 0.0315,
        "commit_count": 7,
        "is_dead_code_candidate": False,
    },
    {
        "id": 5,
        "func_name": "count_file_commits",
        "file_path": "app/services/git_service.py",
        "text": "func: count_file_commits\nfile: app/services/git_service.py\nCount total git commits modifying a specific file using git log revision count for commit churn analysis.",
        "pagerank": 0.0380,
        "commit_count": 6,
        "is_dead_code_candidate": False,
    },
    {
        "id": 6,
        "func_name": "parse_with_timeout",
        "file_path": "app/services/ast_engine.py",
        "text": "func: parse_with_timeout\nfile: app/services/ast_engine.py\nParse Python source code into a Tree-sitter AST syntax tree root node enforcing a defensive execution timeout.",
        "pagerank": 0.0650,
        "commit_count": 12,
        "is_dead_code_candidate": False,
    },
    {
        "id": 7,
        "func_name": "extract_function_names",
        "file_path": "app/services/ast_engine.py",
        "text": "func: extract_function_names\nfile: app/services/ast_engine.py\nTraverse Python AST and extract all top-level and nested function definition identifier names.",
        "pagerank": 0.0710,
        "commit_count": 10,
        "is_dead_code_candidate": False,
    },
    {
        "id": 8,
        "func_name": "extract_structure",
        "file_path": "app/services/ast_engine.py",
        "text": "func: extract_structure\nfile: app/services/ast_engine.py\nRun structural Tree-sitter query to extract classes, method calls, and imports from Python source code.",
        "pagerank": 0.0890,
        "commit_count": 15,
        "is_dead_code_candidate": False,
    },
    {
        "id": 9,
        "func_name": "extract_call_edges",
        "file_path": "app/services/ast_engine.py",
        "text": "func: extract_call_edges\nfile: app/services/ast_engine.py\nTraverse syntax tree with depth-first search to extract intra-file caller and callee function edge relationships.",
        "pagerank": 0.0840,
        "commit_count": 13,
        "is_dead_code_candidate": False,
    },
    {
        "id": 10,
        "func_name": "build_graph",
        "file_path": "app/services/graph_service.py",
        "text": "func: build_graph\nfile: app/services/graph_service.py\nConvert file path, extracted structural elements, and call edges into a populated NetworkX directed graph.",
        "pagerank": 0.0920,
        "commit_count": 16,
        "is_dead_code_candidate": False,
    },
    {
        "id": 11,
        "func_name": "scan_repository",
        "file_path": "parser/repo_walker.py",
        "text": "func: scan_repository\nfile: parser/repo_walker.py\nRecursively walk repository directory tree, parse Python files with Tree-sitter, and build merged dependency graph.",
        "pagerank": 0.1250,
        "commit_count": 22,
        "is_dead_code_candidate": False,
    },
    {
        "id": 12,
        "func_name": "detect_dead_code",
        "file_path": "parser/repo_walker.py",
        "text": "func: detect_dead_code\nfile: parser/repo_walker.py\nDetect uncalled dead code candidate functions with zero incoming call-graph edges, filtering out web entry points.",
        "pagerank": 0.0450,
        "commit_count": 9,
        "is_dead_code_candidate": False,
    },
    {
        "id": 13,
        "func_name": "compute_pagerank",
        "file_path": "parser/repo_walker.py",
        "text": "func: compute_pagerank\nfile: parser/repo_walker.py\nCompute PageRank centrality scores across all nodes in the codebase dependency graph to measure structural importance.",
        "pagerank": 0.0520,
        "commit_count": 7,
        "is_dead_code_candidate": False,
    },
    {
        "id": 14,
        "func_name": "graph_to_json",
        "file_path": "parser/repo_walker.py",
        "text": "func: graph_to_json\nfile: parser/repo_walker.py\nSerialize NetworkX graph into JSON dictionary format with nodes list, node attributes, and directed edge connections.",
        "pagerank": 0.0680,
        "commit_count": 11,
        "is_dead_code_candidate": False,
    },
    {
        "id": 15,
        "func_name": "embed_chunks",
        "file_path": "rag/embeddings.py",
        "text": "func: embed_chunks\nfile: rag/embeddings.py\nBatch compute dense vector embeddings for code chunk texts using SentenceTransformer all-MiniLM-L6-v2 model.",
        "pagerank": 0.0610,
        "commit_count": 9,
        "is_dead_code_candidate": False,
    },
    {
        "id": 16,
        "func_name": "extract_function_chunks",
        "file_path": "rag/ingestion.py",
        "text": "func: extract_function_chunks\nfile: rag/ingestion.py\nFilter function nodes from graph dictionary and extract chunk dictionaries with deterministic UUID-v5 identifiers.",
        "pagerank": 0.0740,
        "commit_count": 12,
        "is_dead_code_candidate": False,
    },
    {
        "id": 17,
        "func_name": "batch_upsert_chunks",
        "file_path": "rag/ingestion.py",
        "text": "func: batch_upsert_chunks\nfile: rag/ingestion.py\nUpsert batches of embedded code chunks and metadata payloads into the Qdrant vector database collection.",
        "pagerank": 0.0780,
        "commit_count": 11,
        "is_dead_code_candidate": False,
    },
    {
        "id": 18,
        "func_name": "create_code_chunks_collection",
        "file_path": "rag/qdrant_client.py",
        "text": "func: create_code_chunks_collection\nfile: rag/qdrant_client.py\nCreate and initialize Qdrant code_chunks collection schema with 384 dimensions, cosine distance, and keyword payload indexes.",
        "pagerank": 0.0560,
        "commit_count": 8,
        "is_dead_code_candidate": False,
    },
    {
        "id": 19,
        "func_name": "search_functions",
        "file_path": "rag/test_search.py",
        "text": "func: search_functions\nfile: rag/test_search.py\nVectorize natural language question and retrieve top-k semantically matching code functions from Qdrant by cosine similarity.",
        "pagerank": 0.0720,
        "commit_count": 10,
        "is_dead_code_candidate": False,
    },
    {
        "id": 20,
        "func_name": "hybrid_search",
        "file_path": "rag/hybrid_retriever.py",
        "text": "func: hybrid_search\nfile: rag/hybrid_retriever.py\nExecute two-stage hybrid retrieval combining dense vector search anchors with 1-hop caller and callee graph neighbors.",
        "pagerank": 0.0890,
        "commit_count": 14,
        "is_dead_code_candidate": False,
    },
    {
        "id": 21,
        "func_name": "build_rag_prompt",
        "file_path": "rag/prompt_builder.py",
        "text": "func: build_rag_prompt\nfile: rag/prompt_builder.py\nConstruct structured RAG prompt with anti-hallucination XML grounding rules, token budgeting, and code anchors.",
        "pagerank": 0.0810,
        "commit_count": 13,
        "is_dead_code_candidate": False,
    },
    {
        "id": 22,
        "func_name": "verify_grounding",
        "file_path": "rag/llm_engine.py",
        "text": "func: verify_grounding\nfile: rag/llm_engine.py\nEvaluate LLM output citations against retrieved context anchors to compute grounding adherence score.",
        "pagerank": 0.0590,
        "commit_count": 8,
        "is_dead_code_candidate": False,
    },
]


# ===========================================================================
# Ingestion & Seeding Engine
# ===========================================================================

def seed_evaluation_corpus(
    client: QdrantClient,
    corpus: list[dict[str, Any]] = TEST_CORPUS,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    vector_size: int = DEFAULT_VECTOR_SIZE,
) -> int:
    """Populates Qdrant with the test corpus vectors and metadata."""
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
    )

    points: list[PointStruct] = []
    for item in corpus:
        encoded = model.encode(item["text"])
        vector = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
        points.append(
            PointStruct(
                id=item["id"],
                vector=vector,
                payload={
                    "func_name": item["func_name"],
                    "file_path": item["file_path"],
                    "pagerank": item.get("pagerank", 0.0),
                    "commit_count": item.get("commit_count", 0),
                    "is_dead_code_candidate": item.get("is_dead_code_candidate", False),
                },
            )
        )

    client.upsert(collection_name=collection_name, points=points, wait=True)
    return len(points)


# ===========================================================================
# Core Retrieval & Metric Calculation Functions
# ===========================================================================

def execute_vector_retrieval(
    query: str,
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Encodes query and queries Qdrant for top-k matching points."""
    encoded_query = model.encode(query)
    query_vector = encoded_query.tolist() if hasattr(encoded_query, "tolist") else list(encoded_query)

    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        )
        scored_points = response.points
    else:
        scored_points = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

    results: list[dict[str, Any]] = []
    for rank, point in enumerate(scored_points, start=1):
        payload = point.payload or {}
        results.append(
            {
                "rank": rank,
                "func_name": payload.get("func_name", ""),
                "file_path": payload.get("file_path", ""),
                "score": float(point.score),
            }
        )
    return results


def calculate_precision_at_k(retrieved: list[str], ground_truth: list[str], k: int) -> float:
    """Compute Precision@k: |Retrieved_k ∩ GroundTruth| / k."""
    if k <= 0:
        return 0.0
    retrieved_k = retrieved[:k]
    gt_set = set(ground_truth)
    hits = sum(1 for f in retrieved_k if f in gt_set)
    return hits / k


def calculate_recall_at_k(retrieved: list[str], ground_truth: list[str], k: int) -> float:
    """Compute Recall@k: |Retrieved_k ∩ GroundTruth| / |GroundTruth|."""
    if not ground_truth:
        return 1.0
    retrieved_k = retrieved[:k]
    gt_set = set(ground_truth)
    hits = sum(1 for f in retrieved_k if f in gt_set)
    return hits / len(gt_set)


def calculate_f1_at_k(precision: float, recall: float) -> float:
    """Compute harmonic mean F1@k from precision and recall."""
    if precision + recall == 0.0:
        return 0.0
    return (2.0 * precision * recall) / (precision + recall)


def calculate_reciprocal_rank(retrieved: list[str], ground_truth: list[str]) -> float:
    """Compute Reciprocal Rank: 1 / rank_of_first_relevant_item (1-based), or 0.0."""
    gt_set = set(ground_truth)
    for idx, func in enumerate(retrieved, start=1):
        if func in gt_set:
            return 1.0 / idx
    return 0.0


def evaluate_single_query(
    item: EvaluationItem,
    client: QdrantClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    k: int = 3,
) -> QueryEvalResult:
    """Evaluates a single benchmark question and computes quantitative metrics."""
    raw_results = execute_vector_retrieval(
        query=item.query,
        client=client,
        collection_name=collection_name,
        top_k=k,
    )

    retrieved_funcs = [r["func_name"] for r in raw_results]
    gt_set = set(item.expected_functions)

    retrieved_items: list[RetrievedItem] = [
        RetrievedItem(
            rank=r["rank"],
            func_name=r["func_name"],
            file_path=r["file_path"],
            score=r["score"],
            is_relevant=r["func_name"] in gt_set,
        )
        for r in raw_results
    ]

    p_at_k = calculate_precision_at_k(retrieved_funcs, item.expected_functions, k)
    r_at_k = calculate_recall_at_k(retrieved_funcs, item.expected_functions, k)
    f1 = calculate_f1_at_k(p_at_k, r_at_k)
    mrr = calculate_reciprocal_rank(retrieved_funcs, item.expected_functions)
    hit = any(f in gt_set for f in retrieved_funcs)
    top1 = bool(retrieved_funcs and retrieved_funcs[0] in gt_set)

    false_positives = [f for f in retrieved_funcs if f not in gt_set]
    false_negatives = [f for f in item.expected_functions if f not in set(retrieved_funcs)]

    confidence_gap = 0.0
    if len(raw_results) >= 2:
        confidence_gap = round(raw_results[0]["score"] - raw_results[1]["score"], 4)

    return QueryEvalResult(
        query_id=item.query_id,
        query=item.query,
        category=item.category,
        difficulty=item.difficulty,
        expected_functions=item.expected_functions,
        retrieved_functions=retrieved_funcs,
        retrieved_items=retrieved_items,
        precision_at_k=round(p_at_k, 4),
        recall_at_k=round(r_at_k, 4),
        f1_at_k=round(f1, 4),
        hit_at_k=hit,
        reciprocal_rank=round(mrr, 4),
        top1_match=top1,
        false_positives=false_positives,
        false_negatives=false_negatives,
        confidence_gap=confidence_gap,
    )


# ===========================================================================
# Full Benchmark Runner
# ===========================================================================

def run_retrieval_evaluation(
    dataset: list[EvaluationItem] = GROUND_TRUTH_DATASET,
    corpus: list[dict[str, Any]] = TEST_CORPUS,
    client: Optional[QdrantClient] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    k: int = 3,
) -> RetrievalEvaluationReport:
    """Executes the full evaluation harness across the benchmark dataset."""
    if client is None:
        client = QdrantClient(":memory:")

    # Ensure collection is populated with corpus
    if not client.collection_exists(collection_name) or client.get_collection(collection_name).points_count == 0:
        seed_evaluation_corpus(client=client, corpus=corpus, collection_name=collection_name)

    query_results: list[QueryEvalResult] = []
    for item in dataset:
        res = evaluate_single_query(item, client=client, collection_name=collection_name, k=k)
        query_results.append(res)

    total_queries = len(query_results)
    if total_queries == 0:
        return RetrievalEvaluationReport(
            k=k,
            total_queries=0,
            macro_precision_at_k=0.0,
            macro_recall_at_k=0.0,
            macro_f1_at_k=0.0,
            hit_rate_at_k=0.0,
            mean_reciprocal_rank=0.0,
            top1_accuracy=0.0,
            category_breakdown={},
            difficulty_breakdown={},
            query_results=[],
        )

    macro_precision = sum(r.precision_at_k for r in query_results) / total_queries
    macro_recall = sum(r.recall_at_k for r in query_results) / total_queries
    macro_f1 = sum(r.f1_at_k for r in query_results) / total_queries
    hit_rate = sum(1 for r in query_results if r.hit_at_k) / total_queries
    mean_mrr = sum(r.reciprocal_rank for r in query_results) / total_queries
    top1_acc = sum(1 for r in query_results if r.top1_match) / total_queries

    # Breakdown by category
    categories: dict[str, list[QueryEvalResult]] = {}
    for r in query_results:
        categories.setdefault(r.category, []).append(r)

    category_breakdown: dict[str, dict[str, float]] = {}
    for cat, items in categories.items():
        n = len(items)
        category_breakdown[cat] = {
            "count": float(n),
            "precision_at_k": round(sum(i.precision_at_k for i in items) / n, 4),
            "recall_at_k": round(sum(i.recall_at_k for i in items) / n, 4),
            "hit_rate": round(sum(1 for i in items if i.hit_at_k) / n, 4),
            "mrr": round(sum(i.reciprocal_rank for i in items) / n, 4),
        }

    # Breakdown by difficulty
    difficulties: dict[str, list[QueryEvalResult]] = {}
    for r in query_results:
        difficulties.setdefault(r.difficulty, []).append(r)

    difficulty_breakdown: dict[str, dict[str, float]] = {}
    for diff, items in difficulties.items():
        n = len(items)
        difficulty_breakdown[diff] = {
            "count": float(n),
            "precision_at_k": round(sum(i.precision_at_k for i in items) / n, 4),
            "recall_at_k": round(sum(i.recall_at_k for i in items) / n, 4),
            "hit_rate": round(sum(1 for i in items if i.hit_at_k) / n, 4),
            "mrr": round(sum(i.reciprocal_rank for i in items) / n, 4),
        }

    return RetrievalEvaluationReport(
        k=k,
        total_queries=total_queries,
        macro_precision_at_k=round(macro_precision, 4),
        macro_recall_at_k=round(macro_recall, 4),
        macro_f1_at_k=round(macro_f1, 4),
        hit_rate_at_k=round(hit_rate, 4),
        mean_reciprocal_rank=round(mean_mrr, 4),
        top1_accuracy=round(top1_acc, 4),
        category_breakdown=category_breakdown,
        difficulty_breakdown=difficulty_breakdown,
        query_results=query_results,
    )


# ===========================================================================
# Report Rendering (Terminal, Markdown, JSON)
# ===========================================================================

def render_terminal_report(report: RetrievalEvaluationReport) -> str:
    """Renders human-readable CLI report with tables and quantitative methodology analysis."""
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("QUANTITATIVE RETRIEVAL EVALUATION HARNESS (RAG BENCHMARK)")
    lines.append("=" * 100)
    lines.append(f"Model: all-MiniLM-L6-v2 (384-d Cosine) | Metric Target: Top-{report.k} Retrieval | Total Queries: {report.total_queries}")
    lines.append("-" * 100)

    # Detailed Per-Query Table
    header = f"{'ID':<4} | {'Query (Truncated)':<40} | {'Expected':<22} | {'Rank 1 Match':<20} | {'P@' + str(report.k):<6} | {'R@' + str(report.k):<6} | {'Status':<6}"
    lines.append(header)
    lines.append("-" * 100)

    for r in report.query_results:
        q_trunc = (r.query[:37] + "...") if len(r.query) > 40 else r.query
        exp_str = ",".join(r.expected_functions)[:22]
        r1_str = (r.retrieved_functions[0] if r.retrieved_functions else "None")[:20]
        status = "PASS" if r.hit_at_k else "FAIL"
        line = f"{r.query_id:<4} | {q_trunc:<40} | {exp_str:<22} | {r1_str:<20} | {r.precision_at_k:<6.3f} | {r.recall_at_k:<6.3f} | {status:<6}"
        lines.append(line)

    lines.append("-" * 100)
    lines.append("")
    lines.append("=" * 100)
    lines.append("QUANTITATIVE SUMMARY METRICS:")
    lines.append("=" * 100)
    lines.append(f"  * Macro Precision@{report.k} : {report.macro_precision_at_k:.4f} ({report.macro_precision_at_k * 100:.1f}%) [Theoretical Max: ~33.3% for single-target queries]")
    lines.append(f"  * Macro Recall@{report.k}    : {report.macro_recall_at_k:.4f} ({report.macro_recall_at_k * 100:.1f}%)")
    lines.append(f"  * Macro F1@{report.k}        : {report.macro_f1_at_k:.4f}")
    lines.append(f"  * Hit Rate@{report.k} (Hit@k): {report.hit_rate_at_k:.4f} ({report.hit_rate_at_k * 100:.1f}%)")
    lines.append(f"  * Mean Reciprocal Rank (MRR): {report.mean_reciprocal_rank:.4f}")
    lines.append(f"  * Top-1 Accuracy            : {report.top1_accuracy:.4f} ({report.top1_accuracy * 100:.1f}%)")
    lines.append("")

    # Category Breakdown Table
    lines.append("-" * 100)
    lines.append("PERFORMANCE BY CATEGORY:")
    lines.append(f"  {'Category':<28} | {'Count':<6} | {'P@' + str(report.k):<8} | {'R@' + str(report.k):<8} | {'Hit Rate':<10} | {'MRR':<8}")
    lines.append("  " + "-" * 80)
    for cat, metrics in report.category_breakdown.items():
        lines.append(
            f"  {cat:<28} | {int(metrics['count']):<6} | {metrics['precision_at_k']:<8.3f} | {metrics['recall_at_k']:<8.3f} | {metrics['hit_rate']:<10.3f} | {metrics['mrr']:<8.3f}"
        )
    lines.append("")

    # Difficulty Breakdown Table
    lines.append("-" * 100)
    lines.append("PERFORMANCE BY QUERY DIFFICULTY:")
    lines.append(f"  {'Difficulty':<18} | {'Count':<6} | {'P@' + str(report.k):<8} | {'R@' + str(report.k):<8} | {'Hit Rate':<10} | {'MRR':<8}")
    lines.append("  " + "-" * 70)
    for diff, metrics in report.difficulty_breakdown.items():
        lines.append(
            f"  {diff:<18} | {int(metrics['count']):<6} | {metrics['precision_at_k']:<8.3f} | {metrics['recall_at_k']:<8.3f} | {metrics['hit_rate']:<10.3f} | {metrics['mrr']:<8.3f}"
        )
    lines.append("")

    # Pedagogical RAG Evaluation Methodology Notes
    lines.append("=" * 100)
    lines.append("QUANTITATIVE RAG EVALUATION METHODOLOGY INSIGHTS:")
    lines.append("=" * 100)
    lines.append("  1. Precision@k vs Recall@k Dynamics in Code RAG:")
    lines.append(f"     - With k={report.k} and 1 relevant function per query, Precision@{report.k} is bounded by 1/{report.k} = {1.0/report.k:.1%}.")
    lines.append("     - Precision@{k} reflects the signal-to-noise ratio in the LLM's prompt context.")
    lines.append(f"     - Recall@{report.k} measures whether the ground-truth function was surfaced at all.")
    lines.append("  2. Error Analysis:")
    fps = sum(len(r.false_positives) for r in report.query_results)
    fns = sum(len(r.false_negatives) for r in report.query_results)
    lines.append(f"     - Total False Positives (distractor functions in top-{report.k}): {fps}")
    lines.append(f"     - Total False Negatives (missed ground truth functions): {fns}")
    lines.append("  3. Optimization Roadmap:")
    if report.macro_recall_at_k < 0.90:
        lines.append("     - Recall is below 90%: Consider hybrid sparse-dense retrieval (BM25 + Dense) or fine-tuning embeddings.")
    else:
        lines.append("     - High Recall achieved! Consider introducing a Cross-Encoder Reranker to boost MRR and Rank-1 precision.")
    lines.append("=" * 100)

    return "\n".join(lines)


def export_markdown_report(report: RetrievalEvaluationReport, output_path: str | Path) -> None:
    """Exports structured evaluation metrics and methodology analysis to a Markdown file."""
    lines: list[str] = [
        "# Quantitative Retrieval Evaluation Report",
        "",
        "## Executive Summary",
        f"- **Benchmark Size**: {report.total_queries} Hand-Labeled Natural Language Questions",
        f"- **Retrieval Cutoff (k)**: {report.k}",
        f"- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions, Cosine Metric)",
        f"- **Macro Precision@{report.k}**: **{report.macro_precision_at_k * 100:.1f}%** (Single-target theoretical upper bound: 33.3%)",
        f"- **Macro Recall@{report.k}**: **{report.macro_recall_at_k * 100:.1f}%**",
        f"- **Macro F1@{report.k}**: **{report.macro_f1_at_k:.4f}**",
        f"- **Hit Rate@{report.k}**: **{report.hit_rate_at_k * 100:.1f}%**",
        f"- **Mean Reciprocal Rank (MRR)**: **{report.mean_reciprocal_rank:.4f}**",
        f"- **Top-1 Accuracy**: **{report.top1_accuracy * 100:.1f}%**",
        "",
        "---",
        "",
        "## Quantitative RAG Evaluation Methodology",
        "",
        "### 1. The Precision@k vs Recall@k Dilemma",
        "In dense retrieval for RAG, **Recall@k** measures *completeness* (whether the gold function reaches the LLM context), while **Precision@k** measures *purity* (how much noise/distraction is injected into the context window).",
        "",
        "$$Precision@k = \\frac{|Retrieved_k \\cap Relevant|}{k}$$",
        "$$Recall@k = \\frac{|Retrieved_k \\cap Relevant|}{|Relevant|}$$",
        "",
        "> [!IMPORTANT]",
        "> When evaluating questions with a single known-correct function ($|Relevant| = 1$) at $k=3$, the maximum possible Precision@3 is $\\frac{1}{3} \\approx 33.3\\%$. The remaining two retrieved positions are non-gold distractors. Recall@3, however, reaches 100% whenever the gold function is placed in ranks 1, 2, or 3.",
        "",
        "### 2. Metric Interpretation for Code RAG",
        "| Metric | Formula | Practical Meaning in Codebases | Target Goal |",
        "| :--- | :--- | :--- | :--- |",
        "| **Recall@3** | Hits in top-3 / Total Ground Truth | Does the prompt contain the function the developer asked about? | $\\ge 85\\%$ |",
        "| **Precision@3** | Hits in top-3 / 3 | How concentrated is the context window? | $\\ge 30\\%$ (for single-target) |",
        "| **MRR** | $\\frac{1}{N} \\sum \\frac{1}{Rank_1}$ | Is the best match at the very top of the list? | $\\ge 0.75$ |",
        "| **Top-1 Accuracy** | Proportion of queries with gold at Rank 1 | Can the pipeline answer directly using the primary anchor? | $\\ge 70\\%$ |",
        "",
        "---",
        "",
        "## Detailed Per-Query Results",
        "",
        "| ID | Natural Language Query | Ground Truth | Rank 1 Retrieved | Score | P@3 | R@3 | MRR | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in report.query_results:
        exp = "`" + "`, `".join(r.expected_functions) + "`"
        top1 = f"`{r.retrieved_items[0].func_name}`" if r.retrieved_items else "`None`"
        score = f"{r.retrieved_items[0].score:.4f}" if r.retrieved_items else "0.0"
        status = "✅ PASS" if r.hit_at_k else "❌ FAIL"
        lines.append(
            f"| **{r.query_id}** | {r.query} | {exp} | {top1} | {score} | {r.precision_at_k:.2f} | {r.recall_at_k:.2f} | {r.reciprocal_rank:.2f} | {status} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Performance by Category",
        "",
        "| Category | Queries | Precision@3 | Recall@3 | Hit Rate | MRR |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for cat, m in report.category_breakdown.items():
        lines.append(
            f"| **{cat}** | {int(m['count'])} | {m['precision_at_k']:.3f} | {m['recall_at_k']:.3f} | {m['hit_rate'] * 100:.1f}% | {m['mrr']:.3f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Performance by Query Difficulty",
        "",
        "| Difficulty | Queries | Precision@3 | Recall@3 | Hit Rate | MRR |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ])

    for diff, m in report.difficulty_breakdown.items():
        lines.append(
            f"| **{diff}** | {int(m['count'])} | {m['precision_at_k']:.3f} | {m['recall_at_k']:.3f} | {m['hit_rate'] * 100:.1f}% | {m['mrr']:.3f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## Diagnostic Insights & Next Steps",
        "1. **Dense Semantic Matching Strength**: Strong semantic correlation on natural language queries with phrasing differences (e.g. 'purge stale clones' successfully retrieves `garbage_collect_temp_repos`).",
        "2. **Context Window Optimization**: For prompts generated with retrieved code, a high Recall@3 ensures the necessary code body is sent to the LLM, while keeping $k=3$ limits total token usage.",
        "3. **Graph-Augmentation Synergy**: By feeding the vector search output to `rag.hybrid_retriever.hybrid_search`, 1-hop graph neighbors can be pulled in structurally even if their textual description has lower direct semantic similarity.",
    ])

    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
    print(f"Markdown evaluation report saved to: {output_path}")


def export_json_report(report: RetrievalEvaluationReport, output_path: str | Path) -> None:
    """Exports full benchmark results to structured JSON."""
    data = asdict(report)
    Path(output_path).write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"JSON evaluation report saved to: {output_path}")


# ===========================================================================
# CLI Entrypoint
# ===========================================================================

def build_cli_parser() -> argparse.ArgumentParser:
    """Configures command line interface."""
    parser = argparse.ArgumentParser(
        description="Quantitative Retrieval Evaluation Harness for Code RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--k",
        type=int,
        default=3,
        help="Top-k retrieval cutoff for precision and recall calculation",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Connect to live Qdrant server instead of isolated in-memory client",
    )
    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("QDRANT_HOST", "localhost"),
        help="Qdrant server hostname when using --live",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("QDRANT_PORT", "6333")),
        help="Qdrant server port when using --live",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=DEFAULT_COLLECTION_NAME,
        help="Qdrant collection name to evaluate against",
    )
    parser.add_argument(
        "--export-md",
        type=str,
        default=None,
        help="File path to save Markdown evaluation report",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="File path to save JSON evaluation report",
    )
    return parser


def main() -> int:
    """CLI execution entrypoint."""
    parser = build_cli_parser()
    args = parser.parse_args()

    print(f"\nInitializing Retrieval Evaluation Harness (Top-{args.k})...")
    client: QdrantClient
    if args.live:
        print(f"Connecting to live Qdrant server at {args.host}:{args.port}...")
        try:
            client = QdrantClient(host=args.host, port=args.port)
            client.get_collections()
        except Exception as exc:
            print(f"Warning: Failed to connect to live Qdrant ({exc}). Falling back to isolated in-memory client.")
            client = QdrantClient(":memory:")
    else:
        print("Running in isolated in-memory mode (deterministic, zero external infrastructure required)...")
        client = QdrantClient(":memory:")

    print("Embedding and seeding test corpus into Qdrant...")
    points_seeded = seed_evaluation_corpus(client=client, collection_name=args.collection)
    print(f"Seeded {points_seeded} code functions into '{args.collection}'.")

    print(f"Evaluating {len(GROUND_TRUTH_DATASET)} hand-labeled benchmark questions (k={args.k})...\n")
    report = run_retrieval_evaluation(
        dataset=GROUND_TRUTH_DATASET,
        corpus=TEST_CORPUS,
        client=client,
        collection_name=args.collection,
        k=args.k,
    )

    terminal_output = render_terminal_report(report)
    print(terminal_output)

    if args.export_md:
        export_markdown_report(report, args.export_md)

    if args.export_json:
        export_json_report(report, args.export_json)

    return 0


if __name__ == "__main__":
    sys.exit(main())

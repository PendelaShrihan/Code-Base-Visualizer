"""
rag/test_search.py
------------------
Semantic Code Retrieval Verification & Vector Similarity Evaluation.

This script accepts plain English questions, vectorizes them using SentenceTransformer
(all-MiniLM-L6-v2), and executes a vector similarity search against the Qdrant
'code_chunks' collection to retrieve and evaluate the top-3 matching functions.

Usage:
    # 1. Direct query via CLI argument:
    python rag/test_search.py "how do I clone a git repository?"

    # 2. Interactive question-and-answer prompt:
    python rag/test_search.py

    # 3. Run built-in evaluation test suite with sample queries:
    python rag/test_search.py --eval

    # 4. Seed demo functions from current codebase into Qdrant if empty:
    python rag/test_search.py --seed

Vector Similarity Evaluation Primer:
------------------------------------
1. Metric: Cosine Similarity
   - Range: [-1.0, 1.0]. For normalized sentence embeddings, typically [0.0, 1.0].
   - Score Interpretation:
       * >= 0.70 : HIGH CONFIDENCE (Strong direct semantic correlation)
       * 0.45 - 0.69 : MODERATE RELEVANCE (Related domain / partial conceptual overlap)
       * < 0.45  : WEAK / NOISE (Low semantic alignment; potential false positive)
2. Confidence Gap (Delta):
   - Margin between Rank 1 and Rank 2 score (Score_1 - Score_2).
   - Delta > 0.10 indicates high discrimination; low delta indicates semantic ambiguity.
3. Graph Metadata Fusion:
   - PageRank & Commit Churn from Qdrant payloads provide structural importance context
     alongside raw semantic similarity.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
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
from qdrant_client.models import PointStruct

from rag.embeddings import model
from rag.qdrant_client import (
    COLLECTION_NAME,
    create_code_chunks_collection,
    get_qdrant_client,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Evaluation Threshold Constants
# ---------------------------------------------------------------------------
SIMILARITY_THRESHOLD_HIGH: float = 0.70
SIMILARITY_THRESHOLD_MODERATE: float = 0.45


def classify_similarity(score: float) -> str:
    """Classify cosine similarity score into qualitative evaluation tiers."""
    if score >= SIMILARITY_THRESHOLD_HIGH:
        return "HIGH CONFIDENCE"
    if score >= SIMILARITY_THRESHOLD_MODERATE:
        return "MODERATE RELEVANCE"
    return "WEAK / NOISE"


# ---------------------------------------------------------------------------
# Core Semantic Search Function
# ---------------------------------------------------------------------------
def search_functions(
    query: str,
    top_k: int = 3,
    client: Optional[QdrantClient] = None,
    collection_name: str = COLLECTION_NAME,
    score_threshold: Optional[float] = None,
) -> list[dict[str, Any]]:
    """
    Vectorizes a plain English query and retrieves top-k matching functions from Qdrant.

    Args:
        query: Plain English natural language question (e.g. "clone repository").
        top_k: Number of matching functions to return (default: 3).
        client: QdrantClient instance (defaults to connected Qdrant client).
        collection_name: Qdrant collection name (default: 'code_chunks').
        score_threshold: Optional minimum cosine similarity cutoff.

    Returns:
        List of result dictionaries containing rank, func_name, file_path, score,
        centrality (pagerank), commit_count, is_dead_code, and evaluation label.
    """
    if not query.strip():
        return []

    client = client or get_qdrant_client()

    # Ensure collection exists before querying
    if not client.collection_exists(collection_name):
        logger.warning(f"Collection '{collection_name}' does not exist. Creating schema...")
        create_code_chunks_collection(client=client, collection_name=collection_name)

    # 1. Encode query into 384-dimensional vector using all-MiniLM-L6-v2
    query_vector: list[float] = model.encode(query).tolist()

    # 2. Query Qdrant vector database (supports modern query_points and legacy search)
    if hasattr(client, "query_points"):
        response = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )
        scored_points = response.points
    else:
        scored_points = client.search(  # type: ignore[attr-defined]
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
        )

    # 3. Format and enrich results with evaluation metrics
    results: list[dict[str, Any]] = []
    for rank, point in enumerate(scored_points, start=1):
        payload = point.payload or {}
        score = float(point.score)
        results.append(
            {
                "rank": rank,
                "point_id": str(point.id),
                "func_name": payload.get("func_name", "<unknown>"),
                "file_path": payload.get("file_path", "<unknown>"),
                "score": round(score, 4),
                "similarity_label": classify_similarity(score),
                "pagerank": round(float(payload.get("pagerank", 0.0)), 6),
                "commit_count": int(payload.get("commit_count", 0)),
                "is_dead_code_candidate": bool(
                    payload.get("is_dead_code_candidate", False)
                ),
            }
        )

    return results


# ---------------------------------------------------------------------------
# Formatting & Evaluation Display
# ---------------------------------------------------------------------------
def format_results(query: str, results: list[dict[str, Any]]) -> str:
    """Renders human-readable report with vector similarity evaluation breakdown."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"QUERY: \"{query}\"")
    lines.append("=" * 80)

    if not results:
        lines.append("No matching functions found in Qdrant collection.")
        lines.append("Tip: Use '--seed' to populate realistic sample code functions.")
        lines.append("=" * 80)
        return "\n".join(lines)

    lines.append(f"Top {len(results)} Matching Functions (ranked by Cosine Similarity):")
    lines.append("-" * 80)

    for res in results:
        rank = res["rank"]
        func_name = res["func_name"]
        file_path = res["file_path"]
        score = res["score"]
        label = res["similarity_label"]
        pagerank = res["pagerank"]
        commits = res["commit_count"]
        dead_code = "YES" if res["is_dead_code_candidate"] else "No"

        lines.append(
            f"  [{rank}] {func_name}()"
        )
        lines.append(f"      File        : {file_path}")
        lines.append(f"      Similarity  : {score:.4f} -> [{label}]")
        lines.append(
            f"      Centrality  : PageRank={pagerank:.6f} | Churn={commits} commits | DeadCode={dead_code}"
        )
        lines.append("")

    # Vector similarity evaluation diagnostics
    lines.append("-" * 80)
    lines.append("VECTOR SIMILARITY EVALUATION:")
    top_score = results[0]["score"]
    lines.append(f"  * Top Confidence Score : {top_score:.4f} ({results[0]['similarity_label']})")

    if len(results) >= 2:
        gap = results[0]["score"] - results[1]["score"]
        lines.append(f"  * Confidence Gap (1 vs 2): +{gap:.4f}")
        if gap >= 0.10:
            lines.append("    -> High discrimination: Rank 1 is distinctly separated from Rank 2.")
        elif gap >= 0.03:
            lines.append("    -> Moderate discrimination: Rank 1 and 2 share conceptual proximity.")
        else:
            lines.append("    -> Low discrimination / Ambiguous: Multiple candidate functions are close ties.")

    lines.append("=" * 80)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Seed Helper for Testing & Verification
# ---------------------------------------------------------------------------
SAMPLE_FUNCTIONS: list[dict[str, Any]] = [
    {
        "id": 1,
        "func_name": "clone_repository",
        "file_path": "app/services/git_service.py",
        "text": "func: clone_repository\nfile: app/services/git_service.py\nClones a remote git repository to local temp directory using GitPython full clone.",
        "pagerank": 0.0825,
        "commit_count": 14,
        "is_dead_code_candidate": False,
    },
    {
        "id": 2,
        "func_name": "cleanup_repo_directory",
        "file_path": "app/services/git_service.py",
        "text": "func: cleanup_repo_directory\nfile: app/services/git_service.py\nRemoves cloned repository directory and performs filesystem garbage collection.",
        "pagerank": 0.0412,
        "commit_count": 8,
        "is_dead_code_candidate": False,
    },
    {
        "id": 3,
        "func_name": "scan_repository",
        "file_path": "parser/repo_walker.py",
        "text": "func: scan_repository\nfile: parser/repo_walker.py\nScans repository directories, extracts AST nodes with Tree-sitter, and computes PageRank centrality.",
        "pagerank": 0.1250,
        "commit_count": 22,
        "is_dead_code_candidate": False,
    },
    {
        "id": 4,
        "func_name": "detect_dead_code",
        "file_path": "parser/repo_walker.py",
        "text": "func: detect_dead_code\nfile: parser/repo_walker.py\nDetects unreferenced candidate dead code functions with in-degree zero in call graph.",
        "pagerank": 0.0350,
        "commit_count": 6,
        "is_dead_code_candidate": False,
    },
    {
        "id": 5,
        "func_name": "embed_chunks",
        "file_path": "rag/embeddings.py",
        "text": "func: embed_chunks\nfile: rag/embeddings.py\nGenerates dense vector embeddings using SentenceTransformer all-MiniLM-L6-v2 in batch mode.",
        "pagerank": 0.0610,
        "commit_count": 9,
        "is_dead_code_candidate": False,
    },
    {
        "id": 6,
        "func_name": "batch_upsert_chunks",
        "file_path": "rag/ingestion.py",
        "text": "func: batch_upsert_chunks\nfile: rag/ingestion.py\nUpserts embedded code chunks and metadata payloads into Qdrant vector database collection.",
        "pagerank": 0.0780,
        "commit_count": 11,
        "is_dead_code_candidate": False,
    },
]


def seed_sample_functions(
    client: Optional[QdrantClient] = None,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """Populates Qdrant with sample functions from the CodeBase Visualizer codebase."""
    client = client or get_qdrant_client()
    create_code_chunks_collection(client=client, collection_name=collection_name, recreate=True)

    points: list[PointStruct] = []
    print("Embedding sample codebase functions...")
    for item in SAMPLE_FUNCTIONS:
        vector = model.encode(item["text"]).tolist()
        points.append(
            PointStruct(
                id=item["id"],
                vector=vector,
                payload={
                    "func_name": item["func_name"],
                    "file_path": item["file_path"],
                    "pagerank": item["pagerank"],
                    "commit_count": item["commit_count"],
                    "is_dead_code_candidate": item["is_dead_code_candidate"],
                },
            )
        )

    client.upsert(collection_name=collection_name, points=points, wait=True)
    print(f"Successfully seeded {len(points)} function vectors into '{collection_name}'.")
    return len(points)


# ---------------------------------------------------------------------------
# Built-in Evaluation Queries
# ---------------------------------------------------------------------------
EVALUATION_QUERIES: list[tuple[str, str]] = [
    ("How do I clone a git repository?", "clone_repository"),
    ("Where is dead code detected in the call graph?", "detect_dead_code"),
    ("Which function calculates or attaches embeddings using SentenceTransformers?", "embed_chunks"),
    ("How are chunks saved into the Qdrant vector database?", "batch_upsert_chunks"),
    ("How does AST scanning and repo walking work?", "scan_repository"),
]


def run_evaluation_suite(
    client: Optional[QdrantClient] = None,
    collection_name: str = COLLECTION_NAME,
) -> None:
    """Executes a benchmark evaluation suite and prints similarity evaluations."""
    client = client or get_qdrant_client()

    info = client.get_collection(collection_name=collection_name)
    if info.points_count == 0:
        print(f"Collection '{collection_name}' is empty. Seeding sample functions first...\n")
        seed_sample_functions(client=client, collection_name=collection_name)

    print("\n" + "#" * 80)
    print("VECTOR SIMILARITY EVALUATION BENCHMARK")
    print("#" * 80 + "\n")

    correct_top1 = 0
    correct_top3 = 0

    for query, expected_top1 in EVALUATION_QUERIES:
        results = search_functions(
            query=query,
            top_k=3,
            client=client,
            collection_name=collection_name,
        )
        print(format_results(query, results))

        retrieved_names = [r["func_name"] for r in results]
        if retrieved_names and retrieved_names[0] == expected_top1:
            correct_top1 += 1
            top1_status = "PASS (Rank 1 match)"
        else:
            top1_status = f"FAIL (Expected '{expected_top1}', got '{retrieved_names[0] if retrieved_names else 'None'}')"

        if expected_top1 in retrieved_names:
            correct_top3 += 1
            top3_status = "PASS (In Top 3)"
        else:
            top3_status = "FAIL (Not in Top 3)"

        print(f">> Top-1 Evaluation: {top1_status}")
        print(f">> Top-3 Recall    : {top3_status}\n")

    total = len(EVALUATION_QUERIES)
    print("=" * 80)
    print("EVALUATION SUMMARY REPORT:")
    print(f"  Total Queries Evaluated: {total}")
    print(f"  Top-1 Accuracy         : {correct_top1}/{total} ({correct_top1/total*100:.1f}%)")
    print(f"  Top-3 Recall (Hit@3)   : {correct_top3}/{total} ({correct_top3/total*100:.1f}%)")
    print("=" * 80 + "\n")


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semantic Code Retrieval Verification & Vector Similarity Evaluation"
    )
    parser.add_argument(
        "question",
        nargs="*",
        help="Plain English question to search for (e.g. 'how to clone git repo')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of matching functions to return (default: 3)",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=COLLECTION_NAME,
        help=f"Qdrant collection name (default: {COLLECTION_NAME})",
    )
    parser.add_argument(
        "--seed",
        action="store_true",
        help="Seed sample functions into Qdrant collection for testing",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Run comprehensive vector similarity evaluation benchmark",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive query prompt",
    )

    args = parser.parse_args()

    client = get_qdrant_client()

    # Handle seed command
    if args.seed:
        seed_sample_functions(client=client, collection_name=args.collection)
        return

    # Handle eval benchmark
    if args.eval:
        run_evaluation_suite(client=client, collection_name=args.collection)
        return

    # Auto-seed if collection is completely empty so user gets instant verification
    try:
        if client.collection_exists(args.collection):
            info = client.get_collection(args.collection)
            if info.points_count == 0:
                print(f"Note: Collection '{args.collection}' is currently empty.")
                print("Auto-seeding sample codebase functions for retrieval verification...\n")
                seed_sample_functions(client=client, collection_name=args.collection)
        else:
            print(f"Creating and seeding collection '{args.collection}'...\n")
            seed_sample_functions(client=client, collection_name=args.collection)
    except Exception as e:
        logger.warning(f"Could not check/seed collection: {e}")

    # Process query from args
    query = " ".join(args.question).strip() if args.question else ""

    if query:
        results = search_functions(
            query=query,
            top_k=args.top_k,
            client=client,
            collection_name=args.collection,
        )
        print(format_results(query, results))
        return

    # If no query provided and not interactive flag, enter interactive loop
    print("=" * 80)
    print("SEMANTIC CODE RETRIEVAL - INTERACTIVE SEARCH")
    print("Type a plain English question and press Enter. Type 'exit' or 'quit' to exit.")
    print("=" * 80)

    while True:
        try:
            user_input = input("\nEnter question: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break
            results = search_functions(
                query=user_input,
                top_k=args.top_k,
                client=client,
                collection_name=args.collection,
            )
            print(format_results(user_input, results))
        except (KeyboardInterrupt, EOFError):
            print("\nExiting search.")
            break


if __name__ == "__main__":
    main()

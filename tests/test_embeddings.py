"""
tests/test_embeddings.py
Unit tests and sanity checks for rag/embeddings.py.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from app.services.ast_engine import extract_function_chunks_from_file
from rag.embeddings import embed_chunks


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two lists of floats."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def test_embed_chunks_empty():
    """Test that passing an empty list returns an empty list."""
    assert embed_chunks([]) == []


def test_embed_chunks_git_service():
    """
    Test extract_function_chunks_from_file on git_service.py fed into embed_chunks.
    Verifies that every chunk gets a 384-dimensional embedding list.
    """
    git_service_path = Path("app/services/git_service.py")
    chunks = extract_function_chunks_from_file(git_service_path)
    assert len(chunks) > 0, "Expected at least one function chunk from git_service.py"

    embedded_chunks = embed_chunks(chunks)

    assert len(embedded_chunks) == len(chunks)
    for chunk in embedded_chunks:
        assert "embedding" in chunk, "Chunk is missing 'embedding' key"
        embedding = chunk["embedding"]
        assert isinstance(embedding, list), "Embedding must be a plain Python list"
        assert len(embedding) == 384, f"Expected 384-dim embedding vector, got {len(embedding)}"
        assert all(isinstance(v, float) for v in embedding), "All embedding elements must be floats"


def test_embeddings_semantic_similarity():
    """
    Sanity check: semantically similar code snippets should have higher cosine
    similarity than unrelated snippets.
    """
    chunks = [
        {"id": "clone_1", "text": "def clone_repository(url: str):\n    return git.Repo.clone_from(url)"},
        {"id": "clone_2", "text": "def download_repo(link: str):\n    return git.Repo.clone_from(link)"},
        {"id": "tax_1", "text": "def calculate_tax(income: float):\n    return income * 0.2"},
    ]

    embedded = embed_chunks(chunks)
    vec_a = embedded[0]["embedding"]
    vec_b = embedded[1]["embedding"]
    vec_c = embedded[2]["embedding"]

    sim_ab = cosine_similarity(vec_a, vec_b)
    sim_ac = cosine_similarity(vec_a, vec_c)
    sim_bc = cosine_similarity(vec_b, vec_c)

    print(f"\nSimilarity (clone_repo vs download_repo): {sim_ab:.4f}")
    print(f"Similarity (clone_repo vs calculate_tax): {sim_ac:.4f}")
    print(f"Similarity (download_repo vs calculate_tax): {sim_bc:.4f}")

    assert sim_ab > sim_ac, f"Expected sim(clone, download) ({sim_ab}) > sim(clone, tax) ({sim_ac})"
    assert sim_ab > sim_bc, f"Expected sim(clone, download) ({sim_ab}) > sim(download, tax) ({sim_bc})"

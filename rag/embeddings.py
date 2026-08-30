"""
rag/embeddings.py
Provides batch embedding utilities using SentenceTransformer.
Loads the model once at module level for efficient reuse.
"""

from __future__ import annotations

from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Embedding Model setup (module-level singleton — loaded once, reused everywhere)
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
model: SentenceTransformer = SentenceTransformer(MODEL_NAME)


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Extracts the 'text' field from each chunk, computes embeddings in a single
    batch call, converts the output NumPy vectors to plain Python lists,
    and attaches them back under the 'embedding' key for each chunk.

    Args:
        chunks: List of chunk dictionaries, each containing at least a 'text' key.

    Returns:
        The same list of chunk dictionaries with the 'embedding' key populated.
    """
    if not chunks:
        return chunks

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)

    for chunk, vector in zip(chunks, embeddings):
        chunk["embedding"] = vector.tolist()

    return chunks

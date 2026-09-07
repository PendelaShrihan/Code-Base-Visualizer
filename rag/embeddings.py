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


def embed_chunks(
    chunks: list[dict],
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> list[dict]:
    """
    Extracts the 'text' field from each chunk, computes embeddings in a single
    batch call, converts the output NumPy vectors to plain Python lists,
    and attaches them back under the 'embedding' key for each chunk.

    Args:
        chunks:            List of chunk dictionaries, each containing at least
                           a 'text' key.
        batch_size:        Number of texts to encode per internal mini-batch
                           inside SentenceTransformer (default: 32).  Increase
                           for GPU-backed workers; leave at default for CPU.
        show_progress_bar: When True, renders a tqdm progress bar during
                           encoding.  Set to False (default) for clean Celery
                           worker logs.

    Returns:
        The same list of chunk dictionaries with the 'embedding' key populated.
    """
    if not chunks:
        return chunks

    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress_bar,
    )

    for chunk, vector in zip(chunks, embeddings):
        chunk["embedding"] = vector.tolist()

    return chunks

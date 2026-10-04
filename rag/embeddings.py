import logging
from typing import Any

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Embedding Model setup (module-level singleton — loaded once, reused everywhere)
# ---------------------------------------------------------------------------
MODEL_NAME = "all-MiniLM-L6-v2"
model: SentenceTransformer = SentenceTransformer(MODEL_NAME)


def get_embedding_model() -> SentenceTransformer:
    """Return the module-level SentenceTransformer model singleton."""
    return model


def embed_chunks(
    chunks: list[dict[str, Any]],
    batch_size: int = 32,
    show_progress_bar: bool = False,
) -> list[dict[str, Any]]:
    """
    Extracts the 'text' field from each chunk, computes embeddings in a single
    batch call, converts the output NumPy vectors to plain Python lists,
    and attaches them back under the 'embedding' key for each chunk.

    Defensive behaviors:
      - Gracefully handles empty or missing 'text' fields using func_name/file_path metadata.
      - Strips null-byte characters (\x00) that can corrupt HuggingFace tokenizers.
      - Handles empty chunk lists immediately without invoking the model.

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

    texts: list[str] = []
    for chunk in chunks:
        raw_text = chunk.get("text")
        if not isinstance(raw_text, str) or not raw_text.strip():
            # Fall back to structured identifiers if 'text' is missing or blank
            fn = chunk.get("func_name", "")
            fp = chunk.get("file_path", "")
            raw_text = f"func: {fn}\nfile: {fp}".strip() or "empty"
        # Sanitize null-bytes that could crash lower-level tokenizers/serializers
        cleaned = raw_text.replace("\x00", "")
        texts.append(cleaned)

    try:
        embeddings = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
        )
    except Exception as exc:
        logger.error("SentenceTransformer model.encode failed for %d texts: %s", len(texts), exc)
        raise

    for chunk, vector in zip(chunks, embeddings):
        chunk["embedding"] = vector.tolist() if hasattr(vector, "tolist") else list(vector)

    return chunks


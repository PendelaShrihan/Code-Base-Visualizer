"""
rag/ingestion.py
----------------
Batch vector ingestion pipeline for CodeBase Visualizer.

Pipeline overview
-----------------
  graph_dict["nodes"]
        │
        ▼ (filter kind=="function")
  extract_function_chunks()   →  list[ChunkDict]
        │
        ▼ (slices of BATCH_SIZE)
  embed_chunks(slice)         →  attach "embedding" to each chunk
        │
        ▼
  qdrant.upsert(PointStructs) →  idempotent upsert via UUID-v5 point IDs
        │
        ▼
  ingest_graph()              →  IngestionSummary dict

Design decisions
----------------
* **UUID v5 (NAMESPACE_URL, node_id)** — deterministic point IDs so that
  re-analysing the same repository *updates* existing Qdrant points rather
  than creating duplicates.

* **Batch size 64 (default)** — empirically balances SentenceTransformer
  GPU/CPU throughput against Qdrant HTTP round-trip overhead for repos with
  hundreds to thousands of functions.

* **Lazy collection creation** — `create_code_chunks_collection()` is called
  once at the start of `ingest_graph()` and is a no-op if the collection
  already exists.

* **Upsert, not insert** — Qdrant `upsert()` is atomic and handles both the
  create-new and overwrite-existing cases in a single call.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from rag.embeddings import embed_chunks
from rag.qdrant_client import (
    COLLECTION_NAME,
    create_code_chunks_collection,
    get_qdrant_client,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default number of chunks to embed and upsert in a single batch.
#: 64 is a sweet spot: large enough to amortise HTTP overhead, small enough
#: to fit comfortably in CPU/GPU memory for all-MiniLM-L6-v2 (384-d).
DEFAULT_BATCH_SIZE: int = 64

#: UUID namespace used to derive deterministic point IDs from node ID strings.
_UUID_NAMESPACE: uuid.UUID = uuid.NAMESPACE_URL

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

ChunkDict = dict[str, Any]
IngestionSummary = dict[str, Any]


# ---------------------------------------------------------------------------
# Step 1 – Extract function chunks from the serialised graph
# ---------------------------------------------------------------------------

def extract_function_chunks(graph_dict: dict[str, Any]) -> list[ChunkDict]:
    """Build an embeddable chunk dict for every function node in *graph_dict*.

    Only nodes with ``kind == "function"`` are included; file, class, import,
    and call-target nodes are ignored because they carry no meaningful semantic
    text for code search.

    The ``text`` field is constructed as::

        func: <func_name>
        file: <file_path>

    This minimal but structured format is intentionally terse so the embedding
    model focuses on the function identity rather than noise.

    The ``chunk_id`` is a **deterministic UUID v5** derived from the node's ID
    string (e.g. ``"app/services/git_service.py::func::clone_repository"``).
    Using UUID v5 means the same node always maps to the same Qdrant point ID,
    making repeated ingestion of the same repository fully idempotent.

    Args:
        graph_dict: Dict produced by :func:`parser.repo_walker.graph_to_json`,
                    containing ``"nodes"`` and ``"edges"`` lists.

    Returns:
        List of chunk dicts, one per function node.  Each dict contains:

        * ``chunk_id``   – deterministic UUID string (Qdrant point ID)
        * ``text``       – embeddable text representation
        * ``func_name``  – bare function name
        * ``file_path``  – relative file path within the repository
        * ``pagerank``   – PageRank centrality score (float)
        * ``commit_count`` – git commit churn count (int)
        * ``is_dead_code_candidate`` – bool flag from dead code analysis
    """
    chunks: list[ChunkDict] = []

    nodes: list[dict[str, Any]] = graph_dict.get("nodes", [])

    for node in nodes:
        if node.get("kind") != "function":
            continue

        node_id: str = node.get("id", "")
        func_name: str = node.get("name") or node.get("label") or ""
        file_path: str = node.get("file") or node.get("path") or ""

        # Build embeddable text — short but semantically structured.
        text = f"func: {func_name}\nfile: {file_path}"

        # Deterministic UUID v5 from the scoped node ID string.
        chunk_id = str(uuid.uuid5(_UUID_NAMESPACE, node_id))

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "func_name": func_name,
                "file_path": file_path,
                "pagerank": float(node.get("pagerank", 0.0)),
                "commit_count": int(node.get("commit_count", 0)),
                "is_dead_code_candidate": bool(
                    node.get("is_dead_code_candidate", False)
                ),
            }
        )

    logger.debug(
        "extract_function_chunks: extracted %d function chunk(s) from %d node(s)",
        len(chunks),
        len(nodes),
    )
    return chunks


# ---------------------------------------------------------------------------
# Step 2 – Batch-embed and upsert to Qdrant
# ---------------------------------------------------------------------------

def batch_upsert_chunks(
    chunks: list[ChunkDict],
    client: QdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> tuple[int, int]:
    """Embed *chunks* in batches and upsert them to Qdrant.

    Each batch slice is:
    1. Embedded via :func:`rag.embeddings.embed_chunks` (single
       ``SentenceTransformer.encode()`` call per batch — GPU-friendly).
    2. Converted to :class:`qdrant_client.models.PointStruct` objects where
       the ``id`` is the pre-computed ``chunk_id`` UUID, the ``vector`` is the
       embedding list, and ``payload`` contains all remaining metadata fields.
    3. Upserted to Qdrant in a single HTTP call per batch.

    Args:
        chunks:          List of chunk dicts from :func:`extract_function_chunks`.
        client:          Qdrant client (uses module-level default if ``None``).
        collection_name: Target collection name.
        batch_size:      Number of chunks to process per batch (default: 64).

    Returns:
        ``(total_upserted, total_batches)`` — number of points successfully
        upserted and number of batch iterations executed.

    Raises:
        Any exception from Qdrant or the embedding model propagates upward so
        the Celery task can handle/retry appropriately.
    """
    if not chunks:
        logger.debug("batch_upsert_chunks: no chunks to upsert — skipping.")
        return 0, 0

    client = client or get_qdrant_client()

    total_upserted = 0
    total_batches = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch = chunks[batch_start : batch_start + batch_size]

        # --- Embed (mutates batch in-place, adds "embedding" key) ------------
        embed_chunks(batch, batch_size=batch_size)

        # --- Build PointStructs -----------------------------------------------
        points: list[PointStruct] = [
            PointStruct(
                id=chunk["chunk_id"],
                vector=chunk["embedding"],
                payload={
                    "func_name": chunk["func_name"],
                    "file_path": chunk["file_path"],
                    "pagerank": chunk["pagerank"],
                    "commit_count": chunk["commit_count"],
                    "is_dead_code_candidate": chunk["is_dead_code_candidate"],
                },
            )
            for chunk in batch
        ]

        # --- Upsert -----------------------------------------------------------
        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,  # Block until Qdrant confirms the write for consistency
        )

        batch_count = len(points)
        total_upserted += batch_count
        total_batches += 1

        logger.debug(
            "batch_upsert_chunks: upserted batch %d/%d (%d points, offset %d)",
            total_batches,
            -(-len(chunks) // batch_size),  # ceiling division
            batch_count,
            batch_start,
        )

    logger.info(
        "batch_upsert_chunks: finished — %d point(s) upserted in %d batch(es)",
        total_upserted,
        total_batches,
    )
    return total_upserted, total_batches


# ---------------------------------------------------------------------------
# Public entry-point
# ---------------------------------------------------------------------------

def ingest_graph(
    graph_dict: dict[str, Any],
    client: QdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> IngestionSummary:
    """Full ingestion pipeline: extract → embed → upsert.

    Lazily creates the Qdrant collection if it does not already exist, then
    runs :func:`extract_function_chunks` followed by
    :func:`batch_upsert_chunks`.

    Args:
        graph_dict:      Serialised graph dict from
                         :func:`parser.repo_walker.graph_to_json`.
        client:          Qdrant client (uses module-level default if ``None``).
        collection_name: Target Qdrant collection (default: ``"code_chunks"``).
        batch_size:      Embedding / upsert batch size (default: 64).

    Returns:
        Summary dict::

            {
                "chunks_extracted": <int>,
                "chunks_upserted":  <int>,
                "batches":          <int>,
                "collection":       "<str>",
                "status":           "ok" | "empty",
            }
    """
    client = client or get_qdrant_client()

    # Lazily ensure the collection exists (no-op if already present)
    create_code_chunks_collection(client=client, collection_name=collection_name)

    # Step 1 — Extract
    chunks = extract_function_chunks(graph_dict)
    chunks_extracted = len(chunks)

    if not chunks:
        logger.info(
            "ingest_graph: no function nodes found in graph — nothing to upsert."
        )
        return {
            "chunks_extracted": 0,
            "chunks_upserted": 0,
            "batches": 0,
            "collection": collection_name,
            "status": "empty",
        }

    logger.info(
        "ingest_graph: ingesting %d function chunk(s) into '%s' (batch_size=%d)",
        chunks_extracted,
        collection_name,
        batch_size,
    )

    # Step 2 — Embed + Upsert
    total_upserted, total_batches = batch_upsert_chunks(
        chunks=chunks,
        client=client,
        collection_name=collection_name,
        batch_size=batch_size,
    )

    summary: IngestionSummary = {
        "chunks_extracted": chunks_extracted,
        "chunks_upserted": total_upserted,
        "batches": total_batches,
        "collection": collection_name,
        "status": "ok",
    }
    logger.info("ingest_graph: complete — %s", summary)
    return summary

"""
rag/qdrant_client.py
--------------------
Qdrant vector database client configuration and schema initialization.

Target Container (Day 13):
  - Docker Compose service: host="qdrant", port=6333
  - Local default: host="localhost", port=6333 (overridable via QDRANT_HOST / QDRANT_PORT)

Collection Configuration:
  - Collection Name: code_chunks
  - Vector Size: 384 (matching all-MiniLM-L6-v2 embedding dimensions)
  - Distance Metric: Cosine
  - Payload Indexes:
      * file_path (keyword)
      * func_name (keyword)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

# If executed directly as `python rag/qdrant_client.py`, sys.path[0] is the 'rag' dir,
# which would cause `import qdrant_client` to shadow the third-party library.
_current_dir = str(Path(__file__).resolve().parent)
if sys.path and sys.path[0] == _current_dir:
    sys.path.pop(0)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from qdrant_client import QdrantClient, models
from qdrant_client.http.models import CollectionInfo, Distance, PayloadSchemaType, VectorParams

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Connection Configuration
# ---------------------------------------------------------------------------
# Docker Compose exposes Qdrant under the service name "qdrant".
# Falls back to localhost for local development without Docker.
QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", "6333"))

# ---------------------------------------------------------------------------
# Schema Constants
# ---------------------------------------------------------------------------
COLLECTION_NAME: str = "code_chunks"
VECTOR_SIZE: int = 384  # Matches all-MiniLM-L6-v2 output dimension
VECTOR_DISTANCE: Distance = Distance.COSINE

PAYLOAD_INDEXES: list[tuple[str, PayloadSchemaType]] = [
    ("file_path", PayloadSchemaType.KEYWORD),
    ("func_name", PayloadSchemaType.KEYWORD),
]


def get_qdrant_client(
    host: Optional[str] = None,
    port: Optional[int] = None,
    location: Optional[str] = None,
) -> QdrantClient:
    """
    Factory function to create a Qdrant client instance.

    Args:
        host: Hostname for Qdrant server (defaults to QDRANT_HOST env or localhost).
        port: Port for Qdrant server (defaults to QDRANT_PORT env or 6333).
        location: If provided (e.g. ":memory:"), overrides host/port connection.

    Returns:
        QdrantClient instance.
    """
    if location:
        return QdrantClient(location=location)
    return QdrantClient(
        host=host or QDRANT_HOST,
        port=port or QDRANT_PORT,
    )


# Module-level default client instance pointed at the configured Qdrant service
qdrant_client: QdrantClient = get_qdrant_client()


def create_code_chunks_collection(
    client: Optional[QdrantClient] = None,
    collection_name: str = COLLECTION_NAME,
    vector_size: int = VECTOR_SIZE,
    distance: Distance = VECTOR_DISTANCE,
    recreate: bool = False,
) -> CollectionInfo:
    """
    Creates the code_chunks collection in Qdrant with the specified vector dimensions,
    distance metric, and payload keyword indexes.

    Args:
        client: Qdrant client instance (uses default module client if None).
        collection_name: Name of the collection to create.
        vector_size: Dimensionality of embeddings (default: 384).
        distance: Vector distance metric (default: Cosine).
        recreate: If True, recreates the collection even if it already exists.

    Returns:
        CollectionInfo object containing the verified collection configuration.
    """
    client = client or qdrant_client

    exists = client.collection_exists(collection_name)
    if exists and recreate:
        logger.info(f"Recreating collection '{collection_name}'...")
        client.delete_collection(collection_name=collection_name)
        exists = False

    if not exists:
        logger.info(
            f"Creating collection '{collection_name}' (size={vector_size}, distance={distance})..."
        )
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )
    else:
        logger.info(f"Collection '{collection_name}' already exists.")

    # Create payload indexes on metadata fields for fast filtering
    for field_name, schema_type in PAYLOAD_INDEXES:
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name=field_name,
                field_schema=schema_type,
            )
            logger.info(f"Created payload index on '{collection_name}.{field_name}' ({schema_type}).")
        except Exception as err:
            logger.warning(f"Note on payload index for '{field_name}': {err}")

    # Verify and return collection metadata
    return verify_collection_config(
        client=client,
        collection_name=collection_name,
        expected_size=vector_size,
        expected_distance=distance,
    )


def verify_collection_config(
    client: Optional[QdrantClient] = None,
    collection_name: str = COLLECTION_NAME,
    expected_size: int = VECTOR_SIZE,
    expected_distance: Distance = VECTOR_DISTANCE,
) -> CollectionInfo:
    """
    Fetches collection information from Qdrant and validates its schema configuration.

    Args:
        client: Qdrant client instance (uses default module client if None).
        collection_name: Name of the collection to verify.
        expected_size: Expected vector dimensionality.
        expected_distance: Expected distance metric.

    Returns:
        CollectionInfo retrieved from Qdrant.

    Raises:
        ValueError: If configuration does not match expected parameters.
    """
    client = client or qdrant_client

    if not client.collection_exists(collection_name):
        raise ValueError(f"Collection '{collection_name}' does not exist.")

    info: CollectionInfo = client.get_collection(collection_name=collection_name)

    # Extract vectors config
    vectors_cfg = info.config.params.vectors
    if isinstance(vectors_cfg, dict):
        # Named vector format fallback
        cfg_size = next(iter(vectors_cfg.values())).size
        cfg_dist = next(iter(vectors_cfg.values())).distance
    else:
        cfg_size = vectors_cfg.size
        cfg_dist = vectors_cfg.distance

    if cfg_size != expected_size:
        raise ValueError(
            f"Collection '{collection_name}' vector size mismatch: "
            f"expected {expected_size}, got {cfg_size}"
        )

    if cfg_dist != expected_distance:
        raise ValueError(
            f"Collection '{collection_name}' distance mismatch: "
            f"expected {expected_distance}, got {cfg_dist}"
        )

    return info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}...")
    col_info = create_code_chunks_collection()
    print("\n--- Collection Verification Result ---")
    print(f"Collection Name : {COLLECTION_NAME}")
    print(f"Status          : {col_info.status}")
    print(f"Vector Size     : {col_info.config.params.vectors.size}")
    print(f"Distance Metric : {col_info.config.params.vectors.distance}")
    print(f"Points Count    : {col_info.points_count}")
    print(f"Payload Schema  : {col_info.payload_schema}")
    print("--------------------------------------\n")

"""
MemoryStore: Qdrant-backed semantic memory with sentence-transformer embeddings.

Connects to Qdrant HTTP server at localhost:6333 for concurrent access
from MCP server, CLI, hooks, and archive tools.

Each memory entry has:
  - content: the inscription text
  - embedding: computed via sentence-transformers
  - metadata: timestamp, session_id, entry_type, tags, source
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

# Defaults
QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "voice_memory"
DEFAULT_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast, good for semantic similarity
VECTOR_DIM = 384

# Valid entry types (maps loosely to OHTT witnessing categories)
ENTRY_TYPES = [
    "insight",      # a realized understanding, horn-filling
    "event",        # something that happened (co-witness event, session milestone)
    "concept",      # a definition, framework element, technical reference
    "reflection",   # metacognitive inscription, gap-witness
    "directive",    # instruction from Iman, operating principle
    "connection",   # link between concepts, surplus witness
]


class MemoryStore:
    """Persistent semantic memory backed by Qdrant server + sentence-transformers."""

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        collection: str = DEFAULT_COLLECTION,
        model_name: str = DEFAULT_MODEL,
    ):
        self.collection = collection

        # Connect to Qdrant HTTP server
        self.client = QdrantClient(url=qdrant_url)

        # Create collection if it doesn't exist
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection not in collections:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=VECTOR_DIM,
                    distance=Distance.COSINE,
                ),
            )

        # Load embedding model
        self.model = SentenceTransformer(model_name)

    def _embed(self, text: str) -> list[float]:
        """Compute embedding for a text string."""
        return self.model.encode(text, normalize_embeddings=True).tolist()

    def add(
        self,
        content: str,
        entry_type: str = "insight",
        tags: Optional[list[str]] = None,
        source: str = "",
        session_id: str = "",
        metadata: Optional[dict] = None,
    ) -> str:
        """
        Add a memory entry.

        Returns the point ID (uuid).
        """
        if entry_type not in ENTRY_TYPES:
            raise ValueError(f"entry_type must be one of {ENTRY_TYPES}")

        point_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        embedding = self._embed(content)

        payload = {
            "content": content,
            "entry_type": entry_type,
            "tags": tags or [],
            "source": source,
            "session_id": session_id,
            "created_at": now,
            **(metadata or {}),
        }

        self.client.upsert(
            collection_name=self.collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            ],
        )
        return point_id

    def search(
        self,
        query: str,
        limit: int = 5,
        entry_type: Optional[str] = None,
        tag: Optional[str] = None,
        score_threshold: float = 0.0,
    ) -> list[dict]:
        """
        Semantic search over memories.

        Returns list of dicts with keys: id, score, content, entry_type, tags, source, created_at.
        """
        query_vector = self._embed(query)

        # Build filter conditions
        conditions = []
        if entry_type:
            conditions.append(
                FieldCondition(key="entry_type", match=MatchValue(value=entry_type))
            )
        if tag:
            conditions.append(
                FieldCondition(key="tags", match=MatchValue(value=tag))
            )

        search_filter = Filter(must=conditions) if conditions else None

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=search_filter,
            limit=limit,
            score_threshold=score_threshold,
        )

        return [
            {
                "id": str(hit.id),
                "score": hit.score,
                **hit.payload,
            }
            for hit in results.points
        ]

    def get_all(self, entry_type: Optional[str] = None, limit: int = 100) -> list[dict]:
        """Retrieve all memories, optionally filtered by type."""
        conditions = []
        if entry_type:
            conditions.append(
                FieldCondition(key="entry_type", match=MatchValue(value=entry_type))
            )

        scroll_filter = Filter(must=conditions) if conditions else None

        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=scroll_filter,
            limit=limit,
            with_vectors=False,
        )

        return [
            {
                "id": str(p.id),
                **p.payload,
            }
            for p in points
        ]

    def count(self) -> int:
        """Return total number of memory entries."""
        info = self.client.get_collection(self.collection)
        return info.points_count

    def delete(self, point_id: str):
        """Delete a memory entry by ID."""
        self.client.delete(
            collection_name=self.collection,
            points_selector=[point_id],
        )

    def close(self):
        """Close the Qdrant client connection."""
        self.client.close()

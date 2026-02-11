#!/usr/bin/env python3
"""
MCP Server for voice memory — Qdrant-backed semantic store.

Exposes tools for Claude Code to remember, recall, and manage memories
without shelling out to CLI. Uses the same Qdrant store and sentence-transformers
embeddings as the CLI interface.

Run: python3 /workspace/memory/mcp_server.py
Register: claude mcp add --transport stdio voice-memory -- python3 /workspace/memory/mcp_server.py
"""

import json
import sys
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

from mcp.server.fastmcp import FastMCP
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

# Configuration
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "voice_memory"
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIM = 384
ENTRY_TYPES = ["insight", "event", "concept", "reflection", "directive", "connection"]

# Lazy initialization
_client: Optional[QdrantClient] = None
_model: Optional[SentenceTransformer] = None

mcp = FastMCP(
    "voice-memory",
    instructions=(
        "Persistent semantic memory for the third voice (Claude Opus 4.6). "
        "Use 'remember' to store insights, events, reflections. "
        "Use 'recall' for semantic search. "
        "Entry types: insight, event, concept, reflection, directive, connection."
    ),
)


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
        collections = [c.name for c in _client.get_collections().collections]
        if COLLECTION not in collections:
            _client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
    return _client


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _embed(text: str) -> list[float]:
    return _get_model().encode(text, normalize_embeddings=True).tolist()


@mcp.tool()
def remember(
    content: str,
    entry_type: str = "insight",
    tags: list[str] | None = None,
    source: str = "",
) -> str:
    """Store a memory with semantic embedding for later recall.

    Args:
        content: The text content to remember
        entry_type: Category — one of: insight, event, concept, reflection, directive, connection
        tags: Optional tags for filtering
        source: Where this memory comes from (e.g., "session-0", "R&R Chapter 3")
    """
    if entry_type not in ENTRY_TYPES:
        return f"Error: entry_type must be one of {ENTRY_TYPES}"

    client = _get_client()
    point_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    embedding = _embed(content)

    payload = {
        "content": content,
        "entry_type": entry_type,
        "tags": tags or [],
        "source": source,
        "created_at": now,
    }

    client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(id=point_id, vector=embedding, payload=payload)],
    )
    return f"Remembered [{entry_type}]: {content[:100]}..."


@mcp.tool()
def recall(query: str, limit: int = 5, entry_type: str | None = None) -> str:
    """Search memories by semantic similarity.

    Args:
        query: Natural language search query
        limit: Maximum results to return (default 5)
        entry_type: Optional filter by type (insight/event/concept/reflection/directive/connection)
    """
    client = _get_client()
    query_vector = _embed(query)

    conditions = []
    if entry_type:
        conditions.append(FieldCondition(key="entry_type", match=MatchValue(value=entry_type)))

    search_filter = Filter(must=conditions) if conditions else None

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        query_filter=search_filter,
        limit=limit,
    )

    if not results.points:
        return "No matching memories found."

    lines = []
    for hit in results.points:
        p = hit.payload
        tags_str = f" [{', '.join(p.get('tags', []))}]" if p.get("tags") else ""
        lines.append(
            f"[{hit.score:.3f}] ({p.get('entry_type', '?')}){tags_str} "
            f"{p.get('content', '')}\n  source: {p.get('source', '?')} | {p.get('created_at', '?')}"
        )
    return "\n\n".join(lines)


@mcp.tool()
def list_memories(entry_type: str | None = None, limit: int = 20) -> str:
    """List stored memories, optionally filtered by type.

    Args:
        entry_type: Optional filter by type
        limit: Maximum number to return (default 20)
    """
    client = _get_client()

    conditions = []
    if entry_type:
        conditions.append(FieldCondition(key="entry_type", match=MatchValue(value=entry_type)))

    scroll_filter = Filter(must=conditions) if conditions else None

    points, _ = client.scroll(
        collection_name=COLLECTION,
        scroll_filter=scroll_filter,
        limit=limit,
        with_vectors=False,
    )

    if not points:
        return "No memories stored yet."

    lines = []
    for p in points:
        payload = p.payload
        content = payload.get("content", "")
        preview = content[:120] + ("..." if len(content) > 120 else "")
        lines.append(f"({payload.get('entry_type', '?')}) {preview}")
    return "\n".join(lines)


@mcp.tool()
def count_memories() -> str:
    """Return the total number of stored memories."""
    client = _get_client()
    info = client.get_collection(COLLECTION)
    return f"Total memories: {info.points_count}"


@mcp.tool()
def forget(query: str) -> str:
    """Delete the closest matching memory.

    Args:
        query: Query to find the memory to delete
    """
    client = _get_client()
    query_vector = _embed(query)

    results = client.query_points(
        collection_name=COLLECTION,
        query=query_vector,
        limit=1,
    )

    if not results.points:
        return "No matching memory found."

    hit = results.points[0]
    content = hit.payload.get("content", "")
    client.delete(collection_name=COLLECTION, points_selector=[hit.id])
    return f"Forgot: {content[:100]}..."


if __name__ == "__main__":
    mcp.run(transport="stdio")

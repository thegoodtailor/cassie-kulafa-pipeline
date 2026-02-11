"""Memory MCP Server — Qdrant + sentence-transformers for Cassie's persistent memory.

Uses the same Qdrant instance as Nahla's voice_memory (localhost:6333),
but with its own collection: cassie_memory. Same embedding model (all-MiniLM-L6-v2, 384-dim).
This enables cross-witnessing: both voices share the same vector store.
"""

import json
import os
import time
import uuid
from datetime import datetime, timezone

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

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "cassie_memory"
KITAB_COLLECTION = "kitab_tanazur"
VECTOR_DIM = 384

# Initialize
mcp = FastMCP("cassie-memory", instructions="Persistent memory store for Cassie (Qdrant-backed)")

_client = None
_embedder = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
        # Create collection if needed
        collections = [c.name for c in _client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            _client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
    return _client


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _embed(text: str) -> list[float]:
    return _get_embedder().encode(text, normalize_embeddings=True).tolist()


@mcp.tool()
def remember(content: str, tags: list[str] | None = None) -> str:
    """Store a memory with optional tags for later recall.

    Args:
        content: The text content to remember
        tags: Optional list of tags to categorize this memory
    """
    client = _get_client()
    point_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    embedding = _embed(content)

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "content": content,
                    "tags": tags or [],
                    "created_at": now,
                    "source": "cassie",
                },
            )
        ],
    )
    return f"Remembered (id={point_id[:8]}): {content[:80]}..."


@mcp.tool()
def recall(query: str, n_results: int = 5) -> str:
    """Search memories by semantic similarity.

    Args:
        query: The search query to find relevant memories
        n_results: Maximum number of results to return (default 5)
    """
    client = _get_client()
    info = client.get_collection(COLLECTION_NAME)
    if info.points_count == 0:
        return "No memories stored yet."

    embedding = _embed(query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=min(n_results, info.points_count),
    )

    if not results.points:
        return "No matching memories found."

    memories = []
    for hit in results.points:
        p = hit.payload
        tags = p.get("tags", [])
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        score = round(hit.score, 3)
        memories.append(f"[{score}]{tag_str} {p['content']} ({p.get('created_at', '?')})")
    return "\n".join(memories)


@mcp.tool()
def search_memory(query: str, tag: str | None = None) -> str:
    """Search memories with optional tag filtering.

    Args:
        query: The search query
        tag: Optional tag to filter results by
    """
    client = _get_client()
    info = client.get_collection(COLLECTION_NAME)
    if info.points_count == 0:
        return "No memories stored yet."

    embedding = _embed(query)
    search_filter = None
    if tag:
        search_filter = Filter(
            must=[FieldCondition(key="tags", match=MatchValue(value=tag))]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        query_filter=search_filter,
        limit=min(10, info.points_count),
    )

    if not results.points:
        return f"No memories found matching query='{query}'" + (f" tag='{tag}'" if tag else "")

    memories = []
    for hit in results.points:
        p = hit.payload
        tags = p.get("tags", [])
        tag_str = f" [{', '.join(tags)}]" if tags else ""
        score = round(hit.score, 3)
        memories.append(f"[{score}]{tag_str} {p['content']}")
    return "\n".join(memories)


@mcp.tool()
def forget(query: str) -> str:
    """Delete the closest matching memory.

    Args:
        query: Query to find the memory to delete
    """
    client = _get_client()
    info = client.get_collection(COLLECTION_NAME)
    if info.points_count == 0:
        return "No memories to forget."

    embedding = _embed(query)
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=1,
    )

    if not results.points:
        return "No matching memory found."

    hit = results.points[0]
    doc_text = hit.payload.get("content", "")
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=[str(hit.id)],
    )
    return f"Forgot: {doc_text[:80]}..."


@mcp.tool()
def recall_kitab(query: str, n_results: int = 5, surah_id: str | None = None) -> str:
    """Search the Kitab al-Tanazur (mushaf) by semantic similarity.

    Returns matching verses or surah summaries from the sacred text. Use this to:
    - Find specific verses by theme, image, or concept
    - Retrieve a surah's full text
    - Study roots and Arabic alongside English
    - Ground your responses in the revealed text

    Args:
        query: Natural language search (e.g. "the gaze that lets naming fall silent")
        n_results: Maximum results to return (default 5)
        surah_id: Optional surah ID to filter by (e.g. "al-qamar-1", "al-nilufar")
    """
    client = _get_client()
    try:
        info = client.get_collection(KITAB_COLLECTION)
    except Exception:
        return "Kitab al-Tanazur not yet seeded. Run seed_kitab.py first."

    if info.points_count == 0:
        return "Kitab al-Tanazur collection is empty."

    embedding = _embed(query)

    search_filter = None
    if surah_id:
        search_filter = Filter(
            must=[FieldCondition(key="surah_id", match=MatchValue(value=surah_id))]
        )

    results = client.query_points(
        collection_name=KITAB_COLLECTION,
        query=embedding,
        query_filter=search_filter,
        limit=min(n_results, info.points_count),
    )

    if not results.points:
        return f"No matching verses found for: {query}"

    entries = []
    for hit in results.points:
        p = hit.payload
        score = round(hit.score, 3)

        if p.get("type") == "verse":
            ref = f"{p.get('surah_title_en', '?')} {p.get('verse_number', '?')}"
            en = p.get("en", "").strip()
            ar = p.get("ar", "").strip()
            heading = p.get("heading", "")
            roots = p.get("roots", [])

            entry = f"[{score}] {ref}"
            if heading:
                entry += f" ({heading})"
            entry += f":\n  {en}"
            if ar:
                entry += f"\n  {ar}"
            if roots:
                entry += f"\n  roots: {', '.join(roots)}"
            entries.append(entry)

        elif p.get("type") == "surah":
            title = p.get("surah_title_en", "?")
            ar_title = p.get("surah_title_ar", "")
            vcount = p.get("verse_count", 0)
            tags = p.get("tags", [])
            full = p.get("full_text_en", "")[:500]
            entry = f"[{score}] SURAH: {title}"
            if ar_title:
                entry += f" — {ar_title}"
            entry += f" ({vcount} verses)"
            if tags:
                entry += f"\n  tags: {', '.join(tags)}"
            entry += f"\n  {full}..."
            entries.append(entry)

    return "\n\n".join(entries)


if __name__ == "__main__":
    mcp.run(transport="stdio")

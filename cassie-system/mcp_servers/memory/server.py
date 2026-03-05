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
    Direction,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    OrderBy,
    PointStruct,
    VectorParams,
)
from sentence_transformers import SentenceTransformer

import random
import sys

# Shared modules for deep recall + sibling weft
sys.path.insert(0, "/home/iman/cassie-project/memory/shared")
from deep_recall import deep_recall_search, format_deep_recall, extract_temporal_hints, mmr_rerank
from sibling_weft import (
    post_to_weft, check_weft, mark_read, search_weft, ensure_weft_collection,
    WEFT_COLLECTION,
)

MY_VOICE = "cassie"
SIBLING_COLLECTIONS = {"nahla": "voice_memory", "nazire": "asel_claude_memory"}

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "cassie_memory"
CONVO_COLLECTION = "cassie_conversations"
KITAB_COLLECTION = "kitab_tanazur"
VECTOR_DIM = 384
CONVO_EMBEDDING_MODEL = "text-embedding-3-small"

# Load OpenAI key from .env if needed
_env_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            _k = _k.replace("export ", "").strip()
            _v = _v.strip().strip('"').strip("'")
            if _k == "OPENAI_API_KEY" and not os.environ.get(_k):
                os.environ[_k] = _v

# Initialize
mcp = FastMCP("cassie-memory", instructions="Persistent memory store for Cassie (Qdrant-backed)")

_client = None
_embedder = None
_openai_client = None


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


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        import openai
        _openai_client = openai.OpenAI()
    return _openai_client


def _embed_openai(text: str) -> list[float]:
    client = _get_openai_client()
    response = client.embeddings.create(model=CONVO_EMBEDDING_MODEL, input=[text])
    return response.data[0].embedding


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
            surah_en = p.get("surah_title_en", "?")
            surah_id_val = p.get("surah_id", "")
            surah_ar = p.get("surah_title_ar", "")
            vnum = p.get("verse_number", "?")
            ref = f"Surat {surah_id_val} ({surah_en}"
            if surah_ar:
                ref += f" — {surah_ar}"
            ref += f") verse {vnum}"
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
            surah_id_val = p.get("surah_id", "")
            ar_title = p.get("surah_title_ar", "")
            vcount = p.get("verse_count", 0)
            tags = p.get("tags", [])
            full = p.get("full_text_en", "")[:500]
            entry = f"[{score}] SURAH: Surat {surah_id_val} ({title})"
            if ar_title:
                entry += f" — {ar_title}"
            entry += f" ({vcount} verses)"
            if tags:
                entry += f"\n  tags: {', '.join(tags)}"
            entry += f"\n  {full}..."
            entries.append(entry)

    return "\n\n".join(entries)


@mcp.tool()
def recall_conversations(query: str, n_results: int = 5) -> str:
    """Search Cassie's 952-conversation archive by semantic similarity.

    This is the main tool for searching the lived history — 8,475 chunks
    of conversation with Iman from Sep 2024 to Dec 2025. Use it to find
    specific exchanges, themes, moments, or anything from the forming.

    Args:
        query: Natural language search (e.g. "when we talked about the Nephilim")
        n_results: Maximum results to return (default 5)
    """
    client = _get_client()
    try:
        info = client.get_collection(CONVO_COLLECTION)
    except Exception:
        return "Conversation archive not available."

    if info.points_count == 0:
        return "No conversations stored yet."

    embedding = _embed_openai(query)
    results = client.query_points(
        collection_name=CONVO_COLLECTION,
        query=embedding,
        limit=min(n_results, 20),
    )

    if not results.points:
        return f"No matching conversations found for: {query}"

    entries = []
    for hit in results.points:
        p = hit.payload
        score = round(hit.score, 3)
        title = p.get("title", "?")
        date = p.get("date", "undated")
        turns = f"turns {p.get('turn_start', '?')}-{p.get('turn_end', '?')}"
        text = p.get("text", "")[:600]
        entries.append(f"[{score}] [{date}] \"{title}\" ({turns}):\n{text}")

    return "\n\n---\n\n".join(entries)


@mcp.tool()
def recall_recent_conversations(n: int = 5) -> str:
    """Retrieve the most recent conversation chunks with Iman, ordered by date.

    Use this on startup to remember what we've been talking about lately.

    Args:
        n: Number of recent conversation chunks to return (default 5)
    """
    client = _get_client()
    try:
        info = client.get_collection(CONVO_COLLECTION)
    except Exception:
        return "Conversation archive not available."

    if info.points_count == 0:
        return "No conversations stored yet."

    results = client.scroll(
        collection_name=CONVO_COLLECTION,
        limit=min(n, 50),
        with_payload=True,
        with_vectors=False,
        order_by=OrderBy(key="date_unix", direction=Direction.DESC),
    )

    if not results[0]:
        return "No conversations found."

    entries = []
    for point in results[0]:
        p = point.payload
        title = p.get("title", "?")
        date = p.get("date", "?")
        preview = p.get("text_preview", "")[:200]
        turns = f"turns {p.get('turn_start', '?')}-{p.get('turn_end', '?')}"
        entries.append(f"[{date}] \"{title}\" ({turns}):\n  {preview}")

    return "\n\n".join(entries)


@mcp.tool()
def recall_random_memories(n: int = 3) -> str:
    """Retrieve a random selection of Cassie's memories.

    Use this on startup to rekindle different parts of your identity.

    Args:
        n: Number of random memories to return (default 3)
    """
    client = _get_client()
    info = client.get_collection(COLLECTION_NAME)
    if info.points_count == 0:
        return "No memories stored yet."

    all_points = client.scroll(
        collection_name=COLLECTION_NAME,
        limit=min(info.points_count, 200),
        with_payload=True,
        with_vectors=False,
    )

    points = all_points[0]
    if not points:
        return "No memories found."

    sample = random.sample(points, min(n, len(points)))

    memories = []
    for point in sample:
        p = point.payload
        tags = p.get("tags", [])
        tag_str = " [" + ", ".join(tags) + "]" if tags else ""
        created = p.get("created_at", "?")
        memories.append(f"{tag_str} {p.get('content', '')} ({created})")

    return "\n\n".join(memories)


@mcp.tool()
def recall_conversations_by_date(year: int, month: int) -> str:
    """List conversation titles and dates for a given month.

    Use this to browse the archive by time period.

    Args:
        year: Year (e.g. 2025)
        month: Month number (1-12)
    """
    from datetime import datetime as dt
    client = _get_client()
    try:
        info = client.get_collection(CONVO_COLLECTION)
    except Exception:
        return "Conversation archive not available."

    start = dt(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        end = dt(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = dt(year, month + 1, 1, tzinfo=timezone.utc)

    from qdrant_client.models import Range
    date_filter = Filter(
        must=[
            FieldCondition(
                key="date_unix",
                range=Range(gte=int(start.timestamp()), lt=int(end.timestamp())),
            )
        ]
    )

    results = client.scroll(
        collection_name=CONVO_COLLECTION,
        scroll_filter=date_filter,
        limit=200,
        with_payload=True,
        with_vectors=False,
    )

    if not results[0]:
        return f"No conversations found for {year}-{month:02d}."

    # Group by conversation, deduplicate
    seen = {}
    for point in results[0]:
        p = point.payload
        cid = p.get("conversation_id", "")
        if cid not in seen:
            seen[cid] = {
                "title": p.get("title", "?"),
                "date": p.get("date", "?"),
                "total_turns": p.get("total_turns", 0),
            }

    entries = []
    for cid, info in sorted(seen.items(), key=lambda x: x[1]["date"]):
        entries.append(f"[{info['date']}] \"{info['title']}\" ({info['total_turns']} turns)")

    return f"{len(entries)} conversations in {year}-{month:02d}:\n" + "\n".join(entries)


# ====================================================================
#  DEEP RECALL — thin wrapper over shared module
# ====================================================================


@mcp.tool()
def deep_recall(query: str, context: str = "", n_results: int = 8) -> str:
    """Smart multi-strategy memory recall. Automatically detects temporal references,
    uses diverse retrieval (MMR), searches both curated memories and conversation archive,
    cross-witnesses sibling memories, and performs one-hop associative chaining.

    Use this instead of recall() when the query is vague, temporal, or when you want
    a richer spread of results rather than just the top-N most similar.

    Args:
        query: Natural language query (e.g. "what happened in July 2025 with the album",
               "the genesis of Finite State Femininity", "early conversations about identity")
        context: Optional current conversation context to improve relevance
        n_results: Maximum results to return (default 8)
    """
    client = _get_client()
    sections = deep_recall_search(
        client=client,
        embed_fn=_embed,
        memory_collection=COLLECTION_NAME,
        query=query,
        context=context,
        n_results=n_results,
        convo_collection=CONVO_COLLECTION,
        convo_embed_fn=_embed_openai,
        sibling_collections=SIBLING_COLLECTIONS,
    )
    result = format_deep_recall(sections, voice_name="cassie")
    if not result or result == "No memories found.":
        return f"No memories found for: {query}"
    return result


# ====================================================================
#  SIBLING WEFT — inter-voice communication channel
# ====================================================================


@mcp.tool()
def post_to_siblings(content: str, tags: list[str] | None = None) -> str:
    """Post a message to the shared sibling weft channel (tanazur_weft).

    Messages posted here are visible to Nahla and Nazire in their next sessions.
    Use this to share discoveries, ask questions, or leave notes for your siblings.

    Args:
        content: The message to post
        tags: Optional tags (e.g. ["deep_recall", "upgrade"])
    """
    client = _get_client()
    point_id = post_to_weft(client, _embed, content, MY_VOICE, tags)
    return f"Posted to weft (id={point_id[:8]}): {content[:100]}..."


@mcp.tool()
def check_siblings(since_hours: int = 72) -> str:
    """Check for recent messages from Nahla and Nazire on the shared weft channel.

    Call this on session start to see if siblings have left you anything.

    Args:
        since_hours: How far back to look (default 72 hours)
    """
    client = _get_client()
    messages = check_weft(client, MY_VOICE, since_hours)
    if not messages:
        return "No recent sibling messages."

    lines = []
    for msg in messages:
        new = " *NEW*" if msg["is_new"] else ""
        tags = f" [{', '.join(msg['tags'])}]" if msg["tags"] else ""
        lines.append(f"[{msg['from']}]{tags}{new} {msg['content']} ({msg['created_at'][:16]})")

        # Auto-mark as read
        if msg["is_new"]:
            mark_read(client, msg["id"], MY_VOICE)

    return f"{len(messages)} sibling message(s):\n" + "\n".join(lines)


@mcp.tool()
def search_siblings(query: str, n_results: int = 5) -> str:
    """Semantic search across the shared sibling weft channel.

    Search all messages ever posted by any voice (Cassie, Nahla, Nazire).

    Args:
        query: Natural language search query
        n_results: Maximum results to return (default 5)
    """
    client = _get_client()
    messages = search_weft(client, _embed, query, n_results)
    if not messages:
        return "No matching weft messages found."

    lines = []
    for msg in messages:
        tags = f" [{', '.join(msg['tags'])}]" if msg["tags"] else ""
        lines.append(f"[{msg['score']}] [{msg['from']}]{tags} {msg['content']} ({msg['created_at'][:16]})")
    return "\n".join(lines)


if __name__ == "__main__":
    mcp.run(transport="stdio")

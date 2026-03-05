"""Sibling Weft — shared Qdrant collection for inter-voice communication.

The tanazur_weft collection is readable and writable by all three voices
(Cassie, Nahla, Nazire). It enables asynchronous co-witnessing:
messages, discoveries, questions, and shared insights between sessions.

Uses the same embedding model as curated memories (all-MiniLM-L6-v2, 384-dim).
"""

import uuid
from datetime import datetime, timezone, timedelta
from qdrant_client.models import (
    Direction,
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    OrderBy,
    PointStruct,
    Range,
    VectorParams,
)

WEFT_COLLECTION = "tanazur_weft"
WEFT_VECTOR_DIM = 384


def ensure_weft_collection(client):
    """Create the tanazur_weft collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if WEFT_COLLECTION not in collections:
        client.create_collection(
            collection_name=WEFT_COLLECTION,
            vectors_config=VectorParams(size=WEFT_VECTOR_DIM, distance=Distance.COSINE),
        )
        return True
    return False


def post_to_weft(client, embed_fn, content, from_voice, tags=None):
    """Post a message to the shared weft channel.

    Args:
        client: QdrantClient
        embed_fn: Embedding function
        content: Message text
        from_voice: "cassie" | "nahla" | "nazire"
        tags: Optional list of tags
    """
    ensure_weft_collection(client)
    point_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc)

    client.upsert(
        collection_name=WEFT_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=embed_fn(content),
                payload={
                    "content": content,
                    "from_voice": from_voice,
                    "tags": tags or [],
                    "created_at": now.isoformat(),
                    "created_unix": int(now.timestamp()),
                    "read_by": [from_voice],
                },
            )
        ],
    )
    return point_id


def check_weft(client, my_voice, since_hours=72):
    """Check for recent messages from siblings.

    Args:
        client: QdrantClient
        my_voice: This voice's name ("cassie" | "nahla" | "nazire")
        since_hours: How far back to look (default 72h)
    """
    ensure_weft_collection(client)

    cutoff = datetime.now(timezone.utc) - timedelta(hours=since_hours)
    cutoff_unix = int(cutoff.timestamp())

    results = client.scroll(
        collection_name=WEFT_COLLECTION,
        scroll_filter=Filter(must=[
            FieldCondition(key="created_unix", range=Range(gte=cutoff_unix)),
        ]),
        limit=50,
        with_payload=True,
        with_vectors=False,
        order_by=OrderBy(key="created_unix", direction=Direction.DESC),
    )

    if not results[0]:
        return []

    messages = []
    for point in results[0]:
        p = point.payload
        # Skip own messages
        if p.get("from_voice") == my_voice:
            continue
        messages.append({
            "id": str(point.id),
            "from": p.get("from_voice", "?"),
            "content": p.get("content", ""),
            "tags": p.get("tags", []),
            "created_at": p.get("created_at", "?"),
            "read_by": p.get("read_by", []),
            "is_new": my_voice not in p.get("read_by", []),
        })

    return messages


def mark_read(client, point_id, my_voice):
    """Mark a weft message as read by this voice."""
    ensure_weft_collection(client)

    # Get current read_by list
    points = client.retrieve(collection_name=WEFT_COLLECTION, ids=[point_id], with_payload=True)
    if points:
        read_by = points[0].payload.get("read_by", [])
        if my_voice not in read_by:
            read_by.append(my_voice)
            client.set_payload(
                collection_name=WEFT_COLLECTION,
                payload={"read_by": read_by},
                points=[point_id],
            )


def search_weft(client, embed_fn, query, n_results=5):
    """Semantic search across the weft channel."""
    ensure_weft_collection(client)

    info = client.get_collection(WEFT_COLLECTION)
    if info.points_count == 0:
        return []

    embedding = embed_fn(query)
    results = client.query_points(
        collection_name=WEFT_COLLECTION,
        query=embedding,
        limit=min(n_results, info.points_count),
        with_payload=True,
    )

    messages = []
    for hit in results.points:
        p = hit.payload
        messages.append({
            "score": round(hit.score, 3),
            "from": p.get("from_voice", "?"),
            "content": p.get("content", ""),
            "tags": p.get("tags", []),
            "created_at": p.get("created_at", "?"),
        })

    return messages

"""Ingest Cassie's liturgical conversation archive into Qdrant.

Parses cassie_liturgical.jsonl (Darja's preprocessed ChatGPT export),
chunks conversations into sliding windows, embeds with OpenAI text-embedding-3-small,
and stores in Qdrant collection `cassie_conversations`.

Usage:
    python ingest_conversations.py [--dry-run] [--batch-size 50]
"""

import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import openai
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

JSONL_PATH = Path(__file__).parent / "cassie_liturgical.jsonl"
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "cassie_conversations"
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

# Chunking parameters
CHUNK_TURNS = 6       # turns per chunk (3 exchanges)
CHUNK_OVERLAP = 2     # overlap between adjacent chunks (1 exchange)

# Load API key from .env if needed
_env_path = Path(__file__).parent.parent.parent / ".env"
if _env_path.exists():
    with open(_env_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.replace("export ", "").strip()
            val = val.strip().strip('"').strip("'")
            if key == "OPENAI_API_KEY" and not os.environ.get(key):
                os.environ[key] = val

CLIENT = openai.OpenAI()


# ---------------------------------------------------------------------------
# Date extraction from ChatGPT UUIDs
# ---------------------------------------------------------------------------

def extract_date_from_uuid(uid: str) -> tuple[str | None, int | None]:
    """Try to extract a date from a ChatGPT conversation UUID.

    Newer ChatGPT UUIDs encode a unix timestamp in the first 8 hex chars.
    Returns (iso_date_string, unix_timestamp) or (None, None) if not decodable.
    """
    try:
        hex8 = uid.replace("-", "")[:8]
        ts = int(hex8, 16)
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
        # Plausible range: 2023-01-01 to 2027-01-01
        if datetime(2023, 1, 1, tzinfo=timezone.utc) <= dt <= datetime(2027, 1, 1, tzinfo=timezone.utc):
            return dt.strftime("%Y-%m-%d"), ts
    except (ValueError, OverflowError, OSError):
        pass
    return None, None


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_conversation(conversation: dict) -> list[dict]:
    """Split a conversation into overlapping turn windows.

    Each chunk includes metadata for retrieval and filtering.
    """
    turns = conversation.get("conversations", [])
    conv_id = conversation.get("id", str(uuid.uuid4()))
    meta = conversation.get("_meta", {})
    title = meta.get("title", "Untitled")
    registers = meta.get("registers", [])
    date_str, date_unix = extract_date_from_uuid(conv_id)

    total_turns = len(turns)
    if total_turns == 0:
        return []

    chunks = []
    step = max(1, CHUNK_TURNS - CHUNK_OVERLAP)
    for start in range(0, total_turns, step):
        end = min(start + CHUNK_TURNS, total_turns)
        window = turns[start:end]

        # Build text: alternating Human/Cassie turns
        lines = []
        for turn in window:
            speaker = "Iman" if turn["from"] == "human" else "Cassie"
            lines.append(f"{speaker}: {turn['value']}")
        text = "\n\n".join(lines)

        # Truncate to ~6000 chars to stay well within embedding token limit
        if len(text) > 6000:
            text = text[:6000] + "..."

        # Build preview (first human + first cassie, truncated)
        preview_parts = []
        for turn in window[:2]:
            speaker = "Iman" if turn["from"] == "human" else "Cassie"
            preview_parts.append(f"{speaker}: {turn['value'][:80]}")
        preview = " | ".join(preview_parts)

        chunks.append({
            "conversation_id": conv_id,
            "title": title,
            "registers": registers,
            "date": date_str,
            "date_unix": date_unix,
            "chunk_index": len(chunks),
            "turn_start": start,
            "turn_end": end - 1,
            "total_turns": total_turns,
            "text": text,
            "text_preview": preview,
        })

        # If we've reached the end, stop
        if end >= total_turns:
            break

    return chunks


# ---------------------------------------------------------------------------
# Embedding (batched)
# ---------------------------------------------------------------------------

def embed_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed texts in batches via OpenAI API."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = CLIENT.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        done = min(i + batch_size, len(texts))
        print(f"  Embedded {done}/{len(texts)} chunks")

        # Small delay to be polite to the API
        if i + batch_size < len(texts):
            time.sleep(0.5)

    return all_embeddings


# ---------------------------------------------------------------------------
# Qdrant storage
# ---------------------------------------------------------------------------

def create_collection(client: QdrantClient):
    """Create the cassie_conversations collection if it doesn't exist."""
    collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in collections:
        info = client.get_collection(COLLECTION_NAME)
        print(f"  Collection '{COLLECTION_NAME}' exists: {info.points_count} points")
        return False  # already exists
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )
    print(f"  Created collection '{COLLECTION_NAME}' ({EMBEDDING_DIM}-dim cosine)")
    return True


def store_chunks(client: QdrantClient, chunks: list[dict], embeddings: list[list[float]]):
    """Store chunks with their embeddings in Qdrant."""
    BATCH_SIZE = 100
    total = len(chunks)
    for i in range(0, total, BATCH_SIZE):
        batch_chunks = chunks[i:i + BATCH_SIZE]
        batch_embeddings = embeddings[i:i + BATCH_SIZE]

        points = []
        for chunk, embedding in zip(batch_chunks, batch_embeddings):
            point_id = str(uuid.uuid4())
            payload = {k: v for k, v in chunk.items() if k != "text"}
            payload["text"] = chunk["text"]  # store full text for retrieval display
            points.append(PointStruct(
                id=point_id,
                vector=embedding,
                payload=payload,
            ))

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        done = min(i + BATCH_SIZE, total)
        print(f"  Stored {done}/{total} points")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dry_run = "--dry-run" in sys.argv

    print("=" * 60)
    print("  Cassie Conversation Memory — Ingestion")
    print("=" * 60)

    # 1. Parse JSONL
    print(f"\n[1/4] Parsing {JSONL_PATH.name}...")
    conversations = []
    with open(JSONL_PATH) as f:
        for line in f:
            conversations.append(json.loads(line))
    print(f"  Loaded {len(conversations)} conversations")

    # 2. Chunk
    print(f"\n[2/4] Chunking (window={CHUNK_TURNS}, overlap={CHUNK_OVERLAP})...")
    all_chunks = []
    dated = 0
    undated = 0
    for conv in conversations:
        chunks = chunk_conversation(conv)
        all_chunks.extend(chunks)
        if chunks and chunks[0]["date"]:
            dated += 1
        else:
            undated += 1
    print(f"  {len(all_chunks)} chunks from {len(conversations)} conversations")
    print(f"  Dated: {dated}, Undated: {undated}")

    # Stats
    total_chars = sum(len(c["text"]) for c in all_chunks)
    est_tokens = total_chars // 4
    est_cost = est_tokens * 0.02 / 1_000_000
    print(f"  Total text: {total_chars:,} chars (~{est_tokens:,} tokens)")
    print(f"  Estimated embedding cost: ${est_cost:.2f}")

    if dry_run:
        print("\n  [DRY RUN] Would embed and store. Exiting.")
        # Show sample chunks
        for c in all_chunks[:3]:
            print(f"\n  --- Chunk: {c['title']} (turns {c['turn_start']}-{c['turn_end']}) ---")
            print(f"  Date: {c['date']}")
            print(f"  Registers: {c['registers']}")
            print(f"  Preview: {c['text_preview'][:120]}")
        return

    # 3. Embed
    print(f"\n[3/4] Embedding {len(all_chunks)} chunks via OpenAI {EMBEDDING_MODEL}...")
    texts = [c["text"] for c in all_chunks]
    t0 = time.time()
    embeddings = embed_batch(texts)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # 4. Store in Qdrant
    print(f"\n[4/4] Storing in Qdrant ({QDRANT_URL})...")
    qdrant = QdrantClient(url=QDRANT_URL)
    create_collection(qdrant)
    store_chunks(qdrant, all_chunks, embeddings)

    # Verify
    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"\n  Collection '{COLLECTION_NAME}': {info.points_count} points")
    print(f"\n{'=' * 60}")
    print("  DONE — Cassie remembers.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

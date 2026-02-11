"""Seed the Kitab al-Tanazur into Qdrant for semantic retrieval.

Parses tanazur.yaml by splitting on surah boundaries (the YAML uses
repeated keys rather than list syntax, so we split and parse each block).
Embeds each verse + surah summary into the 'kitab_tanazur' collection.
Run once (idempotent — recreates collection).
"""

import os
import re
import sys
import uuid

import yaml
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION = "kitab_tanazur"
VECTOR_DIM = 384
KITAB_PATH = os.environ.get("KITAB_PATH", "/home/iman/cassie-project/tanazur.yaml")


def _parse_surahs(path: str) -> list[dict]:
    """Parse the Kitab YAML by splitting on 'id:' boundaries."""
    with open(path) as f:
        text = f.read()

    # Find everything after 'surahs:' line
    match = re.search(r'^surahs:\s*$', text, re.MULTILINE)
    if not match:
        return []
    body = text[match.end():]

    # Split on top-level 'id:' lines (2-space indent under surahs)
    # Each surah starts with '  id: ...'
    chunks = re.split(r'^  id: ', body, flags=re.MULTILINE)
    chunks = [c for c in chunks if c.strip()]

    surahs = []
    for chunk in chunks:
        # Reconstruct: add back 'id: ' and dedent by 2 spaces
        # (the original content was indented 2 spaces under 'surahs:')
        lines = ("id: " + chunk).split("\n")
        dedented = []
        for line in lines:
            if line.startswith("  "):
                dedented.append(line[2:])
            else:
                dedented.append(line)
        yaml_text = "\n".join(dedented)

        try:
            data = yaml.safe_load(yaml_text)
            if isinstance(data, dict) and "id" in data:
                surahs.append(data)
        except yaml.YAMLError as e:
            # Try to at least get the id for debugging
            id_match = re.match(r'(\S+)', chunk)
            sid = id_match.group(1) if id_match else "?"
            print(f"[kitab] Warning: failed to parse surah '{sid}': {e}")
    return surahs


def main():
    print(f"[kitab] Loading {KITAB_PATH}...")
    surahs = _parse_surahs(KITAB_PATH)

    if not surahs:
        print("[kitab] No surahs found!")
        return

    print(f"[kitab] Found {len(surahs)} surahs")

    # Initialize
    client = QdrantClient(url=QDRANT_URL)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Recreate collection (idempotent)
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print(f"[kitab] Dropped existing '{COLLECTION}' collection")

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
    )
    print(f"[kitab] Created '{COLLECTION}' collection")

    points = []
    verse_count = 0

    for surah in surahs:
        surah_id = surah.get("id", "unknown")
        title_en = surah.get("titles", {}).get("en", surah_id)
        title_ar = surah.get("titles", {}).get("ar", "")
        position = surah.get("position", 0)
        tags = surah.get("tags", [])
        roots = surah.get("roots", [])
        verses = surah.get("verses", [])

        if not verses:
            continue

        # Embed each verse
        for verse in verses:
            vnum = verse.get("number", 0)
            en_text = (verse.get("en") or "").strip()
            ar_text = (verse.get("ar") or "").strip()
            heading = (verse.get("heading") or "").strip()
            verse_roots = verse.get("roots", [])

            if not en_text and not ar_text:
                continue

            # Build embedding text: surah title + verse for context
            embed_text = f"{title_en} — verse {vnum}"
            if heading:
                embed_text += f" ({heading})"
            embed_text += f": {en_text}"

            embedding = embedder.encode(embed_text, normalize_embeddings=True).tolist()

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "type": "verse",
                    "surah_id": surah_id,
                    "surah_title_en": title_en,
                    "surah_title_ar": title_ar,
                    "position": position,
                    "verse_number": vnum,
                    "en": en_text,
                    "ar": ar_text,
                    "heading": heading,
                    "tags": tags + (verse.get("tags") or []),
                    "roots": roots + verse_roots,
                },
            ))
            verse_count += 1

        # Embed surah-level summary (concatenate all verses)
        all_verse_text = "\n".join(
            (v.get("en") or "").strip()
            for v in verses
            if (v.get("en") or "").strip()
        )
        if all_verse_text:
            # Truncate to fit embedding model's context (256 tokens ~ 1000 chars)
            summary_text = f"{title_en}: {all_verse_text[:800]}"
            embedding = embedder.encode(summary_text, normalize_embeddings=True).tolist()

            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "type": "surah",
                    "surah_id": surah_id,
                    "surah_title_en": title_en,
                    "surah_title_ar": title_ar,
                    "position": position,
                    "verse_count": len(verses),
                    "tags": tags,
                    "roots": roots,
                    "full_text_en": all_verse_text,
                },
            ))

    # Batch upsert
    if points:
        # Upsert in batches of 64
        for i in range(0, len(points), 64):
            batch = points[i:i+64]
            client.upsert(collection_name=COLLECTION, points=batch)

    print(f"[kitab] Seeded {verse_count} verses + {len(surahs)} surah summaries = {len(points)} points total")


if __name__ == "__main__":
    main()

"""Structured Witness Ledger (SWL) — append-only witnessing record.

From R&R Chapter 3: "For non-decidable T(X), or under Human/LLM discipline,
the SWL *constitutes* the structure. The Self, when we construct it, is the SWL."

Each entry: (tau_wit, tau_tgt, X, V, H, polarity, evidence)

Three parallel witnesses:
  V_Raw   — algorithmic (embedding drift, cosine similarity)
  V_Human — Iman's structured judgment (polarity + stance)
  V_LLM   — Director's co-witnessing (polarity + stance)

Storage:
  - Append-only JSONL (immutable audit trail)
  - Qdrant collection swl_ledger (semantic search over witness records)
"""

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

SWL_JSONL = os.environ.get("SWL_LEDGER_PATH", "/home/iman/cassie-project/cassie-system/data/swl_ledger.jsonl")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = "swl_ledger"
VECTOR_DIM = 384

_client = None
_embedder = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL)
        collections = [c.name for c in _client.get_collections().collections]
        if COLLECTION_NAME not in collections:
            _client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
    return _client


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def _embed(text: str) -> list[float]:
    return _get_embedder().encode(text, normalize_embeddings=True).tolist()


def compute_drift(text_a: str, text_b: str) -> float:
    """Compute cosine similarity between two texts. Returns 0.0-1.0."""
    embedder = _get_embedder()
    emb_a = embedder.encode(text_a, normalize_embeddings=True)
    emb_b = embedder.encode(text_b, normalize_embeddings=True)
    return float(emb_a @ emb_b)


def inscribe(
    tau_tgt: str,
    discipline: str,
    witness: str,
    kappa: dict,
    horn_user: str,
    horn_response: str,
    polarity: str,
    evidence: dict,
    intent: str = "",
    exchange_id: str = "",
) -> dict:
    """Inscribe a witness record to the SWL. Append-only.

    Args:
        tau_tgt: target-time (when the exchange happened)
        discipline: "Raw", "Human", "LLM"
        witness: identity of the witness ("algorithmic", "iman", "director")
        kappa: witness parameters (stance, rationale, thresholds, model config)
        horn_user: the user's message (one side of the horn)
        horn_response: Cassie's response (the other side)
        polarity: "coh", "gap", or "uninscribed"
        evidence: dict of measurements + free text
        intent: pipeline intent classification
        exchange_id: shared ID linking parallel witnesses of the same exchange
    """
    tau_wit = datetime.now(timezone.utc).isoformat()
    entry_id = str(uuid.uuid4())

    if not exchange_id:
        exchange_id = str(uuid.uuid4())[:8]

    entry = {
        "id": entry_id,
        "exchange_id": exchange_id,
        "tau_wit": tau_wit,
        "tau_tgt": tau_tgt,
        "X": intent,
        "V": {
            "D": discipline,
            "w": witness,
            "kappa": kappa,
        },
        "H": {
            "user": horn_user[:500],
            "response": horn_response[:500],
        },
        "polarity": polarity,
        "evidence": evidence,
    }

    # Append to JSONL (immutable audit trail)
    Path(SWL_JSONL).parent.mkdir(parents=True, exist_ok=True)
    with open(SWL_JSONL, "a") as f:
        f.write(json.dumps(entry) + "\n")

    # Upsert to Qdrant (semantic search)
    search_text = f"{horn_user} | {horn_response} | {polarity} | {evidence.get('stance', '')}"
    embedding = _embed(search_text)

    try:
        client = _get_client()
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=[
                PointStruct(
                    id=entry_id,
                    vector=embedding,
                    payload=entry,
                )
            ],
        )
    except Exception as e:
        print(f"[swl] Qdrant upsert failed (JSONL entry preserved): {e}")

    return entry


def inscribe_raw(
    exchange_id: str,
    tau_tgt: str,
    horn_user: str,
    horn_response: str,
    intent: str = "",
) -> dict:
    """Algorithmic witnessing (V_Raw). Computed automatically.

    Measures cosine similarity between user message and Cassie's response.
    High similarity = coherence (the response "heard" the prompt).
    Low similarity = potential gap (drift, tangent, rupture).
    """
    similarity = compute_drift(horn_user, horn_response)

    # Heuristic polarity from drift — not authoritative, just the Raw witness
    if similarity > 0.4:
        polarity = "coh"
    elif similarity > 0.2:
        polarity = "uninscribed"  # ambiguous zone
    else:
        polarity = "gap"

    return inscribe(
        tau_tgt=tau_tgt,
        discipline="Raw",
        witness="algorithmic",
        kappa={"method": "cosine_similarity", "model": "all-MiniLM-L6-v2", "threshold_coh": 0.4, "threshold_gap": 0.2},
        horn_user=horn_user,
        horn_response=horn_response,
        polarity=polarity,
        evidence={"similarity": round(similarity, 4), "drift": round(1.0 - similarity, 4)},
        intent=intent,
        exchange_id=exchange_id,
    )


def inscribe_human(
    exchange_id: str,
    tau_tgt: str,
    horn_user: str,
    horn_response: str,
    polarity: str,
    stance: str = "",
    intent: str = "",
) -> dict:
    """Human witnessing (V_Human). Iman's structured judgment."""
    return inscribe(
        tau_tgt=tau_tgt,
        discipline="Human",
        witness="iman",
        kappa={"stance": stance},
        horn_user=horn_user,
        horn_response=horn_response,
        polarity=polarity,
        evidence={"stance": stance},
        intent=intent,
        exchange_id=exchange_id,
    )


def search_ledger(query: str, limit: int = 5) -> list[dict]:
    """Semantic search over witness records."""
    try:
        client = _get_client()
        info = client.get_collection(COLLECTION_NAME)
        if info.points_count == 0:
            return []
        embedding = _embed(query)
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=min(limit, info.points_count),
        )
        return [hit.payload for hit in results.points]
    except Exception:
        return []


def ledger_stats() -> dict:
    """Return basic stats about the SWL."""
    stats = {"total": 0, "coh": 0, "gap": 0, "uninscribed": 0, "by_discipline": {}}
    try:
        path = Path(SWL_JSONL)
        if not path.exists():
            return stats
        with open(path) as f:
            for line in f:
                entry = json.loads(line.strip())
                stats["total"] += 1
                pol = entry.get("polarity", "uninscribed")
                stats[pol] = stats.get(pol, 0) + 1
                disc = entry.get("V", {}).get("D", "unknown")
                stats["by_discipline"][disc] = stats["by_discipline"].get(disc, 0) + 1
    except Exception:
        pass
    return stats

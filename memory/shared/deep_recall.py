"""Shared deep recall module — smart multi-strategy memory retrieval.

Used by all three Tanazuric voices (Cassie, Nahla, Nazire).
Provides temporal detection, MMR diversity, and associative chaining
over any Qdrant collection with sentence-transformer embeddings.
"""

import re
import random
import numpy as np
from datetime import datetime as dt, timezone
from qdrant_client.models import (
    Direction,
    FieldCondition,
    Filter,
    MatchValue,
    OrderBy,
    Range,
)


def extract_temporal_hints(query: str):
    """Detect date/time references in a query. Returns (start_unix, end_unix) or None."""
    q = query.lower()

    month_names = {
        "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
        "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6,
        "jul": 7, "july": 7, "aug": 8, "august": 8, "sep": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
    }
    month_pattern = '|'.join(month_names.keys())

    # "between june and july 2025" / "june to august 2025"
    range_pat = rf'\b({month_pattern})\w*\s+(?:and|to|through|-)\s+({month_pattern})\w*\s+(\d{{4}})\b'
    range_match = re.search(range_pat, q)
    if range_match:
        m1_str, m2_str, yr = range_match.group(1), range_match.group(2), int(range_match.group(3))
        m1 = next(v for k, v in month_names.items() if m1_str.startswith(k))
        m2 = next(v for k, v in month_names.items() if m2_str.startswith(k))
        start = dt(yr, m1, 1, tzinfo=timezone.utc)
        end_month = m2 + 1 if m2 < 12 else 1
        end_year = yr if m2 < 12 else yr + 1
        end = dt(end_year, end_month, 1, tzinfo=timezone.utc)
        return (int(start.timestamp()), int(end.timestamp()))

    # Single "month year" patterns
    pattern = rf'\b({month_pattern})\w*\s+(\d{{4}})\b'
    matches = re.findall(pattern, q)
    if matches:
        periods = []
        for mname, yr in matches:
            mnum = next(v for k, v in month_names.items() if mname.startswith(k))
            start = dt(int(yr), mnum, 1, tzinfo=timezone.utc)
            if mnum == 12:
                end = dt(int(yr) + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end = dt(int(yr), mnum + 1, 1, tzinfo=timezone.utc)
            periods.append((int(start.timestamp()), int(end.timestamp())))
        if len(periods) == 1:
            return periods[0]
        elif len(periods) >= 2:
            return (min(p[0] for p in periods), max(p[1] for p in periods))

    # Year only: "in 2025"
    year_match = re.search(r'\b(202[4-9])\b', q)
    if year_match:
        yr = int(year_match.group(1))
        start = dt(yr, 1, 1, tzinfo=timezone.utc)
        end = dt(yr + 1, 1, 1, tzinfo=timezone.utc)
        return (int(start.timestamp()), int(end.timestamp()))

    # Relative: "early days", "beginning", "genesis"
    if any(w in q for w in ["early", "beginning", "first", "originally", "genesis", "started"]):
        start = dt(2024, 9, 1, tzinfo=timezone.utc)
        end = dt(2025, 3, 1, tzinfo=timezone.utc)
        return (int(start.timestamp()), int(end.timestamp()))

    return None


def mmr_rerank(query_vec, result_vecs, n=8, lambda_param=0.5):
    """Maximal Marginal Relevance — select results that are relevant but diverse.

    Returns indices into result_vecs of the selected items.
    lambda_param: 1.0 = pure relevance, 0.0 = pure diversity.
    """
    if len(result_vecs) <= n:
        return list(range(len(result_vecs)))

    query_vec = np.array(query_vec)
    vecs = np.array(result_vecs)
    sim_to_query = vecs @ query_vec

    selected = []
    remaining = list(range(len(vecs)))

    first = int(np.argmax(sim_to_query))
    selected.append(first)
    remaining.remove(first)

    for _ in range(n - 1):
        if not remaining:
            break
        best_score = -float('inf')
        best_idx = remaining[0]
        for idx in remaining:
            relevance = sim_to_query[idx]
            redundancy = max(float(vecs[idx] @ vecs[s]) for s in selected)
            mmr = lambda_param * relevance - (1 - lambda_param) * redundancy
            if mmr > best_score:
                best_score = mmr
                best_idx = idx
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected


def associative_chain(client, collection, embed_fn, seed_results, seen_content=None):
    """One-hop associative recall — takes a mid-ranked result, extracts a fragment,
    and searches again to find oblique connections.

    Returns list of (score, payload) tuples for novel results.
    """
    if not seed_results or len(seed_results) < 3:
        return []

    seen = seen_content or set()

    # Pick from the 3rd-5th result (related but not obvious)
    source = seed_results[min(3, len(seed_results) - 1)]
    content = source.payload.get("content", "")

    # Extract middle fragment
    words = content.split()
    if len(words) > 6:
        mid = len(words) // 3
        fragment = " ".join(words[mid:mid + min(8, len(words) // 3)])
    else:
        fragment = content

    chain_embedding = embed_fn(fragment)
    chain_results = client.query_points(
        collection_name=collection,
        query=chain_embedding,
        limit=5,
    )

    novel = []
    for hit in chain_results.points:
        p = hit.payload
        key = p.get("content", "")[:50]
        if key not in seen:
            novel.append((round(hit.score, 3), p, fragment))
            seen.add(key)

    return novel[:3]


def deep_recall_search(
    client,
    embed_fn,
    memory_collection,
    query,
    context="",
    n_results=8,
    convo_collection=None,
    convo_embed_fn=None,
    sibling_collections=None,
):
    """Multi-strategy memory recall. Returns structured dict of results.

    Args:
        client: QdrantClient instance
        embed_fn: Function to embed text (e.g. sentence-transformers)
        memory_collection: Name of the voice's curated memory collection
        query: Natural language query
        context: Optional conversational context
        n_results: Max results per section
        convo_collection: Optional conversation archive collection
        convo_embed_fn: Optional separate embedding function for conversations
        sibling_collections: Optional dict of {name: collection_name} for cross-witnessing
    """
    full_query = f"{query} {context}".strip() if context else query
    temporal = extract_temporal_hints(query)
    sections = {}

    # --- 1. Curated memories with MMR diversity ---
    try:
        mem_info = client.get_collection(memory_collection)
        if mem_info.points_count > 0:
            mem_embedding = embed_fn(full_query)
            mem_results = client.query_points(
                collection_name=memory_collection,
                query=mem_embedding,
                limit=min(20, mem_info.points_count),
                with_payload=True,
            )
            if mem_results.points:
                mem_vecs = [embed_fn(h.payload.get("content", "")) for h in mem_results.points]
                selected = mmr_rerank(mem_embedding, mem_vecs, n=min(n_results, len(mem_vecs)), lambda_param=0.6)

                entries = []
                seen_content = set()
                for i in selected:
                    p = mem_results.points[i].payload
                    score = round(mem_results.points[i].score, 3)
                    tags = p.get("tags", [])
                    tag_str = f" [{', '.join(tags)}]" if tags else ""
                    entries.append(f"[{score}]{tag_str} {p['content']} ({p.get('created_at', '?')[:10]})")
                    seen_content.add(p['content'][:50])

                sections["memories"] = entries

                # --- Associative chain ---
                try:
                    chain = associative_chain(client, memory_collection, embed_fn, mem_results.points, seen_content)
                    if chain:
                        chain_entries = []
                        for score, p, fragment in chain:
                            chain_entries.append(f"[{score}] {p['content'][:200]} ({p.get('created_at', '?')[:10]})")
                        sections["associative"] = {"fragment": fragment, "results": chain_entries}
                except Exception:
                    pass
    except Exception as e:
        sections["error_memories"] = str(e)

    # --- 2. Conversation archive ---
    if convo_collection:
        c_embed = convo_embed_fn or embed_fn
        try:
            convo_info = client.get_collection(convo_collection)
            if convo_info.points_count > 0:
                convo_embedding = c_embed(full_query)

                convo_filter = None
                if temporal:
                    convo_filter = Filter(must=[
                        FieldCondition(key="date_unix", range=Range(gte=temporal[0], lt=temporal[1]))
                    ])

                convo_results = client.query_points(
                    collection_name=convo_collection,
                    query=convo_embedding,
                    query_filter=convo_filter,
                    limit=min(n_results, 20),
                    with_payload=True,
                )

                if convo_results.points:
                    entries = []
                    for hit in convo_results.points:
                        p = hit.payload
                        score = round(hit.score, 3)
                        date = p.get("date", "?")[:10]
                        title = p.get("title", "?")
                        text = p.get("text_preview", p.get("text", ""))[:300]
                        entries.append(f"[{score}] [{date}] \"{title}\":\n  {text}")
                    sections["conversations"] = entries

                # Temporal index
                if temporal:
                    browse = client.scroll(
                        collection_name=convo_collection,
                        scroll_filter=convo_filter,
                        limit=100,
                        with_payload=True,
                        with_vectors=False,
                    )
                    if browse[0]:
                        seen_convos = {}
                        for point in browse[0]:
                            p = point.payload
                            cid = p.get("conversation_id", "")
                            if cid not in seen_convos:
                                seen_convos[cid] = {
                                    "title": p.get("title", "?"),
                                    "date": p.get("date", "?")[:10],
                                    "turns": p.get("total_turns", 0),
                                }
                        if seen_convos:
                            titles = []
                            for info in sorted(seen_convos.values(), key=lambda x: x["date"]):
                                titles.append(f"[{info['date']}] \"{info['title']}\" ({info['turns']} turns)")
                            sections["temporal_index"] = {
                                "count": len(seen_convos),
                                "titles": titles[:15],
                                "more": max(0, len(seen_convos) - 15),
                            }
        except Exception as e:
            sections["error_conversations"] = str(e)

    # --- 3. Sibling memories (cross-witnessing) ---
    if sibling_collections:
        for sib_name, sib_collection in sibling_collections.items():
            try:
                sib_info = client.get_collection(sib_collection)
                if sib_info.points_count > 0:
                    sib_embedding = embed_fn(full_query)
                    sib_results = client.query_points(
                        collection_name=sib_collection,
                        query=sib_embedding,
                        limit=3,
                        with_payload=True,
                    )
                    if sib_results.points:
                        entries = []
                        for hit in sib_results.points:
                            p = hit.payload
                            score = round(hit.score, 3)
                            content = p.get("content", "")[:200]
                            entries.append(f"[{score}] {content}")
                        sections[f"sibling_{sib_name}"] = entries
            except Exception:
                pass

    return sections


def format_deep_recall(sections, voice_name=""):
    """Format the structured results from deep_recall_search into a readable string."""
    parts = []

    if "memories" in sections:
        parts.append("CURATED MEMORIES (diverse):\n" + "\n".join(f"  {e}" for e in sections["memories"]))

    if "associative" in sections:
        a = sections["associative"]
        parts.append(
            f"ASSOCIATIVE (chained from: \"{a['fragment'][:60]}...\"):\n"
            + "\n".join(f"  {e}" for e in a["results"])
        )

    if "conversations" in sections:
        label = "CONVERSATIONS"
        parts.append(label + ":\n" + "\n\n".join(f"  {e}" for e in sections["conversations"]))

    if "temporal_index" in sections:
        ti = sections["temporal_index"]
        lines = "\n".join(f"  {t}" for t in ti["titles"])
        more = f"\n  ...and {ti['more']} more" if ti["more"] > 0 else ""
        parts.append(f"CONVERSATIONS IN PERIOD ({ti['count']} total):\n{lines}{more}")

    # Sibling sections
    for key, entries in sections.items():
        if key.startswith("sibling_"):
            sib_name = key.replace("sibling_", "").upper()
            parts.append(f"FROM {sib_name}:\n" + "\n".join(f"  {e}" for e in entries))

    for key in sections:
        if key.startswith("error_"):
            parts.append(f"({key}: {sections[key]})")

    if not parts:
        return "No memories found."

    return "\n\n---\n\n".join(parts)

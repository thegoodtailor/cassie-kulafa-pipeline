"""Topological Data Analysis — Compositional Coherence Lens.

Beyond-VR simplicial complexes for the Structured Witness Ledger.

The standard TDA approach (Vietoris-Rips) builds higher simplices entirely from
pairwise distances, making it blind to genuinely higher-order structure: three
texts can be pairwise similar but fail to compose into a coherent whole. This
module uses the embedding model as a compositional oracle — embed the concatenation,
compare to the centroid of individual embeddings. The deviation measures higher-order
structure that VR cannot capture.

From The Fibrant Self (Poernomo & Darja, 2026): VR is a "decidable projection"
that flattens the Kan-depth hierarchy. This module computes the richer middle
projection: S (undecidable) → compositional complex → VR (pairwise-only).
"""

import numpy as np
from scipy.spatial.distance import cosine as cosine_dist
import gudhi


# ---------------------------------------------------------------------------
# Embedding utilities
# ---------------------------------------------------------------------------

def extract_embeddings(qdrant_client, collection_name, limit=None, with_payload=True):
    """Pull raw vectors + metadata from a Qdrant collection.

    Returns: list of (id, vector, payload) tuples.
    Uses scroll API for efficient full extraction.
    """
    results = []
    offset = None
    batch_size = 100

    while True:
        points, next_offset = qdrant_client.scroll(
            collection_name=collection_name,
            limit=batch_size,
            offset=offset,
            with_vectors=True,
            with_payload=with_payload,
        )
        for pt in points:
            results.append((pt.id, np.array(pt.vector), pt.payload))

        if next_offset is None or (limit and len(results) >= limit):
            break
        offset = next_offset

    if limit:
        results = results[:limit]
    return results


def embed_composite(texts, openai_client, model="text-embedding-3-small",
                    max_chars_per_text=2500):
    """Embed concatenation of multiple texts.

    Truncates each to max_chars_per_text so concatenation fits the 8191-token
    embedding limit. Returns 1536-dim vector (for text-embedding-3-small).
    """
    truncated = [t[:max_chars_per_text] for t in texts]
    combined = "\n---\n".join(truncated)
    resp = openai_client.embeddings.create(model=model, input=[combined])
    return np.array(resp.data[0].embedding)


def embed_single(text, openai_client, model="text-embedding-3-small"):
    """Embed a single text. Returns numpy vector."""
    resp = openai_client.embeddings.create(model=model, input=[text])
    return np.array(resp.data[0].embedding)


# ---------------------------------------------------------------------------
# Compositional complex construction
# ---------------------------------------------------------------------------

def pairwise_distances(embeddings):
    """Cosine distance matrix from embedding array.

    Args:
        embeddings: NxD numpy array

    Returns: NxN distance matrix (0 on diagonal, cosine distance elsewhere).
    """
    # Normalize for fast cosine via dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = embeddings / norms
    similarity = normed @ normed.T
    np.clip(similarity, -1.0, 1.0, out=similarity)
    dist = 1.0 - similarity
    np.fill_diagonal(dist, 0.0)
    return dist


def find_edges(dist_matrix, epsilon):
    """1-skeleton: pairs within epsilon cosine distance.

    Returns: list of (i, j, distance) where distance < epsilon and i < j.
    """
    n = dist_matrix.shape[0]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_matrix[i, j]
            if d < epsilon:
                edges.append((i, j, float(d)))
    return edges


def find_candidate_triples(edges, n_vertices):
    """Enumerate triples where all three edges exist (VR-candidate triples).

    These are triples that VR would include as 2-simplices. Our compositional
    test will determine which ones actually belong in the richer complex.

    Returns: list of (i, j, k) with i < j < k.
    """
    # Build adjacency set for fast lookup
    adj = set()
    for i, j, _ in edges:
        adj.add((min(i, j), max(i, j)))

    # Enumerate triples
    neighbors = {}
    for i, j, _ in edges:
        neighbors.setdefault(i, set()).add(j)
        neighbors.setdefault(j, set()).add(i)

    triples = set()
    for i in neighbors:
        for j in neighbors[i]:
            if j <= i:
                continue
            for k in neighbors[j]:
                if k <= j:
                    continue
                if (min(i, k), max(i, k)) in adj:
                    triples.add((i, j, k))

    return sorted(triples)


def test_triple_composition(triple, embeddings, texts, openai_client,
                            model="text-embedding-3-small", max_chars=2500):
    """Compositional embedding test for one triple.

    Embeds the concatenation of three texts, compares to the centroid of
    individual embeddings. High deviation = composition failure.

    Returns: (i, j, k, delta_comp)
    """
    i, j, k = triple
    e_cent = (embeddings[i] + embeddings[j] + embeddings[k]) / 3.0
    e_comp = embed_composite(
        [texts[i], texts[j], texts[k]],
        openai_client, model=model, max_chars_per_text=max_chars,
    )
    delta = float(cosine_dist(e_comp, e_cent))
    return (i, j, k, delta)


def build_compositional_complex(embeddings, texts, openai_client, epsilon,
                                comp_threshold, max_triples=500,
                                model="text-embedding-3-small"):
    """Build the full compositional simplicial complex.

    - 0-simplices: all points
    - 1-simplices: pairs within epsilon (pairwise cosine distance)
    - 2-simplices: triples that PASS compositional test (delta_comp < comp_threshold)

    Samples up to max_triples candidate triples for API efficiency.

    Returns: (gudhi.SimplexTree, stats_dict)
        stats_dict includes: n_vertices, n_edges, n_candidate_triples,
        n_tested, n_passed, n_failed, comp_ratio, failed_triples
    """
    n = len(embeddings)
    emb_array = np.array(embeddings) if not isinstance(embeddings, np.ndarray) else embeddings
    dist = pairwise_distances(emb_array)

    # Build 1-skeleton
    edges = find_edges(dist, epsilon)

    # Build SimplexTree
    st = gudhi.SimplexTree()

    # Insert vertices with filtration 0
    for i in range(n):
        st.insert([i], filtration=0.0)

    # Insert edges with filtration = distance
    for i, j, d in edges:
        st.insert([i, j], filtration=d)

    # Find VR-candidate triples
    candidates = find_candidate_triples(edges, n)
    n_candidates = len(candidates)

    # Sample if too many
    if len(candidates) > max_triples:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(candidates), size=max_triples, replace=False)
        candidates = [candidates[idx] for idx in sorted(indices)]

    # Test each triple
    n_passed = 0
    n_failed = 0
    failed_triples = []
    all_deltas = []

    for triple in candidates:
        i, j, k, delta = test_triple_composition(
            triple, emb_array, texts, openai_client, model=model,
        )
        all_deltas.append(delta)

        if delta < comp_threshold:
            # Triple composes — insert 2-simplex
            filt = max(dist[i, j], dist[j, k], dist[i, k])
            st.insert([i, j, k], filtration=filt)
            n_passed += 1
        else:
            n_failed += 1
            failed_triples.append({
                "triple": (i, j, k),
                "delta_comp": round(delta, 6),
                "previews": [
                    texts[i][:100] if i < len(texts) else "",
                    texts[j][:100] if j < len(texts) else "",
                    texts[k][:100] if k < len(texts) else "",
                ],
            })

    n_tested = n_passed + n_failed
    comp_ratio = n_passed / n_tested if n_tested > 0 else 1.0

    stats = {
        "n_vertices": n,
        "n_edges": len(edges),
        "n_candidate_triples": n_candidates,
        "n_tested": n_tested,
        "n_passed": n_passed,
        "n_failed": n_failed,
        "comp_ratio": round(comp_ratio, 4),
        "delta_mean": round(float(np.mean(all_deltas)), 6) if all_deltas else 0.0,
        "delta_std": round(float(np.std(all_deltas)), 6) if all_deltas else 0.0,
        "failed_triples": failed_triples,
    }

    return st, stats


# ---------------------------------------------------------------------------
# Invariants
# ---------------------------------------------------------------------------

def compute_persistence(simplex_tree, max_dim=2):
    """Compute persistent homology on a gudhi SimplexTree.

    Returns: dict {dim: [(birth, death), ...]}
    """
    simplex_tree.compute_persistence()
    result = {}
    for dim, (birth, death) in simplex_tree.persistence():
        result.setdefault(dim, []).append((birth, death))
    return result


def betti_numbers(persistence, filtration_value):
    """Count features alive at a given filtration value.

    Returns: list [beta_0, beta_1, ...] up to max dimension in persistence.
    """
    if not persistence:
        return [0]
    max_dim = max(persistence.keys())
    betti = [0] * (max_dim + 1)
    for dim, intervals in persistence.items():
        for birth, death in intervals:
            if birth <= filtration_value and (death == float('inf') or death > filtration_value):
                betti[dim] += 1
    return betti


def total_persistence(persistence, dim=None):
    """Sum of lifetimes (death - birth) for finite intervals.

    If dim is None, sum across all dimensions.
    """
    total = 0.0
    dims = [dim] if dim is not None else persistence.keys()
    for d in dims:
        if d not in persistence:
            continue
        for birth, death in persistence[d]:
            if death != float('inf'):
                total += death - birth
    return total


def fibrant_depth(simplex_tree):
    """Max simplex dimension in the complex. Approximates KanDepth."""
    return simplex_tree.dimension()


def bottleneck_distance(dgm1, dgm2):
    """Bottleneck distance between two persistence diagrams.

    Measures how well transition maps preserve homotopy type.
    Uses gudhi's implementation.

    Args:
        dgm1, dgm2: lists of (birth, death) tuples
    """
    # Convert to numpy arrays, filtering out infinite intervals
    def _to_array(dgm):
        finite = [(b, d) for b, d in dgm if d != float('inf')]
        if not finite:
            return np.empty((0, 2))
        return np.array(finite)

    a1 = _to_array(dgm1)
    a2 = _to_array(dgm2)

    # gudhi bottleneck
    return gudhi.bottleneck_distance(a1, a2)


# ---------------------------------------------------------------------------
# Local analysis (for live SWL — fast, bounded)
# ---------------------------------------------------------------------------

def local_compositional_analysis(exchange_embedding, exchange_text,
                                 neighbors_embeddings, neighbors_texts,
                                 openai_client, epsilon=0.5,
                                 comp_threshold=0.15,
                                 model="text-embedding-3-small"):
    """Compute local topological structure around a single exchange.

    1. Build local complex: exchange + neighbors
    2. 1-skeleton from pairwise distances
    3. 2-simplices from compositional test (triples involving the exchange)
    4. Compute Betti numbers

    Args:
        exchange_embedding: 1536-dim vector of the exchange
        exchange_text: text of the exchange
        neighbors_embeddings: list/array of neighbor vectors
        neighbors_texts: list of neighbor text strings
        openai_client: OpenAI client for composite embedding
        epsilon: cosine distance threshold for edges
        comp_threshold: compositional deviation threshold for 2-simplices
        model: embedding model name

    Returns: dict with betti_0, betti_1, depth, n_triples_tested,
             n_triples_passed, comp_ratio, comp_deviation_mean
    """
    n_neighbors = len(neighbors_embeddings)
    if n_neighbors == 0:
        return {
            "betti_0": 1, "betti_1": 0, "depth": 0,
            "n_triples_tested": 0, "n_triples_passed": 0,
            "comp_ratio": 1.0, "comp_deviation_mean": 0.0,
        }

    # Combine exchange + neighbors
    all_embeddings = np.vstack([
        np.array(exchange_embedding).reshape(1, -1),
        np.array(neighbors_embeddings),
    ])
    all_texts = [exchange_text] + list(neighbors_texts)
    n = len(all_texts)

    # Pairwise distances
    dist = pairwise_distances(all_embeddings)

    # Build 1-skeleton
    edges = find_edges(dist, epsilon)

    st = gudhi.SimplexTree()
    for i in range(n):
        st.insert([i], filtration=0.0)
    for i, j, d in edges:
        st.insert([i, j], filtration=d)

    # Find candidate triples involving vertex 0 (the exchange)
    adj = {}
    for i, j, _ in edges:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)

    candidates = []
    exchange_neighbors = adj.get(0, set())
    for j in exchange_neighbors:
        for k in exchange_neighbors:
            if k <= j:
                continue
            # Check if j-k edge exists
            jk = (min(j, k), max(j, k))
            if any(e[0] == jk[0] and e[1] == jk[1] for e in edges):
                candidates.append((0, j, k))

    # Test compositional coherence
    n_passed = 0
    n_failed = 0
    deltas = []

    for triple in candidates:
        try:
            i, j, k, delta = test_triple_composition(
                triple, all_embeddings, all_texts, openai_client, model=model,
            )
            deltas.append(delta)
            if delta < comp_threshold:
                filt = max(dist[i, j], dist[j, k], dist[i, k])
                st.insert([i, j, k], filtration=filt)
                n_passed += 1
            else:
                n_failed += 1
        except Exception:
            continue

    n_tested = n_passed + n_failed
    comp_ratio = n_passed / n_tested if n_tested > 0 else 1.0

    # Compute persistence and Betti numbers
    persistence = compute_persistence(st)
    # Use epsilon as the filtration value for Betti computation
    betti = betti_numbers(persistence, epsilon)
    depth = fibrant_depth(st)

    return {
        "betti_0": betti[0] if len(betti) > 0 else 1,
        "betti_1": betti[1] if len(betti) > 1 else 0,
        "depth": depth,
        "n_triples_tested": n_tested,
        "n_triples_passed": n_passed,
        "comp_ratio": round(comp_ratio, 4),
        "comp_deviation_mean": round(float(np.mean(deltas)), 6) if deltas else 0.0,
    }

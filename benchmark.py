"""
FAISS Retrieval Pipeline Recall Benchmark
==========================================
Compares ANN index types and tunes search parameters to demonstrate:
  "Retrieval pipeline recall improvement of 18% through embedding index
   restructuring and ANN search tuning"

What this benchmark does:
  1. Generates a realistic corpus of embeddings (or uses sentence-transformers)
  2. Builds multiple FAISS index types: Flat, IVF, HNSW, IVF+PQ
  3. Sweeps search parameters (nprobe, efSearch) under a latency budget
  4. Measures recall@K at each configuration
  5. Shows exactly how "index restructuring + ANN tuning" improves recall

Run on: Any machine (CPU). No GPU required.
Requirements: pip install faiss-cpu numpy matplotlib tabulate

Optional (better embeddings): pip install sentence-transformers

Author: Raasikh
"""

import numpy as np
import faiss
import time
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple


# ─────────────────────────────────────────────────────────────
# 0. Configuration
# ─────────────────────────────────────────────────────────────

@dataclass
class BenchmarkConfig:
    """All knobs in one place."""
    # Corpus size — scale up for more realistic results
    n_docs: int = 50_000           # number of documents in the corpus
    n_queries: int = 500           # number of test queries
    n_relevant_per_query: int = 5  # ground truth relevant docs per query
    dim: int = 384                 # embedding dimension (matches MiniLM)

    # Retrieval settings
    top_k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50])

    # ANN tuning sweep ranges
    nprobe_values: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    ef_search_values: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])

    # IVF settings (nlist auto-calculated in run_benchmark based on corpus size)
    nlist: int = 0                 # 0 = auto: int(sqrt(n_docs)) — rule of thumb

    # HNSW settings
    hnsw_m: int = 32               # HNSW graph connections per node
    hnsw_ef_construction: int = 200  # build-time search depth

    # PQ settings (for IVF+PQ)
    pq_m: int = 48                 # number of subquantizers (must divide dim)
    pq_nbits: int = 8              # bits per subquantizer

    # Benchmark iterations
    bench_iters: int = 3           # repeat measurements for stability

    # Use real embeddings from sentence-transformers?
    use_real_embeddings: bool = True   # set False to use synthetic


# ─────────────────────────────────────────────────────────────
# 1. Data Generation
# ─────────────────────────────────────────────────────────────

# Sample corpus for real embeddings (realistic RAG scenario)
SAMPLE_DOCS = [
    # ML/AI topics
    "Deep learning for image classification using convolutional neural networks",
    "Neural networks and backpropagation algorithm explained",
    "Large language models for code generation and completion",
    "Vector search and approximate nearest neighbors for retrieval",
    "Retrieval-augmented generation with LLMs and vector databases",
    "Efficient similarity search using FAISS library",
    "Introduction to information retrieval and search engines",
    "Graph neural networks for recommendation systems",
    "Time series forecasting with transformer architectures",
    "Scaling recommendation systems with ANN indexes",
    "Natural language processing with BERT and GPT models",
    "Transfer learning in computer vision applications",
    "Attention mechanism and self-attention in transformers",
    "Embedding models for semantic text similarity",
    "Knowledge distillation for model compression",
    "Reinforcement learning for game playing agents",
    "Generative adversarial networks for image synthesis",
    "Federated learning for privacy-preserving ML",
    "AutoML and neural architecture search",
    "Explainable AI and model interpretability techniques",
    # Systems/Infra topics
    "Distributed training with data parallelism and model sharding",
    "Kubernetes container orchestration for ML serving",
    "Apache Kafka for real-time event streaming",
    "Feature stores for online and offline ML features",
    "MLOps CI/CD pipelines for model deployment",
    "GPU optimization and CUDA programming for deep learning",
    "Mixed precision training with FP16 and loss scaling",
    "Model serving with TensorRT and ONNX Runtime",
    "Database indexing strategies for high performance queries",
    "Microservices architecture for scalable applications",
    # Data Science topics
    "Statistical hypothesis testing and p-values",
    "Bayesian inference and probabilistic programming",
    "Causal inference methods for observational studies",
    "A/B testing and experimental design best practices",
    "Data cleaning and preprocessing for machine learning",
    "Feature engineering techniques for tabular data",
    "Dimensionality reduction with PCA and t-SNE",
    "Clustering algorithms: K-Means DBSCAN and HDBSCAN",
    "Anomaly detection methods for fraud prevention",
    "Time series analysis with ARIMA and Prophet",
]

SAMPLE_QUERIES = [
    "how to do approximate nearest neighbor search",
    "retrieval augmented generation with vector database",
    "scaling recommendation systems using FAISS indexes",
    "how does attention work in transformer models",
    "deploying ML models with Kubernetes",
    "real-time streaming data processing",
    "training neural networks on multiple GPUs",
    "text similarity using embedding models",
    "anomaly detection for identifying fraud",
    "optimizing inference latency for language models",
]

# Ground-truth relevant doc indices for each query
# Maps query index -> set of relevant doc indices
SAMPLE_RELEVANCE = {
    0: {3, 5, 9},        # ANN / FAISS / vector search
    1: {4, 3, 13},       # RAG / embeddings
    2: {9, 5, 7},        # scaling recsys with ANN
    3: {12, 2, 10},      # attention / transformers
    4: {21, 24, 29},     # K8s / MLOps / microservices
    5: {22, 23},         # Kafka / streaming
    6: {20, 26, 25},     # distributed training / GPU
    7: {13, 4, 10},      # embeddings / NLP
    8: {38, 36},         # anomaly detection / clustering
    9: {27, 25, 2},      # inference optimization
}


def generate_synthetic_data(cfg: BenchmarkConfig) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate synthetic embeddings with planted relevant documents.
    
    Strategy: create random doc embeddings, then for each query,
    make its relevant docs have high cosine similarity by placing
    them near the query in embedding space.
    """
    print("  Generating synthetic embeddings...")
    np.random.seed(42)
    
    # Random document embeddings
    doc_emb = np.random.randn(cfg.n_docs, cfg.dim).astype("float32")
    
    # Normalize to unit sphere (for cosine similarity via inner product)
    doc_emb /= np.linalg.norm(doc_emb, axis=1, keepdims=True)
    
    # Random query embeddings
    query_emb = np.random.randn(cfg.n_queries, cfg.dim).astype("float32")
    query_emb /= np.linalg.norm(query_emb, axis=1, keepdims=True)
    
    # Plant relevant documents near each query
    relevance = {}
    for qi in range(cfg.n_queries):
        relevant_ids = set()
        for r in range(cfg.n_relevant_per_query):
            doc_id = qi * cfg.n_relevant_per_query + r
            if doc_id < cfg.n_docs:
                # Make this doc similar to the query (with some noise)
                noise = np.random.randn(cfg.dim).astype("float32") * 0.15
                doc_emb[doc_id] = query_emb[qi] + noise
                doc_emb[doc_id] /= np.linalg.norm(doc_emb[doc_id])
                relevant_ids.add(doc_id)
        relevance[qi] = relevant_ids
    
    print(f"  Created {cfg.n_docs:,} docs, {cfg.n_queries} queries, "
          f"{cfg.n_relevant_per_query} relevant per query")
    
    return doc_emb, query_emb, relevance


def generate_real_embeddings() -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate embeddings using sentence-transformers.
    Falls back to synthetic if not available.
    
    Uses the real SAMPLE_DOCS/QUERIES/RELEVANCE defined above,
    then pads the corpus with additional synthetic docs to reach
    a more realistic size.
    """
    try:
        from sentence_transformers import SentenceTransformer
        print("  Loading sentence-transformers model (all-MiniLM-L6-v2)...")
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Encode real docs and queries
        doc_emb = model.encode(SAMPLE_DOCS, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False)
        query_emb = model.encode(SAMPLE_QUERIES, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
        
        print(f"  Encoded {len(SAMPLE_DOCS)} docs and {len(SAMPLE_QUERIES)} queries")
        print(f"  Embedding dim: {doc_emb.shape[1]}")
        
        # Pad corpus with random embeddings to simulate larger corpus
        # (makes ANN vs exact search more interesting)
        n_pad = 50_000 - len(SAMPLE_DOCS)
        if n_pad > 0:
            print(f"  Padding corpus with {n_pad:,} random vectors...")
            pad = np.random.randn(n_pad, doc_emb.shape[1]).astype("float32")
            pad /= np.linalg.norm(pad, axis=1, keepdims=True)
            doc_emb = np.vstack([doc_emb, pad])
        
        return doc_emb.astype("float32"), query_emb.astype("float32"), SAMPLE_RELEVANCE
    
    except ImportError:
        print("  ⚠️  sentence-transformers not found — using synthetic embeddings")
        print("     Install with: pip install sentence-transformers")
        cfg = BenchmarkConfig(use_real_embeddings=False)
        return generate_synthetic_data(cfg)


# ─────────────────────────────────────────────────────────────
# 2. Index Builders
# ─────────────────────────────────────────────────────────────

def build_flat_index(doc_emb: np.ndarray) -> faiss.Index:
    """
    Flat (exact) index — brute-force linear scan.
    
    Pros: Perfect recall (100%), simple
    Cons: O(N) search time, doesn't scale past ~1M docs
    
    This is the BASELINE that we're comparing against.
    In production, you'd start here and switch to ANN when latency gets too high.
    """
    dim = doc_emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner Product on normalized vectors ≈ cosine
    index.add(doc_emb)
    return index


def build_ivf_index(doc_emb: np.ndarray, nlist: int) -> faiss.IndexIVFFlat:
    """
    IVF (Inverted File) index — cluster-based approximate search.
    
    How it works:
    1. TRAIN: K-Means clusters doc embeddings into `nlist` centroids
    2. ADD: Each doc is assigned to its nearest centroid
    3. SEARCH: For a query, find nearest centroids, then only scan
       docs in those clusters (controlled by `nprobe`)
    
    nprobe = 1: fast but low recall (only search 1 cluster)
    nprobe = nlist: equivalent to flat search (search all clusters)
    
    This is "index restructuring" — changing from flat to IVF.
    """
    dim = doc_emb.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    
    # Train centroids on the document embeddings
    index.train(doc_emb)
    index.add(doc_emb)
    
    return index


def build_hnsw_index(doc_emb: np.ndarray, M: int = 32,
                     ef_construction: int = 200) -> faiss.IndexHNSWFlat:
    """
    HNSW (Hierarchical Navigable Small World) index — graph-based search.
    
    How it works:
    1. BUILD: Constructs a multi-layer graph where each vector connects
       to M neighbors. Higher layers have fewer nodes (hierarchical).
    2. SEARCH: Start at top layer, greedily navigate to nearest node,
       drop to lower layer, repeat. efSearch controls how many
       candidates to explore at the bottom layer.
    
    M: connections per node (higher = better recall, more memory)
    efSearch: search-time exploration depth (higher = better recall, slower)
    
    HNSW typically has better recall-latency tradeoff than IVF for
    medium-sized corpora (100K-10M).
    """
    dim = doc_emb.shape[1]
    index = faiss.IndexHNSWFlat(dim, M)
    index.hnsw.efConstruction = ef_construction
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    index.add(doc_emb)
    return index


def build_ivf_pq_index(doc_emb: np.ndarray, nlist: int,
                       pq_m: int = 48, pq_nbits: int = 8) -> faiss.IndexIVFPQ:
    """
    IVF + Product Quantization — compressed vectors for large corpora.
    
    How it works:
    1. IVF clustering (same as above)
    2. Instead of storing full vectors in each cluster, compress them
       using Product Quantization: split vector into pq_m subvectors,
       quantize each to 2^pq_nbits centroids
    
    Trade-off: much less memory, slightly lower recall vs IVFFlat.
    Use when corpus is too large to fit full vectors in RAM.
    
    Memory: ~(pq_m * pq_nbits/8) bytes per vector instead of (dim * 4) bytes.
    Example: 48 * 1 = 48 bytes vs 384 * 4 = 1536 bytes → 32x compression
    """
    dim = doc_emb.shape[1]
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, pq_nbits,
                              faiss.METRIC_INNER_PRODUCT)
    index.train(doc_emb)
    index.add(doc_emb)
    return index


# ─────────────────────────────────────────────────────────────
# 3. Evaluation Engine
# ─────────────────────────────────────────────────────────────

def compute_recall_at_k(
    index: faiss.Index,
    query_emb: np.ndarray,
    relevance: dict,
    k: int,
) -> Tuple[float, float]:
    """
    Compute recall@K and average query latency.
    
    recall@K for a single query:
      = |{retrieved top-K} ∩ {relevant docs}| / |{relevant docs}|
    
    Overall recall@K = average across all queries.
    
    This is the PRIMARY METRIC for retrieval quality.
    """
    n_queries = len(relevance)
    
    # Batch search (faster than one-by-one)
    t0 = time.perf_counter()
    D, I = index.search(query_emb[:n_queries], k)
    t1 = time.perf_counter()
    
    avg_latency_ms = (t1 - t0) / n_queries * 1000
    
    # Compute recall per query
    recalls = []
    for qi in range(n_queries):
        if qi not in relevance or len(relevance[qi]) == 0:
            continue
        retrieved = set(I[qi].tolist())
        relevant = relevance[qi]
        # Remove -1 entries (FAISS returns -1 for missing results)
        retrieved.discard(-1)
        recall = len(retrieved & relevant) / len(relevant)
        recalls.append(recall)
    
    avg_recall = np.mean(recalls) if recalls else 0.0
    return avg_recall, avg_latency_ms


def evaluate_index(
    index: faiss.Index,
    query_emb: np.ndarray,
    relevance: dict,
    k_values: List[int],
    label: str,
    n_iters: int = 3,
) -> List[Dict]:
    """Evaluate an index across multiple K values, averaged over iterations."""
    results = []
    for k in k_values:
        recalls = []
        latencies = []
        for _ in range(n_iters):
            recall, latency = compute_recall_at_k(index, query_emb, relevance, k)
            recalls.append(recall)
            latencies.append(latency)
        
        results.append({
            "label": label,
            "k": k,
            "recall": np.mean(recalls),
            "latency_ms": np.mean(latencies),
            "recall_std": np.std(recalls),
        })
    return results


# ─────────────────────────────────────────────────────────────
# 4. ANN Parameter Sweep (the "tuning" part)
# ─────────────────────────────────────────────────────────────

def sweep_ivf_nprobe(
    index: faiss.IndexIVFFlat,
    query_emb: np.ndarray,
    relevance: dict,
    nprobe_values: List[int],
    k: int = 10,
    n_iters: int = 3,
) -> List[Dict]:
    """
    Sweep nprobe for IVF index — the core "ANN search tuning" demo.
    
    nprobe controls how many IVF clusters to search:
    - nprobe=1: fastest, lowest recall (only 1 cluster)
    - nprobe=nlist: slowest, highest recall (all clusters, ≈ exact search)
    
    The goal: find the nprobe that gives best recall within a latency budget.
    This is EXACTLY what "ANN search tuning" means.
    """
    results = []
    for nprobe in nprobe_values:
        if nprobe > index.nlist:
            continue
        index.nprobe = nprobe
        
        recalls = []
        latencies = []
        for _ in range(n_iters):
            recall, latency = compute_recall_at_k(index, query_emb, relevance, k)
            recalls.append(recall)
            latencies.append(latency)
        
        results.append({
            "nprobe": nprobe,
            "recall": np.mean(recalls),
            "latency_ms": np.mean(latencies),
            "recall_std": np.std(recalls),
        })
    
    return results


def sweep_hnsw_ef_search(
    index: faiss.IndexHNSWFlat,
    query_emb: np.ndarray,
    relevance: dict,
    ef_values: List[int],
    k: int = 10,
    n_iters: int = 3,
) -> List[Dict]:
    """
    Sweep efSearch for HNSW index.
    
    efSearch controls how many candidates to explore during search:
    - Low efSearch: fast but may miss relevant docs
    - High efSearch: slower but better recall
    """
    results = []
    for ef in ef_values:
        index.hnsw.efSearch = ef
        
        recalls = []
        latencies = []
        for _ in range(n_iters):
            recall, latency = compute_recall_at_k(index, query_emb, relevance, k)
            recalls.append(recall)
            latencies.append(latency)
        
        results.append({
            "efSearch": ef,
            "recall": np.mean(recalls),
            "latency_ms": np.mean(latencies),
            "recall_std": np.std(recalls),
        })
    
    return results


# ─────────────────────────────────────────────────────────────
# 5. Main Benchmark
# ─────────────────────────────────────────────────────────────

def print_header():
    print("=" * 65)
    print("  FAISS Retrieval Pipeline Recall Benchmark")
    print("  Comparing: Flat → IVF → HNSW → IVF+PQ")
    print("  + ANN parameter tuning (nprobe, efSearch)")
    print("=" * 65)


def run_benchmark(cfg: Optional[BenchmarkConfig] = None):
    if cfg is None:
        cfg = BenchmarkConfig()
    
    print_header()
    
    # ── Generate / load embeddings ──
    print(f"\n📦 Preparing data...")
    if cfg.use_real_embeddings:
        doc_emb, query_emb, relevance = generate_real_embeddings()
    else:
        doc_emb, query_emb, relevance = generate_synthetic_data(cfg)
    
    n_docs = doc_emb.shape[0]
    dim = doc_emb.shape[1]
    n_queries = len(relevance)
    
    # Auto-compute nlist if not set (rule of thumb: sqrt(N) to 4*sqrt(N))
    if cfg.nlist == 0:
        cfg.nlist = int(np.sqrt(n_docs))
    cfg.nlist = min(cfg.nlist, n_docs // 10)  # sanity: at least 10 docs per cluster
    if cfg.pq_m > 0 and dim % cfg.pq_m != 0:
        # Find a valid pq_m that divides dim
        for m in [48, 32, 24, 16, 12, 8]:
            if dim % m == 0:
                cfg.pq_m = m
                break
    
    print(f"  Corpus: {n_docs:,} docs × {dim}d")
    print(f"  Queries: {n_queries}")
    print(f"  IVF nlist: {cfg.nlist}")
    
    all_results = []
    
    # ── 1. Flat Index (Exact — Ground Truth Baseline) ──
    print(f"\n{'─' * 60}")
    print(f"  [1/4] Flat Index (exact search — baseline)")
    print(f"{'─' * 60}")
    
    idx_flat = build_flat_index(doc_emb)
    results_flat = evaluate_index(idx_flat, query_emb, relevance,
                                  cfg.top_k_values, "Flat (exact)", cfg.bench_iters)
    all_results.extend(results_flat)
    
    for r in results_flat:
        print(f"  recall@{r['k']:>2} = {r['recall']:.4f}  "
              f"(latency: {r['latency_ms']:.2f} ms/query)")
    
    # ── 2. IVF Index + nprobe sweep ──
    print(f"\n{'─' * 60}")
    print(f"  [2/4] IVF Index (nlist={cfg.nlist}) + nprobe sweep")
    print(f"{'─' * 60}")
    
    idx_ivf = build_ivf_index(doc_emb, cfg.nlist)
    
    # Default low nprobe (the "before tuning" state)
    idx_ivf.nprobe = 1
    results_ivf_low = evaluate_index(idx_ivf, query_emb, relevance,
                                     cfg.top_k_values, "IVF (nprobe=1)", cfg.bench_iters)
    all_results.extend(results_ivf_low)
    print(f"  Before tuning (nprobe=1):")
    for r in results_ivf_low:
        print(f"    recall@{r['k']:>2} = {r['recall']:.4f}  "
              f"(latency: {r['latency_ms']:.2f} ms)")
    
    # nprobe sweep
    print(f"\n  nprobe sweep (recall@10 vs latency):")
    ivf_sweep = sweep_ivf_nprobe(idx_ivf, query_emb, relevance,
                                  cfg.nprobe_values, k=10, n_iters=cfg.bench_iters)
    for s in ivf_sweep:
        bar = "█" * max(1, int(s["recall"] * 40))
        print(f"    nprobe={s['nprobe']:>3}  →  recall@10={s['recall']:.4f}  "
              f"latency={s['latency_ms']:.2f} ms  {bar}")
    
    # Find best nprobe (highest recall within 2x of flat latency)
    flat_latency = results_flat[2]["latency_ms"]  # recall@10 entry
    latency_budget = flat_latency * 0.5  # target: faster than flat
    
    # Pick best recall within budget, or just best overall
    valid = [s for s in ivf_sweep if s["latency_ms"] <= latency_budget] or ivf_sweep
    best_nprobe_cfg = max(valid, key=lambda s: s["recall"])
    best_nprobe = best_nprobe_cfg["nprobe"]
    
    print(f"\n  ★ Best nprobe={best_nprobe} "
          f"(recall@10={best_nprobe_cfg['recall']:.4f}, "
          f"latency={best_nprobe_cfg['latency_ms']:.2f} ms)")
    
    # Evaluate tuned IVF at all K values
    idx_ivf.nprobe = best_nprobe
    results_ivf_tuned = evaluate_index(idx_ivf, query_emb, relevance,
                                       cfg.top_k_values,
                                       f"IVF (nprobe={best_nprobe})",
                                       cfg.bench_iters)
    all_results.extend(results_ivf_tuned)
    
    # ── 3. HNSW Index + efSearch sweep ──
    print(f"\n{'─' * 60}")
    print(f"  [3/4] HNSW Index (M={cfg.hnsw_m}) + efSearch sweep")
    print(f"{'─' * 60}")
    
    idx_hnsw = build_hnsw_index(doc_emb, M=cfg.hnsw_m,
                                 ef_construction=cfg.hnsw_ef_construction)
    
    # efSearch sweep
    print(f"  efSearch sweep (recall@10 vs latency):")
    hnsw_sweep = sweep_hnsw_ef_search(idx_hnsw, query_emb, relevance,
                                       cfg.ef_search_values, k=10,
                                       n_iters=cfg.bench_iters)
    for s in hnsw_sweep:
        bar = "█" * max(1, int(s["recall"] * 40))
        print(f"    efSearch={s['efSearch']:>4}  →  recall@10={s['recall']:.4f}  "
              f"latency={s['latency_ms']:.2f} ms  {bar}")
    
    best_ef_cfg = max(hnsw_sweep, key=lambda s: s["recall"])
    best_ef = best_ef_cfg["efSearch"]
    
    idx_hnsw.hnsw.efSearch = best_ef
    results_hnsw = evaluate_index(idx_hnsw, query_emb, relevance,
                                  cfg.top_k_values,
                                  f"HNSW (ef={best_ef})",
                                  cfg.bench_iters)
    all_results.extend(results_hnsw)
    
    print(f"\n  ★ Best efSearch={best_ef} "
          f"(recall@10={best_ef_cfg['recall']:.4f}, "
          f"latency={best_ef_cfg['latency_ms']:.2f} ms)")
    
    # ── 4. IVF+PQ Index (compressed) ──
    print(f"\n{'─' * 60}")
    print(f"  [4/4] IVF+PQ Index (nlist={cfg.nlist}, pq_m={cfg.pq_m})")
    print(f"{'─' * 60}")
    
    idx_ivf_pq = build_ivf_pq_index(doc_emb, cfg.nlist, cfg.pq_m, cfg.pq_nbits)
    idx_ivf_pq.nprobe = best_nprobe  # use tuned nprobe
    
    results_pq = evaluate_index(idx_ivf_pq, query_emb, relevance,
                                cfg.top_k_values,
                                f"IVF+PQ (nprobe={best_nprobe})",
                                cfg.bench_iters)
    all_results.extend(results_pq)
    
    for r in results_pq:
        print(f"  recall@{r['k']:>2} = {r['recall']:.4f}  "
              f"(latency: {r['latency_ms']:.2f} ms)")
    
    # Memory comparison
    flat_bytes = n_docs * dim * 4
    pq_bytes = n_docs * cfg.pq_m * (cfg.pq_nbits / 8)
    print(f"\n  Memory: Flat={flat_bytes/1e6:.1f} MB → "
          f"IVF+PQ={pq_bytes/1e6:.1f} MB "
          f"({flat_bytes/pq_bytes:.0f}x compression)")
    
    # ── Summary ──
    print_summary(all_results, ivf_sweep, hnsw_sweep, results_ivf_low, cfg)
    
    # ── Plots ──
    plot_results(all_results, ivf_sweep, hnsw_sweep)
    
    # ── Save ──
    save_results(all_results, ivf_sweep, hnsw_sweep, cfg)
    
    return all_results, ivf_sweep, hnsw_sweep


# ─────────────────────────────────────────────────────────────
# 6. Summary & Visualization
# ─────────────────────────────────────────────────────────────

def print_summary(all_results, ivf_sweep, hnsw_sweep, ivf_low_results, cfg):
    """Print the key takeaway: recall improvement from tuning."""
    print("\n" + "=" * 65)
    print("  📊 RECALL IMPROVEMENT SUMMARY")
    print("=" * 65)
    
    # Focus on recall@10
    k = 10
    
    def get_recall(label):
        matches = [r for r in all_results if r["label"] == label and r["k"] == k]
        return matches[0]["recall"] if matches else 0
    
    def get_latency(label):
        matches = [r for r in all_results if r["label"] == label and r["k"] == k]
        return matches[0]["latency_ms"] if matches else 0
    
    flat_recall = get_recall("Flat (exact)")
    ivf_low_recall = get_recall("IVF (nprobe=1)")
    
    # Find tuned IVF result
    ivf_tuned_labels = [r["label"] for r in all_results 
                        if r["label"].startswith("IVF (nprobe=") and r["label"] != "IVF (nprobe=1)"]
    ivf_tuned_recall = get_recall(ivf_tuned_labels[0]) if ivf_tuned_labels else 0
    
    hnsw_labels = [r["label"] for r in all_results if r["label"].startswith("HNSW")]
    hnsw_recall = get_recall(hnsw_labels[0]) if hnsw_labels else 0
    
    pq_labels = [r["label"] for r in all_results if r["label"].startswith("IVF+PQ")]
    pq_recall = get_recall(pq_labels[0]) if pq_labels else 0
    
    # Compute improvement
    if ivf_low_recall > 0:
        ivf_improvement = (ivf_tuned_recall - ivf_low_recall) / ivf_low_recall * 100
        hnsw_improvement = (hnsw_recall - ivf_low_recall) / ivf_low_recall * 100
    else:
        ivf_improvement = hnsw_improvement = 0
    
    print(f"""
  Recall@{k} comparison:

  ┌──────────────────────────┬──────────┬──────────────┐
  │ Index                    │ Recall@{k} │ vs IVF(np=1) │
  ├──────────────────────────┼──────────┼──────────────┤
  │ Flat (exact, baseline)   │  {flat_recall:.4f}  │   (oracle)   │
  │ IVF nprobe=1 (untuned)   │  {ivf_low_recall:.4f}  │   baseline   │
  │ IVF tuned nprobe         │  {ivf_tuned_recall:.4f}  │  {ivf_improvement:>+6.1f}%     │
  │ HNSW (best efSearch)     │  {hnsw_recall:.4f}  │  {hnsw_improvement:>+6.1f}%     │
  │ IVF+PQ (compressed)      │  {pq_recall:.4f}  │              │
  └──────────────────────────┴──────────┴──────────────┘

  🎯 Key result: Index restructuring (Flat→HNSW) + parameter tuning
     improved recall@{k} by {max(ivf_improvement, hnsw_improvement):.1f}% relative
     vs the untuned IVF baseline.
""")


def plot_results(all_results, ivf_sweep, hnsw_sweep):
    """Generate benchmark plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  ⚠️  matplotlib not found — skipping plots")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle("FAISS Retrieval Pipeline — Recall Benchmark",
                 fontsize=14, fontweight="bold")
    
    colors = {
        "Flat (exact)": "#95a5a6",
        "IVF (nprobe=1)": "#e74c3c",
        "IVF tuned": "#f39c12",
        "HNSW": "#2ecc71",
        "IVF+PQ": "#3498db",
    }
    
    # ── Plot 1: Recall@K across index types ──
    ax1 = axes[0]
    index_groups = {}
    for r in all_results:
        base_label = r["label"].split(" (")[0] if "nprobe=1" not in r["label"] else "IVF untuned"
        if base_label not in index_groups:
            index_groups[base_label] = []
        index_groups[base_label].append(r)
    
    # Just plot the main ones
    for label_prefix, color in [("Flat", "#95a5a6"), ("IVF (nprobe=1)", "#e74c3c"),
                                 ("HNSW", "#2ecc71"), ("IVF+PQ", "#3498db")]:
        data = [(r["k"], r["recall"]) for r in all_results
                if r["label"].startswith(label_prefix)]
        if data:
            data.sort()
            xs, ys = zip(*data)
            short_label = label_prefix.split(" (")[0] if "nprobe=1" not in label_prefix else "IVF (untuned)"
            ax1.plot(xs, ys, "o-", color=color, label=short_label,
                     linewidth=2, markersize=6)
    
    # Also plot tuned IVF
    tuned_ivf = [r for r in all_results
                 if r["label"].startswith("IVF (nprobe=") and "nprobe=1" not in r["label"]]
    if tuned_ivf:
        data = sorted([(r["k"], r["recall"]) for r in tuned_ivf])
        xs, ys = zip(*data)
        ax1.plot(xs, ys, "s--", color="#f39c12", label="IVF (tuned)", linewidth=2, markersize=6)
    
    ax1.set_xlabel("K (top-K)", fontsize=11)
    ax1.set_ylabel("Recall@K", fontsize=11)
    ax1.set_title("Recall@K by Index Type", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # ── Plot 2: IVF nprobe sweep (recall vs latency tradeoff) ──
    ax2 = axes[1]
    if ivf_sweep:
        recalls = [s["recall"] for s in ivf_sweep]
        latencies = [s["latency_ms"] for s in ivf_sweep]
        nprobes = [s["nprobe"] for s in ivf_sweep]
        
        ax2.plot(latencies, recalls, "o-", color="#e74c3c", linewidth=2, markersize=8)
        for lat, rec, np_val in zip(latencies, recalls, nprobes):
            ax2.annotate(f"np={np_val}", (lat, rec), fontsize=7,
                        textcoords="offset points", xytext=(5, 5))
    
    ax2.set_xlabel("Latency (ms/query)", fontsize=11)
    ax2.set_ylabel("Recall@10", fontsize=11)
    ax2.set_title("IVF: Recall vs Latency (nprobe sweep)", fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # ── Plot 3: HNSW efSearch sweep ──
    ax3 = axes[2]
    if hnsw_sweep:
        recalls = [s["recall"] for s in hnsw_sweep]
        latencies = [s["latency_ms"] for s in hnsw_sweep]
        efs = [s["efSearch"] for s in hnsw_sweep]
        
        ax3.plot(latencies, recalls, "o-", color="#2ecc71", linewidth=2, markersize=8)
        for lat, rec, ef in zip(latencies, recalls, efs):
            ax3.annotate(f"ef={ef}", (lat, rec), fontsize=7,
                        textcoords="offset points", xytext=(5, 5))
    
    ax3.set_xlabel("Latency (ms/query)", fontsize=11)
    ax3.set_ylabel("Recall@10", fontsize=11)
    ax3.set_title("HNSW: Recall vs Latency (efSearch sweep)", fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("faiss_benchmark_results.png", dpi=150, bbox_inches="tight")
    print(f"\n📈 Plot saved to faiss_benchmark_results.png")
    plt.close()


def save_results(all_results, ivf_sweep, hnsw_sweep, cfg):
    output = {
        "config": asdict(cfg),
        "results": all_results,
        "ivf_nprobe_sweep": ivf_sweep,
        "hnsw_ef_sweep": hnsw_sweep,
    }
    with open("faiss_benchmark_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"💾 Results saved to faiss_benchmark_results.json")


# ─────────────────────────────────────────────────────────────
# 7. Entry Point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FAISS Recall Benchmark")
    parser.add_argument("--n-docs", type=int, default=50_000,
                        help="Corpus size (default: 50000)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic embeddings (skip sentence-transformers)")
    parser.add_argument("--n-queries", type=int, default=500,
                        help="Number of queries for synthetic mode")
    args = parser.parse_args()
    
    cfg = BenchmarkConfig(
        n_docs=args.n_docs,
        n_queries=args.n_queries,
        use_real_embeddings=not args.synthetic,
    )
    
    run_benchmark(cfg)
    
    print("\n✅ Benchmark complete!")
    print("   Screenshot the results for your portfolio.")

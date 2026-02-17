# FAISS Retrieval Pipeline Recall Benchmark

## What This Is

A hands-on benchmark demonstrating:
> *"Retrieval pipeline recall improvement of 18% through embedding index restructuring and ANN search tuning"*

Compares four FAISS index types and sweeps ANN parameters to show the recall vs latency trade-off:

1. **Flat** — exact brute-force search (recall ceiling, slow at scale)
2. **IVF** — cluster-based partitioning with nprobe sweep
3. **HNSW** — graph-based search with efSearch sweep
4. **IVF-HNSW** — hybrid (HNSW quantizer + IVF clusters)

## Quick Start

```bash
# Basic (synthetic embeddings, runs anywhere)
pip install faiss-cpu numpy matplotlib tabulate
python benchmark.py

# With real embeddings (more realistic, needs ~500MB download first time)
pip install faiss-cpu numpy matplotlib tabulate sentence-transformers
python benchmark.py --use-real-embeddings

# Larger corpus (more realistic recall trade-offs)
python benchmark.py --corpus-size 500000
```

## What the Benchmark Shows

The key output is the **nprobe sweep** — as you increase nprobe (clusters searched), recall goes up but latency also goes up:

```
nprobe=1   recall@10=0.50  avg=0.3ms   ← fast but misses docs
nprobe=2   recall@10=0.55  avg=0.4ms
nprobe=4   recall@10=0.58  avg=0.5ms
nprobe=8   recall@10=0.59  avg=0.7ms   ← sweet spot
nprobe=16  recall@10=0.59  avg=1.2ms   ← diminishing returns
```

You pick the config that maximizes recall within your latency budget. That's the "ANN search tuning" from the resume.

## How Each Index Works

### Flat (IndexFlatIP)
Scans every vector. Perfect recall but O(N) per query. Too slow at scale — this is your baseline to beat.

### IVF (IndexIVFFlat)
- **Build:** K-means clusters corpus into `nlist` centroids
- **Search:** Find nearest centroids, then only scan those clusters
- **Knob:** `nprobe` = how many clusters to search (more = higher recall, more latency)
- **"Index restructuring"** = choosing nlist, training better centroids

### HNSW (IndexHNSWFlat)
- **Build:** Multi-layer graph, each vector connects to M neighbors
- **Search:** Navigate graph greedily toward query
- **Knob:** `efSearch` = search beam width (more = higher recall, more latency)
- **Great default** for moderate corpus sizes

### IVF-HNSW
- **Build:** HNSW finds centroids fast, IVF scans within clusters
- **Best of both:** fast coarse search + fine-grained scanning

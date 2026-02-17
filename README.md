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

## Interview Talking Points

**"How did you improve retrieval recall by 18%?"**

> "Our RAG stack was missing relevant documents. I built a labeled evaluation set and measured recall@10 under a latency budget. The baseline was a simple IVF index with default parameters. I restructured the index — tried IVF, HNSW, and IVF-HNSW configurations — and then swept ANN parameters like nprobe and efSearch. For each configuration I measured recall@K and P95 latency. The best config improved recall@10 by about 18% relative at the same latency budget, which directly improved downstream answer quality."

**"How did you get ground-truth relevance labels?"**

> "We sampled real queries from logs, assembled candidate sets using multiple retrieval methods (exact search, BM25, high-recall ANN), and got relevance labels through a mix of human annotation for high-value queries and implicit feedback (dwell time, conversions) as weak labels for the long tail."

**"How did you deal with the latency-recall trade-off?"**

> "I treated it as a joint objective. I swept parameters, measured both recall@K and P95 latency for each config, eliminated anything exceeding our latency target, and picked the highest-recall config from what remained."

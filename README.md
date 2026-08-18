# db_bad_clust — ML-Powered DB Schema Anti-Pattern Detection

Clustering pipeline that detects schema anti-patterns in Oracle databases using BERT embeddings + structural features.

---

## Quick Start

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Notebooks — ML Pipeline

Run in order:

1. `notebooks/01_data_preparation.ipynb` — schema extraction + preprocessing + structural encoding
2. `notebooks/02_ml_embeddings.ipynb` — BERT embeddings + feature building + dimensionality reduction
3. `notebooks/03_clustering.ipynb` — clustering + ground truth comparison
4. `notebooks/04_analysis.ipynb` — evaluation + recommendations + visualization

### Configuration

Set `SKIP_BERT = True` in notebook 02 to use synthetic embeddings (avoids ~1.5GB download).

Best known config:

- alpha=0.35, beta=0.15, gamma=0.50, delta=0.00
- UMAP 5 components, KMeans 5 clusters

## Tests

```bash
python3 -m pytest tests/ -v   # 364 tests, no DB required
```

---

**Results:** UMAP+KMeans achieves ARI=0.4085, NMI=0.6109. See `docs/phase3/02_final_report.md`.

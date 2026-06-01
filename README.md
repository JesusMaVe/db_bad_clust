# db_bad_clust — ML-Powered DB Schema Anti-Pattern Detection

Clustering pipeline that detects schema anti-patterns in Oracle databases using BERT embeddings + structural features.

---

## Quick Start

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 1 — Generate Oracle DB with 22 bad-schema tables

```bash
docker compose up -d                          # Start Oracle 23c
cd scripts && python3 orchestrator.py         # Create tables + dirty data
```

**Note:** Must `cd scripts` first — paths are relative to CWD.

## Phase 2/3 — ML Pipeline

```bash
cd scripts
python3 phase2_orchestrator.py --method umap --cluster-method kmeans --validate
```

### Key flags

| Flag | Default | Description |
|---|---|---|
| `--method` | `pca` | `pca`, `umap`, `svd`, `tsne` |
| `--cluster-method` | `kmeans` | `kmeans`, `dbscan`, `hdbscan`, `agglomerative`, `meanshift` |
| `--validate` | — | Compare clusters against anti-pattern ground truth |
| `--tune` | — | Grid search 360 weight combinations |
| `--skip-bert` | — | Use synthetic embeddings (skip ~1.5 GB download) |

### Best known config

```bash
python3 phase2_orchestrator.py \
  --method umap --alpha 0.35 --beta 0.15 --gamma 0.50 --delta 0.00 \
  --n-components 5 --cluster-method kmeans --n-clusters 5 --validate
```

## Tests

```bash
python3 -m pytest tests/ -v        # 364 tests, no DB required
ruff check scripts/                 # 0 lint errors
```

## Dashboard

Open `dashboard/index.html` in a browser to explore the latest results.

---

**Results:** UMAP+KMeans achieves ARI=0.4085, NMI=0.6109. See `docs/phase3/02_final_report.md`.

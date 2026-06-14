# AGENTS.md — db_bad_clust (ml_bad_db_trainer)

**Status: Phase 1 (COMPLETE) → Phase 2 (COMPLETE) → Phase 3 (COMPLETE) → Notebooks (CURRENT)**

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 1 — Oracle DB Generation (scripts/)

```bash
docker compose up -d
python3 anti_patterns.py  # Generate anti-pattern table catalog
```

## Phase 2/3 — ML Pipeline (Notebooks)

Run in order from `notebooks/` directory:
1. `01_data_preparation.ipynb` — schema extraction + preprocessing + structural encoding
2. `02_ml_embeddings.ipynb` — BERT embeddings + feature building + dimensionality reduction
3. `03_clustering.ipynb` — clustering + ground truth comparison
4. `04_analysis.ipynb` — evaluation + recommendations + visualization

Each notebook saves intermediate data via pickle in `output/` for the next notebook.

## Module Files (root)

Python modules live at the repo root (not in scripts/):

### Data Pipeline
- `db_connector.py` — Oracle connection
- `schema_extractor.py` — metadata extraction from Oracle
- `text_preprocessor.py` — column name preprocessing for BERT
- `structural_encoder.py` — data type + constraint encoding
- `bert_embedder.py` — BERT embedding generation (supports DI for testing)
- `feature_builder.py` — composite vector construction

### ML Pipeline
- `dimensionality_reducer.py` — PCA/UMAP/t-SNE/SVD
- `cluster_engine.py` — KMeans/DBSCAN/HDBSCAN/Agglomerative/MeanShift

### Evaluation (SRP-refactored)
- `evaluator.py` — facade module (delegates to specialized modules)
- `metrics.py` — internal clustering quality metrics (silhouette, DB, CH)
- `validation.py` — external validation against ground truth (ARI, NMI)
- `anomaly.py` — anomaly detection (centroid distance, precision/recall)
- `reporter.py` — formatted text report generation

### Anti-patterns
- `recommender.py` — anti-pattern recommendations
- `recommendation_reporter.py` — formatted text report generation
- `ground_truth.py` — anti-pattern ground truth mapping
- `anti_patterns.py` — table catalog (1430 lines, data-only)
- `exceptions.py` — exception hierarchy

## Configuration

Set `SKIP_BERT = True` in notebook 02 to use synthetic embeddings (avoids ~1.5GB download).

Best known config (hardcoded in notebooks):
- alpha=0.35, beta=0.15, gamma=0.50, delta=0.00
- UMAP 5 components, KMeans 5 clusters

## Tests

```bash
python3 -m pytest tests/ -v   # 396 tests, no DB required
```

## Gotchas

- Notebooks use `sys.path.insert(0, str(Path.cwd().parent))` to import root modules
- BERT downloads ~1.5GB on first run — use `SKIP_BERT=True`
- Table names case-sensitive — always double-quote
- Oracle 23c healthcheck takes ~30s
- `config.yaml` says `tables_count:10` but actual count is 23

## Reference

- `docs/phase2/00_anti_patterns_catalog.md` — anti-pattern catalog
- `docs/phase2/11_completion_report.md` — Phase 2 results

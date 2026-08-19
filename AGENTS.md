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
python3 anti_patterns.py   # Generate anti-pattern table catalog
python3 apply_comments.py  # Apply documentation overlay (comments) to the DB
```

## Phase 2/3 — ML Pipeline (Notebooks)

Run in order from `notebooks/` directory:
1. `01_data_preparation.ipynb` — schema extraction + preprocessing + structural encoding
2. `02_ml_embeddings.ipynb` — BERT embeddings + feature building + dimensionality reduction
3. `03_classification.ipynb` — Rule Engine + ML Fallback
4. `04_analysis.ipynb` — Classification metrics

Each notebook saves intermediate data via pickle in `output/` for the next notebook.

## Module Files (root)

Python modules live at the repo root (not in scripts/):

### Data Pipeline
- `db_connector.py` — Oracle connection
- `schema_extractor.py` — metadata extraction from Oracle (bulk queries: one per dictionary view, incl. comments/identity/virtual columns)
- `apply_comments.py` — applies the documentation overlay (TABLE_COMMENTS/COLUMN_COMMENTS in anti_patterns.py) to the DB
- `text_preprocessor.py` — column name preprocessing for BERT (appends column comments; lowercases only ALL-CAPS tokens)
- `structural_encoder.py` — data type + constraint encoding
- `bert_embedder.py` — sentence embeddings via paraphrase-multilingual-MiniLM-L12-v2 (mean pooling + L2 norm, 384 dims)
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

Set `SKIP_BERT = True` in notebook 02 to use synthetic embeddings (avoids ~470MB download).

Best known config:
- Feature weights (classification notebooks): alpha=0.15, beta=0.35, gamma=0.45, delta=0.05
- Clustering (re-validated, issue #1): alpha=0.00, beta=0.35, gamma=0.45, delta=0.20 + PCA 20D + HDBSCAN (ARI 0.5815)
- UMAP: n_components=5, n_neighbors=25, min_dist=0.05
- Rule Engine: 10 categories, ~73% coverage, accuracy 0.9465 / F1-macro 0.7938
- ML fallback (One-Class SVM) disabled — worse than rules

## Tests

```bash
python3 -m pytest tests/ -v   # 458 tests, no DB required
```

## Gotchas

- Notebooks use `sys.path.insert(0, str(Path.cwd().parent))` to import root modules
- MiniLM-L12 downloads ~470MB on first run — use `SKIP_BERT=True`
- Table names case-sensitive — always double-quote
- Oracle 23c healthcheck takes ~30s
- `config.yaml` says `tables_count:10` but actual count is 23
- venv shebangs are stale (folder was renamed) — use `.venv/bin/python -m <cmd>`, not entrypoints
- Oracle rejects exact duplicate indexes (ORA-01408) — redundant-index anti-pattern is a composite index with duplicated prefix
- Keyword matching in rule_engine/ground_truth uses token-boundary (`_kw_match`) — substring matching caused false positives ('fec' in 'afectada')

## Reference

- `docs/phase2/00_anti_patterns_catalog.md` — anti-pattern catalog
- `docs/phase2/11_completion_report.md` — Phase 2 results

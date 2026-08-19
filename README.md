# db_bad_clust — ML-Powered DB Schema Anti-Pattern Detection

Pipeline that detects schema anti-patterns in Oracle databases using sentence embeddings + structural features + a rule engine.

## Quick Start

```bash
docker compose up -d          # Oracle 23c (healthcheck ~30s)
source .venv/bin/activate
pip install -r requirements.txt
python3 apply_comments.py     # documentation overlay (comments) onto the DB
```

Note: venv shebangs are stale (folder was renamed) — run `.venv/bin/python -m <cmd>` instead of entrypoints.

## Notebooks — ML Pipeline

Run in order from `notebooks/`:

1. `01_data_preparation.ipynb` — schema extraction + preprocessing + structural encoding
2. `02_ml_embeddings.ipynb` — sentence embeddings + feature building + dimensionality reduction
3. `03_classification.ipynb` — Rule Engine + evaluation vs ground truth
4. `04_analysis.ipynb` — metrics + recommendations

Or headless:

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/01_data_preparation.ipynb
# ... same for 02, 03, 04
```

Each notebook saves intermediate data via pickle in `output/` for the next one.

### Configuration

Set `SKIP_BERT = True` in notebook 02 to use synthetic embeddings (avoids ~470MB download).

Best known config:

- Classification (Rule Engine): accuracy 0.9465 / F1-macro 0.7938
- Clustering: alpha=0.00, beta=0.35, gamma=0.45, delta=0.20 + PCA 20D + HDBSCAN (ARI 0.5815)
- ML fallback (One-Class SVM) disabled — worse than rules

## Tests

```bash
python3 -m pytest tests/ -v   # 458 tests, no DB required
```

## Docs

- `docs/implementation.md` — what was actually built + experimental results (§4-bis/4-ter: re-validation and post-improvement summary)
- `docs/research_extraction_preprocessing.md` — best-practices research (primary sources)
- `docs/phase3/02_final_report.md` — Phase 3 final report

---

**Results:** Rule Engine accuracy 0.9465 / F1-macro 0.7938 (243 columns, 10 classes). Clustering best: PCA-20D + HDBSCAN, ARI=0.5815.

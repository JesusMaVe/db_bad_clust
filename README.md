# db_bad_clust — ML-Powered DB Schema Anti-Pattern Detection

Detects schema anti-patterns in Oracle databases: a rule engine classifies columns/tables
(wrong data types, reserved words, EAV, polymorphic keys, giant tables, ...) and an
unsupervised clustering pipeline (BERT embeddings + structural features) validates the
same anti-patterns without labels.

## Quick Start

```bash
docker compose up -d                   # Oracle 23c (healthcheck ~30s)
.venv/bin/python -m pip install -e ".[dev]"
.venv/bin/python scripts/apply_comments.py   # documentation overlay (comments) onto the DB
```

Note: venv shebangs are stale (folder was renamed) — always run `.venv/bin/python -m <cmd>`
instead of bare entrypoints.

## Structure

```
src/db_bad_clust/
├── data/          # db_connector, schema_extractor — Oracle in, dataclasses out
├── features/      # text_preprocessor, structural_encoder, bert_embedder, feature_builder
├── clustering/    # dimensionality_reducer, cluster_engine
├── rules/         # rule_engine (detection), recommender, recommendation_reporter
├── evaluation/    # metrics, validation, anomaly — clustering quality + ground-truth checks
└── generation/    # anti_patterns (catalog), schema_generator, benchmark, ddl_generator, ground_truth, label_export
scripts/           # apply_comments.py — one-shot DB scripts, not imported by the package
notebooks/         # 01-04, the actual pipeline entrypoint
tests/             # pytest, no DB required (mock-based)
docs/              # research + status reports (superpowers/ plans are local-only)
```

## Notebooks — ML Pipeline

Run in order from `notebooks/`:

1. `01_data_preparation.ipynb` — schema extraction + preprocessing + structural encoding
2. `02_ml_embeddings.ipynb` — sentence embeddings + feature building + dimensionality reduction
3. `03_classification.ipynb` — Rule Engine + evaluation vs ground truth
4. `04_analysis.ipynb` — metrics + recommendations + clustering ARI

Or headless:

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/01_data_preparation.ipynb
# ... same for 02, 03, 04
```

Each notebook saves intermediate data via pickle in `output/` for the next one.
Notebooks 01/03/04 need Oracle up; 02 needs it only transitively (reads the pickle).

**Manual ground truth (optional)**: `.venv/bin/python -m db_bad_clust.generation.label_export`
dumps `output/manual_labels.csv` (empty `label` column) from Oracle; fill labels (vocabulary in
`db_bad_clust.generation.ground_truth.MANUAL_LABEL_VOCABULARY`) and notebook 03 auto-uses them
instead of the rule-engine catalog, for an honest, non-circular evaluation.

### Configuration

Set `SKIP_BERT = True` in notebook 02 to use synthetic embeddings (avoids ~470MB download).

Best known config:

- Rule Engine (aligned GT, circular): accuracy 0.9877 / F1-macro 0.9057
- Rule Engine (manual GT, honest): accuracy 0.8930 / F1-macro 0.8467
- Clustering: α=0.00, β=0.35, γ=0.45, δ=0.20 + PCA 20D + HDBSCAN (ARI 0.5815)

## Tests

```bash
.venv/bin/python -m pytest tests/ -v   # 460 tests, no DB required
.venv/bin/python -m ruff check src tests scripts
```

## Docs

- `docs/00_PROJECT_STATUS_REPORT.md` — full status report, phase by phase
- `docs/research_extraction_preprocessing.md` — best-practices research (primary sources)
- `docs/rule_engine_and_future_ml_report.md` — rule engine design + ML fallback exploration
- See `AGENTS.md` for the invariants an agent (or a new contributor) needs before touching the rule engine or the generation catalog.

---

**Results:** Rule Engine accuracy 0.9877 / F1-macro 0.9057 (aligned GT, 243 columns) ·
0.8930 / F1-macro 0.8467 (manual GT, honest). Clustering best: PCA-20D + HDBSCAN, ARI=0.5815.

# AGENTS.md — db_bad_clust (ml_bad_db_trainer)

**Status: Phase 1 (COMPLETE) → Phase 2 (COMPLETE) → Phase 3 (COMPLETE) → Notebooks (CURRENT)**

## Setup

```bash
docker compose up -d          # Oracle 23c, healthcheck ~30s
pip install -r requirements.txt
```

Use `.venv/bin/python -m <cmd>` — venv shebangs are stale (folder was renamed), entrypoints like `jupyter`/`pip`/`nbconvert` fail.

## Oracle DB — NOT fully reproducible from the repo

- The 23 tables were created ad-hoc; there is no DDL generator in the repo (`scripts/` no longer exists).
- `anti_patterns.py` is a **data-only catalog** (no `__main__`) — importing it gives the expected schema; running it does nothing.
- The DB is expected to match the catalog. If they drift (e.g. a catalog table missing in Oracle), metrics silently change — this happened with DATOS_MAESTROS (issue #2).
- The only runnable DB step is the documentation overlay:

```bash
.venv/bin/python apply_comments.py   # applies TABLE_COMMENTS/COLUMN_COMMENTS to Oracle
```

- It requires the Oracle container up (`docker compose up -d`, wait for healthy).

## ML Pipeline (Notebooks)

Run in order; each saves a pickle to `output/` for the next:

1. `notebooks/01_data_preparation.ipynb` — extraction + preprocessing + structural encoding
2. `notebooks/02_ml_embeddings.ipynb` — sentence embeddings + feature building + reduction
3. `notebooks/03_classification.ipynb` — Rule Engine + evaluation vs ground truth
4. `notebooks/04_analysis.ipynb` — metrics + recommendations + clustering ARI (writes `output/notebook_results/`)

**Manual ground truth (optional)**: `label_export.py` dumps `output/manual_labels.csv` (empty `label` column) from Oracle; fill labels (vocabulary in `ground_truth.MANUAL_LABEL_VOCABULARY`) and notebook 03 auto-uses them instead of the rule-engine catalog, giving an honest, non-circular evaluation.

Headless (verified):

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/01_data_preparation.ipynb
```

Notebooks use `sys.path.insert(0, str(Path.cwd().parent))` — run from `notebooks/` in Jupyter, or any dir with nbconvert.
Notebooks 01+03+04 need Oracle up; 02 needs it only transitively (reads pickle). Notebooks 01–02 re-run takes minutes with real BERT (`SKIP_BERT=False`); 03–04 are fast.

## Modules (repo root, not scripts/)

- Data: `db_connector.py`, `schema_extractor.py` (bulk queries: one per dictionary view + redundant-index scan), `text_preprocessor.py`, `structural_encoder.py`, `bert_embedder.py` (MiniLM-L12, mean pooling, 384D), `feature_builder.py`
- ML: `dimensionality_reducer.py`, `cluster_engine.py`
- Evaluation: `evaluator.py` (facade) → `metrics.py`, `validation.py`, `anomaly.py`, `reporter.py`
- Anti-patterns: `rule_engine.py`, `recommender.py`, `recommendation_reporter.py`, `ground_truth.py`, `anti_patterns.py` (catalog), `apply_comments.py`, `exceptions.py`

## Tests & lint

```bash
.venv/bin/python -m pytest tests/ -v                 # 500 tests, NO DB required (mock-based)
.venv/bin/python -m pytest tests/test_rule_engine.py::TestSchemaSpySignals -q   # single test
.venv/bin/python -m ruff check .                     # ~66 pre-existing errors (mostly old tests + rule_engine.py); new code should be lint-clean
```

## Critical invariants (learned the hard way)

- **Ground truth is rule-engine aligned** — `ground_truth.build_ground_truth_map()` and the generator manifest both run `rule_engine.classify()` on the schema, so detection and truth share label semantics. There is no duplicated keyword logic in `ground_truth.py` anymore (delegates to `rule_engine.py`); keep it that way.
- **Keyword lists live only in `rule_engine.py`** (and mirror copies in `ddl_generator.py` / `schema_generator.py` for generation) — any change (keyword, exception, matcher) must be applied in all three or generation/metrics silently diverge.
- **Keyword matching is token-boundary** (`_kw_match`) — substring matching caused false positives ('fec' matched inside 'afectada'). Never revert to `kw in name`.
- **SchemaSpy detections are report-level** (`detect_missing_pk` / `detect_redundant_indexes` / `detect_implicit_fks` are NOT in the classify() rule chain) — adding them there would mask all other detections (all 23 tables lack PK) and pollute classification metrics.
- **Column-level date/number_as_text beats table-level inconsistent_naming** in the merge (deliberate; issue #3).
- **ColumnRuleEngine must be invoked per table** in `classify()` — a flat run collapses same-named columns (ACTIVO × 4) into one detection (issue #2).
- Ground truth is structural per-table; semantic embeddings do NOT improve ARI (α=0 wins) — embeddings are for semantic redundancy, not anti-pattern classification.

## Configuration

- `SKIP_BERT = True` in notebook 02 → synthetic embeddings (avoids ~470MB MiniLM download). Current pickle was built with real embeddings.
- Classification weights (notebooks): α=0.15, β=0.35, γ=0.45, δ=0.05
- Best clustering (re-validated, issue #1): α=0.00, β=0.35, γ=0.45, δ=0.20 + PCA 20D + HDBSCAN → ARI 0.5815
- Rule Engine (aligned GT, circular): accuracy 0.9877 / F1-macro 0.9057 (243 columns); `impossible_data` 0.00 because Oracle extraction loses `fk_references_column`
- Rule Engine (manual GT, honest): accuracy 0.8642 / F1-macro 0.7745 — weakness = TODO_EN_UNO (giant_table en vez de polymorphic) + impossible_data no detectado + sobredetección de clean
- One-Class SVM fallback disabled — worse than rules

## Gotchas

- Table names case-sensitive in Oracle — always double-quote
- Oracle allows only one LONG column per table (ORA-01754) — catalog uses CLOB/BLOB for the rest
- Oracle rejects exact duplicate indexes (ORA-01408) — redundant-index anti-pattern is a composite index duplicating a single-column prefix; the two in the DB (EMPLEADOS, ORDENES_COMPRA) were created manually
- `config.yaml` says `tables_count:10` but actual count is 23 — config value is unused
- `tests/conftest.py` inserts a non-existent `scripts/` path — harmless, imports resolve from root

## Reference

- `docs/implementation.md` — what was built + experimental results; §4-bis/§4-ter are the current post-improvement numbers
- `docs/research_extraction_preprocessing.md` — best-practices research with primary sources
- `docs/phase3/02_final_report.md` — Phase 3 report (pre-revalidation numbers)
- GitHub issues #1–#5 (closed) document each improvement with root-cause analysis

# AGENTS.md — db_bad_clust (ml_bad_db_trainer)

**Status: Phase 1-3 (COMPLETE) → Notebooks (COMPLETE) → Packaging (COMPLETE) → Audit tool (CURRENT)**

The project's goal is now an **Oracle-only schema audit tool**, not hypothesis validation. H₁
is settled and refuted (see "A vs B" below); the ML branch is frozen as documented evidence,
not deleted. Explicitly out of scope, by the owner's decision: multi-SGBD, REST API,
containerization, web dashboard.

## Setup

```bash
docker compose up -d                         # Oracle 23c, healthcheck ~30s
.venv/bin/python -m pip install -e ".[dev]"  # installs db_bad_clust + runtime/dev deps from pyproject
```

Base deps are `oracledb` + `pyyaml` only; all ML (torch, transformers, umap-learn, hdbscan,
sklearn, numpy) is in the `research` extra, which `[dev]` pulls. The audit path imports none
of it — keep it that way.

## The audit tool (CLI)

```bash
.venv/bin/python -m db_bad_clust.cli audit --sql output/fix.sql --json output/audit.json
.venv/bin/python -m db_bad_clust.cli compare --sweep     # A vs B, needs no database
```

`audit` chains extractor → rule_engine → recommender → ddl_generator. Verified against the
live container: 23 tables / 243 columns, 541 lines of remediation SQL. Before the CLI existed
`ddl_generator` had no caller at all.

Use `.venv/bin/python -m <cmd>` — venv shebangs are stale (folder was renamed), entrypoints like
`jupyter`/`pip`/`nbconvert` fail.

## Oracle DB — NOT fully reproducible from the repo

- The 23 tables were created ad-hoc; there is no DDL generator for the *real* DB in the repo
  (`db_bad_clust.generation.ddl_generator` emits fix scripts for detections, not the original schema).
- `db_bad_clust.generation.anti_patterns` is a **data-only catalog** (no `__main__`) — importing it
  gives the expected schema; running it does nothing.
- The DB is expected to match the catalog. If they drift (e.g. a catalog table missing in Oracle),
  metrics silently change — this happened with DATOS_MAESTROS (issue #2).
- The only runnable DB step is the documentation overlay:

```bash
.venv/bin/python scripts/apply_comments.py   # applies TABLE_COMMENTS/COLUMN_COMMENTS to Oracle
```

- It requires the Oracle container up (`docker compose up -d`, wait for healthy).

## ML Pipeline (Notebooks)

Run in order; each saves a pickle to `output/` for the next:

1. `notebooks/01_data_preparation.ipynb` — extraction + preprocessing + structural encoding
2. `notebooks/02_ml_embeddings.ipynb` — sentence embeddings + feature building + reduction
Notebooks 03/04 were deleted (superseded by `audit` / `compare`; their Random Forest
benchmark lives in `evaluation/ml_baselines.py`). **01 and 02 must stay** — they are the
only producers of `output/intermediate_02.pkl`, which `compare` reads.

**Manual ground truth (optional)**: `db_bad_clust.generation.label_export` dumps
`output/manual_labels.csv` (empty `label` column) from Oracle; fill labels (vocabulary in
`db_bad_clust.generation.ground_truth.MANUAL_LABEL_VOCABULARY`) and `compare` uses them
instead of the rule-engine catalog, giving an honest, non-circular evaluation.

Headless (verified):

```bash
.venv/bin/python -m nbconvert --to notebook --execute --inplace notebooks/01_data_preparation.ipynb
```

Notebooks import the installed package (`from db_bad_clust.data.schema_extractor import ...`) —
no `sys.path` hacks. Notebook 01 needs Oracle up; 02 needs it only transitively (reads
pickle). A re-run takes minutes with real BERT (`SKIP_BERT=False`).

## Package layout

```
src/db_bad_clust/
├── cli.py           audit (live Oracle) + compare (A vs B, no DB) — the product surface
├── exceptions.py    BadDBError + subclasses shared across the package
├── data/            db_connector, schema_extractor
├── features/        text_preprocessor, structural_encoder, bert_embedder, feature_builder
├── clustering/       dimensionality_reducer, cluster_engine
├── rules/           rule_engine (detection + SQL_RESERVED_WORDS, _kw_match — the single home
│                    for keyword logic), recommender, recommendation_reporter
├── evaluation/       metrics (internal), validation (external, + cross_table_analysis /
│                    cluster_composition salvaged from the deleted evaluator facade), anomaly,
│                    head_to_head (A vs B, both rulers, no DB)
└── generation/       anti_patterns (catalog), schema_adapter (catalog -> DatabaseSchema),
                     ddl_generator, ground_truth (also owns RULE_TO_MANUAL +
                     MANUAL_LABEL_VOCABULARY), label_export
scripts/             apply_comments.py — one-shot DB script, not imported by anything
```

There is no `evaluator.py` / `reporter.py` facade anymore — they were pure pass-throughs
(deletion test: deleting them removed a hop, concentrated nothing). Call `metrics.py`,
`validation.py`, `anomaly.py` directly.

## Tests & lint

```bash
.venv/bin/python -m pytest tests/ -v                 # 472 tests, NO DB required (mock-based)
.venv/bin/python -m pytest tests/test_rule_engine.py::TestSchemaSpySignals -q   # single test
.venv/bin/python -m ruff check src tests scripts      # 11 pre-existing errors (RUF012 mutable
                                                       # class defaults in rule_engine.py/anti_patterns.py); new code should be lint-clean
```

## Critical invariants (learned the hard way)

- **Ground truth is rule-engine aligned** — `ground_truth.build_ground_truth_map()` and the
  generator manifest both run `rule_engine.classify()` on the schema, so detection and truth
  share label semantics.
- **Keyword lists live only in `rule_engine.py`** — `SQL_RESERVED_WORDS`, `_kw_match`,
  `NUMBER_KEYWORDS`, `DATE_KEYWORDS_HIGH` are imported (not copied) by `ddl_generator.py`.
  The old "mirror copies, keep in sync by hand" invariant is gone — don't reintroduce a local
  copy in a generation module.
- **Rule label → manual label mapping lives only in `ground_truth.RULE_TO_MANUAL`** —
  `recommender.py` and `label_export.py` both import it instead of keeping their own map.
  Same for the label vocabulary (`ground_truth.MANUAL_LABEL_VOCABULARY`).
- **Keyword matching is token-boundary** (`_kw_match`) — substring matching caused false
  positives ('fec' matched inside 'afectada'). Never revert to `kw in name`.
- **SchemaSpy detections are report-level** (`detect_missing_pk` / `detect_redundant_indexes` /
  `detect_implicit_fks` are NOT in the `classify()` rule chain) — adding them there would mask
  all other detections (all 23 tables lack PK) and pollute classification metrics.
- **Column-level date/number_as_text beats table-level inconsistent_naming** in the merge
  (deliberate; issue #3).
- **Column-level polymorphic beats table-level giant_table** in the merge (deliberate; issue #7:
  TODO_EN_UNO `_O_` columns) — eav/inconsistent_naming keep masking polymorphic
  (CONFIGURACION.TIPO_DATO stays eav by design).
- **ColumnRuleEngine must be invoked per table** in `classify()` — a flat run collapses
  same-named columns (ACTIVO × 4) into one detection (issue #2).
- Ground truth is structural per-table; semantic embeddings do NOT improve ARI (α=0 wins) —
  embeddings are for semantic redundancy, not anti-pattern classification.
- **A vs B is settled, measured, and reproducible** (`evaluation/head_to_head.py`, no DB
  needed). Both branches scored with both rulers on the 243 manual labels:

  | Branch          |    ARI |    NMI | Accuracy | F1-macro |
  | --------------- | -----: | -----: | -------: | -------: |
  | Rule engine (A) | 0.7667 | 0.8058 |   0.8930 |   0.8467 |
  | Clustering (B)  | 0.5008 | 0.4199 |   0.7078 |   0.2639 |

  A wins all four, including B's own metrics, and B's accuracy/F1 are an **upper bound**
  (majority-vote cluster naming leaks the ground truth to B). B scores 0.000 on six of ten
  classes — a cluster is named for its majority, so minority anti-patterns are never
  pronounced. `alpha_sweep()`: BERT is actively harmful, ARI 0.5008 (α=0) → ~0.30 once on.
  A's own weaknesses, also measured: 14 columns in CLIENTES_DIRECCIONES over-flagged as
  `inconsistent_naming` (table-level verdict contaminating clean columns), and
  `impossible_data` at 0.000 (limitation #6). Write-up: `docs/veredicto_reglas_vs_ml.html`.
- **`head_to_head._run_branch_a` must apply `RULE_TO_MANUAL`** — the engine emits finer
  labels (`bad_boolean`, `date_as_text`) than the manual vocabulary; skipping the translation
  silently drops A from 0.8930 to 0.8560 and looks like a real regression.
- **One-Class SVM fallback was removed** (was worse than rules, disabled, kept alive only by
  its own tests) — don't resurrect it as a shortcut; fix the rule engine instead.

## Configuration

- `SKIP_BERT = True` in notebook 02 → synthetic embeddings (avoids ~470MB MiniLM download).
  Current pickle was built with real embeddings.
- Classification weights (notebooks): α=0.15, β=0.35, γ=0.45, δ=0.05
- Best clustering (re-validated, issue #1): α=0.00, β=0.35, γ=0.45, δ=0.20 + PCA 20D + HDBSCAN → ARI 0.5815
- Rule Engine (aligned GT, circular): accuracy 0.9877 / F1-macro 0.9057 (243 columns);
  `impossible_data` 0.00 — verified: the extractor is correct; the state
  `is_foreign_key and not fk_references_column` only exists in the synthetic benchmark
  (the deleted schema_generator set FK without reference); real Oracle has zero R constraints and parent
  keys have duplicates/nulls so FKs cannot even be created (issue #6, documented limitation)
- Rule Engine (manual GT, honest): accuracy 0.8930 / F1-macro 0.8467 — polymorphic 1.00/1.00
  (issue #7 fix: `_O_` + giant_table→polymorphic merge exemption; FLAG_* stays
  inconsistent_naming by design) + self_contradictory 1.00/1.00 y wrong_data_types 0.96/0.96
  (CLOB fix: rules 1-3 gate incluye CLOB; BLOB/LONG excluidos) + impossible_data 0.00
  (documented limitation #6: verified — extractor correct, parent keys have duplicates/nulls so
  FKs cannot be created; REGISTRO_ID indetectable from metadata) + sobredetección de clean

## Gotchas

- Table names case-sensitive in Oracle — always double-quote
- Oracle allows only one LONG column per table (ORA-01754) — catalog uses CLOB/BLOB for the rest
- Oracle rejects exact duplicate indexes (ORA-01408) — redundant-index anti-pattern is a
  composite index duplicating a single-column prefix; the two in the DB (EMPLEADOS,
  ORDENES_COMPRA) were created manually
- `config.yaml` says `tables_count:10` but actual count is 23 — config value is unused

## Reference

- `docs/00_PROJECT_STATUS_REPORT.md` — full status report, phase by phase
- `docs/research_extraction_preprocessing.md` — best-practices research with primary sources
- `docs/rule_engine_and_future_ml_report.md` — rule engine design + ML fallback exploration
- GitHub issues #1-#7 (closed) document each improvement with root-cause analysis

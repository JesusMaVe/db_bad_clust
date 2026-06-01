# AGENTS.md — db_bad_clust (ml_bad_db_trainer)

**Status: Phase 1 (COMPLETE) → Phase 2 (COMPLETE) → Phase 3 (COMPLETE)**

## Setup & run

```bash
source .venv/bin/activate
pip install -r requirements.txt

# Phase 1 — start Oracle, generate 23 bad-schema tables, insert dirty data:
docker compose up -d
cd scripts && python3 orchestrator.py          # FAILS if run from repo root

# Phase 2 — ML pipeline over extracted schemas:
cd scripts && python3 phase2_orchestrator.py --tune
# Tests run from repo root (no DB needed, mocks):
cd .. && python3 -m pytest tests/ -v
```

## Gotchas (agent will miss these)

- **Must `cd scripts`** before running orchestrator — `OracleConnector(config_path="../config.yaml")` is relative to CWD. Running from repo root crashes.
- **Two entrypoints**: `orchestrator.py` (Phase 1: generate Oracle) and `phase2_orchestrator.py` (Phase 2: ML pipeline). Both do `sys.path.insert(0, ...)` at the top.
- **`config.yaml` says `tables_count: 10`** — stale comment. Actual count is 23 hardcoded in `anti_patterns.py`.
- **`random` has no seed** — every run produces different data. Tests use mocks so they're deterministic.
- **Oracle 23c**: `bad_schema` user pre-created via `APP_USER` env var in `docker-compose.yml`. Healthcheck takes ~30s.
- **Table names are case-sensitive** — DDL creates unquoted (`CREATE TABLE EMPLEADOS`), but DROP/verification uses quoted (`DROP TABLE "EMPLEADOS"`). Always double-quote table names.
- **pandoc+pdflatex + Unicode** — box-drawing chars (U+251C etc.) crash. Use only ASCII in `docs/phase2/*.md`.
- **BERT downloads `bert-base-multilingual-cased`** on first run (~1.5GB). Use `--skip-bert` to bypass.

## DDL quirks

- CHECK constraints created `DISABLE` — exist in `user_constraints` but never validate.
- FK to a table without PK fails silently; FK with `DISABLE` creates a disabled constraint.
- Redundant indexes (same column list) produce `ORA-01408` logged as warning — expected.
- Types LONG/LONG RAW removed in Oracle 23c — table DATOS_MAESTROS fails to create entirely (this IS the anti-pattern).

## Code quality

```bash
ruff check scripts/         # 0 errors (config in pyproject.toml)
ruff format scripts/        # max 100 cols, double quotes
python3 -m pytest tests/ -v # 364 tests, no DB required
```

- **Exceptions**: custom hierarchy `scripts/exceptions.py` — `BadDBError` base → `DatabaseError`, `SchemaError`, `EmbeddingError`, `ClusteringError`, `ConfigError`, `VisualizationError`, `GenerationError`. No bare `except Exception`.
- **Type hints**: 100% across all scripts.
- **Developer manual**: `DEVELOPER.md` (type conventions, exception rules, tuning guide).

## Phase 3 CLI (Phase 2 + Phase 3)

```
--skip-bert       Use synthetic embeddings (skip BERT download)
--method          pca | umap | svd | tsne
--n-components    Target dims (default: auto, 80% variance)
--n-clusters      Cluster count (default: auto via silhouette)
--cluster-method  kmeans | dbscan | agglomerative | meanshift
--metric          euclidean | cosine (DBSCAN, default euclidean)
--dbscan-eps      DBSCAN eps value (default 0.5, use 'auto' for k-distance)
--dbscan-min-pts  DBSCAN min_samples (default: auto, 5% of data)
--validate        Validate clustering against anti-pattern ground truth
--alpha           BERT weight (default 0.40)
--beta            Type weight (default 0.30)
--gamma           Constraint weight (default 0.25)
--delta           data_length weight (default 0.05, best=0.00)
--tune            Grid search 360 combos (~2s extra, best known: 0.35/0.15/0.50/0.00)
--output-dir      Path for plots/reports (default: ../output/run_<ts>)
```

## Oracle diagnostics

```bash
docker compose exec oracle sqlplus bad_schema/bad_schema_pass@FREEPDB1
```

```sql
SELECT table_name FROM user_tables ORDER BY table_name;
DESC "EMPLEADOS";
SELECT * FROM "EMPLEADOS" WHERE ROWNUM <= 3;
-- Constraints (includes DISABLE, FK, CHECK)
SELECT constraint_name, constraint_type, table_name, status, search_condition
  FROM user_constraints ORDER BY table_name, constraint_type;
-- Indexes (includes redundant)
SELECT index_name, table_name, column_name, column_position
  FROM user_ind_columns ORDER BY table_name, index_name, column_position;
-- FK metadata
SELECT constraint_name, table_name, r_constraint_name
  FROM user_constraints WHERE constraint_type = 'R' ORDER BY table_name;
-- Column types & nullability
SELECT table_name, column_name, data_type, data_length, nullable
  FROM user_tab_columns ORDER BY table_name, column_id;
```

## Reference

- `docs/phase2/00_anti_patterns_catalog.md` — anti-pattern catalog
- `docs/phase2/11_completion_report.md` — Phase 2 results & findings
- `DEVELOPER.md` — code standards, testing, tuning

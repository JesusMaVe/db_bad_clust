# AGENTS.md — db_bad_clust (ml_bad_db_trainer)

## Quick start

```bash
# One-shot (start Oracle, wait, generate):
./run_all.sh

# Step by step:
docker compose up -d
# Wait ~30s for healthcheck
cd scripts && python orchestrator.py
```

## Must know

- **Run everything from `scripts/`** — `OracleConnector(config_path="../config.yaml")` path is relative to CWD. Running from repo root will fail.
- **Need virtualenv** — macOS blocks system pip (`externally-managed-environment`). Use `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- **No tests, lint, typecheck, CI** — none exist.
- **Oracle 23c via Docker** — single service, `bad_schema` user pre-created via `APP_USER` env var. Healthcheck-ready in ~30s.
- **Case-sensitive table names** — DDL creates tables unquoted (`CREATE TABLE EMPLEADOS`), but DROP/verification uses quoted (`DROP TABLE "EMPLEADOS"`, `SELECT * FROM "EMPLEADOS"`). Always use double-quotes for table names to be safe.
- **`random` module without seed** — every run produces different data.
- **Code is Spanish-mixed** — identifiers, comments, docs in Spanish/English mix.
- **`config.yaml` says `tables_count: 10`** — actual count is 13 (not read by code, just a comment).

## Architecture

```
scripts/
  orchestrator.py       # Phase 1 entry — generates 13 tables, inserts, verifies
  anti_patterns.py      # Dataclasses + generators for bad schemas & dirty data
  ddl_generator.py      # CREATE/DROP TABLE, FK, CHECK (DISABLE), indexes
  dml_generator.py      # INSERT with literal SQL values (not bind), dirty data hazard
  db_connector.py       # OracleConnector — reads config.yaml, oracledb thin mode

  # Phase 2 — ML pipeline on extracted schemas
  schema_extractor.py   # Metadata extraction from Oracle dictionary views
  text_preprocessor.py  # Clean names: camelCase split, abbreviation expansion, table context
  bert_embedder.py      # 768-dim embeddings via bert-base-multilingual-cased
  structural_encoder.py # One-hot types (12) + binary constraints (5)
  feature_builder.py    # Weighted composite vector φ = α·BERT + β·type + γ·constraints

sql_init/               # Mounted into container for init SQL (empty)
docs/phase2/            # Docs per Phase 2 module
run_all.sh              # docker compose up + healthcheck wait + orchestrator
```

## Phase 2 pipeline

```
Phase 1 → SchemaExtractor → TextPreprocessor → BERTEmbedder
  → StructuralEncoder → FeatureBuilder → [DimensionalityReducer]
  → [ClusterEngine] → [Evaluator] → [Recommender] → [Visualizer]
```

Modules in `[]` are pending (6–10). Each module has docs in `docs/phase2/`.

## Anti-pattern catalog

Full reference in `docs/phase2/00_anti_patterns_catalog.md`. 13 tables covering:
14 schema-level anti-patterns (no PK, wrong types, FK roto, CHECK DISABLE, redundant indexes, etc.)
~17 data-level anti-patterns (timezone mixing, multi-byte, contradictory data, neg IDs, etc.)

## DDLGenerator quirks

- CHECK constraints are created `DISABLE` — they exist in `user_constraints` but don't validate
- FK to a table without PK will fail silently; FK with `DISABLE` directive creates a disabled constraint
- Redundant indexes (same column list) produce `ORA-01408` logged as warning — expected behavior

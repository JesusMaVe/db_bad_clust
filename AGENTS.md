# AGENTS.md — db_clust (ml_bad_db_trainer)

## Quick start

```bash
docker compose up -d && sleep 30
cd scripts && python orchestrator.py
```

## Must know

- **Run everything from `scripts/`** — `orchestrator.py` hardcodes `../config.yaml` relative to CWD and uses `sys.path.insert(0, ...)` for intra-project imports. Running from the repo root will fail.
- **No `pyproject.toml`** — plain pip. Install with `pip install -r requirements.txt`.
- **No tests, no lint, no typecheck, no CI** — none of these setups exist.
- **Oracle 23c via Docker** — single service in `docker-compose.yml`. The `bad_schema` user is pre-created by `APP_USER` env vars. Container may take ~30s to be healthy.
- **Connection config** — `config.yaml` at repo root. DSN is `localhost:1521/FREEPDB1`.
- **Case-sensitive table names** — all DDL wraps names in double-quotes (`"EMPLEADOS"`). Unquoted queries will fail.
- **`random` module without seed** — runs are not reproducible.
- **Code is Spanish-mixed** — identifiers, comments, and docs are mostly in Spanish.

## Architecture

```
scripts/
  orchestrator.py       # Phase 1 entry point — generates 5 tables, inserts, verifies
  anti_patterns.py      # Data classes + generators for bad schemas
  ddl_generator.py      # CREATE/DROP TABLE with anti-patterns
  dml_generator.py      # INSERT with dirty data
  db_connector.py       # OracleConnector — reads config.yaml, oracledb thin mode

  # Phase 2 — ML pipeline on schemas
  schema_extractor.py   # Metadata extraction from Oracle (tables, columns, constraints)
  text_preprocessor.py  # Name cleaning (camelCase, abbreviations, table context)
  bert_embedder.py      # BERT embeddings via bert-base-multilingual-cased

sql_init/               # Mounted into container for init SQL (currently empty)
config.yaml             # DB creds + anti-pattern generation config
docs/phase2/            # Documentation per Phase 2 module
```

## Phase 2 pipeline

```
Phase 1 → SchemaExtractor → TextPreprocessor → BERTEmbedder
  → StructuralEncoder → FeatureBuilder → DimensionalityReducer
  → ClusterEngine → Evaluator → Recommender → Visualizer
```

Cada módulo de Phase 2 tiene su documentación en `docs/phase2/`. Los módulos 4-10 están pendientes de implementar.

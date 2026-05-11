# AGENTS.md — db_clust (ml_bad_db_trainer)

## Quick start

```bash
docker compose up -d && sleep 30
cd scripts && python orchestrator.py
```

## Must know

- **Run everything from `scripts/`** — `orchestrator.py` hardcodes `../config.yaml` relative to CWD and uses `sys.path.insert(0, ...)` for intra-project imports. Running from the repo root will fail.
- **No `pyproject.toml`** — plain pip. Install with `pip install -r requirements.txt`.
- **Not a git repo** — no `.gitignore` either. `__pycache__/` is tracked.
- **No tests, no lint, no typecheck, no CI** — none of these setups exist.
- **Oracle 23c via Docker** — single service in `docker-compose.yml`. The `bad_schema` user is pre-created by `APP_USER` env vars. Container may take ~30s to be healthy.
- **Connection config** — `config.yaml` at repo root. DSN is `localhost:1521/FREEPDB1`.
- **Case-sensitive table names** — all DDL wraps names in double-quotes (`"EMPLEADOS"`). Unquoted queries will fail.
- **`python-dotenv`, `sqlalchemy`, `pandas` in `requirements.txt` but never imported** — only `oracledb` and `pyyaml` are actually used.
- **`random` module without seed** — runs are not reproducible.
- **Code is Spanish-mixed** — identifiers, comments, and `clustering_db.md` are mostly in Spanish.

## Architecture

```
scripts/
  orchestrator.py     # Entry point — connects, generates 5 tables, inserts, verifies
  anti_patterns.py    # Data classes + table/data generators for intentionally bad schemas
  ddl_generator.py    # CREATE/DROP TABLE with anti-patterns (wrong types, no PKs, cryptic names)
  dml_generator.py    # INSERT with dirty data (HTML in text, inconsistent dates, nulls, dupes)
  db_connector.py     # OracleConnector — reads config.yaml, uses oracledb in thin mode
sql_init/             # Mounted into container for init SQL (currently empty)
config.yaml           # DB creds + anti-pattern generation config (10 tables, 5-20 rows each)
clustering_db.md      # ML pipeline design doc — BERT → PCA/UMAP → clustering (Phase 2)
```

## Two-phase pipeline

```
Phase 1 (implemented)         Phase 2 (design doc, TODO)
 ┌────────────────────┐        ┌──────────────────────────┐
 │ Generate bad DB    │ ─────> │ Extract metadata → BERT  │
 │ (known anti-       │        │ embeddings → clustering  │
 │  patterns = ground │        │ (KMeans/DBSCAN/MeanShift)│
 │  truth)            │        │ → recommendations        │
 └────────────────────┘        └──────────────────────────┘
                                   ↑ evaluate against ↑
                                   ground truth from Phase 1
```

`clustering_db.md` describes Phase 2 in detail (BERT embeddings, DBSCAN, K-Means, Mean Shift, UMAP, evaluation via Silhouette/Davies-Bouldin/Calinski-Harabasz). **None of Phase 2 is implemented.** The current code is only Phase 1 — generating a DB with known anti-patterns so the future ML pipeline can be validated against ground truth.

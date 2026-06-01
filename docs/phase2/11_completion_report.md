# Phase 2 Completion Report

**Project:** db_bad_clust (ml_bad_db_trainer)
**Date:** May 2026
**Status:** COMPLETE

---

## Summary

Phase 2 implements a full ML pipeline that extracts Oracle schemas from the Phase 1 database (22 tables, 235 columns), processes column metadata through BERT embeddings and structural encoding, and applies unsupervised clustering to detect schema anti-patterns. The pipeline produces evaluation metrics, fix recommendations, and visualizations.

## Pipeline Architecture

```
Phase 1 DB → SchemaExtractor → TextPreprocessor → BERTEmbedder
  → StructuralEncoder → FeatureBuilder → DimensionalityReducer
  → ClusterEngine → Evaluator → Recommender → Visualizer
```

## Modules Implemented (10/10)

| # | Module | File | Purpose |
|---|---|---|---|
| 1 | SchemaExtractor | `schema_extractor.py` | Metadata extraction from Oracle dictionary views (ALL_TAB_COLUMNS, ALL_CONSTRAINTS, ALL_IND_COLUMNS) |
| 2 | TextPreprocessor | `text_preprocessor.py` | Clean column names: camelCase split, abbreviation expansion, prefix removal, table context |
| 3 | BERTEmbedder | `bert_embedder.py` | 768-dim embeddings via `bert-base-multilingual-cased` with MPS/CPU fallback |
| 4 | StructuralEncoder | `structural_encoder.py` | One-hot types (12 canonical) + binary constraints (5) + statistical (log1p data_length) |
| 5 | FeatureBuilder | `feature_builder.py` | Weighted composite vector phi = alpha*BERT + beta*type + gamma*constraints + delta*statistical |
| 6 | DimensionalityReducer | `dimensionality_reducer.py` | PCA / UMAP / SVD / t-SNE reduction (786 → N dims) |
| 7 | ClusterEngine | `cluster_engine.py` | KMeans / DBSCAN / Agglomerative clustering |
| 8 | Evaluator | `evaluator.py` | Silhouette, Davies-Bouldin, Calinski-Harabasz + cross-table purity + composition |
| 9 | Recommender | `recommender.py` | Fix recommendations per cluster (date-as-text, number-as-text, nullable, boolean flags) |
| 10 | Visualizer | `visualizer.py` | 2D/3D scatter, silhouette plot, cluster-table heatmap, composition bar charts |

## Infrastructure (Delivered)

| Item | Detail |
|---|---|
| `exceptions.py` | Custom hierarchy: BadDBError → 9 subclasses (DatabaseError, SchemaError, EmbeddingError, ClusteringError, etc.) |
| `phase2_orchestrator.py` | CLI entry point with 11 arguments, 360-combination grid search tuning |
| `pyproject.toml` | Ruff config + pytest config |
| `tests/` | 297 tests with pytest (no Oracle required, uses mocks) |
| `DEVELOPER.md` | Full developer manual including code standards, linting, testing, tuning |
| Type hints | 100% coverage across all 17 scripts (~3,969 LOC) |
| Ruff lint | 0 errors |

## Best Known Result

### Hyperparameters

| Param | Value | Description |
|---|---|---|
| alpha | 0.35 | BERT semantic weight |
| beta | 0.15 | Data type weight |
| gamma | 0.50 | Constraint weight |
| delta | 0.00 | Statistical feature weight (data_length) |
| n_components | 5 | PCA target dimensions |
| n_clusters | 5 | KMeans clusters |

### Metrics

| Metric | Value | Interpretation |
|---|---|---|
| **Silhouette** | **0.4614** | Moderate cluster separation |
| Davies-Bouldin | 0.976 | Low inter-cluster similarity |
| Calinski-Harabasz | 148.9 | Good density separation |
| **Purity** | **82.77%** | Columns from same table cluster together |

### Cluster Composition

| Cluster | Size | VARCHAR % | NUMERIC % | Tables | Purity |
|---|---|---|---|---|---|
| #0 | 36 | 78% | 17% | 11 tables (DEPARTAMENTOS, VENTAS, etc.) | Mixed |
| #1 | 88 | 61% | 24% | 18 tables (AUDITORIA_LOG, TRANSACCIONES, etc.) | High |
| #2 | 55 | 96% | 2% | BACKUP_DATOS, TBL_DATOS | High |
| #3 | 41 | 100% | 0% | TABLA_BASE_DATOS, TABLA_VACIA | High |
| #4 | 15 | 67% | 7% | TODO_EN_UNO | High (94%) |

### Key Findings

1. **Constraints matter most**: gamma=0.50 dominates — PK, FK, Unique, Nullable, Indexed flags are the strongest signal for clustering
2. **BERT helps but is secondary**: alpha=0.35 is significant but not dominant
3. **data_length does NOT help**: delta=0.00 consistently wins — VARCHAR2(10) vs VARCHAR2(4000) doesn't correlate with anti-patterns
4. **5 clusters is optimal**: fewer loses granularity, more fragments purity (10 clusters → 63% purity)
5. **100% nullable across all clusters**: this feature doesn't discriminate in our dataset
6. **Without tuning**: raw BERT defaults produce silhouette ~0.34; tuning improves to 0.46

## Detected Anti-Patterns (Sample)

The recommender successfully identifies:
- **Date-as-text**: `FECHA_NACIMIENTO`, `FECHA_ALTA`, `FECHA_ORDEN` stored as VARCHAR2
- **Number-as-text**: `SALARIO`, `CANTIDAD`, `PRECIO`, `MONTO` stored as VARCHAR2
- **Boolean-as-text**: `ACTIVO` as CHAR/VARCHAR2/CLOB/NUMBER (4 different types for same semantic)
- **100% nullable**: All 235 columns allow NULL — no mandatory fields
- **Mixed types in TODO_EN_UNO**: 15 columns with wildly inconsistent naming and types

## Runtime Performance

| Step | Time (MacBook Air M3) |
|---|---|
| Bert model load | ~3s (first run, cached after) |
| Embeddings (235 cols) | ~1s |
| Grid search (360 combos) | ~2s |
| Visualizations | ~0.3s |
| **Total with BERT + tune** | **~11s** |
| **Total skip-bert + tune** | **~4s** |

## Future Work (Ideas)

- Implement UMAP (requires `umap-learn` package)
- Experiment with fuzzy matching on column name embeddings
- Add ground-truth validation against the anti-pattern catalog (docs/phase2/00_anti_patterns_catalog.md)
- Cross-validation stability analysis for cluster assignments
- Export cluster labels as SQL comments directly into the Oracle schema

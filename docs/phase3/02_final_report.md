# Phase 3 Final Report — Hypothesis Validation

**Project:** db_bad_clust (ml_bad_db_trainer)
**Date:** May 2026
**Status:** COMPLETE

---

## Hypothesis

**H₁ (principal):** BERT embeddings + structural features, clustered with DBSCAN, produces more coherent clusters and detects more anomalies than K-Means or Mean Shift on the same composite feature vector.

**H₀ (nula):** No significant difference in cluster quality between algorithms.

---

## Experimental Setup

| Component | Value |
|---|---|
| Database | Oracle 23c, 22 tables, 235 columns |
| Semantic embeddings | BERT `bert-base-multilingual-cased` (768D) |
| Structural encoding | One-hot types (12) + constraints (5) + data_length (1) |
| Composite vector | φ = α·BERT ⊕ β·type ⊕ γ·constraints ⊕ δ·data_length |
| Reduction | UMAP (5D best), PCA (5-20D baseline) |
| Ground truth | Anti-pattern catalog: 235 columns, 7 categories |
| Evaluation | ARI (external), NMI (external), Silhouette (internal), Purity (table) |

---

## Results: Algorithm Comparison (UMAP 5D, best weights)

| Algorithm | ARI | NMI | Silhouette | Clusters | Pureza | Noise |
|---|---|---|---|---|---|---|
| **KMeans** (γ=0.50, k=5) | **0.4085** | 0.6109 | 0.5783 | 5 | 95.32% | 0 |
| **DBSCAN** (auto-eps) | 0.3636 | **0.6328** | 0.8000 | 6 | 100% | 103 |
| **HDBSCAN** (default) | 0.3064 | 0.5696 | 0.6998 | 4 | 100% | 0 |
| PCA+KMeans (tuned) | 0.3517 | 0.5066 | 0.4614 | 5 | 82.77% | 0 |
| PCA+DBSCAN (eps=4) | 0.2475 | 0.4770 | 0.6184* | 4 | — | 141 |
| **MeanShift** (auto) | **−0.0037** | 0.3222 | — | 3 | — | 0 |

**Conclusion:** H₁ is **partially refuted** — UMAP+KMeans (ARI=0.4085) outperforms UMAP+DBSCAN (ARI=0.3636) on this dataset. However, DBSCAN achieves higher NMI (0.6328 vs 0.6109), indicating better information-theoretic agreement with ground truth. MeanShift fails entirely (ARI ≈ 0).

---

## Best Configuration

### Hyperparameters

| Param | Value | Description |
|---|---|---|
| α (alpha) | 0.35 | BERT semantic weight |
| β (beta) | 0.15 | Data type weight |
| γ (gamma) | 0.50 | **Constraint weight (dominant)** |
| δ (delta) | 0.00 | Statistical feature (data_length — negligible) |
| Reduction | UMAP | Non-linear, 5 dimensions |
| Method | KMeans | k=5 (auto via silhouette) |

### Cluster Composition (Best Run)

| Cluster | Size | Tables | VARCHAR% | NUMERIC% | Priority |
|---|---|---|---|---|---|
| #0 | 47 | EMPLEADOS, EMPLEADOS_HISTORIAL, ORDENES_COMPRA, VENTAS, USUARIOS_WEB, etc. | 74% | 21% | ALTA |
| #1 | 59 | BACKUP_DATOS, TBL_DATOS | 95% | 2% | MEDIA |
| #2 | 73 | AUDITORIA_LOG, CLIENTES_DIRECCIONES, PRODUCTOS, PROVEEDORES, etc. | 60% | 23% | ALTA |
| #3 | 40 | TABLA_BASE_DATOS | 100% | 0% | MEDIA |
| #4 | 16 | TODO_EN_UNO | 69% | 6% | ALTA |

### Key Insights

1. **γ (constraints) is the dominant signal** — γ=0.50 consistently wins. PK, FK, Unique, Nullable, Index flags are the strongest discriminators.
2. **δ (data_length) is irrelevant** — δ=0.00 wins every grid search. VARCHAR2(10) vs VARCHAR2(4000) does not correlate with anti-patterns.
3. **α (BERT) is secondary but important** — dropping BERT entirely (α=0) drops ARI to ~0.25. Best α is 0.35.
4. **UMAP beats PCA** — UMAP 5D (ARI=0.4085) vs PCA 5D (ARI=0.3517). Δ=0.057 confirms H₁_d.
5. **KMeans beats DBSCAN on ARI** — H₁_a is refuted (ΔARI = −0.045 in favor of KMeans). DBSCAN produces too many noise points (103/235).
6. **Anomaly detection works for KMeans** — centroid-distance method identifies 12 structural anomalies (5.1%).
7. **MeanShift is unusable** — ARI ≈ −0.0037. KDE fails in reduced dimensionality.

---

## Ground Truth Validation

The `scripts/ground_truth.py` module auto-generates column→anti-pattern labels from `anti_patterns.py` definitions. After fixing a case-sensitivity bug (40 columns previously excluded), all 235 columns are validated.

### Per-Algorithm Validation

| Algorithm | ARI | NMI | Structural Anomalies |
|---|---|---|---|
| UMAP+KMeans | **0.4085** | 0.6109 | 12 (centroid-based) |
| UMAP+DBSCAN | 0.3636 | **0.6328** | 103 noise (DBSCAN native) |
| PCA+KMeans | 0.3517 | 0.5066 | 12 (centroid-based) |
| UMAP+HDBSCAN | 0.3064 | 0.5696 | 0 |
| PCA+DBSCAN | 0.2475 | 0.4770 | 141 noise |
| MeanShift | −0.0037 | 0.3222 | 0 |

---

## Detected Anti-Patterns

### Date-as-Text
11 columns across 8 tables stored dates as VARCHAR2 (e.g., `FECHA_NACIMIENTO`, `FECHA_ALTA`, `FECHA_ORDEN`, `FECHA_VENTA`).

### Number-as-Text
9 columns store numeric data as VARCHAR2 (e.g., `SALARIO`, `CANTIDAD`, `PRECIO`, `MONTO`, `PRECIO_TOTAL`).

### Boolean-as-Text
4 distinct types for the same semantic field:
- `ACTIVO` as CHAR(1) and VARCHAR2 (different tables)
- `ES_ACTIVO` as CLOB
- `ESTADO` as NUMBER
- `FLAG_S_N`, `FLAG_Y_N`, `FLAG_1_0`, `FLAG_T_F` in TODO_EN_UNO

### 100% Nullable
All 235 columns across 22 tables allow NULL — no mandatory fields anywhere.

### TODO_EN_UNO (Mixed Domain)
16 columns with wildly inconsistent naming (`FECHA_O_DIRECCION`, `CANT_O_PRECIO`, `ESTADO_O_ACTIVO`) and 4 different boolean flag formats.

---

## Pipeline Performance

| Step | Time (MacBook Air M3) |
|---|---|
| BERT model load | ~3s (cached) |
| Embeddings (235 cols) | ~1s |
| Grid search (360 combos) | ~2s |
| UMAP reduction | ~8s |
| Visualizations | ~0.5s |
| **Total end-to-end** | **~12s** |

---

## Hypotheses Evaluation Summary

| ID | Statement | Metric | Result | Verdict |
|---|---|---|---|---|
| H₁_a | DBSCAN ARI > KMeans ARI | ARI | KMeans 0.4085 > DBSCAN 0.3636 | **REFUTED** |
| H₁_b | DBSCAN detects anomalies as noise | F1 | 0.00% (no known anti-pattern = noise) | INCONCLUSIVE |
| H₁_c | Cosine > Euclidean for DBSCAN | ARI | Both tested | NOT SIGNIFICANT |
| H₁_d | UMAP > PCA for all algorithms | ΔARI | UMAP 0.4085 > PCA 0.3517 | **CONFIRMED** |

---

## Key Deliverables

| Artifact | Location |
|---|---|
| Pipeline source | `scripts/phase2_orchestrator.py` |
| Cluster engine (5 methods) | `scripts/cluster_engine.py` |
| Ground truth validation | `scripts/ground_truth.py` |
| Anomaly detection | `scripts/evaluator.py` (centroid-based) |
| Anti-pattern catalog | `docs/phase2/00_anti_patterns_catalog.md` |
| Phase 2 completion | `docs/phase2/11_completion_report.md` |
| Test suite (364 tests) | `tests/` |
| Web dashboard | `dashboard/index.html` |
| Best run output | `output/run_20260531_191027_bert/` |

---

## Future Work

- Sentence-BERT (SBERT) for better semantic embeddings
- Graph Neural Networks over FK relationships
- Cross-validation stability analysis
- Auto-fix SQL generation from recommendations
- Multi-SGBD support via SQLAlchemy

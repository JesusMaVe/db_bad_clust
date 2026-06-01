# Phase 3 Design — Validación de Hipótesis H₁

**Date:** 2026-05-31
**Status:** Draft

## Objective

Validate the principal hypothesis from CONTEXT.md: **BERT + DBSCAN (with cosine metric + UMAP reduction) produces more coherent clusters and detects more anomalies than K-Means or Mean Shift on the same composite feature vector.**

## Scope

- **In scope:** DBSCAN, Mean Shift, UMAP, k-distance graph ε estimation, ground truth validation, cosine metric.
- **Not in scope:** Multi-SGBD (SQLAlchemy), Sentence-BERT, advanced statistical features (cardinality, entropy), GNN, NoSQL.
- **Constraint:** Extend existing Phase 2 modules. Do not break Phase 2.

---

## 1. Module Changes

### 1.1 `cluster_engine.py` — Add DBSCAN + Mean Shift

Add two new methods to the existing `ClusterEngine`:

**DBSCAN:**
- sklearn `DBSCAN(metric=metric, eps=eps, min_samples=min_samples)`
- metric: `"euclidean"` or `"cosine"` (new parameter)
- eps: `"auto"` (k-distance graph) or explicit `float`
- min_samples: new parameter, default 5
- Noise detection: points with label `-1` are exposed as `n_noise` attribute

**Mean Shift:**
- sklearn `MeanShift(bandwidth=None)` — auto bandwidth via `estimate_bandwidth()`
- cluster_all=True (no noise label)

**k-distance graph (new function):**

```python
def estimate_epsilon(X: np.ndarray, k: int = 5) -> float:
    """
    Compute k-distance graph and return epsilon at the elbow point.
    Uses the point of maximum curvature (kneedle algorithm).

    Implementation:
    1. Compute pairwise distances
    2. For each point, find distance to k-th nearest neighbor
    3. Sort distances ascending
    4. Find elbow = point with max perpendicular distance from line
       connecting first and last point
    """
```

### 1.2 `dimensionality_reducer.py` — Add UMAP

Add method `"umap"` to existing `DimensionalityReducer`:

- Import `umap.UMAP` with `try/except ImportError`
- If not installed, raise `ClusteringError` with install instructions
- Parameters: `n_neighbors=15`, `min_dist=0.1`, random seed for reproducibility
- UMAP has no `.transform()` — always `fit_transform()`

### 1.3 `evaluator.py` — Add ground truth validation

New method:

```python
def validate_against_ground_truth(
    labels: np.ndarray,
    ground_truth: list[str | None],
) -> dict[str, float]:
    """
    Compute external validation metrics:
    - Adjusted Rand Index (ARI): cluster agreement adjusted for chance
    - Normalized Mutual Information (NMI): information-theoretic agreement
    - Anomaly detection: precision, recall, F1 for noise points (-1)
      vs known anti-pattern columns
    """
```

### 1.4 `ground_truth.py` — New module

Reads from `scripts/anti_patterns.py` to generate column→anti-pattern mapping.

```python
# Auto-generated from anti_patterns.py table definitions
# Each column maps to its known anti-pattern type or None
GROUND_TRUTH: dict[str, str | None] = {
    "EMPLEADOS.FECHA_NACIMIENTO": "date_as_text",
    "EMPLEADOS.SALARIO": "number_as_text",
    "BACKUP_DATOS.ID": None,  # normal column
    ...
}

def get_ground_truth(column_table_map: list[str]) -> list[str | None]:
    """Return ground truth labels for each column in order."""
```

The mapping is derived from the metadata in `anti_patterns.py` — specifically from the `POORLY_DESIGNED_TABLES` dataclass instances which define anti-patterns per column.

### 1.5 `phase2_orchestrator.py` — CLI changes

New flags:

```
--cluster-method    kmeans | dbscan | agglomerative | meanshift
--metric            euclidean | cosine
--dbscan-eps        auto | <float>
--dbscan-min-pts    <int>
--umap-neighbors    <int>     (default: 15)
--validate          Use ground truth validation against anti-pattern catalog
```

Tuning grid expansion (when `--tune`):
- Add DBSCAN with auto-eps to grid (separate from KMeans grid)
- Evaluate both methods and report best overall

**Note:** Tuning currently assumes KMeans. Phase 3 extends `tune_params()` to optionally tune DBSCAN parameters (eps, min_samples) and compare both algorithms.

---

## 2. Ground Truth Mapping

### Source

`scripts/anti_patterns.py` defines 22 table generators via `POORLY_DESIGNED_TABLES`. Each generator tags columns with anti-pattern categories:

| Category | Columns example | Expected cluster behavior |
|---|---|---|
| `date_as_text` | FECHA_NACIMIENTO, FECHA_ALTA | Should form a sub-cluster |
| `number_as_text` | SALARIO, CANTIDAD, PRECIO | Should form a sub-cluster |
| `boolean_as_text` | ACTIVO, ES_ACTIVO, FLAG_S_N | Should form a sub-cluster |
| `no_primary_key` | All columns in TABLA_VACIA | Table-level anti-pattern |
| `reserved_word` | FROM, TABLE, GROUP (in TABLA_BASE_DATOS) | Should cluster together |
| `normal` | Standard PK/FK columns | Reference baseline |
| `nullable_all` | All columns (universal) | Not discriminative |

### Validation Protocol

1. After clustering, compute:
   - **ARI** (Adjusted Rand Index): agreement between cluster labels and anti-pattern categories. Range [-1, 1], higher = better.
   - **NMI** (Normalized Mutual Information): shared information. Range [0, 1], higher = better.
   - **Anomaly F1**: If using DBSCAN, compare noise points (-1) against known anomalous columns.

2. Compare across algorithms:
   - DBSCAN (cosine + auto-eps) vs K-Means (best k) vs Mean Shift vs Agglomerative

---

## 3. Hypotheses Evaluation

| ID | Statement | Metric | Success criterion |
|---|---|---|---|
| H₁ₐ | DBSCAN achieves higher ARI than K-Means on same phi | ARI | ΔARI > 0.05 |
| H₁_b | DBSCAN detects known anomalous columns as noise | F1 anomaly | F1 > 0.6 |
| H₁_c | Cosine metric outperforms euclidean for DBSCAN | ARI, Silhouette | ΔARI > 0.02 |
| H₁_d | UMAP improves cluster quality over PCA for all algorithms | ARI, Silhouette | ΔARI > 0.03 |

---

## 4. Dependencies

Add to `requirements.txt`:

```
umap-learn>=0.5.0
```

Optional, with graceful fallback (same pattern as `torch` in `bert_embedder.py`).

---

## 5. Testing

### Existing tests: must remain passing
- 297 tests unchanged

### New tests to add

| File | Tests |
|---|---|
| `tests/test_cluster_engine.py` | DBSCAN with cosine, DBSCAN with auto-eps, Mean Shift, noise detection |
| `tests/test_dimensionality_reducer.py` | UMAP (if available), graceful ImportError fallback |
| `tests/test_evaluator.py` | `validate_against_ground_truth()` with known labels |
| `tests/test_ground_truth.py` | Mapping correctness, column coverage |

---

## 6. Acceptance Criteria

1. `--cluster-method dbscan --metric cosine --validate` runs end-to-end against Oracle DB
2. Evaluation report includes ARI, NMI, anomaly F1
3. `--cluster-method meanshift` works (auto-bandwidth)
4. `--method umap` reduces dimensionality
5. DBSCAN auto-eps produces reasonable epsilon (validated via k-distance plot)
6. All 297 existing tests + new tests pass
7. Ruff lint: 0 errors

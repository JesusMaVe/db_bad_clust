# Phase 2 Implementation Plan — Modules 6-10

> **Goal:** Complete the ML pipeline from feature vectors to cluster analysis, recommendations, and visualization.

**Architecture:** Each module is a standalone class in `scripts/`. The `phase2_orchestrator.py` ties them together, reading from Oracle via SchemaExtractor and producing cluster visualizations + recommendations.

**Tech Stack:** numpy, scikit-learn, matplotlib, seaborn, umap-learn (optional), pandas

---

## File Structure

| File | Responsibility |
|------|---------------|
| `scripts/dimensionality_reducer.py` | PCA / UMAP / t-SNE reduction of phi vectors |
| `scripts/cluster_engine.py` | KMeans / DBSCAN / Agglomerative clustering |
| `scripts/evaluator.py` | Silhouette, D-B, Calinski-Harabasz, cross-table analysis |
| `scripts/recommender.py` | Generate fix recommendations per cluster |
| `scripts/visualizer.py` | 2D/3D scatter plots, heatmaps, bar charts |
| `scripts/phase2_orchestrator.py` | End-to-end pipeline entry point |

## Data Flow

```
Oracle DB → SchemaExtractor → TextPreprocessor → BERTEmbedder
  → StructuralEncoder → FeatureBuilder
  → DimensionalityReducer → ClusterEngine → Evaluator
  → Recommender → Visualizer
```

## Modules

### 6. DimensionalityReducer
- Methods: PCA (default), UMAP (optional), TruncatedSVD
- Configurable n_components (default 10)
- fit / transform / fit_transform API
- Returns reduced matrix + explained variance ratio (PCA)

### 7. ClusterEngine
- Methods: KMeans (default), DBSCAN, Agglomerative
- Configurable n_clusters / eps / linkage
- fit_predict returns labels
- Exposes cluster_centers_, labels_, inertia_ (KMeans)

### 8. Evaluator
- silhouette_score, davies_bouldin_score, calinski_harabasz_score
- cross_table_analysis: Do columns from same table cluster together?
- cluster_composition: What anti-patterns appear in each cluster?
- print_report: Text summary of all metrics

### 9. Recommender
- For each cluster, analyze column characteristics (type, constraints)
- Generate natural-language fix recommendations
- Prioritize recommendations by severity/impact
- Returns structured list of {cluster_id, columns, issues, fixes}

### 10. Visualizer
- plot_clusters_2d: Scatter plot with colored clusters
- plot_silhouette: Silhouette diagram
- plot_heatmap: Cluster × table cross-tabulation
- plot_composition: Bar chart of anti-patterns per cluster
- save_all: Generate all plots to output directory

## Orchestrator
- CLI with argparse
- Options: --skip-bert (use random embeddings), --method, --n-clusters, --output-dir
- Connects to Oracle, runs full pipeline, saves plots + report

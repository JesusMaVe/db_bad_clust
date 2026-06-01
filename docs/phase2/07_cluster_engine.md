# Modulo 7: Cluster Engine

## Proposito

Agrupa columnas de base de datos en clusters basados en sus vectores compuestos phi reducidos. Los clusters revelan patrones y anti-patrones de diseno de esquemas.

## Ubicacion

`scripts/cluster_engine.py`

## Metodos Soportados

| Metodo | Descripcion | Parametros Clave |
|--------|-------------|-------------------|
| KMeans | Particional, basado en centroides (default) | n_clusters |
| DBSCAN | Basado en densidad, detecta outliers | eps, min_samples |
| Agglomerative | Jerarquico aglomerativo | n_clusters, linkage |

## API

```python
from cluster_engine import ClusterEngine

engine = ClusterEngine(
    method="kmeans",       # "kmeans" | "dbscan" | "agglomerative"
    n_clusters=5,          # ignorado por DBSCAN
    random_state=42,
    # args especificos:
    # eps=0.5, min_samples=5     (DBSCAN)
    # linkage="ward"              (Agglomerative)
)

labels = engine.fit_predict(X_reduced)  # -> ndarray (N,)

# Despues del ajuste:
engine.labels_        # mismas etiquetas
engine.cluster_info_  # dict con sizes, centroides, n_noise
```

## Output

- Labels: -1 = outlier/ruido (DBSCAN), 0..k-1 = clusters
- cluster_info_: `{n_clusters, n_noise, cluster_sizes, centroids}`

# Modulo 8: Evaluator

## Proposito

Evaluar la calidad del clustering usando metricas internas (silhouette, Davies-Bouldin, Calinski-Harabasz) y analisis cruzado contra tablas y anti-patrones conocidos.

## Ubicacion

`scripts/evaluator.py`

## Metricas

| Metrica | Rango | Mejor |
|---------|-------|-------|
| Silhouette | [-1, 1] | Mayor (cohesion + separacion) |
| Davies-Bouldin | [0, +inf) | Menor (similaridad entre clusters) |
| Calinski-Harabasz | [0, +inf) | Mayor (ratio varianza inter/intra) |

## API

```python
from evaluator import Evaluator

evaluator = Evaluator()

# Metricas internas
metrics = evaluator.evaluate(X_reduced, labels)
# -> {silhouette, davies_bouldin, calinski_harabasz, n_clusters, n_samples, n_noise}

# Analisis por tabla
cross_table = evaluator.cross_table_analysis(column_table_map, labels)
# -> {table_distribution, table_purity, average_purity}

# Composicion por cluster
composition = evaluator.cluster_composition(labels, column_metadata_list)
# -> {cluster_id: {total_columns, pct_nullable, pct_varchar, ...}}

# Reporte formateado
print(evaluator.print_report(metrics, cross_table, composition))
```

## Cross-Table Analysis

Mide que tan "puras" son las tablas dentro de los clusters. Si columnas de una misma tabla caen en clusters distintos, sugiere que la tabla mezcla anti-patrones.

Pureza = (columnas en cluster mayoritario) / (total columnas de la tabla)

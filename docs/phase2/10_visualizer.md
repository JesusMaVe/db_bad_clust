# Modulo 10: Visualizer

## Proposito

Generar graficos para visualizar los clusters de columnas, su distribucion entre tablas, y la composicion de anti-patrones.

## Ubicacion

`scripts/visualizer.py`

## Graficos Generados

| Grafico | Archivo | Descripcion |
|---------|---------|-------------|
| Scatter 2D | `clusters_2d.png` | Dispersion 2D coloreado por cluster |
| Scatter 3D | `clusters_3d.png` | Dispersion 3D (si dims >= 3) |
| Silhouette | `silhouette.png` | Diagrama silhouette por muestra |
| Heatmap | `cluster_table_heatmap.png` | Distribucion cluster x tabla |
| Composicion | `cluster_composition.png` | Tipos de dato por cluster |

## API

```python
from visualizer import Visualizer

viz = Visualizer(output_dir="../output")

# Individual
viz.plot_clusters_2d(X_reduced, labels, column_names)
viz.plot_clusters_3d(X_reduced, labels)
viz.plot_silhouette(X_reduced, labels)
viz.plot_heatmap(column_table_map, labels)
viz.plot_composition(labels, column_metadata_list)

# Todo en uno
files = viz.save_all(X_reduced, labels, column_table_map, column_metadata, column_names)
# -> {nombre_grafico: ruta_archivo}
```

## Requisitos

- matplotlib
- seaborn (importado via pandas en heatmap)

# Modulo 6: Dimensionality Reducer

## Proposito

Reduce el vector compuesto phi(aj) de 785 dimensiones a un espacio de menor dimensionalidad para clustering y visualizacion.

## Ubicacion

`scripts/dimensionality_reducer.py`

## Metodos Soportados

| Metodo | Clase | Uso |
|--------|-------|-----|
| PCA | `sklearn.decomposition.PCA` | Reduccion lineal, deterministica (default) |
| UMAP | `umap.UMAP` | No lineal, preserva estructura global |
| SVD | `sklearn.decomposition.TruncatedSVD` | Para matrices grandes o sparse |
| t-SNE | `sklearn.manifold.TSNE` | Solo visualizacion (no transform) |

## API

```python
from dimensionality_reducer import DimensionalityReducer

reducer = DimensionalityReducer(
    method="pca",        # "pca" | "umap" | "svd" | "tsne"
    n_components=10,     # dimension objetivo
    random_state=42,
)

X_reduced = reducer.fit_transform(phi)   # (N, 785) -> (N, 10)

# Para PCA/SVD:
reducer.explained_variance_ratio  # varianza explicada por componente
reducer.components_                # componentes principales

# Transformar nuevos datos (solo PCA/SVD):
X_new_reduced = reducer.transform(X_new)
```

## Comportamiento

- Si n_components > min(N, D), se ajusta automaticamente
- UMAP requiere `pip install umap-learn`; si no esta, fallback a PCA
- t-SNE no soporta `transform()` solo `fit_transform()`

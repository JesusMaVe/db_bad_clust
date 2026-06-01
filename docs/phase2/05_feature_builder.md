# Modulo 5: Feature Builder

## Proposito

Construye el vector compuesto phi(aj) concatenando embeddings BERT con codificacion estructural, aplicando normalizacion Z-score y ponderacion por importancia.

## Ubicacion

`scripts/feature_builder.py`

## Formula

```
phi(aj) = [alpha  e_texto_norm    beta  e_tipo_norm    gamma  e_rest_norm      e_est_norm]
```

## Pesos por Defecto

| Peso | Componente       | Valor | Justificacion                     |
|------|------------------|-------|-----------------------------------|
| alpha    | BERT (semantica) | 0.60  | Mayor peso  captura significado  |
| beta    | Tipo de dato     | 0.15  | Informativo pero secundario       |
| gamma    | Restricciones    | 0.15  | Contexto estructural              |
|     | Estadisticos     | 0.10  | Opcional (reservado para futuro)  |

## API

```python
from feature_builder import FeatureBuilder

builder = FeatureBuilder(alpha=0.6, beta=0.15, gamma=0.15, delta=0.10)

phi = builder.build(
    e_text,   # (N, 768)   embeddings BERT
    e_type,   # (N, 12)    one-hot tipos
    e_rest,   # (N, 5)     binario constraints
    e_stat=None  # opcional  features estadisticos
)
# -> numpy array shape (N, 785)  = 768 + 12 + 5

builder.weights  # -> {"alpha": 0.6, "beta": 0.15, "gamma": 0.15, "delta": 0.1}
```

## Pipeline Interno

```
e_text  -> Z-score -> alpha x e_text_norm  -
e_type  -> Z-score -> beta x e_type_norm  |-> concatenate -> phi(aj)
e_rest  -> Z-score -> gamma x e_rest_norm  
```

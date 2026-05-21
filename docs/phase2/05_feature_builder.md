# Módulo 5: Feature Builder

## Propósito

Construye el vector compuesto φ(aⱼ) concatenando embeddings BERT con codificación estructural, aplicando normalización Z-score y ponderación por importancia.

## Ubicación

`scripts/feature_builder.py`

## Fórmula

```
φ(aⱼ) = [α · e_texto_norm  ⊕  β · e_tipo_norm  ⊕  γ · e_rest_norm  ⊕  δ · e_est_norm]
```

## Pesos por Defecto

| Peso | Componente       | Valor | Justificación                     |
|------|------------------|-------|-----------------------------------|
| α    | BERT (semántica) | 0.60  | Mayor peso — captura significado  |
| β    | Tipo de dato     | 0.15  | Informativo pero secundario       |
| γ    | Restricciones    | 0.15  | Contexto estructural              |
| δ    | Estadísticos     | 0.10  | Opcional (reservado para futuro)  |

## API

```python
from feature_builder import FeatureBuilder

builder = FeatureBuilder(alpha=0.6, beta=0.15, gamma=0.15, delta=0.10)

phi = builder.build(
    e_text,   # (N, 768)  — embeddings BERT
    e_type,   # (N, 12)   — one-hot tipos
    e_rest,   # (N, 5)    — binario constraints
    e_stat=None  # opcional — features estadísticos
)
# → numpy array shape (N, 785)  = 768 + 12 + 5

builder.weights  # → {"alpha": 0.6, "beta": 0.15, "gamma": 0.15, "delta": 0.1}
```

## Pipeline Interno

```
e_text  → Z-score → α × e_text_norm  ┐
e_type  → Z-score → β × e_type_norm  ├→ concatenate → φ(aⱼ)
e_rest  → Z-score → γ × e_rest_norm  ┘
```

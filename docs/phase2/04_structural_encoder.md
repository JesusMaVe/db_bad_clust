# Módulo 4: Structural Encoder

## Propósito

Convierte los metadatos estructurales de cada columna en vectores numéricos:

1. **One-hot encoding** del tipo de dato canónico (VARCHAR, NUMBER, DATE, etc.)
2. **Codificación binaria** de restricciones (PK, FK, Unique, Nullable, Indexed)

Estos vectores se concatenan con los embeddings BERT en el Feature Builder (Módulo 5).

## Ubicación

`scripts/structural_encoder.py`

## API

```python
from structural_encoder import StructuralEncoder

encoder = StructuralEncoder()

# One-hot de tipos → (N, 12)
e_type = encoder.encode_data_types(columns)

# Binario de constraints → (N, 5)
e_rest = encoder.encode_constraints(columns)

# Todo en uno
encoded = encoder.encode_all(columns)
# → {"data_types": ..., "constraints": ...}
```

## Taxonomía de Tipos Oracle → Canónicos

| Tipo Oracle          | Canónico  | Índice |
| -------------------- | --------- | ------ |
| VARCHAR2, NVARCHAR2  | VARCHAR   | 0      |
| CHAR, NCHAR          | CHAR      | 1      |
| NUMBER, INT, INTEGER | NUMBER    | 2      |
| FLOAT, BINARY_FLOAT  | FLOAT     | 3      |
| DATE                 | DATE      | 4      |
| TIMESTAMP            | TIMESTAMP | 5      |
| CLOB, NCLOB          | CLOB      | 6      |
| BLOB                 | BLOB      | 7      |
| RAW                  | RAW       | 8      |
| ROWID, UROWID        | ROWID     | 9      |
| XMLTYPE              | XML       | 10     |
| (cualquier otro)     | OTHER     | 11     |

## Constraints Binarios

| Posición | Campo            | Descripción            |
| -------- | ---------------- | ---------------------- |
| 0        | `is_primary_key` | 1 si es PK             |
| 1        | `is_foreign_key` | 1 si es FK             |
| 2        | `is_unique`      | 1 si tiene constraint U |
| 3        | `is_indexed`     | 1 si tiene índice       |
| 4        | `nullable`       | 1 si permite NULL       |

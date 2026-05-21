"""
structural_encoder.py — Codificación estructural de metadatos de columnas

Propósito:
  Convierte los metadatos estructurales de cada columna en vectores
  numéricos para el vector compuesto φ(aⱼ).

  Dos componentes:
    1. One-hot encoding del tipo de dato (VARCHAR, NUMBER, DATE, CLOB, etc.)
    2. Codificación binaria de restricciones (PK, FK, Unique, Nullable, Indexed)

Uso:
  encoder = StructuralEncoder()
  e_type = encoder.encode_data_types(columns)    # → (N, n_types)
  e_rest = encoder.encode_constraints(columns)   # → (N, n_constraints)
"""

import numpy as np
from typing import List

from schema_extractor import ColumnMetadata


# ---------------------------------------------------------------------------
# Taxonomía de tipos Oracle → canónicos
# ---------------------------------------------------------------------------

ORACLE_TYPE_MAP = {
    "VARCHAR2":  "VARCHAR",
    "VARCHAR":   "VARCHAR",
    "NVARCHAR2": "VARCHAR",
    "CHAR":      "CHAR",
    "NCHAR":     "CHAR",
    "NUMBER":    "NUMBER",
    "FLOAT":     "FLOAT",
    "BINARY_FLOAT":  "FLOAT",
    "BINARY_DOUBLE": "FLOAT",
    "INT":       "NUMBER",
    "INTEGER":   "NUMBER",
    "BIGINT":    "NUMBER",
    "SMALLINT":  "NUMBER",
    "DECIMAL":   "NUMBER",
    "DATE":      "DATE",
    "TIMESTAMP": "TIMESTAMP",
    "CLOB":      "CLOB",
    "NCLOB":     "CLOB",
    "BLOB":      "BLOB",
    "RAW":       "RAW",
    "ROWID":     "ROWID",
    "UROWID":    "ROWID",
    "XMLTYPE":   "XML",
}

# Orden canónico (el índice en el one-hot corresponde a esta lista)
CANONICAL_TYPES = [
    "VARCHAR", "CHAR", "NUMBER", "FLOAT",
    "DATE", "TIMESTAMP",
    "CLOB", "BLOB",
    "RAW", "ROWID", "XML",
    "OTHER",
]

# Nombres de constraints binarios
CONSTRAINT_NAMES = [
    "is_primary_key",
    "is_foreign_key",
    "is_unique",
    "is_indexed",
    "nullable",
]


# ---------------------------------------------------------------------------
# Structural Encoder
# ---------------------------------------------------------------------------

class StructuralEncoder:
    """
    Codifica metadatos estructurales de columnas en vectores numéricos.

    Atributos:
        type_vocab: List[str] — taxonomía canónica de tipos.
        constraint_names: List[str] — nombres de constraints binarios.
    """

    def __init__(self):
        self.type_vocab = CANONICAL_TYPES
        self.constraint_names = CONSTRAINT_NAMES
        self._type_to_idx = {t: i for i, t in enumerate(self.type_vocab)}

    # ── One-hot de tipos de dato ──────────────────────────────────────

    def _canonical_type(self, oracle_type: str) -> str:
        """Mapea tipo Oracle a tipo canónico."""
        base = oracle_type.upper().split("(")[0].split(" ")[0]
        return ORACLE_TYPE_MAP.get(base, "OTHER")

    def encode_data_types(self, columns: List[ColumnMetadata]) -> np.ndarray:
        """
        One-hot encoding de tipos de dato.

        Returns:
            numpy array shape (len(columns), len(CANONICAL_TYPES)).
        """
        n_types = len(self.type_vocab)
        matrix = np.zeros((len(columns), n_types), dtype=np.float32)

        for i, col in enumerate(columns):
            canonical = self._canonical_type(col.data_type)
            idx = self._type_to_idx.get(canonical, self._type_to_idx["OTHER"])
            matrix[i, idx] = 1.0

        return matrix

    # ── Binario de restricciones ──────────────────────────────────────

    def encode_constraints(self, columns: List[ColumnMetadata]) -> np.ndarray:
        """
        Codificación binaria de restricciones.

        Orden: [is_primary_key, is_foreign_key, is_unique, is_indexed, nullable]

        Returns:
            numpy array shape (len(columns), 5) con valores {0.0, 1.0}.
        """
        matrix = np.zeros((len(columns), 5), dtype=np.float32)

        for i, col in enumerate(columns):
            matrix[i, 0] = float(col.is_primary_key)
            matrix[i, 1] = float(col.is_foreign_key)
            matrix[i, 2] = float(col.is_unique)
            matrix[i, 3] = float(col.is_indexed)
            matrix[i, 4] = float(col.nullable)

        return matrix

    # ── Todo en uno ───────────────────────────────────────────────────

    def encode_all(self, columns: List[ColumnMetadata]) -> dict:
        """
        Codifica tipos y constraints; devuelve dict con ambos.

        Returns:
            {"data_types": ndarray (N, n_types),
             "constraints": ndarray (N, n_constraints)}
        """
        return {
            "data_types": self.encode_data_types(columns),
            "constraints": self.encode_constraints(columns),
        }


# ---------------------------------------------------------------------------
# Cardinaidad de tipos
# ---------------------------------------------------------------------------

def type_vocab_size() -> int:
    """Número de tipos canónicos (útil para calcular dimensión)."""
    return len(CANONICAL_TYPES)


def constraint_dim() -> int:
    """Dimensión del vector de constraints (siempre 5)."""
    return len(CONSTRAINT_NAMES)

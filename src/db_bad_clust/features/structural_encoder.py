"""
structural_encoder.py — Structural encoding of column metadata

Purpose:
  Convert structural metadata of each column into numeric
  vectors for the composite vector φ(aⱼ).

  Two components:
    1. One-hot encoding of data type (VARCHAR, NUMBER, DATE, CLOB, etc.)
    2. Binary encoding of constraints (PK, FK, Unique, Nullable, Indexed)

Usage:
  encoder = StructuralEncoder()
  e_type = encoder.encode_data_types(columns)    # → (N, n_types)
  e_rest = encoder.encode_constraints(columns)   # → (N, n_constraints)
"""

from __future__ import annotations

import numpy as np

from db_bad_clust.data.schema_extractor import ColumnMetadata

# ---------------------------------------------------------------------------
# Oracle type → canonical mapping
# ---------------------------------------------------------------------------

ORACLE_TYPE_MAP = {
    "VARCHAR2": "VARCHAR",
    "VARCHAR": "VARCHAR",
    "NVARCHAR2": "VARCHAR",
    "CHAR": "CHAR",
    "NCHAR": "CHAR",
    "NUMBER": "NUMBER",
    "FLOAT": "FLOAT",
    "BINARY_FLOAT": "FLOAT",
    "BINARY_DOUBLE": "FLOAT",
    "INT": "NUMBER",
    "INTEGER": "NUMBER",
    "BIGINT": "NUMBER",
    "SMALLINT": "NUMBER",
    "DECIMAL": "NUMBER",
    "DATE": "DATE",
    "TIMESTAMP": "TIMESTAMP",
    "CLOB": "CLOB",
    "NCLOB": "CLOB",
    "BLOB": "BLOB",
    "RAW": "RAW",
    "ROWID": "ROWID",
    "UROWID": "ROWID",
    "XMLTYPE": "XML",
}

# Orden canónico (el índice en el one-hot corresponde a esta lista)
CANONICAL_TYPES = [
    "VARCHAR",
    "CHAR",
    "NUMBER",
    "FLOAT",
    "DATE",
    "TIMESTAMP",
    "CLOB",
    "BLOB",
    "RAW",
    "ROWID",
    "XML",
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
    Encodes structural column metadata into numeric vectors.

    Attributes:
        type_vocab: List[str] — canonical type taxonomy.
        constraint_names: List[str] — binary constraint names.
    """

    def __init__(self) -> None:
        self.type_vocab = CANONICAL_TYPES
        self.constraint_names = CONSTRAINT_NAMES
        self._type_to_idx: dict[str, int] = {t: i for i, t in enumerate(self.type_vocab)}

    # ── One-hot data type encoding ────────────────────────────────────

    def _canonical_type(self, oracle_type: str) -> str:
        """Map Oracle type to canonical type."""
        base = oracle_type.upper().split("(")[0].split(" ")[0]
        return ORACLE_TYPE_MAP.get(base, "OTHER")

    def encode_data_types(self, columns: list[ColumnMetadata]) -> np.ndarray:
        """
        One-hot encoding of data types.

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

    # ── Binary constraint encoding ────────────────────────────────────

    def encode_constraints(self, columns: list[ColumnMetadata]) -> np.ndarray:
        """
        Binary encoding of constraints.

        Order: [is_primary_key, is_foreign_key, is_unique, is_indexed, nullable]

        Returns:
            numpy array shape (len(columns), 5) with values {0.0, 1.0}.
        """
        matrix = np.zeros((len(columns), 5), dtype=np.float32)

        for i, col in enumerate(columns):
            matrix[i, 0] = float(col.is_primary_key)
            matrix[i, 1] = float(col.is_foreign_key)
            matrix[i, 2] = float(col.is_unique)
            matrix[i, 3] = float(col.is_indexed)
            matrix[i, 4] = float(col.nullable)

        return matrix

    # ── All-in-one ────────────────────────────────────────────────────

    def encode_all(self, columns: list[ColumnMetadata]) -> dict[str, np.ndarray]:
        """
        Encode types and constraints; return dict with both.

        Returns:
            {"data_types": ndarray (N, n_types),
             "constraints": ndarray (N, n_constraints),
             "statistical": ndarray (N, 1)}
        """
        return {
            "data_types": self.encode_data_types(columns),
            "constraints": self.encode_constraints(columns),
            "statistical": self.encode_statistical(columns),
        }

    # ── Statistical features (continuous) ─────────────────────────────

    def encode_statistical(self, columns: list[ColumnMetadata]) -> np.ndarray:
        """
        Encode continuous statistical features per column.

        Currently: data_length (log-transformed to reduce skew).

        Returns:
            numpy array shape (len(columns), 1) with data_length
            on logarithmic scale (log1p) to reduce skew.
        """
        matrix = np.zeros((len(columns), 1), dtype=np.float32)
        for i, col in enumerate(columns):
            raw = float(col.data_length) if col.data_length is not None else 0.0
            matrix[i, 0] = np.log1p(raw)
        return matrix


# ---------------------------------------------------------------------------
# Type cardinality
# ---------------------------------------------------------------------------


def type_vocab_size() -> int:
    """Number of canonical types (useful for dimension calculation)."""
    return len(CANONICAL_TYPES)


def constraint_dim() -> int:
    """Dimension of the constraint vector (always 5)."""
    return len(CONSTRAINT_NAMES)


def statistical_dim() -> int:
    """Dimension of the statistical vector (currently 1: data_length)."""
    return 1

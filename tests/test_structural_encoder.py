"""
test_structural_encoder.py — Tests for the StructuralEncoder module.

Verifies:
  - encode_data_types returns correct shape (N, 12)
  - encode_constraints returns correct shape (N, 5)
  - encode_statistical returns correct shape (N, 1)
  - encode_all returns dict with 3 keys
  - One-hot encoding positions for specific types
  - Canonical type mapping (NUMBER, FLOAT, CLOB, etc.)
  - Module-level helper functions
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest
import numpy as np
from structural_encoder import (
    StructuralEncoder,
    type_vocab_size,
    constraint_dim,
    statistical_dim,
    CANONICAL_TYPES,
    CONSTRAINT_NAMES,
)
from schema_extractor import ColumnMetadata


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def encoder() -> StructuralEncoder:
    """Default StructuralEncoder instance."""
    return StructuralEncoder()


@pytest.fixture
def single_column() -> list[ColumnMetadata]:
    """A single NUMBER column (trivial case)."""
    return [
        ColumnMetadata(
            name="ID",
            data_type="NUMBER",
            data_length=22,
            nullable=False,
            is_primary_key=True,
        )
    ]


# ─── Module-level helpers ───────────────────────────────────────────────


class TestModuleHelpers:
    """Verify the module-level helper functions."""

    def test_type_vocab_size(self) -> None:
        """type_vocab_size() == 12."""
        assert type_vocab_size() == 12

    def test_constraint_dim(self) -> None:
        """constraint_dim() == 5."""
        assert constraint_dim() == 5

    def test_statistical_dim(self) -> None:
        """statistical_dim() == 1."""
        assert statistical_dim() == 1

    def test_canonical_types_list_length(self) -> None:
        """CANONICAL_TYPES has 12 entries."""
        assert len(CANONICAL_TYPES) == 12

    def test_other_is_last(self) -> None:
        """'OTHER' is the last (fallback) entry in CANONICAL_TYPES."""
        assert CANONICAL_TYPES[-1] == "OTHER"

    def test_constraint_names_order(self) -> None:
        """CONSTRAINT_NAMES order matches encoder usage."""
        assert CONSTRAINT_NAMES == [
            "is_primary_key",
            "is_foreign_key",
            "is_unique",
            "is_indexed",
            "nullable",
        ]


# ─── Data type encoding ────────────────────────────────────────────────


class TestEncodeDataTypes:
    """Verify one-hot encoding of data types."""

    def test_shape(self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]) -> None:
        """encode_data_types returns (N, 12)."""
        encoded = encoder.encode_data_types(sample_columns)
        assert encoded.shape == (5, 12)

    def test_single_column(
        self, encoder: StructuralEncoder, single_column: list[ColumnMetadata]
    ) -> None:
        """Single NUMBER column → shape (1, 12)."""
        encoded = encoder.encode_data_types(single_column)
        assert encoded.shape == (1, 12)

    def test_empty_columns(self, encoder: StructuralEncoder) -> None:
        """Empty list → shape (0, 12)."""
        encoded = encoder.encode_data_types([])
        assert encoded.shape == (0, 12)

    def test_number_position(
        self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]
    ) -> None:
        """NUMBER type is at index 2 (VARCHAR=0, CHAR=1, NUMBER=2)."""
        encoded = encoder.encode_data_types(sample_columns)
        # Sample columns: ID=NUMBER, NOMBRE=VARCHAR2, SALARIO=NUMBER, FECHA_ALTA=DATE, ACTIVO=CHAR
        #   0: VARCHAR index, 1: CHAR index, 2: NUMBER index, 3: FLOAT, 4: DATE, 5: TIMESTAMP, ...
        assert encoded[0, 2] == 1.0  # ID → NUMBER
        assert encoded[1, 0] == 1.0  # NOMBRE → VARCHAR
        assert encoded[2, 2] == 1.0  # SALARIO → NUMBER
        assert encoded[3, 4] == 1.0  # FECHA_ALTA → DATE
        assert encoded[4, 1] == 1.0  # ACTIVO → CHAR

    def test_only_one_one_per_row(
        self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]
    ) -> None:
        """Each row has exactly one 1.0 (one-hot)."""
        encoded = encoder.encode_data_types(sample_columns)
        row_sums = encoded.sum(axis=1)
        assert np.allclose(row_sums, 1.0)

    @pytest.mark.parametrize(
        "oracle_type, expected_canonical",
        [
            ("VARCHAR2", "VARCHAR"),
            ("VARCHAR", "VARCHAR"),
            ("NVARCHAR2", "VARCHAR"),
            ("CHAR", "CHAR"),
            ("NCHAR", "CHAR"),
            ("NUMBER", "NUMBER"),
            ("INT", "NUMBER"),
            ("INTEGER", "NUMBER"),
            ("BIGINT", "NUMBER"),
            ("FLOAT", "FLOAT"),
            ("BINARY_FLOAT", "FLOAT"),
            ("BINARY_DOUBLE", "FLOAT"),
            ("DATE", "DATE"),
            ("TIMESTAMP", "TIMESTAMP"),
            ("CLOB", "CLOB"),
            ("NCLOB", "CLOB"),
            ("BLOB", "BLOB"),
            ("RAW", "RAW"),
            ("ROWID", "ROWID"),
            ("UROWID", "ROWID"),
            ("XMLTYPE", "XML"),
            ("UNKNOWN_TYPE", "OTHER"),
        ],
    )
    def test_type_mapping(
        self,
        encoder: StructuralEncoder,
        oracle_type: str,
        expected_canonical: str,
    ) -> None:
        """Each Oracle type maps to the correct canonical category."""
        col = ColumnMetadata(name="COL", data_type=oracle_type, data_length=10, nullable=True)
        encoded = encoder.encode_data_types([col])
        canonical_idx = CANONICAL_TYPES.index(expected_canonical)
        assert encoded[0, canonical_idx] == 1.0, (
            f"{oracle_type} should map to {expected_canonical} at index {canonical_idx}"
        )

    def test_type_with_precision(self, encoder: StructuralEncoder) -> None:
        """Oracle type with precision suffix like 'NUMBER(10,2)' maps to NUMBER."""
        col = ColumnMetadata(name="COL", data_type="NUMBER(10,2)", data_length=22, nullable=True)
        encoded = encoder.encode_data_types([col])
        number_idx = CANONICAL_TYPES.index("NUMBER")
        assert encoded[0, number_idx] == 1.0

    def test_dtype_is_float32(
        self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]
    ) -> None:
        """Output dtype is np.float32."""
        encoded = encoder.encode_data_types(sample_columns)
        assert encoded.dtype == np.float32


# ─── Constraint encoding ────────────────────────────────────────────────


class TestEncodeConstraints:
    """Verify binary encoding of constraints."""

    def test_shape(self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]) -> None:
        """encode_constraints returns (N, 5)."""
        encoded = encoder.encode_constraints(sample_columns)
        assert encoded.shape == (5, 5)

    def test_single_column(
        self, encoder: StructuralEncoder, single_column: list[ColumnMetadata]
    ) -> None:
        """Single column → shape (1, 5)."""
        encoded = encoder.encode_constraints(single_column)
        assert encoded.shape == (1, 5)

    def test_empty_columns(self, encoder: StructuralEncoder) -> None:
        """Empty list → shape (0, 5)."""
        encoded = encoder.encode_constraints([])
        assert encoded.shape == (0, 5)

    def test_pk_flag(
        self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]
    ) -> None:
        """Primary key column has is_primary_key=1 at position 0."""
        encoded = encoder.encode_constraints(sample_columns)
        # ID has is_primary_key=True
        assert encoded[0, 0] == 1.0
        # Other columns are not PK
        assert encoded[1, 0] == 0.0
        assert encoded[2, 0] == 0.0

    def test_nullable_flag(
        self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]
    ) -> None:
        """Nullable flag at position 4 (1.0 if nullable)."""
        encoded = encoder.encode_constraints(sample_columns)
        # ID is NOT nullable
        assert encoded[0, 4] == 0.0
        # NOMBRE is nullable
        assert encoded[1, 4] == 1.0

    def test_indexed_flag(
        self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]
    ) -> None:
        """Indexed flag at position 3."""
        encoded = encoder.encode_constraints(sample_columns)
        # NOMBRE is indexed
        assert encoded[1, 3] == 1.0
        # SALARIO is not indexed
        assert encoded[2, 3] == 0.0

    def test_fk_flag(self, encoder: StructuralEncoder) -> None:
        """Foreign key flag at position 1."""
        cols = [
            ColumnMetadata(
                name="DEPT_ID",
                data_type="NUMBER",
                data_length=22,
                nullable=True,
                is_foreign_key=True,
            ),
            ColumnMetadata(
                name="NAME",
                data_type="VARCHAR2",
                data_length=50,
                nullable=False,
            ),
        ]
        encoded = encoder.encode_constraints(cols)
        assert encoded[0, 1] == 1.0  # FK
        assert encoded[1, 1] == 0.0  # not FK

    def test_unique_flag(self, encoder: StructuralEncoder) -> None:
        """Unique flag at position 2."""
        cols = [
            ColumnMetadata(
                name="EMAIL",
                data_type="VARCHAR2",
                data_length=100,
                nullable=True,
                is_unique=True,
            ),
        ]
        encoded = encoder.encode_constraints(cols)
        assert encoded[0, 2] == 1.0

    def test_all_flags_independent(self, encoder: StructuralEncoder) -> None:
        """All constraints can be simultaneously True."""
        cols = [
            ColumnMetadata(
                name="ALL_FLAGS",
                data_type="NUMBER",
                data_length=22,
                nullable=True,
                is_primary_key=True,
                is_foreign_key=True,
                is_unique=True,
                is_indexed=True,
            ),
        ]
        encoded = encoder.encode_constraints(cols)
        assert np.allclose(encoded[0], [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_values_are_float32(
        self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]
    ) -> None:
        """Output dtype is np.float32."""
        encoded = encoder.encode_constraints(sample_columns)
        assert encoded.dtype == np.float32


# ─── Statistical encoding ──────────────────────────────────────────────


class TestEncodeStatistical:
    """Verify statistical feature encoding."""

    def test_shape(self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]) -> None:
        """encode_statistical returns (N, 1)."""
        encoded = encoder.encode_statistical(sample_columns)
        assert encoded.shape == (5, 1)

    def test_empty_columns(self, encoder: StructuralEncoder) -> None:
        """Empty list → shape (0, 1)."""
        encoded = encoder.encode_statistical([])
        assert encoded.shape == (0, 1)

    def test_log_transform(self, encoder: StructuralEncoder) -> None:
        """data_length is log1p-transformed (log(1 + value))."""
        cols = [
            ColumnMetadata(name="A", data_type="VARCHAR2", data_length=100, nullable=True),
            ColumnMetadata(name="B", data_type="NUMBER", data_length=22, nullable=True),
        ]
        encoded = encoder.encode_statistical(cols)
        expected_a = np.log1p(100.0)
        expected_b = np.log1p(22.0)
        assert np.isclose(encoded[0, 0], expected_a)
        assert np.isclose(encoded[1, 0], expected_b)

    def test_none_length_becomes_zero(self, encoder: StructuralEncoder) -> None:
        """None data_length defaults to 0, log1p(0) = 0."""
        cols = [
            ColumnMetadata(name="A", data_type="DATE", data_length=None, nullable=True),
        ]
        encoded = encoder.encode_statistical(cols)
        assert encoded[0, 0] == 0.0

    def test_zero_length(self, encoder: StructuralEncoder) -> None:
        """data_length=0 → log1p(0) = 0."""
        cols = [
            ColumnMetadata(name="A", data_type="VARCHAR2", data_length=0, nullable=True),
        ]
        encoded = encoder.encode_statistical(cols)
        assert encoded[0, 0] == 0.0

    def test_dtype(self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]) -> None:
        """Output dtype is np.float32."""
        encoded = encoder.encode_statistical(sample_columns)
        assert encoded.dtype == np.float32


# ─── encode_all ─────────────────────────────────────────────────────────


class TestEncodeAll:
    """Verify the combined encode_all method."""

    def test_returns_dict(
        self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]
    ) -> None:
        """encode_all returns a dict."""
        result = encoder.encode_all(sample_columns)
        assert isinstance(result, dict)

    def test_has_three_keys(
        self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]
    ) -> None:
        """encode_all dict has 3 keys."""
        result = encoder.encode_all(sample_columns)
        assert set(result.keys()) == {"data_types", "constraints", "statistical"}

    def test_data_types_shape(
        self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]
    ) -> None:
        """Data types matrix is (N, 12)."""
        result = encoder.encode_all(sample_columns)
        assert result["data_types"].shape == (5, 12)

    def test_constraints_shape(
        self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]
    ) -> None:
        """Constraints matrix is (N, 5)."""
        result = encoder.encode_all(sample_columns)
        assert result["constraints"].shape == (5, 5)

    def test_statistical_shape(
        self, encoder: StructuralEncoder, sample_columns: list[ColumnMetadata]
    ) -> None:
        """Statistical matrix is (N, 1)."""
        result = encoder.encode_all(sample_columns)
        assert result["statistical"].shape == (5, 1)

    def test_empty_columns(self, encoder: StructuralEncoder) -> None:
        """Empty list returns dict with zero-row arrays."""
        result = encoder.encode_all([])
        assert result["data_types"].shape == (0, 12)
        assert result["constraints"].shape == (0, 5)
        assert result["statistical"].shape == (0, 1)


# ─── Edge cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge-case column definitions."""

    def test_unknown_goes_to_other(self, encoder: StructuralEncoder) -> None:
        """Unrecognized Oracle type maps to 'OTHER' (last position)."""
        col = ColumnMetadata(
            name="WEIRD", data_type="SOMETHING_STRANGE", data_length=99, nullable=True
        )
        encoded = encoder.encode_data_types([col])
        other_idx = CANONICAL_TYPES.index("OTHER")
        assert encoded[0, other_idx] == 1.0

    def test_canonical_type_unicode(self, encoder: StructuralEncoder) -> None:
        """Type names with unicode characters (e.g. NVARCHAR2)."""
        col = ColumnMetadata(
            name="UNICODE_COL", data_type="NVARCHAR2", data_length=200, nullable=True
        )
        encoded = encoder.encode_data_types([col])
        varchar_idx = CANONICAL_TYPES.index("VARCHAR")
        assert encoded[0, varchar_idx] == 1.0

    def test_all_other_types(self, encoder: StructuralEncoder) -> None:
        """Cover each canonical type with a representative."""
        representatives = [
            ("VARCHAR2", "VARCHAR"),
            ("CHAR", "CHAR"),
            ("NUMBER", "NUMBER"),
            ("FLOAT", "FLOAT"),
            ("DATE", "DATE"),
            ("TIMESTAMP", "TIMESTAMP"),
            ("CLOB", "CLOB"),
            ("BLOB", "BLOB"),
            ("RAW", "RAW"),
            ("ROWID", "ROWID"),
            ("XMLTYPE", "XML"),
            ("CUSTOM_TYPE", "OTHER"),
        ]
        cols = [
            ColumnMetadata(name=f"COL{i}", data_type=t[0], data_length=10, nullable=True)
            for i, t in enumerate(representatives)
        ]
        encoded = encoder.encode_data_types(cols)
        assert encoded.shape == (12, 12)
        # Each row should have a single 1 on the diagonal
        for i, (_, canonical) in enumerate(representatives):
            idx = CANONICAL_TYPES.index(canonical)
            assert encoded[i, idx] == 1.0, f"Row {i} ({canonical}) failed"
        assert np.allclose(encoded.sum(axis=1), 1.0)

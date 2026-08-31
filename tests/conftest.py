"""
conftest.py — Shared pytest fixtures (schema/column fixtures for rule engine + clustering).
"""

import numpy as np
import pytest

from db_bad_clust.data.schema_extractor import ColumnMetadata, DatabaseSchema, TableMetadata

# ── Column-level fixtures ──────────────────────────────────────────────


@pytest.fixture
def sample_columns():
    """Five columns with varied types and constraints."""
    return [
        ColumnMetadata(
            name="ID",
            data_type="NUMBER",
            data_length=22,
            nullable=False,
            is_primary_key=True,
        ),
        ColumnMetadata(
            name="NOMBRE",
            data_type="VARCHAR2",
            data_length=100,
            nullable=True,
            is_indexed=True,
        ),
        ColumnMetadata(
            name="SALARIO",
            data_type="NUMBER",
            data_length=22,
            nullable=True,
        ),
        ColumnMetadata(
            name="FECHA_ALTA",
            data_type="DATE",
            data_length=7,
            nullable=True,
        ),
        ColumnMetadata(
            name="ACTIVO",
            data_type="CHAR",
            data_length=1,
            nullable=True,
            is_indexed=True,
        ),
    ]


@pytest.fixture
def sample_columns_with_fk():
    """Columns including a foreign-key relationship."""
    return [
        ColumnMetadata(
            name="ID",
            data_type="NUMBER",
            data_length=22,
            nullable=False,
            is_primary_key=True,
        ),
        ColumnMetadata(
            name="DEPTO_ID",
            data_type="NUMBER",
            data_length=22,
            nullable=True,
            is_foreign_key=True,
            fk_references_table="DEPARTAMENTOS",
            fk_references_column="ID",
        ),
        ColumnMetadata(
            name="NOMBRE",
            data_type="VARCHAR2",
            data_length=50,
            nullable=False,
        ),
    ]


@pytest.fixture
def sample_columns_with_blob():
    """Columns including a BLOB / CLOB for type-vocab coverage."""
    return [
        ColumnMetadata(name="ID", data_type="NUMBER", data_length=22, nullable=False),
        ColumnMetadata(name="FOTO", data_type="BLOB", data_length=4000, nullable=True),
        ColumnMetadata(name="CURRICULUM", data_type="CLOB", data_length=8000, nullable=True),
        ColumnMetadata(name="DATOS_XML", data_type="XMLTYPE", data_length=2000, nullable=True),
    ]


# ── Table-level fixtures ───────────────────────────────────────────────


@pytest.fixture
def sample_tables(sample_columns):
    """Two tables: EMPLEADOS (5 cols) and DEPARTAMENTOS (2 cols)."""
    return [
        TableMetadata(name="EMPLEADOS", columns=sample_columns, row_count_approx=100),
        TableMetadata(
            name="DEPARTAMENTOS",
            columns=[
                ColumnMetadata(
                    name="ID",
                    data_type="NUMBER",
                    data_length=22,
                    nullable=False,
                    is_primary_key=True,
                ),
                ColumnMetadata(name="NOMBRE", data_type="VARCHAR2", data_length=50, nullable=False),
            ],
        ),
    ]


@pytest.fixture
def sample_tables_multi():
    """Three tables with 10 total columns — good for clustering tests."""
    return [
        TableMetadata(
            name="EMPLEADOS",
            columns=[
                ColumnMetadata(
                    name="ID",
                    data_type="NUMBER",
                    data_length=22,
                    nullable=False,
                    is_primary_key=True,
                ),
                ColumnMetadata(
                    name="NOMBRE",
                    data_type="VARCHAR2",
                    data_length=100,
                    nullable=False,
                ),
                ColumnMetadata(name="SALARIO", data_type="NUMBER", data_length=22, nullable=True),
                ColumnMetadata(name="FECHA_ALTA", data_type="DATE", data_length=7, nullable=True),
                ColumnMetadata(
                    name="ACTIVO",
                    data_type="CHAR",
                    data_length=1,
                    nullable=False,
                ),
            ],
        ),
        TableMetadata(
            name="DEPARTAMENTOS",
            columns=[
                ColumnMetadata(
                    name="ID",
                    data_type="NUMBER",
                    data_length=22,
                    nullable=False,
                    is_primary_key=True,
                ),
                ColumnMetadata(
                    name="NOMBRE",
                    data_type="VARCHAR2",
                    data_length=50,
                    nullable=False,
                ),
                ColumnMetadata(
                    name="FECHA_CREACION",
                    data_type="DATE",
                    data_length=7,
                    nullable=True,
                ),
            ],
        ),
        TableMetadata(
            name="PRODUCTOS",
            columns=[
                ColumnMetadata(
                    name="ID",
                    data_type="NUMBER",
                    data_length=22,
                    nullable=False,
                    is_primary_key=True,
                ),
                ColumnMetadata(name="PRECIO", data_type="NUMBER", data_length=12, nullable=False),
            ],
        ),
    ]


# ── Schema-level fixtures ──────────────────────────────────────────────


@pytest.fixture
def sample_schema(sample_tables):
    """DatabaseSchema wrapping sample_tables."""
    return DatabaseSchema(tables=sample_tables)


@pytest.fixture
def sample_schema_multi(sample_tables_multi):
    """DatabaseSchema wrapping sample_tables_multi (10 cols)."""
    return DatabaseSchema(tables=sample_tables_multi)


# ── Numeric fixtures ───────────────────────────────────────────────────


@pytest.fixture
def random_embeddings():
    """5 x 768 BERT-like embedding matrix (fixed seed for reproducibility)."""
    np.random.seed(42)
    return np.random.randn(5, 768).astype(np.float32)


@pytest.fixture
def random_embeddings_10():
    """10 x 768 embedding matrix for multi-table tests."""
    np.random.seed(123)
    return np.random.randn(10, 768).astype(np.float32)


@pytest.fixture
def sample_reduced_data():
    """Small 2D array for clustering tests (10 samples, 2 features)."""
    np.random.seed(42)
    # Create 2 clear clusters
    cluster0 = np.random.randn(5, 2) + np.array([2, 2])
    cluster1 = np.random.randn(5, 2) + np.array([-2, -2])
    return np.vstack([cluster0, cluster1]).astype(np.float32)


# ── Encoding fixtures ──────────────────────────────────────────────────


@pytest.fixture
def sample_type_encoding(sample_columns):
    """One-hot type encoding for sample_columns (5 x 12)."""
    from db_bad_clust.features.structural_encoder import StructuralEncoder

    encoder = StructuralEncoder()
    return encoder.encode_data_types(sample_columns)


@pytest.fixture
def sample_constraint_encoding(sample_columns):
    """Binary constraint encoding for sample_columns (5 x 5)."""
    from db_bad_clust.features.structural_encoder import StructuralEncoder

    encoder = StructuralEncoder()
    return encoder.encode_constraints(sample_columns)


@pytest.fixture
def sample_statistical_encoding(sample_columns):
    """Statistical features for sample_columns (5 x 1)."""
    from db_bad_clust.features.structural_encoder import StructuralEncoder

    encoder = StructuralEncoder()
    return encoder.encode_statistical(sample_columns)


# ── Label fixtures ─────────────────────────────────────────────────────


@pytest.fixture
def sample_labels():
    """Cluster labels for 10 columns (3 clusters)."""
    return np.array([0, 0, 1, 1, 2, 0, 1, 2, 2, 0], dtype=int)


@pytest.fixture
def column_table_map():
    """Table name for each of the 10 columns in sample_tables_multi."""
    return [
        "EMPLEADOS",  # ID
        "EMPLEADOS",  # NOMBRE
        "EMPLEADOS",  # SALARIO
        "EMPLEADOS",  # FECHA_ALTA
        "EMPLEADOS",  # ACTIVO
        "DEPARTAMENTOS",  # ID
        "DEPARTAMENTOS",  # NOMBRE
        "DEPARTAMENTOS",  # FECHA_CREACION
        "PRODUCTOS",  # ID
        "PRODUCTOS",  # PRECIO
    ]

"""
Tests for ddl_generator.py.

All tests are mock-based and do not require a running Oracle database.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from ddl_generator import DDLGenerator, _guess_target_type, _quote_if_reserved
from schema_extractor import ColumnMetadata, DatabaseSchema, TableMetadata


@pytest.fixture
def sample_schema() -> DatabaseSchema:
    """Schema with one issue per label family."""
    return DatabaseSchema(
        tables=[
            TableMetadata(
                name="EMPLEADOS",
                columns=[
                    ColumnMetadata(name="ID", data_type="NUMBER", data_length=22, nullable=False),
                    ColumnMetadata(name="FECHA_NACIMIENTO", data_type="VARCHAR2", data_length=20, nullable=True),
                    ColumnMetadata(name="SALARIO", data_type="VARCHAR2", data_length=20, nullable=True),
                    ColumnMetadata(name="ACTIVO", data_type="VARCHAR2", data_length=10, nullable=True),
                    ColumnMetadata(name="FROM", data_type="VARCHAR2", data_length=50, nullable=True),
                ],
            ),
            TableMetadata(
                name="PEDIDOS",
                columns=[
                    ColumnMetadata(name="ID", data_type="NUMBER", data_length=22, nullable=True),
                    ColumnMetadata(name="CLIENTE_ID", data_type="NUMBER", data_length=22, nullable=True),
                    ColumnMetadata(name="FECHA_PEDIDO", data_type="DATE", data_length=7, nullable=True),
                ],
                row_count_approx=2_000_000,
            ),
            TableMetadata(
                name="LEGACY",
                columns=[
                    ColumnMetadata(name="ID", data_type="NUMBER", data_length=22, nullable=False),
                    ColumnMetadata(name="TEXTO", data_type="LONG", data_length=0, nullable=True),
                ],
            ),
        ],
        fk_columns={"PEDIDOS": {"FK_CLIENTE": ["CLIENTE_ID"]}},
        index_columns={"PEDIDOS": {"IDX_ID": ["ID"]}},
    )


class TestDDLGenerator:
    def test_generates_non_empty_script(self, sample_schema: DatabaseSchema) -> None:
        sql = DDLGenerator().from_schema(sample_schema)
        assert "ALTER TABLE" in sql
        assert "-- Summary" in sql

    def test_date_as_text_ddl(self, sample_schema: DatabaseSchema) -> None:
        sql = DDLGenerator().from_schema(sample_schema)
        assert "ALTER TABLE EMPLEADOS MODIFY (FECHA_NACIMIENTO DATE);" in sql

    def test_number_as_text_ddl(self, sample_schema: DatabaseSchema) -> None:
        sql = DDLGenerator().from_schema(sample_schema)
        assert "ALTER TABLE EMPLEADOS MODIFY (SALARIO NUMBER);" in sql

    def test_reserved_word_quoted_rename(self, sample_schema: DatabaseSchema) -> None:
        sql = DDLGenerator().from_schema(sample_schema)
        assert "RENAME COLUMN \"FROM\" TO FROM_COL" in sql

    def test_fk_without_index_ddl(self, sample_schema: DatabaseSchema) -> None:
        sql = DDLGenerator().from_schema(sample_schema)
        assert "CREATE INDEX IDX_PEDIDOS_CLIENTE_ID ON PEDIDOS (CLIENTE_ID);" in sql

    def test_obsolete_type_ddl(self, sample_schema: DatabaseSchema) -> None:
        sql = DDLGenerator().from_schema(sample_schema)
        assert "ALTER TABLE LEGACY MODIFY (TEXTO CLOB);" in sql

    def test_missing_pk_ddl(self, sample_schema: DatabaseSchema) -> None:
        sql = DDLGenerator().from_schema(sample_schema)
        assert "ADD CONSTRAINT PK_PEDIDOS PRIMARY KEY (ID)" in sql

    def test_partition_candidate_comment(self, sample_schema: DatabaseSchema) -> None:
        sql = DDLGenerator().from_schema(sample_schema)
        assert "Partition candidate PEDIDOS" in sql

    def test_deduplicates_identical_statements(self, sample_schema: DatabaseSchema) -> None:
        sql = DDLGenerator().from_schema(sample_schema)
        # FECHA_NACIMIENTO triggers both date_as_text and wrong_data_types,
        # but only one ALTER statement should appear.
        assert sql.count("ALTER TABLE EMPLEADOS MODIFY (FECHA_NACIMIENTO DATE);") == 1


class TestHelpers:
    def test_guess_target_type(self) -> None:
        assert _guess_target_type("FECHA_ALTA") == "DATE"
        assert _guess_target_type("SALARIO") == "NUMBER"
        assert _guess_target_type("UNKNOWN") == "NUMBER"

    def test_quote_if_reserved(self) -> None:
        assert _quote_if_reserved("FROM") == '"FROM"'
        assert _quote_if_reserved("NOMBRE") == "NOMBRE"


class TestCLI:
    def test_loads_pickle_and_writes_sql(self, tmp_path: Path, sample_schema: DatabaseSchema) -> None:
        pickle_path = tmp_path / "schema.pkl"
        output_path = tmp_path / "remediation.sql"
        with pickle_path.open("wb") as f:
            pickle.dump(sample_schema, f)

        from ddl_generator import _load_schema_from_pickle

        loaded = _load_schema_from_pickle(pickle_path)
        assert loaded.total_columns() == sample_schema.total_columns()

        sql = DDLGenerator().from_schema(loaded)
        output_path.write_text(sql, encoding="utf-8")
        assert output_path.exists()
        assert "ALTER TABLE" in output_path.read_text(encoding="utf-8")

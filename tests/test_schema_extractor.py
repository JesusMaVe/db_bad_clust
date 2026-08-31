"""
test_schema_extractor.py — Tests for the bulk SchemaExtractor logic.

Uses a fake cursor/connection (no DB required) to verify:
  - One bulk query per dictionary view (no per-table N+1)
  - Grouping of columns/constraints/comments by table
  - PK/FK/unique/index/comment assignment onto ColumnMetadata
  - Identity and virtual column flags from user_tab_cols
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from db_bad_clust.data.schema_extractor import SchemaExtractor


class FakeCursor:
    """Returns canned rows per SQL keyword; records executed statements."""

    def __init__(self):
        self.executed: list[str] = []

    def execute(self, sql, params=None, **kwargs):
        self.executed.append(sql)
        sql_l = sql.lower()
        if "from user_tables" in sql_l:
            self._rows = [("EMPLEADOS", 100), ("CONFIGURACION", 5)]
        elif "from user_tab_cols" in sql_l:
            self._rows = [
                ("EMPLEADOS", "ID", "NUMBER", "N", 22, None, 0, None, "YES", "NO"),
                ("EMPLEADOS", "EMAIL", "DATE", "Y", 7, None, None, None, "NO", "NO"),
                ("CONFIGURACION", "CLAVE", "VARCHAR2", "Y", 100, None, None, None, "NO", "YES"),
            ]
        elif "constraint_type = :ctype" in sql_l:
            # PK: EMPLEADOS.ID / U: none
            self._rows = [("EMPLEADOS", "ID")] if kwargs.get("ctype") == "P" else []
        elif "constraint_type = 'r'" in sql_l and "r_constraint_name" in sql_l:
            # Existing FK reference lookup (with c2.table_name / cc2.column_name)
            self._rows = [
                ("EMPLEADOS", "EMAIL", "CONFIGURACION", "CLAVE", "FK_EMAIL")
            ]
        elif "constraint_type = 'r'" in sql_l:
            # Ordered FK columns (new in Phase 2)
            self._rows = [("EMPLEADOS", "FK_EMAIL", "EMAIL", 1)]
        elif "user_ind_columns" in sql_l:
            # Both _get_index_columns() and get_redundant_indexes() use 4 columns.
            # Column order distinguishes the two callers.
            if "index_name, column_name" in sql_l:
                self._rows = [
                    ("EMPLEADOS", "IDX_EMP_0", "ID", 1),
                    ("EMPLEADOS", "IDX_EMP_RED", "ID", 1),
                    ("EMPLEADOS", "IDX_EMP_RED", "ACTIVO", 2),
                    ("EMPLEADOS", "IDX_EMAIL", "EMAIL", 1),
                ]
            else:
                # get_redundant_indexes order: table_name, column_name, index_name, column_position
                self._rows = [
                    ("EMPLEADOS", "ID", "IDX_EMP_0", 1),
                    ("EMPLEADOS", "ID", "IDX_EMP_RED", 1),
                    ("EMPLEADOS", "ACTIVO", "IDX_EMP_RED", 2),
                ]
        elif "user_tab_statistics" in sql_l:
            self._rows = [("EMPLEADOS", "NO", 100, None), ("CONFIGURACION", "NO", 5, None)]
        elif "status, validated" in sql_l:
            self._rows = []
        elif "user_tab_comments" in sql_l:
            self._rows = [("EMPLEADOS", "Tabla de empleados")]
        elif "user_col_comments" in sql_l:
            self._rows = [("EMPLEADOS", "EMAIL", "Correo del empleado")]
        else:
            self._rows = []

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class FakeConnection:
    """Duck-typed oracledb.Connection returning a FakeCursor."""

    def __init__(self, cursor: FakeCursor | None = None):
        self._cursor = cursor or FakeCursor()

    def cursor(self) -> FakeCursor:
        return self._cursor


@pytest.fixture
def schema():
    extractor = SchemaExtractor(FakeConnection())  # type: ignore[arg-type]
    return extractor.extract_all()


class TestBulkExtraction:
    def test_no_n_plus_one(self):
        """Exactly 12 dictionary queries total (bulk, no per-table N+1)."""
        conn = FakeConnection()
        SchemaExtractor(conn).extract_all()  # type: ignore[arg-type]
        assert len(conn._cursor.executed) == 12
        assert not any(":table_name" in s for s in conn._cursor.executed)

    def test_tables_and_columns(self, schema):
        assert schema.table_names() == ["EMPLEADOS", "CONFIGURACION"]
        assert schema.total_columns() == 3

    def test_constraints_assigned(self, schema):
        emp = next(t for t in schema.tables if t.name == "EMPLEADOS")
        id_col = next(c for c in emp.columns if c.name == "ID")
        email = next(c for c in emp.columns if c.name == "EMAIL")
        assert id_col.is_primary_key
        assert email.is_foreign_key
        assert email.fk_references_table == "CONFIGURACION"
        assert email.fk_references_column == "CLAVE"
        assert email.is_indexed
        assert not email.is_unique

    def test_comments_assigned(self, schema):
        emp = next(t for t in schema.tables if t.name == "EMPLEADOS")
        email = next(c for c in emp.columns if c.name == "EMAIL")
        assert emp.table_comment == "Tabla de empleados"
        assert email.comments == "Correo del empleado"

    def test_identity_and_virtual_flags(self, schema):
        emp = next(t for t in schema.tables if t.name == "EMPLEADOS")
        id_col = next(c for c in emp.columns if c.name == "ID")
        cfg = next(t for t in schema.tables if t.name == "CONFIGURACION")
        clave = next(c for c in cfg.columns if c.name == "CLAVE")
        assert id_col.is_identity
        assert not id_col.is_virtual
        assert clave.is_virtual

    def test_empty_db(self):
        class EmptyCursor(FakeCursor):
            def execute(self, sql, params=None, **kwargs):
                self.executed.append(sql)
                self._rows = []

        schema = SchemaExtractor(FakeConnection(EmptyCursor())).extract_all()  # type: ignore[arg-type]
        assert schema.tables == []


class TestRedundantIndexes:
    """Verify composite index duplicating a single index is detected."""

    def test_redundant_pair_found(self, schema):
        # FakeCursor: IDX_EMP_0(ID) single + IDX_EMP_RED(ID, ACTIVO) composite
        assert schema.redundant_indexes == [
            ("EMPLEADOS", "ID", "IDX_EMP_0", "IDX_EMP_RED")
        ]

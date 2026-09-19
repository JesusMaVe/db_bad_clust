"""
test_schema_extractor.py — Tests for the bulk SchemaExtractor logic.

Uses a fake cursor/connection (no DB required) to verify:
  - One bulk query per dictionary view (no per-table N+1)
  - Grouping of columns/constraints/comments by table
  - PK/FK/unique/index/comment assignment onto ColumnMetadata
  - Identity and virtual column flags from all_tab_cols
  - Every dictionary query filtered by the schema owner (ALL_* views)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from db_bad_clust.data.schema_extractor import SchemaExtractor


class FakeCursor:
    """Returns canned rows per SQL keyword; records executed statements."""

    def __init__(self, session_user: str = "APP"):
        self.executed: list[str] = []
        self.binds: list[dict] = []
        self.session_user = session_user

    def execute(self, sql, params=None, **kwargs):
        self.executed.append(sql)
        self.binds.append(kwargs)
        sql_l = sql.lower()
        if "from dual" in sql_l:
            self._rows = [(self.session_user,)]
        elif "from all_tables" in sql_l:
            self._rows = [("EMPLEADOS", 100), ("CONFIGURACION", 5)]
        elif "from all_tab_cols" in sql_l:
            self._rows = [
                ("EMPLEADOS", "ID", "NUMBER", "N", 22, None, 0, None, "YES", "NO", 0),
                ("EMPLEADOS", "EMAIL", "DATE", "Y", 7, None, None, None, "NO", "NO", 0),
                ("CONFIGURACION", "CLAVE", "VARCHAR2", "Y", 400, None, None, None, "NO", "YES", 100),
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
        elif "all_ind_columns" in sql_l:
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
        elif "all_tab_statistics" in sql_l:
            self._rows = [("EMPLEADOS", "NO", 100, None), ("CONFIGURACION", "NO", 5, None)]
        elif "status, validated" in sql_l:
            self._rows = []
        elif "all_tab_comments" in sql_l:
            self._rows = [("EMPLEADOS", "Tabla de empleados")]
        elif "all_col_comments" in sql_l:
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
        """12 dictionary queries (bulk, no per-table N+1), plus one to resolve
        the session user when no owner is given."""
        conn = FakeConnection()
        SchemaExtractor(conn).extract_all()  # type: ignore[arg-type]
        assert len(conn._cursor.executed) == 13
        assert not any(":table_name" in s for s in conn._cursor.executed)

        given = FakeConnection()
        SchemaExtractor(given, owner="HR").extract_all()  # type: ignore[arg-type]
        assert len(given._cursor.executed) == 12

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
                self._rows = [("APP",)] if "dual" in sql.lower() else []

        schema = SchemaExtractor(FakeConnection(EmptyCursor())).extract_all()  # type: ignore[arg-type]
        assert schema.tables == []


class TestRedundantIndexes:
    """Verify composite index duplicating a single index is detected."""

    def test_redundant_pair_found(self, schema):
        # FakeCursor: IDX_EMP_0(ID) single + IDX_EMP_RED(ID, ACTIVO) composite
        assert schema.redundant_indexes == [
            ("EMPLEADOS", "ID", "IDX_EMP_0", "IDX_EMP_RED")
        ]


class TestOwner:
    """ALL_* views filtered by owner: any schema the connected user can see."""

    def test_every_dictionary_query_reads_all_views_filtered_by_owner(self):
        conn = FakeConnection()
        SchemaExtractor(conn, owner="HR").extract_all()  # type: ignore[arg-type]
        for sql, binds in zip(conn._cursor.executed, conn._cursor.binds, strict=True):
            assert "user_" not in sql.lower()
            assert binds.get("owner") == "HR", sql

    def test_without_owner_the_session_user_is_used(self):
        conn = FakeConnection(FakeCursor(session_user="APP"))
        extractor = SchemaExtractor(conn)  # type: ignore[arg-type]
        extractor.extract_all()
        assert extractor.owner == "APP"
        dictionary = [b for s, b in zip(conn._cursor.executed, conn._cursor.binds, strict=True) if "dual" not in s.lower()]
        assert all(b.get("owner") == "APP" for b in dictionary)

    def test_session_user_is_resolved_once(self):
        conn = FakeConnection()
        SchemaExtractor(conn).extract_all()  # type: ignore[arg-type]
        assert sum("dual" in s.lower() for s in conn._cursor.executed) == 1

    @pytest.mark.parametrize(
        ("given", "expected"),
        [("hr", "HR"), ("  Sales ", "SALES"), ('"MixedCase"', "MixedCase")],
    )
    def test_owner_follows_oracle_identifier_rules(self, given, expected):
        assert SchemaExtractor(FakeConnection(), owner=given).owner == expected  # type: ignore[arg-type]

    def test_foreign_keys_join_on_owner_and_follow_cross_schema_references(self):
        conn = FakeConnection()
        SchemaExtractor(conn, owner="HR").extract_all()  # type: ignore[arg-type]
        fk_sql = next(s.lower() for s in conn._cursor.executed if "r_constraint_name" in s.lower())
        assert "c.r_owner = c2.owner" in fk_sql
        assert "c.owner = cc.owner" in fk_sql


class TestCharLength:
    def test_char_length_is_read_and_zero_means_not_applicable(self, schema):
        """ALL_TAB_COLS reports char_length 0 for non-character types."""
        cfg = next(t for t in schema.tables if t.name == "CONFIGURACION")
        emp = next(t for t in schema.tables if t.name == "EMPLEADOS")
        clave = next(c for c in cfg.columns if c.name == "CLAVE")
        id_col = next(c for c in emp.columns if c.name == "ID")
        assert clave.char_length == 100
        assert clave.data_length == 400  # bytes, under character semantics
        assert id_col.char_length is None

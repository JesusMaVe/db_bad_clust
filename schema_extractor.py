"""
schema_extractor.py — Metadata extraction from Oracle Database

Purpose:
  Extract the full schema from an Oracle database: tables, columns,
  data types, constraints (PK, FK, Unique), indexes, and nullability.

  This module is the entry point for Phase 2 of the pipeline
  (ML on database schemas). The extracted data feeds the text
  preprocessing, BERT embedding, and clustering modules.

Flow:
  1. Connect to Oracle (reuses OracleConnector)
  2. Query data dictionary views (one bulk query per view):
     - user_tables
     - user_tab_cols (includes hidden/virtual/identity columns)
     - user_constraints / user_cons_columns
     - user_ind_columns
     - user_tab_comments / user_col_comments
  3. Populate nested dataclasses: DatabaseSchema -> TableMetadata -> ColumnMetadata
  4. Serialize to dict/JSON for consumption by other modules

Usage:
  extractor = SchemaExtractor(connection)
  schema = extractor.extract_all()
  for table in schema.tables:
      print(table.name, len(table.columns))
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

import oracledb

from exceptions import SchemaError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ColumnMetadata:
    """Metadata for a single database column."""

    name: str
    data_type: str
    nullable: bool
    data_length: int | None = None
    data_precision: int | None = None
    data_scale: int | None = None
    is_primary_key: bool = False
    is_foreign_key: bool = False
    is_unique: bool = False
    is_indexed: bool = False
    fk_references_table: str | None = None
    fk_references_column: str | None = None
    fk_name: str | None = None
    default_value: str | None = None
    comments: str | None = None
    is_identity: bool = False
    is_virtual: bool = False


@dataclass
class TableMetadata:
    """Metadata for a single database table."""

    name: str
    columns: list[ColumnMetadata] = field(default_factory=list)
    table_comment: str | None = None
    row_count_approx: int | None = None


@dataclass
class DatabaseSchema:
    """Container for the full extracted database schema."""

    tables: list[TableMetadata] = field(default_factory=list)
    # SchemaSpy-style signal: (table, column, single_index, composite_index)
    redundant_indexes: list[tuple[str, str, str, str]] = field(default_factory=list)

    def table_names(self) -> list[str]:
        """Return the list of table names."""
        return [t.name for t in self.tables]

    def total_columns(self) -> int:
        """Return the total number of columns across all tables."""
        return sum(len(t.columns) for t in self.tables)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the schema to a dictionary."""
        return {"tables": [asdict(t) for t in self.tables]}


# ---------------------------------------------------------------------------
# Schema Extractor
# ---------------------------------------------------------------------------


class SchemaExtractor:
    """
    Extracts schema metadata from an Oracle database.

    Queries the data dictionary views for the current user to build
    a complete DatabaseSchema with column types, constraints, indexes,
    and comments.

    Uses one bulk query per dictionary view (no per-table N+1): the
    same pattern as professional catalog tools like DataHub.
    """

    def __init__(self, connection: oracledb.Connection) -> None:
        self.connection = connection
        self.cursor = connection.cursor()

    # ── Main method ────────────────────────────────────────────────────

    def extract_all(self) -> DatabaseSchema:
        """
        Extract the full database schema.

        Returns:
            DatabaseSchema with all tables, columns, and metadata.

        Raises:
            SchemaError: If any query against the data dictionary fails.
        """
        try:
            table_names = self._get_table_names()
            if not table_names:
                return DatabaseSchema(tables=[])

            columns_by_table = self._get_all_columns()
            pk_by_table = self._get_all_constraints("P")
            unique_by_table = self._get_all_constraints("U")
            fk_by_table = self._get_all_foreign_keys()
            indexed_by_table = self._get_all_indexed_columns()
            table_comments = self._get_table_comments()
            col_comments = self._get_column_comments()

            tables: list[TableMetadata] = []
            for tname in table_names:
                columns = columns_by_table.get(tname, [])
                pk_cols = pk_by_table.get(tname, set())
                unique_cols = unique_by_table.get(tname, set())
                fk_info = fk_by_table.get(tname, {})
                indexed_cols = indexed_by_table.get(tname, set())

                for col in columns:
                    col.is_primary_key = col.name in pk_cols
                    col.is_foreign_key = col.name in fk_info
                    if col.is_foreign_key:
                        ref = fk_info[col.name]
                        col.fk_references_table = ref["ref_table"]
                        col.fk_references_column = ref["ref_column"]
                        col.fk_name = ref["fk_name"]
                    col.is_unique = col.name in unique_cols
                    col.is_indexed = col.name in indexed_cols
                    col.comments = col_comments.get((tname, col.name))

                tables.append(
                    TableMetadata(
                        name=tname,
                        columns=columns,
                        table_comment=table_comments.get(tname),
                    )
                )

            logger.info(
                "Schema extracted: %d tables, %d columns",
                len(tables),
                sum(len(t.columns) for t in tables),
            )
            return DatabaseSchema(
                tables=tables,
                redundant_indexes=self.get_redundant_indexes(),
            )
        except oracledb.Error as exc:
            raise SchemaError(f"Failed to extract schema: {exc}") from exc

    # ── Bulk data dictionary queries (one per view, grouped in Python) ─

    def _get_table_names(self) -> list[str]:
        """Return table names for the current user."""
        self.cursor.execute("SELECT table_name FROM user_tables ORDER BY table_name")
        return [row[0] for row in self.cursor.fetchall()]

    def _get_all_columns(self) -> dict[str, list[ColumnMetadata]]:
        """Return columns for all tables, grouped by table name.

        Uses user_tab_cols (not user_tab_columns) to also see hidden and
        virtual columns, plus identity flags — as DataHub does.
        """
        self.cursor.execute(
            """
            SELECT
                table_name,
                column_name,
                data_type,
                nullable,
                data_length,
                data_precision,
                data_scale,
                data_default,
                identity_column,
                virtual_column
            FROM user_tab_cols
            ORDER BY table_name, column_id
            """
        )
        result: dict[str, list[ColumnMetadata]] = {}
        for row in self.cursor.fetchall():
            col = ColumnMetadata(
                name=row[1],
                data_type=row[2],
                nullable=(row[3] == "Y"),
                data_length=row[4],
                data_precision=row[5],
                data_scale=row[6],
                default_value=row[7],
                is_identity=(row[8] == "YES"),
                is_virtual=(row[9] == "YES"),
            )
            result.setdefault(row[0], []).append(col)
        return result

    def _get_all_constraints(self, constraint_type: str) -> dict[str, set[str]]:
        """Return {table_name -> {column_name}} for PK or UNIQUE constraints."""
        self.cursor.execute(
            """
            SELECT c.table_name, cc.column_name
            FROM user_constraints c
            JOIN user_cons_columns cc
              ON c.constraint_name = cc.constraint_name
            WHERE c.constraint_type = :ctype
            """,
            ctype=constraint_type,
        )
        result: dict[str, set[str]] = {}
        for table_name, column_name in self.cursor.fetchall():
            result.setdefault(table_name, set()).add(column_name)
        return result

    def _get_all_foreign_keys(self) -> dict[str, dict[str, dict[str, str]]]:
        """Return {table_name -> {col_name -> {ref_table, ref_column, fk_name}}}."""
        self.cursor.execute(
            """
            SELECT
                c.table_name,
                cc.column_name,
                c2.table_name  AS ref_table,
                cc2.column_name AS ref_column,
                c.constraint_name AS fk_name
            FROM user_constraints c
            JOIN user_cons_columns cc
              ON c.constraint_name = cc.constraint_name
            JOIN user_constraints c2
              ON c.r_constraint_name = c2.constraint_name
            JOIN user_cons_columns cc2
              ON c2.constraint_name = cc2.constraint_name
             AND cc2.position = cc.position
            WHERE c.constraint_type = 'R'
            """
        )
        result: dict[str, dict[str, dict[str, str]]] = {}
        for table_name, col_name, ref_table, ref_column, fk_name in self.cursor.fetchall():
            result.setdefault(table_name, {})[col_name] = {
                "ref_table": ref_table,
                "ref_column": ref_column,
                "fk_name": fk_name,
            }
        return result

    def _get_all_indexed_columns(self) -> dict[str, set[str]]:
        """Return {table_name -> {column_name}} for indexed columns."""
        self.cursor.execute(
            "SELECT table_name, column_name FROM user_ind_columns"
        )
        result: dict[str, set[str]] = {}
        for table_name, column_name in self.cursor.fetchall():
            result.setdefault(table_name, set()).add(column_name)
        return result

    def get_redundant_indexes(self) -> list[tuple[str, str, str, str]]:
        """Return redundant index pairs: a composite index whose leading
        column already has a single-column index (SchemaSpy-style signal).

        Returns:
            List of (table_name, column_name, single_index, composite_index).
        """
        self.cursor.execute(
            """
            SELECT table_name, column_name, index_name, column_position
            FROM user_ind_columns
            ORDER BY table_name, index_name, column_position
            """
        )
        by_index: dict[tuple[str, str], list[tuple[str, int]]] = {}
        for table_name, column_name, index_name, position in self.cursor.fetchall():
            by_index.setdefault((table_name, index_name), []).append((column_name, position))

        # single-column indexes per table
        singles: dict[tuple[str, str], str] = {}
        for (tname, iname), cols in by_index.items():
            if len(cols) == 1:
                singles[(tname, cols[0][0])] = iname

        # composite indexes whose leading column has a single index
        redundant: list[tuple[str, str, str, str]] = []
        for (tname, iname), cols in by_index.items():
            if len(cols) > 1:
                leading = min(cols, key=lambda c: c[1])[0]
                single = singles.get((tname, leading))
                if single:
                    redundant.append((tname, leading, single, iname))
        return redundant

    def _get_table_comments(self) -> dict[str, str]:
        """Return {table_name -> comment} for tables that have one."""
        self.cursor.execute(
            """
            SELECT table_name, comments
            FROM user_tab_comments
            WHERE comments IS NOT NULL
            """
        )
        return {row[0]: row[1] for row in self.cursor.fetchall()}

    def _get_column_comments(self) -> dict[tuple[str, str], str]:
        """Return {(table_name, column_name) -> comment} for commented columns."""
        self.cursor.execute(
            """
            SELECT table_name, column_name, comments
            FROM user_col_comments
            WHERE comments IS NOT NULL
            """
        )
        return {(row[0], row[1]): row[2] for row in self.cursor.fetchall()}

    # ── Utility ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the internal cursor."""
        self.cursor.close()

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
  2. Query data dictionary views:
     - user_tables
     - user_tab_columns
     - user_constraints / user_cons_columns
     - user_ind_columns
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
    a complete DatabaseSchema with column types, constraints, and indexes.
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
            tables: list[TableMetadata] = []

            for tname in table_names:
                columns = self._get_columns(tname)
                pk_cols = self._get_primary_key_columns(tname)
                fk_info = self._get_foreign_keys(tname)
                unique_cols = self._get_unique_columns(tname)
                indexed_cols = self._get_indexed_columns(tname)

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

                tables.append(TableMetadata(name=tname, columns=columns))

            logger.info(
                "Schema extracted: %d tables, %d columns",
                len(tables),
                sum(len(t.columns) for t in tables),
            )
            return DatabaseSchema(tables=tables)
        except oracledb.Error as exc:
            raise SchemaError(f"Failed to extract schema: {exc}") from exc

    # ── Data dictionary queries ────────────────────────────────────────

    def _get_table_names(self) -> list[str]:
        """Return table names for the current user."""
        self.cursor.execute("SELECT table_name FROM user_tables ORDER BY table_name")
        return [row[0] for row in self.cursor.fetchall()]

    def _get_columns(self, table_name: str) -> list[ColumnMetadata]:
        """
        Return columns for a table with type, nullability, and length.
        """
        self.cursor.execute(
            """
            SELECT
                column_name,
                data_type,
                nullable,
                data_length,
                data_precision,
                data_scale,
                data_default
            FROM user_tab_columns
            WHERE table_name = :table_name
            ORDER BY column_id
            """,
            table_name=table_name,
        )
        columns: list[ColumnMetadata] = []
        for row in self.cursor.fetchall():
            col = ColumnMetadata(
                name=row[0],
                data_type=row[1],
                nullable=(row[2] == "Y"),
                data_length=row[3],
                data_precision=row[4],
                data_scale=row[5],
                default_value=row[6],
            )
            columns.append(col)
        return columns

    def _get_primary_key_columns(self, table_name: str) -> set[str]:
        """
        Return the set of column names that form the primary key.
        """
        self.cursor.execute(
            """
            SELECT cc.column_name
            FROM user_constraints c
            JOIN user_cons_columns cc
              ON c.constraint_name = cc.constraint_name
            WHERE c.table_name = :table_name
              AND c.constraint_type = 'P'
            """,
            table_name=table_name,
        )
        return {row[0] for row in self.cursor.fetchall()}

    def _get_foreign_keys(self, table_name: str) -> dict[str, dict[str, str]]:
        """
        Return a dict {col_name -> {ref_table, ref_column, fk_name}}
        for columns that are foreign keys.
        """
        self.cursor.execute(
            """
            SELECT
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
            WHERE c.table_name = :table_name
              AND c.constraint_type = 'R'
            """,
            table_name=table_name,
        )
        fk_map: dict[str, dict[str, str]] = {}
        for row in self.cursor.fetchall():
            fk_map[row[0]] = {
                "ref_table": row[1],
                "ref_column": row[2],
                "fk_name": row[3],
            }
        return fk_map

    def _get_unique_columns(self, table_name: str) -> set[str]:
        """
        Return the set of columns with a UNIQUE constraint.
        """
        self.cursor.execute(
            """
            SELECT cc.column_name
            FROM user_constraints c
            JOIN user_cons_columns cc
              ON c.constraint_name = cc.constraint_name
            WHERE c.table_name = :table_name
              AND c.constraint_type = 'U'
            """,
            table_name=table_name,
        )
        return {row[0] for row in self.cursor.fetchall()}

    def _get_indexed_columns(self, table_name: str) -> set[str]:
        """
        Return the set of columns that have an index.
        """
        self.cursor.execute(
            """
            SELECT column_name
            FROM user_ind_columns
            WHERE table_name = :table_name
            """,
            table_name=table_name,
        )
        return {row[0] for row in self.cursor.fetchall()}

    # ── Utility ────────────────────────────────────────────────────────

    def close(self) -> None:
        """Close the internal cursor."""
        self.cursor.close()

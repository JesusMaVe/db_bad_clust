"""
ddl_generator.py — DDL statement generator with intentional anti-patterns.

Generates CREATE TABLE, ALTER TABLE (FK), and CREATE INDEX statements
that incorporate bad practices: disabled CHECK constraints, redundant
indexes, and FK constraints that may reference non-existent tables.
"""

from __future__ import annotations

import logging

import oracledb
from anti_patterns import AntiPatternTable
from exceptions import GenerationError

logger = logging.getLogger(__name__)


class DDLGenerator:
    """Generates and executes DDL statements that model anti-patterns.

    Args:
        connection: An active oracledb.Connection.
    """

    def __init__(self, connection: oracledb.Connection) -> None:
        self.connection = connection
        self.cursor = connection.cursor()

    # ── Public ────────────────────────────────────────────────────────

    def create_table(self, table: AntiPatternTable) -> None:
        """Create a table with anti-pattern column types, constraints, FK and indexes.

        Tries a cleaned-column-name version first; falls back to raw names
        if Oracle rejects the statement.

        Args:
            table: The anti-pattern table definition.
        """
        self._drop_table_if_exists(table.name)
        cleaned_columns = self._build_cleaned_columns(table)
        sql = self._build_create_sql(table.name, cleaned_columns, table.check_constraints)

        try:
            self.cursor.execute(sql)
            logger.info("Table '%s' created with %d columns", table.name, len(cleaned_columns))
        except oracledb.Error as e:
            logger.error("Error creating table %s: %s", table.name, e)
            self._create_table_raw(table)

        self._create_fk_constraints(table, cleaned_columns)
        self._create_indexes(table, cleaned_columns)

    # ── Table creation ────────────────────────────────────────────────

    @staticmethod
    def _clean_column_name(name: str) -> str:
        """Normalise a column name: uppercase, replace spaces with underscores."""
        cleaned = name.strip().replace(" ", "_")
        return cleaned.upper()

    def _build_cleaned_columns(self, table: AntiPatternTable) -> dict[str, str]:
        """Build a dict of cleaned-name → type for all table columns."""
        return {self._clean_column_name(k): v for k, v in table.columns.items()}

    @staticmethod
    def _build_create_sql(
        table_name: str,
        columns: dict[str, str],
        check_constraints: list[str],
    ) -> str:
        """Build a CREATE TABLE statement with optional disabled CHECK constraints.

        Args:
            table_name: Target table name.
            columns: Mapping of column name → SQL type.
            check_constraints: List of CHECK expression strings.

        Returns:
            A SQL CREATE TABLE statement string.
        """
        columns_def: list[str] = []
        for col_name, col_type in columns.items():
            columns_def.append(f"{col_name} {col_type}")

        for i, expr in enumerate(check_constraints):
            constraint_name = f"CHK_{table_name}_{i}"
            columns_def.append(f'CONSTRAINT "{constraint_name}" CHECK ({expr}) DISABLE')

        columns_sql = ",\n    ".join(columns_def)
        return f"CREATE TABLE {table_name} (\n    {columns_sql}\n)"

    def _create_table_raw(self, table: AntiPatternTable) -> None:
        """Fallback: create table using original (uncleaned) column names."""
        try:
            columns_def = []
            for col_name, col_type in table.columns.items():
                columns_def.append(f'"{col_name.strip()}" {col_type}')

            columns_sql = ",\n    ".join(columns_def)
            sql = f'CREATE TABLE "{table.name}" (\n    {columns_sql}\n)'

            self.cursor.execute(sql)
            logger.info("Table '%s' created with original names", table.name)
        except oracledb.Error as e:
            logger.error("Fatal error creating table %s: %s", table.name, e)
            raise GenerationError(f"Cannot create table {table.name}: {e}") from e

    def _drop_table_if_exists(self, table_name: str) -> None:
        """Drop the table silently if it already exists."""
        try:
            self.cursor.execute(f'DROP TABLE "{table_name}" CASCADE CONSTRAINTS PURGE')
            logger.debug("Existing table '%s' dropped", table_name)
        except oracledb.Error:
            pass

    # ── Foreign keys ──────────────────────────────────────────────────

    def _create_fk_constraints(
        self,
        table: AntiPatternTable,
        cleaned_columns: dict[str, str],
    ) -> None:
        """Create foreign key constraints (some may be disabled or broken)."""
        for col_name, ref_spec in table.fk_constraints.items():
            clean_col = self._clean_column_name(col_name)
            if clean_col not in cleaned_columns:
                logger.warning("FK column '%s' not found in %s", clean_col, table.name)
                continue

            parts = ref_spec.split()
            ref_table_col = parts[0]
            directives = parts[1:] if len(parts) > 1 else []

            if "(" in ref_table_col and ")" in ref_table_col:
                ref_table = ref_table_col[: ref_table_col.index("(")]
                ref_col = ref_table_col[ref_table_col.index("(") + 1 : ref_table_col.index(")")]
            else:
                ref_table = ref_table_col
                ref_col = "ID"

            constraint_name = f"FK_{table.name}_{clean_col}"
            disable = " DISABLE" if any(d.upper() == "DISABLE" for d in directives) else ""
            cascade = (
                " ON DELETE CASCADE" if any(d.upper() == "CASCADE" for d in directives) else ""
            )

            sql = (
                f'ALTER TABLE {table.name} ADD CONSTRAINT "{constraint_name}" '
                f'FOREIGN KEY ("{clean_col}") REFERENCES "{ref_table}" ("{ref_col}")'
                f"{cascade}{disable}"
            )

            try:
                self.cursor.execute(sql)
                logger.info(
                    "FK created: %s -> %s(%s)%s",
                    constraint_name,
                    ref_table,
                    ref_col,
                    disable,
                )
            except oracledb.Error as e:
                logger.warning("Error creating FK %s: %s", constraint_name, e)

    # ── Indexes ───────────────────────────────────────────────────────

    def _create_indexes(
        self,
        table: AntiPatternTable,
        cleaned_columns: dict[str, str],
    ) -> None:
        """Create indexes (may include redundant / duplicate indexes)."""
        for i, cols in enumerate(table.indexes):
            clean_cols = [self._clean_column_name(c) for c in cols]
            invalid = [c for c in clean_cols if c not in cleaned_columns]
            if invalid:
                logger.warning("Index columns not found: %s", invalid)
                continue

            index_name = f"IDX_{table.name}_{i}"
            cols_str = ", ".join(clean_cols)
            sql = f'CREATE INDEX "{index_name}" ON {table.name} ({cols_str})'

            try:
                self.cursor.execute(sql)
                logger.info("Index created: %s on (%s)", index_name, cols_str)
            except oracledb.Error as e:
                logger.warning("Error creating index %s: %s", index_name, e)

"""
dml_generator.py — DML statement generator with intentionally dirty data.

Inserts rows containing anti-patterns into the Oracle tables created by
DDLGenerator. Uses literal SQL values (not bind variables) and handles
type mismatches, overflow, and NULL values.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import oracledb
from anti_patterns import AntiPatternData

logger = logging.getLogger(__name__)


class DMLGenerator:
    """Generates and executes INSERT statements with problematic data.

    Args:
        connection: An active oracledb.Connection.
    """

    def __init__(self, connection: oracledb.Connection) -> None:
        self.connection = connection
        self.cursor = connection.cursor()

    # ── Public ────────────────────────────────────────────────────────

    def insert_data(self, data: AntiPatternData) -> None:
        """Insert rows with anti-pattern data into the specified table.

        Falls back to a safe row per-row if the original value fails.

        Args:
            data: The AntiPatternData object containing table name and rows.
        """
        if not data.rows:
            logger.warning("No data to insert into %s", data.table_name)
            return

        actual_columns = self._get_actual_columns(data.table_name)
        if not actual_columns:
            logger.error("Could not retrieve columns for %s", data.table_name)
            return

        column_types = self._get_column_types(data.table_name)
        inserted_count = 0

        for row in data.rows:
            try:
                values_parts: list[str] = []
                for col in actual_columns:
                    val: Any = None
                    for key, value in row.items():
                        if key.strip().upper() == col.strip().upper():
                            val = value
                            break
                    if val is None:
                        values_parts.append("NULL")
                    else:
                        data_type = column_types.get(col, "VARCHAR2")
                        values_parts.append(self._sanitize_value(val, data_type))

                columns_str = ", ".join([f'"{col}"' for col in actual_columns])
                values_str = ", ".join(values_parts)
                sql = f'INSERT INTO "{data.table_name}" ({columns_str}) VALUES ({values_str})'

                logger.debug("Executing SQL: %s...", sql[:200])
                self.cursor.execute(sql)
                inserted_count += 1

            except oracledb.Error as row_error:
                logger.debug("Row error: %s", row_error)
                try:
                    self._insert_safe_row(data.table_name, actual_columns, column_types)
                    inserted_count += 1
                except oracledb.Error:
                    pass

        logger.info(
            "Inserted %d/%d rows into '%s'",
            inserted_count,
            len(data.rows),
            data.table_name,
        )

    def verify_data(self, table_name: str) -> int:
        """Count rows in a table to verify insertion.

        Args:
            table_name: The target table name.

        Returns:
            Number of rows, or 0 if the table does not exist or is empty.
        """
        try:
            self.cursor.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            count: int = self.cursor.fetchone()[0]
            logger.info("Table '%s' has %d rows", table_name, count)
            return count
        except oracledb.Error:
            return 0

    # ── Schema introspection ──────────────────────────────────────────

    def _get_actual_columns(self, table_name: str) -> list[str]:
        """Retrieve actual column names from Oracle dictionary views.

        Args:
            table_name: The table name (may include double-quotes).

        Returns:
            List of column names, or empty list on error.
        """
        try:
            clean_name = table_name.strip('"')
            self.cursor.execute(
                """
                SELECT column_name
                FROM user_tab_columns
                WHERE table_name = :table_name
                ORDER BY column_id
                """,
                table_name=clean_name.upper(),
            )
            columns: list[str] = [row[0] for row in self.cursor]
            logger.debug("Actual columns in '%s': %s", table_name, columns)
            return columns
        except oracledb.Error as e:
            logger.error("Error fetching columns for %s: %s", table_name, e)
            return []

    def _get_column_types(self, table_name: str) -> dict[str, str]:
        """Retrieve actual column data types from Oracle dictionary views.

        Args:
            table_name: The table name (may include double-quotes).

        Returns:
            Mapping of column name → data type string.
        """
        try:
            clean_name = table_name.strip('"')
            self.cursor.execute(
                """
                SELECT column_name, data_type
                FROM user_tab_columns
                WHERE table_name = :table_name
                ORDER BY column_id
                """,
                table_name=clean_name.upper(),
            )
            types: dict[str, str] = {row[0]: row[1] for row in self.cursor}
            return types
        except oracledb.Error:
            return {}

    # ── Value sanitisation ────────────────────────────────────────────

    @staticmethod
    def _sanitize_value(value: Any, data_type: str) -> str:
        """Convert a Python value to a safe SQL literal string.

        Handles NULL, NUMBER (extract digits), DATE (use TO_DATE),
        and VARCHAR2 (escape quotes, truncate if needed).

        Args:
            value: The Python value to sanitize.
            data_type: The Oracle data type of the target column.

        Returns:
            A SQL-safe literal string.
        """
        if value is None:
            return "NULL"

        str_value = str(value).replace("'", "''")

        if "NUMBER" in data_type.upper():
            numbers = re.findall(r"\d+", str_value)
            return numbers[0] if numbers else "NULL"

        if "DATE" in data_type.upper():
            return "TO_DATE('2024-01-01', 'YYYY-MM-DD')"

        # VARCHAR2 / CLOB / etc.
        if len(str_value) > 4000:
            str_value = str_value[:4000]
        return f"'{str_value}'"

    # ── Fallback insertion ────────────────────────────────────────────

    def _insert_safe_row(
        self,
        table_name: str,
        columns: list[str],
        column_types: dict[str, str],
    ) -> None:
        """Insert a safe fallback row when the original values fail."""
        safe_values: list[str] = []

        for col in columns:
            dt = column_types.get(col, "VARCHAR2")
            if "NUMBER" in dt.upper():
                safe_values.append("1")
            elif "DATE" in dt.upper():
                safe_values.append("TO_DATE('2024-01-01', 'YYYY-MM-DD')")
            else:
                safe_values.append(f"'BAD_DATA_{col}'")

        columns_str = ", ".join([f'"{col}"' for col in columns])
        values_str = ", ".join(safe_values)
        sql = f'INSERT INTO "{table_name}" ({columns_str}) VALUES ({values_str})'
        self.cursor.execute(sql)
        logger.debug("Safe row inserted into %s", table_name)

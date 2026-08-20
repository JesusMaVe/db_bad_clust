"""
ddl_generator.py — Generate executable Oracle DDL scripts from schema analysis.

Takes a DatabaseSchema (or a pickled one) and emits a SQL migration script
with one statement per detected anti-pattern. Statements are grouped by
table and tagged with severity / effort / impact. The script is read-only:
this module never executes DDL against the database.

Usage:
    .venv/bin/python ddl_generator.py --pickle output/intermediate_01.pkl
    .venv/bin/python ddl_generator.py --config config.yaml --output fixes.sql
"""

from __future__ import annotations

import argparse
import pickle
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from exceptions import GenerationError
from rule_engine import (
    DATE_KEYWORDS_HIGH,
    NUMBER_KEYWORDS,
    ColumnRuleEngine,
    Severity,
    TableRuleEngine,
)
from schema_extractor import DatabaseSchema


@dataclass
class DDLOperation:
    """A single DDL statement derived from a detection."""

    statement: str
    table_name: str
    issue_label: str
    severity: Severity
    effort: str
    impact: str
    comment: str = ""


_EFFORT_HIGH = {"giant_table", "eav", "polymorphic", "partition_candidate", "inconsistent_naming"}
_EFFORT_MEDIUM = {"self_referencing", "implicit_fk", "missing_pk", "wrong_data_types"}
_IMPACT_HIGH = {"missing_pk", "fk_without_index", "disabled_constraint", "stale_statistics", "impossible_data"}
_IMPACT_MEDIUM = {"wrong_data_types", "obsolete_type", "oversized_varchar", "reserved_words", "bad_boolean"}


_SQL_RESERVED_WORDS = {
    "SELECT", "FROM", "WHERE", "INSERT", "UPDATE", "DELETE", "DROP",
    "CREATE", "ALTER", "TABLE", "INDEX", "VIEW", "TRIGGER", "PROCEDURE",
    "FUNCTION", "PACKAGE", "BODY", "BEGIN", "END", "IF", "THEN", "ELSE",
    "LOOP", "FOR", "WHILE", "RETURN", "DECLARE", "EXCEPTION", "CURSOR",
    "OPEN", "CLOSE", "FETCH", "COMMIT", "ROLLBACK", "SAVEPOINT",
    "GRANT", "REVOKE", "EXECUTE", "DESCRIBE", "EXPLAIN", "PLAN",
    "NULL", "NOT", "AND", "OR", "IN", "EXISTS", "BETWEEN", "LIKE",
    "IS", "TRUE", "FALSE", "DEFAULT", "CHECK", "CONSTRAINT",
    "PRIMARY", "KEY", "FOREIGN", "REFERENCES", "UNIQUE", "INDEX",
    "ASC", "DESC", "ORDER", "GROUP", "HAVING", "DISTINCT", "UNION",
    "ALL", "INTERSECT", "MINUS", "CONNECT", "BY", "START", "WITH",
    "LEVEL", "ROWNUM", "ROWID", "SYSDATE", "UID", "USER",
    "VARCHAR", "VARCHAR2", "NUMBER", "DATE", "CHAR", "CLOB", "BLOB",
    "INTEGER", "INT", "FLOAT", "TIMESTAMP",
}


def _quote_if_reserved(name: str) -> str:
    if name.upper() in _SQL_RESERVED_WORDS:
        return f'"{name}"'
    return name


def _safe_identifier(base: str, max_len: int = 30) -> str:
    """Return an Oracle-safe identifier, truncated if necessary."""
    clean = re.sub(r"[^A-Za-z0-9_]", "_", base).upper()
    if len(clean) > max_len:
        return clean[:max_len]
    return clean


def _kw_match(name_lower: str, keyword: str) -> bool:
    """Token-boundary keyword match (mirrors rule_engine._kw_match)."""
    tokens = name_lower.replace("-", "_").split("_")
    return any(
        t == keyword or t.startswith(keyword) or t.endswith(keyword)
        for t in tokens
        if t
    )


def _has_number_keyword(name_lower: str) -> bool:
    """Mirror rule_engine number-keyword heuristic."""
    for kw in NUMBER_KEYWORDS:
        if _kw_match(name_lower, kw):
            return True
    return False


def _guess_target_type(col_name: str) -> str:
    """Guess DATE vs NUMBER from column name for wrong_data_types issues."""
    name_lower = col_name.lower()
    if any(_kw_match(name_lower, kw) for kw in DATE_KEYWORDS_HIGH):
        return "DATE"
    if _has_number_keyword(name_lower):
        return "NUMBER"
    return "NUMBER"  # fallback


class DDLGenerator:
    """Generate an Oracle DDL script from anti-pattern detections."""

    def __init__(self, include_comments: bool = True) -> None:
        self.include_comments = include_comments

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def from_schema(self, schema: DatabaseSchema) -> str:
        """Run all detectors on schema and render DDL."""
        results = self._collect_results(schema)
        return self.from_results(results)

    def from_results(self, results: list[Any]) -> str:
        """Render DDL from a list of ClassificationResult."""
        operations: list[DDLOperation] = []
        seen: set[str] = set()
        for r in results:
            for op in self._to_operations(r):
                if op.statement in seen:
                    continue
                seen.add(op.statement)
                operations.append(op)
        return self._render(operations)

    # ------------------------------------------------------------------
    # Detection aggregation
    # ------------------------------------------------------------------

    def _collect_results(self, schema: DatabaseSchema) -> list[Any]:
        col_engine = ColumnRuleEngine()
        table_engine = TableRuleEngine()
        all_columns = [c for t in schema.tables for c in t.columns]

        results: list[Any] = []
        results.extend(col_engine.classify(all_columns, schema))
        results.extend(table_engine.classify(schema))
        # Report-level signals (must stay out of classify)
        results.extend(col_engine.detect_implicit_fks(all_columns, schema))
        results.extend(col_engine.detect_obsolete_types(all_columns, schema))
        results.extend(col_engine.detect_oversized_varchars(all_columns, schema))
        results.extend(table_engine.detect_missing_pk(schema))
        results.extend(table_engine.detect_redundant_indexes(schema))
        results.extend(table_engine.detect_fk_without_index(schema))
        results.extend(table_engine.detect_stale_statistics(schema))
        results.extend(table_engine.detect_disabled_constraints(schema))
        results.extend(table_engine.detect_partition_candidates(schema))
        return results

    # ------------------------------------------------------------------
    # Result → DDL operation
    # ------------------------------------------------------------------

    def _to_operations(self, result: Any) -> list[DDLOperation]:
        label = result.predicted_label
        table = result.table_name
        column = result.column_name
        fn = _DDL_HANDLERS.get(label, _handle_generic)
        return fn(result, table, column)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, operations: list[DDLOperation]) -> str:
        lines: list[str] = [
            "-- Oracle remediation script generated by db_bad_clust",
            f"-- Generated at: {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "-- Review before executing. Dangerous statements are commented out.",
            "",
        ]

        if not operations:
            lines.append("-- No actionable issues detected.")
            return "\n".join(lines)

        by_table: dict[str, list[DDLOperation]] = {}
        for op in operations:
            by_table.setdefault(op.table_name, []).append(op)

        summary: dict[str, int] = {}
        for op in operations:
            summary[op.issue_label] = summary.get(op.issue_label, 0) + 1

        lines.append("-- Summary")
        for label, count in sorted(summary.items(), key=lambda x: -x[1]):
            lines.append(f"--   {label}: {count}")
        lines.append("")

        for table_name in sorted(by_table):
            lines.append(f"-- {'='*60}")
            lines.append(f"-- Table: {table_name}")
            lines.append(f"-- {'='*60}")
            for op in by_table[table_name]:
                if op.comment:
                    lines.append(f"-- {op.comment}")
                lines.append(
                    f"-- [severity={op.severity.value}, effort={op.effort}, impact={op.impact}]"
                )
                lines.append(op.statement)
                lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-label handlers
# ---------------------------------------------------------------------------


def _op(
    statement: str,
    table: str,
    label: str,
    severity: Severity,
    comment: str = "",
) -> DDLOperation:
    effort = "high" if label in _EFFORT_HIGH else ("medium" if label in _EFFORT_MEDIUM else "low")
    impact = "high" if label in _IMPACT_HIGH else ("medium" if label in _IMPACT_MEDIUM else "low")
    return DDLOperation(
        statement=statement,
        table_name=table,
        issue_label=label,
        severity=severity,
        effort=effort,
        impact=impact,
        comment=comment,
    )


def _handle_date_as_text(result: Any, table: str, column: str) -> list[DDLOperation]:
    col = _quote_if_reserved(column)
    return [_op(
        f"ALTER TABLE {table} MODIFY ({col} DATE);",
        table, "date_as_text", result.severity,
        f"Review data cleanliness before changing {column} to DATE.",
    )]


def _handle_number_as_text(result: Any, table: str, column: str) -> list[DDLOperation]:
    col = _quote_if_reserved(column)
    return [_op(
        f"ALTER TABLE {table} MODIFY ({col} NUMBER);",
        table, "number_as_text", result.severity,
        f"Review data cleanliness before changing {column} to NUMBER.",
    )]


def _handle_wrong_data_types(result: Any, table: str, column: str) -> list[DDLOperation]:
    # Table-level wrong_data_types may join multiple columns with ", ".
    cols = [c.strip() for c in column.split(",")]
    ops: list[DDLOperation] = []
    for c in cols:
        target = _guess_target_type(c)
        quoted = _quote_if_reserved(c)
        ops.append(_op(
            f"ALTER TABLE {table} MODIFY ({quoted} {target});",
            table, "wrong_data_types", result.severity,
            f"Guessed target type {target} from column name; validate data first.",
        ))
    return ops


def _handle_reserved_words(result: Any, table: str, column: str) -> list[DDLOperation]:
    new_name = _safe_identifier(f"{column}_COL")
    old_col = _quote_if_reserved(column)
    return [_op(
        f"ALTER TABLE {table} RENAME COLUMN {old_col} TO {new_name};",
        table, "reserved_words", result.severity,
        f"Column {column} is a SQL reserved word.",
    )]


def _handle_bad_boolean(result: Any, table: str, column: str) -> list[DDLOperation]:
    col = _quote_if_reserved(column)
    chk = _safe_identifier(f"{table}_CHK_{column}")
    return [_op(
        f"ALTER TABLE {table} MODIFY ({col} NUMBER(1));\n"
        f"ALTER TABLE {table} ADD CONSTRAINT {chk} CHECK ({col} IN (0, 1));",
        table, "bad_boolean", result.severity,
    )]


def _handle_impossible_data(result: Any, table: str, column: str) -> list[DDLOperation]:
    col = _quote_if_reserved(column)
    return [_op(
        f"ALTER TABLE {table} MODIFY ({col} NOT NULL);",
        table, "impossible_data", result.severity,
        f"Primary-key column {column} should not allow NULL.",
    )]


def _handle_self_referencing(result: Any, table: str, column: str) -> list[DDLOperation]:
    return [_op(
        f"-- Self-referencing column {column} in {table}: verify hierarchy intent.",
        table, "self_referencing", result.severity, "Manual review required.",
    )]


def _handle_polymorphic(result: Any, table: str, column: str) -> list[DDLOperation]:
    return [_op(
        f"-- Polymorphic table {table}: split into one table per type or add per-type FKs.",
        table, "polymorphic", result.severity, "Manual refactor required.",
    )]


def _handle_giant_table(result: Any, table: str, column: str) -> list[DDLOperation]:
    return [_op(
        f"-- Giant table {table}: split into smaller, focused tables.",
        table, "giant_table", result.severity, "Manual refactor required.",
    )]


def _handle_eav(result: Any, table: str, column: str) -> list[DDLOperation]:
    return [_op(
        f"-- EAV table {table}: normalize to relational columns or use JSON/XML.",
        table, "eav", result.severity, "Manual refactor required.",
    )]


def _handle_inconsistent_naming(result: Any, table: str, column: str) -> list[DDLOperation]:
    return [_op(
        f"-- Table {table} has inconsistent naming; standardize columns.",
        table, "inconsistent_naming", result.severity, "Manual rename campaign required.",
    )]


def _handle_implicit_fk(result: Any, table: str, column: str) -> list[DDLOperation]:
    return [_op(
        f"-- Implicit FK {column} in {table}: add FOREIGN KEY referencing target table.",
        table, "implicit_fk", result.severity, "Manual reference resolution required.",
    )]


def _handle_missing_pk(result: Any, table: str, column: str) -> list[DDLOperation]:
    # If the table has an ID-like column, propose it; otherwise leave as comment.
    candidate = "ID"
    return [_op(
        f"ALTER TABLE {table} ADD CONSTRAINT {_safe_identifier(f'PK_{table}')} PRIMARY KEY ({candidate});",
        table, "missing_pk", result.severity,
        f"Verify {candidate} is unique and not nullable before enabling.",
    )]


def _handle_redundant_index(result: Any, table: str, column: str) -> list[DDLOperation]:
    meta = result.metadata or {}
    single = meta.get("single_index") or "REDUNDANT_IDX"
    return [_op(
        f"DROP INDEX {single};",
        table, "redundant_index", result.severity,
        "Keep the composite index instead.",
    )]


def _handle_fk_without_index(result: Any, table: str, column: str) -> list[DDLOperation]:
    meta = result.metadata or {}
    cols = meta.get("fk_columns") or [column]
    quoted = ", ".join(_quote_if_reserved(c) for c in cols)
    idx_name = _safe_identifier(f"IDX_{table}_{'_'.join(cols)}")
    return [_op(
        f"CREATE INDEX {idx_name} ON {table} ({quoted});",
        table, "fk_without_index", result.severity,
        f"Covering index for FK {meta.get('fk_name', '')}.",
    )]


def _handle_stale_statistics(result: Any, table: str, column: str) -> list[DDLOperation]:
    return [_op(
        f"EXEC DBMS_STATS.GATHER_TABLE_STATS(ownname => USER, tabname => '{table}');",
        table, "stale_statistics", result.severity,
    )]


def _handle_disabled_constraint(result: Any, table: str, column: str) -> list[DDLOperation]:
    meta = result.metadata or {}
    name = meta.get("constraint_name", "UNKNOWN")
    validated = meta.get("validated")
    if validated != "VALIDATED":
        stmt = f"ALTER TABLE {table} ENABLE VALIDATE CONSTRAINT {name};"
    else:
        stmt = f"ALTER TABLE {table} ENABLE CONSTRAINT {name};"
    return [_op(stmt, table, "disabled_constraint", result.severity)]


def _handle_obsolete_type(result: Any, table: str, column: str) -> list[DDLOperation]:
    meta = result.metadata or {}
    source = (meta.get("source_type") or "LONG").upper()
    target = "BLOB" if "RAW" in source else "CLOB"
    col = _quote_if_reserved(column)
    return [_op(
        f"ALTER TABLE {table} MODIFY ({col} {target});",
        table, "obsolete_type", result.severity,
        f"Convert deprecated {source} to {target}.",
    )]


def _handle_partition_candidate(result: Any, table: str, column: str) -> list[DDLOperation]:
    meta = result.metadata or {}
    date_cols = ", ".join(meta.get("date_columns", [column]))
    return [_op(
        f"-- Partition candidate {table}: consider PARTITION BY RANGE ({date_cols}).",
        table, "partition_candidate", result.severity,
        "Requires new table design and data migration.",
    )]


def _handle_oversized_varchar(result: Any, table: str, column: str) -> list[DDLOperation]:
    meta = result.metadata or {}
    size = meta.get("suggested_size", 100)
    col = _quote_if_reserved(column)
    return [_op(
        f"ALTER TABLE {table} MODIFY ({col} VARCHAR2({size}));",
        table, "oversized_varchar", result.severity,
        f"Suggested size {size}; validate actual data length first.",
    )]


def _handle_generic(result: Any, table: str, column: str) -> list[DDLOperation]:
    return [_op(
        f"-- {result.predicted_label}: {result.explanation}",
        table, result.predicted_label, result.severity,
    )]


_DDL_HANDLERS: dict[str, Callable[[Any, str, str], list[DDLOperation]]] = {
    "date_as_text": _handle_date_as_text,
    "number_as_text": _handle_number_as_text,
    "wrong_data_types": _handle_wrong_data_types,
    "reserved_words": _handle_reserved_words,
    "bad_boolean": _handle_bad_boolean,
    "impossible_data": _handle_impossible_data,
    "self_referencing": _handle_self_referencing,
    "polymorphic": _handle_polymorphic,
    "giant_table": _handle_giant_table,
    "eav": _handle_eav,
    "inconsistent_naming": _handle_inconsistent_naming,
    "implicit_fk": _handle_implicit_fk,
    "missing_pk": _handle_missing_pk,
    "redundant_index": _handle_redundant_index,
    "fk_without_index": _handle_fk_without_index,
    "stale_statistics": _handle_stale_statistics,
    "disabled_constraint": _handle_disabled_constraint,
    "obsolete_type": _handle_obsolete_type,
    "partition_candidate": _handle_partition_candidate,
    "oversized_varchar": _handle_oversized_varchar,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _load_schema_from_pickle(path: Path) -> DatabaseSchema:
    with path.open("rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, DatabaseSchema):
        return obj
    if isinstance(obj, dict) and "schema" in obj:
        schema = obj["schema"]
        if isinstance(schema, DatabaseSchema):
            return schema
    raise GenerationError(
        f"Pickled object is not a DatabaseSchema or pipeline dict: {type(obj)}"
    )


def _load_schema_from_db(config_path: str) -> DatabaseSchema:
    from db_connector import OracleConnector
    from schema_extractor import SchemaExtractor

    connector = OracleConnector(config_path)
    connection = connector.connect()
    try:
        extractor = SchemaExtractor(connection)
        return extractor.extract_all()
    finally:
        connection.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Oracle DDL remediation script from schema analysis."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pickle",
        type=str,
        help="Path to a pickled DatabaseSchema produced by the pipeline.",
    )
    source.add_argument(
        "--config",
        type=str,
        help="Path to a YAML config with database credentials (extracts live schema).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/remediation.sql",
        help="Path to write the generated SQL script.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.pickle:
        schema = _load_schema_from_pickle(Path(args.pickle))
    else:
        schema = _load_schema_from_db(args.config)

    sql = DDLGenerator().from_schema(schema)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(sql, encoding="utf-8")

    print(f"Wrote remediation script: {out_path}")
    print(f"Statements generated: {sql.count('ALTER TABLE') + sql.count('CREATE INDEX') + sql.count('DROP INDEX') + sql.count('EXEC DBMS_STATS')}")


if __name__ == "__main__":
    main()

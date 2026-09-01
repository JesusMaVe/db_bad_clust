"""
schema_adapter.py — Anti-pattern catalog to DatabaseSchema.

Converts the hand-written catalog in `anti_patterns.py` into the same
`DatabaseSchema` shape the Oracle extractor produces, so the rule engine can be
run against the catalog without a database connection.

This is the seam between the catalog (what a bad schema looks like) and
detection (what the engine sees). `ground_truth.build_ground_truth_map()` is
its caller.

Usage:
    from db_bad_clust.generation.schema_adapter import to_database_schema

    schema = to_database_schema(generate_poorly_designed_tables())
"""

from __future__ import annotations

import re
from typing import Any

from db_bad_clust.data.schema_extractor import ColumnMetadata, DatabaseSchema, TableMetadata
from db_bad_clust.generation.anti_patterns import AntiPatternTable


def parse_data_length(data_type: str) -> int | None:
    """Extract the length/precision size from an Oracle type string."""
    m = re.search(r"\((\d+)\)", data_type)
    if m:
        return int(m.group(1))
    if data_type.upper().startswith("DATE"):
        return 7
    if data_type.upper() in {"NUMBER", "LONG", "LONG RAW"}:
        return 22
    return None


def parse_fk_reference(expr: str) -> dict[str, Any] | None:
    """Parse an fk_constraints value like 'TABLE(COL) [ON DELETE ...] [DISABLE]'."""
    pattern = re.compile(
        r"^\s*([A-Z_][A-Z0-9_]*)\s*\(\s*([A-Z_][A-Z0-9_]*)\s*\)"
        r"(?:\s+(ON\s+DELETE\s+(?:CASCADE|SET\s+NULL|NO\s+ACTION)))?"
        r"(?:\s+(DISABLE))?\s*$",
        re.IGNORECASE,
    )
    m = pattern.match(expr)
    if not m:
        return None
    return {
        "ref_table": m.group(1),
        "ref_column": m.group(2),
        "on_delete": m.group(3) or "",
        "disabled": bool(m.group(4)),
    }


def to_database_schema(tables: list[AntiPatternTable]) -> DatabaseSchema:
    """Convert catalog tables into a DatabaseSchema.

    Populates the metadata fields the rule engine reads (fk_columns,
    index_columns, table_statistics, constraint_status) so the full engine can
    run against the catalog without needing Oracle.
    """
    schema_tables: list[TableMetadata] = []
    fk_columns: dict[str, dict[str, list[str]]] = {}
    index_columns: dict[str, dict[str, list[str]]] = {}
    table_statistics: dict[str, dict[str, Any]] = {}
    constraint_status: dict[str, list[dict[str, Any]]] = {}

    for table in tables:
        columns: list[ColumnMetadata] = []
        for col_name, data_type in table.columns.items():
            is_pk = col_name == "ID" and not table.no_primary_key
            columns.append(
                ColumnMetadata(
                    name=col_name,
                    data_type=data_type,
                    nullable=not is_pk,
                    data_length=parse_data_length(data_type),
                    is_primary_key=is_pk,
                    is_foreign_key=col_name in table.fk_constraints,
                )
            )

        schema_tables.append(
            TableMetadata(name=table.name, columns=columns, row_count_approx=None)
        )

        # FKs ordered by constraint name
        if table.fk_constraints:
            fk_columns[table.name] = {}
            for col_name, ref_expr in table.fk_constraints.items():
                parsed = parse_fk_reference(ref_expr)
                fk_name = f"FK_{table.name}_{col_name}"[:30]
                fk_columns[table.name][fk_name] = [col_name]
                if parsed and parsed.get("disabled"):
                    constraint_status.setdefault(table.name, []).append(
                        {
                            "name": fk_name,
                            "type": "R",
                            "status": "DISABLED",
                            "validated": "NOT VALIDATED",
                        }
                    )

        # Indexes ordered by index name
        if table.indexes:
            index_columns[table.name] = {}
            for i, idx_cols in enumerate(table.indexes):
                index_columns[table.name][f"IDX_{table.name}_{i + 1}"[:30]] = list(idx_cols)

        table_statistics[table.name] = {
            "stale_stats": False,
            "num_rows": 0,
            "last_analyzed": None,
        }

        # Disabled check constraints
        for i, check_expr in enumerate(table.check_constraints):
            if check_expr.upper().rstrip().endswith(" DISABLE"):
                constraint_status.setdefault(table.name, []).append(
                    {
                        "name": f"{table.name}_CHK{i + 1}"[:30],
                        "type": "C",
                        "status": "DISABLED",
                        "validated": "NOT VALIDATED",
                    }
                )

    return DatabaseSchema(
        tables=schema_tables,
        fk_columns=fk_columns,
        index_columns=index_columns,
        table_statistics=table_statistics,
        constraint_status=constraint_status,
    )

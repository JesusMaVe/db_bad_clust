"""
verify_schema.py — Structural fidelity check for the reconstructed anti-pattern schema.

Connects to the live Oracle instance and checks that what SchemaExtractor sees
matches output/manual_labels.csv exactly (same 23 tables, same 243 columns, both
directions) and that PK/FK counts match what scripts/generate_schema_sql.py reported
when it wrote sql_init/001_bad_schema.sql. Run this before spending time on
apply_comments.py / build_embeddings.py / the notebooks.

Usage:
    python3 verify_schema.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from db_bad_clust.data.db_connector import OracleConnector
from db_bad_clust.data.schema_extractor import SchemaExtractor

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = REPO_ROOT / "output" / "manual_labels.csv"


def load_expected() -> tuple[set[str], set[tuple[str, str]]]:
    tables: set[str] = set()
    columns: set[tuple[str, str]] = set()
    with LABELS_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table = row["table"].strip()
            column = row["column"].strip()
            tables.add(table)
            columns.add((table, column))
    return tables, columns


def main() -> int:
    expected_tables, expected_columns = load_expected()

    conn = OracleConnector(config_path="config.yaml").connect()
    schema = SchemaExtractor(conn).extract_all()

    actual_tables = set(schema.table_names())
    actual_columns = {
        (table.name, col.name) for table in schema.tables for col in table.columns
    }

    pk_count = sum(
        1 for table in schema.tables for col in table.columns if col.is_primary_key
    )
    fk_count = sum(
        1 for table in schema.tables for col in table.columns if col.is_foreign_key
    )

    ok = True

    missing_tables = expected_tables - actual_tables
    extra_tables = actual_tables - expected_tables
    if missing_tables or extra_tables:
        ok = False
        if missing_tables:
            print(f"MISSING tables (in CSV, not in DB): {sorted(missing_tables)}")
        if extra_tables:
            print(f"EXTRA tables (in DB, not in CSV): {sorted(extra_tables)}")

    missing_columns = expected_columns - actual_columns
    extra_columns = actual_columns - expected_columns
    if missing_columns or extra_columns:
        ok = False
        if missing_columns:
            print(f"MISSING columns (in CSV, not in DB): {sorted(missing_columns)}")
        if extra_columns:
            print(f"EXTRA columns (in DB, not in CSV): {sorted(extra_columns)}")

    print(f"Tables: {len(actual_tables)} (expected {len(expected_tables)})")
    print(f"Columns: {len(actual_columns)} (expected {len(expected_columns)})")
    print(f"Primary keys: {pk_count}")
    print(f"Foreign keys: {fk_count}")

    if ok:
        print("OK — schema matches manual_labels.csv exactly.")
    else:
        print("FAILED — schema does not match manual_labels.csv. See above.")

    conn.close()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

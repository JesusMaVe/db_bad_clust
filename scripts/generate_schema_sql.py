"""
generate_schema_sql.py — Bootstrap DDL generator for the synthetic anti-pattern schema.

Reads output/manual_labels.csv (the only surviving spec of the 23-table / 243-column
research corpus — the original Oracle schema was created ad-hoc and never versioned,
see AGENTS.md "Oracle — no es reproducible al 100% desde el repo") and emits
sql_init/001_bad_schema.sql: CREATE TABLE + FK ALTER TABLE statements, no INSERTs.

This is a one-time bootstrap for THIS synthetic labeled corpus, not part of the
reusable pipeline. SchemaExtractor/TextPreprocessor/build_embeddings.py are schema-
agnostic already and need no changes to point at a different Oracle database.

Usage:
    python3 generate_schema_sql.py
"""

from __future__ import annotations

import csv
from collections import OrderedDict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS_PATH = REPO_ROOT / "output" / "manual_labels.csv"
OUTPUT_PATH = REPO_ROOT / "sql_init" / "001_bad_schema.sql"

DB_USER = "bad_schema"
DB_PASSWORD = "bad_schema_pass"
DB_SERVICE = "localhost:1521/FREEPDB1"

EXCLUDED_PK_FK_LABELS = {"impossible_data", "inconsistent_naming"}

TYPE_MAP = {
    "VARCHAR2": "VARCHAR2(255)",
    "NVARCHAR2": "NVARCHAR2(255)",
    "CHAR": "CHAR(1)",
    "RAW": "RAW(255)",
    "NUMBER": "NUMBER",
    "DATE": "DATE",
    "CLOB": "CLOB",
    "BLOB": "BLOB",
}


def load_columns() -> OrderedDict[str, list[dict[str, str]]]:
    tables: OrderedDict[str, list[dict[str, str]]] = OrderedDict()
    with LABELS_PATH.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            table = row["table"].strip()
            tables.setdefault(table, []).append(
                {
                    "name": row["column"].strip(),
                    "data_type": row["data_type"].strip(),
                    "label": row["label"].strip(),
                }
            )
    return tables


def choose_pk(columns: list[dict[str, str]]) -> str | None:
    eligible = [
        c
        for c in columns
        if c["data_type"] == "NUMBER" and c["label"] not in EXCLUDED_PK_FK_LABELS
    ]
    for c in eligible:
        if c["name"] == "ID":
            return c["name"]
    for c in eligible:
        if c["name"].startswith("ID_") or c["name"].endswith("_ID"):
            return c["name"]
    return None


def find_target_table(prefix: str, self_table: str, pk_map: dict[str, str]) -> str | None:
    prefix_u = prefix.upper()
    for table, pk in pk_map.items():
        if table == self_table or pk is None:
            continue
        table_u = table.upper()
        if (
            table_u == prefix_u
            or table_u == prefix_u + "S"
            or (table_u.endswith("S") and table_u[:-1] == prefix_u)
        ):
            return table
    return None


def choose_fks(
    table: str, columns: list[dict[str, str]], pk_col: str | None, pk_map: dict[str, str]
) -> list[tuple[str, str, str]]:
    """Returns list of (column_name, target_table, target_pk_column)."""
    fks: list[tuple[str, str, str]] = []
    for c in columns:
        if c["name"] == pk_col or c["data_type"] != "NUMBER":
            continue
        if c["label"] in EXCLUDED_PK_FK_LABELS:
            continue
        if c["label"] == "self_referencing":
            if pk_col:
                fks.append((c["name"], table, pk_col))
            continue
        if c["name"].endswith("_ID"):
            prefix = c["name"][: -len("_ID")]
            target = find_target_table(prefix, table, pk_map)
            if target:
                fks.append((c["name"], target, pk_map[target]))
    return fks


def render_create_table(table: str, columns: list[dict[str, str]], pk_col: str | None) -> str:
    lines = []
    for c in columns:
        ddl_type = TYPE_MAP[c["data_type"]]
        col_line = f'    "{c["name"]}" {ddl_type}'
        if c["name"] == pk_col:
            col_line += f' CONSTRAINT PK_{table} PRIMARY KEY'
        lines.append(col_line)
    body = ",\n".join(lines)
    return f'CREATE TABLE "{table}" (\n{body}\n);'


def render_fk(table: str, column: str, target_table: str, target_pk: str) -> str:
    constraint = f"FK_{table}_{column}"
    return (
        f'ALTER TABLE "{table}" ADD CONSTRAINT {constraint}\n'
        f'    FOREIGN KEY ("{column}") REFERENCES "{target_table}" ("{target_pk}");'
    )


def main() -> None:
    tables = load_columns()

    pk_map: dict[str, str | None] = {t: choose_pk(cols) for t, cols in tables.items()}

    fk_statements: list[str] = []
    fk_count = 0
    for table, cols in tables.items():
        for column, target_table, target_pk in choose_fks(table, cols, pk_map[table], pk_map):
            fk_statements.append(render_fk(table, column, target_table, target_pk))
            fk_count += 1

    create_statements = [
        render_create_table(table, cols, pk_map[table]) for table, cols in tables.items()
    ]

    total_columns = sum(len(cols) for cols in tables.values())
    pk_count = sum(1 for pk in pk_map.values() if pk is not None)

    header = f"""WHENEVER SQLERROR EXIT SQL.SQLCODE
CONNECT {DB_USER}/{DB_PASSWORD}@{DB_SERVICE}

-- Generado por scripts/generate_schema_sql.py a partir de output/manual_labels.csv.
-- Bootstrap de una sola vez del corpus sintetico de investigacion
-- ({len(tables)} tablas / {total_columns} columnas). No representa el esquema Oracle
-- original (creado ad-hoc, nunca versionado — ver AGENTS.md). PK/FK/longitudes son
-- una reconstruccion fiel en nombres y tipos, inventada en constraints.
-- Sin INSERTs: el pipeline solo lee metadata de catalogo, nunca contenido de filas.

-- ============================================================
-- Fase 1: tablas ({pk_count} con PK, {len(tables) - pk_count} sin PK)
-- ============================================================
"""

    footer = "\nEXIT\n"

    sql = (
        header
        + "\n\n".join(create_statements)
        + "\n\n-- ============================================================\n"
        + f"-- Fase 2: foreign keys ({fk_count} en total)\n"
        + "-- ============================================================\n\n"
        + "\n\n".join(fk_statements)
        + footer
    )

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(sql, encoding="utf-8")

    print(f"Wrote {OUTPUT_PATH.relative_to(REPO_ROOT)}")
    print(f"Tables: {len(tables)}  Columns: {total_columns}")
    print(f"Primary keys: {pk_count}  Foreign keys: {fk_count}")


if __name__ == "__main__":
    main()

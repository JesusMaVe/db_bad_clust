"""
schema_generator.py — Synthetic Oracle schema generator for benchmarking.

Generates arbitrary-size Oracle schemas with planted anti-patterns and
emits:
  1. A DDL script (.sql) ready to review/run against an Oracle instance.
  2. A manifest (.json) with the planted ground truth per table/column.

Usage:
    .venv/bin/python schema_generator.py --tables 100 --output output/generated
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from db_bad_clust.data.schema_extractor import ColumnMetadata, DatabaseSchema, TableMetadata
from db_bad_clust.exceptions import GenerationError
from db_bad_clust.generation.anti_patterns import AntiPatternTable
from db_bad_clust.rules.rule_engine import SQL_RESERVED_WORDS


@dataclass
class GeneratedAntiPatternTable(AntiPatternTable):
    """AntiPatternTable extended with Phase-2 Oracle detector flags."""

    fk_without_index: bool = False
    stale_statistics: bool = False
    disabled_constraints: bool = False
    obsolete_types: bool = False
    partition_candidate: bool = False
    partition_row_count: int = 0
    oversized_varchars: dict[str, int] = field(default_factory=dict)


@dataclass
class SchemaGeneratorConfig:
    """Configuration for synthetic schema generation."""

    num_tables: int = 100
    seed: int | None = 42
    anti_pattern_density: float = 0.4
    include_report_signals: bool = True


_Template = Callable[[random.Random, int], AntiPatternTable]


def _clean_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Small clean reference table; occasionally missing a PK."""
    return GeneratedAntiPatternTable(
        name=f"DEP_{idx:04d}",
        columns={
            "ID": "NUMBER",
            "NOMBRE": f"VARCHAR2({rng.randint(50, 200)})",
            "FECHA_CREACION": "DATE",
        },
        no_primary_key=rng.random() < 0.25,
    )


def _employee_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Dates/numbers stored as text + boolean-as-text (wrong_data_types/self_contradictory)."""
    return GeneratedAntiPatternTable(
        name=f"EMP_{idx:04d}",
        columns={
            "ID": "NUMBER",
            "NOMBRE_EMPLEADO": "VARCHAR2(100)",
            "FECHA_NACIMIENTO": "VARCHAR2(20)",
            "SALARIO": "VARCHAR2(50)",
            "DEPARTAMENTO_ID": "NUMBER",
            "EMAIL": "DATE",
            "ACTIVO": "VARCHAR2(10)",
        },
        no_primary_key=True,
        wrong_data_types=True,
        self_contradictory=True,
        fk_constraints={"DEPARTAMENTO_ID": "DEPARTAMENTOS(ID) ON DELETE CASCADE"},
    )


def _orders_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Transaction table with text numeric/date columns and an implicit FK."""
    return GeneratedAntiPatternTable(
        name=f"ORD_{idx:04d}",
        columns={
            "ORDER_ID": "NUMBER",
            "FECHA_ORDEN": "VARCHAR2(50)",
            "CLIENTE_ID": "NUMBER",
            "PRODUCTO_ID": "NUMBER",
            "CANTIDAD": "VARCHAR2(10)",
            "PRECIO": "VARCHAR2(20)",
            "NOTAS": "CLOB",
            "ESTADO": "NUMBER",
        },
        no_primary_key=True,
        wrong_data_types=True,
    )


def _audit_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Log table with a date stored as text."""
    return GeneratedAntiPatternTable(
        name=f"AUD_{idx:04d}",
        columns={
            "ID": "NUMBER",
            "TABLA_AFECTADA": "VARCHAR2(50)",
            "REGISTRO_ID": "NUMBER",
            "ACCION": "VARCHAR2(20)",
            "USUARIO": "VARCHAR2(50)",
            "FECHA": "VARCHAR2(30)",
            "DETALLE": "VARCHAR2(4000)",
        },
        no_primary_key=True,
        wrong_data_types=True,
    )


def _report_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Table using SQL reserved words as column names."""
    return GeneratedAntiPatternTable(
        name=f"RPT_{idx:04d}",
        columns={
            "ID": "NUMBER",
            "FROM": "VARCHAR2(100)",
            "WHERE": "VARCHAR2(100)",
            "SELECT": "VARCHAR2(4000)",
            "NULL": "VARCHAR2(200)",
            "TIMESTAMP": "VARCHAR2(30)",
        },
        no_primary_key=True,
        reserved_word_columns=True,
    )


def _eav_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Entity-Attribute-Value configuration table."""
    return GeneratedAntiPatternTable(
        name=f"CFG_{idx:04d}",
        columns={
            "ID": "NUMBER",
            "CLAVE": "VARCHAR2(100)",
            "VALOR": "VARCHAR2(4000)",
            "TIPO_DATO": "VARCHAR2(20)",
            "ACTIVO": "CHAR(1)",
            "ULTIMA_MODIFICACION": "VARCHAR2(30)",
        },
        no_primary_key=True,
        eav_antipattern=True,
    )


def _polymorphic_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Polymorphic table with a type discriminator and generic value columns."""
    return GeneratedAntiPatternTable(
        name=f"POL_{idx:04d}",
        columns={
            "ID": "NUMBER",
            "TIPO_REGISTRO": "VARCHAR2(20)",
            "NOMBRE_O_DESCRIPCION": "VARCHAR2(4000)",
            "CANT_O_PRECIO": "VARCHAR2(4000)",
            "FECHA_O_DIRECCION": "VARCHAR2(4000)",
            "ESTADO_O_ACTIVO": "VARCHAR2(4000)",
            "FLAG_S_N": "CHAR(1)",
            "FLAG_1_0": "VARCHAR2(3)",
        },
        no_primary_key=True,
        polymorphic=True,
        wrong_data_types=True,
    )


def _hierarchy_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Self-referencing hierarchy table."""
    return GeneratedAntiPatternTable(
        name=f"HIE_{idx:04d}",
        columns={
            "ID": "NUMBER",
            "NOMBRE": "VARCHAR2(100)",
            "CAT_PADRE_ID": "NUMBER",
            "NIVEL": "VARCHAR2(10)",
            "ACTIVO": "CHAR(1)",
            "RUTA_COMPLETA": "VARCHAR2(1000)",
        },
        no_primary_key=True,
        self_referencing=True,
        wrong_data_types=True,
    )


def _giant_dump_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Wide table with generic/inconsistent column names."""
    columns: dict[str, str] = {"ID": "NUMBER"}
    letters = [chr(ord("A") + i) for i in range(26)]
    for i in range(35):
        prefix = letters[i // 26] if i >= 26 else ""
        col = f"COL_{prefix}{letters[i % 26]}"
        columns[col] = "VARCHAR2(4000)"
    columns["TEMP_DATA"] = "CLOB"
    return GeneratedAntiPatternTable(
        name=f"DMP_{idx:04d}",
        columns=columns,
        no_primary_key=True,
        inconsistent_naming=True,
        giant_table=True,
    )


def _backup_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Backup/volume table with many numbered columns."""
    columns = {"ID": "NUMBER"}
    for i in range(1, 41):
        columns[f"COL_{i:03d}"] = "VARCHAR2(4000)"
    return GeneratedAntiPatternTable(
        name=f"BAK_{idx:04d}",
        columns=columns,
        no_primary_key=True,
        inconsistent_naming=True,
        giant_table=True,
    )


def _fk_without_index_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Child table with FK constraint but no index on the FK column."""
    return GeneratedAntiPatternTable(
        name=f"CHD_{idx:04d}",
        columns={
            "ID": "NUMBER",
            "NOMBRE": "VARCHAR2(100)",
            "DEPARTAMENTO_ID": "NUMBER",
        },
        no_primary_key=True,
        fk_constraints={"DEPARTAMENTO_ID": "DEPARTAMENTOS(ID)"},
        indexes=[
            ["NOMBRE"],  # index exists, but not on the FK column
        ],
        fk_without_index=True,
    )


def _stale_stats_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Table flagged as having stale/missing Optimizer statistics."""
    return GeneratedAntiPatternTable(
        name=f"STT_{idx:04d}",
        columns={
            "ID": "NUMBER",
            "NOMBRE": "VARCHAR2(100)",
            "FECHA_ALTA": "DATE",
        },
        no_primary_key=True,
        stale_statistics=True,
    )


def _disabled_constraint_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Table with a disabled check constraint and a disabled FK."""
    return GeneratedAntiPatternTable(
        name=f"DIS_{idx:04d}",
        columns={
            "ID": "NUMBER",
            "ESTADO": "VARCHAR2(20)",
            "DEPARTAMENTO_ID": "NUMBER",
        },
        no_primary_key=True,
        fk_constraints={"DEPARTAMENTO_ID": "DEPARTAMENTOS(ID) DISABLE"},
        check_constraints=["ESTADO IN ('A', 'I') DISABLE"],
        disabled_constraints=True,
    )


def _obsolete_type_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Table using deprecated LONG / LONG RAW / RAW columns."""
    return GeneratedAntiPatternTable(
        name=f"OBS_{idx:04d}",
        columns={
            "ID": "NUMBER",
            "TEXTO_LARGO": "LONG",
            "BINARIO": "LONG RAW",
            "CODIGO_HEX": f"RAW({rng.randint(100, 2000)})",
            "FECHA": "DATE",
        },
        no_primary_key=True,
        obsolete_types=True,
    )


def _partition_candidate_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Large table with a date column, candidate for partitioning."""
    return GeneratedAntiPatternTable(
        name=f"PAR_{idx:04d}",
        columns={
            "ID": "NUMBER",
            "FECHA_EVENTO": "DATE",
            "DESCRIPCION": "VARCHAR2(500)",
            "CANTIDAD": "NUMBER",
        },
        no_primary_key=True,
        partition_candidate=True,
        partition_row_count=2_000_000,
    )


def _oversized_varchar_table(rng: random.Random, idx: int) -> AntiPatternTable:
    """Short-domain columns declared with oversized VARCHAR2."""
    return GeneratedAntiPatternTable(
        name=f"OVS_{idx:04d}",
        columns={
            "ID": "NUMBER",
            "FLAG_ACTIVO": "VARCHAR2(4000)",
            "CODIGO_ESTADO": "VARCHAR2(2000)",
            "TIPO_REGISTRO": "VARCHAR2(1000)",
            "DESCRIPCION": "VARCHAR2(4000)",
        },
        no_primary_key=True,
        oversized_varchars={
            "FLAG_ACTIVO": 1,
            "CODIGO_ESTADO": 5,
            "TIPO_REGISTRO": 10,
        },
    )


# Templates weighted roughly: 60% anti-pattern tables, 40% clean reference tables.
_DIRTY_TEMPLATES: list[_Template] = [
    _employee_table,
    _orders_table,
    _audit_table,
    _report_table,
    _eav_table,
    _polymorphic_table,
    _hierarchy_table,
    _giant_dump_table,
    _backup_table,
    _fk_without_index_table,
    _stale_stats_table,
    _disabled_constraint_table,
    _obsolete_type_table,
    _partition_candidate_table,
    _oversized_varchar_table,
]
_ALL_TEMPLATES: list[_Template] = [_clean_table, *_DIRTY_TEMPLATES]


def _parse_fk_reference(expr: str) -> dict[str, Any] | None:
    """Parse a simple fk_constraints value like 'TABLE(COL) [ON DELETE ...] [DISABLE]'."""
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


def _quote_if_reserved(name: str) -> str:
    """Quote identifiers that are Oracle reserved words."""
    return f'"{name}"' if name.upper() in SQL_RESERVED_WORDS else name


class SyntheticSchemaGenerator:
    """Generate synthetic Oracle schemas with planted anti-patterns."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self, config: SchemaGeneratorConfig | None = None
    ) -> tuple[list[AntiPatternTable], dict[str, Any]]:
        """Generate tables and a ground-truth manifest.

        Returns:
            (tables, manifest) where manifest is a JSON-serializable dict.
        """
        if config is None:
            config = SchemaGeneratorConfig()

        tables: list[AntiPatternTable] = []
        for i in range(config.num_tables):
            if self.rng.random() < config.anti_pattern_density:
                template = self.rng.choice(_DIRTY_TEMPLATES)
            else:
                template = _clean_table
            tables.append(template(self.rng, i))

        manifest = self._build_manifest(tables, config)
        return tables, manifest

    def to_ddl(self, tables: list[AntiPatternTable]) -> str:
        """Render CREATE TABLE / INDEX / CONSTRAINT DDL for Oracle."""
        lines: list[str] = [
            "-- Synthetic schema generated for db_bad_clust benchmarking",
            f"-- Generated at: {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
            "-- Review before executing; this script does NOT drop objects.",
            "",
        ]

        for table in tables:
            lines.extend(self._render_table_ddl(table))
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------

    def _build_manifest(
        self, tables: list[AntiPatternTable], config: SchemaGeneratorConfig
    ) -> dict[str, Any]:
        """Build manifest with ground-truth labels aligned to the rule engine."""
        from db_bad_clust.rules.rule_engine import classify as classify_schema

        table_entries: list[dict[str, Any]] = []
        total_columns = 0
        label_counts: dict[str, int] = {}

        # Column-level labels come from the same classify() merge the rule engine uses,
        # so the benchmark measures the detectors against their own label semantics.
        schema = SyntheticSchemaGenerator.to_database_schema(tables)
        classify_results = classify_schema(schema=schema)
        classify_labels = {
            f"{r.table_name}.{r.column_name}".upper(): r.predicted_label
            for r in classify_results
        }

        for table in tables:
            columns: list[dict[str, Any]] = []
            for col_name, data_type in table.columns.items():
                total_columns += 1
                key = f"{table.name}.{col_name}".upper()
                label = classify_labels.get(key, "clean")
                label_counts[label] = label_counts.get(label, 0) + 1
                columns.append(
                    {
                        "name": col_name,
                        "data_type": data_type,
                        "anti_pattern": label,
                    }
                )

            table_flags = [
                flag
                for flag, present in {
                    "no_primary_key": table.no_primary_key,
                    "inconsistent_naming": table.inconsistent_naming,
                    "reserved_word_columns": table.reserved_word_columns,
                    "wrong_data_types": table.wrong_data_types,
                    "self_referencing": table.self_referencing,
                    "polymorphic": table.polymorphic,
                    "giant_table": table.giant_table,
                    "eav_antipattern": table.eav_antipattern,
                    "impossible_data": table.impossible_data,
                    "self_contradictory": table.self_contradictory,
                }.items()
                if present
            ]

            report_flags = {}
            if isinstance(table, GeneratedAntiPatternTable):
                report_flags = {
                    "fk_without_index": table.fk_without_index,
                    "stale_statistics": table.stale_statistics,
                    "disabled_constraints": table.disabled_constraints,
                    "obsolete_types": table.obsolete_types,
                    "partition_candidate": table.partition_candidate,
                    "partition_row_count": table.partition_row_count,
                    "oversized_varchars": table.oversized_varchars,
                }

            table_entries.append(
                {
                    "name": table.name,
                    "anti_patterns": table_flags,
                    "report_flags": report_flags,
                    "columns": columns,
                    "has_redundant_indexes": len(table.indexes) > 1
                    and any(len(idx) > 1 for idx in table.indexes),
                }
            )

        return {
            "metadata": {
                "generated_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "num_tables": len(tables),
                "seed": config.seed,
                "anti_pattern_density": config.anti_pattern_density,
            },
            "summary": {
                "total_columns": total_columns,
                "column_label_counts": label_counts,
            },
            "tables": table_entries,
        }

    # ------------------------------------------------------------------
    # DDL rendering
    # ------------------------------------------------------------------

    def _render_table_ddl(self, table: AntiPatternTable) -> list[str]:
        lines: list[str] = []
        pk_cols: list[str] = []
        inline_constraints: list[str] = []

        # Columns
        col_lines: list[str] = []
        for col_name, data_type in table.columns.items():
            quoted = _quote_if_reserved(col_name)
            nullable = "NOT NULL" if col_name == "ID" else "NULL"
            col_lines.append(f"    {quoted} {data_type} {nullable}")

            if col_name == "ID" and not table.no_primary_key:
                pk_cols.append(quoted)

        # Inline check constraints
        for i, check_expr in enumerate(table.check_constraints):
            expr = check_expr
            disabled = ""
            if expr.upper().rstrip().endswith(" DISABLE"):
                expr = expr[: -len(" DISABLE")].rstrip()
                disabled = " DISABLE"
            inline_constraints.append(
                f"    CONSTRAINT {table.name}_CHK{i + 1} CHECK ({expr}){disabled}"
            )

        # Inline FK constraints
        for col_name, ref_expr in table.fk_constraints.items():
            parsed = _parse_fk_reference(ref_expr)
            if not parsed:
                raise GenerationError(
                    f"Cannot parse FK reference '{ref_expr}' on {table.name}.{col_name}"
                )
            ref_table = _quote_if_reserved(parsed["ref_table"])
            ref_column = _quote_if_reserved(parsed["ref_column"])
            quoted_col = _quote_if_reserved(col_name)
            fk_name = f"FK_{table.name}_{col_name}"[:30]
            on_delete = f" {parsed['on_delete']}" if parsed["on_delete"] else ""
            disabled = " DISABLE" if parsed["disabled"] else ""
            inline_constraints.append(
                f"    CONSTRAINT {fk_name} FOREIGN KEY ({quoted_col})"
                f" REFERENCES {ref_table}({ref_column}){on_delete}{disabled}"
            )

        all_constraints = col_lines + inline_constraints
        if pk_cols:
            pk_name = f"PK_{table.name}"[:30]
            all_constraints.append(
                f"    CONSTRAINT {pk_name} PRIMARY KEY ({', '.join(pk_cols)})"
            )

        lines.append(f"CREATE TABLE {table.name} (")
        lines.append(",\n".join(all_constraints))
        lines.append(");")

        # Indexes
        for i, idx_cols in enumerate(table.indexes):
            idx_name = f"IDX_{table.name}_{i + 1}"[:30]
            quoted_cols = ", ".join(_quote_if_reserved(c) for c in idx_cols)
            lines.append(
                f"CREATE INDEX {idx_name} ON {table.name} ({quoted_cols});"
            )

        # Explicit redundant-index pair: single-col index on ID plus composite starting with ID.
        # ponytail: report-level signal kept simple; real redundancy scanning stays in SchemaExtractor.
        if len(table.indexes) > 1 and any(len(idx) > 1 for idx in table.indexes):
            single_name = f"IDX_{table.name}_R1"[:30]
            comp_name = f"IDX_{table.name}_R2"[:30]
            col_names = list(table.columns.keys())
            first_col = col_names[0]
            second_col = col_names[1]
            lines.append(
                f"CREATE INDEX {single_name} ON {table.name} ({_quote_if_reserved(first_col)});"
            )
            lines.append(
                f"CREATE INDEX {comp_name} ON {table.name} ({_quote_if_reserved(first_col)}, {_quote_if_reserved(second_col)});"
            )

        return lines

    # ------------------------------------------------------------------
    # Conversion to DatabaseSchema (for benchmark/evaluation)
    # ------------------------------------------------------------------

    @staticmethod
    def to_database_schema(tables: list[AntiPatternTable]) -> DatabaseSchema:
        """Convert generated AntiPatternTable objects into a DatabaseSchema.

        Populates the Phase-2 metadata fields (fk_columns, index_columns,
        table_statistics, constraint_status) so the full rule engine can run
        against a synthetic schema without needing Oracle.
        """
        schema_tables: list[TableMetadata] = []
        fk_columns: dict[str, dict[str, list[str]]] = {}
        index_columns: dict[str, dict[str, list[str]]] = {}
        table_statistics: dict[str, dict[str, Any]] = {}
        constraint_status: dict[str, list[dict[str, Any]]] = {}

        for table in tables:
            columns: list[ColumnMetadata] = []
            for col_name, data_type in table.columns.items():
                length = _parse_data_length(data_type)
                is_pk = col_name == "ID" and not table.no_primary_key
                columns.append(
                    ColumnMetadata(
                        name=col_name,
                        data_type=data_type,
                        nullable=not is_pk,
                        data_length=length,
                        is_primary_key=is_pk,
                        is_foreign_key=col_name in table.fk_constraints,
                    )
                )

            row_count = 0
            if isinstance(table, GeneratedAntiPatternTable):
                if table.partition_candidate:
                    row_count = table.partition_row_count
                elif table.stale_statistics:
                    row_count = 10_000

            schema_tables.append(
                TableMetadata(
                    name=table.name,
                    columns=columns,
                    row_count_approx=row_count if row_count > 0 else None,
                )
            )

            # FKs ordered by constraint name
            if table.fk_constraints:
                fk_columns[table.name] = {}
                for col_name, ref_expr in table.fk_constraints.items():
                    parsed = _parse_fk_reference(ref_expr)
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
                    idx_name = f"IDX_{table.name}_{i + 1}"[:30]
                    index_columns[table.name][idx_name] = list(idx_cols)

            # Table statistics
            if isinstance(table, GeneratedAntiPatternTable) and table.stale_statistics:
                table_statistics[table.name] = {
                    "stale_stats": True,
                    "num_rows": 10_000,
                    "last_analyzed": None,
                }
            elif isinstance(table, GeneratedAntiPatternTable) and table.partition_candidate:
                table_statistics[table.name] = {
                    "stale_stats": False,
                    "num_rows": table.partition_row_count,
                    "last_analyzed": None,
                }
            else:
                table_statistics[table.name] = {
                    "stale_stats": False,
                    "num_rows": row_count,
                    "last_analyzed": None,
                }

            # Disabled check constraints
            for i, check_expr in enumerate(table.check_constraints):
                if check_expr.upper().rstrip().endswith(" DISABLE"):
                    chk_name = f"{table.name}_CHK{i + 1}"[:30]
                    constraint_status.setdefault(table.name, []).append(
                        {
                            "name": chk_name,
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_data_length(data_type: str) -> int | None:
    """Extract the length/precision size from an Oracle type string."""
    m = re.search(r"\((\d+)\)", data_type)
    if m:
        return int(m.group(1))
    if data_type.upper().startswith("DATE"):
        return 7
    if data_type.upper() in {"NUMBER", "LONG", "LONG RAW"}:
        return 22
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Oracle schema for benchmarking."
    )
    parser.add_argument(
        "--tables", type=int, default=100, help="Number of tables to generate."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.4,
        help="Probability of generating a dirty table (0.0 - 1.0).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/generated",
        help="Directory to write generated_schema.sql and manifest.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config = SchemaGeneratorConfig(
        num_tables=args.tables,
        seed=args.seed,
        anti_pattern_density=args.density,
    )

    generator = SyntheticSchemaGenerator(seed=config.seed)
    tables, manifest = generator.generate(config)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    sql_path = out_dir / "generated_schema.sql"
    sql_path.write_text(generator.to_ddl(tables), encoding="utf-8")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"Generated {len(tables)} tables, {manifest['summary']['total_columns']} columns")
    print(f"  DDL: {sql_path}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()

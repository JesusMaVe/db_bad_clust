"""
ground_truth.py — Anti-pattern ground truth for H1 validation

Purpose:
  Map each column from the anti-pattern tables to its primary
  anti-pattern category at COLUMN level.

  Column-level rules:
    - wrong_data_types: Column stored as VARCHAR2 but name suggests numeric/date
    - reserved_words: Column name is a SQL reserved word
    - self_contradictory: Boolean column stored as VARCHAR2/CHAR
    - impossible_data: Primary key that allows NULL
    - self_referencing: FK referencing own table
    - polymorphic: Type indicator column
    - giant_table: Table with >20 columns (all columns flagged)
    - inconsistent_naming: Table with mixed naming conventions (minority flagged)
    - eav: Entity-Attribute-Value pattern (all columns flagged)
    - clean: No anti-pattern detected

Usage:
  from ground_truth import get_ground_truth, build_ground_truth_map

  gt_map = build_ground_truth_map()
  truth_labels = get_ground_truth(column_table_map, column_names, gt_map)
"""

from __future__ import annotations

import logging
import re

from anti_patterns import AntiPatternTable, generate_poorly_designed_tables
from structural_encoder import ORACLE_TYPE_MAP

logger = logging.getLogger(__name__)

# Keywords for detecting wrong data types (same as rule_engine.py)
DATE_KEYWORDS_HIGH = {
    "fecha", "fec", "fch", "fh",
    "date", "time", "timestamp",
    "dia", "mes", "anio", "ano",
    "hor", "hora",
    "created", "updated", "modified",
    "start", "end", "begin",
    "inicio", "fin",
    "alta", "baja", "apertura", "cierre",
    "f_alta", "f_baja", "f_mod", "f_cre",
    "creado", "modificado",
}

NUMBER_KEYWORDS_HIGH = {
    "precio", "costo", "salario", "sueldo",
    "importe", "monto", "cantidad",
    "presupuesto",
    "intento", "intentos",
}

NUMBER_KEYWORDS_MEDIUM = {
    "id", "fk", "pk",
    "num", "number", "amt", "amount",
    "count", "qty",
    "total", "sum", "suma",
    "price", "cost", "salary",
    "peso", "kg", "lt", "mts",
    "porcentaje", "tasa", "rate",
    "fee", "tarifa", "comision",
    "saldo", "balance", "deuda",
    "credito", "cargo", "abono",
    "valor", "monto_total", "importe_total",
    "presup",
    "nivel", "level",
    "intento", "intentos",
    "orden", "order",
    "stock", "existencia",
}

NUMBER_KEYWORDS_EXCEPTIONS = {
    "codigo_postal", "codigo_serie", "numero_serie",
    "numero_documento", "codigo_documento",
}

BOOLEAN_KEYWORDS_WORD = {
    "flag", "ind", "active", "enabled",
    "activo", "habilitado",
    "valid", "valido",
    "aprobado", "approved",
    "bloqueado", "blocked",
    "eliminado", "deleted",
    "existente", "existing",
}

BOOLEAN_PREFIX_PATTERNS = {
    "es_", "has_", "is_",
}

DATE_KEYWORDS = DATE_KEYWORDS_HIGH
NUMBER_KEYWORDS = NUMBER_KEYWORDS_HIGH | NUMBER_KEYWORDS_MEDIUM
BOOLEAN_KEYWORDS = BOOLEAN_KEYWORDS_WORD | BOOLEAN_PREFIX_PATTERNS


def _has_boolean_keyword(name_lower: str) -> bool:
    """Check if column name contains a boolean keyword using word boundaries."""
    words = name_lower.replace("-", "_").split("_")
    for word in words:
        if word in BOOLEAN_KEYWORDS_WORD:
            return True
    for prefix in BOOLEAN_PREFIX_PATTERNS:
        if name_lower.startswith(prefix) and len(name_lower) > len(prefix):
            return True
    return False


def _kw_match(name_lower: str, keyword: str) -> bool:
    """Token-boundary keyword match (same as rule_engine._kw_match)."""
    tokens = name_lower.replace("-", "_").split("_")
    return any(
        t == keyword or t.startswith(keyword) or t.endswith(keyword)
        for t in tokens
        if t
    )


def _has_number_keyword(name_lower: str) -> bool:
    """Check if column name contains a number keyword, with exceptions."""
    if name_lower in NUMBER_KEYWORDS_EXCEPTIONS:
        return False
    for kw in NUMBER_KEYWORDS_HIGH:
        if _kw_match(name_lower, kw):
            return True
    words = name_lower.replace("-", "_").split("_")
    for word in words:
        if word in NUMBER_KEYWORDS_MEDIUM:
            return True
    return False

POLYMORPHIC_TYPE_NAMES = {
    "TYPE", "TIPO", "DISCRIMINATOR", "KIND", "OBJECTION_TYPE",
    "PAYMENT_TYPE", "DOCUMENT_TYPE", "RECORD_TYPE", "ENTITY_TYPE",
    "TIPO_REGISTRO", "TIPO_DATO", "TIPO_DOCUMENTO",
}

SELF_REF_NAMES = {
    "parent_id", "padre_id", "superior_id", "jefe_id", "manager_id",
    "father_id", "boss_id", "supervisor_id", "referencia_id",
}

GIANT_TABLE_THRESHOLD = 20

LABEL_CLEAN = "clean"
LABEL_UNKNOWN = None


def _get_column_anti_pattern(col_name: str, data_type: str, table: AntiPatternTable) -> str:
    """Determine the anti-pattern for a single column.

    Priority order (highest first):
      polymorphic > eav > reserved_words > self_referencing > giant_table
      > inconsistent_naming > self_contradictory > impossible_data
      > wrong_data_types > clean

    Args:
        col_name: Column name.
        data_type: Column data type (e.g., "VARCHAR2(20)").
        table: The parent AntiPatternTable.

    Returns:
        Anti-pattern label string.
    """
    # Check for polymorphic columns (highest priority)
    if table.polymorphic and col_name.upper() in POLYMORPHIC_TYPE_NAMES:
        return "polymorphic"

    # Check for EAV pattern (table-level, all columns flagged)
    if table.eav_antipattern:
        return "eav"

    # Check for reserved words
    if table.reserved_word_columns and col_name.upper() in {
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
    }:
        return "reserved_words"

    # Check for self-referencing (FK-like names in self-ref tables)
    if table.self_referencing:
        name_lower = col_name.lower()
        if name_lower in SELF_REF_NAMES or (name_lower.endswith("_id") and "padre" in name_lower):
            return "self_referencing"

    # Check for giant table (table-level, all columns flagged)
    if table.giant_table:
        return "giant_table"

    # Check for inconsistent naming (table-level, all columns flagged)
    if table.inconsistent_naming:
        return "inconsistent_naming"

    # Check for bad boolean (self_contradictory)
    if table.self_contradictory:
        col_type = data_type.upper().split("(")[0].split(" ")[0]
        mapped_type = ORACLE_TYPE_MAP.get(col_type, col_type)
        text_types = set(ORACLE_TYPE_MAP.get(t, t) for t in ["VARCHAR", "CHAR"])
        if mapped_type in text_types:
            name_lower = col_name.lower()
            if _has_boolean_keyword(name_lower):
                return "self_contradictory"

    # Check for impossible data (PK with NULL)
    if table.impossible_data:
        if col_name.upper() == "ID":
            return "impossible_data"

    # Check for wrong data types (column-level, ANY table: the catalog
    # deliberately mistypes columns in tables without the table-level flag)
    col_type = data_type.upper().split("(")[0].split(" ")[0]
    mapped_type = ORACLE_TYPE_MAP.get(col_type, col_type)
    text_types = set(ORACLE_TYPE_MAP.get(t, t) for t in ["VARCHAR", "CHAR"])

    if mapped_type in text_types:
        name_lower = col_name.lower()
        is_date = any(_kw_match(name_lower, kw) for kw in DATE_KEYWORDS_HIGH)
        is_number = _has_number_keyword(name_lower)
        if is_date or is_number:
            return "wrong_data_types"

    return LABEL_CLEAN


def build_ground_truth_map() -> dict[str, str]:
    """Build a mapping from TABLE.COLUMN to anti-pattern label at COLUMN level.

    Returns:
        Dict keyed by "TABLE_NAME.COLUMN_NAME" with anti-pattern label.
    """
    tables: list[AntiPatternTable] = generate_poorly_designed_tables()
    gt: dict[str, str] = {}

    for table in tables:
        for col_name, data_type in table.columns.items():
            key: str = f"{table.name}.{col_name}".upper()
            label: str = _get_column_anti_pattern(col_name, data_type, table)
            gt[key] = label

    return gt


def get_ground_truth(
    column_table_map: list[str],
    column_names: list[str] | None = None,
    gt_map: dict[str, str] | None = None,
) -> list[str | None]:
    """Build ground truth label list matching the pipeline's column order.

    Args:
        column_table_map: List of table names per column (index-parallel).
        column_names: Optional list of "TABLE.COLUMN" names. If provided,
            these are used for lookup directly.
        gt_map: Pre-built ground truth map (built once and cached).

    Returns:
        List of ground truth labels (str) or None for unknown columns.
    """
    if gt_map is None:
        gt_map = build_ground_truth_map()

    truth: list[str | None] = []
    if column_names:
        for name in column_names:
            truth.append(gt_map.get(name.upper(), LABEL_UNKNOWN))
    else:
        for i, table_name in enumerate(column_table_map):
            truth.append(LABEL_UNKNOWN)

    return truth

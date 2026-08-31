"""
ground_truth.py — Anti-pattern ground truth for H1 validation

Purpose:
  Map each column from the anti-pattern tables to its primary
  anti-pattern category at COLUMN level.

  The labels are produced by the same rule engine used for detection
  (rule_engine.classify) applied to the catalog tables converted to
  DatabaseSchema, so ground truth and detection share label semantics.

Usage:
  from db_bad_clust.generation.ground_truth import get_ground_truth, build_ground_truth_map

  gt_map = build_ground_truth_map()
  truth_labels = get_ground_truth(column_table_map, column_names, gt_map)
"""

from __future__ import annotations

import csv
from pathlib import Path

from db_bad_clust.generation.anti_patterns import AntiPatternTable, generate_poorly_designed_tables

LABEL_CLEAN = "clean"
LABEL_UNKNOWN = None

# Rule-engine labels -> manual vocabulary (identity for labels that already match).
RULE_TO_MANUAL = {
    "date_as_text": "wrong_data_types",
    "number_as_text": "wrong_data_types",
    "bad_boolean": "self_contradictory",
    "reserved_word": "reserved_words",
    "impossible_data": "impossible_data",
    "self_referencing": "self_referencing",
    "polymorphic": "polymorphic",
    "giant_table": "giant_table",
    "eav_pattern": "eav",
    "inconsistent_naming": "inconsistent_naming",
}

MANUAL_LABEL_VOCABULARY = {
    "clean",
    "wrong_data_types",
    "reserved_words",
    "self_contradictory",
    "impossible_data",
    "self_referencing",
    "polymorphic",
    "giant_table",
    "inconsistent_naming",
    "eav",
}


def load_manual_ground_truth(path: str | Path) -> dict[str, str]:
    """Load manually labeled ground truth from a CSV (see label_export.py).

    CSV columns: table,column,data_type,label. Labels must be in
    MANUAL_LABEL_VOCABULARY and every row must be filled.

    Returns:
        Dict keyed by "TABLE_NAME.COLUMN_NAME" with the label.
    """
    gt: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8") as f:
        for i, row in enumerate(csv.DictReader(f), start=2):
            label = (row.get("label") or "").strip().lower()
            if not label:
                raise ValueError(f"Empty label at CSV row {i}: {row['table']}.{row['column']}")
            if label not in MANUAL_LABEL_VOCABULARY:
                raise ValueError(f"Invalid label '{label}' at CSV row {i} (allowed: {sorted(MANUAL_LABEL_VOCABULARY)})")
            gt[f"{row['table']}.{row['column']}".upper()] = label
    if not gt:
        raise ValueError(f"No rows in {path}")
    return gt


def build_ground_truth_map() -> dict[str, str]:
    """Build a mapping from TABLE.COLUMN to anti-pattern label at COLUMN level.

    The labels are produced by running rule_engine.classify on the catalog
    tables converted to DatabaseSchema, keeping ground truth aligned with the
    detector's label semantics.

    Returns:
        Dict keyed by "TABLE_NAME.COLUMN_NAME" with anti-pattern label.
    """
    from db_bad_clust.generation.schema_generator import SyntheticSchemaGenerator
    from db_bad_clust.rules.rule_engine import classify as classify_schema

    tables: list[AntiPatternTable] = generate_poorly_designed_tables()
    schema = SyntheticSchemaGenerator.to_database_schema(tables)
    results = classify_schema(schema=schema)

    gt: dict[str, str] = {}
    for r in results:
        key = f"{r.table_name}.{r.column_name}".upper()
        gt[key] = r.predicted_label

    # Every catalog column must have an entry.
    for table in tables:
        for col_name in table.columns:
            key = f"{table.name}.{col_name}".upper()
            gt.setdefault(key, LABEL_CLEAN)

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

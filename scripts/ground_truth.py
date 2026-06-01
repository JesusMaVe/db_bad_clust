"""
ground_truth.py — Anti-pattern ground truth for H1 validation

Purpose:
  Map each column from the 22 anti-pattern tables to its primary
  anti-pattern category. This ground truth is used for external
  clustering validation (ARI, NMI) to test H1:
    "Columns affected by similar anti-patterns cluster together."

  Each table gets a single primary anti-pattern label by priority:
    polymorphic > eav > reserved_words > self_referencing > giant_table
    > inconsistent_naming > self_contradictory > impossible_data
    > wrong_data_types > clean (no distinctive anti-pattern)

Usage:
  from ground_truth import get_ground_truth, build_ground_truth_map

  gt_map = build_ground_truth_map()
  truth_labels = get_ground_truth(column_table_map, column_names, gt_map)
"""

from __future__ import annotations

import logging

from anti_patterns import AntiPatternTable, generate_poorly_designed_tables

logger = logging.getLogger(__name__)

ANTI_PATTERN_PRIORITY: list[tuple[str, str]] = [
    ("polymorphic", "polymorphic"),
    ("eav_antipattern", "eav"),
    ("reserved_word_columns", "reserved_words"),
    ("self_referencing", "self_referencing"),
    ("giant_table", "giant_table"),
    ("inconsistent_naming", "inconsistent_naming"),
    ("self_contradictory", "self_contradictory"),
    ("impossible_data", "impossible_data"),
    ("wrong_data_types", "wrong_data_types"),
]

LABEL_CLEAN = "clean"
LABEL_UNKNOWN = None


def get_primary_anti_pattern(table: AntiPatternTable) -> str:
    """Determine the primary anti-pattern category for a table by priority.

    Args:
        table: An AntiPatternTable instance.

    Returns:
        Anti-pattern label string (e.g. "wrong_data_types", "clean").
    """
    for attr, label in ANTI_PATTERN_PRIORITY:
        if getattr(table, attr, False):
            return label
    return LABEL_CLEAN


def build_ground_truth_map() -> dict[str, str]:
    """Build a mapping from TABLE.COLUMN to primary anti-pattern label.

    Returns:
        Dict keyed by "TABLE_NAME.COLUMN_NAME" with anti-pattern label.
    """
    tables: list[AntiPatternTable] = generate_poorly_designed_tables()
    gt: dict[str, str] = {}
    for table in tables:
        label: str = get_primary_anti_pattern(table)
        for col_name in table.columns:
            key: str = f"{table.name}.{col_name}".upper()
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
            # column_names not available, we only have table-level info
            truth.append(LABEL_UNKNOWN)

    return truth

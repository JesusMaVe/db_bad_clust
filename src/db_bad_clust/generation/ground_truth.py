"""
ground_truth.py — The manual ground truth for the clustering evaluation.

The 243 columns of the Oracle instance were labelled by hand, one anti-pattern
per column, into `output/manual_labels.csv`. That file is the only ground truth
this branch recognises.

An earlier version of this module *derived* the ground truth by running the rule
engine over the schema. That made "accuracy" circular — the detector was scored
against its own output — and it is gone along with the rule engine (see the
`rule-engine` branch). What remains is the loader and the vocabulary a label is
allowed to take.

Usage:
    from db_bad_clust.generation.ground_truth import load_manual_ground_truth

    gt = load_manual_ground_truth("output/manual_labels.csv")
    gt["EMPLEADOS.FECHA_INGRESO"]  # → "wrong_data_types"
"""

from __future__ import annotations

import csv
from pathlib import Path

LABEL_CLEAN = "clean"

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
    """Load the hand-labelled ground truth from a CSV (see label_export.py).

    CSV columns: table,column,data_type,label. Every row must be filled and
    every label must be in MANUAL_LABEL_VOCABULARY — a blank or misspelled
    label would silently become a class of its own and distort every metric.

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
                raise ValueError(
                    f"Invalid label '{label}' at CSV row {i} "
                    f"(allowed: {sorted(MANUAL_LABEL_VOCABULARY)})"
                )
            gt[f"{row['table']}.{row['column']}".upper()] = label
    if not gt:
        raise ValueError(f"No rows in {path}")
    return gt

"""
table_aggregates.py — Per-table aggregate features, attached to every column of that table.

`giant_table`/`eav`/`polymorphic` are properties of a TABLE, not a column — no per-column
encoder can see them from one column's text alone. AGENTS.md documents that most of the document
embedding's edge on `giant_table` on the full corpus is identity leakage through the table
comment (memorizing which two tables are the known giants), not a generalisable "this table has
too many generic columns" concept. This module gives every column a numeric, non-identifying
summary of the table it belongs to instead: column count, nullable ratio, type diversity,
generic-name rate, comment ratio. All are statistics computed from structure alone — no table
name, no hand-written rule about what makes a table "bad".

Deep Sets (arXiv:1703.06114) is the ML precedent for this shape of feature: aggregating
instance-level information (columns) into a permutation-invariant set-level summary (the table),
then attaching that summary back onto every instance.

Usage:
    from db_bad_clust.features.table_aggregates import build_table_block

    e_table = build_table_block(schema, column_index)   # (N, 5)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from db_bad_clust.data.schema_extractor import DatabaseSchema

N_TABLE_FEATURES = 5

# A short, code-like identifier (A, F1, COL_001, AAA...) rather than a word — the
# structural shape a generic/auto-generated column name tends to have, independent
# of any specific naming convention this corpus happens to use.
_GENERIC_NAME = re.compile(r"^[A-Z]{1,4}_?\d{0,4}$")


def _is_generic_name(name: str) -> bool:
    return bool(_GENERIC_NAME.fullmatch(name))


def table_stats(schema: DatabaseSchema) -> dict[str, np.ndarray]:
    """table_name -> [log1p(n_cols), nullable_ratio, type_diversity, generic_name_rate, comment_ratio].

    type_diversity is n_distinct_types / n_columns (0 when every column shares one
    type, up to 1.0 when every column has a distinct type).
    """
    stats: dict[str, np.ndarray] = {}
    for table in schema.tables:
        cols = table.columns
        n = len(cols)
        if n == 0:
            stats[table.name] = np.zeros(N_TABLE_FEATURES, dtype=np.float64)
            continue
        n_types = len({c.data_type for c in cols})
        stats[table.name] = np.array(
            [
                np.log1p(n),
                sum(c.nullable for c in cols) / n,
                n_types / n,
                sum(_is_generic_name(c.name) for c in cols) / n,
                sum(c.comments is not None for c in cols) / n,
            ],
            dtype=np.float64,
        )
    return stats


def build_table_block(schema: DatabaseSchema, column_index: list[str]) -> np.ndarray:
    """One row per entry in `column_index`, carrying its table's aggregate stats.

    Shape (len(column_index), 5). A column whose table isn't in `schema` (should not
    happen with a consistent pickle) gets a zero row rather than raising.
    """
    stats = table_stats(schema)
    zero = np.zeros(N_TABLE_FEATURES, dtype=np.float64)
    rows = [stats.get(key.split(".", 1)[0], zero) for key in column_index]
    return np.array(rows, dtype=np.float64)

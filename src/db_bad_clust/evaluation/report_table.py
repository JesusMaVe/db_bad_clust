"""
report_table.py — One place that knows how to pad and align a text table.

Every report in `evaluation/` (`experiments.format_table`/`format_diagnostics`,
`ml_baselines.format_baselines`, `health_report.format_health_report`) built its
own header/row padding by hand, each with its own column widths and separator
length picked ad hoc — none of them agreed with each other, and the separator
length in `format_table` didn't even match its own header's actual width. This
module is the single seam: callers format each cell's *value* (a `f"{x:.4f}"` is
still the caller's call — this module only knows about layout), and get back an
aligned block with a separator sized to the header it actually printed.

Usage:
    from db_bad_clust.evaluation.report_table import Column, render_table

    columns = [Column("Configuration", 28, "<"), Column("ARI", 8)]
    rows = [["structure only", f"{0.0762:.4f}"]]
    print(render_table(columns, rows))
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    """One column's layout: header text, character width, and alignment.

    `align` is a Python format-spec alignment character: "<" left, ">" right.
    Cell values are handed in already formatted as strings — this module pads
    and aligns them, it does not decide how a float or percentage is written.
    """

    header: str
    width: int
    align: str = ">"


def render_table(columns: Sequence[Column], rows: Sequence[Sequence[str]]) -> str:
    """Render a header, a separator sized to it, and one line per row.

    Each row must have exactly as many cells as `columns`, each already
    formatted as a string (composite cells like "24/243" are fine — this
    function only pads and joins).
    """
    header = " ".join(f"{c.header:{c.align}{c.width}}" for c in columns)
    lines = [header, "-" * len(header)]
    for row in rows:
        if len(row) != len(columns):
            raise ValueError(f"row has {len(row)} cells, expected {len(columns)}")
        lines.append(
            " ".join(f"{cell:{c.align}{c.width}}" for c, cell in zip(columns, row, strict=True))
        )
    return "\n".join(lines)

"""
recommendation_reporter.py — Report generation for recommendations

Purpose:
  Generate formatted text reports for anti-pattern recommendations.
  Separates column-level and table-level issues.

Usage:
  from recommendation_reporter import RecommendationReporter

  reporter = RecommendationReporter()
  report = reporter.format_recommendations(recommendations)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class RecommendationReporter:
    """Generate formatted recommendation reports."""

    @staticmethod
    def format_recommendations(recommendations: dict[str, Any]) -> str:
        """Format recommendations as text.

        Args:
            recommendations: Dict with 'column_issues' and 'table_issues' groups.

        Returns:
            Multi-line string report with consolidated recommendations.
        """
        lines: list[str] = []

        # Column-level issues
        col_issues = recommendations.get("column_issues", [])
        if col_issues:
            lines.append("=" * 60)
            lines.append("COLUMN-LEVEL ISSUES")
            lines.append("=" * 60)

            for rec in col_issues:
                severity_icon = {"alta": "🔴", "media": "🟡", "baja": "🟢"}.get(
                    rec["severity"], "⚪"
                )
                lines.append("")
                lines.append(
                    f"{severity_icon} [{rec['label'].upper()}] "
                    f"Found in {rec['total_columns']} columns:"
                )

                # Show first 10 columns
                cols = rec["columns"][:10]
                cols_str = ", ".join(cols)
                if len(rec["columns"]) > 10:
                    cols_str += f"... (+{len(rec['columns']) - 10} more)"
                lines.append(f"   Columns: {cols_str}")

                # Action
                lines.append(f"   ✅ Action: {rec['action']}")

        # Table-level issues
        table_issues = recommendations.get("table_issues", [])
        if table_issues:
            lines.append("")
            lines.append("=" * 60)
            lines.append("TABLE-LEVEL ISSUES")
            lines.append("=" * 60)

            for rec in table_issues:
                severity_icon = {"alta": "🔴", "media": "🟡", "baja": "🟢"}.get(
                    rec["severity"], "⚪"
                )
                lines.append("")
                lines.append(
                    f"{severity_icon} [{rec['label'].upper()}] "
                    f"{rec['total_tables']} table(s) affected:"
                )

                # Show tables
                tables = rec["tables"][:5]
                tables_str = ", ".join(tables)
                if len(rec["tables"]) > 5:
                    tables_str += f"... (+{len(rec['tables']) - 5} more)"
                lines.append(f"   Tables: {tables_str}")

                # Action
                lines.append(f"   ✅ Action: {rec['action']}")

        if not col_issues and not table_issues:
            lines.append("No anti-patterns detected.")

        return "\n".join(lines)

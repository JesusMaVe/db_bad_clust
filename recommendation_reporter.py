"""
recommendation_reporter.py — Report generation for recommendations

Purpose:
  Generate formatted text reports for anti-pattern recommendations.

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
    def format_recommendations(recommendations: list[dict[str, Any]]) -> str:
        """Format recommendations as text.

        Args:
            recommendations: List of recommendation dicts from Recommender.

        Returns:
            Multi-line string report.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("RECOMMENDATIONS PER CLUSTER")
        lines.append("=" * 60)

        for rec in recommendations:
            lines.append(
                f"\nCluster #{rec['cluster_id']} "
                f"[{rec['severity'].upper()}] "
                f"({rec['total_columns']} columns, "
                f"{len(rec['tables_involved'])} tables)"
            )
            lines.append(f"  Dominant type: {rec['dominant_type']}")
            lines.append(f"  Tables: {', '.join(rec['tables_involved'][:5])}")
            if len(rec["tables_involved"]) > 5:
                lines.append(f"    ... and {len(rec['tables_involved']) - 5} more")

            for issue in rec["issues"]:
                lines.append(f"  - {issue}")

            for fix in rec["recommendations"]:
                lines.append(f"    -> {fix}")

        return "\n".join(lines)

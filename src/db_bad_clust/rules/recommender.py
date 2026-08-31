"""
recommender.py — Recommendation generation for fixing anti-patterns

Purpose:
  Analyze schema and produce actionable recommendations grouped by
  anti-pattern type. Separates column-level and table-level issues
  for clean reporting.

Usage:
  recommender = Recommender()
  recommendations = recommender.recommend(schema)
  report = recommender.print_recommendations(recommendations)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from db_bad_clust.data.schema_extractor import DatabaseSchema
from db_bad_clust.rules.recommendation_reporter import RecommendationReporter
from db_bad_clust.rules.rule_engine import (
    ClassificationResult,
    ColumnRuleEngine,
    Severity,
    TableRuleEngine,
)

logger = logging.getLogger(__name__)

# Action recommendations per anti-pattern type
ANTI_PATTERN_ACTIONS = {
    "date_as_text": "ALTER TABLE ... MODIFY (... DATE);",
    "number_as_text": "ALTER TABLE ... MODIFY (... NUMBER);",
    "reserved_word": "Rename column to {col}_COL or similar;",
    "bad_boolean": "Unify to CHAR(1) with CHECK ('S','N') or NUMBER(1);",
    "impossible_data": "Set primary key column to NOT NULL;",
    "self_referencing": "Valid for hierarchies, but verify intent;",
    "polymorphic": "Use separate tables per type or FK constraint per type;",
    "giant_table": "Split into smaller, focused tables;",
    "eav_pattern": "Normalize to proper relational design or use JSON/XML;",
    "inconsistent_naming": "Standardize to {primary} convention;",
    "implicit_fk": "Add a FOREIGN KEY constraint to the referenced table;",
    "missing_pk": "Add a PRIMARY KEY constraint on a unique identifier column;",
    "redundant_index": "Drop the redundant index or extend it;",
    "fk_without_index": "Create a covering index on the foreign-key columns;",
    "stale_statistics": "Run DBMS_STATS.GATHER_TABLE_STATS on the table;",
    "disabled_constraint": "Enable and validate the constraint;",
    "obsolete_type": "Convert LONG/LONG RAW/RAW columns to CLOB/BLOB or standard types;",
    "partition_candidate": "Evaluate range/list partitioning, typically by date;",
    "oversized_varchar": "Reduce VARCHAR2 size to match actual data length;",
}


class Recommender:
    """
    Generates correction recommendations based on schema analysis.
    Separates column-level and table-level issues.
    """

    def __init__(self) -> None:
        self.col_engine = ColumnRuleEngine()
        self.table_engine = TableRuleEngine()

    def recommend(self, schema: DatabaseSchema) -> dict[str, Any]:
        """
        Generate recommendations grouped by anti-pattern type.

        Args:
            schema: DatabaseSchema extracted from Oracle.

        Returns:
            Dict with 'column_issues' and 'table_issues' groups.
        """
        all_columns = []
        for table in schema.tables:
            for col in table.columns:
                all_columns.append(col)

        if not all_columns:
            return {"column_issues": [], "table_issues": []}

        col_results = self.col_engine.classify(all_columns, schema)
        table_results = self.table_engine.classify(schema)

        # SchemaSpy-style report-level findings (not column labels)
        col_results = col_results + self.col_engine.detect_implicit_fks(all_columns, schema)
        col_results = col_results + self.col_engine.detect_obsolete_types(all_columns, schema)
        col_results = col_results + self.col_engine.detect_oversized_varchars(all_columns, schema)
        table_results = (
            table_results
            + self.table_engine.detect_missing_pk(schema)
            + self.table_engine.detect_redundant_indexes(schema)
            + self.table_engine.detect_fk_without_index(schema)
            + self.table_engine.detect_stale_statistics(schema)
            + self.table_engine.detect_disabled_constraints(schema)
            + self.table_engine.detect_partition_candidates(schema)
        )

        col_groups = self._group_column_results(col_results)
        table_groups = self._group_table_results(table_results)

        return {
            "column_issues": col_groups,
            "table_issues": table_groups,
        }

    def _group_column_results(
        self, results: list[ClassificationResult]
    ) -> list[dict[str, Any]]:
        """Group column-level results by predicted label and data type."""
        grouped = defaultdict(lambda: defaultdict(list))

        for r in results:
            # Map rule label to ground truth label
            label = self._map_label(r.predicted_label)
            grouped[label][r.method].append({
                "column": r.column_name,
                "table": r.table_name,
                "confidence": r.confidence,
                "explanation": r.explanation,
            })

        recommendations = []
        for label, methods in grouped.items():
            total_columns = sum(len(cols) for cols in methods.values())
            all_columns = []
            for cols in methods.values():
                all_columns.extend([c["column"] for c in cols])

            action = ANTI_PATTERN_ACTIONS.get(label, "Review and fix;")
            severity = self._get_max_severity(results, label)

            recommendations.append({
                "label": label,
                "total_columns": total_columns,
                "columns": all_columns,
                "severity": severity,
                "action": action,
                "details": methods,
            })

        return sorted(recommendations, key=lambda x: -x["total_columns"])

    def _group_table_results(
        self, results: list[ClassificationResult]
    ) -> list[dict[str, Any]]:
        """Group table-level results by predicted label."""
        grouped = defaultdict(list)

        for r in results:
            label = self._map_label(r.predicted_label)
            grouped[label].append({
                "table": r.table_name,
                "confidence": r.confidence,
                "explanation": r.explanation,
            })

        recommendations = []
        for label, tables in grouped.items():
            table_names = [t["table"] for t in tables]
            action = ANTI_PATTERN_ACTIONS.get(label, "Review and fix;")
            severity = self._get_max_severity(results, label)

            recommendations.append({
                "label": label,
                "total_tables": len(tables),
                "tables": table_names,
                "severity": severity,
                "action": action,
                "details": tables,
            })

        return sorted(recommendations, key=lambda x: -x["total_tables"])

    def _map_label(self, rule_label: str) -> str:
        """Map rule engine labels to ground truth labels (shared map in ground_truth)."""
        from db_bad_clust.generation.ground_truth import RULE_TO_MANUAL

        return RULE_TO_MANUAL.get(rule_label, rule_label)

    def _get_max_severity(
        self, results: list[ClassificationResult], label: str
    ) -> str:
        """Get maximum severity for a given label."""
        severity_map = {
            Severity.HIGH: "alta",
            Severity.MEDIUM: "media",
            Severity.LOW: "baja",
        }
        for r in results:
            if self._map_label(r.predicted_label) == label:
                return severity_map.get(r.severity, "media")
        return "media"

    @staticmethod
    def print_recommendations(recommendations: dict[str, Any]) -> str:
        """Format recommendations as text.

        Delegates to RecommendationReporter for formatting.
        """
        return RecommendationReporter.format_recommendations(recommendations)

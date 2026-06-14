"""
evaluator.py — Clustering quality evaluation (facade)

Purpose:
  Facade module that provides a unified interface to the evaluation
  subsystem. Delegates to specialized modules:
  - metrics.py: Internal clustering quality metrics
  - validation.py: External validation against ground truth
  - anomaly.py: Anomaly detection
  - reporter.py: Report generation

Usage:
  evaluator = Evaluator()
  metrics = evaluator.evaluate(X_reduced, labels)
  report = evaluator.print_report(schema, labels, metrics)
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from metrics import ClusteringMetrics
from validation import GroundTruthValidator
from anomaly import AnomalyDetector
from reporter import EvaluationReporter

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluates clustering quality and generates reports.

    This class acts as a facade, delegating to specialized modules
    while maintaining backward compatibility.
    """

    def __init__(self) -> None:
        self._metrics = ClusteringMetrics()
        self._validator = GroundTruthValidator()
        self._anomaly_detector = AnomalyDetector()
        self._reporter = EvaluationReporter()

    # ── Internal metrics (delegated to ClusteringMetrics) ──────────────

    @staticmethod
    def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
        """Silhouette Score (-1 to 1, higher = better)."""
        return ClusteringMetrics.silhouette(X, labels)

    @staticmethod
    def davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
        """Davies-Bouldin Index (0 to inf, lower = better)."""
        return ClusteringMetrics.davies_bouldin(X, labels)

    @staticmethod
    def calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
        """Calinski-Harabasz Index (higher = better)."""
        return ClusteringMetrics.calinski_harabasz(X, labels)

    # ── External validation (delegated to GroundTruthValidator) ────────

    @staticmethod
    def adjusted_rand_index(
        labels_pred: np.ndarray,
        labels_true: np.ndarray,
    ) -> float:
        """Adjusted Rand Index (-1 to 1, higher = better agreement)."""
        return GroundTruthValidator.adjusted_rand_index(labels_pred, labels_true)

    @staticmethod
    def normalized_mutual_info(
        labels_pred: np.ndarray,
        labels_true: np.ndarray,
    ) -> float:
        """Normalized Mutual Information (0 to 1, higher = better)."""
        return GroundTruthValidator.normalized_mutual_info(labels_pred, labels_true)

    # ── Anomaly detection (delegated to AnomalyDetector) ──────────────

    @staticmethod
    def anomaly_detection_metrics(
        labels_pred: np.ndarray,
        labels_true: np.ndarray,
    ) -> dict[str, float]:
        """Compute anomaly precision/recall/F1 treating -1 as anomaly."""
        return AnomalyDetector.detection_metrics(labels_pred, labels_true)

    @staticmethod
    def centroid_anomaly_scores(
        X: np.ndarray,
        labels: np.ndarray,
        percentile: float = 95.0,
    ) -> dict[str, Any]:
        """Detect structural anomalies via distance to cluster centroid."""
        return AnomalyDetector.centroid_scores(X, labels, percentile)

    # ── Composite validation ──────────────────────────────────────────

    def validate_against_ground_truth(
        self,
        labels_pred: np.ndarray,
        labels_true: list[str | None],
        X: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """External validation: compare predicted clusters against ground truth."""
        return self._validator.validate(labels_pred, labels_true, X)

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        """Compute all internal metrics."""
        return self._metrics.evaluate(X, labels)

    # ── Cross-table analysis ────────────────────────────────────────────

    @staticmethod
    def cross_table_analysis(
        column_table_map: list[str],
        labels: np.ndarray,
    ) -> dict[str, Any]:
        """Analyze how columns from each table are distributed across clusters."""
        table_clusters: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))  # type: ignore[arg-type]
        for table_name, label in zip(column_table_map, labels):
            table_clusters[table_name][int(label)] += 1

        # Purity: fraction of a table's columns in their majority cluster
        purity: dict[str, dict[str, Any]] = {}
        for table, clusters in table_clusters.items():
            total = sum(clusters.values())
            majority = max(clusters.values())
            purity[table] = {
                "total_columns": total,
                "majority_cluster": max(clusters, key=clusters.get),
                "majority_count": majority,
                "purity": majority / total if total > 0 else 0.0,
                "cluster_distribution": dict(clusters),
            }

        purity_values = [p["purity"] for p in purity.values()]
        avg_purity = np.mean(purity_values) if purity_values else 0.0

        return {
            "table_distribution": dict(table_clusters),
            "table_purity": purity,
            "average_purity": float(avg_purity),
        }

    # ── Anti-pattern composition per cluster ────────────────────────────

    @staticmethod
    def cluster_composition(
        labels: np.ndarray,
        column_metadata: list[Any],
    ) -> dict[int, dict[str, Any]]:
        """Analyze which anti-pattern characteristics appear in each cluster."""
        from schema_extractor import ColumnMetadata  # noqa: F401  # used for type info

        clusters: dict[int, list[Any]] = defaultdict(list)
        for label, col in zip(labels, column_metadata):
            clusters[int(label)].append(col)

        composition: dict[int, dict[str, Any]] = {}
        for cluster_id, cols in clusters.items():
            total = len(cols)
            _ = sum(1 for c in cols if not c.is_primary_key and c.name.upper() in ("ID",))

            null_count = sum(1 for c in cols if c.nullable)
            text_types = sum(1 for c in cols if c.data_type.upper().startswith("VARCHAR"))
            numeric_types = sum(
                1 for c in cols if c.data_type.upper() in ("NUMBER", "FLOAT", "INTEGER", "INT")
            )

            composition[int(cluster_id)] = {
                "total_columns": total,
                "pct_nullable": null_count / total * 100 if total else 0,
                "pct_varchar": text_types / total * 100 if total else 0,
                "pct_numeric": numeric_types / total * 100 if total else 0,
                "types_distribution": dict(Counter(c.data_type for c in cols)),
            }

        return composition

    # ── Report (delegated to EvaluationReporter) ────────────────────────

    @staticmethod
    def print_report(
        metrics: dict[str, Any],
        cross_table: dict[str, Any] | None = None,
        composition: dict[int, dict[str, Any]] | None = None,
    ) -> str:
        """Generate a formatted text report with all results."""
        return EvaluationReporter.internal_metrics_report(metrics, cross_table, composition)

    @staticmethod
    def print_validation_report(validation: dict[str, Any]) -> str:
        """Generate a formatted report for ground truth validation."""
        return EvaluationReporter.validation_report(validation)

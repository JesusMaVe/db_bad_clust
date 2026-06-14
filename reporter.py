"""
reporter.py — Report generation for clustering evaluation

Purpose:
  Generate formatted text reports for clustering evaluation results,
  including internal metrics, cross-table analysis, cluster composition,
  and ground truth validation.

Usage:
  from reporter import EvaluationReporter

  reporter = EvaluationReporter()
  report = reporter.internal_metrics_report(metrics)
  report += reporter.validation_report(validation)
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class EvaluationReporter:
    """Generate formatted evaluation reports."""

    @staticmethod
    def internal_metrics_report(
        metrics: dict[str, Any],
        cross_table: dict[str, Any] | None = None,
        composition: dict[int, dict[str, Any]] | None = None,
    ) -> str:
        """Generate a formatted text report with internal metrics.

        Args:
            metrics: Dict from ClusteringMetrics.evaluate().
            cross_table: Optional cross-table analysis results.
            composition: Optional cluster composition results.

        Returns:
            Multi-line string report.
        """
        lines: list[str] = []
        lines.append("=" * 60)
        lines.append("EVALUACION DE CLUSTERING")
        lines.append("=" * 60)

        if "error" in metrics:
            lines.append(f"ERROR: {metrics['error']}")
            return "\n".join(lines)

        lines.append("\n--- Metricas Internas ---")
        lines.append(f"  Silhouette Score:         {metrics['silhouette']:.4f}")
        lines.append(f"  Davies-Bouldin Index:     {metrics['davies_bouldin']:.4f}")
        lines.append(f"  Calinski-Harabasz Index:  {metrics['calinski_harabasz']:.2f}")
        lines.append(f"  Clusters formados:        {metrics['n_clusters']}")
        lines.append(f"  Muestras totales:         {metrics['n_samples']}")
        lines.append(f"  Outliers (DBSCAN):        {metrics['n_noise']}")

        if cross_table:
            lines.append("\n--- Distribucion por Tabla ---")
            lines.append(f"  Pureza promedio: {cross_table['average_purity']:.2%}")
            for table, info in sorted(cross_table.get("table_purity", {}).items()):
                lines.append(
                    f"  {table}: pureza={info['purity']:.0%} "
                    f"({info['total_columns']} cols, cluster mayoritario #{info['majority_cluster']})"
                )

        if composition:
            lines.append("\n--- Composicion por Cluster ---")
            for cid, info in sorted(composition.items()):
                if cid == -1:
                    continue
                lines.append(f"  Cluster #{cid}: {info['total_columns']} columnas")
                lines.append(f"    Nullable: {info['pct_nullable']:.0f}%")
                lines.append(f"    VARCHAR:  {info['pct_varchar']:.0f}%")
                lines.append(f"    Numerico: {info['pct_numeric']:.0f}%")

        lines.append("\n" + "=" * 60)
        return "\n".join(lines)

    @staticmethod
    def validation_report(validation: dict[str, Any]) -> str:
        """Generate a formatted report for ground truth validation.

        Args:
            validation: Dict from GroundTruthValidator.validate().

        Returns:
            Multi-line string report.
        """
        lines: list[str] = []
        lines.append("-" * 50)
        lines.append("VALIDACION H1 — Ground Truth")
        lines.append("-" * 50)

        if "error" in validation:
            lines.append(f"  ERROR: {validation['error']}")
            lines.append(f"  Columnas con GT: {validation.get('n_gt_labels', 0)}")
        else:
            lines.append(f"  Adjusted Rand Index (ARI):  {validation['ari']:.4f}")
            lines.append(f"  Normalized Mutual Info:    {validation['nmi']:.4f}")
            lines.append(f"  Columnas con GT:           {validation['n_gt_labels']}")
            lines.append("  Anomaly (noise) detection:")
            lines.append(f"    Precision: {validation['anomaly_precision']:.2%}")
            lines.append(f"    Recall:    {validation['anomaly_recall']:.2%}")
            lines.append(f"    F1:        {validation['anomaly_f1']:.2%}")
            lines.append(f"    Support:   {validation['anomaly_support']}")
            if "structural_anomalies" in validation:
                lines.append("  Structural anomalies (centroid distance):")
                lines.append(f"    Count:  {validation['structural_anomalies']}")
                lines.append(f"    Ratio:  {validation['structural_anomaly_ratio']:.2%}")
                lines.append(f"    Mean distance: {validation['centroid_mean_distance']:.4f}")
                lines.append(f"    Std distance:  {validation['centroid_std_distance']:.4f}")

        lines.append("-" * 50)
        return "\n".join(lines)

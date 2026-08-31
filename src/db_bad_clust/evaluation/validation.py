"""
validation.py — External validation against ground truth

Purpose:
  Compare predicted cluster labels against known ground truth labels
  to validate clustering quality using external metrics.

Metrics:
  - Adjusted Rand Index (ARI): -1 to 1, higher = better agreement
  - Normalized Mutual Information (NMI): 0 to 1, higher = better

Usage:
  from db_bad_clust.evaluation.validation import GroundTruthValidator

  validator = GroundTruthValidator()
  result = validator.validate(labels_pred, labels_true)
"""

from __future__ import annotations

import logging
from collections import Counter, defaultdict
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class GroundTruthValidator:
    """Validate clustering against ground truth labels."""

    @staticmethod
    def adjusted_rand_index(
        labels_pred: np.ndarray,
        labels_true: np.ndarray,
    ) -> float:
        """Adjusted Rand Index (-1 to 1, higher = better agreement).

        Measures the similarity between two clusterings, adjusted for chance.
        """
        from sklearn.metrics import adjusted_rand_score

        return float(adjusted_rand_score(labels_true, labels_pred))

    @staticmethod
    def normalized_mutual_info(
        labels_pred: np.ndarray,
        labels_true: np.ndarray,
    ) -> float:
        """Normalized Mutual Information (0 to 1, higher = better).

        Measures the mutual information between two clusterings,
        normalized by the average entropy.
        """
        from sklearn.metrics import normalized_mutual_info_score

        return float(normalized_mutual_info_score(labels_true, labels_pred))

    def validate(
        self,
        labels_pred: np.ndarray,
        labels_true: list[str | None],
        X: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Validate predicted clusters against ground truth.

        Args:
            labels_pred: Predicted cluster labels from clustering.
            labels_true: Ground truth anti-pattern labels per column.
            X: Optional reduced feature matrix for centroid anomaly scoring.

        Returns:
            Dict with ari, nmi, n_gt_labels, and optional anomaly metrics.
        """
        from db_bad_clust.evaluation.anomaly import AnomalyDetector

        # Filter to columns with known ground truth
        mask: np.ndarray = np.array([t is not None for t in labels_true])
        if mask.sum() < 2:
            return {
                "error": "Less than 2 columns with ground truth labels",
                "ari": 0.0,
                "nmi": 0.0,
                "n_gt_labels": int(mask.sum()),
            }

        pred_known: np.ndarray = labels_pred[mask]
        true_known: np.ndarray = np.array([t for t in labels_true if t is not None])

        ari: float = self.adjusted_rand_index(pred_known, true_known)
        nmi: float = self.normalized_mutual_info(pred_known, true_known)

        detector = AnomalyDetector()
        anomaly: dict[str, float] = detector.detection_metrics(labels_pred, labels_true)

        result: dict[str, Any] = {
            "ari": ari,
            "nmi": nmi,
            "n_gt_labels": int(mask.sum()),
            "anomaly_precision": anomaly["precision"],
            "anomaly_recall": anomaly["recall"],
            "anomaly_f1": anomaly["f1"],
            "anomaly_support": anomaly["support"],
        }

        if X is not None and -1 not in set(labels_pred):
            centroid_anomaly: dict[str, Any] = detector.centroid_scores(X, labels_pred)
            result["structural_anomalies"] = centroid_anomaly["n_anomalies"]
            result["structural_anomaly_ratio"] = (
                centroid_anomaly["n_anomalies"] / X.shape[0]
                if X.shape[0] > 0
                else 0.0
            )
            result["centroid_mean_distance"] = centroid_anomaly["mean_distance"]
            result["centroid_std_distance"] = centroid_anomaly["std_distance"]

        return result


def cross_table_analysis(
    column_table_map: list[str],
    labels: np.ndarray,
) -> dict[str, Any]:
    """Analyze how columns from each table are distributed across clusters.

    Salvaged from the deleted evaluator facade (was a module-level helper
    exposed through it).
    """
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


def cluster_composition(
    labels: np.ndarray,
    column_metadata: list[Any],
) -> dict[int, dict[str, Any]]:
    """Analyze which anti-pattern characteristics appear in each cluster."""
    clusters: dict[int, list[Any]] = defaultdict(list)
    for label, col in zip(labels, column_metadata):
        clusters[int(label)].append(col)

    composition: dict[int, dict[str, Any]] = {}
    for cluster_id, cols in clusters.items():
        total = len(cols)
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

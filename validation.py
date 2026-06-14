"""
validation.py — External validation against ground truth

Purpose:
  Compare predicted cluster labels against known ground truth labels
  to validate clustering quality using external metrics.

Metrics:
  - Adjusted Rand Index (ARI): -1 to 1, higher = better agreement
  - Normalized Mutual Information (NMI): 0 to 1, higher = better

Usage:
  from validation import GroundTruthValidator

  validator = GroundTruthValidator()
  result = validator.validate(labels_pred, labels_true)
"""

from __future__ import annotations

import logging
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
        from anomaly import AnomalyDetector

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

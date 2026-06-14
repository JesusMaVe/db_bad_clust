"""
anomaly.py — Anomaly detection in clusters

Purpose:
  Detect anomalies and outliers in clustering results using
  multiple methods:
  - Noise detection: Points labeled as -1 by density-based algorithms
  - Centroid distance: Points far from cluster centroids
  - Precision/Recall/F1: Compare predicted anomalies against ground truth

Usage:
  from anomaly import AnomalyDetector

  detector = AnomalyDetector()
  metrics = detector.detection_metrics(labels_pred, labels_true)
  scores = detector.centroid_scores(X, labels)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detect anomalies in clustering results."""

    @staticmethod
    def detection_metrics(
        labels_pred: np.ndarray,
        labels_true: np.ndarray,
    ) -> dict[str, float]:
        """Compute anomaly precision/recall/F1 treating -1 as anomaly.

        Compares predicted noise (-1) against columns with unknown
        ground truth (None) as the anomaly class.

        Args:
            labels_pred: Predicted cluster labels (-1 = noise).
            labels_true: Ground truth labels (None = unknown/noise).

        Returns:
            Dict with precision, recall, f1, support.
        """
        from sklearn.metrics import precision_recall_fscore_support

        # Binarize: -1 in pred = anomaly (1), else 0
        # Binarize: None in true = anomaly (1), else 0
        pred_bin: np.ndarray = np.where(labels_pred == -1, 1, 0).astype(int)
        true_bin: np.ndarray = np.where(
            np.array([t is None for t in labels_true]),
            1,
            0,
        ).astype(int)

        if true_bin.sum() == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0}

        precision, recall, f1, _ = precision_recall_fscore_support(
            true_bin,
            pred_bin,
            average="binary",
            zero_division=0.0,
        )
        support: int = int(true_bin.sum())
        return {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": support,
        }

    @staticmethod
    def centroid_scores(
        X: np.ndarray,
        labels: np.ndarray,
        percentile: float = 95.0,
    ) -> dict[str, Any]:
        """Detect structural anomalies via distance to cluster centroid.

        For each point, computes the distance to its cluster centroid.
        Points beyond the given percentile are flagged as structural
        anomalies. Only for methods with centroids (e.g. KMeans).

        Args:
            X: Reduced feature matrix, shape (N, d).
            labels: Cluster assignments (no -1 expected).
            percentile: Distance percentile threshold (default 95).

        Returns:
            Dict with anomaly_indices, anomaly_distances, n_anomalies,
            mean_distance, std_distance.
        """
        unique: np.ndarray = np.unique(labels)
        if len(unique) <= 1 or -1 in unique:
            return {
                "anomaly_indices": np.array([], dtype=int),
                "anomaly_distances": np.array([], dtype=float),
                "n_anomalies": 0,
                "mean_distance": 0.0,
                "std_distance": 0.0,
            }

        # Vectorized centroid computation
        centroids: dict[int, np.ndarray] = {}
        for cid in unique:
            mask_c: np.ndarray = labels == cid
            centroids[int(cid)] = X[mask_c].mean(axis=0)

        # Vectorized distance computation (6x faster than loop)
        centroid_array = np.array([centroids[int(l)] for l in labels])
        distances: np.ndarray = np.linalg.norm(X - centroid_array, axis=1)

        threshold: float = float(np.percentile(distances, percentile))
        anomaly_mask: np.ndarray = distances > threshold
        anomaly_indices: np.ndarray = np.where(anomaly_mask)[0]

        return {
            "anomaly_indices": anomaly_indices,
            "anomaly_distances": distances[anomaly_mask],
            "n_anomalies": int(anomaly_mask.sum()),
            "mean_distance": float(distances.mean()),
            "std_distance": float(distances.std()),
            "threshold": float(threshold),
            "percentile": float(percentile),
        }

"""
metrics.py — Internal clustering quality metrics

Purpose:
  Compute internal metrics to evaluate clustering quality without
  ground truth labels. These metrics measure intra-cluster cohesion
  and inter-cluster separation.

Metrics:
  - Silhouette Score: -1 to 1, higher = better
  - Davies-Bouldin Index: 0 to inf, lower = better
  - Calinski-Harabasz Index: higher = better

Usage:
  from db_bad_clust.evaluation.metrics import ClusteringMetrics

  metrics = ClusteringMetrics()
  result = metrics.evaluate(X_reduced, labels)
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from db_bad_clust.exceptions import EvaluationError

logger = logging.getLogger(__name__)


class ClusteringMetrics:
    """Compute internal clustering quality metrics."""

    @staticmethod
    def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
        """Silhouette Score (-1 to 1, higher = better).

        Measures how similar points are to their own cluster compared
        to other clusters.
        """
        from sklearn.metrics import silhouette_score

        unique = set(labels)
        if len(unique) <= 1 or len(unique) >= X.shape[0]:
            return 0.0
        try:
            try:
                return float(silhouette_score(X, labels))
            except Exception as exc:
                raise EvaluationError("Silhouette calculation failed") from exc
        except EvaluationError as e:
            logger.warning("Error computing silhouette: %s", e)
            return 0.0

    @staticmethod
    def davies_bouldin(X: np.ndarray, labels: np.ndarray) -> float:
        """Davies-Bouldin Index (0 to inf, lower = better).

        Average similarity between clusters, where similarity compares
        intra-cluster scatter to inter-cluster separation.
        """
        from sklearn.metrics import davies_bouldin_score

        unique = set(labels) - {-1}
        if len(unique) <= 1:
            return 0.0
        try:
            try:
                return float(davies_bouldin_score(X, labels))
            except Exception as exc:
                raise EvaluationError("Davies-Bouldin calculation failed") from exc
        except EvaluationError as e:
            logger.warning("Error computing Davies-Bouldin: %s", e)
            return 0.0

    @staticmethod
    def calinski_harabasz(X: np.ndarray, labels: np.ndarray) -> float:
        """Calinski-Harabasz Index (higher = better).

        Ratio of between-cluster dispersion to within-cluster dispersion.
        """
        from sklearn.metrics import calinski_harabasz_score

        unique = set(labels) - {-1}
        if len(unique) <= 1:
            return 0.0
        try:
            try:
                return float(calinski_harabasz_score(X, labels))
            except Exception as exc:
                raise EvaluationError("Calinski-Harabasz calculation failed") from exc
        except EvaluationError as e:
            logger.warning("Error computing CH: %s", e)
            return 0.0

    def evaluate(self, X: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        """Compute all internal metrics.

        Args:
            X: Feature matrix, shape (N, d).
            labels: Cluster labels (-1 = noise).

        Returns:
            Dict with computed metrics.
        """
        mask = labels != -1
        if mask.sum() == 0:
            return {"error": "No samples with assigned cluster"}

        X_clean = X[mask]
        labels_clean = labels[mask]

        return {
            "silhouette": self.silhouette(X_clean, labels_clean),
            "davies_bouldin": self.davies_bouldin(X_clean, labels_clean),
            "calinski_harabasz": self.calinski_harabasz(X_clean, labels_clean),
            "n_clusters": len(set(labels_clean)),
            "n_samples": len(labels),
            "n_noise": int((labels == -1).sum()),
        }

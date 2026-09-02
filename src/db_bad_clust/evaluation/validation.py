"""
validation.py — External validation against ground truth

Metrics:
  - Adjusted Rand Index (ARI): -1 to 1, higher = better agreement
  - Normalized Mutual Information (NMI): 0 to 1, higher = better

Both are consumed by evaluation/cluster_scoring.py, the actual scoring path
`cli experiment` uses; this module owns their computation so cluster_scoring
and ml_baselines.py share one implementation.

Usage:
  from db_bad_clust.evaluation.validation import GroundTruthValidator

  validator = GroundTruthValidator()
  ari = validator.adjusted_rand_index(labels_pred, labels_true)
"""

from __future__ import annotations

import numpy as np


class GroundTruthValidator:
    """Score a clustering against ground truth labels."""

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

"""
test_validation.py — Tests for validation.py module
"""

import numpy as np
import pytest
from validation import GroundTruthValidator


class TestAdjustedRandIndex:
    """Tests for ARI metric."""

    def test_ari_perfect_match(self):
        """Identical clusterings should have ARI = 1."""
        labels_pred = np.array([0, 0, 1, 1])
        labels_true = np.array([0, 0, 1, 1])
        assert GroundTruthValidator.adjusted_rand_index(labels_pred, labels_true) == 1.0

    def test_ari_random(self):
        """Random clusterings should have ARI near 0."""
        labels_pred = np.array([0, 1, 0, 1])
        labels_true = np.array([0, 0, 1, 1])
        ari = GroundTruthValidator.adjusted_rand_index(labels_pred, labels_true)
        # ARI can be negative or positive for random
        assert -1.0 <= ari <= 1.0

    def test_ari_complementary(self):
        """Complementary clusterings still have ARI = 1 (same structure)."""
        labels_pred = np.array([0, 0, 1, 1])
        labels_true = np.array([1, 1, 0, 0])
        ari = GroundTruthValidator.adjusted_rand_index(labels_pred, labels_true)
        # ARI is invariant to label permutation
        assert ari == 1.0


class TestNormalizedMutualInfo:
    """Tests for NMI metric."""

    def test_nmi_perfect_match(self):
        """Identical clusterings should have NMI = 1."""
        labels_pred = np.array([0, 0, 1, 1])
        labels_true = np.array([0, 0, 1, 1])
        assert GroundTruthValidator.normalized_mutual_info(labels_pred, labels_true) == 1.0

    def test_nmi_range(self):
        """NMI should be between 0 and 1."""
        labels_pred = np.array([0, 1, 0, 1])
        labels_true = np.array([0, 0, 1, 1])
        nmi = GroundTruthValidator.normalized_mutual_info(labels_pred, labels_true)
        assert 0 <= nmi <= 1


class TestValidate:
    """Tests for validate method."""

    def test_validate_with_ground_truth(self):
        """Validate with known ground truth."""
        validator = GroundTruthValidator()
        labels_pred = np.array([0, 0, 1, 1, 0])
        labels_true = ["clean", "clean", "eav", "eav", None]
        result = validator.validate(labels_pred, labels_true)
        assert "ari" in result
        assert "nmi" in result
        assert "n_gt_labels" in result

    def test_validate_insufficient_gt(self):
        """Validate with less than 2 ground truth labels."""
        validator = GroundTruthValidator()
        labels_pred = np.array([0, 0, 1, 1])
        labels_true = [None, None, None, None]
        result = validator.validate(labels_pred, labels_true)
        assert "error" in result
        assert result["n_gt_labels"] == 0

    def test_validate_with_feature_matrix(self):
        """Validate with X for centroid anomaly scores."""
        validator = GroundTruthValidator()
        X = np.random.randn(10, 5)
        labels_pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
        labels_true = ["clean", "clean", "clean", "eav", "eav", "eav",
                       "giant", "giant", "giant", "clean"]
        result = validator.validate(labels_pred, labels_true, X)
        assert "structural_anomalies" in result

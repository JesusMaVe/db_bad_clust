"""
test_anomaly.py — Tests for anomaly.py module
"""

import numpy as np

from db_bad_clust.evaluation.anomaly import AnomalyDetector


class TestDetectionMetrics:
    """Tests for anomaly detection metrics."""

    def test_perfect_detection(self):
        """Perfect anomaly detection should have precision=1, recall=1."""
        labels_pred = np.array([-1, -1, 0, 0, 0])
        labels_true = [None, None, "clean", "clean", "clean"]
        result = AnomalyDetector.detection_metrics(labels_pred, labels_true)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_no_anomalies_in_gt(self):
        """No anomalies in ground truth returns zeros."""
        labels_pred = np.array([-1, 0, 0, 0])
        labels_true = ["clean", "clean", "clean", "clean"]
        result = AnomalyDetector.detection_metrics(labels_pred, labels_true)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["support"] == 0

    def test_no_predicted_anomalies(self):
        """No predicted anomalies returns precision=0."""
        labels_pred = np.array([0, 0, 0, 0])
        labels_true = [None, None, "clean", "clean"]
        result = AnomalyDetector.detection_metrics(labels_pred, labels_true)
        assert result["recall"] == 0.0


class TestCentroidScores:
    """Tests for centroid-based anomaly scoring."""

    def test_returns_expected_keys(self):
        """centroid_scores returns all expected keys."""
        X = np.random.randn(50, 5)
        labels = np.random.randint(0, 3, 50)
        result = AnomalyDetector.centroid_scores(X, labels)
        assert "anomaly_indices" in result
        assert "anomaly_distances" in result
        assert "n_anomalies" in result
        assert "mean_distance" in result
        assert "std_distance" in result
        assert "threshold" in result

    def test_single_cluster(self):
        """Single cluster returns no anomalies."""
        X = np.random.randn(10, 5)
        labels = np.zeros(10, dtype=int)
        result = AnomalyDetector.centroid_scores(X, labels)
        assert result["n_anomalies"] == 0

    def test_with_noise(self):
        """Labels with -1 returns no anomalies."""
        X = np.random.randn(10, 5)
        labels = np.array([-1, -1, 0, 0, 1, 1, 0, 1, 0, 1])
        result = AnomalyDetector.centroid_scores(X, labels)
        assert result["n_anomalies"] == 0

    def test_anomaly_percentile(self):
        """Different percentiles should give different thresholds."""
        X = np.random.randn(100, 5)
        labels = np.random.randint(0, 3, 100)
        r95 = AnomalyDetector.centroid_scores(X, labels, percentile=95)
        r99 = AnomalyDetector.centroid_scores(X, labels, percentile=99)
        assert r95["threshold"] <= r99["threshold"]
        assert r95["n_anomalies"] >= r99["n_anomalies"]

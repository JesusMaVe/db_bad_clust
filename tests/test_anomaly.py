"""
test_anomaly.py — Tests for anomaly.py module
"""

import numpy as np
import pytest
from anomaly import AnomalyDetector, OneClassAnalyzer


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


class TestOneClassAnalyzer:
    """Tests for OneClassAnalyzer ML-based anomaly detection."""

    def test_fit_predict(self):
        """Fit on clean data and predict returns correct shape."""
        rng = np.random.RandomState(42)
        X_clean = rng.randn(100, 5)
        X_test = rng.randn(20, 5)

        oc = OneClassAnalyzer(nu=0.1)
        oc.fit(X_clean)
        preds = oc.predict(X_test)

        assert preds.shape == (20,)
        assert set(np.unique(preds)).issubset({-1, 1})

    def test_score_samples(self):
        """score_samples returns correct shape and finite values."""
        rng = np.random.RandomState(42)
        X_clean = rng.randn(100, 5)
        X_test = rng.randn(30, 5)

        oc = OneClassAnalyzer(nu=0.1)
        oc.fit(X_clean)
        scores = oc.score_samples(X_test)

        assert scores.shape == (30,)
        assert np.all(np.isfinite(scores))

    def test_anomaly_detection(self):
        """Clean points should mostly be normal, outliers should be flagged."""
        rng = np.random.RandomState(42)
        X_clean = rng.randn(200, 5)
        X_outliers = rng.uniform(10, 20, size=(20, 5))

        oc = OneClassAnalyzer(nu=0.1)
        oc.fit(X_clean)

        preds_normal = oc.predict(X_clean[:50])
        preds_outliers = oc.predict(X_outliers)

        normal_ratio = (preds_normal == 1).mean()
        outlier_ratio = (preds_outliers == -1).mean()

        assert normal_ratio > 0.7
        assert outlier_ratio > 0.5

    def test_unfitted_raises(self):
        """Calling predict/score before fit raises RuntimeError."""
        oc = OneClassAnalyzer()
        X = np.random.randn(10, 5)

        with pytest.raises(RuntimeError):
            oc.predict(X)
        with pytest.raises(RuntimeError):
            oc.score_samples(X)

    def test_custom_params(self):
        """Custom kernel/nu/gamma are accepted."""
        X = np.random.randn(50, 3)
        oc = OneClassAnalyzer(nu=0.2, kernel="linear", gamma="auto")
        oc.fit(X)
        preds = oc.predict(X[:10])
        assert preds.shape == (10,)

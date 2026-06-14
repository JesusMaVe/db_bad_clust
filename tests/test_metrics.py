"""
test_metrics.py — Tests for metrics.py module
"""

import numpy as np
import pytest
from metrics import ClusteringMetrics


class TestSilhouette:
    """Tests for silhouette score."""

    def test_silhouette_perfect_clusters(self):
        """Well-separated clusters should have high silhouette."""
        X = np.array([
            [0, 0], [0.1, 0], [0, 0.1],
            [10, 10], [10.1, 10], [10, 10.1],
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        score = ClusteringMetrics.silhouette(X, labels)
        assert score > 0.5

    def test_silhouette_single_cluster(self):
        """Single cluster returns 0."""
        X = np.array([[0, 0], [1, 1]])
        labels = np.array([0, 0])
        assert ClusteringMetrics.silhouette(X, labels) == 0.0

    def test_silhouette_all_noise(self):
        """All noise (-1) returns 0."""
        X = np.array([[0, 0], [1, 1]])
        labels = np.array([-1, -1])
        assert ClusteringMetrics.silhouette(X, labels) == 0.0


class TestDaviesBouldin:
    """Tests for Davies-Bouldin index."""

    def test_davies_bouldin_separated_clusters(self):
        """Well-separated clusters should have low DB index."""
        X = np.array([
            [0, 0], [0.1, 0], [0, 0.1],
            [10, 10], [10.1, 10], [10, 10.1],
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        score = ClusteringMetrics.davies_bouldin(X, labels)
        assert score < 1.0

    def test_davies_bouldin_single_cluster(self):
        """Single cluster returns 0."""
        X = np.array([[0, 0], [1, 1]])
        labels = np.array([0, 0])
        assert ClusteringMetrics.davies_bouldin(X, labels) == 0.0


class TestCalinskiHarabasz:
    """Tests for Calinski-Harabasz index."""

    def test_calinski_harabasz_separated_clusters(self):
        """Well-separated clusters should have high CH index."""
        X = np.array([
            [0, 0], [0.1, 0], [0, 0.1],
            [10, 10], [10.1, 10], [10, 10.1],
        ])
        labels = np.array([0, 0, 0, 1, 1, 1])
        score = ClusteringMetrics.calinski_harabasz(X, labels)
        assert score > 100

    def test_calinski_harabasz_single_cluster(self):
        """Single cluster returns 0."""
        X = np.array([[0, 0], [1, 1]])
        labels = np.array([0, 0])
        assert ClusteringMetrics.calinski_harabasz(X, labels) == 0.0


class TestEvaluate:
    """Tests for evaluate method."""

    def test_evaluate_returns_all_keys(self):
        """Evaluate returns all expected metric keys."""
        X = np.random.randn(100, 5)
        labels = np.random.randint(0, 3, 100)
        metrics = ClusteringMetrics().evaluate(X, labels)
        assert "silhouette" in metrics
        assert "davies_bouldin" in metrics
        assert "calinski_harabasz" in metrics
        assert "n_clusters" in metrics
        assert "n_samples" in metrics
        assert "n_noise" in metrics

    def test_evaluate_all_noise(self):
        """All noise samples returns error."""
        X = np.random.randn(10, 5)
        labels = np.full(10, -1)
        result = ClusteringMetrics().evaluate(X, labels)
        assert "error" in result

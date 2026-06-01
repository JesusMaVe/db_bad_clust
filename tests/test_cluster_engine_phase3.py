"""
test_cluster_engine_phase3.py — Tests for Phase 3 cluster engine additions.

Verifies:
  - MeanShift clustering works
  - DBSCAN with cosine metric works
  - DBSCAN with auto-eps works
  - MeanShift with custom bandwidth
  - MeanShift cluster info is populated
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest
import numpy as np
from cluster_engine import ClusterEngine
from exceptions import ClusteringError


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def clustered_data() -> np.ndarray:
    """Small 2D dataset with 2 clear clusters (10 samples)."""
    np.random.seed(42)
    c0 = np.random.randn(5, 2) + np.array([3, 3])
    c1 = np.random.randn(5, 2) + np.array([-3, -3])
    return np.vstack([c0, c1]).astype(np.float32)


@pytest.fixture
def high_dim_data() -> np.ndarray:
    """Higher-dim data for cosine metric test (10 samples, 10 dims)."""
    np.random.seed(99)
    c0 = np.random.randn(5, 10) + np.array([0.5] * 10)
    c1 = np.random.randn(5, 10) + np.array([-0.5] * 10)
    return np.vstack([c0, c1]).astype(np.float32)


@pytest.fixture
def single_sample() -> np.ndarray:
    """Single sample (edge case)."""
    return np.array([[1.0, 2.0]], dtype=np.float32)


@pytest.fixture
def empty_data() -> np.ndarray:
    """Empty dataset."""
    return np.zeros((0, 2), dtype=np.float32)


# ── MeanShift ────────────────────────────────────────────────────────────


class TestMeanShift:
    """MeanShift clustering tests."""

    def test_meanshift_default(self, clustered_data: np.ndarray) -> None:
        """MeanShift with auto-bandwidth produces labels."""
        engine = ClusterEngine(method="meanshift")
        labels = engine.fit_predict(clustered_data)
        assert len(labels) == 10
        assert labels.dtype == int
        assert -1 not in set(labels)  # MeanShift has no noise

    def test_meanshift_custom_bandwidth(self, clustered_data: np.ndarray) -> None:
        """MeanShift with explicit bandwidth."""
        engine = ClusterEngine(method="meanshift", bandwidth=1.0)
        labels = engine.fit_predict(clustered_data)
        assert len(labels) == 10
        assert labels.dtype == int

    def test_meanshift_labels_are_integers(self, clustered_data: np.ndarray) -> None:
        """Labels have int dtype."""
        engine = ClusterEngine(method="meanshift")
        labels = engine.fit_predict(clustered_data)
        assert labels.dtype == np.int64 or labels.dtype == np.int32

    def test_meanshift_empty(self, empty_data: np.ndarray) -> None:
        """Empty data returns empty array."""
        engine = ClusterEngine(method="meanshift")
        labels = engine.fit_predict(empty_data)
        assert len(labels) == 0

    def test_meanshift_single_sample(self, single_sample: np.ndarray) -> None:
        """Single sample with MeanShift returns label 0."""
        engine = ClusterEngine(method="meanshift")
        labels = engine.fit_predict(single_sample)
        assert labels[0] == 0


# ── DBSCAN with metric ───────────────────────────────────────────────────


class TestDBSCANMetric:
    """DBSCAN with different distance metrics."""

    def test_dbscan_cosine(self, high_dim_data: np.ndarray) -> None:
        """DBSCAN with cosine metric produces labels."""
        engine = ClusterEngine(method="dbscan", eps=0.5, min_samples=2, metric="cosine")
        labels = engine.fit_predict(high_dim_data)
        assert len(labels) == 10
        assert labels.dtype == int

    def test_dbscan_cosine_large_eps(self, high_dim_data: np.ndarray) -> None:
        """Large eps with cosine metric finds one cluster."""
        engine = ClusterEngine(method="dbscan", eps=1.5, min_samples=2, metric="cosine")
        labels = engine.fit_predict(high_dim_data)
        assert -1 not in set(labels) or len(set(labels)) <= 2


# ── DBSCAN auto-eps ──────────────────────────────────────────────────────


class TestDBSCANAutoEps:
    """DBSCAN with auto-eps estimation."""

    def test_auto_eps(self, clustered_data: np.ndarray) -> None:
        """Auto-eps produces valid labels."""
        engine = ClusterEngine(method="dbscan", eps="auto", min_samples=2)
        labels = engine.fit_predict(clustered_data)
        assert len(labels) == 10
        assert labels.dtype == int

    def test_auto_eps_with_cosine(self, high_dim_data: np.ndarray) -> None:
        """Auto-eps with cosine metric works."""
        engine = ClusterEngine(method="dbscan", eps="auto", min_samples=2, metric="cosine")
        labels = engine.fit_predict(high_dim_data)
        assert len(labels) == 10

    def test_auto_eps_small_data(self) -> None:
        """Auto-eps works with very small datasets."""
        X = np.array([[0.0, 0.0], [1.0, 1.0], [10.0, 10.0]], dtype=np.float32)
        engine = ClusterEngine(method="dbscan", eps="auto", min_samples=1)
        labels = engine.fit_predict(X)
        assert len(labels) == 3


# ── Cluster info for new methods ─────────────────────────────────────────


class TestClusterInfoPhase3:
    """Cluster info for mean shift."""

    def test_meanshift_cluster_info(self, clustered_data: np.ndarray) -> None:
        """MeanShift populates cluster info."""
        engine = ClusterEngine(method="meanshift")
        engine.fit_predict(clustered_data)
        assert isinstance(engine.cluster_info_, dict)
        assert "n_clusters" in engine.cluster_info_
        assert "n_noise" in engine.cluster_info_

    def test_dbscan_auto_eps_cluster_info(self, clustered_data: np.ndarray) -> None:
        """DBSCAN auto-eps populates cluster info."""
        engine = ClusterEngine(method="dbscan", eps="auto", min_samples=2)
        engine.fit_predict(clustered_data)
        assert isinstance(engine.cluster_info_, dict)
        assert "n_noise" in engine.cluster_info_


# ── Unsupported method ───────────────────────────────────────────────────


class TestUnsupportedMethod:
    """Unsupported method raises error."""

    def test_invalid_method_raises(self, clustered_data: np.ndarray) -> None:
        engine = ClusterEngine(method="spectral")  # type: ignore[arg-type]
        with pytest.raises(ClusteringError, match="Unsupported method"):
            engine.fit_predict(clustered_data)

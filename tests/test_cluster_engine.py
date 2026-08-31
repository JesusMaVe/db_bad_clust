"""
test_cluster_engine.py — Tests for the ClusterEngine module.

Verifies:
  - KMeans, DBSCAN, and Agglomerative produce correct number of labels
  - Labels are integer arrays
  - Empty/edge cases are handled gracefully
  - Unsupported method raises ClusteringError
  - Cluster info is populated after fit_predict
  - get_cluster_members works correctly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import numpy as np
import pytest

from db_bad_clust.clustering.cluster_engine import ClusterEngine
from db_bad_clust.exceptions import ClusteringError

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def clustered_data() -> np.ndarray:
    """Small 2D dataset with 2 clear clusters (10 samples)."""
    np.random.seed(42)
    c0 = np.random.randn(5, 2) + np.array([3, 3])
    c1 = np.random.randn(5, 2) + np.array([-3, -3])
    return np.vstack([c0, c1]).astype(np.float32)


@pytest.fixture
def three_cluster_data() -> np.ndarray:
    """3 clear clusters (15 samples)."""
    np.random.seed(99)
    c0 = np.random.randn(5, 2) + np.array([0, 0])
    c1 = np.random.randn(5, 2) + np.array([5, 5])
    c2 = np.random.randn(5, 2) + np.array([-5, -5])
    return np.vstack([c0, c1, c2]).astype(np.float32)


@pytest.fixture
def single_sample() -> np.ndarray:
    """Just one sample (edge case)."""
    return np.array([[1.0, 2.0]], dtype=np.float32)


@pytest.fixture
def empty_data() -> np.ndarray:
    """Empty dataset (0 samples)."""
    return np.zeros((0, 2), dtype=np.float32)


# ── KMeans ──────────────────────────────────────────────────────────────


class TestKMeans:
    """KMeans clustering tests."""

    def test_default_n_clusters(self, clustered_data: np.ndarray) -> None:
        """Default KMeans creates n_clusters=5 (or fewer if data is smaller)."""
        engine = ClusterEngine(method="kmeans")
        labels = engine.fit_predict(clustered_data)
        # 10 samples → min(5, 10) = 5 clusters by default
        assert len(labels) == 10
        assert labels.dtype == int

    def test_two_clusters(self, clustered_data: np.ndarray) -> None:
        """KMeans with n_clusters=2 produces 2 clusters."""
        engine = ClusterEngine(method="kmeans", n_clusters=2)
        labels = engine.fit_predict(clustered_data)
        assert len(set(labels)) == 2
        assert len(labels) == 10

    def test_three_clusters(self, three_cluster_data: np.ndarray) -> None:
        """KMeans with n_clusters=3 produces 3 clusters."""
        engine = ClusterEngine(method="kmeans", n_clusters=3)
        labels = engine.fit_predict(three_cluster_data)
        assert len(set(labels)) == 3
        assert len(labels) == 15

    def test_more_clusters_than_samples(self, single_sample: np.ndarray) -> None:
        """n_clusters > n_samples caps at n_samples."""
        engine = ClusterEngine(method="kmeans", n_clusters=10)
        labels = engine.fit_predict(single_sample)
        # min(10, 1) = 1 cluster
        assert len(set(labels)) == 1
        assert labels[0] == 0

    def test_labels_are_integers(self, clustered_data: np.ndarray) -> None:
        """Labels array has integer dtype."""
        engine = ClusterEngine(method="kmeans", n_clusters=2)
        labels = engine.fit_predict(clustered_data)
        assert labels.dtype == np.int64 or labels.dtype == np.int32

    def test_random_state_reproducibility(self, clustered_data: np.ndarray) -> None:
        """Same random_state produces same labels."""
        engine1 = ClusterEngine(method="kmeans", n_clusters=2, random_state=42)
        engine2 = ClusterEngine(method="kmeans", n_clusters=2, random_state=42)
        labels1 = engine1.fit_predict(clustered_data)
        labels2 = engine2.fit_predict(clustered_data)
        np.testing.assert_array_equal(labels1, labels2)


# ── DBSCAN ──────────────────────────────────────────────────────────────


class TestDBSCAN:
    """DBSCAN clustering tests."""

    def test_dbscan_default(self, clustered_data: np.ndarray) -> None:
        """DBSCAN with default params produces labels (some may be -1)."""
        engine = ClusterEngine(method="dbscan")
        labels = engine.fit_predict(clustered_data)
        assert len(labels) == 10
        assert labels.dtype == int

    def test_dbscan_eps_large(self, clustered_data: np.ndarray) -> None:
        """Large eps makes DBSCAN find one big cluster."""
        engine = ClusterEngine(method="dbscan", eps=10.0)
        labels = engine.fit_predict(clustered_data)
        # All samples are in one cluster (no -1's)
        assert -1 not in set(labels)
        assert len(set(labels)) == 1

    def test_dbscan_eps_small(self, clustered_data: np.ndarray) -> None:
        """Very small eps makes all samples outliers."""
        engine = ClusterEngine(method="dbscan", eps=0.01, min_samples=1)
        labels = engine.fit_predict(clustered_data)
        # With min_samples=1, each point is its own cluster
        # (since eps is tiny, no two points are close enough)
        # In DBSCAN, with min_samples=1, every point is a core point = cluster
        # Actually with eps=0.01 only really close points group
        assert len(set(labels)) > 1  # at least some outliers or many clusters

    def test_dbscan_min_samples(self, clustered_data: np.ndarray) -> None:
        """Custom min_samples is respected."""
        engine = ClusterEngine(method="dbscan", eps=0.5, min_samples=3)
        labels = engine.fit_predict(clustered_data)
        assert len(labels) == 10


# ── Agglomerative ───────────────────────────────────────────────────────


class TestAgglomerative:
    """Agglomerative clustering tests."""

    def test_agglomerative_default(self, clustered_data: np.ndarray) -> None:
        """Default agglomerative clustering works."""
        engine = ClusterEngine(method="agglomerative", n_clusters=2)
        labels = engine.fit_predict(clustered_data)
        assert len(set(labels)) == 2
        assert len(labels) == 10

    def test_agglomerative_ward_linkage(self, clustered_data: np.ndarray) -> None:
        """Ward linkage produces valid labels."""
        engine = ClusterEngine(method="agglomerative", n_clusters=2, linkage="ward")
        labels = engine.fit_predict(clustered_data)
        assert len(set(labels)) == 2

    def test_agglomerative_single_linkage(self, clustered_data: np.ndarray) -> None:
        """Single linkage is accepted."""
        engine = ClusterEngine(method="agglomerative", n_clusters=2, linkage="single")
        labels = engine.fit_predict(clustered_data)
        assert len(set(labels)) == 2

    def test_agglomerative_three_clusters(self, three_cluster_data: np.ndarray) -> None:
        """Agglomerative with 3 clusters on 3-cluster data."""
        engine = ClusterEngine(method="agglomerative", n_clusters=3)
        labels = engine.fit_predict(three_cluster_data)
        assert len(set(labels)) == 3

    def test_agglomerative_more_clusters_than_samples(self, single_sample: np.ndarray) -> None:
        """n_clusters > n_samples: sklearn requires >= 2 samples."""
        engine = ClusterEngine(method="agglomerative", n_clusters=10)
        with pytest.raises(ValueError, match="minimum of 2 is required"):
            engine.fit_predict(single_sample)


# ── Empty / edge cases ──────────────────────────────────────────────────


class TestEdgeCases:
    """Edge-case handling."""

    def test_empty_dataset(self, empty_data: np.ndarray) -> None:
        """Empty dataset returns empty labels."""
        engine = ClusterEngine(method="kmeans")
        labels = engine.fit_predict(empty_data)
        assert isinstance(labels, np.ndarray)
        assert len(labels) == 0
        assert labels.dtype == int

    def test_empty_dataset_dbscan(self, empty_data: np.ndarray) -> None:
        """Empty dataset with DBSCAN returns empty labels."""
        engine = ClusterEngine(method="dbscan")
        labels = engine.fit_predict(empty_data)
        assert len(labels) == 0

    def test_empty_dataset_agglomerative(self, empty_data: np.ndarray) -> None:
        """Empty dataset with Agglomerative returns empty labels."""
        engine = ClusterEngine(method="agglomerative")
        labels = engine.fit_predict(empty_data)
        assert len(labels) == 0

    def test_single_sample_kmeans(self, single_sample: np.ndarray) -> None:
        """Single sample with KMeans returns label 0."""
        engine = ClusterEngine(method="kmeans", n_clusters=2)
        labels = engine.fit_predict(single_sample)
        assert labels[0] == 0

    def test_single_sample_dbscan(self, single_sample: np.ndarray) -> None:
        """Single sample with DBSCAN returns label 0 (or -1 with default min_samples)."""
        engine = ClusterEngine(method="dbscan", eps=0.5)
        labels = engine.fit_predict(single_sample)
        # With default min_samples=2, a single point is noise
        # Actually, min_samples default is max(2, int(1*0.05)) = max(2, 0) = 2
        # So the single point will be an outlier (-1)
        assert labels[0] == -1 or labels[0] == 0

    def test_single_sample_agglomerative(self, single_sample: np.ndarray) -> None:
        """Single sample with Agglomerative raises (sklearn needs >= 2)."""
        engine = ClusterEngine(method="agglomerative", n_clusters=2)
        with pytest.raises(ValueError, match="minimum of 2 is required"):
            engine.fit_predict(single_sample)


# ── Unsupported method ──────────────────────────────────────────────────


class TestUnsupportedMethod:
    """Unsupported clustering method raises ClusteringError."""

    def test_invalid_method_raises(self, clustered_data: np.ndarray) -> None:
        """Invalid method string raises ClusteringError."""
        engine = ClusterEngine(method="spectral")  # type: ignore[arg-type]
        with pytest.raises(ClusteringError, match="Unsupported method"):
            engine.fit_predict(clustered_data)

    def test_invalid_method_no_side_effects(self, clustered_data: np.ndarray) -> None:
        """Labels is None after a failed fit."""
        engine = ClusterEngine(method="spectral")  # type: ignore[arg-type]
        with pytest.raises(ClusteringError):
            engine.fit_predict(clustered_data)
        assert engine.labels_ is None


# ── Cluster info ────────────────────────────────────────────────────────


class TestClusterInfo:
    """Verify cluster_info_ is populated after fit_predict."""

    def test_cluster_info_exists(self, clustered_data: np.ndarray) -> None:
        """cluster_info_ is a dict with expected keys."""
        engine = ClusterEngine(method="kmeans", n_clusters=2)
        engine.fit_predict(clustered_data)
        assert isinstance(engine.cluster_info_, dict)
        assert "n_clusters" in engine.cluster_info_
        assert "n_noise" in engine.cluster_info_
        assert "cluster_sizes" in engine.cluster_info_
        assert "centroids" in engine.cluster_info_

    def test_cluster_info_n_clusters(self, clustered_data: np.ndarray) -> None:
        """n_clusters in info matches output."""
        engine = ClusterEngine(method="kmeans", n_clusters=2)
        engine.fit_predict(clustered_data)
        assert engine.cluster_info_["n_clusters"] == 2

    def test_cluster_info_sizes(self, clustered_data: np.ndarray) -> None:
        """cluster_sizes sum to total samples."""
        engine = ClusterEngine(method="kmeans", n_clusters=2)
        engine.fit_predict(clustered_data)
        total = sum(engine.cluster_info_["cluster_sizes"].values())
        assert total == 10

    def test_cluster_info_no_noise(self, clustered_data: np.ndarray) -> None:
        """KMeans should have n_noise = 0."""
        engine = ClusterEngine(method="kmeans", n_clusters=2)
        engine.fit_predict(clustered_data)
        assert engine.cluster_info_["n_noise"] == 0

    def test_cluster_info_centroids(self, clustered_data: np.ndarray) -> None:
        """Centroids have the same dimension as input."""
        engine = ClusterEngine(method="kmeans", n_clusters=2)
        engine.fit_predict(clustered_data)
        for centroid in engine.cluster_info_["centroids"].values():
            assert len(centroid) == 2  # 2D input

    def test_cluster_info_empty(self, empty_data: np.ndarray) -> None:
        """Empty data: cluster_info_ stays None (early return in fit_predict)."""
        engine = ClusterEngine(method="kmeans")
        engine.fit_predict(empty_data)
        # Early return for empty data skips _compute_cluster_info
        assert engine.cluster_info_ is None
        assert len(engine.labels_) == 0  # type: ignore[arg-type]

    def test_cluster_info_dbscan_noise(self, clustered_data: np.ndarray) -> None:
        """DBSCAN may have n_noise > 0."""
        engine = ClusterEngine(method="dbscan", eps=0.1, min_samples=2)
        engine.fit_predict(clustered_data)
        # With small eps, there may be noise
        assert "n_noise" in engine.cluster_info_


# ── get_cluster_members ─────────────────────────────────────────────────


class TestGetClusterMembers:
    """Verify get_cluster_members works correctly."""

    def test_get_members(self, clustered_data: np.ndarray) -> None:
        """get_cluster_members returns correct indices."""
        engine = ClusterEngine(method="kmeans", n_clusters=2)
        labels = engine.fit_predict(clustered_data)
        members = engine.get_cluster_members(labels, 0)
        assert isinstance(members, np.ndarray)
        # All returned indices should have label 0
        assert np.all(labels[members] == 0)

    def test_get_members_cluster_id_not_present(self, clustered_data: np.ndarray) -> None:
        """Requesting a non-existent cluster returns empty array."""
        engine = ClusterEngine(method="kmeans", n_clusters=2)
        labels = engine.fit_predict(clustered_data)
        members = engine.get_cluster_members(labels, 99)
        assert len(members) == 0

    def test_get_members_returns_indices(self, clustered_data: np.ndarray) -> None:
        """Members are valid indices into the original array."""
        engine = ClusterEngine(method="kmeans", n_clusters=2)
        labels = engine.fit_predict(clustered_data)
        members = engine.get_cluster_members(labels, 0)
        assert np.all(members >= 0)
        assert np.all(members < len(labels))

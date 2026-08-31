"""
test_dimensionality_reducer.py — Tests for the DimensionalityReducer module.

Verifies:
  - PCA reduces dimensions correctly
  - SVD works
  - explained_variance_ratio is populated after fit_transform
  - Raises ClusteringError for invalid method
  - t-SNE works (one-shot transform only)
  - Transform after fit works for PCA/SVD
  - Edge cases (single sample, fewer features than components)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import numpy as np
import pytest

from db_bad_clust.clustering.dimensionality_reducer import DimensionalityReducer
from db_bad_clust.exceptions import ClusteringError

# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def tsne_kwargs() -> dict:
    """t-SNE needs perplexity < n_samples for the default perplexity=30."""
    return {"perplexity": 5}


@pytest.fixture
def high_dim_data() -> np.ndarray:
    """20 samples with 50 features (simulates reduced ~785-dim data)."""
    np.random.seed(42)
    return np.random.randn(20, 50).astype(np.float32)


@pytest.fixture
def moderate_data() -> np.ndarray:
    """50 samples with 10 features."""
    np.random.seed(1)
    return np.random.randn(50, 10).astype(np.float32)


@pytest.fixture
def single_sample() -> np.ndarray:
    """Single 50-dim sample."""
    return np.random.randn(1, 50).astype(np.float32)


@pytest.fixture
def two_samples() -> np.ndarray:
    """Two 50-dim samples."""
    return np.random.randn(2, 50).astype(np.float32)


# ── PCA ─────────────────────────────────────────────────────────────────


class TestPCA:
    """PCA reduction tests."""

    def test_pca_reduces_dimensions(self, high_dim_data: np.ndarray) -> None:
        """PCA reduces 50 → 10 dimensions."""
        reducer = DimensionalityReducer(method="pca", n_components=10)
        reduced = reducer.fit_transform(high_dim_data)
        assert reduced.shape == (20, 10)

    def test_pca_two_components(self, high_dim_data: np.ndarray) -> None:
        """PCA reduces 50 → 2."""
        reducer = DimensionalityReducer(method="pca", n_components=2)
        reduced = reducer.fit_transform(high_dim_data)
        assert reduced.shape == (20, 2)

    def test_pca_fitted_flag(self, high_dim_data: np.ndarray) -> None:
        """_fitted is True after fit_transform."""
        reducer = DimensionalityReducer(method="pca", n_components=5)
        reducer.fit_transform(high_dim_data)
        assert reducer._fitted

    def test_pca_explained_variance(self, high_dim_data: np.ndarray) -> None:
        """explained_variance_ratio is populated after fit."""
        reducer = DimensionalityReducer(method="pca", n_components=5)
        reducer.fit_transform(high_dim_data)
        evr = reducer.explained_variance_ratio
        assert evr is not None
        assert len(evr) == 5
        # Values should be between 0 and 1, sum should be positive
        assert np.all(evr >= 0)
        assert np.all(evr <= 1)

    def test_pca_transform_new_data(self, high_dim_data: np.ndarray) -> None:
        """transform() works after fit_transform for PCA."""
        reducer = DimensionalityReducer(method="pca", n_components=5)
        reducer.fit_transform(high_dim_data)
        new_data = np.random.randn(3, 50).astype(np.float32)
        transformed = reducer.transform(new_data)
        assert transformed.shape == (3, 5)

    def test_pca_fewer_samples_than_components(self, two_samples: np.ndarray) -> None:
        """n_components > n_samples caps at n_samples."""
        reducer = DimensionalityReducer(method="pca", n_components=10)
        reduced = reducer.fit_transform(two_samples)
        # min(10, 2, 50) = 2
        assert reduced.shape == (2, 2)

    def test_pca_single_sample(self, single_sample: np.ndarray) -> None:
        """Single sample returns unchanged (guard condition)."""
        reducer = DimensionalityReducer(method="pca", n_components=5)
        reduced = reducer.fit_transform(single_sample)
        # min(5, 1, 50) = 1, < 1? No. So it proceeds to PCA with n=1
        # PCA with n_components=1 works
        assert reduced.shape == (1, 1)

    def test_pca_components_property(self, high_dim_data: np.ndarray) -> None:
        """components_ property returns principal components."""
        reducer = DimensionalityReducer(method="pca", n_components=5)
        reducer.fit_transform(high_dim_data)
        components = reducer.components_
        assert components is not None
        assert components.shape == (5, 50)


# ── SVD ─────────────────────────────────────────────────────────────────


class TestSVD:
    """TruncatedSVD reduction tests."""

    def test_svd_reduces_dimensions(self, high_dim_data: np.ndarray) -> None:
        """SVD reduces 50 → 10 dimensions."""
        reducer = DimensionalityReducer(method="svd", n_components=10)
        reduced = reducer.fit_transform(high_dim_data)
        assert reduced.shape == (20, 10)

    def test_svd_two_components(self, high_dim_data: np.ndarray) -> None:
        """SVD reduces 50 → 2."""
        reducer = DimensionalityReducer(method="svd", n_components=2)
        reduced = reducer.fit_transform(high_dim_data)
        assert reduced.shape == (20, 2)

    def test_svd_explained_variance(self, high_dim_data: np.ndarray) -> None:
        """explained_variance_ratio is populated after SVD fit."""
        reducer = DimensionalityReducer(method="svd", n_components=5)
        reducer.fit_transform(high_dim_data)
        evr = reducer.explained_variance_ratio
        assert evr is not None
        assert len(evr) == 5
        assert np.all(evr >= 0)

    def test_svd_transform_new_data(self, high_dim_data: np.ndarray) -> None:
        """transform() works after fit_transform for SVD."""
        reducer = DimensionalityReducer(method="svd", n_components=5)
        reducer.fit_transform(high_dim_data)
        new_data = np.random.randn(3, 50).astype(np.float32)
        transformed = reducer.transform(new_data)
        assert transformed.shape == (3, 5)

    def test_svd_single_sample(self, single_sample: np.ndarray) -> None:
        """SVD with single sample returns reduced array."""
        reducer = DimensionalityReducer(method="svd", n_components=2)
        reduced = reducer.fit_transform(single_sample)
        # min(2, 1, 50) = 1, < 1? No, it's 1. So it proceeds.
        assert reduced.shape == (1, 1)


# ── t-SNE ───────────────────────────────────────────────────────────────


class TestTSNE:
    """t-SNE reduction tests (visualization only, no transform)."""

    def test_tsne_reduces_dimensions(self, high_dim_data: np.ndarray, tsne_kwargs: dict) -> None:
        """t-SNE reduces 50 → 2."""
        reducer = DimensionalityReducer(method="tsne", n_components=2, **tsne_kwargs)
        reduced = reducer.fit_transform(high_dim_data)
        assert reduced.shape == (20, 2)

    def test_tsne_no_transform(self, high_dim_data: np.ndarray, tsne_kwargs: dict) -> None:
        """t-SNE does not support transform() — model is None."""
        reducer = DimensionalityReducer(method="tsne", n_components=2, **tsne_kwargs)
        reducer.fit_transform(high_dim_data)
        assert reducer._model is None  # t-SNE stores no model
        with pytest.raises(ClusteringError, match="Transform not supported"):
            reducer.transform(high_dim_data)

    def test_tsne_explained_variance_none(
        self, high_dim_data: np.ndarray, tsne_kwargs: dict
    ) -> None:
        """t-SNE does not provide explained_variance_ratio."""
        reducer = DimensionalityReducer(method="tsne", n_components=2, **tsne_kwargs)
        reducer.fit_transform(high_dim_data)
        assert reducer.explained_variance_ratio is None

    def test_tsne_components_none(self, high_dim_data: np.ndarray, tsne_kwargs: dict) -> None:
        """t-SNE does not provide components_."""
        reducer = DimensionalityReducer(method="tsne", n_components=2, **tsne_kwargs)
        reducer.fit_transform(high_dim_data)
        assert reducer.components_ is None


# ── Invalid method ──────────────────────────────────────────────────────


class TestInvalidMethod:
    """Invalid method raises ClusteringError."""

    def test_invalid_method_raises(self, high_dim_data: np.ndarray) -> None:
        """Unrecognized method string raises ClusteringError."""
        reducer = DimensionalityReducer(method="invalid_method")  # type: ignore[arg-type]
        with pytest.raises(ClusteringError, match="Unsupported method"):
            reducer.fit_transform(high_dim_data)

    def test_case_sensitive(self, high_dim_data: np.ndarray) -> None:
        """Method names are case-sensitive (PCA != pca)."""
        reducer = DimensionalityReducer(method="PCA")  # type: ignore[arg-type]
        with pytest.raises(ClusteringError):
            reducer.fit_transform(high_dim_data)


# ── Transform without fit ───────────────────────────────────────────────


class TestTransformWithoutFit:
    """Calling transform() before fit_transform() raises error."""

    def test_transform_before_fit(self, high_dim_data: np.ndarray) -> None:
        """transform() before fit_transform raises ClusteringError."""
        reducer = DimensionalityReducer(method="pca", n_components=5)
        with pytest.raises(ClusteringError, match="not been fitted"):
            reducer.transform(high_dim_data)


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """Various edge cases."""

    def test_reduce_to_one_component(self, moderate_data: np.ndarray, tsne_kwargs: dict) -> None:
        """Reduction to 1D works for all methods."""
        for method, kwargs in (("pca", {}), ("svd", {}), ("tsne", tsne_kwargs)):
            reducer = DimensionalityReducer(method=method, n_components=1, **kwargs)  # type: ignore[arg-type]
            reduced = reducer.fit_transform(moderate_data)
            assert reduced.shape == (50, 1), f"{method} failed for 1D"

    def test_n_components_equal_to_features(self, moderate_data: np.ndarray) -> None:
        """n_components == n_features produces same number of output dims."""
        reducer = DimensionalityReducer(method="pca", n_components=10)
        reduced = reducer.fit_transform(moderate_data)
        assert reduced.shape == (50, 10)

    def test_n_components_greater_than_features(self, moderate_data: np.ndarray) -> None:
        """n_components > n_features caps at min(n_samples, n_features)."""
        reducer = DimensionalityReducer(method="pca", n_components=20)
        reduced = reducer.fit_transform(moderate_data)
        # min(20, 50, 10) = 10
        assert reduced.shape == (50, 10)

    def test_random_state_reproducibility(self, high_dim_data: np.ndarray) -> None:
        """Same random_state produces same reduction (PCA)."""
        reducer1 = DimensionalityReducer(method="pca", n_components=5, random_state=42)
        reducer2 = DimensionalityReducer(method="pca", n_components=5, random_state=42)
        r1 = reducer1.fit_transform(high_dim_data)
        r2 = reducer2.fit_transform(high_dim_data)
        np.testing.assert_array_almost_equal(r1, r2)

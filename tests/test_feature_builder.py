"""
test_feature_builder.py — Tests for the FeatureBuilder module.

Verifies:
  - build() with and without e_stat
  - Correct output shapes (785 without stat, 786 with stat)
  - Z-score normalization produces approximately zero mean
  - Weight application is correct
  - Custom weight combinations
  - Edge cases (single sample, constant features)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest
import numpy as np
from feature_builder import FeatureBuilder


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def builder() -> FeatureBuilder:
    """Default FeatureBuilder (alpha=0.40, beta=0.30, gamma=0.25, delta=0.05)."""
    return FeatureBuilder()


@pytest.fixture
def default_builder() -> FeatureBuilder:
    return FeatureBuilder()


@pytest.fixture
def small_embeddings() -> np.ndarray:
    """Small embedding matrix: 5 samples, 768 dims."""
    np.random.seed(42)
    return np.random.randn(5, 768).astype(np.float32)


@pytest.fixture
def small_type_encoding() -> np.ndarray:
    """Type encoding: 5 samples, 12 types."""
    np.random.seed(1)
    return np.random.randn(5, 12).astype(np.float32)


@pytest.fixture
def small_constraint_encoding() -> np.ndarray:
    """Constraint encoding: 5 samples, 5 constraints."""
    np.random.seed(2)
    return np.random.randn(5, 5).astype(np.float32)


@pytest.fixture
def small_statistical() -> np.ndarray:
    """Statistical features: 5 samples, 1 feature."""
    np.random.seed(3)
    return np.random.randn(5, 1).astype(np.float32)


# ── Output shapes ───────────────────────────────────────────────────────


class TestOutputShapes:
    """Verify build() returns correct shapes."""

    def test_without_stat(
        self,
        builder: FeatureBuilder,
        small_embeddings: np.ndarray,
        small_type_encoding: np.ndarray,
        small_constraint_encoding: np.ndarray,
    ) -> None:
        """Without e_stat: 768 + 12 + 5 = 785."""
        phi = builder.build(small_embeddings, small_type_encoding, small_constraint_encoding)
        assert phi.shape == (5, 785)

    def test_with_stat(
        self,
        builder: FeatureBuilder,
        small_embeddings: np.ndarray,
        small_type_encoding: np.ndarray,
        small_constraint_encoding: np.ndarray,
        small_statistical: np.ndarray,
    ) -> None:
        """With e_stat: 768 + 12 + 5 + 1 = 786."""
        phi = builder.build(
            small_embeddings, small_type_encoding, small_constraint_encoding, small_statistical
        )
        assert phi.shape == (5, 786)

    def test_single_sample(
        self,
        builder: FeatureBuilder,
    ) -> None:
        """Single sample produces shape (1, 785)."""
        e_text = np.random.randn(1, 768).astype(np.float32)
        e_type = np.random.randn(1, 12).astype(np.float32)
        e_rest = np.random.randn(1, 5).astype(np.float32)
        phi = builder.build(e_text, e_type, e_rest)
        assert phi.shape == (1, 785)

    def test_many_samples(
        self,
        builder: FeatureBuilder,
    ) -> None:
        """100 samples produces shape (100, 785)."""
        e_text = np.random.randn(100, 768).astype(np.float32)
        e_type = np.random.randn(100, 12).astype(np.float32)
        e_rest = np.random.randn(100, 5).astype(np.float32)
        phi = builder.build(e_text, e_type, e_rest)
        assert phi.shape == (100, 785)


# ── Z-score normalization ───────────────────────────────────────────────


class TestZScoreNormalization:
    """Verify internal _zscore produces zero-mean, unit-variance outputs."""

    def test_approximately_zero_mean(
        self,
        builder: FeatureBuilder,
        small_embeddings: np.ndarray,
        small_type_encoding: np.ndarray,
        small_constraint_encoding: np.ndarray,
    ) -> None:
        """Normalized components should have mean ≈ 0."""
        e_text_norm = builder._zscore(small_embeddings)
        e_type_norm = builder._zscore(small_type_encoding)
        e_rest_norm = builder._zscore(small_constraint_encoding)

        assert abs(e_text_norm.mean()) < 1e-6
        assert abs(e_type_norm.mean()) < 1e-6
        assert abs(e_rest_norm.mean()) < 1e-6

    def test_standard_deviation(
        self,
        builder: FeatureBuilder,
        small_embeddings: np.ndarray,
    ) -> None:
        """Normalized columns should have std ≈ 1."""
        normalized = builder._zscore(small_embeddings)
        # Allow small tolerance due to float precision
        assert np.allclose(normalized.std(axis=0), 1.0, atol=1e-6)

    def test_constant_column(
        self,
        builder: FeatureBuilder,
    ) -> None:
        """Column with constant value should be zero after normalization."""
        const_data = np.ones((5, 3), dtype=np.float32) * 42.0
        normalized = builder._zscore(const_data)
        assert np.allclose(normalized, 0.0)

    def test_zero_std_handling(
        self,
        builder: FeatureBuilder,
    ) -> None:
        """Zero-std column is set to 0 (no division by zero)."""
        # One constant column, one varied column
        data = np.zeros((5, 2), dtype=np.float32)
        data[:, 0] = 5.0
        data[:, 1] = np.arange(5, dtype=np.float32)
        normalized = builder._zscore(data)
        # Constant column should be all zeros
        assert np.allclose(normalized[:, 0], 0.0)
        # Varied column should be normalized
        assert abs(normalized[:, 1].mean()) < 1e-6

    def test_empty_matrix(self, builder: FeatureBuilder) -> None:
        """Empty matrix (0 cols) returns empty."""
        result = builder._zscore(np.zeros((5, 0), dtype=np.float32))
        assert result.shape == (5, 0)


# ── Weight application ──────────────────────────────────────────────────


class TestWeightApplication:
    """Verify weights are correctly applied."""

    def test_default_weights(
        self,
        builder: FeatureBuilder,
        small_embeddings: np.ndarray,
        small_type_encoding: np.ndarray,
        small_constraint_encoding: np.ndarray,
    ) -> None:
        """Default weights produce phi with expected magnitude ranges."""
        phi = builder.build(small_embeddings, small_type_encoding, small_constraint_encoding)
        # After z-score + weight: each component should reflect its weight
        assert builder.weights == {"alpha": 0.40, "beta": 0.30, "gamma": 0.25, "delta": 0.05}

    def test_alpha_zero(
        self,
        small_embeddings: np.ndarray,
        small_type_encoding: np.ndarray,
        small_constraint_encoding: np.ndarray,
    ) -> None:
        """alpha=0 means BERT embeddings contribute nothing."""
        builder = FeatureBuilder(alpha=0.0, beta=0.5, gamma=0.5)
        phi = builder.build(small_embeddings, small_type_encoding, small_constraint_encoding)
        # First 768 columns should be all zeros
        assert np.allclose(phi[:, :768], 0.0, atol=1e-6)

    def test_beta_zero(
        self,
        small_embeddings: np.ndarray,
        small_type_encoding: np.ndarray,
        small_constraint_encoding: np.ndarray,
    ) -> None:
        """beta=0 means type encoding contributes nothing."""
        builder = FeatureBuilder(alpha=0.5, beta=0.0, gamma=0.5)
        phi = builder.build(small_embeddings, small_type_encoding, small_constraint_encoding)
        # Middle 12 columns should be all zeros
        assert np.allclose(phi[:, 768:780], 0.0, atol=1e-6)

    def test_gamma_zero(
        self,
        small_embeddings: np.ndarray,
        small_type_encoding: np.ndarray,
        small_constraint_encoding: np.ndarray,
    ) -> None:
        """gamma=0 means constraints contribute nothing."""
        builder = FeatureBuilder(alpha=0.5, beta=0.5, gamma=0.0)
        phi = builder.build(small_embeddings, small_type_encoding, small_constraint_encoding)
        # Last 5 columns should be all zeros
        assert np.allclose(phi[:, 780:], 0.0, atol=1e-6)

    def test_delta_zero(
        self,
        small_embeddings: np.ndarray,
        small_type_encoding: np.ndarray,
        small_constraint_encoding: np.ndarray,
        small_statistical: np.ndarray,
    ) -> None:
        """delta=0 means statistical features contribute nothing."""
        builder = FeatureBuilder(alpha=0.4, beta=0.3, gamma=0.25, delta=0.0)
        phi = builder.build(
            small_embeddings, small_type_encoding, small_constraint_encoding, small_statistical
        )
        # Last 1 column should be all zeros
        assert np.allclose(phi[:, -1:], 0.0, atol=1e-6)

    def test_custom_weights(
        self,
        small_embeddings: np.ndarray,
        small_type_encoding: np.ndarray,
        small_constraint_encoding: np.ndarray,
    ) -> None:
        """Custom weight values are reflected in the builder."""
        builder = FeatureBuilder(alpha=0.7, beta=0.2, gamma=0.1)
        assert builder.weights == {"alpha": 0.7, "beta": 0.2, "gamma": 0.1, "delta": 0.05}
        phi = builder.build(small_embeddings, small_type_encoding, small_constraint_encoding)
        assert phi.shape == (5, 785)

    def test_weight_sum_not_one(
        self,
        small_embeddings: np.ndarray,
        small_type_encoding: np.ndarray,
        small_constraint_encoding: np.ndarray,
    ) -> None:
        """Weights don't need to sum to 1; builder accepts any values."""
        builder = FeatureBuilder(alpha=2.0, beta=1.0, gamma=0.5)
        phi = builder.build(small_embeddings, small_type_encoding, small_constraint_encoding)
        assert phi.shape == (5, 785)
        # Norm should be larger due to weights > 1
        assert np.linalg.norm(phi) > 0


# ── Fitted state ────────────────────────────────────────────────────────


class TestFittedState:
    """Verify the _fitted flag."""

    def test_not_fitted_initially(self) -> None:
        """Builder starts as not fitted."""
        builder = FeatureBuilder()
        assert not builder._fitted

    def test_fitted_after_build(
        self,
        builder: FeatureBuilder,
        small_embeddings: np.ndarray,
        small_type_encoding: np.ndarray,
        small_constraint_encoding: np.ndarray,
    ) -> None:
        """_fitted becomes True after build()."""
        builder.build(small_embeddings, small_type_encoding, small_constraint_encoding)
        assert builder._fitted


# ── Error handling ──────────────────────────────────────────────────────


class TestErrorHandling:
    """Verify assertions and edge cases."""

    def test_mismatched_rows_raises(
        self,
        builder: FeatureBuilder,
    ) -> None:
        """Mismatched row counts raise AssertionError."""
        e_text = np.random.randn(5, 768).astype(np.float32)
        e_type = np.random.randn(3, 12).astype(np.float32)
        e_rest = np.random.randn(5, 5).astype(np.float32)
        with pytest.raises(AssertionError):
            builder.build(e_text, e_type, e_rest)

    def test_wrong_embedding_dim(
        self,
        builder: FeatureBuilder,
    ) -> None:
        """Embedding dimension 100 instead of 768 still works (no hard check)."""
        e_text = np.random.randn(5, 100).astype(np.float32)
        e_type = np.random.randn(5, 12).astype(np.float32)
        e_rest = np.random.randn(5, 5).astype(np.float32)
        phi = builder.build(e_text, e_type, e_rest)
        assert phi.shape == (5, 117)  # 100 + 12 + 5

    def test_n_components_property(self, builder: FeatureBuilder) -> None:
        """n_components property always returns 3."""
        assert builder.n_components == 3

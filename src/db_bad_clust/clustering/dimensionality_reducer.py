"""
dimensionality_reducer.py — Dimensionality reduction for the composite phi vector

Purpose:
  Reduce the composite vector phi(aj) from 785 dimensions to a
  lower-dimensional space for clustering and visualization.

  Supported methods:
    - PCA (deterministic, default)
    - UMAP (non-linear, preserves global structure)
    - TruncatedSVD (for sparse matrices)
    - t-SNE (visualization only, no transform)

Usage:
  reducer = DimensionalityReducer(method='pca', n_components=10)
  X_reduced = reducer.fit_transform(phi)
  # -> numpy array shape (N, 10)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np

from db_bad_clust.exceptions import ClusteringError

logger = logging.getLogger(__name__)


class DimensionalityReducer:
    """
    Reduces dimensionality of the composite phi vector.
    Caches the fitted model for transforming new data.
    """

    def __init__(
        self,
        method: Literal["pca", "umap", "svd", "tsne"] = "pca",
        n_components: int = 10,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            method: Reduction algorithm ('pca', 'umap', 'svd', 'tsne').
            n_components: Output dimensions.
            random_state: Random seed for reproducibility.
            **kwargs: Additional arguments for the chosen method.
        """
        self.method = method
        self.n_components = n_components
        self.random_state = random_state
        self.kwargs = kwargs
        self._model = None
        self._fitted = False

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and transform the data.

        Args:
            X: numpy array shape (N, D) with composite vectors.

        Returns:
            numpy array shape (N, n_components).
        """
        n_samples, n_features = X.shape

        if min(self.n_components, n_samples, n_features) < 1:
            return X

        if self.method == "pca":
            return self._fit_transform_pca(X)
        elif self.method == "umap":
            return self._fit_transform_umap(X)
        elif self.method == "svd":
            return self._fit_transform_svd(X)
        elif self.method == "tsne":
            return self._fit_transform_tsne(X)
        else:
            raise ClusteringError(f"Unsupported method: {self.method}")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform new data using the fitted model.

        Only available for PCA and SVD (not for UMAP/t-SNE).
        """
        if not self._fitted:
            raise ClusteringError("Model has not been fitted. Call fit_transform first.")
        if self._model is None:
            raise ClusteringError(f"Transform not supported for method '{self.method}'")
        return self._model.transform(X)

    # ── PCA ────────────────────────────────────────────────────────────

    def _fit_transform_pca(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import PCA

        n = min(self.n_components, *X.shape)
        self._model = PCA(n_components=n, random_state=self.random_state, **self.kwargs)
        result = self._model.fit_transform(X)
        logger.info(
            "PCA: %s -> %s dims, explained variance=%s",
            X.shape[1],
            n,
            self._model.explained_variance_ratio_.sum(),
        )
        self._fitted = True
        return result

    # ── UMAP ───────────────────────────────────────────────────────────

    def _fit_transform_umap(self, X: np.ndarray) -> np.ndarray:
        try:
            import umap
        except ImportError:
            logger.warning("umap-learn not installed. Using PCA as fallback.")
            return self._fit_transform_pca(X)

        n = min(self.n_components, *X.shape)

        # Performance-optimized defaults
        umap_kwargs = {
            "n_neighbors": 15,
            "min_dist": 0.1,
            "metric": "euclidean",
            "n_jobs": -1,  # Use all CPU cores
        }
        umap_kwargs.update(self.kwargs)

        self._model = umap.UMAP(
            n_components=n,
            random_state=self.random_state,
            **umap_kwargs,
        )
        result = self._model.fit_transform(X)
        logger.info("UMAP: %s -> %s dims (n_neighbors=%s, metric=%s)",
                    X.shape[1], n, umap_kwargs["n_neighbors"], umap_kwargs["metric"])
        self._fitted = True
        return result

    # ── Truncated SVD ──────────────────────────────────────────────────

    def _fit_transform_svd(self, X: np.ndarray) -> np.ndarray:
        from sklearn.decomposition import TruncatedSVD

        n = min(self.n_components, *X.shape)
        self._model = TruncatedSVD(n_components=n, random_state=self.random_state, **self.kwargs)
        result = self._model.fit_transform(X)
        logger.info(
            "SVD: %s -> %s dims, explained variance=%s",
            X.shape[1],
            n,
            self._model.explained_variance_ratio_.sum(),
        )
        self._fitted = True
        return result

    # ── t-SNE (transform only, no inverse) ─────────────────────────────

    def _fit_transform_tsne(self, X: np.ndarray) -> np.ndarray:
        from sklearn.manifold import TSNE

        n = min(self.n_components, *X.shape)
        model = TSNE(n_components=n, random_state=self.random_state, **self.kwargs)
        result = model.fit_transform(X)
        logger.info("t-SNE: %s -> %s dims", X.shape[1], n)
        self._fitted = True
        return result

    # ── Utilities ──────────────────────────────────────────────────────

    @property
    def explained_variance_ratio(self) -> np.ndarray | None:
        """Explained variance per component (PCA/SVD only)."""
        if hasattr(self._model, "explained_variance_ratio_"):
            return self._model.explained_variance_ratio_
        return None

    @property
    def components_(self) -> np.ndarray | None:
        """Principal components (PCA/SVD only)."""
        if hasattr(self._model, "components_"):
            return self._model.components_
        return None

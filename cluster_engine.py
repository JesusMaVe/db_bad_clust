"""
cluster_engine.py — Clustering engine for attribute vectors

Purpose:
  Group database columns into clusters based on their reduced
  composite phi vectors. Clusters reveal schema design patterns
  and anti-patterns.

  Supported methods:
    - KMeans (partitional, default)
    - DBSCAN (density-based, detects outliers)
    - Agglomerative (hierarchical)
    - MeanShift (mode-seeking, auto-bandwidth)

Usage:
  engine = ClusterEngine(method='kmeans', n_clusters=5)
  labels = engine.fit_predict(X_reduced)
  # -> numpy array shape (N,) with cluster ids (-1 = outlier)
"""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
from exceptions import ClusteringError

logger = logging.getLogger(__name__)


class ClusterEngine:
    """
    Groups attribute vectors into semantic clusters.

    Output attributes (available after fit_predict):
      labels_       : ndarray (N,) — cluster label per sample
      cluster_info_ : dict — clustering metrics and metadata
    """

    def __init__(
        self,
        method: Literal["kmeans", "dbscan", "agglomerative", "meanshift", "hdbscan"] = "kmeans",
        n_clusters: int = 5,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Args:
            method: Clustering algorithm.
            n_clusters: Number of clusters (KMeans, Agglomerative).
            random_state: Random seed for reproducibility.
            **kwargs: Method-specific arguments:
                - dbscan: eps (default 0.5), min_samples (default 5),
                  metric (default 'euclidean', can be 'cosine')
                - agglomerative: linkage (default 'ward')
                - meanshift: bandwidth (default None = auto)
        """
        self.method = method
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kwargs = kwargs
        self._model = None
        self.labels_: np.ndarray | None = None
        self.cluster_info_: dict[str, Any] | None = None

    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and predict cluster labels.

        Args:
            X: numpy array shape (N, d) with reduced vectors.

        Returns:
            numpy array shape (N,) with cluster labels (-1 = outlier).
        """
        if X.shape[0] == 0:
            logger.warning("Empty matrix, returning empty array")
            self.labels_ = np.array([], dtype=int)
            return self.labels_

        if self.method == "kmeans":
            labels = self._fit_kmeans(X)
        elif self.method == "dbscan":
            labels = self._fit_dbscan(X)
        elif self.method == "agglomerative":
            labels = self._fit_agglomerative(X)
        elif self.method == "meanshift":
            labels = self._fit_meanshift(X)
        elif self.method == "hdbscan":
            labels = self._fit_hdbscan(X)
        else:
            raise ClusteringError(f"Unsupported method: {self.method}")

        self.labels_ = labels
        self._compute_cluster_info(X)
        return labels

    # ── KMeans ─────────────────────────────────────────────────────────

    def _fit_kmeans(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import KMeans

        n = min(self.n_clusters, X.shape[0])
        self._model = KMeans(
            n_clusters=n,
            random_state=self.random_state,
            n_init="auto",
            **self.kwargs,
        )
        labels = self._model.fit_predict(X)
        logger.info(
            "KMeans: %s samples -> %s clusters (inertia=%s)",
            X.shape[0],
            n,
            self._model.inertia_,
        )
        return labels.astype(int)

    # ── DBSCAN ─────────────────────────────────────────────────────────

    @staticmethod
    def _estimate_eps_k_distance(X: np.ndarray, k: int = 5) -> float:
        """Estimate eps via k-distance graph knee detection.

        Finds the knee/elbow in the sorted k-distance curve using the
        maximum distance to the line connecting the first and last points.

        Args:
            X: Data array shape (N, d).
            k: Number of nearest neighbors (default 5).

        Returns:
            Estimated eps value at the knee of the k-distance curve.
        """
        from sklearn.neighbors import NearestNeighbors
        n = min(k + 1, X.shape[0])
        nn = NearestNeighbors(n_neighbors=n)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        di = np.sort(distances[:, min(k, distances.shape[1] - 1)])

        # Knee detection: find point with max perpendicular distance
        # to the line connecting (0, di[0]) and (len(di)-1, di[-1])
        n_pts = len(di)
        if n_pts < 4:
            return max(float(di[-1]), 1e-6)

        x0, y0 = 0.0, float(di[0])
        x1, y1 = float(n_pts - 1), float(di[-1])
        dx = x1 - x0
        dy = y1 - y0
        line_len_sq = dx * dx + dy * dy
        if line_len_sq == 0.0:
            return max(float(di[n_pts // 2]), 1e-6)

        # Vectorized knee detection
        indices = np.arange(1, n_pts - 1)
        xi = indices.astype(float)
        yi = di[1:-1].astype(float)

        t = ((xi - x0) * dx + (yi - y0) * dy) / line_len_sq
        proj_x = x0 + t * dx
        proj_y = y0 + t * dy
        dists = (xi - proj_x) ** 2 + (yi - proj_y) ** 2

        knee_idx = int(np.argmax(dists)) + 1  # +1 because we started from index 1

        eps: float = float(di[knee_idx])
        return max(eps, 1e-6)

    def _fit_dbscan(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import DBSCAN

        eps = self.kwargs.get("eps", 0.5)
        if isinstance(eps, str) and eps == "auto":
            eps = self._estimate_eps_k_distance(X)
            logger.info("  Auto-eps estimated: %.4f", eps)
        min_samples = self.kwargs.get("min_samples", max(2, int(X.shape[0] * 0.05)))
        metric: str = self.kwargs.get("metric", "euclidean")
        self._model = DBSCAN(eps=eps, min_samples=min_samples, metric=metric)
        labels = self._model.fit_predict(X)
        n_clusters_found = len(set(labels) - {-1})
        n_noise = (labels == -1).sum()
        logger.info(
            "DBSCAN: %s samples -> %s clusters + %s outliers (eps=%s, min_samples=%s, metric=%s)",
            X.shape[0],
            n_clusters_found,
            n_noise,
            eps,
            min_samples,
            metric,
        )
        return labels.astype(int)

    # ── Agglomerative ──────────────────────────────────────────────────

    def _fit_agglomerative(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import AgglomerativeClustering

        n = min(self.n_clusters, X.shape[0])
        linkage = self.kwargs.get("linkage", "ward")
        self._model = AgglomerativeClustering(
            n_clusters=n,
            linkage=linkage,
        )
        labels = self._model.fit_predict(X)
        logger.info(
            "Agglomerative: %s samples -> %s clusters (linkage=%s)",
            X.shape[0],
            n,
            linkage,
        )
        return labels.astype(int)

    # ── MeanShift ──────────────────────────────────────────────────────

    def _fit_meanshift(self, X: np.ndarray) -> np.ndarray:
        from sklearn.cluster import MeanShift, estimate_bandwidth

        bandwidth = self.kwargs.get("bandwidth", None)
        if bandwidth is None:
            bandwidth = estimate_bandwidth(X, quantile=0.3, random_state=self.random_state)
            logger.info("  Auto-bandwidth estimated: %.4f", bandwidth)
        # sklearn requires bandwidth > 0 for MeanShift; fallback for 0.0 (single sample)
        if bandwidth is not None and bandwidth <= 0.0:
            bandwidth = 1.0
            logger.info("  Bandwidth was <= 0, using default 1.0")
        self._model = MeanShift(
            bandwidth=bandwidth, **{k: v for k, v in self.kwargs.items() if k not in ("bandwidth",)}
        )
        labels = self._model.fit_predict(X)
        n_clusters_found = len(set(labels))
        logger.info(
            "MeanShift: %s samples -> %s clusters (bandwidth=%s)",
            X.shape[0],
            n_clusters_found,
            bandwidth,
        )
        return labels.astype(int)

    # ── HDBSCAN ───────────────────────────────────────────────────────────

    def _fit_hdbscan(self, X: np.ndarray) -> np.ndarray:
        try:
            import hdbscan as _hdbscan
        except ImportError:
            raise ClusteringError(
                "hdbscan is not installed. Run: pip install hdbscan"
            ) from None

        min_cluster_size: int = int(self.kwargs.get("min_cluster_size", 5))
        min_samples: int | None = self.kwargs.get("min_samples", None)
        metric: str = self.kwargs.get("metric", "euclidean")
        cluster_selection_epsilon: float = self.kwargs.get(
            "cluster_selection_epsilon", 0.0
        )

        self._model = _hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric,
            cluster_selection_epsilon=cluster_selection_epsilon,
            gen_min_span_tree=True,
            prediction_data=True,
        )
        labels = self._model.fit_predict(X)
        n_noise = int((labels == -1).sum())
        n_clusters_found = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(
            "HDBSCAN: %s samples -> %s clusters + %s outliers "
            "(min_cluster_size=%s, metric=%s)",
            X.shape[0],
            n_clusters_found,
            n_noise,
            min_cluster_size,
            metric,
        )
        return labels.astype(int)

    # ── Cluster info computation ───────────────────────────────────────

    def _compute_cluster_info(self, X: np.ndarray) -> None:
        """Compute descriptive metrics per cluster."""
        labels = self.labels_
        unique_labels = sorted(set(labels))
        info: dict[str, Any] = {
            "n_clusters": len(unique_labels) - (1 if -1 in unique_labels else 0),
            "n_noise": int((labels == -1).sum()),
            "cluster_sizes": {},
            "centroids": {},
        }

        for label in unique_labels:
            mask = labels == label
            size = int(mask.sum())
            info["cluster_sizes"][int(label)] = size
            if label != -1:
                info["centroids"][int(label)] = X[mask].mean(axis=0).tolist()

        self.cluster_info_ = info

    def get_cluster_members(self, labels: np.ndarray, cluster_id: int) -> np.ndarray:
        """Return indices of samples in a specific cluster."""
        return np.where(labels == cluster_id)[0]

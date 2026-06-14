"""
feature_builder.py — Construction of the composite vector φ(aⱼ)

Purpose:
  Combine BERT embeddings (semantics) with structural encoding
  (types + constraints) into a single feature vector.

  Pipeline per attribute:
    1. Z-score normalization of each component
    2. Weighting with alpha, beta, gamma, delta coefficients
    3. Concatenation → φ(aⱼ)

  phi = [alpha * e_text_norm  +  beta * e_type_norm  +  gamma * e_rest_norm]

Usage:
  builder = FeatureBuilder(alpha=0.60, beta=0.15, gamma=0.15)
  phi = builder.build(e_text, e_type, e_rest)
  # → numpy array shape (N, 768 + 12 + 5)
"""

from __future__ import annotations

import numpy as np


class FeatureBuilder:
    """
    Builds the composite vector φ(aⱼ) for each attribute.

    Default weights:
      alpha (BERT)           = 0.40  — semantic weight
      beta (data type)       = 0.30  — data type weight
      gamma (constraints)    = 0.25  — constraint weight
      delta (statistical)    = 0.05  — statistical feature weight (data_length)
    """

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.30,
        gamma: float = 0.25,
        delta: float = 0.05,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self._fitted = False

    # ── Z-score normalization ─────────────────────────────────────────

    @staticmethod
    def _zscore(matrix: np.ndarray) -> np.ndarray:
        """
        Normalize each column with Z-score: (x - mean) / std.
        Columns with std = 0 are left as 0.
        """
        if matrix.shape[1] == 0:
            return matrix
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        std[std == 0] = 1.0  # avoid division by zero
        return (matrix - mean) / std

    # ── Composite vector construction ─────────────────────────────────

    def build(
        self,
        e_text: np.ndarray,
        e_type: np.ndarray,
        e_rest: np.ndarray,
        e_stat: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Build the composite vector φ(aⱼ).

        Args:
            e_text: BERT embeddings           shape (N, 768)
            e_type: One-hot types              shape (N, n_types)
            e_rest: Binary constraints         shape (N, 5)
            e_stat: Statistical (optional)     shape (N, s)

        Returns:
            numpy array shape (N, d) where d = 768 + n_types + 5 + (s or 0)
        """
        assert e_text.shape[0] == e_type.shape[0] == e_rest.shape[0], (
            "All matrices must have the same number of rows"
        )

        # Normalize each component
        e_text_norm = self._zscore(e_text)
        e_type_norm = self._zscore(e_type)
        e_rest_norm = self._zscore(e_rest)

        # Weight
        e_text_w = self.alpha * e_text_norm
        e_type_w = self.beta * e_type_norm
        e_rest_w = self.gamma * e_rest_norm

        # Concatenate
        components = [e_text_w, e_type_w, e_rest_w]

        if e_stat is not None:
            e_stat_norm = self._zscore(e_stat)
            e_stat_w = self.delta * e_stat_norm
            components.append(e_stat_w)

        phi = np.concatenate(components, axis=1)

        self._fitted = True
        return phi

    # ── Dimensionality query ──────────────────────────────────────────

    @property
    def n_components(self) -> int:
        """Number of active components (excluding statistics)."""
        return 3  # text + type + constraints

    @property
    def weights(self) -> dict[str, float]:
        """Return current weight configuration."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
        }

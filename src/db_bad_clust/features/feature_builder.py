"""
feature_builder.py — Construction of the composite vector φ(aⱼ)

Purpose:
  Combine BERT embeddings (semantics) with structural encoding
  (types + constraints) into a single feature vector.

  Pipeline per attribute:
    1. Z-score normalization of each component
    2. Scaling of each block to unit total variance ("block" mode)
    3. Weighting with alpha, beta, gamma, delta coefficients
    4. Concatenation → φ(aⱼ)

Why step 2 exists — a correction, not a refinement:
  A z-scored block has variance 1 in *every* dimension, so its total variance
  equals its dimension count: 384 for the embeddings against 12 for the types
  and 5 for the constraints. Multiplying by alpha afterwards cannot undo that.
  Measured on the 243-column corpus, under the original "zscore" mode:

      alpha nominal    real share of variance taken by the embeddings
           0.00                          0%
           0.15                         89%
           0.30                         98%

  So alpha never weighted anything: it was a switch. Any sweep run in that mode
  compares "no embeddings" against "embeddings and almost nothing else", and a
  drop across it says nothing about what the embeddings contribute at 15%.

  "block" mode divides each block by its total variance before weighting, so
  the shares follow alpha²:beta²:gamma²:delta². It is the default. "zscore" is
  kept only to reproduce the historical numbers.

Usage:
  builder = FeatureBuilder(alpha=0.60, beta=0.15, gamma=0.15)
  phi = builder.build(e_text, e_type, e_rest)
  # → numpy array shape (N, e_text_dim + 12 + 5)
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

    Args:
      normalize: "block" (default) scales every block to unit total variance
        before weighting, so the weights are the real variance shares.
        "zscore" reproduces the original behaviour, in which the weights are
        overwhelmed by the blocks' dimension counts — see the module docstring.
    """

    VALID_NORMALIZERS = ("block", "zscore")

    def __init__(
        self,
        alpha: float = 0.40,
        beta: float = 0.30,
        gamma: float = 0.25,
        delta: float = 0.05,
        normalize: str = "block",
    ) -> None:
        if normalize not in self.VALID_NORMALIZERS:
            raise ValueError(
                f"normalize must be one of {self.VALID_NORMALIZERS}, got {normalize!r}"
            )
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.normalize = normalize
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

    # ── Block scaling ─────────────────────────────────────────────────

    @staticmethod
    def _unit_variance_block(matrix: np.ndarray) -> np.ndarray:
        """Scale a z-scored block so its *total* variance is 1.

        Without this the block's dimension count is its weight. A block with
        no variance at all (every dimension constant) is returned untouched
        rather than divided by zero.
        """
        total = matrix.var(axis=0).sum()
        if total <= 0.0:
            return matrix
        # Keep the block's own dtype: promoting float32 embeddings to float64
        # here would silently double the memory and, worse, hide how much of a
        # score depends on the arithmetic width (see `precision_sensitivity`).
        return matrix / np.sqrt(total)

    def _normalize(self, matrix: np.ndarray) -> np.ndarray:
        """Z-score, then scale to unit total variance unless in legacy mode."""
        normalized = self._zscore(matrix)
        if self.normalize == "block":
            return self._unit_variance_block(normalized)
        return normalized

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
            e_text: Text embeddings           shape (N, d_text)
            e_type: One-hot types              shape (N, n_types)
            e_rest: Binary constraints         shape (N, 5)
            e_stat: Statistical (optional)     shape (N, s)

        Returns:
            numpy array shape (N, d) where d = d_text + n_types + 5 + (s or 0)
        """
        assert e_text.shape[0] == e_type.shape[0] == e_rest.shape[0], (
            "All matrices must have the same number of rows"
        )

        # Normalize each component
        e_text_norm = self._normalize(e_text)
        e_type_norm = self._normalize(e_type)
        e_rest_norm = self._normalize(e_rest)

        # Weight
        e_text_w = self.alpha * e_text_norm
        e_type_w = self.beta * e_type_norm
        e_rest_w = self.gamma * e_rest_norm

        # Concatenate
        components = [e_text_w, e_type_w, e_rest_w]

        if e_stat is not None:
            e_stat_norm = self._normalize(e_stat)
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

    @property
    def variance_shares(self) -> dict[str, float]:
        """The share of variance each block actually receives in "block" mode.

        Undefined for "zscore" mode, where the share also depends on each
        block's dimension count and so cannot be read off the weights.
        """
        if self.normalize != "block":
            raise ValueError('variance_shares is only meaningful for normalize="block"')
        squared = {k: v**2 for k, v in self.weights.items()}
        total = sum(squared.values()) or 1.0
        return {k: v / total for k, v in squared.items()}

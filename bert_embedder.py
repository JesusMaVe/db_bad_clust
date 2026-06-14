"""
bert_embedder.py — BERT embedding generation for database column names.

Generates dense ℝ⁷⁶⁸ vectors for each column using
bert-base-multilingual-cased from Hugging Face. Takes preprocessed
text (via TextPreprocessor) and extracts the [CLS] token from the
last hidden layer.

Usage:
    embedder = BERTEmbedder()
    vectors = embedder.encode(["employees: date of birth", ...])
    # → numpy array shape (N, 768)
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING

import numpy as np
from exceptions import EmbeddingError

if TYPE_CHECKING:
    from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class BERTEmbedder:
    """Generates BERT embeddings for database attribute texts.

    Caches the model in memory to avoid reloading on successive calls.

    Args:
        model_name: Hugging Face model identifier.
        device: "auto" (CPU/MPS/CUDA), "cpu", "cuda", or "mps".
        max_length: Maximum token count per text (short names → 32).
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        device: str = "auto",
        max_length: int = 32,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._model: AutoModel | None = None
        self._tokenizer: AutoTokenizer | None = None
        self._device: str = self._resolve_device(device)

    # ── Lazy model initialisation ────────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Determine the optimal compute device.

        Returns:
            Device string: "cuda", "mps", or "cpu".
        """
        if device != "auto":
            return device
        try:
            import torch as _

            if _.cuda.is_available():
                return "cuda"
            if hasattr(_.backends, "mps") and _.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self) -> None:
        """Load the model and tokenizer once (idempotent)."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore[import-untyped]
        except ImportError as e:
            raise EmbeddingError(
                "Missing optional dependencies: pip install transformers torch"
            ) from e

        logger.info("Loading BERT model: %s (device=%s)", self.model_name, self._device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

        try:
            import torch as _  # noqa: F401 — verify availability

            self._model.to(self._device)
        except (ImportError, RuntimeError, ValueError):
            self._device = "cpu"

        logger.info("BERT model loaded (dim=%d)", self._model.config.hidden_size)

    @property
    def embedding_dim(self) -> int:
        """Return the embedding dimension (768 for bert-base)."""
        self._load_model()
        return int(self._model.config.hidden_size)

    # ── Encoding ──────────────────────────────────────────────────────

    def encode(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for a list of preprocessed texts.

        Args:
            texts: List of preprocessed column-name strings.

        Returns:
            numpy array of shape (len(texts), embedding_dim).

        Raises:
            EmbeddingError: If torch / transformers are missing or the
                model fails to produce embeddings.
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 0)

        self._load_model()

        try:
            import torch
        except ImportError as e:
            raise EmbeddingError("torch is required for BERT inference") from e

        try:
            inputs = self._tokenizer(  # type: ignore[misc]
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )

            with torch.no_grad():
                inputs = {k: v.to(self._device) for k, v in inputs.items()}
                outputs = self._model(**inputs)  # type: ignore[misc]
                cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            return cls_vectors.astype(np.float32)
        except Exception as e:
            raise EmbeddingError(f"BERT encoding failed: {e}") from e

    def encode_single(self, text: str) -> np.ndarray:
        """Generate an embedding for a single text string.

        Returns:
            numpy array of shape (embedding_dim,).
        """
        return self.encode([text])[0]

    # ── Cache cleanup ─────────────────────────────────────────────────

    def unload(self) -> None:
        """Release the model from memory."""
        self._model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("BERT model unloaded")

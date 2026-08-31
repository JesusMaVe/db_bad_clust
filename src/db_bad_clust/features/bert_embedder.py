"""
bert_embedder.py — Sentence embedding generation for database column names.

Generates dense vectors (ℝ³⁸⁴) for each column using
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 from
Hugging Face. Takes preprocessed text (via TextPreprocessor) and
computes mean pooling over the last hidden layer (attention-mask
weighted), followed by L2 normalization.

Why not bert-base-multilingual-cased + [CLS]: the SBERT paper
(arXiv:1908.10084) shows raw BERT [CLS] vectors are unsuitable for
unsupervised similarity/clustering (29.19 vs 74.89 Spearman on STS).
This model is trained for clustering/semantic search with mean pooling.

Usage:
    embedder = BERTEmbedder()
    vectors = embedder.encode(["employees: date of birth", ...])
    # → numpy array shape (N, 384)
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING, Protocol

import numpy as np

from db_bad_clust.exceptions import EmbeddingError

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Tokenizer(Protocol):
    """Protocol for tokenizers (AutoTokenizer or mock)."""

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 32,
        return_tensors: str = "pt",
    ) -> dict: ...


class Model(Protocol):
    """Protocol for models (AutoModel or mock)."""

    config: dict
    def eval(self) -> None: ...
    def to(self, device: str) -> None: ...
    def __call__(self, **kwargs) -> object: ...


class BERTEmbedder:
    """Generates sentence embeddings for database attribute texts.

    Caches the model in memory to avoid reloading on successive calls.

    Args:
        model_name: Hugging Face model identifier.
        device: "auto" (CPU/MPS/CUDA), "cpu", "cuda", or "mps".
        max_length: Maximum token count per text (model max is 128).
        tokenizer: Optional pre-loaded tokenizer (for testing/DI).
        model: Optional pre-loaded model (for testing/DI).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device: str = "auto",
        max_length: int = 128,
        tokenizer: Tokenizer | None = None,
        model: Model | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self._model: Model | None = model
        self._tokenizer: Tokenizer | None = tokenizer
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
        """Return the embedding dimension (384 for MiniLM-L12)."""
        self._load_model()
        return int(self._model.config.hidden_size)

    # ── Encoding ──────────────────────────────────────────────────────

    def encode(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Generate embeddings for a list of preprocessed texts.

        Args:
            texts: List of preprocessed column-name strings.
            batch_size: Number of texts to process per batch (default 32).

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
            all_embeddings = []

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                inputs = self._tokenizer(  # type: ignore[misc]
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )

                with torch.no_grad():
                    inputs = {k: v.to(self._device) for k, v in inputs.items()}
                    outputs = self._model(**inputs)  # type: ignore[misc]
                    # Mean pooling: weight token embeddings by attention mask
                    token_embeddings = outputs.last_hidden_state
                    mask = inputs["attention_mask"]
                    mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                    summed = torch.sum(token_embeddings * mask_expanded, dim=1)
                    counts = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                    mean_vectors = summed / counts
                    # L2 normalize so cosine similarity == dot product
                    mean_vectors = torch.nn.functional.normalize(mean_vectors, p=2, dim=1)
                    all_embeddings.append(mean_vectors.cpu().numpy())

            return np.concatenate(all_embeddings, axis=0).astype(np.float32)
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

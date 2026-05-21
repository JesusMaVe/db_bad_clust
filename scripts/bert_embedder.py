"""
bert_embedder.py — Generación de embeddings BERT para atributos de BD

Propósito:
  Genera vectores densos ℝ⁷⁶⁸ para cada atributo (columna) usando
  bert-base-multilingual-cased de Hugging Face.

  Toma el texto preprocesado por TextPreprocessor y extrae el vector
  del token [CLS] de la última capa como representación semántica.

Uso:
  embedder = BERTEmbedder()
  vectors = embedder.encode(["empleados: fecha nacimiento", ...])
  # → numpy array shape (N, 768)
"""

import logging
import numpy as np
from typing import List, Optional

logger = logging.getLogger(__name__)


class BERTEmbedder:
    """
    Genera embeddings BERT para textos de atributos de base de datos.

    Cachea el modelo en memoria para evitar recargar en llamadas sucesivas.
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        device: str = "auto",
        max_length: int = 32,
    ):
        """
        Args:
            model_name: Nombre del modelo Hugging Face.
            device: "auto" (CPU/MPS/CUDA), "cpu", "cuda", o "mps".
            max_length: Máximo de tokens por texto (nombres cortos → 32).
        """
        self.model_name = model_name
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._device = self._resolve_device(device)

    # ── Inicialización lazy del modelo ────────────────────────────────

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Determina el dispositivo óptimo."""
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _load_model(self):
        """Carga el modelo y tokenizer (una sola vez)."""
        if self._model is not None:
            return

        try:
            from transformers import AutoTokenizer, AutoModel
        except ImportError:
            raise ImportError(
                "Se necesita 'transformers' y 'torch': pip install transformers torch"
            )

        logger.info(
            "Cargando modelo BERT: %s (device=%s)", self.model_name, self._device
        )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name)
        self._model.eval()

        try:
            import torch
            self._model.to(self._device)
        except Exception:
            self._device = "cpu"

        logger.info("Modelo BERT cargado (dim=%d)", self._model.config.hidden_size)

    @property
    def embedding_dim(self) -> int:
        """Dimensión del embedding (768 para bert-base)."""
        self._load_model()
        return self._model.config.hidden_size

    # ── Encoding ──────────────────────────────────────────────────────

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Genera embeddings para una lista de textos.

        Args:
            texts: Lista de strings preprocesados.

        Returns:
            numpy array shape (len(texts), embedding_dim).
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, 0)

        self._load_model()

        try:
            import torch
        except ImportError:
            raise ImportError("Se necesita 'torch' para ejecutar el modelo BERT")

        inputs = self._tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        with torch.no_grad():
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            outputs = self._model(**inputs)
            # Usar el vector [CLS] (primer token) de la última capa
            cls_vectors = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return cls_vectors.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """
        Genera embedding para un solo texto.

        Returns:
            numpy array shape (embedding_dim,).
        """
        return self.encode([text])[0]

    # ── Cache cleanup ─────────────────────────────────────────────────

    def unload(self):
        """Libera el modelo de memoria."""
        self._model = None
        self._tokenizer = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        logger.info("Modelo BERT descargado")

"""
feature_builder.py — Construcción del vector compuesto φ(aⱼ)

Propósito:
  Combina los embeddings BERT (semántica) con la codificación estructural
  (tipos + restricciones) en un vector de features único.

  Pipeline por atributo:
    1. Normalización Z-score de cada componente
    2. Ponderación con pesos α, β, γ, δ
    3. Concatenación → φ(aⱼ)

  φ = [α·e_texto_norm  ⊕  β·e_tipo_norm  ⊕  γ·e_rest_norm]

Uso:
  builder = FeatureBuilder(alpha=0.60, beta=0.15, gamma=0.15)
  phi = builder.build(e_text, e_type, e_rest)
  # → numpy array shape (N, 768 + 12 + 5)
"""

import numpy as np
from typing import Optional


class FeatureBuilder:
    """
    Construye el vector compuesto φ(aⱼ) para cada atributo.

    Los pesos por defecto siguen la recomendación del diseño:
      α (BERT)       = 0.60  — peso semántico
      β (tipo dato)  = 0.15  — peso del tipo de dato
      γ (restricc.)  = 0.15  — peso de constraints
      δ (estadíst.)  = 0.10  — reservado para features estadísticos futuros
    """

    def __init__(
        self,
        alpha: float = 0.60,
        beta: float = 0.15,
        gamma: float = 0.15,
        delta: float = 0.10,
    ):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self._fitted = False

    # ── Normalización Z-score ─────────────────────────────────────────

    @staticmethod
    def _zscore(matrix: np.ndarray) -> np.ndarray:
        """
        Normaliza cada columna con Z-score: (x - μ) / σ.
        Columnas con σ = 0 se dejan como 0.
        """
        if matrix.shape[1] == 0:
            return matrix
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        std[std == 0] = 1.0  # evitar división por cero
        return (matrix - mean) / std

    # ── Construcción del vector compuesto ─────────────────────────────

    def build(
        self,
        e_text: np.ndarray,
        e_type: np.ndarray,
        e_rest: np.ndarray,
        e_stat: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Construye el vector compuesto φ(aⱼ).

        Args:
            e_text: Embeddings BERT           shape (N, 768)
            e_type: One-hot tipos              shape (N, n_types)
            e_rest: Binario restricciones      shape (N, 5)
            e_stat: Estadísticos (opcional)    shape (N, s)

        Returns:
            numpy array shape (N, d) donde d = 768 + n_types + 5 + (s o 0)
        """
        assert e_text.shape[0] == e_type.shape[0] == e_rest.shape[0], (
            "Todas las matrices deben tener el mismo número de filas"
        )

        # Normalizar cada componente
        e_text_norm = self._zscore(e_text)
        e_type_norm = self._zscore(e_type)
        e_rest_norm = self._zscore(e_rest)

        # Ponderar
        e_text_w = self.alpha * e_text_norm
        e_type_w = self.beta * e_type_norm
        e_rest_w = self.gamma * e_rest_norm

        # Concatenar
        components = [e_text_w, e_type_w, e_rest_w]

        if e_stat is not None:
            e_stat_norm = self._zscore(e_stat)
            e_stat_w = self.delta * e_stat_norm
            components.append(e_stat_w)

        phi = np.concatenate(components, axis=1)

        self._fitted = True
        return phi

    # ── Consulta de dimensionalidad ───────────────────────────────────

    @property
    def n_components(self) -> int:
        """Número de componentes activos (sin contar estadísticos)."""
        return 3  # texto + tipo + restricciones

    @property
    def weights(self) -> dict:
        """Devuelve los pesos actuales."""
        return {
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
        }

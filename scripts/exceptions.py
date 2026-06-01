"""
exceptions.py — Jerarquía de excepciones para db_bad_clust

Todas las excepciones del proyecto heredan de BadDBError.
Los orquestadores pueden capturar BadDBError para manejo genérico
o subtipos específicos para respuestas más precisas.
"""


class BadDBError(Exception):
    """Base exception for all db_bad_clust errors."""

    def __init__(self, message: str = "", cause: Exception | None = None) -> None:
        super().__init__(message)
        self.cause = cause


# ── Database ──────────────────────────────────────────────────────────


class DatabaseError(BadDBError):
    """Base for database connection / query errors."""


class ConnectionError(DatabaseError):
    """Database connection failure (wrong credentials, timeout, etc.)."""


class QueryError(DatabaseError):
    """SQL query execution failure."""


# ── Schema ────────────────────────────────────────────────────────────


class SchemaError(BadDBError):
    """Schema extraction / metadata error."""


# ── Embeddings ────────────────────────────────────────────────────────


class EmbeddingError(BadDBError):
    """BERT / embedding generation error."""


# ── Clustering ────────────────────────────────────────────────────────


class ClusteringError(BadDBError):
    """Dimensionality reduction or clustering error."""


# ── Evaluation ────────────────────────────────────────────────────────


class EvaluationError(BadDBError):
    """Metrics evaluation error."""


# ── Configuration ─────────────────────────────────────────────────────


class ConfigError(BadDBError):
    """Configuration / YAML loading error."""


# ── DDL / DML generation ──────────────────────────────────────────────


class GenerationError(BadDBError):
    """DDL or DML generation error."""


# ── Visualization ─────────────────────────────────────────────────────


class VisualizationError(BadDBError):
    """Plot / chart generation error."""

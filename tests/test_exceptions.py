"""
test_exceptions.py — Tests for the custom exception hierarchy.

Verifies:
  - Every exception class can be instantiated
  - BadDBError is the root base class
  - isinstance checks follow the hierarchy
  - Cause chaining via the `cause` attribute works
  - String representation includes the message
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from exceptions import (
    BadDBError,
    DatabaseError,
    ConnectionError,
    QueryError,
    SchemaError,
    EmbeddingError,
    ClusteringError,
    EvaluationError,
    ConfigError,
    GenerationError,
    VisualizationError,
)

import pytest


# ── All exceptions can be instantiated ─────────────────────────────────


class TestInstantiation:
    """Each exception class can be created with and without a message."""

    @pytest.mark.parametrize(
        "exc_class",
        [
            BadDBError,
            DatabaseError,
            ConnectionError,
            QueryError,
            SchemaError,
            EmbeddingError,
            ClusteringError,
            EvaluationError,
            ConfigError,
            GenerationError,
            VisualizationError,
        ],
    )
    def test_default_construction(self, exc_class: type) -> None:
        """Exception can be raised with no arguments."""
        instance = exc_class()
        assert isinstance(instance, Exception)

    @pytest.mark.parametrize(
        "exc_class, message",
        [
            (BadDBError, "generic error"),
            (DatabaseError, "db failure"),
            (ConnectionError, "cannot connect"),
            (QueryError, "bad SQL"),
            (SchemaError, "missing table"),
            (EmbeddingError, "model failed"),
            (ClusteringError, "no convergence"),
            (EvaluationError, "bad metric"),
            (ConfigError, "YAML parse error"),
            (GenerationError, "DDL generation failed"),
            (VisualizationError, "plot error"),
        ],
    )
    def test_with_message(self, exc_class: type, message: str) -> None:
        """Exception can be raised with a string message."""
        instance = exc_class(message)
        assert str(instance) == message


# ── Hierarchy via isinstance ────────────────────────────────────────────


class TestHierarchy:
    """Verify the isinstance chain for each exception branch."""

    def test_bad_db_error_is_base(self) -> None:
        """Every custom exception should be an instance of BadDBError."""
        for exc in [
            BadDBError(),
            DatabaseError(),
            ConnectionError(),
            QueryError(),
            SchemaError(),
            EmbeddingError(),
            ClusteringError(),
            EvaluationError(),
            ConfigError(),
            GenerationError(),
            VisualizationError(),
        ]:
            assert isinstance(exc, BadDBError), f"{type(exc).__name__} not a BadDBError"

    def test_database_hierarchy(self) -> None:
        """DatabaseError ← ConnectionError, QueryError."""
        assert isinstance(ConnectionError(), DatabaseError)
        assert isinstance(QueryError(), DatabaseError)
        assert issubclass(ConnectionError, DatabaseError)
        assert issubclass(QueryError, DatabaseError)

    def test_concrete_is_not_parent(self) -> None:
        """Child exceptions are not instances of unrelated siblings."""
        conn_err = ConnectionError()
        assert not isinstance(conn_err, QueryError)
        assert not isinstance(conn_err, SchemaError)
        assert not isinstance(conn_err, EmbeddingError)

    def test_schema_is_separate_branch(self) -> None:
        """SchemaError is a BadDBError but not a DatabaseError."""
        assert isinstance(SchemaError(), BadDBError)
        assert not isinstance(SchemaError(), DatabaseError)

    def test_embedding_is_separate_branch(self) -> None:
        """EmbeddingError is a BadDBError but not a DatabaseError."""
        assert isinstance(EmbeddingError(), BadDBError)
        assert not isinstance(EmbeddingError(), DatabaseError)

    def test_clustering_is_separate_branch(self) -> None:
        """ClusteringError is a BadDBError but not a DatabaseError."""
        assert isinstance(ClusteringError(), BadDBError)
        assert not isinstance(ClusteringError(), DatabaseError)

    def test_evaluation_is_separate_branch(self) -> None:
        """EvaluationError is a BadDBError but not a ClusteringError."""
        assert isinstance(EvaluationError(), BadDBError)
        assert not isinstance(EvaluationError(), ClusteringError)


# ── Cause chaining ──────────────────────────────────────────────────────


class TestCauseChaining:
    """Verify the `cause` attribute and __cause__ chaining."""

    def test_cause_attribute(self) -> None:
        """BadDBError stores the cause in the `cause` attribute."""
        inner = ValueError("inner failure")
        outer = BadDBError("wrapping failure", cause=inner)
        assert outer.cause is inner
        assert str(outer) == "wrapping failure"

    def test_cause_is_none_by_default(self) -> None:
        """Cause is None when not provided."""
        exc = BadDBError("no cause")
        assert exc.cause is None

    def test_raise_from_chaining(self) -> None:
        """Using 'raise ... from ...' sets __cause__."""
        inner = RuntimeError("original problem")
        with pytest.raises(ClusteringError) as exc_info:
            raise ClusteringError("clustering failed") from inner
        assert exc_info.value.__cause__ is inner

    def test_nested_cause_chain(self) -> None:
        """Two-level cause chain: BadDBError → Exception → ValueError."""
        original = ValueError("root cause")
        middle = Exception("middle", original)
        top = BadDBError("top level", cause=middle)
        assert top.cause is middle
        assert isinstance(top.cause, Exception)

    def test_subclass_inherits_cause_attribute(self) -> None:
        """Subclasses like ConnectionError inherit the cause attribute."""
        inner = TimeoutError("timeout")
        exc = ConnectionError("connection refused", cause=inner)
        assert exc.cause is inner

    @pytest.mark.parametrize(
        "exc_class",
        [
            BadDBError,
            DatabaseError,
            ConnectionError,
            QueryError,
            SchemaError,
            EmbeddingError,
            ClusteringError,
            EvaluationError,
            ConfigError,
            GenerationError,
            VisualizationError,
        ],
    )
    def test_all_exceptions_accept_cause(self, exc_class: type) -> None:
        """Every exception class accepts the optional cause argument."""
        cause = KeyError("missing key")
        instance = exc_class("wrapped", cause=cause)
        assert instance.cause is cause

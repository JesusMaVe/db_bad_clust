"""
test_recommender.py — Tests for the Recommender module.

Verifies:
  - recommend() returns a list of dicts
  - Each recommendation has expected keys
  - Cluster IDs are correctly assigned
  - Recommendations detect issues (nullable, date-as-text, etc.)
  - print_recommendations() returns a formatted string
  - Works with minimal schema + labels
  - Edge cases: all noise, single cluster, empty schema
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest
import numpy as np
from recommender import Recommender, ANTI_PATTERN_SIGNATURES
from schema_extractor import ColumnMetadata, TableMetadata, DatabaseSchema


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def recommender() -> Recommender:
    """Default Recommender instance."""
    return Recommender()


@pytest.fixture
def simple_schema() -> DatabaseSchema:
    """A single table with 5 columns (one a VARCHAR date-as-text)."""
    return DatabaseSchema(
        tables=[
            TableMetadata(
                name="EMPLEADOS",
                columns=[
                    ColumnMetadata(
                        name="ID",
                        data_type="NUMBER",
                        data_length=22,
                        nullable=False,
                        is_primary_key=True,
                    ),
                    ColumnMetadata(
                        name="NOMBRE",
                        data_type="VARCHAR2",
                        data_length=100,
                        nullable=False,
                    ),
                    ColumnMetadata(
                        name="FECHA_ALTA",
                        data_type="VARCHAR2",
                        data_length=10,
                        nullable=True,
                    ),
                    ColumnMetadata(
                        name="SALARIO",
                        data_type="VARCHAR2",
                        data_length=20,
                        nullable=True,
                    ),
                    ColumnMetadata(
                        name="ACTIVO",
                        data_type="CHAR",
                        data_length=1,
                        nullable=False,
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def mixed_schema() -> DatabaseSchema:
    """Two tables with various data types."""
    return DatabaseSchema(
        tables=[
            TableMetadata(
                name="EMPLEADOS",
                columns=[
                    ColumnMetadata(
                        name="ID",
                        data_type="NUMBER",
                        data_length=22,
                        nullable=False,
                        is_primary_key=True,
                    ),
                    ColumnMetadata(
                        name="FECHA_ALTA",
                        data_type="VARCHAR2",
                        data_length=10,
                        nullable=True,
                    ),
                    ColumnMetadata(
                        name="SALARIO",
                        data_type="NUMBER",
                        data_length=22,
                        nullable=True,
                    ),
                ],
            ),
            TableMetadata(
                name="PRODUCTOS",
                columns=[
                    ColumnMetadata(
                        name="ID",
                        data_type="NUMBER",
                        data_length=22,
                        nullable=False,
                        is_primary_key=True,
                    ),
                    ColumnMetadata(
                        name="PRECIO",
                        data_type="NUMBER",
                        data_length=12,
                        nullable=False,
                    ),
                    ColumnMetadata(
                        name="ACTIVO",
                        data_type="CHAR",
                        data_length=1,
                        nullable=False,
                    ),
                ],
            ),
        ],
    )


@pytest.fixture
def simple_labels() -> np.ndarray:
    """Labels: 5 columns in 2 clusters."""
    return np.array([0, 0, 1, 1, 0], dtype=int)


@pytest.fixture
def mixed_labels() -> np.ndarray:
    """Labels: 6 columns in 2 clusters."""
    return np.array([0, 0, 0, 1, 1, 1], dtype=int)


@pytest.fixture
def all_noise_labels() -> np.ndarray:
    """All columns marked as noise."""
    return np.full(5, -1, dtype=int)


# ── recommend() basic ───────────────────────────────────────────────────


class TestRecommendBasic:
    """Basic recommend() functionality."""

    def test_recommend_returns_list(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """recommend() returns a list."""
        result = recommender.recommend(simple_schema, simple_labels)
        assert isinstance(result, list)

    def test_recommend_contains_dicts(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """Each element in the list is a dict."""
        result = recommender.recommend(simple_schema, simple_labels)
        for rec in result:
            assert isinstance(rec, dict)

    def test_recommend_expected_keys(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """Each recommendation dict has expected keys."""
        result = recommender.recommend(simple_schema, simple_labels)
        expected_keys = {
            "cluster_id",
            "total_columns",
            "tables_involved",
            "dominant_type",
            "pct_nullable",
            "severity",
            "issues",
            "recommendations",
            "columns",
        }
        for rec in result:
            assert set(rec.keys()) == expected_keys

    def test_cluster_count(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """Number of recommendations equals number of clusters (excluding noise)."""
        result = recommender.recommend(simple_schema, simple_labels)
        assert len(result) == 2  # cluster 0 and 1

    def test_cluster_ids(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """Cluster IDs in recommendations match input labels."""
        result = recommender.recommend(simple_schema, simple_labels)
        cluster_ids = sorted([rec["cluster_id"] for rec in result])
        assert cluster_ids == [0, 1]

    def test_total_columns_match(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """Sum of total_columns across clusters equals total columns."""
        result = recommender.recommend(simple_schema, simple_labels)
        total = sum(rec["total_columns"] for rec in result)
        assert total == 5


# ── Domain-specific issues ──────────────────────────────────────────────


class TestDomainIssues:
    """Verify issue detection in recommendations."""

    def test_date_as_text_detected(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """FECHA_ALTA as VARCHAR2 should be flagged as 'date as text'."""
        result = recommender.recommend(simple_schema, simple_labels)
        all_issues = []
        for rec in result:
            all_issues.extend(rec["issues"])

        fecha_issues = [i for i in all_issues if "FECHA_ALTA" in i]
        assert len(fecha_issues) >= 1, f"FECHA_ALTA should be flagged but issues were: {all_issues}"

    def test_number_as_text_detected(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """SALARIO as VARCHAR2 should be flagged as 'number as text'."""
        result = recommender.recommend(simple_schema, simple_labels)
        all_issues = []
        for rec in result:
            all_issues.extend(rec["issues"])

        salario_issues = [i for i in all_issues if "SALARIO" in i]
        assert len(salario_issues) >= 1

    def test_fix_recommendations_present(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """Each recommendation includes fix suggestions."""
        result = recommender.recommend(simple_schema, simple_labels)
        for rec in result:
            assert len(rec["recommendations"]) > 0

    def test_severity_alta_for_text_issues(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """Clusters with date/number-as-text get severity 'alta'."""
        result = recommender.recommend(simple_schema, simple_labels)
        severities = [rec["severity"] for rec in result]
        assert "alta" in severities


# ── column_table_map ────────────────────────────────────────────────────


class TestColumnTableMap:
    """recommend() with explicit column_table_map."""

    def test_custom_column_table_map(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """column_table_map is accepted as parameter (table names come from schema)."""
        custom_map = ["CUSTOM_TABLE"] * 5
        # The parameter is accepted; table names are read from schema metadata
        result = recommender.recommend(simple_schema, simple_labels, column_table_map=custom_map)
        # Table names still come from the schema, not the map
        for rec in result:
            assert all(t == "EMPLEADOS" for t in rec["tables_involved"])

    def test_column_table_map_length_mismatch(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """column_table_map shorter than labels still works (zips)."""
        short_map = ["T1", "T1", "T1"]  # only 3 entries
        result = recommender.recommend(simple_schema, simple_labels, column_table_map=short_map)
        # No crash expected; zip stops at shortest
        assert isinstance(result, list)


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for the recommender."""

    def test_all_noise(
        self, recommender: Recommender, simple_schema: DatabaseSchema, all_noise_labels: np.ndarray
    ) -> None:
        """All labels are -1 (noise) → no recommendations (noise is skipped)."""
        result = recommender.recommend(simple_schema, all_noise_labels)
        assert result == []

    def test_single_column(self, recommender: Recommender) -> None:
        """Single column in a single table → 1 recommendation."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="T1",
                    columns=[
                        ColumnMetadata(
                            name="ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                            is_primary_key=True,
                        ),
                    ],
                ),
            ],
        )
        labels = np.array([0], dtype=int)
        result = recommender.recommend(schema, labels)
        assert len(result) == 1
        assert result[0]["total_columns"] == 1

    def test_single_cluster(self, recommender: Recommender, simple_schema: DatabaseSchema) -> None:
        """All columns in one cluster → 1 recommendation."""
        labels = np.zeros(5, dtype=int)
        result = recommender.recommend(simple_schema, labels)
        assert len(result) == 1
        assert result[0]["total_columns"] == 5

    def test_empty_schema(self, recommender: Recommender) -> None:
        """Empty schema (no tables) → no recommendations."""
        schema = DatabaseSchema(tables=[])
        labels = np.array([], dtype=int)
        result = recommender.recommend(schema, labels)
        assert result == []


# ── print_recommendations ───────────────────────────────────────────────


class TestPrintRecommendations:
    """Formatted output from print_recommendations()."""

    def test_print_returns_string(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """print_recommendations returns a string."""
        recs = recommender.recommend(simple_schema, simple_labels)
        output = recommender.print_recommendations(recs)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_print_contains_header(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """Output contains the header."""
        recs = recommender.recommend(simple_schema, simple_labels)
        output = recommender.print_recommendations(recs)
        assert "RECOMMENDATIONS PER CLUSTER" in output

    def test_print_contains_cluster_ids(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """Output mentions cluster IDs."""
        recs = recommender.recommend(simple_schema, simple_labels)
        output = recommender.print_recommendations(recs)
        assert "Cluster #0" in output
        assert "Cluster #1" in output

    def test_print_contains_issues(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """Output includes detected issues."""
        recs = recommender.recommend(simple_schema, simple_labels)
        output = recommender.print_recommendations(recs)
        assert "->" in output  # fix suggestions start with '->'

    def test_print_empty_recommendations(self, recommender: Recommender) -> None:
        """Empty recommendations list returns a basic header."""
        output = recommender.print_recommendations([])
        assert isinstance(output, str)
        assert "RECOMMENDATIONS PER CLUSTER" in output


# ── Custom signatures ───────────────────────────────────────────────────


class TestCustomSignatures:
    """Recommender with custom anti-pattern signatures."""

    def test_custom_signatures(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """Custom signatures are used when provided."""
        custom = [
            {
                "name": "Custom test",
                "description": "A custom test signature",
                "severity": "baja",
                "fix": "Do something",
            },
        ]
        engine = Recommender(signatures=custom)
        assert engine.signatures == custom

    def test_default_signatures(self, recommender: Recommender) -> None:
        """Default signatures match the module-level constant."""
        assert recommender.signatures == ANTI_PATTERN_SIGNATURES

    def test_signatures_not_modified(
        self, recommender: Recommender, simple_schema: DatabaseSchema, simple_labels: np.ndarray
    ) -> None:
        """Calling recommend() does not mutate the signatures list."""
        original_len = len(recommender.signatures)
        recommender.recommend(simple_schema, simple_labels)
        assert len(recommender.signatures) == original_len


# ── Integration with mixed schema ───────────────────────────────────────


class TestMixedSchema:
    """Recommendations with multiple tables."""

    def test_multiple_tables_involved(
        self, recommender: Recommender, mixed_schema: DatabaseSchema, mixed_labels: np.ndarray
    ) -> None:
        """Recommendations mention all involved tables."""
        result = recommender.recommend(mixed_schema, mixed_labels)
        all_tables = set()
        for rec in result:
            all_tables.update(rec["tables_involved"])
        assert "EMPLEADOS" in all_tables
        assert "PRODUCTOS" in all_tables

    def test_columns_with_types(
        self, recommender: Recommender, mixed_schema: DatabaseSchema, mixed_labels: np.ndarray
    ) -> None:
        """Each column entry in recommendation includes name and type."""
        result = recommender.recommend(mixed_schema, mixed_labels)
        for rec in result:
            for col in rec["columns"]:
                assert "name" in col
                assert "type" in col
                assert "table" in col

    def test_dominant_type_detected(
        self, recommender: Recommender, mixed_schema: DatabaseSchema, mixed_labels: np.ndarray
    ) -> None:
        """Dominant type is a string."""
        result = recommender.recommend(mixed_schema, mixed_labels)
        for rec in result:
            assert isinstance(rec["dominant_type"], str)
            assert len(rec["dominant_type"]) > 0

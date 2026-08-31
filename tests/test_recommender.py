"""
test_recommender.py — Tests for the Recommender module.

Verifies:
  - recommend() returns a dict with column_issues and table_issues
  - Each recommendation has expected keys
  - Recommendations detect issues (date-as-text, number-as-text, etc.)
  - print_recommendations() returns a formatted string
  - Works with minimal schema
  - Edge cases: empty schema
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from db_bad_clust.data.schema_extractor import ColumnMetadata, DatabaseSchema, TableMetadata
from db_bad_clust.rules.recommender import Recommender

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


# ── recommend() basic ───────────────────────────────────────────────────


class TestRecommendBasic:
    """Basic recommend() functionality."""

    def test_recommend_returns_dict(self, recommender: Recommender, simple_schema: DatabaseSchema) -> None:
        """recommend() returns a dict."""
        result = recommender.recommend(simple_schema)
        assert isinstance(result, dict)

    def test_recommend_has_column_issues(self, recommender: Recommender, simple_schema: DatabaseSchema) -> None:
        """recommend() returns column_issues."""
        result = recommender.recommend(simple_schema)
        assert "column_issues" in result
        assert isinstance(result["column_issues"], list)

    def test_recommend_has_table_issues(self, recommender: Recommender, simple_schema: DatabaseSchema) -> None:
        """recommend() returns table_issues."""
        result = recommender.recommend(simple_schema)
        assert "table_issues" in result
        assert isinstance(result["table_issues"], list)

    def test_column_issues_have_expected_keys(self, recommender: Recommender, simple_schema: DatabaseSchema) -> None:
        """Each column issue has expected keys."""
        result = recommender.recommend(simple_schema)
        expected_keys = {"label", "total_columns", "columns", "severity", "action", "details"}
        for rec in result["column_issues"]:
            assert set(rec.keys()) == expected_keys

    def test_table_issues_have_expected_keys(self, recommender: Recommender) -> None:
        """Each table issue has expected keys."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="BIG_TABLE",
                    columns=[
                        ColumnMetadata(
                            name=f"COL_{i}",
                            data_type="VARCHAR2",
                            data_length=100,
                            nullable=True,
                        )
                        for i in range(25)
                    ],
                )
            ]
        )
        result = recommender.recommend(schema)
        expected_keys = {"label", "total_tables", "tables", "severity", "action", "details"}
        for rec in result["table_issues"]:
            assert set(rec.keys()) == expected_keys


# ── Domain-specific issues ──────────────────────────────────────────────


class TestDomainIssues:
    """Verify issue detection in recommendations."""

    def test_date_as_text_detected(self, recommender: Recommender, simple_schema: DatabaseSchema) -> None:
        """FECHA_ALTA as VARCHAR2 should be flagged."""
        result = recommender.recommend(simple_schema)
        all_columns = []
        for rec in result["column_issues"]:
            all_columns.extend(rec["columns"])

        assert "FECHA_ALTA" in all_columns

    def test_number_as_text_detected(self, recommender: Recommender, simple_schema: DatabaseSchema) -> None:
        """SALARIO as VARCHAR2 should be flagged."""
        result = recommender.recommend(simple_schema)
        all_columns = []
        for rec in result["column_issues"]:
            all_columns.extend(rec["columns"])

        assert "SALARIO" in all_columns

    def test_bad_boolean_detected(self, recommender: Recommender, simple_schema: DatabaseSchema) -> None:
        """ACTIVO as CHAR should be flagged."""
        result = recommender.recommend(simple_schema)
        all_columns = []
        for rec in result["column_issues"]:
            all_columns.extend(rec["columns"])

        assert "ACTIVO" in all_columns

    def test_severity_present(self, recommender: Recommender, simple_schema: DatabaseSchema) -> None:
        """Each recommendation has a severity."""
        result = recommender.recommend(simple_schema)
        for rec in result["column_issues"]:
            assert rec["severity"] in ("alta", "media", "baja")

    def test_action_present(self, recommender: Recommender, simple_schema: DatabaseSchema) -> None:
        """Each recommendation has an action."""
        result = recommender.recommend(simple_schema)
        for rec in result["column_issues"]:
            assert len(rec["action"]) > 0


# ── Edge cases ──────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for the recommender."""

    def test_empty_schema(self, recommender: Recommender) -> None:
        """Empty schema (no tables) → no recommendations."""
        schema = DatabaseSchema(tables=[])
        result = recommender.recommend(schema)
        assert result == {"column_issues": [], "table_issues": []}

    def test_single_clean_column(self, recommender: Recommender) -> None:
        """Single well-typed column produces no column issues."""
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
        result = recommender.recommend(schema)
        assert len(result["column_issues"]) == 0

    def test_giant_table_detected(self, recommender: Recommender) -> None:
        """Table with >20 columns is detected as table-level issue."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="BIG_TABLE",
                    columns=[
                        ColumnMetadata(
                            name=f"COL_{i}",
                            data_type="VARCHAR2",
                            data_length=100,
                            nullable=True,
                        )
                        for i in range(25)
                    ],
                )
            ]
        )
        result = recommender.recommend(schema)
        table_labels = [rec["label"] for rec in result["table_issues"]]
        assert "giant_table" in table_labels


# ── print_recommendations ───────────────────────────────────────────────


class TestPrintRecommendations:
    """Formatted output from print_recommendations()."""

    def test_print_returns_string(self, recommender: Recommender, simple_schema: DatabaseSchema) -> None:
        """print_recommendations returns a string."""
        recs = recommender.recommend(simple_schema)
        output = recommender.print_recommendations(recs)
        assert isinstance(output, str)
        assert len(output) > 0

    def test_print_contains_column_header(self, recommender: Recommender, simple_schema: DatabaseSchema) -> None:
        """Output contains the column-level header."""
        recs = recommender.recommend(simple_schema)
        output = recommender.print_recommendations(recs)
        assert "COLUMN-LEVEL ISSUES" in output

    def test_print_contains_severity_icon(self, recommender: Recommender, simple_schema: DatabaseSchema) -> None:
        """Output contains severity icons."""
        recs = recommender.recommend(simple_schema)
        output = recommender.print_recommendations(recs)
        assert any(icon in output for icon in ["🔴", "🟡", "🟢"])

    def test_print_empty_recommendations(self, recommender: Recommender) -> None:
        """Empty recommendations returns 'No anti-patterns detected.'"""
        output = recommender.print_recommendations({"column_issues": [], "table_issues": []})
        assert "No anti-patterns detected" in output


class TestPhase2Recommendations:
    """Phase 2 Oracle-specific detectors surface in recommend()."""

    def test_fk_without_index_in_recommendations(self, recommender: Recommender) -> None:
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="PEDIDOS",
                    columns=[
                        ColumnMetadata(name="ID", data_type="NUMBER", data_length=22, nullable=False),
                        ColumnMetadata(name="CLIENTE_ID", data_type="NUMBER", data_length=22, nullable=True),
                    ],
                )
            ],
            fk_columns={"PEDIDOS": {"FK_CLIENTE": ["CLIENTE_ID"]}},
            index_columns={"PEDIDOS": {"IDX_ID": ["ID"]}},
        )
        result = recommender.recommend(schema)
        table_labels = [r["label"] for r in result["table_issues"]]
        assert "fk_without_index" in table_labels

    def test_obsolete_type_in_recommendations(self, recommender: Recommender) -> None:
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="LEGACY",
                    columns=[
                        ColumnMetadata(name="ID", data_type="NUMBER", data_length=22, nullable=False),
                        ColumnMetadata(name="TEXTO", data_type="LONG", data_length=0, nullable=True),
                    ],
                )
            ]
        )
        result = recommender.recommend(schema)
        column_labels = [r["label"] for r in result["column_issues"]]
        assert "obsolete_type" in column_labels

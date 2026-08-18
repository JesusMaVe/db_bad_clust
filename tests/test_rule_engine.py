"""
test_rule_engine.py — Tests for ColumnRuleEngine and TableRuleEngine.

Verifies:
  - RuleMatch creation and fields
  - ClassificationResult creation and fields
  - Column-level rules: date, number, boolean, reserved, self-ref, impossible, polymorphic
  - Table-level rules: giant_table, eav, inconsistent_naming
  - Integration with recommender
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from rule_engine import (
    ColumnRuleEngine,
    TableRuleEngine,
    RuleMatch,
    ClassificationResult,
    Severity,
    SQL_RESERVED_WORDS,
    GIANT_TABLE_THRESHOLD,
)
from schema_extractor import ColumnMetadata, TableMetadata, DatabaseSchema


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def col_engine() -> ColumnRuleEngine:
    """Default ColumnRuleEngine instance."""
    return ColumnRuleEngine()


@pytest.fixture
def table_engine() -> TableRuleEngine:
    """Default TableRuleEngine instance."""
    return TableRuleEngine()


@pytest.fixture
def empty_schema() -> DatabaseSchema:
    """Empty database schema."""
    return DatabaseSchema(tables=[])


# ─── RuleMatch ─────────────────────────────────────────────────────────


class TestRuleMatch:
    """Verify RuleMatch dataclass creation."""

    def test_creation(self) -> None:
        """RuleMatch can be created with required fields."""
        match = RuleMatch(
            rule_name="test_rule",
            anti_pattern="test_pattern",
            confidence=0.9,
            severity=Severity.HIGH,
            explanation="Test explanation",
            fix="Test fix",
        )
        assert match.rule_name == "test_rule"
        assert match.anti_pattern == "test_pattern"
        assert match.confidence == 0.9
        assert match.severity == Severity.HIGH

    def test_severity_enum(self) -> None:
        """Severity enum has all values."""
        assert Severity.LOW.value == "low"
        assert Severity.MEDIUM.value == "medium"
        assert Severity.HIGH.value == "high"


# ─── ClassificationResult ──────────────────────────────────────────────


class TestClassificationResult:
    """Verify ClassificationResult dataclass creation."""

    def test_creation(self) -> None:
        """ClassificationResult can be created with all fields."""
        result = ClassificationResult(
            column_name="FECHA",
            table_name="EMPLEADOS",
            predicted_label="date_as_text",
            confidence=0.9,
            method="rule_engine",
            explanation="Date stored as text",
            fix="Change to DATE type",
            needs_review=True,
        )
        assert result.column_name == "FECHA"
        assert result.table_name == "EMPLEADOS"
        assert result.predicted_label == "date_as_text"
        assert result.needs_review is True


# ─── Rule 1: wrong_date_as_text ────────────────────────────────────────


class TestWrongDateAsText:
    """Verify date-as-text detection."""

    def test_detects_date_in_varchar(self, col_engine: ColumnRuleEngine) -> None:
        """VARCHAR column with date name is detected."""
        cols = [
            ColumnMetadata(
                name="FECHA_ALTA",
                data_type="VARCHAR2",
                data_length=10,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "date_as_text" in labels

    def test_no_false_positive_on_date_type(self, col_engine: ColumnRuleEngine) -> None:
        """DATE column with date name is NOT flagged."""
        cols = [
            ColumnMetadata(
                name="FECHA_ALTA",
                data_type="DATE",
                data_length=7,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "date_as_text" not in labels

    def test_no_flag_on_generic_name(self, col_engine: ColumnRuleEngine) -> None:
        """VARCHAR column without date-like name is NOT flagged."""
        cols = [
            ColumnMetadata(
                name="NOMBRE",
                data_type="VARCHAR2",
                data_length=50,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "date_as_text" not in labels

    def test_detects_created_timestamp(self, col_engine: ColumnRuleEngine) -> None:
        """Column named 'CREATED' with VARCHAR is detected."""
        cols = [
            ColumnMetadata(
                name="CREATED",
                data_type="VARCHAR2",
                data_length=20,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "date_as_text" in labels


# ─── Rule 2: wrong_number_as_text ──────────────────────────────────────


class TestWrongNumberAsText:
    """Verify number-as-text detection."""

    def test_detects_number_in_varchar(self, col_engine: ColumnRuleEngine) -> None:
        """VARCHAR column with numeric name is detected."""
        cols = [
            ColumnMetadata(
                name="SALARY",
                data_type="VARCHAR2",
                data_length=10,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "number_as_text" in labels

    def test_detects_precio(self, col_engine: ColumnRuleEngine) -> None:
        """Spanish name PRECIO with VARCHAR is detected."""
        cols = [
            ColumnMetadata(
                name="PRECIO",
                data_type="VARCHAR2",
                data_length=10,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "number_as_text" in labels

    def test_no_false_positive_on_number_type(self, col_engine: ColumnRuleEngine) -> None:
        """NUMBER column with numeric name is NOT flagged."""
        cols = [
            ColumnMetadata(
                name="SALARY",
                data_type="NUMBER",
                data_length=22,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "number_as_text" not in labels

    def test_no_flag_on_generic_name(self, col_engine: ColumnRuleEngine) -> None:
        """VARCHAR column without numeric name is NOT flagged."""
        cols = [
            ColumnMetadata(
                name="DESCRIPTION",
                data_type="VARCHAR2",
                data_length=200,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "number_as_text" not in labels


# ─── Rule 3: bad_boolean ──────────────────────────────────────────────


class TestBadBoolean:
    """Verify bad-boolean detection."""

    def test_detects_flag_in_varchar(self, col_engine: ColumnRuleEngine) -> None:
        """VARCHAR column with boolean name is detected."""
        cols = [
            ColumnMetadata(
                name="ACTIVE_FLAG",
                data_type="VARCHAR2",
                data_length=1,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "bad_boolean" in labels

    def test_detects_is_prefix(self, col_engine: ColumnRuleEngine) -> None:
        """Column 'IS_ACTIVE' with VARCHAR is detected."""
        cols = [
            ColumnMetadata(
                name="IS_ACTIVE",
                data_type="VARCHAR2",
                data_length=1,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "bad_boolean" in labels

    def test_detects_activo(self, col_engine: ColumnRuleEngine) -> None:
        """Spanish 'ACTIVO' with CHAR is detected."""
        cols = [
            ColumnMetadata(
                name="ACTIVO",
                data_type="CHAR",
                data_length=1,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "bad_boolean" in labels

    def test_no_flag_on_number_1(self, col_engine: ColumnRuleEngine) -> None:
        """NUMBER(1) is acceptable for boolean."""
        cols = [
            ColumnMetadata(
                name="IS_ACTIVE",
                data_type="NUMBER",
                data_length=1,
                data_precision=1,
                data_scale=0,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "bad_boolean" not in labels

    def test_no_flag_on_generic_name(self, col_engine: ColumnRuleEngine) -> None:
        """VARCHAR column without boolean name is NOT flagged."""
        cols = [
            ColumnMetadata(
                name="STATUS",
                data_type="VARCHAR2",
                data_length=10,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "bad_boolean" not in labels


# ─── Rule 4: reserved_wordss ───────────────────────────────────────────


class TestReservedWords:
    """Verify reserved-word detection."""

    def test_detects_select(self, col_engine: ColumnRuleEngine) -> None:
        """Column named 'SELECT' is detected."""
        cols = [
            ColumnMetadata(
                name="SELECT",
                data_type="VARCHAR2",
                data_length=10,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "reserved_words" in labels

    def test_detects_date_reserved(self, col_engine: ColumnRuleEngine) -> None:
        """Column named 'DATE' is detected."""
        cols = [
            ColumnMetadata(
                name="DATE",
                data_type="VARCHAR2",
                data_length=10,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "reserved_words" in labels

    def test_no_flag_on_normal_name(self, col_engine: ColumnRuleEngine) -> None:
        """Normal column name is NOT flagged."""
        cols = [
            ColumnMetadata(
                name="EMPLOYEE_NAME",
                data_type="VARCHAR2",
                data_length=50,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "reserved_words" not in labels

    def test_reserved_words_not_reviewed(self, col_engine: ColumnRuleEngine) -> None:
        """Reserved word matches are definite (needs_review=False)."""
        cols = [
            ColumnMetadata(
                name="TABLE",
                data_type="NUMBER",
                data_length=22,
                nullable=True,
            )
        ]
        results = col_engine.classify(cols)
        reserved = [r for r in results if r.predicted_label == "reserved_words"]
        assert len(reserved) == 1
        assert reserved[0].needs_review is False


# ─── Rule 5: self_referencing ─────────────────────────────────────────


class TestSelfReferencing:
    """Verify self-referencing FK detection."""

    def test_detects_self_ref(self, col_engine: ColumnRuleEngine) -> None:
        """FK referencing own table is detected."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="EMPLOYEES",
                    columns=[
                        ColumnMetadata(
                            name="ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnMetadata(
                            name="MANAGER_ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=True,
                            is_foreign_key=True,
                            fk_references_table="EMPLOYEES",
                            fk_references_column="ID",
                        ),
                    ],
                )
            ]
        )
        results = col_engine.classify(schema.tables[0].columns, schema)
        labels = [r.predicted_label for r in results]
        assert "self_referencing" in labels

    def test_no_false_positive_on_different_table(self, col_engine: ColumnRuleEngine) -> None:
        """FK referencing different table is NOT flagged."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="EMPLOYEES",
                    columns=[
                        ColumnMetadata(
                            name="ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnMetadata(
                            name="DEPT_ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=True,
                            is_foreign_key=True,
                            fk_references_table="DEPARTMENTS",
                            fk_references_column="ID",
                        ),
                    ],
                )
            ]
        )
        results = col_engine.classify(schema.tables[0].columns, schema)
        labels = [r.predicted_label for r in results]
        assert "self_referencing" not in labels

    def test_no_flag_without_schema(self, col_engine: ColumnRuleEngine) -> None:
        """Without schema, self-referencing cannot be detected."""
        cols = [
            ColumnMetadata(
                name="PARENT_ID",
                data_type="NUMBER",
                data_length=22,
                nullable=True,
                is_foreign_key=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "self_referencing" not in labels


# ─── Rule 6: impossible_data ──────────────────────────────────────────


class TestImpossibleData:
    """Verify impossible-data (nullable PK) detection."""

    def test_detects_nullable_pk(self, col_engine: ColumnRuleEngine) -> None:
        """PK column that allows NULL is detected."""
        cols = [
            ColumnMetadata(
                name="ORDER_ID",
                data_type="NUMBER",
                data_length=22,
                nullable=True,
                is_primary_key=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "impossible_data" in labels

    def test_no_flag_on_non_null_pk(self, col_engine: ColumnRuleEngine) -> None:
        """PK column with NOT NULL is NOT flagged."""
        cols = [
            ColumnMetadata(
                name="ORDER_ID",
                data_type="NUMBER",
                data_length=22,
                nullable=False,
                is_primary_key=True,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "impossible_data" not in labels

    def test_no_flag_on_non_pk_nullable(self, col_engine: ColumnRuleEngine) -> None:
        """Non-PK column that allows NULL is NOT flagged."""
        cols = [
            ColumnMetadata(
                name="ORDER_ID",
                data_type="NUMBER",
                data_length=22,
                nullable=False,
                is_primary_key=True,
            ),
            ColumnMetadata(
                name="NOTE",
                data_type="VARCHAR2",
                data_length=500,
                nullable=True,
            ),
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "impossible_data" not in labels


# ─── Rule 7: polymorphic ─────────────────────────────────────────────


class TestPolymorphic:
    """Verify polymorphic-association detection."""

    def test_detects_tipo_column(self, col_engine: ColumnRuleEngine) -> None:
        """Column named 'TIPO' is detected."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="DOCUMENTS",
                    columns=[
                        ColumnMetadata(
                            name="ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="TIPO",
                            data_type="VARCHAR2",
                            data_length=20,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="CONTENT",
                            data_type="CLOB",
                            data_length=4000,
                            nullable=True,
                        ),
                    ],
                )
            ]
        )
        results = col_engine.classify([], schema)
        labels = [r.predicted_label for r in results]
        assert "polymorphic" in labels

    def test_detects_type_column(self, col_engine: ColumnRuleEngine) -> None:
        """Column named 'TYPE' is detected."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="PAYMENTS",
                    columns=[
                        ColumnMetadata(
                            name="ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="TYPE",
                            data_type="VARCHAR2",
                            data_length=20,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="AMOUNT",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                        ),
                    ],
                )
            ]
        )
        results = col_engine.classify([], schema)
        labels = [r.predicted_label for r in results]
        assert "polymorphic" in labels

    def test_detects_discriminator(self, col_engine: ColumnRuleEngine) -> None:
        """Column named 'DISCRIMINATOR' is detected."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="INHERITANCE_TABLE",
                    columns=[
                        ColumnMetadata(
                            name="ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="DISCRIMINATOR",
                            data_type="NUMBER",
                            data_length=2,
                            nullable=False,
                        ),
                    ],
                )
            ]
        )
        results = col_engine.classify([], schema)
        labels = [r.predicted_label for r in results]
        assert "polymorphic" in labels

    def test_no_flag_on_normal_column(self, col_engine: ColumnRuleEngine) -> None:
        """Table without type indicator column is NOT flagged."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="ORDERS",
                    columns=[
                        ColumnMetadata(
                            name="ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="STATUS",
                            data_type="VARCHAR2",
                            data_length=20,
                            nullable=False,
                        ),
                    ],
                )
            ]
        )
        results = col_engine.classify([], schema)
        labels = [r.predicted_label for r in results]
        assert "polymorphic" not in labels

    def test_no_flag_without_schema(self, col_engine: ColumnRuleEngine) -> None:
        """Without schema, polymorphic cannot be detected."""
        cols = [
            ColumnMetadata(
                name="TYPE",
                data_type="VARCHAR2",
                data_length=20,
                nullable=False,
            )
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert "polymorphic" not in labels


# ─── Table-Level Rules ────────────────────────────────────────────────


class TestGiantTable:
    """Verify giant-table detection."""

    def test_detects_giant_table(self, table_engine: TableRuleEngine) -> None:
        """Table with >20 columns is detected."""
        cols = [
            ColumnMetadata(
                name=f"COL_{i}",
                data_type="VARCHAR2",
                data_length=100,
                nullable=True,
            )
            for i in range(25)
        ]
        schema = DatabaseSchema(
            tables=[TableMetadata(name="BIG_TABLE", columns=cols)]
        )
        results = table_engine.classify(schema)
        labels = [r.predicted_label for r in results]
        assert "giant_table" in labels

    def test_no_flag_on_normal_table(self, table_engine: TableRuleEngine) -> None:
        """Table with <=20 columns is NOT flagged."""
        cols = [
            ColumnMetadata(
                name=f"COL_{i}",
                data_type="VARCHAR2",
                data_length=100,
                nullable=True,
            )
            for i in range(10)
        ]
        schema = DatabaseSchema(
            tables=[TableMetadata(name="NORMAL_TABLE", columns=cols)]
        )
        results = table_engine.classify(schema)
        labels = [r.predicted_label for r in results]
        assert "giant_table" not in labels

    def test_high_severity_for_very_giant(self, table_engine: TableRuleEngine) -> None:
        """Table with >30 columns gets HIGH severity."""
        cols = [
            ColumnMetadata(
                name=f"COL_{i}",
                data_type="VARCHAR2",
                data_length=100,
                nullable=True,
            )
            for i in range(35)
        ]
        schema = DatabaseSchema(
            tables=[TableMetadata(name="HUGE_TABLE", columns=cols)]
        )
        results = table_engine.classify(schema)
        giant = [r for r in results if r.predicted_label == "giant_table"]
        assert len(giant) == 1
        assert giant[0].severity == Severity.HIGH


class TestInconsistentNaming:
    """Verify inconsistent-naming detection."""

    def test_detects_language_mixing(self, table_engine: TableRuleEngine) -> None:
        """Table mixing Spanish and English names is detected."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="MIXED_LANG_TABLE",
                    columns=[
                        ColumnMetadata(
                            name="NOMBRE_CLIENTE",
                            data_type="VARCHAR2",
                            data_length=100,
                            nullable=True,
                        ),
                        ColumnMetadata(
                            name="FECHA_ORDEN",
                            data_type="DATE",
                            nullable=True,
                        ),
                        ColumnMetadata(
                            name="PHONE_NUMBER",
                            data_type="VARCHAR2",
                            data_length=20,
                            nullable=True,
                        ),
                    ],
                )
            ]
        )
        results = table_engine.classify(schema)
        labels = [r.predicted_label for r in results]
        assert "inconsistent_naming" in labels

    def test_detects_generic_names(self, table_engine: TableRuleEngine) -> None:
        """Table with generic names (C1, C2, COL_EXTRA) is detected."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="GENERIC_TABLE",
                    columns=[
                        ColumnMetadata(
                            name="C1",
                            data_type="VARCHAR2",
                            data_length=100,
                            nullable=True,
                        ),
                        ColumnMetadata(
                            name="C2",
                            data_type="VARCHAR2",
                            data_length=100,
                            nullable=True,
                        ),
                        ColumnMetadata(
                            name="COL_EXTRA",
                            data_type="VARCHAR2",
                            data_length=100,
                            nullable=True,
                        ),
                        ColumnMetadata(
                            name="NOMBRE_CLIENTE",
                            data_type="VARCHAR2",
                            data_length=100,
                            nullable=True,
                        ),
                    ],
                )
            ]
        )
        results = table_engine.classify(schema)
        labels = [r.predicted_label for r in results]
        assert "inconsistent_naming" in labels

    def test_no_flag_on_consistent(self, table_engine: TableRuleEngine) -> None:
        """Table with consistent naming is NOT flagged."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="CONSISTENT_TABLE",
                    columns=[
                        ColumnMetadata(
                            name="FIRST_NAME",
                            data_type="VARCHAR2",
                            data_length=50,
                            nullable=True,
                        ),
                        ColumnMetadata(
                            name="LAST_NAME",
                            data_type="VARCHAR2",
                            data_length=50,
                            nullable=True,
                        ),
                        ColumnMetadata(
                            name="PHONE_NUMBER",
                            data_type="VARCHAR2",
                            data_length=20,
                            nullable=True,
                        ),
                    ],
                )
            ]
        )
        results = table_engine.classify(schema)
        labels = [r.predicted_label for r in results]
        assert "inconsistent_naming" not in labels

    def test_no_flag_on_few_columns(self, table_engine: TableRuleEngine) -> None:
        """Table with <3 columns is NOT checked for naming."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="SMALL_TABLE",
                    columns=[
                        ColumnMetadata(
                            name="id",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="NAME",
                            data_type="VARCHAR2",
                            data_length=50,
                            nullable=True,
                        ),
                    ],
                )
            ]
        )
        results = table_engine.classify(schema)
        labels = [r.predicted_label for r in results]
        assert "inconsistent_naming" not in labels


class TestEavPattern:
    """Verify EAV pattern detection."""

    def test_detects_eav_table(self, table_engine: TableRuleEngine) -> None:
        """Table with entity_id, attribute, value columns is detected."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="OBJECT_ATTRIBUTES",
                    columns=[
                        ColumnMetadata(
                            name="OBJECT_ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="ATTRIBUTE",
                            data_type="VARCHAR2",
                            data_length=100,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="VALUE",
                            data_type="VARCHAR2",
                            data_length=4000,
                            nullable=True,
                        ),
                    ],
                )
            ]
        )
        results = table_engine.classify(schema)
        labels = [r.predicted_label for r in results]
        assert "eav" in labels

    def test_detects_eav_with_valor(self, table_engine: TableRuleEngine) -> None:
        """Table with Spanish 'VALOR' column is detected."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="PROPiedades",
                    columns=[
                        ColumnMetadata(
                            name="ENTITY_ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="ATTR_NAME",
                            data_type="VARCHAR2",
                            data_length=100,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="VALOR",
                            data_type="VARCHAR2",
                            data_length=4000,
                            nullable=True,
                        ),
                    ],
                )
            ]
        )
        results = table_engine.classify(schema)
        labels = [r.predicted_label for r in results]
        assert "eav" in labels

    def test_no_flag_on_normal_table(self, table_engine: TableRuleEngine) -> None:
        """Normal table without EAV pattern is NOT flagged."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="EMPLOYEES",
                    columns=[
                        ColumnMetadata(
                            name="ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="NAME",
                            data_type="VARCHAR2",
                            data_length=100,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="SALARY",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=True,
                        ),
                    ],
                )
            ]
        )
        results = table_engine.classify(schema)
        labels = [r.predicted_label for r in results]
        assert "eav" not in labels


# ─── Integration ────────────────────────────────────────────────────────


class TestIntegration:
    """Verify end-to-end classify behavior."""

    def test_multiple_rules_fired(self, col_engine: ColumnRuleEngine) -> None:
        """Multiple rules can fire on the same column list."""
        cols = [
            ColumnMetadata(
                name="FECHA_ALTA",
                data_type="VARCHAR2",
                data_length=10,
                nullable=True,
            ),
            ColumnMetadata(
                name="SELECT",
                data_type="NUMBER",
                data_length=22,
                nullable=True,
            ),
        ]
        results = col_engine.classify(cols)
        labels = {r.predicted_label for r in results}
        assert "date_as_text" in labels
        assert "reserved_words" in labels

    def test_results_sorted_by_confidence(self, col_engine: ColumnRuleEngine) -> None:
        """Results are sorted by confidence descending."""
        cols = [
            ColumnMetadata(
                name="FECHA_ALTA",
                data_type="VARCHAR2",
                data_length=10,
                nullable=True,
            ),
            ColumnMetadata(
                name="SELECT",
                data_type="NUMBER",
                data_length=22,
                nullable=True,
            ),
        ]
        results = col_engine.classify(cols)
        confidences = [r.confidence for r in results]
        assert confidences == sorted(confidences, reverse=True)

    def test_empty_columns(self, col_engine: ColumnRuleEngine) -> None:
        """Empty column list returns empty results."""
        results = col_engine.classify([])
        assert results == []


# ─── Full Pipeline Integration ─────────────────────────────────────────


class TestFullPipeline:
    """End-to-end: schema → rule engine → recommendations."""

    def test_schema_to_recommendations(self) -> None:
        """Full pipeline: DatabaseSchema → ColumnRuleEngine → Recommender."""
        from recommender import Recommender

        schema = DatabaseSchema(
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
                            nullable=True,
                        ),
                    ],
                ),
            ],
        )
        col_engine = ColumnRuleEngine()
        all_columns = [col for t in schema.tables for col in t.columns]
        results = col_engine.classify(all_columns, schema)
        labels = {r.predicted_label for r in results}
        assert "date_as_text" in labels
        assert "number_as_text" in labels
        assert "bad_boolean" in labels

        recommender = Recommender()
        recommendations = recommender.recommend(schema)
        assert len(recommendations["column_issues"]) >= 1

    def test_schema_with_multiple_tables(self) -> None:
        """Pipeline handles multi-table schema correctly."""
        from recommender import Recommender

        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="ORDERS",
                    columns=[
                        ColumnMetadata(
                            name="ORDER_ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=True,
                            is_primary_key=True,
                        ),
                        ColumnMetadata(
                            name="CREATED",
                            data_type="VARCHAR2",
                            data_length=20,
                            nullable=True,
                        ),
                        ColumnMetadata(
                            name="TABLE",
                            data_type="VARCHAR2",
                            data_length=50,
                            nullable=True,
                        ),
                    ],
                ),
                TableMetadata(
                    name="ITEMS",
                    columns=[
                        ColumnMetadata(
                            name="ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnMetadata(
                            name="TOTAL_QTY",
                            data_type="VARCHAR2",
                            data_length=10,
                            nullable=True,
                        ),
                    ],
                ),
            ],
        )
        col_engine = ColumnRuleEngine()
        all_columns = [col for t in schema.tables for col in t.columns]
        results = col_engine.classify(all_columns, schema)
        labels = {r.predicted_label for r in results}
        assert "date_as_text" in labels
        assert "reserved_words" in labels
        assert "impossible_data" in labels
        assert "number_as_text" in labels

        recommender = Recommender()
        recommendations = recommender.recommend(schema)
        assert len(recommendations["column_issues"]) >= 1

    def test_pipeline_with_self_referencing_and_eav(self) -> None:
        """Pipeline detects structural anti-patterns in schema."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="CATEGORIES",
                    columns=[
                        ColumnMetadata(
                            name="ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnMetadata(
                            name="PARENT_ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=True,
                            is_foreign_key=True,
                            fk_references_table="CATEGORIES",
                            fk_references_column="ID",
                        ),
                    ],
                ),
                TableMetadata(
                    name="OBJ_ATTRIBUTES",
                    columns=[
                        ColumnMetadata(
                            name="ENTITY_ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="ATTRIBUTE",
                            data_type="VARCHAR2",
                            data_length=100,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="VALUE",
                            data_type="VARCHAR2",
                            data_length=4000,
                            nullable=True,
                        ),
                    ],
                ),
            ],
        )
        col_engine = ColumnRuleEngine()
        table_engine = TableRuleEngine()
        all_columns = [col for t in schema.tables for col in t.columns]

        col_results = col_engine.classify(all_columns, schema)
        table_results = table_engine.classify(schema)

        col_labels = {r.predicted_label for r in col_results}
        table_labels = {r.predicted_label for r in table_results}

        assert "self_referencing" in col_labels
        assert "eav" in table_labels


# ─── Edge Cases ────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases for rule engine classification."""

    def test_empty_schema_no_results(self, col_engine: ColumnRuleEngine) -> None:
        """Empty schema produces no classification results."""
        schema = DatabaseSchema(tables=[])
        all_columns: list[ColumnMetadata] = []
        results = col_engine.classify(all_columns, schema)
        assert results == []

    def test_clean_schema_no_flags(self, col_engine: ColumnRuleEngine) -> None:
        """Schema with well-designed tables produces no anti-pattern flags."""
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="EMPLOYEES",
                    columns=[
                        ColumnMetadata(
                            name="EMPLOYEE_ID",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=False,
                            is_primary_key=True,
                        ),
                        ColumnMetadata(
                            name="FIRST_NAME",
                            data_type="VARCHAR2",
                            data_length=50,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="LAST_NAME",
                            data_type="VARCHAR2",
                            data_length=50,
                            nullable=False,
                        ),
                        ColumnMetadata(
                            name="HIRE_DATE",
                            data_type="DATE",
                            data_length=7,
                            nullable=True,
                        ),
                        ColumnMetadata(
                            name="ANNUAL_SALARY",
                            data_type="NUMBER",
                            data_length=22,
                            nullable=True,
                        ),
                        ColumnMetadata(
                            name="IS_ACTIVE",
                            data_type="NUMBER",
                            data_length=1,
                            data_precision=1,
                            data_scale=0,
                            nullable=False,
                        ),
                    ],
                ),
            ],
        )
        all_columns = [col for t in schema.tables for col in t.columns]
        results = col_engine.classify(all_columns, schema)
        assert results == []

    def test_single_clean_column(self, col_engine: ColumnRuleEngine) -> None:
        """Single well-typed column produces no flags."""
        cols = [
            ColumnMetadata(
                name="ID",
                data_type="NUMBER",
                data_length=22,
                nullable=False,
                is_primary_key=True,
            )
        ]
        results = col_engine.classify(cols)
        assert results == []

    def test_all_reserved_wordss(self, col_engine: ColumnRuleEngine) -> None:
        """All columns named with reserved words are all flagged."""
        cols = [
            ColumnMetadata(
                name="SELECT",
                data_type="NUMBER",
                data_length=22,
                nullable=True,
            ),
            ColumnMetadata(
                name="FROM",
                data_type="VARCHAR2",
                data_length=50,
                nullable=True,
            ),
            ColumnMetadata(
                name="WHERE",
                data_type="VARCHAR2",
                data_length=50,
                nullable=True,
            ),
        ]
        results = col_engine.classify(cols)
        labels = [r.predicted_label for r in results]
        assert all(label == "reserved_words" for label in labels)
        assert len(results) == 3

    def test_giant_table_only(self, table_engine: TableRuleEngine) -> None:
        """Schema with only a giant table produces only giant_table flag."""
        cols = [
            ColumnMetadata(
                name=f"COL_{i}",
                data_type="NUMBER",
                data_length=22,
                nullable=True,
            )
            for i in range(25)
        ]
        schema = DatabaseSchema(
            tables=[TableMetadata(name="BIG_TABLE", columns=cols)]
        )
        results = table_engine.classify(schema)
        assert len(results) == 1
        assert results[0].predicted_label == "giant_table"


# ─── Regression: duplicate column names across tables ─────────────────

class TestDuplicateColumnNames:
    """Same column name in several tables must yield one detection each.

    Regression: ColumnRuleEngine used to emit table_name="" and the merge
    keyed by "{table}.{column}", collapsing e.g. ACTIVO in 4 tables into
    a single detection.
    """

    @staticmethod
    def _schema() -> DatabaseSchema:
        return DatabaseSchema(
            tables=[
                TableMetadata(
                    name=t,
                    columns=[
                        ColumnMetadata(
                            name="ACTIVO",
                            data_type="CHAR",
                            data_length=1,
                            nullable=True,
                        ),
                    ],
                )
                for t in ("EMPLEADOS", "USUARIOS_WEB", "CATEGORIAS")
            ]
        )

    def test_column_engine_detects_all_duplicates(self) -> None:
        """Column engine attributes each ACTIVO to its own table."""
        schema = self._schema()
        results = ColumnRuleEngine().classify(
            [c for t in schema.tables for c in t.columns], schema
        )
        assert len(results) == 3
        assert {r.table_name for r in results} == {
            "EMPLEADOS", "USUARIOS_WEB", "CATEGORIAS"
        }

    def test_classify_merge_no_collision(self) -> None:
        """Module-level classify() labels every duplicate column."""
        from rule_engine import classify as classify_schema

        results = classify_schema(schema=self._schema())
        assert len(results) == 3
        assert all(r.predicted_label == "bad_boolean" for r in results)
        assert {r.table_name for r in results} == {
            "EMPLEADOS", "USUARIOS_WEB", "CATEGORIAS"
        }

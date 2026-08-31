"""
test_text_preprocessor.py — Tests for the TextPreprocessor module.

Verifies:
  - CamelCase, PascalCase splitting
  - Snake_case / UPPER_CASE handling
  - Abbreviation expansion
  - Redundant prefix removal
  - Table context prefixing
  - Batch processing
  - Quick-access function
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import pytest

from db_bad_clust.features.text_preprocessor import TextPreprocessor, preprocess

# ── Fixture ─────────────────────────────────────────────────────────────


@pytest.fixture
def preprocessor() -> TextPreprocessor:
    """Default TextPreprocessor instance."""
    return TextPreprocessor()


# ── CamelCase / PascalCase splitting ────────────────────────────────────


class TestCamelCaseSplitting:
    """Verify _split_camel_case handles common patterns."""

    def test_simple_camel_case(self, preprocessor: TextPreprocessor) -> None:
        """fechaNacimiento → fecha Nacimiento."""
        result = preprocessor._split_camel_case("fechaNacimiento")
        assert result == "fecha Nacimiento"

    def test_pascal_case(self, preprocessor: TextPreprocessor) -> None:
        """FechaNacimiento → Fecha Nacimiento."""
        result = preprocessor._split_camel_case("FechaNacimiento")
        assert result == "Fecha Nacimiento"

    def test_all_uppercase_no_split(self, preprocessor: TextPreprocessor) -> None:
        """All-uppercase text like 'FECHA' should not be split."""
        result = preprocessor._split_camel_case("FECHA")
        assert result == "FECHA"

    def test_acronym_followed_by_word(self, preprocessor: TextPreprocessor) -> None:
        """XMLParser → XML Parser."""
        result = preprocessor._split_camel_case("XMLParser")
        assert result == "XML Parser"

    def test_multiple_camel_humps(self, preprocessor: TextPreprocessor) -> None:
        """fechaNacimientoEmpleado → fecha Nacimiento Empleado."""
        result = preprocessor._split_camel_case("fechaNacimientoEmpleado")
        assert result == "fecha Nacimiento Empleado"

    def test_single_word(self, preprocessor: TextPreprocessor) -> None:
        """Single lowercase word stays unchanged."""
        result = preprocessor._split_camel_case("nombre")
        assert result == "nombre"

    def test_empty_string(self, preprocessor: TextPreprocessor) -> None:
        """Empty string returns empty."""
        result = preprocessor._split_camel_case("")
        assert result == ""


# ── Snake_case / underscore splitting ──────────────────────────────────


class TestSnakeCaseSplitting:
    """Verify _split_on_underscores works correctly."""

    def test_snake_case(self, preprocessor: TextPreprocessor) -> None:
        """fecha_nacimiento → fecha nacimiento."""
        result = preprocessor._split_on_underscores("fecha_nacimiento")
        assert result == "fecha nacimiento"

    def test_upper_snake_case(self, preprocessor: TextPreprocessor) -> None:
        """FECHA_NACIMIENTO → FECHA NACIMIENTO."""
        result = preprocessor._split_on_underscores("FECHA_NACIMIENTO")
        assert result == "FECHA NACIMIENTO"

    def test_multiple_underscores(self, preprocessor: TextPreprocessor) -> None:
        """a__b → a  b (double underscore becomes two spaces)."""
        result = preprocessor._split_on_underscores("a__b")
        assert result == "a  b"

    def test_leading_underscore(self, preprocessor: TextPreprocessor) -> None:
        """_name →  name."""
        result = preprocessor._split_on_underscores("_name")
        assert result == " name"

    def test_no_underscores(self, preprocessor: TextPreprocessor) -> None:
        """No underscores → unchanged."""
        result = preprocessor._split_on_underscores("nombre")
        assert result == "nombre"


# ── Redundant prefix removal ────────────────────────────────────────────


class TestPrefixRemoval:
    """Verify _remove_redundant_prefixes strips known prefixes."""

    @pytest.mark.parametrize(
        "input_text, expected",
        [
            ("tbl_employees", "employees"),
            ("col_name", "name"),
            ("fld_value", "value"),
            ("tab_info", "info"),
            ("employee_name", "employee_name"),  # no prefix
            ("", ""),
        ],
    )
    def test_prefix_removal(
        self, preprocessor: TextPreprocessor, input_text: str, expected: str
    ) -> None:
        """Known prefixes (tbl, col, fld, tab) should be stripped."""
        # _remove_redundant_prefixes runs after _split_on_underscores,
        # so simulate the post-split input here.
        text = input_text.replace("_", " ")
        result = preprocessor._remove_redundant_prefixes(text)
        assert result.strip() == expected.replace("_", " ")

    def test_prefix_tbl(self, preprocessor: TextPreprocessor) -> None:
        """'tbl employees' → 'employees'."""
        result = preprocessor._remove_redundant_prefixes("tbl employees")
        assert result == "employees"

    def test_prefix_col(self, preprocessor: TextPreprocessor) -> None:
        """'col nombre' → 'nombre'."""
        result = preprocessor._remove_redundant_prefixes("col nombre")
        assert result == "nombre"

    def test_prefix_fld(self, preprocessor: TextPreprocessor) -> None:
        """'fld value' → 'value'."""
        result = preprocessor._remove_redundant_prefixes("fld value")
        assert result == "value"

    def test_no_prefix_unchanged(self, preprocessor: TextPreprocessor) -> None:
        """No recognized prefix → unchanged."""
        result = preprocessor._remove_redundant_prefixes("employee name")
        assert result == "employee name"

    def test_case_insensitive_prefix(self, preprocessor: TextPreprocessor) -> None:
        """'Tbl Employees' → 'Employees' (case-insensitive)."""
        result = preprocessor._remove_redundant_prefixes("Tbl Employees")
        assert result == "Employees"


# ── Abbreviation expansion ──────────────────────────────────────────────


class TestAbbreviationExpansion:
    """Verify _expand_token expands known abbreviations."""

    @pytest.mark.parametrize(
        "token, expected",
        [
            ("id", "identifier"),
            ("pk", "primary key"),
            ("fk", "foreign key"),
            ("num", "number"),
            ("desc", "description"),
            ("emp", "employee"),
            ("dept", "department"),
            ("qty", "quantity"),
            ("config", "configuration"),
            ("uuid", "unique identifier"),
            ("unknown_token", "unknown_token"),  # not in dict
        ],
    )
    def test_known_abbreviations(
        self, preprocessor: TextPreprocessor, token: str, expected: str
    ) -> None:
        """Known abbreviations should be expanded."""
        result = preprocessor._expand_token(token)
        assert result == expected

    def test_case_insensitive_expansion(self, preprocessor: TextPreprocessor) -> None:
        """Abbreviation lookup is case-insensitive."""
        assert preprocessor._expand_token("ID") == "identifier"
        assert preprocessor._expand_token("Id") == "identifier"
        assert preprocessor._expand_token("FK") == "foreign key"

    def test_token_with_punctuation(self, preprocessor: TextPreprocessor) -> None:
        """Punctuation is stripped for lookup."""
        result = preprocessor._expand_token("id,")
        assert result == "identifier"

    def test_custom_abbreviations(self) -> None:
        """Custom abbreviation dict overrides defaults."""
        custom = {"myabbr": "my expansion"}
        pp = TextPreprocessor(abbreviations=custom)
        assert pp._expand_token("myabbr") == "my expansion"
        # Defaults are not available
        assert pp._expand_token("id") == "id"


# ── Full pipeline ───────────────────────────────────────────────────────


class TestFullPipeline:
    """End-to-end tests of TextPreprocessor.process()."""

    def test_snake_case_column(self, preprocessor: TextPreprocessor) -> None:
        """FECHA_NACIMIENTO → fecha nacimiento (no table context)."""
        result = preprocessor.process("FECHA_NACIMIENTO")
        assert result == "fecha nacimiento"

    def test_camel_case_column(self, preprocessor: TextPreprocessor) -> None:
        """fechaNacimiento → fecha Nacimiento (mixed case preserved)."""
        result = preprocessor.process("fechaNacimiento")
        assert result == "fecha Nacimiento"

    def test_with_table_context(self, preprocessor: TextPreprocessor) -> None:
        """FECHA_NACIMIENTO with table EMPLEADOS → empleados: fecha nacimiento."""
        result = preprocessor.process("FECHA_NACIMIENTO", table_name="EMPLEADOS")
        assert result == "empleados: fecha nacimiento"

    def test_column_with_abbreviations(self, preprocessor: TextPreprocessor) -> None:
        """EMP_ID → empleado identifier (with emp and id expanded)."""
        result = preprocessor.process("EMP_ID")
        assert result == "employee identifier"

    def test_fk_column(self, preprocessor: TextPreprocessor) -> None:
        """FK_DEPTO_ID → foreign key department identifier."""
        result = preprocessor.process("FK_DEPTO_ID")
        # "depto" is not in abbreviations, so it stays
        assert "foreign key" in result
        assert "identifier" in result

    def test_prefixed_column(self, preprocessor: TextPreprocessor) -> None:
        """TBL_EMPLEADOS: 'tbl ' prefix removed, 'EMPLEADOS' left as-is."""
        result = preprocessor.process("TBL_EMPLEADOS")
        # 'tbl' is removed as prefix, 'EMPLEADOS' has no abbreviation
        assert result == "empleados"

    def test_column_with_table_and_abbrev(self, preprocessor: TextPreprocessor) -> None:
        """EMP_SALARIO in table EMPLEADOS → empleados: employee salary."""
        # 'salario' is not in the abbreviations dict, so it stays as 'salario'
        result = preprocessor.process("EMP_SALARIO", table_name="EMPLEADOS")
        # "EMP" expands to "employee", "salario" stays as is
        assert "empleados:" in result
        assert "employee" in result
        assert "salario" in result

    def test_empty_column_name(self, preprocessor: TextPreprocessor) -> None:
        """Empty column name produces empty string."""
        result = preprocessor.process("")
        assert result == ""

    def test_short_column_name(self, preprocessor: TextPreprocessor) -> None:
        """Column name 'ID' → 'identifier'."""
        result = preprocessor.process("ID")
        assert result == "identifier"


# ── Comment appending ───────────────────────────────────────────────────


class TestCommentAppending:
    """Verify comments are appended verbatim to the embedding text."""

    def test_comment_appended(self, preprocessor: TextPreprocessor) -> None:
        """Comment is appended after the contextualized name."""
        result = preprocessor.process(
            "EMAIL",
            table_name="EMPLEADOS",
            comment="Correo corporativo del empleado.",
        )
        assert result == "empleados: email | Correo corporativo del empleado."

    def test_comment_keeps_its_case(self, preprocessor: TextPreprocessor) -> None:
        """Natural-language comments are not case-normalized."""
        result = preprocessor.process("SALARIO", comment="Salario bruto anual en euros.")
        assert "Salario bruto anual en euros." in result

    def test_no_comment_unchanged(self, preprocessor: TextPreprocessor) -> None:
        """None or empty comment leaves the text unchanged."""
        assert preprocessor.process("SALARIO") == "salario"
        assert preprocessor.process("SALARIO", comment=None) == "salario"
        assert preprocessor.process("SALARIO", comment="   ") == "salario"


# ── Smart lowercasing ───────────────────────────────────────────────────


class TestSmartLowercasing:
    """Verify only ALL-CAPS tokens are lowercased (Oracle convention)."""

    def test_all_caps_token_lowercased(self, preprocessor: TextPreprocessor) -> None:
        """ALL-CAPS (Oracle default) → lowercase to avoid WordPiece fragmentation."""
        assert preprocessor.process("FECHA_NACIMIENTO") == "fecha nacimiento"

    def test_mixed_case_token_preserved(self, preprocessor: TextPreprocessor) -> None:
        """Mixed-case tokens (camelCase remnants) keep their case."""
        assert preprocessor.process("fechaNacimiento") == "fecha Nacimiento"

    def test_acronym_case_distinction_preserved(self, preprocessor: TextPreprocessor) -> None:
        """'Depto' vs 'DEPTO' are distinguishable after preprocessing."""
        assert "Depto" in preprocessor.process("DeptoId")
        assert "depto" in preprocessor.process("DEPTO_ID")


# ── Batch processing ────────────────────────────────────────────────────


class TestBatchProcessing:
    """Verify TextPreprocessor.process_batch()."""

    def test_batch_without_table(self, preprocessor: TextPreprocessor) -> None:
        """List of column names processed without table context."""
        columns = ["ID", "NOMBRE", "SALARIO"]
        results = preprocessor.process_batch(columns)
        assert len(results) == 3
        assert results[0] == "identifier"
        assert results[1] == "nombre"
        assert results[2] == "salario"

    def test_batch_with_table(self, preprocessor: TextPreprocessor) -> None:
        """List with shared table context."""
        columns = ["ID", "NOMBRE"]
        results = preprocessor.process_batch(columns, table_name="EMPLEADOS")
        assert len(results) == 2
        assert results[0] == "empleados: identifier"
        assert results[1] == "empleados: nombre"

    def test_empty_batch(self, preprocessor: TextPreprocessor) -> None:
        """Empty list → empty list."""
        assert preprocessor.process_batch([]) == []

    def test_batch_mixed_patterns(self, preprocessor: TextPreprocessor) -> None:
        """Mixed input patterns all get processed."""
        columns = ["FECHA_NACIMIENTO", "tbl_employees", "fkDeptoId"]
        results = preprocessor.process_batch(columns)
        assert len(results) == 3
        # ALL-CAPS tokens are lowercased; mixed-case tokens keep their case
        assert results[0] == "fecha nacimiento"
        assert results[1] == "employees"
        assert results[2] == "foreign key Depto identifier"


# ── Quick access function ──────────────────────────────────────────────


class TestQuickAccessFunction:
    """Verify the module-level preprocess() function."""

    def test_preprocess_no_table(self) -> None:
        """Module-level function works without table_name."""
        result = preprocess("FECHA_NACIMIENTO")
        assert result == "fecha nacimiento"

    def test_preprocess_with_table(self) -> None:
        """Module-level function works with table_name."""
        result = preprocess("FECHA_NACIMIENTO", table_name="EMPLEADOS")
        assert result == "empleados: fecha nacimiento"

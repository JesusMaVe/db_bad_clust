"""
Tests for schema_generator.py.

All tests are mock-based and do not require a running Oracle database.
"""

from __future__ import annotations

import pytest

from db_bad_clust.generation.schema_generator import (
    SchemaGeneratorConfig,
    SyntheticSchemaGenerator,
    _parse_fk_reference,
    _quote_if_reserved,
)


@pytest.fixture
def generator():
    """A deterministic schema generator."""
    return SyntheticSchemaGenerator(seed=42)


@pytest.fixture
def sample_output(generator):
    """A small generated schema + manifest."""
    return generator.generate(SchemaGeneratorConfig(num_tables=10, seed=42))


class TestSyntheticSchemaGenerator:
    def test_generates_requested_table_count(self, generator):
        tables, _ = generator.generate(SchemaGeneratorConfig(num_tables=50, seed=1))
        assert len(tables) == 50

    def test_reproducible_with_same_seed(self):
        g1 = SyntheticSchemaGenerator(seed=7)
        g2 = SyntheticSchemaGenerator(seed=7)
        tables1, _ = g1.generate(SchemaGeneratorConfig(num_tables=20, seed=7))
        tables2, _ = g2.generate(SchemaGeneratorConfig(num_tables=20, seed=7))
        assert [t.name for t in tables1] == [t.name for t in tables2]

    def test_density_increases_dirty_tables(self):
        low = SyntheticSchemaGenerator(seed=1)
        high = SyntheticSchemaGenerator(seed=1)
        _, manifest_low = low.generate(
            SchemaGeneratorConfig(num_tables=100, seed=1, anti_pattern_density=0.0)
        )
        _, manifest_high = high.generate(
            SchemaGeneratorConfig(num_tables=100, seed=1, anti_pattern_density=1.0)
        )
        clean_low = manifest_low["summary"]["column_label_counts"].get("clean", 0)
        clean_high = manifest_high["summary"]["column_label_counts"].get("clean", 0)
        assert clean_high < clean_low

    def test_manifest_structure(self, sample_output):
        tables, manifest = sample_output
        assert manifest["metadata"]["num_tables"] == len(tables)
        assert manifest["summary"]["total_columns"] == sum(
            len(t.columns) for t in tables
        )
        assert len(manifest["tables"]) == len(tables)
        first = manifest["tables"][0]
        assert "name" in first
        assert "anti_patterns" in first
        assert "columns" in first


class TestDDLRendering:
    def test_ddl_contains_all_tables(self, generator, sample_output):
        tables, _ = sample_output
        ddl = generator.to_ddl(tables)
        for table in tables:
            assert f"CREATE TABLE {table.name}" in ddl

    def test_reserved_words_quoted(self, generator):
        tables, _ = generator.generate(SchemaGeneratorConfig(num_tables=50, seed=3))
        ddl = generator.to_ddl(tables)
        assert '"FROM"' in ddl or '"WHERE"' in ddl or '"SELECT"' in ddl or '"NULL"' in ddl

    def test_pk_present_unless_missing(self, generator):
        tables, _ = generator.generate(SchemaGeneratorConfig(num_tables=50, seed=4))
        ddl = generator.to_ddl(tables)
        pk_count = ddl.count("PRIMARY KEY")
        assert pk_count > 0
        # Some tables are explicitly missing PKs.
        assert "CREATE TABLE" in ddl

    def test_disabled_constraints_rendered(self, generator):
        tables, _ = generator.generate(SchemaGeneratorConfig(num_tables=500, seed=5))
        ddl = generator.to_ddl(tables)
        assert " DISABLE" in ddl

    def test_long_types_rendered(self, generator):
        tables, _ = generator.generate(SchemaGeneratorConfig(num_tables=500, seed=6))
        ddl = generator.to_ddl(tables)
        assert " LONG" in ddl or " LONG RAW" in ddl or " RAW(" in ddl


class TestPhase2ReportFlags:
    def test_report_flags_present_at_high_density(self):
        generator = SyntheticSchemaGenerator(seed=1)
        _, manifest = generator.generate(
            SchemaGeneratorConfig(num_tables=1000, seed=1, anti_pattern_density=1.0)
        )
        flag_counts: dict[str, int] = {}
        for t in manifest["tables"]:
            flags = t.get("report_flags", {})
            for key, value in flags.items():
                if key == "oversized_varchars" and value:
                    flag_counts[key] = flag_counts.get(key, 0) + 1
                elif value:
                    flag_counts[key] = flag_counts.get(key, 0) + 1
        assert flag_counts.get("fk_without_index", 0) > 0
        assert flag_counts.get("stale_statistics", 0) > 0
        assert flag_counts.get("disabled_constraints", 0) > 0
        assert flag_counts.get("obsolete_types", 0) > 0
        assert flag_counts.get("partition_candidate", 0) > 0
        assert flag_counts.get("oversized_varchars", 0) > 0


class TestDatabaseSchemaConversion:
    def test_to_database_schema_populates_metadata(self):
        generator = SyntheticSchemaGenerator(seed=1)
        tables, _ = generator.generate(
            SchemaGeneratorConfig(num_tables=200, seed=1, anti_pattern_density=1.0)
        )
        schema = SyntheticSchemaGenerator.to_database_schema(tables)
        assert len(schema.tables) == 200
        assert schema.fk_columns or schema.index_columns or schema.table_statistics

    def test_to_database_schema_disabled_constraints(self):
        generator = SyntheticSchemaGenerator(seed=2)
        tables, _ = generator.generate(
            SchemaGeneratorConfig(num_tables=500, seed=2, anti_pattern_density=1.0)
        )
        schema = SyntheticSchemaGenerator.to_database_schema(tables)
        disabled = any(
            c["status"] == "DISABLED"
            for constraints in schema.constraint_status.values()
            for c in constraints
        )
        assert disabled


class TestHelpers:
    @pytest.mark.parametrize(
        "expr,expected",
        [
            ("DEPARTAMENTOS(ID)", {"ref_table": "DEPARTAMENTOS", "ref_column": "ID", "on_delete": "", "disabled": False}),
            ("DEPARTAMENTOS(ID) ON DELETE CASCADE", {"ref_table": "DEPARTAMENTOS", "ref_column": "ID", "on_delete": "ON DELETE CASCADE", "disabled": False}),
            ("TODO_EN_UNO(ID) DISABLE", {"ref_table": "TODO_EN_UNO", "ref_column": "ID", "on_delete": "", "disabled": True}),
        ],
    )
    def test_parse_fk_reference(self, expr, expected):
        parsed = _parse_fk_reference(expr)
        assert parsed == expected

    def test_parse_fk_reference_invalid(self):
        assert _parse_fk_reference("not valid") is None

    def test_quote_if_reserved(self):
        assert _quote_if_reserved("FROM") == '"FROM"'
        assert _quote_if_reserved("NOMBRE") == "NOMBRE"

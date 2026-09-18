"""Tests for the per-table aggregate feature block. No DB required."""

from __future__ import annotations

import numpy as np
import pytest

from db_bad_clust.data.schema_extractor import ColumnMetadata, DatabaseSchema, TableMetadata
from db_bad_clust.features.table_aggregates import (
    N_TABLE_FEATURES,
    _is_generic_name,
    build_table_block,
    table_stats,
)


def _col(name, data_type="VARCHAR2", nullable=True, comments=None):
    return ColumnMetadata(
        name=name, data_type=data_type, data_length=50, nullable=nullable, comments=comments
    )


class TestIsGenericName:
    def test_short_letter_codes_are_generic(self):
        assert _is_generic_name("A")
        assert _is_generic_name("AAA")
        assert _is_generic_name("F1")

    def test_col_number_pattern_is_generic(self):
        assert _is_generic_name("COL_001")
        assert _is_generic_name("COL_50")

    def test_real_words_are_not_generic(self):
        assert not _is_generic_name("SALARIO")
        assert not _is_generic_name("FECHA_NACIMIENTO")
        assert not _is_generic_name("NOMBRE_EMPLEADO")


class TestTableStats:
    def test_shape_is_five(self):
        schema = DatabaseSchema(
            tables=[TableMetadata(name="T", columns=[_col("A"), _col("B", nullable=False)])]
        )
        stats = table_stats(schema)
        assert stats["T"].shape == (N_TABLE_FEATURES,)

    def test_column_count_is_log1p(self):
        schema = DatabaseSchema(
            tables=[TableMetadata(name="T", columns=[_col(f"C{i}") for i in range(9)])]
        )
        stats = table_stats(schema)
        assert stats["T"][0] == np.log1p(9)

    def test_nullable_ratio(self):
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="T",
                    columns=[_col("A", nullable=True), _col("B", nullable=False)],
                )
            ]
        )
        assert table_stats(schema)["T"][1] == 0.5

    def test_type_diversity(self):
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="T",
                    columns=[
                        _col("A", data_type="NUMBER"),
                        _col("B", data_type="NUMBER"),
                        _col("C", data_type="VARCHAR2"),
                        _col("D", data_type="DATE"),
                    ],
                )
            ]
        )
        # 3 distinct types / 4 columns
        assert table_stats(schema)["T"][2] == 0.75

    def test_generic_name_rate(self):
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="T",
                    columns=[_col("A"), _col("SALARIO"), _col("COL_001"), _col("NOMBRE")],
                )
            ]
        )
        # A and COL_001 are generic → 2/4
        assert table_stats(schema)["T"][3] == 0.5

    def test_comment_ratio(self):
        schema = DatabaseSchema(
            tables=[
                TableMetadata(
                    name="T",
                    columns=[
                        _col("A", comments="algo"),
                        _col("B", comments=None),
                        _col("C", comments=None),
                    ],
                )
            ]
        )
        assert table_stats(schema)["T"][4] == pytest.approx(1 / 3)

    def test_empty_table_is_zero_vector(self):
        schema = DatabaseSchema(tables=[TableMetadata(name="EMPTY", columns=[])])
        assert np.array_equal(table_stats(schema)["EMPTY"], np.zeros(N_TABLE_FEATURES))


class TestBuildTableBlock:
    def test_every_column_of_a_table_shares_its_stats(self):
        schema = DatabaseSchema(
            tables=[TableMetadata(name="T", columns=[_col("A"), _col("B")])]
        )
        block = build_table_block(schema, ["T.A", "T.B"])
        assert block.shape == (2, N_TABLE_FEATURES)
        np.testing.assert_array_equal(block[0], block[1])

    def test_different_tables_get_different_rows_when_stats_differ(self):
        schema = DatabaseSchema(
            tables=[
                TableMetadata(name="SMALL", columns=[_col("A")]),
                TableMetadata(
                    name="BIG", columns=[_col(f"C{i}") for i in range(20)]
                ),
            ]
        )
        block = build_table_block(schema, ["SMALL.A", "BIG.C0"])
        assert block[0, 0] != block[1, 0]  # column-count feature differs

    def test_column_order_follows_column_index(self):
        schema = DatabaseSchema(
            tables=[
                TableMetadata(name="A", columns=[_col("X")]),
                TableMetadata(name="B", columns=[_col(f"Y{i}") for i in range(5)]),
            ]
        )
        block = build_table_block(schema, ["B.Y0", "A.X", "B.Y1"])
        assert block[0, 0] == block[2, 0]  # both from table B
        assert block[1, 0] != block[0, 0]  # table A differs

    def test_unknown_table_gets_zero_row(self):
        schema = DatabaseSchema(tables=[TableMetadata(name="T", columns=[_col("A")])])
        block = build_table_block(schema, ["OTHER.X"])
        assert np.array_equal(block[0], np.zeros(N_TABLE_FEATURES))

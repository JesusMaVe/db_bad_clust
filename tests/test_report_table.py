"""Tests for the shared table-rendering seam. Pure layout, no I/O."""

from __future__ import annotations

import pytest

from db_bad_clust.evaluation.report_table import Column, render_table


class TestRenderTable:
    def test_header_and_rows_are_padded_to_column_width(self):
        columns = [Column("name", 10, "<"), Column("value", 6)]
        text = render_table(columns, [["a", "1.00"]])
        lines = text.splitlines()
        assert lines[0] == f"{'name':<10} {'value':>6}"
        assert lines[2] == f"{'a':<10} {'1.00':>6}"

    def test_separator_length_matches_the_header(self):
        columns = [Column("name", 10, "<"), Column("value", 6)]
        text = render_table(columns, [])
        header, separator = text.splitlines()
        assert separator == "-" * len(header)

    def test_no_rows_still_renders_header_and_separator(self):
        columns = [Column("a", 4)]
        text = render_table(columns, [])
        assert len(text.splitlines()) == 2

    def test_left_and_right_alignment_both_respected(self):
        columns = [Column("left", 8, "<"), Column("right", 8, ">")]
        text = render_table(columns, [["x", "y"]])
        row = text.splitlines()[2]
        assert row.startswith("x       ")  # left-padded
        assert row.endswith("       y")  # right-padded

    def test_wrong_cell_count_raises(self):
        columns = [Column("a", 4), Column("b", 4)]
        with pytest.raises(ValueError, match="cells"):
            render_table(columns, [["only-one"]])

    def test_composite_string_cells_pass_through_unformatted(self):
        """Callers may hand in already-composed cells like '24/243'."""
        columns = [Column("name", 10, "<"), Column("ratio", 10)]
        text = render_table(columns, [["structure", "24/243"]])
        assert "24/243" in text.splitlines()[2]

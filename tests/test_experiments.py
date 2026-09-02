"""Tests for the experiment harness. No DB and no pickle required."""

from __future__ import annotations

import numpy as np
import pytest

from db_bad_clust.evaluation.experiments import (
    Dataset,
    evaluate,
    format_table,
    run_clustering,
)


def _dataset(n: int = 40) -> Dataset:
    """Two well-separated groups, half in each of two tables."""
    rng = np.random.default_rng(0)
    half = n // 2
    e_text = np.vstack([rng.normal(0, 0.1, (half, 8)), rng.normal(5, 0.1, (half, 8))])
    e_type = np.vstack([np.tile([1, 0], (half, 1)), np.tile([0, 1], (half, 1))]).astype(float)
    e_rest = rng.normal(0, 1, (n, 3))
    e_stat = rng.normal(0, 1, (n, 1))
    return Dataset(
        e_text=e_text,
        e_type=e_type,
        e_rest=e_rest,
        e_stat=e_stat,
        column_index=[f"T{i // half}.C{i}" for i in range(n)],
        truth=["eav"] * half + ["clean"] * half,
    )


class TestDataset:
    def test_table_of_splits_on_the_first_dot(self):
        data = Dataset(
            e_text=np.zeros((2, 1)),
            e_type=np.zeros((2, 1)),
            e_rest=np.zeros((2, 1)),
            e_stat=np.zeros((2, 1)),
            column_index=["EMPLEADOS.FECHA", "TODO_EN_UNO.NOMBRE_O_DESC"],
            truth=["clean", "polymorphic"],
        )
        assert data.table_of == ["EMPLEADOS", "TODO_EN_UNO"]

    def test_n_columns_matches_the_index(self):
        assert _dataset(40).n_columns == 40

    def test_rejects_blocks_of_differing_length(self):
        with pytest.raises(ValueError, match="same number of rows"):
            Dataset(
                e_text=np.zeros((3, 1)),
                e_type=np.zeros((2, 1)),
                e_rest=np.zeros((2, 1)),
                e_stat=np.zeros((2, 1)),
                column_index=["T.A", "T.B"],
                truth=["clean", "clean"],
            )

    def test_subset_keeps_only_the_named_tables(self):
        data = _dataset(40)
        kept = data.without_tables({"T0"})
        assert kept.n_columns == 20
        assert set(kept.table_of) == {"T1"}
        assert kept.e_text.shape == (20, 8)


class TestRunClustering:
    def test_returns_one_label_per_column(self):
        data = _dataset(40)
        labels = run_clustering(data, {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25})
        assert labels.shape == (40,)

    def test_is_deterministic(self):
        data = _dataset(40)
        weights = {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25}
        first = run_clustering(data, weights)
        second = run_clustering(data, weights)
        assert np.array_equal(first, second)

    def test_separable_groups_are_found(self):
        """Two clouds 5 sigma apart must not be merged into one cluster."""
        data = _dataset(40)
        labels = run_clustering(
            data,
            {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0},
            min_cluster_size=5,
        )
        assert len(set(labels.tolist()) - {-1}) >= 2


class TestEvaluate:
    def test_recovers_the_planted_grouping(self):
        """The two clouds are the two classes, so naming them must score high."""
        data = _dataset(40)
        result = evaluate(
            "semantic only",
            data,
            {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0},
            min_cluster_size=5,
        )
        assert result.score.name == "semantic only"
        assert result.score.accuracy == pytest.approx(1.0)
        assert result.score.ari == pytest.approx(1.0)
        assert len(result.predicted) == 40

    def test_carries_the_raw_cluster_ids(self):
        data = _dataset(40)
        result = evaluate("x", data, {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0})
        assert len(result.cluster_ids) == 40


class TestFormatTable:
    def test_renders_one_line_per_row_with_the_name(self):
        rows = [
            evaluate("solo BERT", _dataset(40), {"alpha": 1.0, "beta": 0, "gamma": 0, "delta": 0}).score
        ]
        text = format_table(rows)
        assert "solo BERT" in text
        assert "ARI" in text

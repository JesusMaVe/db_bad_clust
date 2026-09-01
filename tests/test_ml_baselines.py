"""Tests for the losing-ML baselines. No DB and no pickle required."""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from db_bad_clust.evaluation.ml_baselines import (
    clustering_algorithm_comparison,
    format_baselines,
    random_forest_baseline,
)


@pytest.fixture
def fixture_paths(tmp_path):
    """A tiny but well-formed pickle + label file, enough to exercise both baselines."""
    rng = np.random.default_rng(0)
    n = 40
    columns = [f"T{i % 4}.C{i}" for i in range(n)]
    # two well-separated blobs so clustering has something real to find
    half = n // 2
    e_rest = np.vstack([rng.normal(0, 0.2, (half, 5)), rng.normal(6, 0.2, (n - half, 5))])
    data = {
        "column_index": columns,
        "e_text": rng.normal(0, 1, (n, 8)),
        "e_type": rng.normal(0, 1, (n, 4)),
        "e_rest": e_rest,
        "e_stat": rng.normal(0, 1, (n, 1)),
        "phi": np.hstack([e_rest, rng.normal(0, 1, (n, 3))]),
    }
    pkl = tmp_path / "inter.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump(data, fh)

    labels = tmp_path / "labels.csv"
    rows = ["table,column,data_type,label"]
    for i, key in enumerate(columns):
        table, column = key.split(".")
        rows.append(f"{table},{column},NUMBER,{'clean' if i < half else 'eav'}")
    labels.write_text("\n".join(rows) + "\n")
    return pkl, labels


class TestRandomForestBaseline:
    def test_reports_a_score_and_the_rarest_classes(self, fixture_paths):
        pkl, labels = fixture_paths
        result = random_forest_baseline(pickle_path=pkl, labels_path=labels, folds=2)
        assert 0.0 <= result["f1_macro_mean"] <= 1.0
        assert result["folds"] == 2
        assert result["n_features"] == 8
        assert len(result["rarest_classes"]) == 2

    def test_top_features_are_index_weight_pairs(self, fixture_paths):
        pkl, labels = fixture_paths
        result = random_forest_baseline(pickle_path=pkl, labels_path=labels, folds=2)
        assert len(result["top_features"]) == 8  # capped at 10, only 8 exist
        for index, weight in result["top_features"]:
            assert isinstance(index, int)
            assert weight >= 0.0

    def test_degenerate_folds_are_surfaced_not_hidden(self, tmp_path, fixture_paths):
        """A class with one member is the finding — it must reach the caller."""
        pkl, labels = fixture_paths
        text = labels.read_text().splitlines()
        text[1] = text[1].rsplit(",", 1)[0] + ",self_referencing"  # a class of exactly one
        rare = tmp_path / "rare.csv"
        rare.write_text("\n".join(text) + "\n")
        result = random_forest_baseline(pickle_path=pkl, labels_path=rare, folds=5)
        assert result["split_warnings"]
        assert ("self_referencing", 1) in result["rarest_classes"]


class TestClusteringComparison:
    def test_scores_every_method_and_sorts_by_ari(self, fixture_paths):
        pkl, labels = fixture_paths
        rows = clustering_algorithm_comparison(
            pickle_path=pkl, labels_path=labels, methods=("kmeans", "agglomerative")
        )
        assert len(rows) == 2
        scored = [r["ari"] for r in rows if "ari" in r]
        assert scored == sorted(scored, reverse=True)

    def test_separable_blobs_are_found(self, fixture_paths):
        pkl, labels = fixture_paths
        rows = clustering_algorithm_comparison(
            pickle_path=pkl, labels_path=labels, methods=("kmeans",)
        )
        assert rows[0]["ari"] > 0.5

    def test_a_failing_method_is_recorded_not_raised(self, fixture_paths):
        pkl, labels = fixture_paths
        rows = clustering_algorithm_comparison(
            pickle_path=pkl, labels_path=labels, methods=("no_such_method",)
        )
        assert "error" in rows[0]


class TestFormatting:
    def test_report_names_both_baselines_and_the_ground_truth_caveat(self, fixture_paths):
        pkl, labels = fixture_paths
        text = format_baselines(
            random_forest_baseline(pickle_path=pkl, labels_path=labels, folds=2),
            clustering_algorithm_comparison(
                pickle_path=pkl, labels_path=labels, methods=("kmeans",)
            ),
        )
        assert "Random Forest" in text
        assert "kmeans" in text
        assert "circular" in text

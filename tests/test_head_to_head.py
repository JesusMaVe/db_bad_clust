"""Tests for the A-vs-B comparison harness. No DB and no pickle required."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from db_bad_clust.evaluation.head_to_head import (
    BranchScore,
    ComparisonResult,
    _score,
    clusters_to_labels,
    format_report,
    load_manual_labels,
    majority_vote_map,
    write_csv,
)


class TestMajorityVoteMap:
    def test_names_cluster_after_dominant_label(self):
        clusters = np.array([0, 0, 0, 1, 1])
        truth = ["eav", "eav", "clean", "giant_table", "giant_table"]
        assert majority_vote_map(clusters, truth) == {0: "eav", 1: "giant_table"}

    def test_noise_cluster_is_named_too(self):
        """-1 is treated as an ordinary cluster: the generous reading for B."""
        clusters = np.array([-1, -1, 0])
        truth = ["clean", "clean", "eav"]
        mapping = majority_vote_map(clusters, truth)
        assert mapping[-1] == "clean"

    def test_single_member_cluster_takes_its_own_label(self):
        assert majority_vote_map(np.array([7]), ["polymorphic"]) == {7: "polymorphic"}


class TestClustersToLabels:
    def test_maps_every_column(self):
        clusters = np.array([0, 1, 0, 1])
        truth = ["eav", "clean", "eav", "clean"]
        assert clusters_to_labels(clusters, truth) == truth

    def test_minority_members_inherit_the_majority_name(self):
        """A cluster cannot express two labels — the minority is lost."""
        clusters = np.array([0, 0, 0])
        truth = ["clean", "clean", "reserved_words"]
        assert clusters_to_labels(clusters, truth) == ["clean", "clean", "clean"]


class TestScore:
    def test_perfect_prediction_scores_one(self):
        truth = ["eav", "eav", "clean", "clean"]
        score = _score("perfect", truth, truth)
        assert score.accuracy == pytest.approx(1.0)
        assert score.f1_macro == pytest.approx(1.0)
        assert score.ari == pytest.approx(1.0)

    def test_grouping_overrides_label_names_for_ari(self):
        """ARI is computed on the partition, so cluster ids work directly."""
        truth = ["eav", "eav", "clean", "clean"]
        score = _score("clusters", truth, truth, group_ids=np.array([3, 3, 9, 9]))
        assert score.ari == pytest.approx(1.0)
        assert score.n_groups == 2

    def test_all_wrong_scores_zero_accuracy(self):
        score = _score("wrong", ["clean"] * 4, ["eav"] * 4)
        assert score.accuracy == pytest.approx(0.0)


class TestLoadManualLabels:
    def _write(self, path, rows):
        with open(path, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["table", "column", "data_type", "label"])
            writer.writerows(rows)

    def test_returns_labels_in_column_index_order(self, tmp_path):
        path = tmp_path / "labels.csv"
        self._write(path, [["T", "B", "NUMBER", "eav"], ["T", "A", "VARCHAR2", "clean"]])
        assert load_manual_labels(path, ["T.A", "T.B"]) == ["clean", "eav"]

    def test_matching_is_case_insensitive(self, tmp_path):
        path = tmp_path / "labels.csv"
        self._write(path, [["t", "a", "NUMBER", "clean"]])
        assert load_manual_labels(path, ["T.A"]) == ["clean"]

    def test_raises_when_a_column_is_unlabelled(self, tmp_path):
        path = tmp_path / "labels.csv"
        self._write(path, [["T", "A", "NUMBER", "clean"]])
        with pytest.raises(ValueError, match="lack a manual label"):
            load_manual_labels(path, ["T.A", "T.MISSING"])


def _result() -> ComparisonResult:
    return ComparisonResult(
        rules=BranchScore("Rule engine (A)", 0.76, 0.80, 0.89, 0.84, 9),
        clustering=BranchScore("Clustering (B)", 0.50, 0.42, 0.71, 0.26, 13),
        n_columns=3,
        truth=["eav", "clean", "polymorphic"],
        rules_pred=["eav", "clean", "polymorphic"],
        clustering_pred=["eav", "clean", "clean"],
        column_index=["T.A", "T.B", "T.C"],
        cluster_raw=[0, 1, 1],
        per_class_f1={
            "rules": {"clean": 1.0, "eav": 1.0, "polymorphic": 1.0},
            "clustering": {"clean": 0.6, "eav": 1.0, "polymorphic": 0.0},
        },
        cluster_contents={0: __import__("collections").Counter({"eav": 1})},
    )


class TestDisagreements:
    def test_finds_columns_where_a_wins(self):
        rows = _result().disagreements()
        assert [r["column"] for r in rows] == ["T.C"]
        assert rows[0]["truth"] == "polymorphic"
        assert rows[0]["clustering"] == "clean"

    def test_reverse_is_empty_when_b_never_wins_alone(self):
        assert _result().reverse_disagreements() == []

    def test_limit_truncates(self):
        assert _result().disagreements(limit=0) == []


class TestReporting:
    def test_report_contains_both_branches_and_the_caveat(self):
        text = format_report(_result())
        assert "Rule engine (A)" in text
        assert "Clustering (B)" in text
        assert "upper bound" in text

    def test_write_csv_emits_one_row_per_column(self, tmp_path):
        path = tmp_path / "out.csv"
        write_csv(_result(), path)
        rows = list(csv.DictReader(open(path)))
        assert len(rows) == 3
        assert rows[2] == {
            "column": "T.C",
            "truth": "polymorphic",
            "rules_pred": "polymorphic",
            "cluster_id": "1",
            "clustering_pred": "clean",
        }

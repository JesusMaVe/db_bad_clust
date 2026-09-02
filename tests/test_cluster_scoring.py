"""Tests for the clustering scoring harness. No DB and no pickle required."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from db_bad_clust.evaluation.cluster_scoring import (
    ClusterScore,
    clusters_to_labels,
    load_manual_labels,
    majority_vote_map,
    score,
    write_per_column_csv,
)


class TestMajorityVoteMap:
    def test_names_cluster_after_dominant_label(self):
        clusters = np.array([0, 0, 0, 1, 1])
        truth = ["eav", "eav", "clean", "giant_table", "giant_table"]
        assert majority_vote_map(clusters, truth) == {0: "eav", 1: "giant_table"}

    def test_noise_cluster_is_named_too(self):
        """-1 is treated as an ordinary cluster: the generous reading."""
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
        result = score("perfect", truth, truth)
        assert result.accuracy == pytest.approx(1.0)
        assert result.f1_macro == pytest.approx(1.0)
        assert result.ari == pytest.approx(1.0)

    def test_grouping_overrides_label_names_for_ari(self):
        """ARI is computed on the partition, so cluster ids work directly."""
        truth = ["eav", "eav", "clean", "clean"]
        result = score("clusters", truth, truth, group_ids=np.array([3, 3, 9, 9]))
        assert result.ari == pytest.approx(1.0)
        assert result.n_groups == 2

    def test_all_wrong_scores_zero_accuracy(self):
        assert score("wrong", ["clean"] * 4, ["eav"] * 4).accuracy == pytest.approx(0.0)

    def test_as_row_rounds_to_four_places(self):
        row = ClusterScore("x", 1 / 3, 0.5, 0.5, 0.5, 2).as_row()
        assert row["ari"] == 0.3333
        assert row["branch"] == "x"


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


class TestWritePerColumnCsv:
    def test_emits_one_row_per_column(self, tmp_path):
        path = tmp_path / "out.csv"
        write_per_column_csv(
            path,
            column_index=["T.A", "T.B", "T.C"],
            truth=["eav", "clean", "polymorphic"],
            cluster_ids=[0, 1, 1],
            predicted=["eav", "clean", "clean"],
        )
        rows = list(csv.DictReader(open(path)))
        assert len(rows) == 3
        assert rows[2] == {
            "column": "T.C",
            "truth": "polymorphic",
            "cluster_id": "1",
            "predicted": "clean",
        }

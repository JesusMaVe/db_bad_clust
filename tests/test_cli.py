"""Tests for the command-line entry point. No database required."""

from __future__ import annotations

import csv
import pickle

import numpy as np
import pytest

from db_bad_clust.cli import build_parser, main


@pytest.fixture
def fixture_paths(tmp_path):
    """A minimal pickle + label CSV pair, in the shape notebook 02 produces."""
    n, half = 40, 20
    rng = np.random.default_rng(0)
    column_index = [f"T{i // half}.C{i}" for i in range(n)]
    pickle_path = tmp_path / "intermediate.pkl"
    with open(pickle_path, "wb") as fh:
        pickle.dump(
            {
                "e_text": np.vstack(
                    [rng.normal(0, 0.1, (half, 8)), rng.normal(5, 0.1, (half, 8))]
                ),
                "e_type": np.vstack(
                    [np.tile([1, 0], (half, 1)), np.tile([0, 1], (half, 1))]
                ).astype(float),
                "e_rest": rng.normal(0, 1, (n, 3)),
                "e_stat": rng.normal(0, 1, (n, 1)),
                "phi": rng.normal(0, 1, (n, 12)),
                "column_index": column_index,
            },
            fh,
        )

    labels_path = tmp_path / "labels.csv"
    with open(labels_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["table", "column", "data_type", "label"])
        for i, key in enumerate(column_index):
            table, column = key.split(".")
            writer.writerow([table, column, "NUMBER", "eav" if i < half else "clean"])

    return str(pickle_path), str(labels_path)


class TestLoadDataset:
    def test_blocks_are_widened_to_float64(self, fixture_paths):
        """The pickle stores float32; the working width is pinned deliberately."""
        from db_bad_clust.evaluation.experiments import load_dataset

        pickle_path, labels_path = fixture_paths
        dataset = load_dataset(pickle_path, labels_path)
        assert all(block.dtype == np.float64 for block in dataset.blocks)


class TestParser:
    def test_requires_a_subcommand(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_experiment_defaults_need_no_database(self):
        args = build_parser().parse_args(["experiment"])
        assert args.pickle == "output/intermediate_02.pkl"
        assert args.labels == "output/manual_labels.csv"
        assert args.baselines is False

    def test_audit_command_is_gone(self):
        """The audit product lives on the rule-engine branch, not here."""
        with pytest.raises(SystemExit):
            build_parser().parse_args(["audit"])


class TestExperiment:
    def test_reports_the_corpus_size_and_a_score_table(self, fixture_paths, capsys):
        pickle_path, labels_path = fixture_paths
        code = main(["experiment", "--pickle", pickle_path, "--labels", labels_path, "--csv", ""])
        out = capsys.readouterr().out
        assert code == 0
        assert "40 columns, 2 tables" in out
        assert "structure only (alpha=0)" in out
        assert "upper bound" in out
        assert "ARI float32" in out
        assert "largest tie" in out

    def test_writes_per_column_verdicts_when_asked(self, fixture_paths, tmp_path, capsys):
        pickle_path, labels_path = fixture_paths
        target = tmp_path / "nested" / "per_column.csv"
        code = main(
            ["experiment", "--pickle", pickle_path, "--labels", labels_path, "--csv", str(target)]
        )
        assert code == 0
        rows = list(csv.DictReader(open(target)))
        assert len(rows) == 40
        assert set(rows[0]) == {"column", "truth", "cluster_id", "predicted"}
        assert str(target) in capsys.readouterr().out


class TestErrorHandling:
    def test_missing_input_file_reports_the_path(self, capsys):
        code = main(["experiment", "--pickle", "does/not/exist.pkl"])
        assert code == 1
        assert "does/not/exist.pkl" in capsys.readouterr().err

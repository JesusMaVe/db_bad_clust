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
    groups = np.vstack(
        [np.tile([1, 0, 0, 0, 0, 0], (half, 1)), np.tile([0, 0, 0, 1, 0, 0], (half, 1))]
    )

    def conflict_block(noise: float) -> np.ndarray:
        return np.hstack(
            [groups + rng.normal(0, noise, (n, 6)), groups, rng.normal(0, noise, (n, 1))]
        )

    e_conflict = conflict_block(0.05)
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
                "e_conflict": e_conflict,
                "e_conflict_variants": {
                    "orig": e_conflict,
                    "W2": conflict_block(0.08),
                    "W3": conflict_block(0.10),
                },
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


class TestSweepAndAblation:
    def test_sweep_reports_one_row_per_alpha(self, fixture_paths, capsys):
        pickle_path, labels_path = fixture_paths
        code = main(
            ["experiment", "--pickle", pickle_path, "--labels", labels_path,
             "--csv", "", "--sweep"]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "alpha=0.00" in out
        assert "alpha=1.00" in out

    def test_ablation_compares_the_given_pickles_against_the_floor(
        self, fixture_paths, capsys
    ):
        pickle_path, labels_path = fixture_paths
        code = main(
            ["experiment", "--pickle", pickle_path, "--labels", labels_path,
             "--csv", "", "--ablation", pickle_path]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "structure only (alpha=0)" in out

    def test_without_giants_says_what_it_dropped_and_why(self, fixture_paths, capsys):
        pickle_path, labels_path = fixture_paths
        code = main(
            ["experiment", "--pickle", pickle_path, "--labels", labels_path,
             "--csv", "", "--without-giants"]
        )
        out = capsys.readouterr().out
        assert code == 0
        assert "property" in out
        assert "40 columns" in out  # the fixture has no giant tables to drop


class TestErrorHandling:
    def test_missing_input_file_reports_the_path(self, capsys):
        code = main(["experiment", "--pickle", "does/not/exist.pkl"])
        assert code == 1
        assert "does/not/exist.pkl" in capsys.readouterr().err


class TestConflictFlags:
    def test_conflict_prints_every_config_under_both_algorithms(self, fixture_paths, capsys):
        pickle_path, labels_path = fixture_paths
        assert (
            main(["experiment", "--pickle", pickle_path, "--labels", labels_path, "--conflict"])
            == 0
        )
        out = capsys.readouterr().out
        for config in ("document (alpha=1)", "conflict fused"):
            assert config in out
        assert "ward" in out and "hdbscan" in out
        assert "k=" in out  # Ward's blind choice is printed
        assert "ARI~tbl" in out
        assert "Cluster composition, conflict fused / ward" in out
        assert "tables=" in out

    def test_robustness_prints_one_row_per_wording_and_the_mean(self, fixture_paths, capsys):
        pickle_path, labels_path = fixture_paths
        assert (
            main(["experiment", "--pickle", pickle_path, "--labels", labels_path, "--robustness"])
            == 0
        )
        out = capsys.readouterr().out
        for wording in ("orig", "W2", "W3"):
            assert wording in out
        assert "mean" in out and "smallest k" in out
        assert "conflict fused / ward" in out and "conflict fused / hdbscan" in out

    def test_robustness_without_variants_is_refused_with_a_hint(
        self, fixture_paths, tmp_path, capsys
    ):
        pickle_path, labels_path = fixture_paths
        with open(pickle_path, "rb") as fh:
            data = pickle.load(fh)
        del data["e_conflict_variants"]
        bare = tmp_path / "novariants.pkl"
        with open(bare, "wb") as fh:
            pickle.dump(data, fh)
        assert (
            main(["experiment", "--pickle", str(bare), "--labels", labels_path, "--robustness"])
            == 1
        )
        assert "from-pickle" in capsys.readouterr().err

    def test_stability_prints_the_grid_and_a_blind_pick(self, fixture_paths, capsys):
        pickle_path, labels_path = fixture_paths
        assert (
            main(["experiment", "--pickle", pickle_path, "--labels", labels_path, "--stability"])
            == 0
        )
        out = capsys.readouterr().out
        assert "validity" in out
        assert "mcs" in out

    def test_a_pickle_without_the_block_is_refused_with_a_hint(
        self, fixture_paths, tmp_path, capsys
    ):
        pickle_path, labels_path = fixture_paths
        with open(pickle_path, "rb") as fh:
            data = pickle.load(fh)
        del data["e_conflict"]
        bare = tmp_path / "bare.pkl"
        with open(bare, "wb") as fh:
            pickle.dump(data, fh)
        assert (
            main(["experiment", "--pickle", str(bare), "--labels", labels_path, "--conflict"]) == 1
        )
        assert "from-pickle" in capsys.readouterr().err

    def test_bootstrap_prints_intervals_and_paired_differences(self, fixture_paths, capsys):
        pickle_path, labels_path = fixture_paths
        args = ["experiment", "--pickle", pickle_path, "--labels", labels_path]
        assert main([*args, "--bootstrap", "3", "--bootstrap-frac", "0.6"]) == 0
        out = capsys.readouterr().out
        assert "Bootstrap, 3 resamples" in out
        assert "95% CI" in out
        assert "conflict mean / ward - document / ward" in out

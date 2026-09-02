"""Tests for the experiment harness. No DB and no pickle required."""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from db_bad_clust.evaluation.experiments import (
    Dataset,
    ablation,
    block_variance_shares,
    build_phi,
    evaluate,
    format_diagnostics,
    format_table,
    precision_sensitivity,
    representation_degeneracy,
    run_clustering,
    sweep,
    weights_for,
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


class TestNormalizeIsHonoured:
    """The fusion mode must reach FeatureBuilder, not be accepted and dropped."""

    # A weight of exactly 0 silences a block under either mode, so the gap only
    # opens at a small but non-zero alpha — the case the sweep cared about.
    WEIGHTS: ClassVar[dict[str, float]] = {
        "alpha": 0.1,
        "beta": 0.9,
        "gamma": 0.0,
        "delta": 0.0,
    }

    @staticmethod
    def _lopsided() -> Dataset:
        """A 384-dim text block against a 2-dim type block, as in the real corpus."""
        rng = np.random.default_rng(3)
        n, half = 60, 30
        return Dataset(
            e_text=rng.normal(0, 1, (n, 384)),
            e_type=np.vstack(
                [np.tile([0.0, 8.0], (half, 1)), np.tile([8.0, 0.0], (half, 1))]
            ),
            e_rest=rng.normal(0, 1, (n, 2)),
            e_stat=rng.normal(0, 1, (n, 1)),
            column_index=[f"T{i // half}.C{i}" for i in range(n)],
            truth=["eav"] * half + ["clean"] * half,
        )

    def test_block_mode_gives_the_text_block_the_share_alpha_asks_for(self):
        """alpha=0.1 against beta=0.9 → 0.01 : 0.81, so ~1.2% of the variance."""
        data = self._lopsided()
        shares = block_variance_shares(data, build_phi(data, self.WEIGHTS, normalize="block"))
        assert shares["e_text"] == pytest.approx(0.01 / 0.82, abs=0.005)

    def test_zscore_mode_hands_it_seventy_percent_instead(self):
        """Same weights: 384 dims x 0.01 against 2 dims x 0.81 → the text wins 70:30."""
        data = self._lopsided()
        shares = block_variance_shares(data, build_phi(data, self.WEIGHTS, normalize="zscore"))
        assert shares["e_text"] == pytest.approx(3.84 / 5.46, abs=0.02)

    def test_the_two_modes_are_not_the_same_space(self):
        data = self._lopsided()
        block = build_phi(data, self.WEIGHTS, normalize="block")
        zscore = build_phi(data, self.WEIGHTS, normalize="zscore")
        assert not np.allclose(block, zscore)

    def test_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="normalize"):
            run_clustering(
                _dataset(40),
                {"alpha": 1.0, "beta": 0, "gamma": 0, "delta": 0},
                normalize="minmax",
            )


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


class TestRepresentationDegeneracy:
    """How many of the columns can the representation actually tell apart?"""

    @staticmethod
    def _tied() -> Dataset:
        """A type block with two distinct rows and a text block with forty."""
        rng = np.random.default_rng(11)
        n, half = 40, 20
        return Dataset(
            e_text=rng.normal(0, 1, (n, 16)),
            e_type=np.vstack([np.tile([1.0, 0.0], (half, 1)), np.tile([0.0, 1.0], (half, 1))]),
            e_rest=np.ones((n, 3)),
            e_stat=np.ones((n, 1)),
            column_index=[f"T{i // half}.C{i}" for i in range(n)],
            truth=["eav"] * half + ["clean"] * half,
        )

    def test_structure_only_collapses_the_corpus_to_its_distinct_types(self):
        """alpha=0 leaves only the type block, which has two distinct rows."""
        data = self._tied()
        result = representation_degeneracy(
            data, {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "delta": 0.0}
        )
        assert result["n_columns"] == 40
        assert result["n_distinct"] == 2
        assert result["largest_tie_group"] == 20
        assert result["duplicate_fraction"] == pytest.approx(1.0)

    def test_the_text_block_individuates_every_column(self):
        data = self._tied()
        result = representation_degeneracy(
            data, {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0}
        )
        assert result["n_distinct"] == 40
        assert result["duplicate_fraction"] == pytest.approx(0.0)
        assert result["largest_tie_group"] == 1


class TestPrecisionSensitivity:
    """A score that moves with the float width is not a measurement."""

    def test_a_well_separated_dataset_agrees_across_precisions(self):
        result = precision_sensitivity(
            _dataset(40), {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0}
        )
        assert result["ari_float32"] == pytest.approx(result["ari_float64"])
        assert result["gap"] == pytest.approx(0.0)

    def test_reports_both_precisions_and_their_gap(self):
        result = precision_sensitivity(
            _dataset(40), {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0}
        )
        assert set(result) == {"ari_float32", "ari_float64", "gap"}
        assert result["gap"] == pytest.approx(abs(result["ari_float32"] - result["ari_float64"]))


class TestWeightsFor:
    def test_alpha_zero_leaves_the_structural_ratios_intact(self):
        weights = weights_for(0.0, {"alpha": 0.0, "beta": 0.35, "gamma": 0.45, "delta": 0.20})
        assert weights == {"alpha": 0.0, "beta": 0.35, "gamma": 0.45, "delta": 0.20}

    def test_alpha_one_silences_the_structural_blocks(self):
        weights = weights_for(1.0)
        assert weights["alpha"] == 1.0
        assert weights["beta"] == weights["gamma"] == weights["delta"] == 0.0

    def test_the_structural_blocks_keep_their_relative_proportions(self):
        """Half the space to the semantics, the rest split 35:45:20 as before."""
        weights = weights_for(0.5, {"alpha": 0.0, "beta": 0.35, "gamma": 0.45, "delta": 0.20})
        assert weights["beta"] == pytest.approx(0.175)
        assert weights["gamma"] == pytest.approx(0.225)
        assert weights["delta"] == pytest.approx(0.100)


class TestSweep:
    def test_one_row_per_alpha_named_after_it(self):
        rows = sweep(_dataset(40), alphas=(0.0, 0.5), min_cluster_size=5)
        assert [r.name for r in rows] == ["alpha=0.00", "alpha=0.50"]

    def test_every_row_carries_ami_alongside_nmi(self):
        for row in sweep(_dataset(40), alphas=(0.5,), min_cluster_size=5):
            assert -1.0 <= row.ami <= 1.0


class TestAblation:
    def test_puts_the_structural_floor_first(self):
        rows = ablation({"docs": _dataset(40)}, alpha=1.0, min_cluster_size=5)
        assert rows[0].name == "structure only (alpha=0)"
        assert [r.name for r in rows[1:]] == ["docs"]

    def test_scores_every_representation_given(self):
        rows = ablation(
            {"a": _dataset(40), "b": _dataset(40)}, alpha=1.0, min_cluster_size=5
        )
        assert len(rows) == 3

    def test_no_datasets_yields_no_rows(self):
        assert ablation({}) == []


class TestFormatDiagnostics:
    def test_each_name_is_scored_against_its_own_dataset(self):
        """Reusing one dataset for every name reports one result three times."""
        tied = TestRepresentationDegeneracy._tied()
        rich = _dataset(40)
        text = format_diagnostics(
            {
                "tied": (tied, {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "delta": 0.0}),
                "rich": (rich, {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0}),
            }
        )
        assert "2/40" in text
        assert "40/40" in text


class TestFormatTable:
    def test_renders_one_line_per_row_with_the_name(self):
        rows = [
            evaluate("solo BERT", _dataset(40), {"alpha": 1.0, "beta": 0, "gamma": 0, "delta": 0}).score
        ]
        text = format_table(rows)
        assert "solo BERT" in text
        assert "ARI" in text

    def test_reports_ami_and_says_which_metric_to_compare(self):
        rows = [evaluate("x", _dataset(40), weights_for(1.0)).score]
        text = format_table(rows)
        assert "AMI" in text
        assert "different k" in text

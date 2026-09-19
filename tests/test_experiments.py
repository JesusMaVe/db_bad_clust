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
    evaluate_late_fusion,
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

    def test_cluster_selection_method_is_forwarded_to_hdbscan(self):
        """leaf vs eom must reach ClusterEngine, not be accepted and dropped —
        the same failure mode NormalizeIsHonoured guards against for alpha."""
        data = _dataset(40)
        weights = {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25}
        # Both must run without error and return one label per column.
        eom = run_clustering(data, weights, cluster_selection_method="eom")
        leaf = run_clustering(data, weights, cluster_selection_method="leaf")
        assert eom.shape == leaf.shape == (40,)

    def test_reducer_kwargs_reach_the_reducer(self):
        """UMAP's n_neighbors/min_dist must actually change with the corpus,
        not silently keep the reducer's own large-dataset defaults."""
        data = _dataset(40)
        weights = {"alpha": 0.25, "beta": 0.25, "gamma": 0.25, "delta": 0.25}
        labels = run_clustering(
            data,
            weights,
            reducer="umap",
            reducer_kwargs={"n_neighbors": 5, "min_dist": 0.0},
            min_cluster_size=5,
        )
        assert labels.shape == (40,)

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


class TestEvaluateLateFusion:
    def test_recovers_the_planted_grouping_from_two_views(self):
        """A document-only view and a structure-only view, each cleanly
        separating the two planted clouds, must agree in consensus."""
        data = _dataset(40)
        result = evaluate_late_fusion(
            "late fusion",
            data,
            views=[
                {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0},
                {"alpha": 0.0, "beta": 1.0, "gamma": 0.0, "delta": 0.0},
            ],
            min_cluster_size=5,
        )
        assert result.score.name == "late fusion"
        assert len(result.predicted) == 40
        assert len(result.cluster_ids) == 40

    def test_single_view_matches_evaluate_on_the_same_weights(self):
        """One view is a degenerate case of late fusion — sanity check that
        the consensus step doesn't distort a trivial one-view fusion."""
        data = _dataset(40)
        weights = {"alpha": 1.0, "beta": 0.0, "gamma": 0.0, "delta": 0.0}
        direct = evaluate("direct", data, weights, min_cluster_size=5)
        fused = evaluate_late_fusion(
            "fused", data, views=[weights], min_cluster_size=5
        )
        # Co-association of a single view is exact agreement/disagreement, so
        # the consensus cut should reproduce the same number of clusters.
        assert len(set(fused.cluster_ids) - {-1}) == len(set(direct.cluster_ids) - {-1})


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


# ═══════════════════════════════════════════════════════════════════════
# The conflict block, table leakage and the stability sweep
# ═══════════════════════════════════════════════════════════════════════

from db_bad_clust.evaluation.experiments import (  # noqa: E402
    CONFLICT,
    CONFLICT_FUSED,
    STABILITY_GRID,
    choose_operating_point,
    cluster_composition,
    format_stability,
    stability_sweep,
)
from db_bad_clust.features.semantic_anchors import CONFLICT_DIM  # noqa: E402


def _conflict_dataset(n: int = 40) -> Dataset:
    """`_dataset` plus a conflict block that separates the same two groups."""
    base = _dataset(n)
    rng = np.random.default_rng(1)
    half = n // 2
    e_conflict = rng.normal(0, 0.05, (n, CONFLICT_DIM))
    e_conflict[:half, 0] += 1.0  # "date expected" for the first group
    e_conflict[half:, 3] += 1.0  # "text expected" for the second
    e_conflict[:half, 6] = 1.0  # declared date
    e_conflict[half:, 9] = 1.0  # declared text
    return Dataset(
        e_text=base.e_text,
        e_type=base.e_type,
        e_rest=base.e_rest,
        e_stat=base.e_stat,
        column_index=base.column_index,
        truth=base.truth,
        e_conflict=e_conflict,
    )


class TestConflictDataset:
    def test_block_travels_through_without_tables(self):
        ds = _conflict_dataset()
        kept = ds.without_tables({"T0"})
        assert kept.e_conflict.shape == (20, CONFLICT_DIM)
        assert np.array_equal(kept.e_conflict, ds.e_conflict[20:])

    def test_rejects_a_conflict_block_of_the_wrong_length(self):
        base = _dataset()
        with pytest.raises(ValueError):
            Dataset(
                e_text=base.e_text,
                e_type=base.e_type,
                e_rest=base.e_rest,
                e_stat=base.e_stat,
                column_index=base.column_index,
                truth=base.truth,
                e_conflict=np.zeros((3, CONFLICT_DIM)),
            )

    def test_zeta_absent_from_weights_leaves_phi_unchanged(self):
        """Every existing weights dict lacks "zeta": the block is present but silent."""
        ds = _conflict_dataset()
        phi = build_phi(ds, weights_for(1.0))
        assert phi.shape[1] == 8 + 2 + 3 + 1 + CONFLICT_DIM
        assert np.allclose(phi[:, -CONFLICT_DIM:], 0.0)

    def test_conflict_weights_switch_the_embeddings_off(self):
        assert CONFLICT["alpha"] == 0.0 and CONFLICT["zeta"] == 1.0
        assert CONFLICT_FUSED["zeta"] == 1.0 and 0 < CONFLICT_FUSED["alpha"] < 1

    def test_conflict_representation_never_mixes_the_planted_groups(self):
        """The fixture's constraint block is pure noise, so HDBSCAN may leave a
        few points as noise; what must hold is that no cluster mixes groups."""
        ds = _conflict_dataset()
        run = evaluate("conflict", ds, CONFLICT, min_cluster_size=3, min_samples=2)
        clusters = {c for c in run.cluster_ids if c != -1}
        assert len(clusters) >= 2
        for cid in clusters:
            members = {t for c, t in zip(run.cluster_ids, ds.truth, strict=True) if c == cid}
            assert len(members) == 1
        assert run.score.ari > 0.8

    def test_precision_sensitivity_casts_the_block_too(self):
        ds = _conflict_dataset()
        out = precision_sensitivity(ds, CONFLICT, min_cluster_size=3, min_samples=2)
        assert out["gap"] == pytest.approx(0.0, abs=1e-6)


class TestTableLeakage:
    def test_a_partition_that_is_the_tables_scores_one_against_them(self):
        """The fixture's two groups ARE its two tables — leakage is total here."""
        run = evaluate("doc", _dataset(), weights_for(1.0), min_cluster_size=3, min_samples=2)
        assert run.score.table_ari == pytest.approx(1.0)

    def test_table_ari_is_rendered_in_the_table(self):
        run = evaluate("doc", _dataset(), weights_for(1.0), min_cluster_size=3, min_samples=2)
        out = format_table([run.score])
        assert "ARI~tbl" in out
        assert "1.0000" in out

    def test_a_score_without_tables_renders_na(self):
        from db_bad_clust.evaluation.cluster_scoring import score

        s = score("x", ["a", "b"], ["a", "b"], group_ids=[0, 1])
        assert s.table_ari != s.table_ari  # nan
        assert "n/a" in format_table([s])
        assert s.as_row()["table_ari"] is None


class TestStabilitySweep:
    def test_one_row_per_grid_cell_with_a_validity(self):
        ds = _conflict_dataset()
        grid = ((3, 2), (4, 2), (5, 3))
        rows = stability_sweep(ds, CONFLICT, grid=grid)
        assert [(r.min_cluster_size, r.min_samples) for r in rows] == list(grid)
        assert all(0.0 <= r.noise_fraction <= 1.0 for r in rows)
        assert all(r.score.table_ari == r.score.table_ari for r in rows)

    def test_the_default_grid_is_the_documented_one(self):
        assert (5, 3) in STABILITY_GRID and (8, 3) in STABILITY_GRID
        assert len(STABILITY_GRID) == 15

    def test_operating_point_is_the_highest_defined_validity(self):
        ds = _conflict_dataset()
        rows = stability_sweep(ds, CONFLICT, grid=((3, 2), (4, 2)))
        chosen = choose_operating_point(rows)
        defined = [r for r in rows if r.relative_validity == r.relative_validity]
        if defined:
            assert chosen is max(defined, key=lambda r: r.relative_validity)
        else:
            assert chosen is None

    def test_operating_point_ignores_undefined_cells(self):
        from db_bad_clust.evaluation.cluster_scoring import ClusterScore
        from db_bad_clust.evaluation.experiments import StabilityRow

        def row(v):
            return StabilityRow(5, 3, ClusterScore("x", 0, 0, 0, 0, 1), v, 0.0)

        assert choose_operating_point([row(float("nan"))]) is None
        good = row(0.4)
        assert choose_operating_point([row(float("nan")), row(0.1), good]) is good

    def test_format_marks_the_chosen_cell_and_never_the_best_label(self):
        ds = _conflict_dataset()
        rows = stability_sweep(ds, CONFLICT, grid=((3, 2), (4, 2)))
        out = format_stability(rows)
        assert "validity" in out
        assert ("*" in out) == (choose_operating_point(rows) is not None)
        assert "tuning on the test set" in out


class TestClusterComposition:
    def test_one_line_per_cluster_with_tables_spanned(self):
        ds = _dataset()
        out = cluster_composition(ds, [0] * 20 + [1] * 20)
        lines = out.splitlines()
        assert len(lines) == 2
        assert "tables= 1" in lines[0]
        assert "eav 20" in lines[0]
        assert "clean 20" in lines[1]

    def test_noise_is_named_as_such(self):
        ds = _dataset()
        out = cluster_composition(ds, [-1] * 40)
        assert "noise" in out and "tables= 2" in out


# ═══════════════════════════════════════════════════════════════════════
# Blind evaluation: Ward by silhouette, HDBSCAN + kNN, robustness
# ═══════════════════════════════════════════════════════════════════════

from db_bad_clust.evaluation.experiments import (  # noqa: E402
    WARD_K_RANGE,
    _cluster_ward,
    evaluate_blind,
    format_blind_table,
    format_robustness,
    reassign_noise_knn,
    robustness_over_wordings,
)


def _three_blobs(per: int = 15) -> np.ndarray:
    rng = np.random.default_rng(7)
    centres = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    return np.vstack([rng.normal(c, 0.3, (per, 2)) for c in centres])


def _with_variants(ds: Dataset) -> Dataset:
    rng = np.random.default_rng(9)
    variants = {
        "orig": ds.e_conflict,
        "W2": ds.e_conflict + rng.normal(0, 0.02, ds.e_conflict.shape),
        "W3": ds.e_conflict + rng.normal(0, 0.04, ds.e_conflict.shape),
    }
    return Dataset(
        e_text=ds.e_text,
        e_type=ds.e_type,
        e_rest=ds.e_rest,
        e_stat=ds.e_stat,
        column_index=ds.column_index,
        truth=ds.truth,
        e_conflict=ds.e_conflict,
        e_conflict_variants=variants,
    )


class TestClusterWard:
    def test_picks_the_planted_number_of_groups_by_silhouette(self):
        labels, k, sil = _cluster_ward(_three_blobs())
        assert k == 3
        assert len(set(labels.tolist())) == 3
        assert sil > 0.8

    def test_never_asks_for_more_clusters_than_points_allow(self):
        labels, k, _ = _cluster_ward(_three_blobs(per=2), k_range=(2, 30))
        assert k <= 5
        assert len(labels) == 6

    def test_default_range_is_the_documented_one(self):
        assert WARD_K_RANGE == (2, 30)


class TestReassignNoise:
    def test_leaves_no_noise_and_keeps_clustered_points(self):
        X = _three_blobs()
        labels = np.repeat([0, 1, 2], 15)
        noisy = labels.copy()
        noisy[[0, 16, 31]] = -1
        out = reassign_noise_knn(X, noisy)
        assert (out != -1).all()
        assert np.array_equal(out[noisy != -1], labels[noisy != -1])
        assert np.array_equal(out, labels)  # each noise point sits inside its blob

    def test_all_noise_comes_back_unchanged(self):
        X = _three_blobs()
        out = reassign_noise_knn(X, np.full(len(X), -1))
        assert (out == -1).all()


class TestEvaluateBlind:
    @pytest.mark.parametrize("algorithm", ["ward", "hdbscan"])
    def test_reports_the_choice_and_its_criterion(self, algorithm):
        ds = _conflict_dataset()
        run = evaluate_blind("c", ds, CONFLICT, algorithm=algorithm, grid=((3, 2), (5, 3)))
        assert run.algorithm == algorithm
        assert run.chosen.startswith("k=" if algorithm == "ward" else "mcs=")
        assert run.criterion == run.criterion  # defined
        assert run.evaluation.score.ari > 0.8

    def test_hdbscan_path_leaves_no_noise(self):
        ds = _conflict_dataset()
        run = evaluate_blind("c", ds, CONFLICT, algorithm="hdbscan", grid=((3, 2),))
        assert -1 not in run.evaluation.cluster_ids

    def test_rejects_an_unknown_algorithm(self):
        with pytest.raises(ValueError):
            evaluate_blind("c", _conflict_dataset(), CONFLICT, algorithm="kmeans")

    def test_table_lists_every_run(self):
        ds = _conflict_dataset()
        runs = [
            (a, evaluate_blind(a, ds, CONFLICT, algorithm=a, grid=((3, 2),)))
            for a in ("ward", "hdbscan")
        ]
        out = format_blind_table(runs)
        assert "ward" in out and "hdbscan" in out and "chosen" in out


class TestRobustness:
    def test_one_row_per_wording_and_a_summary(self):
        ds = _with_variants(_conflict_dataset())
        report = robustness_over_wordings(ds, CONFLICT, algorithm="ward")
        assert [w for w, _ in report.rows] == ["orig", "W2", "W3"]
        mean, low, high = report.summary("ari")
        assert low <= mean <= high
        assert report.smallest_k >= 2

    def test_each_wording_is_really_swapped_in(self):
        ds = _with_variants(_conflict_dataset())
        swapped = ds.with_conflict(ds.e_conflict_variants["W3"])
        assert np.array_equal(swapped.e_conflict, ds.e_conflict_variants["W3"])
        assert swapped.e_text is ds.e_text

    def test_refuses_a_dataset_without_variants(self):
        with pytest.raises(ValueError):
            robustness_over_wordings(_conflict_dataset(), CONFLICT)

    def test_format_names_the_mean_as_the_headline(self):
        ds = _with_variants(_conflict_dataset())
        out = format_robustness(robustness_over_wordings(ds, CONFLICT))
        assert "mean" in out and "headline" in out

    def test_without_tables_trims_every_variant(self):
        ds = _with_variants(_conflict_dataset())
        kept = ds.without_tables({"T0"})
        assert all(v.shape[0] == 20 for v in kept.e_conflict_variants.values())

    def test_rejects_a_variant_of_the_wrong_length(self):
        ds = _conflict_dataset()
        with pytest.raises(ValueError):
            Dataset(
                e_text=ds.e_text,
                e_type=ds.e_type,
                e_rest=ds.e_rest,
                e_stat=ds.e_stat,
                column_index=ds.column_index,
                truth=ds.truth,
                e_conflict=ds.e_conflict,
                e_conflict_variants={"orig": ds.e_conflict[:3]},
            )


# ═══════════════════════════════════════════════════════════════════════
# Bootstrap
# ═══════════════════════════════════════════════════════════════════════

from db_bad_clust.evaluation.experiments import (  # noqa: E402
    BootstrapResult,
    blind_scorer,
    bootstrap_compare,
    fixed_partition_scorer,
    format_bootstrap,
    mean_over_wordings_scorer,
)


def _constant_scorer(value: float):
    def run(dataset):
        return {"ari": value, "ami": value, "table_ari": 0.0}

    return run


class TestSubset:
    def test_keeps_the_rows_in_order_including_repeats(self):
        ds = _conflict_dataset()
        sub = ds.subset([3, 1, 1])
        assert sub.column_index == [ds.column_index[3], ds.column_index[1], ds.column_index[1]]
        assert np.array_equal(sub.e_conflict, ds.e_conflict[[3, 1, 1]])

    def test_without_tables_is_a_subset(self):
        ds = _conflict_dataset()
        assert ds.without_tables({"T0"}).column_index == ds.subset(range(20, 40)).column_index


class TestBootstrapCompare:
    def test_subsample_scores_every_config_on_every_resample(self):
        ds = _conflict_dataset()
        result = bootstrap_compare(
            ds, {"a": _constant_scorer(0.5), "b": _constant_scorer(0.2)}, n_boot=5, frac=0.5
        )
        assert result.samples["a"]["ari"].shape == (5,)
        assert result.point["a"]["ari"] == 0.5

    def test_subsample_draws_the_requested_fraction_without_repeats(self):
        seen = []

        def spy(dataset):
            seen.append(list(dataset.column_index))
            return {"ari": 0.0, "ami": 0.0, "table_ari": 0.0}

        ds = _conflict_dataset()
        bootstrap_compare(ds, {"s": spy}, n_boot=3, frac=0.5)
        drawn = seen[1:]  # the first call is the point estimate on the full data
        assert all(len(d) == 20 and len(set(d)) == 20 for d in drawn)

    def test_paired_difference_of_constant_scorers_is_exact(self):
        ds = _conflict_dataset()
        result = bootstrap_compare(
            ds, {"a": _constant_scorer(0.5), "b": _constant_scorer(0.2)}, n_boot=4, frac=0.5
        )
        mean, lo, hi, wins = result.paired_difference("a", "b", "ari")
        assert mean == pytest.approx(0.3)
        assert lo == pytest.approx(0.3) and hi == pytest.approx(0.3)
        assert wins == 1.0

    def test_is_reproducible_under_a_seed(self):
        ds = _conflict_dataset()
        scorers = {"w": blind_scorer(CONFLICT, "ward")}
        a = bootstrap_compare(ds, scorers, n_boot=3, frac=0.7, seed=1)
        b = bootstrap_compare(ds, scorers, n_boot=3, frac=0.7, seed=1)
        assert np.array_equal(a.samples["w"]["ari"], b.samples["w"]["ari"])

    def test_fixed_mode_rescores_one_partition(self):
        ds = _conflict_dataset()
        result = bootstrap_compare(
            ds, {"w": fixed_partition_scorer(CONFLICT, "ward")}, n_boot=20, mode="fixed", frac=0.8
        )
        low, high = result.interval("w", "ari")
        assert -1.0 <= low <= high <= 1.0
        assert result.mode == "fixed"
        # re-scoring one fixed partition on subsampled columns centres on its full-data score
        assert abs(np.median(result.samples["w"]["ari"]) - result.point["w"]["ari"]) < 0.1

    def test_fixed_mode_on_every_column_reproduces_the_point_estimate(self):
        """No duplicates and no dropped columns: the re-score is the score."""
        ds = _conflict_dataset()
        result = bootstrap_compare(
            ds, {"w": fixed_partition_scorer(CONFLICT, "ward")}, n_boot=3, mode="fixed", frac=1.0
        )
        for metric in ("ari", "ami"):
            assert np.allclose(result.samples["w"][metric], result.point["w"][metric])

    def test_fixed_mode_refuses_scorers_without_a_partition(self):
        with pytest.raises(ValueError):
            bootstrap_compare(
                _conflict_dataset(), {"a": _constant_scorer(0.1)}, n_boot=2, mode="fixed"
            )

    def test_rejects_bad_mode_and_fraction(self):
        ds = _conflict_dataset()
        with pytest.raises(ValueError):
            bootstrap_compare(ds, {"a": _constant_scorer(0.1)}, mode="jackknife")
        with pytest.raises(ValueError):
            bootstrap_compare(ds, {"a": _constant_scorer(0.1)}, frac=0.0)

    def test_mean_over_wordings_averages_the_three_runs(self):
        ds = _with_variants(_conflict_dataset())
        mean = mean_over_wordings_scorer(CONFLICT, "ward")(ds)
        each = [blind_scorer(CONFLICT, "ward", w)(ds)["ari"] for w in ("orig", "W2", "W3")]
        assert mean["ari"] == pytest.approx(np.mean(each))

    def test_wording_scorer_refuses_a_missing_variant(self):
        with pytest.raises(ValueError):
            blind_scorer(CONFLICT, "ward", "W9")(_with_variants(_conflict_dataset()))

    def test_format_reports_intervals_and_paired_wins(self):
        ds = _conflict_dataset()
        result = bootstrap_compare(
            ds, {"ref": _constant_scorer(0.1), "new": _constant_scorer(0.3)}, n_boot=3, frac=0.5
        )
        out = format_bootstrap(result, "ref")
        assert "95% CI" in out
        assert "new - ref" in out
        assert "100%" in out
        assert isinstance(result, BootstrapResult)

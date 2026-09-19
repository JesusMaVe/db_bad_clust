"""Tests for the name signals and the table level. No model download, no DB."""

from __future__ import annotations

import numpy as np
import pytest

from db_bad_clust.clustering.table_level import (
    NORMAL,
    combine_levels,
    flag_anomalous_tables,
    tukey_fence,
)
from db_bad_clust.features.name_signals import (
    TABLE_NAMING_FEATURES,
    internal_divergence,
    semantic_emptiness,
    sibling_signals,
    table_naming_features,
)


class AxisEmbedder:
    """Every known word sits on its own unit axis; unknown words share the last one."""

    def __init__(self, words: list[str]) -> None:
        self.words = words

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), len(self.words) + 1))
        for i, text in enumerate(texts):
            j = self.words.index(text) if text in self.words else len(self.words)
            out[i, j] = 1.0
        return out


class TestSemanticEmptiness:
    def test_zero_on_an_anchor_and_one_when_orthogonal_to_all(self):
        anchors = np.eye(3)[:2]
        names = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        assert np.allclose(semantic_emptiness(names, anchors), [0.0, 1.0])


class TestInternalDivergence:
    def test_two_unrelated_words_are_maximally_divergent(self):
        emb = AxisEmbedder(["fecha", "direccion"])
        assert internal_divergence(["fecha o direccion"], emb)[0] == pytest.approx(1.0)

    def test_a_single_word_and_joiners_score_zero(self):
        emb = AxisEmbedder(["fecha"])
        out = internal_divergence(["fecha", "fecha de o", "c1"], emb)
        assert np.allclose(out, 0.0)

    def test_repeated_meaning_is_not_divergent(self):
        emb = AxisEmbedder(["precio"])
        assert internal_divergence(["precio precio"], emb)[0] == pytest.approx(0.0)


class TestSiblingSignals:
    def test_an_outlier_name_is_atypical_and_a_lone_table_scores_zero(self):
        vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        out = sibling_signals(vectors, ["A", "B", "C", "D"], ["T", "T", "T", "U"])
        assert out[2, 0] == pytest.approx(1.0)  # orthogonal to its siblings' centroid
        assert out[0, 0] == pytest.approx(1 - 1 / np.sqrt(2))
        assert np.allclose(out[3], 0.0)  # single-column table

    def test_clash_is_a_near_synonym_spelt_differently(self):
        vectors = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        out = sibling_signals(vectors, ["FLAG_S_N", "FLAG_Y_N", "NOTAS"], ["T", "T", "T"])
        assert out[0, 1] == pytest.approx(1.0)
        assert out[2, 1] == pytest.approx(0.0)

    def test_a_repeated_spelling_is_not_a_clash(self):
        vectors = np.array([[1.0, 0.0], [1.0, 0.0]])
        out = sibling_signals(vectors, ["ID", "ID"], ["T", "T"])
        assert np.allclose(out[:, 1], 0.0)

    def test_heterogeneity_is_shared_by_the_whole_table(self):
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
        out = sibling_signals(vectors, ["A", "B", "C"], ["T", "T", "T"])
        assert np.allclose(out[:, 2], out[0, 2])
        assert out[0, 2] == pytest.approx(1 - 1 / 3)


class TestTableNamingFeatures:
    def test_one_row_per_table_in_first_appearance_order(self):
        intrinsic = np.array([[0.2, 0.0], [0.4, 0.5], [0.9, 0.1]])
        vectors = np.eye(3)
        order, X = table_naming_features(
            intrinsic, vectors, ["A", "B", "C"], ["T2", "T2", "T1"], np.array([1, 0, 0])
        )
        assert order == ["T2", "T1"]
        assert X.shape == (2, len(TABLE_NAMING_FEATURES))
        assert X[0, 0] == pytest.approx(0.3)  # mean emptiness
        assert X[0, 5] == pytest.approx(0.5)  # conflict rate
        assert X[1, 5] == pytest.approx(0.0)


class TestFlagAnomalousTables:
    def _population(self, outliers: list[list[float]]) -> np.ndarray:
        rng = np.random.default_rng(0)
        return np.vstack([rng.normal(0, 0.1, (18, 3)), np.array(outliers)])

    def test_a_far_table_is_flagged_and_the_rest_stay_normal(self):
        groups = flag_anomalous_tables(self._population([[5.0, 5.0, 5.0]]))
        assert groups[-1] != NORMAL
        assert (groups[:-1] == NORMAL).all()

    def test_two_close_outliers_share_a_group_and_a_distant_one_does_not(self):
        X = self._population([[5.0, 5.0, 5.0], [5.1, 5.0, 5.0], [-5.0, 5.0, -5.0]])
        groups = flag_anomalous_tables(X)
        assert groups[-3] == groups[-2] != NORMAL
        assert groups[-1] not in (NORMAL, groups[-3])

    def test_a_homogeneous_population_has_no_anomaly(self):
        X = np.random.default_rng(1).normal(0, 1, (20, 3))
        X[np.abs(X) > 1.5] = 0  # no heavy tail
        assert (flag_anomalous_tables(X) == NORMAL).sum() >= 18

    def test_a_table_too_small_to_have_siblings_is_never_judged(self):
        """A one-column table scores zero on every sibling signal by construction;
        flagging it would be an artefact, so it is held out and stays normal."""
        X = self._population([[5.0, 5.0, 5.0]])
        sizes = [6] * 18 + [1]
        assert flag_anomalous_tables(X, n_columns=sizes)[-1] == NORMAL
        assert flag_anomalous_tables(X, n_columns=[6] * 19)[-1] != NORMAL

    def test_held_out_tables_do_not_distort_the_fence_for_the_rest(self):
        base = self._population([[5.0, 5.0, 5.0]])
        extreme_tiny = np.vstack([base, [[-40.0, 40.0, -40.0]]])
        groups = flag_anomalous_tables(extreme_tiny, n_columns=[6] * 19 + [1])
        assert groups[-1] == NORMAL
        assert groups[-2] != NORMAL  # the real outlier is still caught

    def test_too_few_tables_are_all_normal(self):
        assert (flag_anomalous_tables(np.eye(3)) == NORMAL).all()

    def test_fence_is_q3_plus_one_and_a_half_iqr(self):
        assert tukey_fence(np.array([1.0, 2.0, 3.0, 4.0, 5.0])) == pytest.approx(4.0 + 1.5 * 2.0)


class TestCombineLevels:
    def test_normal_tables_keep_column_clusters_and_anomalous_ones_share_a_group(self):
        out = combine_levels(
            column_ids=[0, 1, 0, 1, 2],
            column_tables=["A", "A", "B", "B", "C"],
            table_order=["A", "B", "C"],
            table_groups=np.array([NORMAL, 0, 0]),
        )
        assert out[0] == 0 and out[1] == 1
        assert out[2] == out[3] == out[4]
        assert out[2] not in (0, 1, 2)

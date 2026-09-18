"""Tests for consensus (late-fusion) clustering. No DB, no real model."""

from __future__ import annotations

import numpy as np
import pytest

from db_bad_clust.clustering.late_fusion import co_association_matrix, consensus_clustering


class TestCoAssociationMatrix:
    def test_identical_views_give_zero_or_one(self):
        labels = np.array([0, 0, 1, 1])
        matrix = co_association_matrix([labels, labels])
        assert matrix[0, 1] == 1.0  # both 0 in both views
        assert matrix[0, 2] == 0.0  # different clusters in both views

    def test_disagreeing_views_average_to_half(self):
        a = np.array([0, 0, 1])
        b = np.array([0, 1, 1])
        matrix = co_association_matrix([a, b])
        # columns 0,1: together in view a, apart in view b -> 0.5
        assert matrix[0, 1] == pytest.approx(0.5)
        # columns 1,2: apart in view a, together in view b -> 0.5
        assert matrix[1, 2] == pytest.approx(0.5)

    def test_noise_never_counts_as_agreement(self):
        a = np.array([-1, -1, 0])
        b = np.array([-1, -1, 0])
        matrix = co_association_matrix([a, b])
        # both columns are noise (-1) in every view -> never counted together
        assert matrix[0, 1] == 0.0

    def test_diagonal_is_one(self):
        labels = np.array([0, 1, 2])
        matrix = co_association_matrix([labels])
        assert np.all(np.diag(matrix) == 1.0)

    def test_mismatched_view_lengths_raise(self):
        with pytest.raises(ValueError, match="same number"):
            co_association_matrix([np.array([0, 1]), np.array([0, 1, 2])])

    def test_empty_views_list_raises(self):
        with pytest.raises(ValueError, match="one view"):
            co_association_matrix([])


class TestConsensusClustering:
    def test_columns_agreeing_in_every_view_end_up_together(self):
        # Two well-separated pairs, agreeing across both views.
        a = np.array([0, 0, 1, 1])
        b = np.array([0, 0, 1, 1])
        labels = consensus_clustering([a, b], distance_threshold=0.5)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]

    def test_columns_that_never_agree_end_up_apart(self):
        a = np.array([0, 1, 2, 3])
        b = np.array([3, 2, 1, 0])  # no pair ever shares a cluster in both views
        labels = consensus_clustering([a, b], distance_threshold=0.1)
        assert len(set(labels.tolist())) == 4

    def test_empty_input_returns_empty_array(self):
        labels = consensus_clustering([np.array([]), np.array([])])
        assert labels.shape == (0,)

    def test_single_view_reproduces_its_own_partition_shape(self):
        a = np.array([0, 0, 1, 1, 2])
        labels = consensus_clustering([a], distance_threshold=0.5)
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert len(labels) == 5

    def test_lower_threshold_never_merges_more_than_higher_threshold(self):
        """A stricter (lower) distance_threshold requires more agreement to
        merge, so it can only produce as many or more distinct clusters."""
        a = np.array([0, 0, 1, 1, 2, 2])
        b = np.array([0, 1, 1, 2, 2, 0])
        strict = consensus_clustering([a, b], distance_threshold=0.1)
        loose = consensus_clustering([a, b], distance_threshold=0.9)
        assert len(set(strict.tolist())) >= len(set(loose.tolist()))
